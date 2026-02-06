"""
单文件、可直接训练的离散手势脚本（不依赖项目内其它模块）
"""

import logging
import math
import os
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm.auto import tqdm

# ==========================
# 基础参数（全部写死）
# ==========================
SEED = 0
DATA_LOCATION = os.path.expanduser("~/emg_data")
CSV_FILENAME = os.path.join(DATA_LOCATION, "discrete_gestures_corpus.csv")

WINDOW_LENGTH = 16_000  # 8 秒 @ 2kHz
STRIDE = 16_000
BATCH_SIZE = 64
NUM_WORKERS = 0

MAX_EPOCHS = 250
ACCELERATOR = "auto"  # cpu/gpu/auto

LEARNING_RATE = 5e-4
WARMUP_START_FACTOR = 0.001
WARMUP_END_FACTOR = 1.0
WARMUP_TOTAL_EPOCHS = 5
LR_SCHEDULER_MILESTONES = [25]
LR_SCHEDULER_FACTOR = 0.5
GRADIENT_CLIP_VAL = 0.5

PULSE_WINDOW = [0.08, 0.12]  # 秒
ROTATION_AUG = 2

LOG_EVERY = 50

# ==========================
# 常量 / 枚举
# ==========================
EMG_NUM_CHANNELS = 16
EMG_SAMPLE_RATE = 2000  # Hz


class GestureType(Enum):
    index_press = 0
    index_release = 1
    middle_press = 2
    middle_release = 3
    thumb_click = 4
    thumb_down = 5
    thumb_in = 6
    thumb_out = 7
    thumb_up = 8


# ==========================
# 工具函数
# ==========================


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_full_dataset_path(root: str, dataset: str) -> Path:
    path = Path(root).expanduser().joinpath(f"{dataset}")
    if not path.suffix:
        path = path.with_suffix(".hdf5")
    return path


# ==========================
# 数据切分
# ==========================


@dataclass
class DataSplit:
    train: dict[str, list[tuple[float, float]] | None]
    val: dict[str, list[tuple[float, float]] | None]
    test: dict[str, list[tuple[float, float]] | None]

    @classmethod
    def from_csv(cls, csv_filename: str, pool_test_partitions: bool = True) -> "DataSplit":
        df = pd.read_csv(csv_filename)
        splits: dict[str, dict[str, list[tuple[float, float]]]] = {
            "train": {},
            "val": {},
            "test": {},
        }

        for split in ["train", "val", "test"]:
            for dataset in df[df["split"] == split]["dataset"].unique():
                dataset_rows = df[(df["split"] == split) & (df["dataset"] == dataset)]
                if split == "test" and pool_test_partitions:
                    first_start = dataset_rows["start"].min()
                    last_end = dataset_rows["end"].max()
                    splits[split][dataset] = [(first_start, last_end)]
                else:
                    splits[split][dataset] = [
                        (row.start, row.end) for row in dataset_rows.itertuples()
                    ]

        return cls(**splits)


# ==========================
# 数据读取
# ==========================


class EmgRecording:
    def __init__(self, hdf5_path: Path, start_time: float = -np.inf, end_time: float = np.inf) -> None:
        self.hdf5_path = hdf5_path
        self.start_time = start_time
        self.end_time = end_time

        self._file = h5py.File(self.hdf5_path, "r")
        self.timeseries = self._file["data"]
        self.prompts = pd.read_hdf(hdf5_path, "prompts") if "prompts" in self._file.keys() else None

        timestamps = self.timeseries["time"]
        assert (np.diff(timestamps) >= 0).all(), "Timestamps are not monotonic"
        self.start_idx, self.end_idx = timestamps.searchsorted([self.start_time, self.end_time])

    def __len__(self) -> int:
        return self.end_idx - self.start_idx

    def __getitem__(self, key: slice) -> np.ndarray:
        if not isinstance(key, slice):
            raise TypeError("Only slices are supported")
        start = key.start if key.start is not None else 0
        stop = key.stop if key.stop is not None else len(self)
        start += self.start_idx
        stop += self.start_idx
        return self.timeseries[start:stop]


class WindowedEmgDataset(Dataset):
    def __init__(
        self,
        hdf5_path: Path,
        start: float,
        end: float,
        transform: Callable[[np.ndarray, pd.DataFrame | None], dict[str, torch.Tensor]],
        emg_augmentation: Callable[[torch.Tensor], torch.Tensor] | None = None,
        window_length: int | None = 10_000,
        stride: int | None = None,
        jitter: bool = False,
    ) -> None:
        self.hdf5_path = hdf5_path
        self.start = start
        self.end = end
        self.transform = transform
        self.emg_augmentation = emg_augmentation
        self.window_length = window_length
        self.stride = stride
        self.jitter = jitter

        self.emg_recording = EmgRecording(self.hdf5_path, self.start, self.end)
        self.window_length = window_length if window_length is not None else len(self.emg_recording)
        self.stride = stride if stride is not None else self.window_length

    def __len__(self) -> int:
        return int(max(len(self.emg_recording) - self.window_length, 0) // self.stride + 1)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | pd.DataFrame | np.ndarray]:
        start_sample = idx * self.stride

        leftover = len(self.emg_recording) - (start_sample + self.window_length)
        if leftover < 0:
            raise IndexError(f"Index {idx} out of bounds")
        if leftover > 0 and self.jitter:
            start_sample += np.random.randint(0, min(self.stride, leftover))

        start = start_sample
        end = start_sample + self.window_length
        timeseries = self.emg_recording[start:end]

        datum = self.transform(timeseries, self.emg_recording.prompts)

        if self.emg_augmentation is not None:
            datum["emg"] = self.emg_augmentation(datum["emg"])

        datum["timestamps"] = timeseries["time"].copy()
        datum["prompts"] = self.emg_recording.prompts
        return datum


def make_dataset(
    data_location: str,
    partition_dict: dict[str, list[tuple[float, float]] | None],
    transform: Callable[[np.ndarray, pd.DataFrame | None], dict[str, torch.Tensor]],
    emg_augmentation: Callable[[torch.Tensor], torch.Tensor] | None,
    window_length: int | None,
    stride: int | None,
    jitter: bool,
    split_label: str | None = None,
) -> ConcatDataset:
    datasets: list[Dataset] = []
    for dataset, partitions in tqdm(
        partition_dict.items(), desc=f"[setup] Loading datasets for split {split_label}"
    ):
        if partitions is None:
            partitions = [(-np.inf, np.inf)]

        for start, end in partitions:
            if window_length is not None:
                partition_samples = (end - start) * EMG_SAMPLE_RATE
                if partition_samples < window_length:
                    continue

            datasets.append(
                WindowedEmgDataset(
                    get_full_dataset_path(data_location, dataset),
                    start=start,
                    end=end,
                    transform=transform,
                    window_length=window_length,
                    stride=stride,
                    jitter=jitter,
                    emg_augmentation=emg_augmentation,
                )
            )

    return ConcatDataset(datasets)


# ==========================
# 数据变换 + 增强
# ==========================


@dataclass
class RotationAugmentation:
    rotation: int = 1

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        rotation = np.random.choice(np.arange(-self.rotation, self.rotation + 1))
        return torch.roll(data, rotation, dims=-1)


@dataclass
class DiscreteGesturesTransform:
    pulse_window: list[float]

    def __call__(self, timeseries: np.ndarray, prompts: pd.DataFrame | None) -> dict[str, torch.Tensor]:
        assert prompts is not None

        tlim = (timeseries["time"][0], timeseries["time"][-1])
        prompts = prompts[prompts["time"].between(*tlim)]
        prompts = prompts[prompts["name"].isin([g.name for g in GestureType])]

        targets = self.gesture_times_to_targets(
            timeseries["time"],
            prompts["time"],
            prompts["name"].map({g.name: g.value for g in GestureType}),
        )

        emg = torch.from_numpy(timeseries["emg"].T).float()
        return {"emg": emg, "targets": targets}

    def gesture_times_to_targets(
        self,
        times: np.ndarray,
        event_start_times: np.ndarray,
        event_ids: pd.Series,
    ) -> torch.Tensor:
        num_timesteps = len(times)
        duration = times[-1] - times[0]
        sampling_freq = int(num_timesteps / duration)

        event_ids = event_ids.to_numpy()
        event_time_indices = np.searchsorted(times, event_start_times)

        pulse = torch.zeros(len(GestureType), num_timesteps, dtype=torch.float32)

        valid_events = (event_time_indices > 0) & (event_time_indices < num_timesteps)
        valid_indices = np.where(valid_events)[0]

        for idx in valid_indices:
            event_start = event_time_indices[idx]
            event_id = event_ids[idx]

            start_offset = int(self.pulse_window[0] * sampling_freq)
            end_offset = int(self.pulse_window[1] * sampling_freq)

            start_idx = max(0, event_start + start_offset)
            end_idx = min(num_timesteps, event_start + end_offset)

            if start_idx < end_idx:
                pulse[event_id, start_idx:end_idx] = 1.0

        return pulse


# ==========================
# 模型
# ==========================


class ReinhardCompression(nn.Module):
    def __init__(self, range: float, midpoint: float) -> None:
        super().__init__()
        self.range = range
        self.midpoint = midpoint

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.range * inputs / (self.midpoint + torch.abs(inputs))


class DiscreteGesturesArchitecture(nn.Module):
    def __init__(
        self,
        input_channels: int = 16,
        conv_output_channels: int = 512,
        kernel_width: int = 21,
        stride: int = 10,
        lstm_hidden_size: int = 512,
        lstm_num_layers: int = 3,
        output_channels: int = 9,
    ) -> None:
        super().__init__()
        self.left_context = kernel_width - 1
        self.stride = stride

        self.compression = ReinhardCompression(range=64.0, midpoint=32.0)
        self.conv_layer = nn.Conv1d(
            input_channels,
            conv_output_channels,
            kernel_size=kernel_width,
            stride=stride,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.post_conv_layer_norm = nn.LayerNorm(normalized_shape=conv_output_channels)

        self.lstm = nn.LSTM(
            input_size=conv_output_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=0.1,
        )

        self.post_lstm_layer_norm = nn.LayerNorm(normalized_shape=lstm_hidden_size)
        self.projection = nn.Linear(lstm_hidden_size, output_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.compression(inputs)
        x = self.conv_layer(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        x = self.post_conv_layer_norm(x)
        x, _ = self.lstm(x)
        x = self.post_lstm_layer_norm(x)
        x = self.projection(x)
        x = x.permute(0, 2, 1)
        return x


class FingerStateMaskGenerator(nn.Module):
    def __init__(self, lpad: int = 0, rpad: int = 0) -> None:
        super().__init__()
        self.lpad = lpad
        self.rpad = rpad
        self.INDEX_FINGER = 0
        self.MIDDLE_FINGER = 1

    def forward(self, gesture_labels: torch.Tensor) -> torch.Tensor:
        batch_size, _, time_steps = gesture_labels.shape
        finger_masks = torch.zeros(
            (batch_size, 2, time_steps),
            device=gesture_labels.device,
            dtype=torch.float32,
        )

        for b in range(batch_size):
            self._process_finger(
                gesture_labels[b],
                finger_masks[b],
                press_channel=GestureType.index_press.value,
                release_channel=GestureType.index_release.value,
                output_channel=self.INDEX_FINGER,
                time_steps=time_steps,
            )
            self._process_finger(
                gesture_labels[b],
                finger_masks[b],
                press_channel=GestureType.middle_press.value,
                release_channel=GestureType.middle_release.value,
                output_channel=self.MIDDLE_FINGER,
                time_steps=time_steps,
            )

        return finger_masks

    def _process_finger(
        self,
        gesture_labels: torch.Tensor,
        finger_masks: torch.Tensor,
        press_channel: int,
        release_channel: int,
        output_channel: int,
        time_steps: int,
    ) -> None:
        press_signal = gesture_labels[press_channel]
        release_signal = gesture_labels[release_channel]

        zero_tensor = torch.zeros(1, device=gesture_labels.device)
        press_diff = torch.diff(press_signal, n=1, prepend=zero_tensor)
        release_diff = torch.diff(release_signal, n=1, prepend=zero_tensor)

        press_onsets = torch.nonzero(press_diff > 0, as_tuple=True)[0]
        release_onsets = torch.nonzero(release_diff > 0, as_tuple=True)[0]

        if press_onsets.numel() == 0 or release_onsets.numel() == 0:
            return

        for press_idx in press_onsets:
            future_releases = release_onsets[release_onsets > press_idx]
            if future_releases.numel() == 0:
                release_idx = torch.tensor(time_steps - 1, device=finger_masks.device)
            else:
                release_idx = future_releases[0]

            start_idx = torch.clamp(press_idx - self.lpad, min=0)
            end_idx = torch.clamp(release_idx + self.rpad + 1, max=time_steps)
            finger_masks[output_channel, start_idx:end_idx] = 1.0


# ==========================
# 训练/验证
# ==========================


def build_dataloaders() -> tuple[DataLoader, DataLoader, DataLoader]:
    data_split = DataSplit.from_csv(CSV_FILENAME, pool_test_partitions=True)

    transform = DiscreteGesturesTransform(pulse_window=PULSE_WINDOW)
    augmentation = RotationAugmentation(rotation=ROTATION_AUG)

    train_dataset = make_dataset(
        data_location=DATA_LOCATION,
        partition_dict=data_split.train,
        transform=transform,
        emg_augmentation=augmentation,
        window_length=WINDOW_LENGTH,
        stride=STRIDE,
        jitter=True,
        split_label="train",
    )

    val_dataset = make_dataset(
        data_location=DATA_LOCATION,
        partition_dict=data_split.val,
        transform=transform,
        emg_augmentation=None,
        window_length=WINDOW_LENGTH,
        stride=STRIDE,
        jitter=False,
        split_label="val",
    )

    test_dataset = make_dataset(
        data_location=DATA_LOCATION,
        partition_dict=data_split.test,
        transform=transform,
        emg_augmentation=None,
        window_length=None,
        stride=None,
        jitter=False,
        split_label="test",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=NUM_WORKERS,
        shuffle=False,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    mask_gen = FingerStateMaskGenerator(lpad=0, rpad=7).to(device)

    total_loss = 0.0
    total_count = 0

    for batch in loader:
        emg = batch["emg"].to(device)
        targets = batch["targets"].to(device)
        targets = targets[:, :, model.left_context :: model.stride]

        release_mask = mask_gen(targets)
        mask = torch.ones_like(targets)
        mask[:, [GestureType.index_release.value, GestureType.middle_release.value], :] = release_mask

        preds = model(emg)
        loss = loss_fn(preds, targets)
        loss = (loss * mask).sum() / mask.sum()

        total_loss += loss.item()
        total_count += 1

    return total_loss / max(total_count, 1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    mask_gen = FingerStateMaskGenerator(lpad=0, rpad=7).to(device)

    total_loss = 0.0
    total_count = 0

    for step, batch in enumerate(loader, start=1):
        emg = batch["emg"].to(device)
        targets = batch["targets"].to(device)
        targets = targets[:, :, model.left_context :: model.stride]

        release_mask = mask_gen(targets)
        mask = torch.ones_like(targets)
        mask[:, [GestureType.index_release.value, GestureType.middle_release.value], :] = release_mask

        preds = model(emg)
        loss = loss_fn(preds, targets)
        loss = (loss * mask).sum() / mask.sum()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VAL)
        optimizer.step()

        total_loss += loss.item()
        total_count += 1

        if step % LOG_EVERY == 0:
            logging.info("epoch %d step %d loss %.6f", epoch, step, loss.item())

    return total_loss / max(total_count, 1)


# ==========================
# 主程序
# ==========================


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    seed_everything(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and ACCELERATOR != "cpu" else "cpu"
    )
    logging.info("device=%s", device)

    train_loader, val_loader, test_loader = build_dataloaders()

    model = DiscreteGesturesArchitecture(output_channels=len(GestureType)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=WARMUP_START_FACTOR,
        end_factor=WARMUP_END_FACTOR,
        total_iters=WARMUP_TOTAL_EPOCHS,
    )

    multistep_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=LR_SCHEDULER_MILESTONES,
        gamma=LR_SCHEDULER_FACTOR,
    )

    scheduler = torch.optim.lr_scheduler.ChainedScheduler(
        [warmup_scheduler, multistep_scheduler]
    )

    best_val = math.inf
    os.makedirs("logs", exist_ok=True)

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = evaluate(model, val_loader, device)

        scheduler.step()

        logging.info("epoch %d train_loss %.6f val_loss %.6f", epoch, train_loss, val_loss)

        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = os.path.join("logs", "best_discrete_gestures.pt")
            torch.save({"model": model.state_dict()}, ckpt_path)
            logging.info("saved best checkpoint to %s", ckpt_path)

    test_loss = evaluate(model, test_loader, device)
    logging.info("test_loss %.6f", test_loss)


if __name__ == "__main__":
    main()
