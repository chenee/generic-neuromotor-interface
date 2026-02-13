import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from generic_neuromotor_interface.cler import compute_cler
from generic_neuromotor_interface.constants import EMG_NUM_CHANNELS

SEQ_LEN_DEFAULT = None
STRIDE_DEFAULT = 20


class OnlineNormalizer:
    """
    Adaptive normalization (dual-EMA) matching the embedded logic.
    """

    def __init__(
        self,
        num_channels=EMG_NUM_CHANNELS,
        alpha_mean=0.01,
        alpha_std=0.0001,
        init_mean=None,
        init_std=None,
    ):
        self.alpha_mean = alpha_mean
        self.alpha_std = alpha_std
        self.mean = (
            np.array(init_mean, dtype=np.float32)
            if init_mean is not None
            else np.zeros(num_channels, dtype=np.float32)
        )
        self.std = (
            np.array(init_std, dtype=np.float32)
            if init_std is not None
            else np.ones(num_channels, dtype=np.float32)
        )
        self.initialized = init_mean is not None
        self.gating_threshold = 2.0

    def update(self, window):
        curr_mean = window.mean(axis=0)
        curr_std = window.std(axis=0)

        z_score = np.abs(curr_mean - self.mean) / (self.std + 1e-6)
        is_rest = np.mean(z_score) < self.gating_threshold

        if not self.initialized:
            self.mean, self.std = curr_mean, np.maximum(curr_std, 1e-4)
            self.initialized = True
        elif is_rest:
            self.mean = self.alpha_mean * curr_mean + (1 - self.alpha_mean) * self.mean
            self.std = self.alpha_std * curr_std + (1 - self.alpha_std) * self.std
            self.std = np.maximum(self.std, 1e-4)

    def normalize(self, window, update=True):
        if update:
            self.update(window)
        return (window - self.mean) / (self.std + 1e-6)


class TFLiteWrapper:
    """Minimal TFLite wrapper (float32/int8) with [raw, abs] features."""

    def __init__(self, path, mean, std, seq_len=None):
        try:
            import tensorflow as tf
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "TensorFlow is required for TFLite inference. "
                "Install it with: pip install tensorflow"
            ) from exc

        self.interpreter = tf.lite.Interpreter(model_path=str(path))
        self.interpreter.allocate_tensors()

        inputs = self.interpreter.get_input_details()

        emg_input = None
        state_input = None

        for d in inputs:
            if "emg" in d["name"]:
                emg_input = d
            elif "state" in d["name"]:
                state_input = d

        if emg_input is None or state_input is None:
            ranked = sorted(inputs, key=lambda x: int(x["shape"][1]), reverse=True)
            if emg_input is None:
                emg_input = ranked[0]
            if state_input is None:
                state_input = ranked[1] if len(ranked) > 1 else ranked[0]

        self.emg_idx = emg_input["index"]
        self.is_quant = emg_input["dtype"] == np.int8
        self.emg_q = emg_input.get("quantization", (1.0, 0))

        self.state_in_idx = state_input["index"]
        self.state_q = state_input.get("quantization", (1.0, 0))
        self.state_shape = tuple(state_input["shape"].tolist())
        self.hidden_size = int(state_input["shape"][-1])

        outputs = self.interpreter.get_output_details()
        for d in outputs:
            if d["shape"][-1] == self.hidden_size:
                self.state_out_idx = d["index"]
                self.state_out_q = d.get("quantization", (1.0, 0))
            else:
                self.probs_idx = d["index"]
                self.probs_q = d.get("quantization", (1.0, 0))

        self.mean = mean
        self.std = std

        self.seq_len = int(seq_len) if seq_len is not None else int(emg_input["shape"][1])

        zp_in = self.state_q[1] if self.is_quant else 0
        self.hx = np.full(
            self.state_shape,
            zp_in,
            dtype=np.int8 if self.is_quant else np.float32,
        )

        if int(emg_input["shape"][1]) != self.seq_len:
            self.interpreter.resize_tensor_input(self.emg_idx, [1, self.seq_len, 32])
            self.interpreter.allocate_tensors()

    def __call__(self, x, warmup=False, apply_norm=True):
        if warmup:
            zp_in = self.state_q[1] if self.is_quant else 0
            self.hx = np.full(
                self.state_shape,
                zp_in,
                dtype=np.int8 if self.is_quant else np.float32,
            )

        if apply_norm:
            x = (x - self.mean) / (self.std + 1e-6)

        x_full = np.concatenate([x, np.abs(x)], axis=-1).astype(np.float32)
        inp = x_full[np.newaxis, ...]

        if self.is_quant:
            s, zp = self.emg_q
            inp = np.clip(np.round(inp / s + zp), -128, 127).astype(np.int8)

        self.interpreter.set_tensor(self.emg_idx, inp)
        self.interpreter.set_tensor(self.state_in_idx, self.hx)
        self.interpreter.invoke()

        raw_probs = self.interpreter.get_tensor(self.probs_idx)
        raw_state = self.interpreter.get_tensor(self.state_out_idx)

        s, zp = self.probs_q
        probs = (raw_probs.astype(np.float32) - zp) * s if self.is_quant else raw_probs
        probs = np.squeeze(probs)

        if self.is_quant:
            s_in, zp_in = self.state_q
            s_out, zp_out = self.state_out_q
            if s_in != s_out or zp_in != zp_out:
                f_state = (raw_state.astype(np.float32) - zp_out) * s_out
                raw_state = np.clip(np.round(f_state / s_in + zp_in), -128, 127).astype(
                    np.int8
                )
        self.hx = raw_state
        return probs


def load_dataset(path):
    with h5py.File(path, "r") as f:
        emg = f["data"]["emg"][:]
        times = f["data"]["time"][:]
    prompts = pd.read_hdf(path, "prompts")
    return emg, times, prompts


def run(args):
    dataset_path = Path(args.dataset).expanduser()
    tflite_path = Path(args.tflite).expanduser()

    emg, times, prompts = load_dataset(dataset_path)

    mean = np.zeros(EMG_NUM_CHANNELS, dtype=np.float32)
    std = np.ones(EMG_NUM_CHANNELS, dtype=np.float32)

    model = TFLiteWrapper(tflite_path, mean, std, args.seq_len)
    normalizer = OnlineNormalizer(init_mean=mean, init_std=std) if args.adaptive_norm else None

    seq_len = model.seq_len
    t_start = time.time()

    indices = np.arange(seq_len, len(emg), args.stride)
    if args.max_windows is not None:
        indices = indices[: args.max_windows]
    all_probs = []

    for idx, i in enumerate(indices):
        chunk = emg[i - seq_len : i]
        if normalizer is not None:
            start = max(0, i - args.stride)
            normalizer.update(emg[start:i])
            chunk = normalizer.normalize(chunk, update=False)
            probs = model(chunk, warmup=(idx == 0), apply_norm=False)
        else:
            probs = model(chunk, warmup=(idx == 0), apply_norm=True)
        all_probs.append(probs)

    probs_arr = np.stack(all_probs, axis=1)
    prob_times = times[indices - 1]

    cler = compute_cler(
        probs_arr,
        prob_times,
        prompts,
        threshold=args.threshold,
        debounce=args.debounce,
        tolerance=(args.tol_left, args.tol_right),
    )

    t_end = time.time()

    print(f"CLER: {cler:.4f}")
    print(f"Windows: {probs_arr.shape[1]} | Duration: {times[indices[-1]] - times[0]:.1f}s")
    print(f"Labels: {len(prompts)}")
    print(f"Inference wall time: {t_end - t_start:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tflite", required=True)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN_DEFAULT)
    parser.add_argument("--stride", type=int, default=STRIDE_DEFAULT)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--debounce", type=float, default=0.05)
    parser.add_argument("--tol-left", type=float, default=-0.05)
    parser.add_argument("--tol-right", type=float, default=0.25)
    parser.add_argument("--adaptive-norm", action="store_true")
    parser.add_argument("--max-windows", type=int, default=None)
    run(parser.parse_args())
