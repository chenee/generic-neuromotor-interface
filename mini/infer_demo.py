import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import warnings
import h5py
import numpy as np
import pandas as pd
import torch
from model import SEMG_MiniRNN

# Suppress environment-specific warnings
warnings.filterwarnings("ignore", message=".*weights_only.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.serialization")
warnings.filterwarnings("ignore", message=".*tf.lite.Interpreter is deprecated.*")

# Global hyper-parameters
SEQ_LEN = 50
STRIDE = 20
SAMPLING_RATE = 2000

class OnlineNormalizer:
    """
    Adaptive Normalization using Dual-EMA baseline tracking.
    Matches the C++ OnlineNormalizer logic for cross-user generalization.
    """
    def __init__(self, num_channels=16, alpha_mean=0.01, alpha_std=0.0001, init_mean=None, init_std=None):
        self.alpha_mean = alpha_mean
        self.alpha_std = alpha_std
        self.mean = np.array(init_mean, dtype=np.float32) if init_mean is not None else np.zeros(num_channels)
        self.std = np.array(init_std, dtype=np.float32) if init_std is not None else np.ones(num_channels)
        self.initialized = init_mean is not None
        self.gating_threshold = 2.0 # Z-score threshold to stop updates during gestures

    def update(self, window):
        curr_mean = window.mean(axis=0)
        curr_std = window.std(axis=0)
        
        # Determine if signal is in 'Rest' state to update baseline statistics
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
        if update: self.update(window)
        return (window - self.mean) / (self.std + 1e-6)

class TFLiteWrapper:
    """Efficient wrapper for TFLite inference (Float32 or Int8)."""
    def __init__(self, path, mean, std, hidden_size=128):
        import tensorflow as tf
        self.interpreter = tf.lite.Interpreter(path)
        self.interpreter.allocate_tensors()
        
        # Identify tensor indices and quantization parameters
        inputs = self.interpreter.get_input_details()
        for d in inputs:
            if 'emg' in d['name']:
                self.emg_idx, self.is_quant = d['index'], (d['dtype'] == np.int8)
                self.emg_q = d.get('quantization', (1.0, 0))
            elif 'state' in d['name']:
                self.state_in_idx, self.state_q = d['index'], d.get('quantization', (1.0, 0))
        
        outputs = self.interpreter.get_output_details()
        for d in outputs:
            if d['shape'][-1] == hidden_size:
                self.state_out_idx, self.state_out_q = d['index'], d.get('quantization', (1.0, 0))
            else:
                self.probs_idx, self.probs_q = d['index'], d.get('quantization', (1.0, 0))

        self.mean, self.std, self.hidden_size = mean, std, hidden_size
        # Correctly initialize hx to zero_point for Int8 or zero for Float32
        zp_in = self.state_q[1] if self.is_quant else 0
        self.hx = np.full((1, hidden_size), zp_in, dtype=np.int8 if self.is_quant else np.float32)
        
        # Resize input for specific SEQ_LEN
        self.interpreter.resize_tensor_input(self.emg_idx, [1, SEQ_LEN, 32])
        self.interpreter.allocate_tensors()

    def __call__(self, x, warmup=False, apply_norm=True):
        if warmup:
            zp_in = self.state_q[1] if self.is_quant else 0
            self.hx = np.full((1, self.hidden_size), zp_in, dtype=np.int8 if self.is_quant else np.float32)

        if apply_norm: x = (x - self.mean) / (self.std + 1e-6)
        
        # Feature engineering: [Raw, Abs]
        x_full = np.concatenate([x, np.abs(x)], axis=-1).astype(np.float32)
        inp = x_full[np.newaxis, ...]
        
        if self.is_quant:
            q = lambda v, p: np.clip(np.round(v / p[0] + p[1]), -128, 127).astype(np.int8)
            inp = q(np.clip(inp, -7.0, 7.0), self.emg_q)
        
        self.interpreter.set_tensor(self.emg_idx, inp)
        self.interpreter.set_tensor(self.state_in_idx, self.hx)
        self.interpreter.invoke()
        
        raw_probs = self.interpreter.get_tensor(self.probs_idx)[0]
        raw_state = self.interpreter.get_tensor(self.state_out_idx)
        
        # Dequantize probabilities
        s, zp = self.probs_q
        probs = (raw_probs.astype(np.float32) - zp) * s if self.is_quant else raw_probs
        
        # Handle state quantization re-mapping if scales differ
        if self.is_quant:
            s_in, zp_in = self.state_q
            s_out, zp_out = self.state_out_q
            if s_in != s_out or zp_in != zp_out:
                f_state = (raw_state.astype(np.float32) - zp_out) * s_out
                raw_state = q(f_state, self.state_q)
        self.hx = raw_state
        return probs

class FingerStateMachine:
    """Prevents logically impossible finger state transitions."""
    def __init__(self, id_to_name):
        self.state = "NEUTRAL"
        self.id_to_name = id_to_name

    def should_filter(self, gid):
        name = self.id_to_name.get(gid, "")
        if self.state == "NEUTRAL": return name in ("index_release", "middle_release")
        if self.state == "INDEX": return name == "middle_release"
        if self.state == "MIDDLE": return name == "index_release"
        return False

    def update(self, gid):
        name = self.id_to_name.get(gid, "")
        if name == "index_press": self.state = "INDEX"
        elif name == "middle_press": self.state = "MIDDLE"
        elif name in ("index_release", "middle_release"): self.state = "NEUTRAL"

class GestureDebouncer:
    """Hysteresis and competition logic for robust detection."""
    def __init__(self, threshold=0.4, margin=0.15, lockout=20):
        self.threshold, self.margin, self.lockout_frames = threshold, margin, lockout
        self.energies = np.zeros(9)
        self.lockouts = {}
        self.global_refractory = 0

    def process(self, probs):
        for gid in list(self.lockouts.keys()):
            self.lockouts[gid] -= 1
            if self.lockouts[gid] <= 0: del self.lockouts[gid]
        if self.global_refractory > 0: self.global_refractory -= 1

        # Energy integration (Alpha=0.1)
        self.energies = self.energies * 0.9 + probs.flatten() * 0.1
        
        best_id = np.argmax(self.energies)
        sorted_p = np.sort(probs.flatten())
        is_ambiguous = (sorted_p[-1] - sorted_p[-2]) < self.margin
        
        if not is_ambiguous and self.energies[best_id] > self.threshold:
            if best_id not in self.lockouts and self.global_refractory == 0:
                self.lockouts[best_id] = self.lockout_frames
                self.global_refractory = 5 # 50ms global gap
                self.energies[:] = 0
                return int(best_id)
        return -1

class GTEvaluator:
    """Matches detections to ground truth prompts to calculate CLER."""
    def __init__(self, df, id_to_name, tol=0.2):
        self.df = df.reset_index(drop=True)
        self.id_to_name, self.tol = id_to_name, tol
        self.results = ["Missed"] * len(self.df)
        self.fps = {}
        self.del_ptr = 0

    def process_detection(self, gid, ts):
        mask = self.df['time'].between(ts - self.tol, ts + self.tol)
        matches = self.df.index[mask].tolist()
        
        best_match = -1
        min_off = float('inf')
        for idx in matches:
            if self.results[idx] == "Missed":
                off = abs(self.df.loc[idx, 'time'] - ts)
                if off < min_off: min_off, best_match = off, idx

        if best_match != -1:
            gt_n, gt_t = self.df.loc[best_match, 'name'], self.df.loc[best_match, 'time']
            if gt_n == self.id_to_name[gid]:
                self.results[best_match] = "Correct"
                return "✔", f" (off: {(ts-gt_t)*1000:+.0f}ms)"
            self.results[best_match] = f"Sub:{self.id_to_name[gid]}"
            return "✘", f" ({gt_n}) (off: {(ts-gt_t)*1000:+.0f}ms)"
        
        self.fps[gid] = self.fps.get(gid, 0) + 1
        return "+", ""

    def cleanup(self, ts):
        dels = []
        while self.del_ptr < len(self.df) and self.df.loc[self.del_ptr, 'time'] < ts - self.tol:
            if self.results[self.del_ptr] == "Missed":
                self.results[self.del_ptr] = "Deleted"
                dels.append((self.df.loc[self.del_ptr, 'time'], self.df.loc[self.del_ptr, 'name']))
            self.del_ptr += 1
        return dels

    def get_summary_stats(self):
        stats = []
        class_errors = []
        for gid, name in self.id_to_name.items():
            idx = self.df[self.df['name'] == name].index
            if len(idx) == 0: continue
            
            matches = sum(1 for k in idx if self.results[k] == "Correct")
            subs = sum(1 for k in idx if "Sub:" in str(self.results[k]))
            dels = sum(1 for k in idx if self.results[k] in ("Missed", "Deleted"))
            ins = self.fps.get(gid, 0)
            
            err = (subs + dels) / len(idx)
            class_errors.append(err)
            
            stats.append({
                'name': name, 'count': len(idx), 'matches': matches,
                'subs': subs, 'dels': dels, 'ins': ins, 'err': err
            })
            
        return stats, np.mean(class_errors) if class_errors else 0

def run_simulation(args):
    # 1. Initialization
    checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
    id_to_name = checkpoint.get('id_to_name')
    stats = checkpoint.get('norm_stats', {'mean': np.zeros(16), 'std': np.ones(16)})
    mean, std = np.array(stats['mean']), np.array(stats['std'])
    
    # 2. Model Loading
    if args.tflite:
        model = TFLiteWrapper(args.tflite, mean, std)
    else:
        model_obj = SEMG_MiniRNN(num_gestures=checkpoint['num_gestures'], hidden_size=checkpoint['hidden_size'])
        model_obj.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model_obj.eval()
        hx = None
        def model(x, warmup=False, apply_norm=True):
            nonlocal hx
            if warmup: hx = None
            if apply_norm: x = (x - mean) / (std + 1e-6)
            with torch.no_grad():
                logits, hx = model_obj(torch.from_numpy(x[None].astype(np.float32)), hx=hx)
                return torch.sigmoid(logits).numpy().flatten()

    # 3. Data and Simulation Loop
    with h5py.File(args.dataset, 'r') as f:
        emg = f['data']['emg'][:]
        ts = f['data']['time'][:]
        prompts = pd.read_hdf(args.dataset, 'prompts')
    
    debouncer = GestureDebouncer(args.threshold, args.margin, args.lockout)
    finger_state = FingerStateMachine(id_to_name)
    evaluator = GTEvaluator(prompts, id_to_name, args.tolerance)
    normalizer = OnlineNormalizer(init_mean=mean, init_std=std) if args.adaptive_norm else None
    
    line_w = 70
    print("-" * line_w)
    print(f"Mini-RNN Real-time Simulation | Dataset: {args.dataset}")
    print(f"Threshold: {args.threshold}, Margin: {args.margin}, Lockout: {args.lockout}")
    print(f"AdaptiveNorm: {'ON' if args.adaptive_norm else 'OFF'}")
    print("-" * line_w)

    indices = np.arange(SEQ_LEN, len(emg), STRIDE)
    filtered_count = 0
    total_detections = 0
    
    for idx, i in enumerate(indices):
        chunk = emg[i - SEQ_LEN : i]
        if normalizer:
            normalizer.update(emg[i - STRIDE : i])
            chunk = normalizer.normalize(chunk, update=False)
        
        probs = model(chunk, warmup=(idx == 0), apply_norm=(not args.adaptive_norm))
        gid = debouncer.process(probs)
        
        if gid != -1:
            if finger_state.should_filter(gid):
                filtered_count += 1
                continue
            
            finger_state.update(gid)
            total_detections += 1
            sym, info = evaluator.process_detection(gid, ts[i-1])
            top3 = [f"{id_to_name[k]}/{probs[k]:.2f}" for k in np.argsort(probs)[::-1][:3] if probs[k] >= 0.2]
            print(f"{ts[i-1]:.3f}: [{sym}] {', '.join(top3)}{info}")
            
        if idx % 200 == 0:
            for t, n in evaluator.cleanup(ts[i-1]): print(f"{t:.3f}: [-] {n}")

    # 4. Final Report
    total_seconds = len(emg) / SAMPLING_RATE
    stats, avg_cler = evaluator.get_summary_stats()
    
    print("\n" + "=" * line_w)
    print(f"FINAL SUMMARY ({total_seconds:.1f}s simulated)")
    print("-" * line_w)
    print(f"{'Gesture':<18} | {'GT':>3} | {'M':>3} | {'S':>3} | {'D':>3} | {'I':>3} | {'Error'}")
    print("-" * line_w)
    
    total_gt = total_m = total_s = total_d = total_i = 0
    for s in stats:
        total_gt += s['count']; total_m += s['matches']
        total_s += s['subs']; total_d += s['dels']; total_i += s['ins']
        print(f"{s['name']:<18} | {s['count']:>3} | {s['matches']:>3} | {s['subs']:>3} | {s['dels']:>3} | {s['ins']:>3} | {s['err']:>6.1%}")
    
    total_err = (total_s + total_d) / total_gt if total_gt > 0 else 0
    precision = total_m / total_detections if total_detections > 0 else 0
    
    print("-" * line_w)
    print(f"{'TOTAL':<18} | {total_gt:>3} | {total_m:>3} | {total_s:>3} | {total_d:>3} | {total_i:>3} | {total_err:>6.1%}")
    print("=" * line_w)
    print(f"Balanced CLER: {avg_cler:.2%} | Precision: {precision:.2%} | FA/min: {total_i / (total_seconds / 60.0):.2f}")
    if filtered_count > 0:
        print(f"State Machine Filtered: {filtered_count} invalid releases")
    print("=" * line_w)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("--model", type=str, default="emg_models/discrete_gestures/mini_rnn_v1.pth")
    parser.add_argument("--tflite", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--lockout", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=0.2)
    parser.add_argument("--adaptive-norm", action="store_true")
    parser.add_argument("--margin", type=float, default=0.15)
    run_simulation(parser.parse_args())
