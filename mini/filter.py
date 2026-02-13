

import numpy as np
from scipy.signal import butter, sosfiltfilt

class SEMGProcessor:
    def __init__(
        self,
        fs: float,
        lowpass: float = 850.0,
        highpass: float = 20.0,
        order: int = 4,
        notch_freq_base: float = 50.0,
        notch_harmonics: int = 9,
        notch_bandwidth: float = 6.0
    ):
        self.fs = fs

        self.sos_bandpass = butter(
            order,
            [highpass, lowpass],
            btype="bandpass",
            fs=fs,
            output="sos"
        )

        self.sos_notches = []
        for k in range(1, notch_harmonics + 1):
            f0 = notch_freq_base * k
            if f0 >= fs / 2:
                break
            sos = butter(
                2,
                [f0 - notch_bandwidth / 2, f0 + notch_bandwidth / 2],
                btype="bandstop",
                fs=fs,
                output="sos"
            )
            self.sos_notches.append(sos)

    def process(self, raw: np.ndarray):
        bandpass = sosfiltfilt(self.sos_bandpass, raw, axis=0)
        notched = bandpass
        for sos in self.sos_notches:
            notched = sosfiltfilt(sos, notched, axis=0)
        return bandpass, notched