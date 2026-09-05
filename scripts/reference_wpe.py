#!/usr/bin/env python3
"""NumPy reference for the de-reverberation stage, mirroring src/dsp/wpe.ts.

Offline weighted prediction error, the NARA-WPE formulation. Per frequency bin, a filter
over `TAPS` past frames -- starting `DELAY` frames back so the direct sound and early
reflections survive -- is fitted to the observation weighted by the inverse of the current
speech power, and the prediction is subtracted.

The constants below must match src/dsp/wpe.ts. They are what the site actually runs, so
anything generated here (the idle charts, for instance) describes the shipped behaviour.

    python scripts/reference_wpe.py input.wav output.wav
"""
import argparse

import numpy as np
import soundfile as sf

N_FFT = 4096
HOP = N_FFT // 4
TAPS = 12
DELAY = 2
ITERATIONS = 2
# Reverberation energy is concentrated low, and the fit costs scale with the bin count.
FIT_CEILING_HZ = 8000
# Hann applied on both analysis and synthesis at 75% overlap sums to 1.5.
OVERLAP_SCALE = 2.0 / 3.0


def _window() -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)


def _stft(x: np.ndarray) -> np.ndarray:
    frames = max(1, int(np.ceil((len(x) + N_FFT) / HOP)))
    # Frame f covers input [(f + 1) * HOP - N_FFT, ... + N_FFT), matching the TypeScript.
    # In padded coordinates, where padded[k] is input sample k - N_FFT, that starts at
    # (f + 1) * HOP.
    padded = np.concatenate([np.zeros(N_FFT), x, np.zeros((frames + 1) * HOP + 2 * N_FFT)])
    index = np.arange(N_FFT)[None, :] + HOP * (np.arange(frames) + 1)[:, None]
    return np.fft.rfft(padded[index] * _window(), axis=1)


def _istft(spec: np.ndarray, length: int) -> np.ndarray:
    frames = spec.shape[0]
    out = np.zeros((frames + 2) * HOP + 2 * N_FFT)
    blocks = np.fft.irfft(spec, n=N_FFT, axis=1) * _window() * OVERLAP_SCALE
    for f in range(frames):
        start = (f + 1) * HOP
        out[start : start + N_FFT] += blocks[f]
    return out[N_FFT : N_FFT + length]


def _smooth(values: np.ndarray) -> np.ndarray:
    padded = np.concatenate([values[:1], values, values[-1:]])
    return 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]


def dereverberate(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Returns the full residual, sample-aligned with the input. Blend it yourself."""
    observed = _stft(samples.astype(np.float64))
    frames, bin_count = observed.shape
    fit_bins = min(bin_count, int(round(FIT_CEILING_HZ / (sample_rate / 2) * (N_FFT // 2))) + 1)
    estimate = observed.copy()

    for _ in range(ITERATIONS):
        for bin_index in range(fit_bins):
            power = _smooth(np.abs(estimate[:, bin_index]) ** 2)
            weight = 1.0 / np.maximum(power, 1e-12)

            usable = np.arange(DELAY + TAPS, frames)
            if usable.size == 0:
                continue
            # x[t, l] = observed[t - DELAY - l]
            x = np.stack([observed[usable - DELAY - l, bin_index] for l in range(TAPS)], axis=1)
            w = weight[usable]
            r_matrix = (x * w[:, None]).T @ x.conj()
            r_vector = (x * w[:, None]).T @ observed[usable, bin_index].conj()

            ridge = 1e-6 * np.trace(r_matrix).real / TAPS + 1e-12
            r_matrix = r_matrix + np.eye(TAPS) * ridge
            try:
                gains = np.linalg.solve(r_matrix, r_vector)
            except np.linalg.LinAlgError:
                continue

            prediction = np.zeros(frames, dtype=complex)
            prediction[usable] = x @ gains.conj()
            estimate[:, bin_index] = observed[:, bin_index] - prediction

    return _istft(estimate, len(samples)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--amount", type=float, default=1.0, help="blend against the input")
    args = parser.parse_args()

    audio, rate = sf.read(args.source, dtype="float32", always_2d=True)
    channels = []
    for index in range(audio.shape[1]):
        dry = audio[:, index]
        residual = dereverberate(dry, rate)
        channels.append(dry + args.amount * (residual - dry))
    sf.write(args.destination, np.stack(channels, axis=1), rate, subtype="FLOAT")
    print(f"{args.destination}: {len(channels)} channel(s), amount {args.amount:.2f}")


if __name__ == "__main__":
    main()
