#!/usr/bin/env python3
"""Independent NumPy reference for the browser pipeline.

Transcribed from the desktop plug-in's NeuralEnhancer.cpp rather than from src/, and it
uses numpy's own FFT instead of the site's Bluestein transform, so agreement between the
two is evidence and not a shared bug. Used by verify_against_reference.py.

    pip install numpy onnxruntime soundfile
    python scripts/reference_dereverb.py input.wav output.wav

The input must already be 48 kHz; this script does no resampling. In the browser that job
belongs to decodeAudioData.
"""
import argparse
import json
import pathlib

import numpy as np
import onnxruntime as ort
import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "public/models"


def _shipped_model() -> pathlib.Path:
    """Whichever network the site ships, so this cannot drift from src/models.ts."""
    found = sorted(MODEL_DIR.glob("*.onnx"))
    if len(found) != 1:
        raise SystemExit(f"expected exactly one .onnx in {MODEL_DIR}, found {len(found)}")
    return found[0]


MODEL = _shipped_model()
METADATA = MODEL.with_suffix(".meta.json")


def load_metadata() -> dict:
    if not METADATA.exists():
        raise SystemExit(f"{METADATA} is missing; run `node scripts/extract-model-metadata.mjs`")
    return json.loads(METADATA.read_text())


def vorbis_window(size: int) -> np.ndarray:
    """sin(pi/2 * sin^2(pi (i + 0.5) / N)), power-complementary at 50% overlap."""
    i = np.arange(size)
    return np.sin(0.5 * np.pi * np.sin(np.pi * (i + 0.5) / size) ** 2)


def seed_state(meta: dict) -> np.ndarray:
    state = np.zeros(meta["stateSize"], dtype=np.float32)
    for segment in meta["stateInit"]:
        values = np.asarray(segment["values"], dtype=np.float32)
        state[segment["offset"] : segment["offset"] + len(values)] = values
    return state


# The network's own algorithmic delay, in windows: an impulse fed in at sample n leaves at
# n + 2 * win_len. Upstream's offline path removes it by trimming 2 * win_len off the front
# of its ISTFT; the real-time path cannot and leaves it in.
MODEL_DELAY_WINDOWS = 2


def dereverb(samples: np.ndarray, meta: dict, session: ort.InferenceSession) -> np.ndarray:
    size, hop, bins = meta["fftSize"], meta["hopSize"], meta["bins"]
    pad = size - hop
    delay = MODEL_DELAY_WINDOWS * size
    window = vorbis_window(size)

    frames = max(1, int(np.ceil((len(samples) + delay + 2 * pad - size) / hop)) + 1)
    padded = np.concatenate(
        [np.zeros(pad, np.float64), samples.astype(np.float64), np.zeros(frames * hop + size)]
    )
    out = np.zeros(len(padded) + size, np.float64)
    state = seed_state(meta)

    for frame in range(frames):
        start = frame * hop
        block = padded[start : start + size] * window
        spectrum = np.fft.fft(block)
        spec = np.stack([spectrum[:bins].real, spectrum[:bins].imag], -1)
        spec = spec.astype(np.float32).reshape(1, 1, bins, 2)

        enhanced, state = session.run(["spec_e", "state_out"], {"spec": spec, "state_in": state})

        half = enhanced[0, 0, :, 0].astype(np.float64) + 1j * enhanced[0, 0, :, 1].astype(np.float64)
        full = np.empty(size, complex)
        full[:bins] = half
        # Hermitian mirror of the bins the model does not return.
        full[bins:] = np.conj(half[1 : size - bins + 1][::-1])
        out[start : start + size] += np.real(np.fft.ifft(full)) * window

    # Skipping `delay` samples aligns the result with the input; the run was lengthened by
    # the same amount above so the tail survives the shift.
    start = pad + delay
    return out[start : start + len(samples)].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()

    meta = load_metadata()
    audio, rate = sf.read(args.source, dtype="float32", always_2d=True)
    if rate != meta["sampleRate"]:
        raise SystemExit(f"{args.source} is {rate} Hz; the model needs {meta['sampleRate']} Hz")

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(MODEL), options, providers=["CPUExecutionProvider"])

    channels = [dereverb(audio[:, c], meta, session) for c in range(audio.shape[1])]
    result = np.stack(channels, axis=1)
    sf.write(args.destination, result, rate, subtype="FLOAT")

    before = np.mean(audio**2) + 1e-12
    after = np.mean(result**2) + 1e-12
    print(
        f"{args.destination}: {result.shape[1]} channel(s), {len(result) / rate:.2f} s, "
        f"level change {10 * np.log10(after / before):+.2f} dB"
    )


if __name__ == "__main__":
    main()
