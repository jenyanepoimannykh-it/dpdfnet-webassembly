#!/usr/bin/env python3
"""Regenerate the built-in example: the audio the page offers to play, and the trace the
idle chart draws from it. Both come from one degradation so the chart is a preview of the
clip you can actually hear.

Degrades dry speech with pink noise and optionally a synthetic room, runs it through the
model via reference_dereverb.py, and writes both level envelopes as a small TypeScript
module. The idle chart is therefore a real measurement rather than a drawing.

    python scripts/make_demo_trace.py speech-48k-mono.wav
"""
import argparse
import pathlib
import subprocess

import numpy as np
import onnxruntime as ort
import soundfile as sf

from reference_dereverb import MODEL, ROOT, dereverb, load_metadata
from reference_wpe import dereverberate

BUCKETS = 600


def room_impulse(rate: int, rt60: float, drr_db: float, seed: int = 7) -> np.ndarray:
    """Direct sound plus an exponentially decaying noise tail at a chosen direct-to-
    reverberant ratio. Crude next to a measured room, but the decay law is the part that
    matters here."""
    rng = np.random.default_rng(seed)
    length = int(min(1.2, rt60 * 2) * rate)
    t = np.arange(length) / rate
    tail = rng.normal(size=length) * np.exp(-6.9 * t / rt60)
    tail[: int(0.0035 * rate)] = 0  # roughly a metre of extra path to the first reflection
    tail *= np.sqrt(10 ** (-drr_db / 10) / np.sum(tail**2))
    tail[0] = 1.0
    return tail


def pink_noise(length: int, rng: np.random.Generator) -> np.ndarray:
    """Roughly 1/f. Closer to room tone, fans and traffic than white noise is."""
    spectrum = np.fft.rfft(rng.normal(size=length))
    freqs = np.arange(len(spectrum), dtype=float)
    freqs[0] = 1.0
    return np.fft.irfft(spectrum / np.sqrt(freqs), n=length)


def envelope(samples: np.ndarray, buckets: int) -> np.ndarray:
    edges = np.linspace(0, len(samples), buckets + 1).astype(int)
    return np.array([np.max(np.abs(samples[a:b])) for a, b in zip(edges, edges[1:])])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("speech", help="dry 48 kHz mono speech")
    parser.add_argument("--noise-snr", type=float, default=8.0, help="dB; use inf for none")
    parser.add_argument("--rt60", type=float, default=0.7, help="0 disables the room")
    parser.add_argument("--drr", type=float, default=0.0, help="direct-to-reverberant ratio, dB")
    parser.add_argument("--seconds", type=float, default=9.0)
    parser.add_argument("--skip", type=float, default=0.4, help="seconds to trim from the start")
    parser.add_argument("--dereverb-amount", type=float, default=0.6, help="the page's default")
    args = parser.parse_args()

    meta = load_metadata()
    speech, rate = sf.read(args.speech, dtype="float64", always_2d=True)
    if rate != meta["sampleRate"]:
        raise SystemExit(f"{args.speech} is {rate} Hz; the model needs {meta['sampleRate']} Hz")
    speech = speech[:, 0]

    rng = np.random.default_rng(3)
    degraded = speech
    if args.rt60 > 0:
        degraded = np.convolve(degraded, room_impulse(rate, args.rt60, args.drr))[: len(speech)]
    if np.isfinite(args.noise_snr):
        noise = pink_noise(len(degraded), rng)
        noise /= np.sqrt(np.mean(noise**2))
        noise *= np.sqrt(np.mean(degraded**2)) / (10 ** (args.noise_snr / 20))
        degraded = degraded + noise
    degraded *= 0.35 / np.max(np.abs(degraded))

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(MODEL), options, providers=["CPUExecutionProvider"])

    # The whole chain, at the settings the page defaults to, so each module's idle chart
    # shows what that module does rather than what the one below it does.
    residual = dereverberate(degraded.astype(np.float32), rate)
    dereverbed = degraded + args.dereverb_amount * (residual - degraded)
    denoised = dereverb(dereverbed.astype(np.float32), meta, session)

    start, stop = int(args.skip * rate), int((args.skip + args.seconds) * rate)
    stop = min(stop, len(denoised))
    room = envelope(degraded[start:stop], BUCKETS)
    mid = envelope(dereverbed[start:stop], BUCKETS)
    dry = envelope(denoised[start:stop], BUCKETS)
    peak = room.max()
    room, mid, dry = room / peak, mid / peak, dry / peak

    # The audio the page plays is exactly the clip these envelopes were measured from.
    example_dir = ROOT / "public/example"
    example_dir.mkdir(parents=True, exist_ok=True)
    raw = example_dir / "example.wav"
    sf.write(raw, degraded[start:stop].astype(np.float32), rate, subtype="PCM_16")
    mp3 = example_dir / "example.mp3"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-ac", "1", "-b:a", "128k", str(mp3)],
        check=True,
    )
    raw.unlink()
    print(f"wrote {mp3} ({mp3.stat().st_size // 1024} kB)")

    trim = lambda x: (f"{x:.3f}".rstrip("0").rstrip(".") or "0")
    pack = lambda values: ",".join(trim(v) for v in values)
    out = ROOT / "src/ui/demo-trace.ts"
    out.write_text(
        f"""// The idle charts are a real measurement, not an illustration, and they are measured
// from the very clip the page offers to play. {args.seconds:.0f} seconds of speech under pink noise at
// {args.noise_snr:.0f} dB SNR in a room (RT60 {args.rt60} s, direct-to-reverberant {args.drr:+.0f} dB), run through
// the whole chain at its default settings by scripts/reference_wpe.py and
// scripts/reference_dereverb.py. Peak magnitude per bucket, normalised to the degraded
// peak. Regenerate with scripts/make_demo_trace.py.

const decode = (packed: string): Float32Array => Float32Array.from(packed.split(','), Number)

/** What the microphone heard: the input to stage one. */
export const demoNoisy = decode(
  '{pack(room)}',
)

/** After de-reverberation at {args.dereverb_amount:.0%}: stage one's output, stage two's input. */
export const demoDereverbed = decode(
  '{pack(mid)}',
)

/** After the network: stage two's output. */
export const demoClean = decode(
  '{pack(dry)}',
)
"""
    )
    print(f"wrote {out} ({out.stat().st_size} bytes, {BUCKETS} buckets)")


if __name__ == "__main__":
    main()
