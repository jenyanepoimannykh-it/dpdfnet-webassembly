#!/usr/bin/env python3
"""Check that the browser's output matches the NumPy reference sample for sample.

Run the dev server, load a 48 kHz file in the page, then in DevTools:

    copy(JSON.stringify({length: deadroom.dry[0].length,
      dry: btoa(String.fromCharCode(...new Uint8Array(deadroom.dry[0].buffer))),
      wet: btoa(String.fromCharCode(...new Uint8Array(deadroom.wet[0].buffer)))}))

Save that to a file and pass it here. The window.deadroom handle only exists in dev builds.
Feeding the reference the browser's own decoded samples is what makes the comparison mean
something: any difference left is the pipeline's, not the media decoder's.

    python scripts/verify_against_reference.py browser-output.json
"""
import argparse
import base64
import json

import numpy as np
import onnxruntime as ort

from reference_dereverb import MODEL, dereverb, load_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", help="JSON written from the page, see the docstring")
    parser.add_argument("--min-snr", type=float, default=100.0, help="dB, fails below this")
    args = parser.parse_args()

    payload = json.loads(open(args.dump).read())
    unpack = lambda key: np.frombuffer(base64.b64decode(payload[key]), np.float32)
    dry, wet = unpack("dry"), unpack("wet")

    meta = load_metadata()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(MODEL), options, providers=["CPUExecutionProvider"])
    expected = dereverb(dry, meta, session)

    error = wet.astype(np.float64) - expected.astype(np.float64)
    signal = np.sqrt(np.mean(expected.astype(np.float64) ** 2))
    noise = np.sqrt(np.mean(error**2))
    snr = 20 * np.log10(signal / noise) if noise > 0 else float("inf")

    print(f"samples      {len(wet)}")
    print(f"max |error|  {np.max(np.abs(error)):.3e}  (signal peak {np.max(np.abs(expected)):.4f})")
    print(f"SNR          {snr:.1f} dB")
    if snr < args.min_snr:
        raise SystemExit(f"FAIL: {snr:.1f} dB is below the {args.min_snr:.0f} dB floor")
    print("PASS")


if __name__ == "__main__":
    main()
