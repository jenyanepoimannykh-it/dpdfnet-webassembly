# Deadroom

Neural room removal for voice, running entirely in the browser. Drop in audio or video,
[DPDFNet](https://github.com/ceva-ip/DPDFNet) removes the reverberation, and you get a
48 kHz WAV back. No upload, no server, no account.

The model is the same one embedded in the JenyaDereverb2 VST3/AU plug-in, running through
ONNX Runtime compiled to WebAssembly. On an M-series Mac it processes roughly **5.7× faster
than real time** on one thread.

## Running it

```sh
npm install
npm run dev
```

The model weights are already in `public/models/`. `npm run build` emits a static `dist/`
that can be served from anywhere; set `BASE_PATH` if it will live in a subdirectory:

```sh
BASE_PATH=/deadroom/ npm run build
```

## How the audio path works

`src/dsp/dereverb.ts` is a port of `NeuralEnhancer.cpp` from the plug-in:

| | |
|---|---|
| Sample rate | 48 kHz, enforced — `decodeAudioData` resamples on the way in |
| Transform | 960-point STFT, hop 480, Vorbis window (power-complementary at 50% overlap) |
| Model I/O | `spec[1,1,481,2]` + `state_in[56436]` → `spec_e` + `state_out` |
| State | Recurrent, per channel, seeded from the model's own `erb_norm_init` / `spec_norm_init` metadata |
| Latency | None. The plug-in reports 960 samples because it must stream; offline the frames are simply walked, so the output is sample-aligned with the input |

960 is not a power of two, so `src/dsp/fft.ts` implements the transform with Bluestein's
algorithm over a 2048-point radix-2 FFT. It agrees with a naive DFT to 1e-13.

onnxruntime-web cannot read a model's custom metadata map, so the recurrent state seed is
lifted out at build time by `scripts/extract-model-metadata.mjs` into
`public/models/dpdfnet2_48khz_hr.meta.json`. Re-run it if the weights are ever replaced.

## Verifying the port

`scripts/reference_dereverb.py` is an independent NumPy implementation, transcribed from
the C++ rather than from `src/`, using numpy's own FFT. Feeding both the same samples:

```
samples      384000
max |error|  2.421e-08  (signal peak 0.1301)
SNR          138.7 dB
PASS
```

That is float32 precision — the two pipelines are the same pipeline. See
`scripts/verify_against_reference.py` for how to capture the browser's output.

## Does it actually work?

Speech convolved with synthetic rooms, measured as the level in the gaps between words
(where reverberation lives and nothing else does):

| Room | Tails before | Tails after | Speech-to-tail ratio |
|---|---|---|---|
| RT60 0.35 s, DRR +6 dB | −63.8 dB | −57.8 dB | 38.1 → 31.2 dB |
| RT60 0.50 s, DRR +3 dB | −57.5 dB | −58.3 dB | 32.1 → 31.9 dB |
| RT60 0.60 s, DRR 0 dB | −53.2 dB | −57.6 dB | 28.5 → **31.6** dB |
| RT60 0.80 s, DRR −3 dB | −48.3 dB | −56.2 dB | 24.2 → **30.5** dB |

The model drives room tails to a floor near −57 dB whatever it is given, so the wetter the
room the more it gains. The first row is a room already drier than that floor — the model
has nothing to remove and adds a little of its own noise, 57 dB down and inaudible.

## Notes on the interface

The mix knob is a true crossfade, `dry + mix * (wet - dry)`, applied identically in the
player and the exporter. **Intermediate settings can cancel**: the model reconstructs
spectral phase, so dry and wet are not phase-coherent and individual bins can null. Both
ends of the knob are safe; the page says so when the knob leaves them.

Stereo is summed to mono by default. Speech de-reverberation gains nothing from a second
correlated channel and it doubles the work, but "Keep both channels" processes each with
its own recurrent state.

The idle chart is a real measurement — 9 s of speech through a synthetic room and back
through this model — not a drawing. Regenerate it with `scripts/make_demo_trace.py`.

## Browser support

Needs WebAssembly, Web Audio, and Web Workers: current Chrome, Edge, Firefox and Safari.
Which containers can be opened is down to the browser's own decoders; WAV, MP3, M4A/AAC,
FLAC, Ogg and the audio track of MP4/MOV all work in Chrome and Safari. Video files come
back as a WAV — muxing the cleaned track back into the original video is not built yet.

## Licences

DPDFNet is © CEVA, Inc., under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
ONNX Runtime is © Microsoft, MIT. Archivo is under the SIL Open Font License.
