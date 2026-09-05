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
| Network delay | 1920 samples (40 ms, two windows), removed. See below |

960 is not a power of two, so `src/dsp/fft.ts` implements the transform with Bluestein's
algorithm over a 2048-point radix-2 FFT. It agrees with a naive DFT to 1e-13.

### The 40 ms network delay

The network has an algorithmic delay of two windows. An impulse fed in at sample *n* leaves
at *n + 1920*, confirmed three ways: the peak of the measured impulse response, the
cross-correlation lag against the input, and the shift at which this pipeline lines up with
`dpdfnet.enhance()`. Upstream's offline path removes it by trimming `2 * win_len` off the
front of its ISTFT; the real-time path cannot, and reports only one window of latency.

Offline there is no constraint, so the whole delay is taken out here and the result is
sample-aligned with the input. That matters more than it sounds: it keeps exported audio in
sync with the video it came from, and it makes the mix control a real blend. Left
uncompensated, dry and wet sit 40 ms apart, the difference signal comes out **louder** than
the input (+3.4 dB), and intermediate mix settings comb-filter. Compensated, the difference
drops to −24.5 dB, 0.3% of loud bins are more than 90° out of phase, and the median phase
difference is 0.1°.

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

Against the upstream package itself, on the same input: **65.5 dB** versus
`dpdfnet.enhance()` with no shift, and **63.3 dB** versus `StreamEnhancer` when the delay
compensation is removed. Two independent implementations agreeing at 65 dB is ordinary
numerical difference — different FFT libraries, float32 accumulation, and reflect versus
zero padding at the first frame.

One upstream caveat worth knowing: `dpdfnet.enhance(..., onnx_path=...)` overrides the model
file but *not* `info.sample_rate`, so passing a 48 kHz model without also naming it silently
resamples the audio to 16 kHz and back. Pass `model=` alongside `onnx_path=`.

## What the model actually does

CEVA describe DPDFNet as a family of causal speech enhancement models for **real-time noise
suppression**, built on DeepFilterNet2 with Dual-Path RNN blocks. Noise suppression is what
it is good at; de-reverberation it does as a side effect. SI-SDR against the clean source,
48 kHz model:

| Condition | Input | After | Gain |
|---|---|---|---|
| White noise, 0 dB SNR | 0.0 dB | 14.9 dB | **+14.9** |
| Pink noise, 0 dB SNR | 0.5 dB | 13.3 dB | **+12.8** |
| White noise, +5 dB SNR | 5.0 dB | 17.8 dB | **+12.8** |
| Pink noise, +10 dB SNR | 10.3 dB | 20.7 dB | **+10.4** |
| Pink noise, +15 dB SNR | 15.0 dB | 23.0 dB | +7.9 |
| Reverb, RT60 0.7 s | 0.5 dB | 1.6 dB | +1.2 |
| Reverb + noise, +5 dB SNR | −1.7 dB | 1.2 dB | +3.0 |

Measured on reverberation alone, by the level in the gaps between words rather than SI-SDR,
it is more flattering — the model drives room tails to a floor near −57 dB whatever it is
given, so speech-to-tail improves by 4.9 dB at RT60 0.7 s and 6.3 dB at RT60 0.8 s. Both
readings are true; the model does audibly shorten a room, but it removes noise far better
than it removes reflections.

### Why this build ships `dpdfnet2_48khz_hr` and not `dpdfnet8_48khz_hr`

The 48 kHz family has two members and both have an identical ONNX signature — only
`state_size` differs (56436 against 90228), so either is a drop-in here. Measured, the
larger one is not worth it:

| | Params | MACs | Download | Speed (1 thread, native) | Best case gain |
|---|---|---|---|---|---|
| `dpdfnet2_48khz_hr` | 2.58 M | 2.42 G | 10.5 MB | 11.0× real time | — |
| `dpdfnet8_48khz_hr` | 3.63 M | 7.17 G | 14.9 MB | 3.7× real time | +0.1 to +0.4 dB |

Three times the compute and 4.4 MB more download for a fraction of a dB.

CEVA also publish 8 kHz and 16 kHz variants (`baseline` at 0.36 GMACs is seven times cheaper
than what ships here) which would suit weak devices or transcription prep, at the cost of
bandwidth. Note that `dpdfnet8_48khz_hr` carries the wrong `profile` string in its metadata
— it says `dpdfnet2_48khz_hr` — so models must be named locally rather than trusted from the
file.

## Notes on the interface

The mix knob is a true crossfade, `dry + mix * (wet - dry)`, applied identically in the
player and the exporter. It is equivalent to upstream's `--attn-limit-db`, which blends the
same two spectra with `alpha = 10 ** (-dB / 20)`; a mix of *m* is an attenuation limit of
`-20 * log10(1 - m)` dB.

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
