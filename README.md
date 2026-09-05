# Hush

Neural noise removal for voice, running entirely in the browser. Drop in audio or video,
[DPDFNet](https://github.com/ceva-ip/DPDFNet) lifts the voice out from under the noise, and
you get a 48 kHz WAV back. No upload, no server, no account.

Two stages, stacked in the order the signal passes through them and each switchable on its
own:

1. **Dereverb** — weighted prediction error. A linear filter fitted to the reverberation and
   subtracted. No model, no weights.
2. **Denoise** — DPDFNet-8 through ONNX Runtime compiled to WebAssembly, the same network
   embedded in the JenyaDereverb2 VST3/AU plug-in.

On an M-series Mac, in Chrome, the chain runs about **2× faster than real time**; the
de-reverberation stage alone manages 44×.

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

## Why two stages

They are good at different things, and one of them is worth switching off. SI-SDR against
the clean source:

| Condition | Input | Dereverb | Denoise | **Both** |
|---|---|---|---|---|
| Reverb 0.7 s | 0.48 | 1.64 | 1.45 | **2.20** |
| Reverb 0.7 s + noise 5 dB | −1.76 | −1.23 | 1.27 | **1.65** |
| Reverb 0.5 s + noise 10 dB | 1.95 | 2.73 | 3.48 | **4.10** |
| Noise 5 dB, no room | 5.00 | 4.87 | **17.92** | 14.51 |

De-reverberation earns its place on a room and costs 3.4 dB on material that has none, which
is exactly why it is a stage with a switch rather than something applied silently. It
defaults to on at 60%, which keeps most of the gain on a room and halves the loss without
one.

## Stage one: de-reverberation

`src/dsp/wpe.ts`, the offline formulation published by NARA-WPE. Per bin, a filter over
`taps` past frames — starting `delay` frames back so the direct sound and early reflections
survive — is fitted to the observation weighted by the inverse of the current speech power,
and the prediction is subtracted. Two passes: the first weights by the observed power, the
second by the power of the first pass's residual.

4096-point STFT, hop 1024, 12 taps, delay 2, 2 iterations, fitted in blocks of 20 s with 2 s
of context either side. The fit only runs up to 8 kHz: reverberation energy is concentrated
low, and stopping there is eight times cheaper than the full band while keeping 89% of the
measured benefit. Measured on speech in an RT60 0.7 s room, it is worth **+1.23 dB** SI-SDR
and **+1.6 dB** of speech-to-tail ratio, at 44× real time.

`amount` plays no part in the fit — the residual returned is always the full one — so the
interface re-blends the stage without re-running it.

> This is **not** `OnlineWpe.cpp` from the plug-in repository. That file is a recursive
> least-squares variant which is absent from `CMakeLists.txt` and referenced by nothing;
> ported faithfully and measured, it made material worse (SI-SDR −13 dB, residual peaks
> above the input). It looks like abandoned v0.2 work. The batch form above is the one WPE
> is known for, and nothing here has to be causal anyway.

## Stage two: the network

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

### The model

`dpdfnet8_48khz_hr`, the largest of the family: 3.63 M parameters, 7.17 GMACs, 14.9 MB.
CEVA publish six others — the smaller `dpdfnet2_48khz_hr` is three times cheaper for
0.1 to 0.4 dB less, and the 8 and 16 kHz variants trade fullband quality for speed. All
share an identical ONNX signature, differing only in `state_size`, so swapping is a matter
of dropping a file into `public/models/` and re-running
`scripts/extract-model-metadata.mjs`.

`dpdfnet8_48khz_hr` carries the wrong `profile` string in its own metadata (it says
`dpdfnet2_48khz_hr`), so `src/models.ts` names the model locally rather than trusting the
file.

## Notes on the interface

The mix knob is a true crossfade, `dry + mix * (wet - dry)`, applied identically in the
player and the exporter. It is equivalent to upstream's `--attn-limit-db`, which blends the
same two spectra with `alpha = 10 ** (-dB / 20)`; a mix of *m* is an attenuation limit of
`-20 * log10(1 - m)` dB.

Each module plots what its own stage did: the stage's input as the filled envelope, its
output as the trace. Read down the rack and you follow the signal. The number on each is the
change in the quiet floor between words — where reverberation tails and steady noise both
live — rather than broadband level, which barely moves when a stage works well. Both
readings are clamped at −90 dBFS, since the network takes the gaps to digital silence and
the unclamped ratio runs past 80 dB and says nothing.

"Hear the original" bypasses both stages without moving the playhead.

Stereo is summed to mono by default. Speech enhancement gains nothing from a second
correlated channel and it doubles the work, but "Keep both channels" processes each with
its own recurrent state.

The charts plot level in dB, not amplitude. A noise floor sits 20–50 dB under the speech
above it, so on a linear axis the whole story is a few pixels tall.

**Play the example** runs the built-in clip through the real chain: 9 s of speech under pink
noise at 8 dB SNR in an RT60 0.7 s room. It is the same audio the idle charts are drawn
from, so the charts are a preview of the clip you can hear.
`scripts/make_demo_trace.py` regenerates both the audio and the traces together.

## Browser support

Needs WebAssembly, Web Audio, and Web Workers: current Chrome, Edge, Firefox and Safari.
Which containers can be opened is down to the browser's own decoders; WAV, MP3, M4A/AAC,
FLAC, Ogg and the audio track of MP4/MOV all work in Chrome and Safari. Video files come
back as a WAV — muxing the cleaned track back into the original video is not built yet.

## Licences

DPDFNet is © CEVA, Inc., under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
ONNX Runtime is © Microsoft, MIT. Archivo is under the SIL Open Font License.
