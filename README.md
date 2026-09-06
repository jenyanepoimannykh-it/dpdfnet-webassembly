# Hush

Neural noise removal for voice, running entirely in the browser. Drop in audio or video,
[DPDFNet](https://github.com/ceva-ip/DPDFNet) lifts the voice out from under the noise, and
you get a 48 kHz WAV back. No upload, no server, no account.

The main interface is `/dereverb/`, with `/denoise/` as a separate secondary tool. Both run
the same DPDFNet pass, which is what the JenyaDereverb2 plug-in ships as its de-reverb: the
network takes room and noise out together and has no control for either separately. The two
pages differ in what they say and in the material they are pointed at, not in what runs.

One pass of DPDFNet-8 through ONNX Runtime compiled to WebAssembly — the same network
embedded in the JenyaDereverb2 VST3/AU plug-in. It removes noise and about a decibel of room
together, because that is what it was trained to do; it has no separate control for either.
On an M-series Mac, in Chrome, it runs about **2× faster than real time** on one thread.

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

### Why there is no separate de-reverberation control

DPDFNet takes one input and returns one output, with no mode switch, so the only way to
"use it for de-reverberation" is to run it twice. Measured, a second pass is worth **+0.03
dB** and costs a whole extra pass:

| Condition | Input | One pass | Two passes |
|---|---|---|---|
| Reverb 0.7 s | 0.48 | 1.45 | 1.48 |
| Reverb 0.7 s + noise 5 dB | −1.76 | 1.27 | 1.28 |
| Reverb 0.5 s + noise 10 dB | 1.95 | 3.48 | 3.49 |
| Noise 5 dB, no room | 5.00 | 17.92 | 17.67 |

It is a masking suppressor: after one pass the reverberation it knows how to attenuate is
already attenuated, and feeding it its own output gives it nothing new. `voice-enh` reached
the same conclusion independently — "one restrained noise-suppression pass; no second cleanup
pass, preserving a stable natural floor".

### The linear room filter, again

`src/dsp/wpe.ts` came back to run `/dereverb/` on its own and has been removed a second
time, so this is the record of what it was worth. Two of its constants were badly chosen and
had never been measured. The weighting floored each frame's power at an absolute `1e-12`,
which handed frames of near-silence weights in the billions and let them decide a fit that
had no reverberation in it to cancel — the floor belongs at about `1e-2` of the loudest
frame in the same band. And 12 taps reached only 299 ms, short of the tails the tool is
pointed at; 30 is the knee. Fitting up to 12 kHz rather than 8 kHz is worth another 0.1 to
0.3 dB, and above 12 kHz there is nothing left to take. Scored on the level in the gaps
between words of the page's own example clip:

| WPE as first written | −2.20 dB |
|---|---|
| WPE with all three fixed | −4.88 dB |
| **DPDFNet, one pass** | **−7.14 dB** |

Tuning it was worth a lot in relative terms and not enough in absolute ones. The network
takes half again as much room out of the same clip, and it is what the plug-in ships, so it
runs both pages and the linear filter is gone rather than kept as dead weight — the same
call, for the same reason, as the commit that removed it the first time. What it cost is
speed: tuned WPE ran at 9× real time where the network runs at about 2×.

Three things that look like they should help WPE do not, recorded so they are not tried a
third time. A second pass over the residual makes it *worse* (+2.62 → +1.69 dB at RT60 0.4):
after one fit the predictable tail is gone and the second starts cancelling signal. Delaying
the filter a third frame scores far better against a target that keeps 50 ms of early
reflections (+4.18 vs +2.62) and exactly the same against one that keeps 25 ms — the metric
being flattered for removing less, not more room coming out. And a shorter analysis window,
which should localise the tail better, loses across the board: 2048-point scores +0.60 dB
where 4096 scores +2.62.

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

Each pass is all or nothing. The Off/On switch is the crossfade `dry + mix * (wet - dry)`
pinned to its ends, applied identically in the player and the exporter: On is the processed
signal alone, Off is the original alone, and both are whole. The blend itself is still
equivalent to upstream's `--attn-limit-db`, which mixes the same two spectra with
`alpha = 10 ** (-dB / 20)`, should an intermediate setting ever come back.

The chart plots what the pass did: the input as the filled envelope, the output as the
trace. What still shows through the fill is what came out. The number beside it is the change
in the quiet floor between words — where reverberation tails and steady noise both live —
rather than broadband level, which barely moves when the model works well: speech dominates
the average, and speech is what it is trying to keep. It is clamped at −90 dBFS, since the
network takes the gaps to digital silence and the unclamped ratio runs past 80 dB and says
nothing.

Stereo is summed to mono by default. Speech enhancement gains nothing from a second
correlated channel and it doubles the work, but "Keep both channels" processes each with
its own recurrent state.

The chart plots level in dB, not amplitude. A noise floor sits 20–50 dB under the speech
above it, so on a linear axis the whole story is a few pixels tall.

**Play the example** runs the built-in clip through the real model: 9 s of speech under pink
noise at 8 dB SNR in an RT60 0.7 s room. It is the same audio the idle chart is drawn from,
so the chart is a preview of the clip you can hear. `scripts/make_demo_trace.py` regenerates
both the audio and the trace together.

## Browser support

Needs WebAssembly, Web Audio, and Web Workers: current Chrome, Edge, Firefox and Safari.
Which containers can be opened is down to the browser's own decoders; WAV, MP3, M4A/AAC,
FLAC, Ogg and the audio track of MP4/MOV all work in Chrome and Safari. Video files come
back as a WAV — muxing the cleaned track back into the original video is not built yet.

## Licences

DPDFNet is © CEVA, Inc., under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
ONNX Runtime is © Microsoft, MIT. Archivo is under the SIL Open Font License.
