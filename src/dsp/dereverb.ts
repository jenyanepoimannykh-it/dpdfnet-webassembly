// A direct port of the DPDFNet inference loop in JenyaDereverb2's NeuralEnhancer.cpp:
// 960-point STFT, hop 480, Vorbis window, spec[1,1,481,2] + state_in[56436] in and
// spec_e + state_out back. The plug-in runs it sample-by-sample through ring buffers to
// stay real-time; offline there is no such constraint, so this walks whole frames and
// overlap-adds into the output, which produces the same signal without the 960-sample
// latency the plug-in has to report to its host.
import * as ort from 'onnxruntime-web/wasm'
import { Dft } from './fft'

export interface StateSegment {
  readonly offset: number
  readonly values: readonly number[]
}

export interface ModelMetadata {
  readonly modelType: string
  readonly profile: string
  readonly sampleRate: number
  readonly fftSize: number
  readonly hopSize: number
  readonly bins: number
  readonly windowType: string
  readonly stateSize: number
  readonly stateInit: readonly StateSegment[]
}

export interface ChannelProgress {
  /** Frames finished for this channel. */
  readonly frame: number
  readonly totalFrames: number
}

export interface ProcessOptions {
  readonly onProgress?: (progress: ChannelProgress) => void
  readonly shouldCancel?: () => boolean
}

export interface ChannelResult {
  /** Fully processed signal, sample-aligned with the input and the same length. */
  readonly wet: Float32Array
  /** Per-frame level change in dB, negative where the model removed energy. */
  readonly reductionDb: Float32Array
}

export class CancelledError extends Error {
  constructor() {
    super('processing cancelled')
    this.name = 'CancelledError'
  }
}

export interface LoadOptions {
  readonly modelUrl: string
  readonly metadataUrl: string
  /** Location of ort-wasm-simd-threaded.wasm; Vite hashes it, so it is passed in. */
  readonly wasmBinaryUrl: string
  readonly numThreads?: number
  readonly onModelProgress?: (loadedBytes: number, totalBytes: number) => void
}

/** How often the frame loop hands control back, in frames (~0.6 s of audio). */
const progressInterval = 64

/**
 * The network's own algorithmic delay, in windows. An impulse fed in at sample n leaves at
 * n + 2 * fftSize — 1920 samples, 40 ms at 48 kHz — which is confirmed three ways: the peak
 * of the measured impulse response, the cross-correlation lag against the input, and the
 * shift at which this pipeline lines up with dpdfnet.enhance() to 65 dB.
 *
 * The real-time plug-in cannot remove this; it reports one window of latency and leaves the
 * rest, which is why its dry and wet paths are 20 ms apart and why partial mix settings
 * comb-filter there. Offline there is no such constraint, so the whole delay is taken out
 * and the returned signal is sample-aligned with the input. The upstream offline path does
 * the same thing, by trimming 2 * win_len off the front of its ISTFT.
 */
const modelDelayWindows = 2

export class DereverbEngine {
  private readonly session: ort.InferenceSession
  readonly metadata: ModelMetadata
  private readonly window: Float64Array
  private readonly dft: Dft
  private readonly stateSeed: Float32Array

  private constructor(session: ort.InferenceSession, metadata: ModelMetadata) {
    this.session = session
    this.metadata = metadata
    const { fftSize } = metadata
    this.dft = new Dft(fftSize)
    // Vorbis power-complementary window: at 50% overlap the analysis and synthesis passes
    // multiply to sin^2 + cos^2 = 1, so the overlap-add needs no further normalisation.
    this.window = new Float64Array(fftSize)
    for (let i = 0; i < fftSize; i += 1) {
      const half = Math.sin((Math.PI * (i + 0.5)) / fftSize)
      this.window[i] = Math.sin(0.5 * Math.PI * half * half)
    }
    this.stateSeed = new Float32Array(metadata.stateSize)
    for (const segment of metadata.stateInit) {
      this.stateSeed.set(segment.values, segment.offset)
    }
  }

  static async load(options: LoadOptions): Promise<DereverbEngine> {
    ort.env.wasm.wasmPaths = { wasm: options.wasmBinaryUrl }
    ort.env.wasm.numThreads = options.numThreads ?? 1
    ort.env.logLevel = 'error'

    const [metadata, modelBytes] = await Promise.all([
      fetchMetadata(options.metadataUrl),
      fetchModel(options.modelUrl, options.onModelProgress),
    ])
    const session = await ort.InferenceSession.create(modelBytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    })
    const expected = ['spec', 'state_in']
    const missing = expected.filter((name) => !session.inputNames.includes(name))
    if (missing.length > 0) {
      throw new Error(`model does not expose the expected inputs (missing ${missing.join(', ')})`)
    }
    return new DereverbEngine(session, metadata)
  }

  /** Samples of network delay to take back out of the result. */
  private get modelDelay(): number {
    return modelDelayWindows * this.metadata.fftSize
  }

  /**
   * Frames covering `input` with `fftSize - hopSize` samples of silence padded onto each
   * end, which is exactly the state the plug-in's ring buffers start and end in. Every
   * original sample then falls under two windows and comes back at unity gain. The run is
   * extended by the model delay so the tail is not cut off once the result is shifted back.
   */
  private frameCount(length: number): number {
    const { fftSize, hopSize } = this.metadata
    const padded = length + this.modelDelay + 2 * (fftSize - hopSize)
    return Math.max(1, Math.ceil((padded - fftSize) / hopSize) + 1)
  }

  async processChannel(input: Float32Array, options: ProcessOptions = {}): Promise<ChannelResult> {
    const { fftSize, hopSize, bins, stateSize } = this.metadata
    const pad = fftSize - hopSize
    const delay = this.modelDelay
    const totalFrames = this.frameCount(input.length)
    const output = new Float32Array((totalFrames - 1) * hopSize + fftSize)
    const reductionDb = new Float32Array(totalFrames)

    const real = new Float64Array(fftSize)
    const imag = new Float64Array(fftSize)
    const spec = new Float32Array(bins * 2)
    const state = new Float32Array(stateSize)
    state.set(this.stateSeed)

    const specTensor = new ort.Tensor('float32', spec, [1, 1, bins, 2])
    const stateTensor = new ort.Tensor('float32', state, [stateSize])
    const feeds = { spec: specTensor, state_in: stateTensor }

    for (let frame = 0; frame < totalFrames; frame += 1) {
      if (frame % progressInterval === 0) {
        if (options.shouldCancel?.()) throw new CancelledError()
        options.onProgress?.({ frame, totalFrames })
      }

      const start = frame * hopSize - pad
      for (let i = 0; i < fftSize; i += 1) {
        const index = start + i
        const sample = index >= 0 && index < input.length ? input[index] : 0
        // One non-finite sample would otherwise live on in the overlap-add tail and in the
        // recurrent state, silencing everything after it.
        real[i] = Number.isFinite(sample) ? sample * this.window[i] : 0
        imag[i] = 0
      }
      this.dft.forward(real, imag)

      let before = 1e-12
      for (let bin = 0; bin < bins; bin += 1) {
        const re = real[bin]
        const im = imag[bin]
        spec[bin * 2] = re
        spec[bin * 2 + 1] = im
        before += re * re + im * im
      }

      const results = await this.session.run(feeds)
      const enhanced = results.spec_e.data as Float32Array
      const nextState = results.state_out.data as Float32Array

      // The model is recurrent, so a diverged frame would feed itself forever. Reseed the
      // state and drop the frame rather than latch a dead channel.
      if (!Number.isFinite(enhanced[0]) || !Number.isFinite(nextState[0])) {
        state.set(this.stateSeed)
        reductionDb[frame] = 0
        continue
      }
      state.set(nextState)

      let after = 1e-12
      for (let bin = 0; bin < bins; bin += 1) {
        const re = enhanced[bin * 2]
        const im = enhanced[bin * 2 + 1]
        real[bin] = re
        imag[bin] = im
        after += re * re + im * im
      }
      for (let bin = bins; bin < fftSize; bin += 1) {
        const mirror = fftSize - bin
        real[bin] = real[mirror]
        imag[bin] = -imag[mirror]
      }
      this.dft.inverse(real, imag)

      const scale = 1 / fftSize
      const base = start + pad
      for (let i = 0; i < fftSize; i += 1) {
        output[base + i] += real[i] * this.window[i] * scale
      }
      reductionDb[frame] = Math.min(0, 10 * Math.log10(after / before))
    }

    options.onProgress?.({ frame: totalFrames, totalFrames })
    // Skipping `delay` samples is what aligns the result with the input; the run was
    // lengthened by the same amount above so nothing is lost off the end.
    const from = pad + delay
    return { wet: output.subarray(from, from + input.length), reductionDb }
  }

  release(): Promise<void> {
    return this.session.release()
  }
}

async function fetchMetadata(url: string): Promise<ModelMetadata> {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`could not load model metadata (HTTP ${response.status})`)
  return (await response.json()) as ModelMetadata
}

/** Cache name is versioned with the weights so a new model does not serve a stale entry. */
const modelCacheName = 'deadroom-model-1'

async function fetchModel(
  url: string,
  onProgress?: (loadedBytes: number, totalBytes: number) => void,
): Promise<Uint8Array> {
  // The weights are the only download that hurts, and they are immutable for the life of
  // this model, so they are kept explicitly rather than left to HTTP revalidation.
  const cache = typeof caches !== 'undefined' ? await caches.open(modelCacheName).catch(() => null) : null
  const cached = await cache?.match(url)
  if (cached) return new Uint8Array(await cached.arrayBuffer())

  const response = await fetch(url)
  if (!response.ok) throw new Error(`could not load the model (HTTP ${response.status})`)
  if (cache) await cache.put(url, response.clone()).catch(() => undefined)
  const declared = Number(response.headers.get('content-length') ?? 0)
  if (!onProgress || !response.body || declared === 0) {
    return new Uint8Array(await response.arrayBuffer())
  }
  const reader = response.body.getReader()
  const chunks: Uint8Array[] = []
  let loaded = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    loaded += value.byteLength
    onProgress(loaded, declared)
  }
  const bytes = new Uint8Array(loaded)
  let offset = 0
  for (const chunk of chunks) {
    bytes.set(chunk, offset)
    offset += chunk.byteLength
  }
  return bytes
}
