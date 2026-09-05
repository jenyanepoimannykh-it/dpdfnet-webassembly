// The signal chain runs here so a long file never blocks the interface: de-reverberation
// first, then the network. That order is deliberate — weighted prediction error fits a
// linear model to the observed reverberation, and it wants the signal before a non-linear
// suppressor has been near it.
//
// The de-reverberation fit is cached per source. Changing its blend, or switching the
// network off, re-runs only what actually depends on the change.
import wasmBinaryUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url'
import { CancelledError, DereverbEngine } from '../dsp/dereverb'
import { WpeDereverb } from '../dsp/wpe'
import { modelUrls } from '../models'
import type { ProcessRequest, WorkerRequest, WorkerResponse } from './protocol'

interface WorkerScope {
  postMessage(message: WorkerResponse, transfer?: Transferable[]): void
  addEventListener(type: 'message', listener: (event: MessageEvent<WorkerRequest>) => void): void
}

const scope = self as unknown as WorkerScope
const post = (message: WorkerResponse, transfer?: Transferable[]) => scope.postMessage(message, transfer)

let engine: DereverbEngine | null = null
let loading: Promise<DereverbEngine> | null = null
let cancelRequested = false

const wpe = new WpeDereverb()
let cache: { source: number; dry: Float32Array[]; residual: Float32Array[] } | null = null

/** Intra-op threads for the convolutions. Capped low: the graph is small and per-frame,
 *  so beyond a few threads the synchronisation costs more than it saves. */
function threadCount(): number {
  if (typeof crossOriginIsolated !== 'undefined' && !crossOriginIsolated) return 1
  const cores = navigator.hardwareConcurrency ?? 1
  return Math.max(1, Math.min(4, cores - 1))
}

async function ensureEngine(): Promise<DereverbEngine> {
  if (engine) return engine
  if (loading) return loading
  const startedAt = performance.now()
  loading = DereverbEngine.load({
    ...modelUrls(),
    wasmBinaryUrl,
    numThreads: threadCount(),
    onModelProgress: (loadedBytes, totalBytes) =>
      post({ type: 'model-progress', loadedBytes, totalBytes }),
  }).then((loaded) => {
    engine = loaded
    post({ type: 'ready', loadMs: performance.now() - startedAt })
    return loaded
  })
  return loading
}

function blend(dry: Float32Array, wet: Float32Array, amount: number): Float32Array {
  if (amount >= 1) return wet.slice()
  if (amount <= 0) return dry.slice()
  const out = new Float32Array(dry.length)
  for (let i = 0; i < dry.length; i += 1) out[i] = dry[i] + amount * (wet[i] - dry[i])
  return out
}

async function process(request: ProcessRequest): Promise<void> {
  if (request.channels) {
    cache = { source: request.source, dry: request.channels, residual: [] }
  }
  if (!cache || cache.source !== request.source) {
    throw new Error('the worker no longer holds that audio; load the file again')
  }
  const dry = cache.dry
  const sampleRate = request.sampleRate
  const totalSeconds = dry.reduce((sum, channel) => sum + channel.length / sampleRate, 0)
  const startedAt = performance.now()
  let lastPost = 0
  const report = (stage: 'dereverb' | 'denoise', done: number) => {
    const now = performance.now()
    if (now - lastPost < 100) return
    lastPost = now
    post({
      type: 'process-progress',
      job: request.job,
      stage,
      share: Math.min(1, done / Math.max(totalSeconds, 1e-6)),
      processedSeconds: done,
      elapsedMs: now - startedAt,
    })
  }

  if (request.dereverb.enabled && cache.residual.length === 0) {
    let done = 0
    const residual: Float32Array[] = []
    for (const channel of dry) {
      const seconds = channel.length / sampleRate
      const result = wpe.process(channel, {
        sampleRate,
        shouldCancel: () => cancelRequested,
        onProgress: (frame, frames) => report('dereverb', done + (frame / frames) * seconds),
      })
      residual.push(result.residual.slice())
      done += seconds
    }
    cache.residual = residual
  }
  if (cancelRequested) throw new CancelledError()

  const dereverbed =
    request.dereverb.enabled && cache.residual.length > 0
      ? dry.map((channel, index) => blend(channel, cache!.residual[index], request.dereverb.amount))
      : dry.map((channel) => channel.slice())

  let denoised: Float32Array[]
  if (request.denoise) {
    const active = await ensureEngine()
    if (active.metadata.sampleRate !== sampleRate) {
      throw new Error(`the model runs at ${active.metadata.sampleRate} Hz but received ${sampleRate} Hz`)
    }
    denoised = []
    let done = 0
    for (const channel of dereverbed) {
      const seconds = channel.length / sampleRate
      const result = await active.processChannel(channel, {
        shouldCancel: () => cancelRequested,
        onProgress: ({ frame, totalFrames }) =>
          report('denoise', done + (frame / totalFrames) * seconds),
      })
      denoised.push(result.wet.slice())
      done += seconds
    }
  } else {
    denoised = dereverbed.map((channel) => channel.slice())
  }

  post(
    {
      type: 'processed',
      job: request.job,
      dereverbed,
      denoised,
      elapsedMs: performance.now() - startedAt,
    },
    [...dereverbed.map((c) => c.buffer), ...denoised.map((c) => c.buffer)],
  )
}

scope.addEventListener('message', (event) => {
  const request = event.data
  if (request.type === 'cancel') {
    cancelRequested = true
    return
  }
  if (request.type === 'load') {
    ensureEngine().catch((error) => post({ type: 'error', message: describe(error) }))
    return
  }
  cancelRequested = false
  process(request).catch((error) => {
    if (error instanceof CancelledError || describe(error) === 'cancelled') {
      post({ type: 'cancelled', job: request.job })
    } else {
      post({ type: 'error', message: describe(error) })
    }
  })
})

function describe(error: unknown): string {
  if (error instanceof Error) return error.message
  return String(error)
}
