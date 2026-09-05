// Inference runs here so a long file never blocks the interface. onnxruntime-web is
// imported inside the worker too, which keeps its ~1 MB of glue off the first paint.
import wasmBinaryUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url'
import { CancelledError, DereverbEngine } from '../dsp/dereverb'
import { modelUrls } from '../models'
import type { WorkerRequest, WorkerResponse } from './protocol'

interface WorkerScope {
  postMessage(message: WorkerResponse, transfer?: Transferable[]): void
  addEventListener(type: 'message', listener: (event: MessageEvent<WorkerRequest>) => void): void
}

const scope = self as unknown as WorkerScope
const post = (message: WorkerResponse, transfer?: Transferable[]) => scope.postMessage(message, transfer)

// Only one engine is held at a time. Both models together would sit in the WebAssembly
// heap for the whole session, and switching is rare enough not to be worth that.
let engine: DereverbEngine | null = null
let engineId: string | null = null
let loading: Promise<DereverbEngine> | null = null
let loadingId: string | null = null
let cancelRequested = false

/** Intra-op threads for the convolutions. Capped low: the graph is small and per-frame,
 *  so beyond a few threads the synchronisation costs more than it saves. */
function threadCount(): number {
  if (typeof crossOriginIsolated !== 'undefined' && !crossOriginIsolated) return 1
  const cores = navigator.hardwareConcurrency ?? 1
  return Math.max(1, Math.min(4, cores - 1))
}

async function ensureEngine(model: string): Promise<DereverbEngine> {
  if (engine && engineId === model) return engine
  if (loading && loadingId === model) return loading

  const previous = engine
  engine = null
  engineId = null
  void previous?.release().catch(() => undefined)

  const startedAt = performance.now()
  loadingId = model
  loading = DereverbEngine.load({
    ...modelUrls(model),
    wasmBinaryUrl,
    numThreads: threadCount(),
    onModelProgress: (loadedBytes, totalBytes) =>
      post({ type: 'model-progress', loadedBytes, totalBytes }),
  }).then((loaded) => {
    engine = loaded
    engineId = model
    post({
      type: 'ready',
      model,
      metadata: loaded.metadata,
      loadMs: performance.now() - startedAt,
    })
    return loaded
  })
  return loading
}

async function process(request: ProcessRequestLike): Promise<void> {
  const active = await ensureEngine(request.model)
  if (cancelRequested) throw new CancelledError()
  const { hopSize, sampleRate } = active.metadata
  if (request.sampleRate !== sampleRate) {
    throw new Error(`the model runs at ${sampleRate} Hz but received ${request.sampleRate} Hz`)
  }
  const totalSeconds = (request.channels.length * request.channels[0].length) / sampleRate
  const startedAt = performance.now()
  let secondsBefore = 0
  let lastPost = 0

  const wet: Float32Array[] = []
  const reduction: Float32Array[] = []
  for (const channel of request.channels) {
    const result = await active.processChannel(channel, {
      shouldCancel: () => cancelRequested,
      onProgress: ({ frame, totalFrames }) => {
        const now = performance.now()
        if (frame !== totalFrames && now - lastPost < 100) return
        lastPost = now
        post({
          type: 'process-progress',
          job: request.job,
          processedSeconds: secondsBefore + (Math.min(frame, totalFrames) * hopSize) / sampleRate,
          totalSeconds,
          elapsedMs: now - startedAt,
        })
      },
    })
    // subarray shares the padded buffer; slice so only the audible span is transferred.
    wet.push(result.wet.slice())
    reduction.push(result.reductionDb)
    secondsBefore += channel.length / sampleRate
  }

  post(
    {
      type: 'processed',
      job: request.job,
      channels: wet,
      reductionDb: reduction,
      hopSize,
      elapsedMs: performance.now() - startedAt,
    },
    [...wet.map((channel) => channel.buffer), ...reduction.map((channel) => channel.buffer)],
  )
}

interface ProcessRequestLike {
  readonly job: number
  readonly model: string
  readonly channels: Float32Array[]
  readonly sampleRate: number
}

scope.addEventListener('message', (event) => {
  const request = event.data
  if (request.type === 'cancel') {
    cancelRequested = true
    return
  }
  if (request.type === 'load') {
    ensureEngine(request.model).catch((error) => post({ type: 'error', message: describe(error) }))
    return
  }
  cancelRequested = false
  process(request).catch((error) => {
    if (error instanceof CancelledError) post({ type: 'cancelled', job: request.job })
    else post({ type: 'error', message: describe(error) })
  })
})

function describe(error: unknown): string {
  if (error instanceof Error) return error.message
  return String(error)
}
