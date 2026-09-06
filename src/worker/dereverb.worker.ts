// Inference runs here so a long file never blocks the interface. onnxruntime-web is
// imported inside the worker too, which keeps its glue off the first paint.
import wasmBinaryUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url'
import { CancelledError, DereverbEngine } from '../dsp/dereverb'
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
let queue: Promise<void> = Promise.resolve()

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

// Both tools are the same pass. The network takes noise and room out together, which is
// what the plug-in ships as its de-reverb; the two pages differ in what they say about it
// and in the material they are pointed at, not in what runs.
async function process(request: ProcessRequest): Promise<void> {
  const active = await ensureEngine()
  if (cancelRequested) throw new CancelledError()
  const sampleRate = request.sampleRate
  if (active.metadata.sampleRate !== sampleRate) {
    throw new Error(`the model runs at ${active.metadata.sampleRate} Hz but received ${sampleRate} Hz`)
  }

  const totalSeconds = request.channels.reduce((sum, c) => sum + c.length / sampleRate, 0)
  const startedAt = performance.now()
  let done = 0
  let lastPost = 0

  const output: Float32Array[] = []
  for (const channel of request.channels) {
    const seconds = channel.length / sampleRate
    const result = await active.processChannel(channel, {
      shouldCancel: () => cancelRequested,
      onProgress: ({ frame, totalFrames }) => {
        const now = performance.now()
        if (now - lastPost < 100) return
        lastPost = now
        const processedSeconds = done + (frame / totalFrames) * seconds
        post({
          type: 'process-progress',
          job: request.job,
          share: Math.min(1, processedSeconds / Math.max(totalSeconds, 1e-6)),
          processedSeconds,
          elapsedMs: now - startedAt,
        })
      },
    })
    // subarray shares the padded buffer; slice so only the audible span is transferred.
    output.push(result.wet.slice())
    done += seconds
  }

  post(
    { type: 'processed', job: request.job, channels: output, elapsedMs: performance.now() - startedAt },
    output.map((channel) => channel.buffer),
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
  // One run at a time. The engine carries recurrent state between frames, and the example
  // the idle play button renders can still be in flight when a picked file arrives.
  queue = queue.then(async () => {
    cancelRequested = false
    try {
      await process(request)
    } catch (error) {
      if (cancelRequested || error instanceof CancelledError) post({ type: 'cancelled', job: request.job })
      else post({ type: 'error', message: describe(error) })
    }
  })
})

function describe(error: unknown): string {
  if (error instanceof Error) return error.message
  return String(error)
}
