// Messages exchanged with the processing worker. Audio buffers travel as transferables in
// both directions, so neither side keeps a copy of a long recording alive.

export interface LoadRequest {
  readonly type: 'load'
}

export interface ProcessRequest {
  readonly type: 'process'
  /** Echoed back on every reply so a cancelled run's late results can be discarded. */
  readonly job: number
  readonly channels: Float32Array[]
  readonly sampleRate: number
}

export interface CancelRequest {
  readonly type: 'cancel'
}

export type WorkerRequest = LoadRequest | ProcessRequest | CancelRequest

export interface ModelProgressMessage {
  readonly type: 'model-progress'
  readonly loadedBytes: number
  readonly totalBytes: number
}

export interface ReadyMessage {
  readonly type: 'ready'
  readonly loadMs: number
}

export interface ProcessProgressMessage {
  readonly type: 'process-progress'
  readonly job: number
  readonly share: number
  readonly processedSeconds: number
  readonly elapsedMs: number
}

export interface ProcessedMessage {
  readonly type: 'processed'
  readonly job: number
  readonly channels: Float32Array[]
  readonly elapsedMs: number
}

export interface CancelledMessage {
  readonly type: 'cancelled'
  readonly job: number
}

export interface ErrorMessage {
  readonly type: 'error'
  readonly message: string
}

export type WorkerResponse =
  | ModelProgressMessage
  | ReadyMessage
  | ProcessProgressMessage
  | ProcessedMessage
  | CancelledMessage
  | ErrorMessage
