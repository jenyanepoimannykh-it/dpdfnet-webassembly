// Decoding runs through the browser's own media decoders, so whatever the user can play
// they can process: wav, mp3, m4a/aac, ogg, opus, flac, and the audio track of mp4/mov
// video. decodeAudioData resamples to the context's rate, which is how the material
// reaches the 48 kHz the model requires without a resampler of our own.

export interface DecodedAudio {
  /** One Float32Array per channel, already at `sampleRate`. */
  readonly channels: Float32Array[]
  readonly sampleRate: number
  readonly duration: number
}

export class DecodeError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options)
    this.name = 'DecodeError'
  }
}

export async function decodeFile(file: File, sampleRate: number): Promise<DecodedAudio> {
  const bytes = await file.arrayBuffer()
  // A one-frame context is enough: it is never rendered, it only sets the target rate.
  const context = new OfflineAudioContext(1, 1, sampleRate)
  let buffer: AudioBuffer
  try {
    buffer = await context.decodeAudioData(bytes)
  } catch (cause) {
    throw new DecodeError(
      `This browser could not decode ${file.name}. Try exporting the audio as WAV, MP3 or M4A first.`,
      { cause },
    )
  }
  if (buffer.length === 0) throw new DecodeError(`${file.name} contains no audio.`)
  const channels: Float32Array[] = []
  for (let channel = 0; channel < buffer.numberOfChannels; channel += 1) {
    channels.push(buffer.getChannelData(channel).slice())
  }
  return { channels, sampleRate: buffer.sampleRate, duration: buffer.duration }
}

/** Averages every channel into one, which halves inference time on stereo material. */
export function downmixToMono(channels: readonly Float32Array[]): Float32Array[] {
  if (channels.length <= 1) return channels.map((channel) => channel.slice())
  const length = channels[0].length
  const mono = new Float32Array(length)
  for (const channel of channels) {
    for (let i = 0; i < length; i += 1) mono[i] += channel[i]
  }
  const scale = 1 / channels.length
  for (let i = 0; i < length; i += 1) mono[i] *= scale
  return [mono]
}
