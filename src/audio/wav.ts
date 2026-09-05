// RIFF/WAVE writer. 24-bit integer is the default because it matches what the companion
// CLI writes and imports cleanly everywhere; 32-bit float is offered for anyone who wants
// the model's output with no quantisation at all.

export type WavBitDepth = 16 | 24 | 32

export interface WavOptions {
  readonly sampleRate: number
  readonly bitDepth?: WavBitDepth
}

export function encodeWav(channels: readonly Float32Array[], options: WavOptions): ArrayBuffer {
  const bitDepth = options.bitDepth ?? 24
  const channelCount = channels.length
  if (channelCount === 0) throw new Error('encodeWav needs at least one channel')
  const frames = channels[0].length
  const bytesPerSample = bitDepth / 8
  const blockAlign = channelCount * bytesPerSample
  const dataBytes = frames * blockAlign
  const isFloat = bitDepth === 32

  const buffer = new ArrayBuffer(44 + dataBytes)
  const view = new DataView(buffer)
  const text = (offset: number, value: string) => {
    for (let i = 0; i < value.length; i += 1) view.setUint8(offset + i, value.charCodeAt(i))
  }

  text(0, 'RIFF')
  view.setUint32(4, 36 + dataBytes, true)
  text(8, 'WAVE')
  text(12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, isFloat ? 3 : 1, true)
  view.setUint16(22, channelCount, true)
  view.setUint32(24, options.sampleRate, true)
  view.setUint32(28, options.sampleRate * blockAlign, true)
  view.setUint16(32, blockAlign, true)
  view.setUint16(34, bitDepth, true)
  text(36, 'data')
  view.setUint32(40, dataBytes, true)

  let offset = 44
  for (let frame = 0; frame < frames; frame += 1) {
    for (let channel = 0; channel < channelCount; channel += 1) {
      const sample = clamp(channels[channel][frame])
      if (isFloat) {
        view.setFloat32(offset, sample, true)
      } else if (bitDepth === 24) {
        // Asymmetric scaling: full-scale negative is exactly -2^23, positive stops one
        // step short, so a 0 dBFS peak cannot wrap to the opposite rail.
        const value = Math.round(sample < 0 ? sample * 8388608 : sample * 8388607)
        view.setUint8(offset, value & 0xff)
        view.setUint8(offset + 1, (value >> 8) & 0xff)
        view.setUint8(offset + 2, (value >> 16) & 0xff)
      } else {
        view.setInt16(offset, Math.round(sample < 0 ? sample * 32768 : sample * 32767), true)
      }
      offset += bytesPerSample
    }
  }
  return buffer
}

function clamp(sample: number): number {
  if (!Number.isFinite(sample)) return 0
  if (sample > 1) return 1
  if (sample < -1) return -1
  return sample
}
