// Both versions play from the same clock so the blend slider is a true crossfade: moving it
// mid-sentence changes only the balance, never the position. The gain law matches what the
// exporter writes, so what you hear is what lands in the file.

export interface PlayerState {
  readonly playing: boolean
  readonly currentTime: number
  readonly duration: number
}

export class Player {
  private context: AudioContext | null = null
  private dryBuffer: AudioBuffer | null = null
  private wetBuffer: AudioBuffer | null = null
  private drySource: AudioBufferSourceNode | null = null
  private wetSource: AudioBufferSourceNode | null = null
  private dryGain: GainNode | null = null
  private wetGain: GainNode | null = null
  private startedAt = 0
  private offset = 0
  private playing = false
  private mix = 1
  private onChange: (state: PlayerState) => void = () => {}

  get duration(): number {
    return this.dryBuffer?.duration ?? 0
  }

  get currentTime(): number {
    if (!this.playing || !this.context) return this.offset
    return Math.min(this.duration, this.offset + (this.context.currentTime - this.startedAt))
  }

  get isPlaying(): boolean {
    return this.playing
  }

  subscribe(listener: (state: PlayerState) => void): void {
    this.onChange = listener
  }

  load(dry: readonly Float32Array[], wet: readonly Float32Array[], sampleRate: number): void {
    this.stopSources()
    const context = this.ensureContext(sampleRate)
    this.dryBuffer = toAudioBuffer(context, dry, sampleRate)
    this.wetBuffer = toAudioBuffer(context, wet, sampleRate)
    this.offset = 0
    this.playing = false
    this.emit()
  }

  setMix(mix: number): void {
    this.mix = Math.min(1, Math.max(0, mix))
    if (this.dryGain && this.wetGain && this.context) {
      // A short ramp rather than a step: an abrupt gain change on a running buffer clicks.
      const at = this.context.currentTime
      this.dryGain.gain.setTargetAtTime(1 - this.mix, at, 0.01)
      this.wetGain.gain.setTargetAtTime(this.mix, at, 0.01)
    }
  }

  play(): void {
    if (this.playing || !this.dryBuffer || !this.wetBuffer) return
    const context = this.ensureContext(this.dryBuffer.sampleRate)
    void context.resume()
    if (this.offset >= this.duration - 0.005) this.offset = 0

    this.dryGain = context.createGain()
    this.wetGain = context.createGain()
    this.dryGain.gain.value = 1 - this.mix
    this.wetGain.gain.value = this.mix
    this.dryGain.connect(context.destination)
    this.wetGain.connect(context.destination)

    this.drySource = context.createBufferSource()
    this.wetSource = context.createBufferSource()
    this.drySource.buffer = this.dryBuffer
    this.wetSource.buffer = this.wetBuffer
    this.drySource.connect(this.dryGain)
    this.wetSource.connect(this.wetGain)
    this.drySource.onended = () => {
      if (!this.playing) return
      this.playing = false
      this.offset = this.duration
      this.emit()
    }

    const at = context.currentTime + 0.02
    this.drySource.start(at, this.offset)
    this.wetSource.start(at, this.offset)
    this.startedAt = at
    this.playing = true
    this.emit()
  }

  pause(): void {
    if (!this.playing) return
    this.offset = this.currentTime
    this.stopSources()
    this.playing = false
    this.emit()
  }

  toggle(): void {
    if (this.playing) this.pause()
    else this.play()
  }

  seek(seconds: number): void {
    const target = Math.min(this.duration, Math.max(0, seconds))
    const wasPlaying = this.playing
    if (wasPlaying) {
      this.stopSources()
      this.playing = false
    }
    this.offset = target
    if (wasPlaying) this.play()
    else this.emit()
  }

  dispose(): void {
    this.stopSources()
    this.dryBuffer = null
    this.wetBuffer = null
    this.offset = 0
    this.playing = false
  }

  private ensureContext(sampleRate: number): AudioContext {
    if (!this.context || this.context.state === 'closed') {
      this.context = new AudioContext({ sampleRate })
    }
    return this.context
  }

  private stopSources(): void {
    for (const source of [this.drySource, this.wetSource]) {
      if (!source) continue
      source.onended = null
      try {
        source.stop()
      } catch {
        // Already stopped; nothing to unwind.
      }
      source.disconnect()
    }
    this.drySource = null
    this.wetSource = null
    this.dryGain?.disconnect()
    this.wetGain?.disconnect()
    this.dryGain = null
    this.wetGain = null
  }

  private emit(): void {
    this.onChange({ playing: this.playing, currentTime: this.currentTime, duration: this.duration })
  }
}

function toAudioBuffer(
  context: BaseAudioContext,
  channels: readonly Float32Array[],
  sampleRate: number,
): AudioBuffer {
  const buffer = context.createBuffer(channels.length, channels[0].length, sampleRate)
  for (let channel = 0; channel < channels.length; channel += 1) {
    // Arrays that have crossed a worker boundary are typed over ArrayBufferLike; the Web
    // Audio signature insists on ArrayBuffer, and a transferred buffer is always one.
    buffer.copyToChannel(channels[channel] as Float32Array<ArrayBuffer>, channel)
  }
  return buffer
}
