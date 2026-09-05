// Both versions play from the same clock, so moving the mix mid-sentence changes only the
// balance, never the position. The gain law matches what the exporter writes, so what you
// hear is what lands in the file.

export interface PlayerState {
  readonly playing: boolean
  readonly currentTime: number
  readonly duration: number
}

export class Player {
  private context: AudioContext | null = null
  private buffers: AudioBuffer[] = []
  private sources: AudioBufferSourceNode[] = []
  private gains: GainNode[] = []
  private startedAt = 0
  private offset = 0
  private playing = false
  private mix = 1
  private bypassed = false
  private onChange: (state: PlayerState) => void = () => {}

  get duration(): number {
    return this.buffers[0]?.duration ?? 0
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

  /** The original and the processed result, both the same length. */
  load(
    original: readonly Float32Array[],
    processed: readonly Float32Array[],
    sampleRate: number,
  ): void {
    this.stopSources()
    const context = this.ensureContext(sampleRate)
    this.buffers = [original, processed].map((channels) =>
      toAudioBuffer(context, channels, sampleRate),
    )
    this.offset = 0
    this.playing = false
    this.emit()
  }

  setMix(mix: number): void {
    this.mix = Math.min(1, Math.max(0, mix))
    this.applyGains()
  }

  /** Monitor the untouched input without moving the mix, so the export is unaffected. */
  setBypassed(bypassed: boolean): void {
    this.bypassed = bypassed
    this.applyGains()
  }

  get isBypassed(): boolean {
    return this.bypassed
  }

  private targetGains(): number[] {
    if (this.bypassed) return [1, 0]
    return [1 - this.mix, this.mix]
  }

  private applyGains(): void {
    if (!this.context || this.gains.length === 0) return
    // A short ramp rather than a step: an abrupt gain change on a running buffer clicks.
    const at = this.context.currentTime
    const targets = this.targetGains()
    this.gains.forEach((gain, index) => gain.gain.setTargetAtTime(targets[index], at, 0.01))
  }

  play(): void {
    if (this.playing || this.buffers.length === 0) return
    const context = this.ensureContext(this.buffers[0].sampleRate)
    void context.resume()
    if (this.offset >= this.duration - 0.005) this.offset = 0

    const targets = this.targetGains()
    this.gains = this.buffers.map((_, index) => {
      const gain = context.createGain()
      gain.gain.value = targets[index]
      gain.connect(context.destination)
      return gain
    })
    this.sources = this.buffers.map((buffer, index) => {
      const source = context.createBufferSource()
      source.buffer = buffer
      source.connect(this.gains[index])
      return source
    })
    this.sources[0].onended = () => {
      if (!this.playing) return
      this.playing = false
      this.offset = this.duration
      this.emit()
    }

    const at = context.currentTime + 0.02
    for (const source of this.sources) source.start(at, this.offset)
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
    this.buffers = []
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
    for (const source of this.sources) {
      source.onended = null
      try {
        source.stop()
      } catch {
        // Already stopped; nothing to unwind.
      }
      source.disconnect()
    }
    for (const gain of this.gains) gain.disconnect()
    this.sources = []
    this.gains = []
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
