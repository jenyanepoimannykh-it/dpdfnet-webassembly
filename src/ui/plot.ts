// The chart is the plug-in's "Room tail" display widened to a whole file. Both states use
// the same grammar as the desktop UI: the dim filled envelope is what the microphone
// captured, the bright phosphor line is what the model left. The dim area standing above
// the line is the room that came out.
//
// The vertical axis is level in dB, not amplitude. Reverberation lives 20-50 dB under the
// speech that caused it, so on a linear axis the whole story is a few pixels tall; in dB
// the tails between words open out and the horizontal grid lines read as 15 dB steps.
import { demoDry, demoRoom } from './demo-trace'

export interface AudioView {
  readonly dry: readonly Float32Array[]
  readonly wet: readonly Float32Array[]
  readonly sampleRate: number
}

interface Palette {
  fill: string
  phosphor: string
  grid: string
  text: string
}

/** Envelope resolution held in memory; every canvas width is folded down from this. */
const bucketCount = 4096

/** Bottom of the level axis. Anything quieter than this is silence for our purposes. */
const floorDb = 60

function levelOf(amplitude: number): number {
  if (amplitude <= 0) return 0
  const db = 20 * Math.log10(amplitude)
  return Math.min(1, Math.max(0, (db + floorDb) / floorDb))
}

export class Plot {
  private readonly canvas: HTMLCanvasElement
  private readonly layer = document.createElement('canvas')
  private readonly context: CanvasRenderingContext2D
  private peaks: { dry: Float32Array; wet: Float32Array } | null = null
  private duration = 0
  private playhead: number | null = null
  private width = 0
  private height = 0
  private ratio = 1
  private seekListener: ((seconds: number) => void) | null = null

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas
    const context = canvas.getContext('2d')
    if (!context) throw new Error('this browser does not provide a 2D canvas context')
    this.context = context

    new ResizeObserver(() => this.resize()).observe(canvas)

    canvas.addEventListener('pointerdown', (event) => {
      if (!this.seekListener || this.duration === 0) return
      const bounds = canvas.getBoundingClientRect()
      const fraction = Math.min(1, Math.max(0, (event.clientX - bounds.left) / bounds.width))
      this.seekListener(fraction * this.duration)
    })
  }

  onSeek(listener: (seconds: number) => void): void {
    this.seekListener = listener
  }

  showAudio(view: AudioView): void {
    this.duration = view.dry[0].length / view.sampleRate
    const dry = envelope(view.dry, bucketCount)
    const wet = envelope(view.wet, bucketCount)
    let loudest = 0
    for (const value of dry) if (value > loudest) loudest = value
    const scale = loudest > 0 ? 1 / loudest : 1
    for (let i = 0; i < bucketCount; i += 1) {
      dry[i] *= scale
      wet[i] *= scale
    }
    this.peaks = { dry, wet }
    this.playhead = null
    this.repaint()
  }

  clear(): void {
    this.peaks = null
    this.duration = 0
    this.playhead = null
    this.repaint()
  }

  setPlayhead(seconds: number | null): void {
    this.playhead = seconds
    this.blit()
  }

  private resize(): void {
    const bounds = this.canvas.getBoundingClientRect()
    if (bounds.width === 0 || bounds.height === 0) return
    this.ratio = Math.min(2, window.devicePixelRatio || 1)
    this.width = Math.round(bounds.width * this.ratio)
    this.height = Math.round(bounds.height * this.ratio)
    this.canvas.width = this.width
    this.canvas.height = this.height
    this.layer.width = this.width
    this.layer.height = this.height
    this.repaint()
  }

  private palette(): Palette {
    const styles = getComputedStyle(this.canvas)
    return {
      fill: styles.getPropertyValue('--fill').trim(),
      phosphor: styles.getPropertyValue('--phosphor').trim(),
      grid: styles.getPropertyValue('--track').trim(),
      text: styles.getPropertyValue('--text').trim(),
    }
  }

  private repaint(): void {
    if (this.width === 0) {
      this.resize()
      return
    }
    const context = this.layer.getContext('2d')
    if (!context) return
    context.clearRect(0, 0, this.width, this.height)
    const palette = this.palette()
    this.grid(context, palette)
    if (this.peaks) this.paint(context, palette, this.peaks.dry, this.peaks.wet)
    else this.paint(context, palette, demoRoom, demoDry)
    this.blit()
  }

  private blit(): void {
    if (this.width === 0) return
    this.context.clearRect(0, 0, this.width, this.height)
    this.context.drawImage(this.layer, 0, 0)
    if (this.playhead === null || this.duration === 0) return
    const x = Math.round((this.playhead / this.duration) * this.width) + 0.5
    this.context.strokeStyle = this.palette().text
    this.context.lineWidth = this.ratio
    this.context.beginPath()
    this.context.moveTo(x, 0)
    this.context.lineTo(x, this.height)
    this.context.stroke()
  }

  private grid(context: CanvasRenderingContext2D, palette: Palette): void {
    const columns = this.duration > 0 ? Math.min(12, Math.max(4, Math.round(this.duration / 5))) : 8
    context.strokeStyle = palette.grid
    context.lineWidth = 1
    context.beginPath()
    for (let i = 1; i < columns; i += 1) {
      const x = Math.round((i / columns) * this.width) + 0.5
      context.moveTo(x, 0)
      context.lineTo(x, this.height)
    }
    for (let i = 1; i < 4; i += 1) {
      const y = Math.round((i / 4) * this.height) + 0.5
      context.moveTo(0, y)
      context.lineTo(this.width, y)
    }
    context.stroke()
  }

  /** Bottom-anchored envelopes: a filled one for the input, a stroked one for the output. */
  private paint(
    context: CanvasRenderingContext2D,
    palette: Palette,
    dry: Float32Array,
    wet: Float32Array,
  ): void {
    const floor = this.height - Math.max(1, this.ratio)
    const span = this.height * 0.94
    const columnHeight = (source: Float32Array, x: number) => {
      const buckets = source.length
      const from = Math.floor((x / this.width) * buckets)
      const to = Math.max(from + 1, Math.floor(((x + 1) / this.width) * buckets))
      let peak = 0
      for (let bucket = from; bucket < to && bucket < buckets; bucket += 1) {
        if (source[bucket] > peak) peak = source[bucket]
      }
      return floor - levelOf(peak) * span
    }

    context.fillStyle = palette.fill
    context.beginPath()
    context.moveTo(0, floor)
    for (let x = 0; x < this.width; x += 1) context.lineTo(x, columnHeight(dry, x))
    context.lineTo(this.width, floor)
    context.closePath()
    context.fill()

    context.strokeStyle = palette.phosphor
    context.lineWidth = Math.max(1, 1.25 * this.ratio)
    context.lineJoin = 'round'
    // A little bloom, the way the trace sits on the plug-in's panel.
    context.shadowColor = palette.phosphor
    context.shadowBlur = 5 * this.ratio
    context.beginPath()
    for (let x = 0; x < this.width; x += 1) {
      const y = columnHeight(wet, x)
      if (x === 0) context.moveTo(x, y)
      else context.lineTo(x, y)
    }
    context.stroke()
    context.shadowBlur = 0
  }
}

/** Peak magnitude per bucket across every channel. Callers scale both traces by the
 *  input's loudest bucket, so the gap between them is a real level difference rather than
 *  two separately normalised pictures. */
function envelope(channels: readonly Float32Array[], buckets: number): Float32Array {
  const peaks = new Float32Array(buckets)
  const length = channels[0]?.length ?? 0
  if (length === 0) return peaks
  for (let bucket = 0; bucket < buckets; bucket += 1) {
    const from = Math.floor((bucket / buckets) * length)
    const to = Math.max(from + 1, Math.floor(((bucket + 1) / buckets) * length))
    let peak = 0
    for (const channel of channels) {
      for (let i = from; i < to && i < length; i += 1) {
        const magnitude = Math.abs(channel[i])
        if (magnitude > peak) peak = magnitude
      }
    }
    peaks[bucket] = peak
  }
  return peaks
}
