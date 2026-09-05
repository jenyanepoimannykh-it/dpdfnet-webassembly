// The plug-in's mix control, redrawn for the page. The native range input stays in the DOM
// and keeps every keyboard and assistive-technology behaviour it already has; the arc is
// painted over it and pointer drags are translated into value changes.

export class Knob {
  private readonly input: HTMLInputElement
  private readonly arc: SVGPathElement
  private readonly value: HTMLElement
  private readonly surface: HTMLElement
  private listener: (fraction: number) => void = () => {}

  constructor(root: HTMLElement, input: HTMLInputElement, arc: SVGPathElement, value: HTMLElement) {
    this.input = input
    this.arc = arc
    this.value = value
    this.surface = root

    input.addEventListener('input', () => this.publish())

    let dragging = false
    let originY = 0
    let originValue = 0
    input.addEventListener('pointerdown', (event) => {
      if (input.disabled) return
      dragging = true
      originY = event.clientY
      originValue = Number(input.value)
      input.setPointerCapture(event.pointerId)
      event.preventDefault()
    })
    input.addEventListener('pointermove', (event) => {
      if (!dragging) return
      // 160 px of travel covers the full range, the usual feel for a plug-in knob.
      const range = Number(input.max) - Number(input.min)
      const moved = ((originY - event.clientY) / 160) * range
      input.value = String(Math.round(clamp(originValue + moved, Number(input.min), Number(input.max))))
      this.publish()
    })
    const release = (event: PointerEvent) => {
      if (!dragging) return
      dragging = false
      input.releasePointerCapture(event.pointerId)
    }
    input.addEventListener('pointerup', release)
    input.addEventListener('pointercancel', release)

    this.render()
  }

  onChange(listener: (fraction: number) => void): void {
    this.listener = listener
  }

  get fraction(): number {
    return Number(this.input.value) / Number(this.input.max)
  }

  setEnabled(enabled: boolean): void {
    this.input.disabled = !enabled
    this.surface.classList.toggle('knob--live', enabled)
  }

  private publish(): void {
    this.render()
    this.listener(this.fraction)
  }

  private render(): void {
    const percent = Math.round(this.fraction * 100)
    // pathLength is 100 on the arc, so the lit length is the percentage directly.
    this.arc.style.strokeDasharray = `${this.fraction * 100} 100`
    this.value.textContent = `${percent}%`
    this.input.setAttribute('aria-valuetext', `${percent} percent dereverbed`)
  }
}

function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value))
}
