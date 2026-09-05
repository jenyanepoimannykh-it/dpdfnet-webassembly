// One box in the rack: a chart of what the stage did, a measured readout, a knob, and a
// switch. Both stages are the same shape, so they are the same object.
import { Knob } from './knob'
import { Plot, type AudioView, type IdleTrace } from './plot'

export interface RackModuleElements {
  readonly root: string
  readonly canvas: string
  readonly stat: string
  readonly value: string
  readonly unit: string
  readonly knob: string
  readonly arc: string
  readonly knobValue: string
  readonly input: string
  readonly toggle: string
}

const element = <T extends HTMLElement>(id: string): T => {
  const found = document.getElementById(id)
  if (!found) throw new Error(`the page is missing #${id}`)
  return found as T
}

export class RackModule {
  readonly plot: Plot
  readonly knob: Knob
  private readonly root: HTMLElement
  private readonly stat: HTMLElement
  private readonly value: HTMLElement
  private readonly unit: HTMLElement
  private readonly toggle: HTMLInputElement
  private readonly idleUnit: string

  constructor(ids: RackModuleElements, idle: IdleTrace) {
    this.root = element(ids.root)
    this.stat = element(ids.stat)
    this.value = element(ids.value)
    this.unit = element(ids.unit)
    this.toggle = element<HTMLInputElement>(ids.toggle)
    this.idleUnit = this.unit.textContent ?? ''
    this.plot = new Plot(element<HTMLCanvasElement>(ids.canvas), idle)
    this.knob = new Knob(
      element(ids.knob),
      element<HTMLInputElement>(ids.input),
      document.getElementById(ids.arc) as unknown as SVGPathElement,
      element(ids.knobValue),
    )
    this.root.classList.toggle('module--off', !this.toggle.checked)
  }

  get enabled(): boolean {
    return this.toggle.checked
  }

  get amount(): number {
    return this.knob.fraction
  }

  onToggle(listener: (enabled: boolean) => void): void {
    this.toggle.addEventListener('change', () => {
      this.root.classList.toggle('module--off', !this.toggle.checked)
      listener(this.toggle.checked)
    })
  }

  onAmount(listener: (amount: number) => void): void {
    this.knob.onChange(listener)
  }

  /** `removedDb` is negative where the stage took energy out. */
  show(view: AudioView, removedDb: number): void {
    this.plot.showAudio(view)
    this.knob.setEnabled(true)
    this.stat.classList.remove('stat--idle')
    this.value.textContent = Math.abs(removedDb).toFixed(1)
    this.unit.textContent = removedDb <= 0 ? 'dB off the floor' : 'dB added to the floor'
  }

  clear(): void {
    this.plot.clear()
    this.knob.setEnabled(false)
    this.stat.classList.add('stat--idle')
    this.value.textContent = ''
    this.unit.textContent = this.idleUnit
  }

  setPlayhead(seconds: number | null): void {
    this.plot.setPlayhead(seconds)
  }
}
