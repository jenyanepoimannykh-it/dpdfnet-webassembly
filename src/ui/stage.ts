// The processing box: a chart of what the model did, a measured readout, and a mix knob.
import { Knob } from './knob'
import { Plot, type AudioView, type IdleTrace } from './plot'

export interface StageElements {
  readonly canvas: string
  readonly stat: string
  readonly value: string
  readonly unit: string
  readonly knob: string
  readonly arc: string
  readonly knobValue: string
  readonly input: string
}

const element = <T extends HTMLElement>(id: string): T => {
  const found = document.getElementById(id)
  if (!found) throw new Error(`the page is missing #${id}`)
  return found as T
}

export class Stage {
  readonly plot: Plot
  readonly knob: Knob
  private readonly stat: HTMLElement
  private readonly value: HTMLElement
  private readonly unit: HTMLElement
  private readonly idleUnit: string

  constructor(ids: StageElements, idle: IdleTrace) {
    this.stat = element(ids.stat)
    this.value = element(ids.value)
    this.unit = element(ids.unit)
    this.idleUnit = this.unit.textContent ?? ''
    this.plot = new Plot(element<HTMLCanvasElement>(ids.canvas), idle)
    this.knob = new Knob(
      element(ids.knob),
      element<HTMLInputElement>(ids.input),
      document.getElementById(ids.arc) as unknown as SVGPathElement,
      element(ids.knobValue),
    )
  }

  get amount(): number {
    return this.knob.fraction
  }

  onAmount(listener: (amount: number) => void): void {
    this.knob.onChange(listener)
  }

  /** `removedDb` is negative where the model took energy out. */
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
