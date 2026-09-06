// The processing box: a chart of what the pass did and a measured readout beside it.
import { Plot, type AudioView, type IdleTrace } from './plot'

export interface StageElements {
  readonly canvas: string
  readonly stat: string
  readonly value: string
  readonly unit: string
}

const element = <T extends HTMLElement>(id: string): T => {
  const found = document.getElementById(id)
  if (!found) throw new Error(`the page is missing #${id}`)
  return found as T
}

export class Stage {
  readonly plot: Plot
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
  }

  /** `removedDb` is negative where the model took energy out. */
  show(view: AudioView, removedDb: number): void {
    this.plot.showAudio(view)
    this.stat.classList.remove('stat--idle')
    this.value.textContent = Math.abs(removedDb).toFixed(1)
    this.unit.textContent = removedDb <= 0 ? 'dB off the floor' : 'dB added to the floor'
  }

  clear(): void {
    this.plot.clear()
    this.stat.classList.add('stat--idle')
    this.value.textContent = ''
    this.unit.textContent = this.idleUnit
  }

  setPlayhead(seconds: number | null): void {
    this.plot.setPlayhead(seconds)
  }

  setIdleDuration(seconds: number): void {
    this.plot.setIdleDuration(seconds)
  }

  /** Draws the idle chart from the example clip itself rather than a canned trace. */
  setIdleAudio(dry: readonly Float32Array[], wet: readonly Float32Array[]): void {
    this.plot.setIdleAudio(dry, wet)
  }
}
