import './styles.css'
import { decodeFile, DecodeError, downmixToMono } from './audio/decode'
import { blend, levelDifferenceDb } from './audio/blend'
import { encodeWav, type WavBitDepth } from './audio/wav'
import { Player } from './audio/player'
import { Plot } from './ui/plot'
import { Knob } from './ui/knob'
import { formatChannels, formatClock, formatSpeed } from './ui/format'
import type { WorkerRequest, WorkerResponse } from './worker/protocol'

const modelSampleRate = 48000

const element = <T extends HTMLElement>(id: string): T => {
  const found = document.getElementById(id)
  if (!found) throw new Error(`the page is missing #${id}`)
  return found as T
}

const dom = {
  engineStatus: element('engine-status'),
  plotCanvas: element<HTMLCanvasElement>('plot-canvas'),
  panelReading: element('panel-reading'),
  stat: element('stat'),
  statValue: element('stat-value'),
  statUnit: element('stat-unit'),
  knob: element('knob'),
  knobArc: document.getElementById('knob-arc') as unknown as SVGPathElement,
  knobValue: element('knob-value'),
  mixSlider: element<HTMLInputElement>('mix-slider'),
  readout: element<HTMLDListElement>('file-readout'),
  stagePick: element('stage-pick'),
  stageBusy: element('stage-busy'),
  stageDone: element('stage-done'),
  fileInput: element<HTMLInputElement>('file-input'),
  filepickButton: element('file-input').nextElementSibling as HTMLElement,
  stereoCheck: element<HTMLInputElement>('stereo-check'),
  progressFill: element('progress-fill'),
  progressLabel: element('progress-label'),
  cancelButton: element<HTMLButtonElement>('cancel-button'),
  playButton: element<HTMLButtonElement>('play-button'),
  transportTime: element('transport-time'),
  downloadButton: element<HTMLButtonElement>('download-button'),
  depthSelect: element<HTMLSelectElement>('depth-select'),
  resetButton: element<HTMLButtonElement>('reset-button'),
  alert: element('alert'),
  dropveil: element('dropveil'),
}

interface Result {
  readonly name: string
  readonly dry: Float32Array[]
  readonly wet: Float32Array[]
  readonly sampleRate: number
}

const plot = new Plot(dom.plotCanvas)
const player = new Player()
const knob = new Knob(dom.knob, dom.mixSlider, dom.knobArc, dom.knobValue)

let result: Result | null = null
let worker: Worker | null = null
let modelReady = false
let currentJob = 0
let frameHandle = 0

// --- worker --------------------------------------------------------------

function ensureWorker(): Worker {
  if (worker) return worker
  const created = new Worker(new URL('./worker/dereverb.worker.ts', import.meta.url), {
    type: 'module',
  })
  created.addEventListener('message', (event: MessageEvent<WorkerResponse>) => handle(event.data))
  created.addEventListener('error', () =>
    fail('The processing worker stopped. Reload the page and try again.'),
  )
  worker = created
  return created
}

function send(request: WorkerRequest, transfer?: Transferable[]): void {
  ensureWorker().postMessage(request, transfer ?? [])
}

function handle(message: WorkerResponse): void {
  switch (message.type) {
    case 'model-progress': {
      const share = message.loadedBytes / message.totalBytes
      setProgress(share * 0.25, `Downloading the model, ${Math.round(share * 100)}%`)
      break
    }
    case 'ready': {
      modelReady = true
      dom.engineStatus.classList.add('status--live')
      setStatus('DPDFNET AI · 48 kHz · runs on your machine')
      break
    }
    case 'process-progress': {
      if (message.job !== currentJob) return
      const share = message.processedSeconds / Math.max(message.totalSeconds, 1e-6)
      setProgress(
        0.25 + share * 0.75,
        `Removing the room, ${Math.round(share * 100)}%. ${formatSpeed(message.processedSeconds, message.elapsedMs)}`,
      )
      break
    }
    case 'processed': {
      if (message.job !== currentJob) return
      finish(message.channels, message.elapsedMs)
      break
    }
    case 'cancelled': {
      if (message.job === currentJob) reset()
      break
    }
    case 'error': {
      fail(message.message)
      break
    }
  }
}

// --- stages --------------------------------------------------------------

function show(stage: 'pick' | 'busy' | 'done'): void {
  dom.stagePick.hidden = stage !== 'pick'
  dom.stageBusy.hidden = stage !== 'busy'
  dom.stageDone.hidden = stage !== 'done'
}

function setStatus(text: string): void {
  const lamp = dom.engineStatus.firstElementChild
  dom.engineStatus.textContent = text
  if (lamp) dom.engineStatus.prepend(lamp)
}

function setProgress(share: number, label: string): void {
  dom.progressFill.style.width = `${Math.min(100, Math.max(0, share * 100))}%`
  dom.progressLabel.textContent = label
}

function fail(message: string): void {
  dom.alert.hidden = false
  dom.alert.textContent = message
  show(result?.wet.length ? 'done' : 'pick')
}

function clearAlert(): void {
  dom.alert.hidden = true
  dom.alert.textContent = ''
}

async function load(file: File): Promise<void> {
  clearAlert()
  player.dispose()
  result = null
  plot.clear()
  knob.setEnabled(false)
  dom.stat.classList.add('stat--idle')
  dom.statUnit.textContent = 'Waiting for a file'
  dom.readout.replaceChildren()
  currentJob += 1
  const job = currentJob

  show('busy')
  setProgress(0.02, 'Reading the file')
  dom.panelReading.textContent = file.name

  let decoded
  try {
    decoded = await decodeFile(file, modelSampleRate)
  } catch (error) {
    fail(error instanceof DecodeError ? error.message : `Could not read ${file.name}.`)
    return
  }
  if (job !== currentJob) return

  const channels =
    decoded.channels.length > 1 && !dom.stereoCheck.checked
      ? downmixToMono(decoded.channels)
      : decoded.channels

  setProgress(modelReady ? 0.25 : 0.05, modelReady ? 'Removing the room' : 'Loading the model')
  result = { name: file.name, dry: channels.map((c) => c.slice()), wet: [], sampleRate: modelSampleRate }
  send(
    { type: 'process', job, channels, sampleRate: modelSampleRate },
    channels.map((channel) => channel.buffer),
  )
}

function finish(wet: Float32Array[], elapsedMs: number): void {
  if (!result) return
  result = { ...result, wet }
  const seconds = result.dry[0].length / result.sampleRate
  const removed = levelDifferenceDb(result.dry, wet)

  plot.showAudio({ dry: result.dry, wet, sampleRate: result.sampleRate })
  dom.panelReading.textContent = `${result.name}, ${formatClock(seconds)}`
  dom.stat.classList.remove('stat--idle')
  dom.statValue.textContent = Math.abs(removed).toFixed(1)
  dom.statUnit.textContent = removed <= 0 ? 'dB removed' : 'dB added'
  dom.readout.replaceChildren(
    ...(
      [
        ['Channels', formatChannels(result.dry.length)],
        ['Rate', '48 kHz'],
        ['Took', `${(elapsedMs / 1000).toFixed(1)} s, ${formatSpeed(seconds, elapsedMs)}`],
      ] as const
    ).map(([label, value]) => {
      const row = document.createElement('div')
      const term = document.createElement('dt')
      term.textContent = label
      const detail = document.createElement('dd')
      detail.textContent = value
      row.append(term, detail)
      return row
    }),
  )

  if (import.meta.env.DEV) {
    // Handle for the numerical comparison against the NumPy reference in scripts/;
    // the branch and its contents are stripped from production builds.
    ;(window as unknown as { deadroom?: unknown }).deadroom = result
  }

  knob.setEnabled(true)
  player.load(result.dry, wet, result.sampleRate)
  player.setMix(knob.fraction)
  show('done')
  dom.playButton.focus()
}

function reset(): void {
  player.dispose()
  result = null
  currentJob += 1
  plot.clear()
  clearAlert()
  knob.setEnabled(false)
  dom.stat.classList.add('stat--idle')
  dom.statUnit.textContent = 'Waiting for a file'
  dom.panelReading.textContent = 'Example: 9 s in an untreated room'
  dom.readout.replaceChildren()
  dom.fileInput.value = ''
  show('pick')
}

// --- transport -----------------------------------------------------------

player.subscribe((state) => {
  dom.playButton.textContent = state.playing ? 'Pause' : 'Play'
  dom.transportTime.textContent = `${formatClock(state.currentTime)} / ${formatClock(state.duration)}`
  plot.setPlayhead(state.currentTime)
  if (state.playing && frameHandle === 0) tick()
  if (!state.playing && frameHandle !== 0) {
    cancelAnimationFrame(frameHandle)
    frameHandle = 0
  }
})

function tick(): void {
  frameHandle = requestAnimationFrame(() => {
    frameHandle = 0
    if (!player.isPlaying) return
    dom.transportTime.textContent = `${formatClock(player.currentTime)} / ${formatClock(player.duration)}`
    plot.setPlayhead(player.currentTime)
    tick()
  })
}

plot.onSeek((seconds) => {
  if (result?.wet.length) player.seek(seconds)
})

knob.onChange((fraction) => player.setMix(fraction))

// --- events --------------------------------------------------------------

dom.filepickButton.addEventListener('pointerenter', warmModel)
dom.fileInput.addEventListener('focus', warmModel)
dom.fileInput.addEventListener('change', () => {
  const file = dom.fileInput.files?.[0]
  if (file) void load(file)
})

function warmModel(): void {
  if (!modelReady) send({ type: 'load' })
}

dom.cancelButton.addEventListener('click', () => {
  send({ type: 'cancel' })
  setProgress(0, 'Stopping')
})
dom.resetButton.addEventListener('click', reset)
dom.playButton.addEventListener('click', () => player.toggle())

dom.downloadButton.addEventListener('click', () => {
  if (!result?.wet.length) return
  const wav = encodeWav(blend(result.dry, result.wet, knob.fraction), {
    sampleRate: result.sampleRate,
    bitDepth: Number(dom.depthSelect.value) as WavBitDepth,
  })
  const url = URL.createObjectURL(new Blob([wav], { type: 'audio/wav' }))
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = `${result.name.replace(/\.[^.]+$/, '')} dereverbed.wav`
  anchor.click()
  URL.revokeObjectURL(url)
})

document.addEventListener('keydown', (event) => {
  if (event.code !== 'Space' || !result?.wet.length) return
  const target = event.target as HTMLElement | null
  if (target && ['INPUT', 'SELECT', 'BUTTON', 'TEXTAREA'].includes(target.tagName)) return
  event.preventDefault()
  player.toggle()
})

let dragDepth = 0
window.addEventListener('dragenter', (event) => {
  if (!event.dataTransfer?.types.includes('Files')) return
  dragDepth += 1
  dom.dropveil.hidden = false
  warmModel()
})
window.addEventListener('dragleave', () => {
  dragDepth = Math.max(0, dragDepth - 1)
  if (dragDepth === 0) dom.dropveil.hidden = true
})
window.addEventListener('dragover', (event) => event.preventDefault())
window.addEventListener('drop', (event) => {
  event.preventDefault()
  dragDepth = 0
  dom.dropveil.hidden = true
  const file = event.dataTransfer?.files?.[0]
  if (file) void load(file)
})
