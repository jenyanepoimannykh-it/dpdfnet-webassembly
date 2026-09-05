import './styles.css'
import { decodeFile, DecodeError, downmixToMono } from './audio/decode'
import { blend, floorDifferenceDb } from './audio/blend'
import { encodeWav, type WavBitDepth } from './audio/wav'
import { Player } from './audio/player'
import { model } from './models'
import { Stage } from './ui/stage'
import { demoClean, demoNoisy } from './ui/demo-trace'
import { formatChannels, formatClock, formatSpeed } from './ui/format'
import type { WorkerRequest, WorkerResponse } from './worker/protocol'

const modelSampleRate = model.sampleRate

const element = <T extends HTMLElement>(id: string): T => {
  const found = document.getElementById(id)
  if (!found) throw new Error(`the page is missing #${id}`)
  return found as T
}

const dom = {
  engineStatus: element('engine-status'),
  readout: element<HTMLDListElement>('file-readout'),
  stagePick: element('stage-pick'),
  stageBusy: element('stage-busy'),
  stageDone: element('stage-done'),
  fileInput: element<HTMLInputElement>('file-input'),
  filepickButton: element('file-input').nextElementSibling as HTMLElement,
  exampleButton: element<HTMLButtonElement>('example-button'),
  stereoCheck: element<HTMLInputElement>('stereo-check'),
  progressFill: element('progress-fill'),
  progressLabel: element('progress-label'),
  cancelButton: element<HTMLButtonElement>('cancel-button'),
  playButton: element<HTMLButtonElement>('play-button'),
  transportTime: element('transport-time'),
  downloadButton: element<HTMLButtonElement>('download-button'),
  depthSelect: element<HTMLSelectElement>('depth-select'),
  resetButton: element<HTMLButtonElement>('reset-button'),
  bypassCheck: element<HTMLInputElement>('bypass-check'),
  alert: element('alert'),
  dropveil: element('dropveil'),
}

const stage = new Stage(
  {
    canvas: 'stage-canvas',
    stat: 'stage-stat',
    value: 'stage-value',
    unit: 'stage-unit',
    knob: 'stage-knob',
    arc: 'stage-arc',
    knobValue: 'stage-knob-value',
    input: 'stage-mix',
  },
  { before: demoNoisy, after: demoClean },
)

const player = new Player()

interface Result {
  readonly name: string
  readonly dry: Float32Array[]
  readonly wet: Float32Array[]
  readonly sampleRate: number
}

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
      setProgress(share * 0.2, `Downloading the model, ${Math.round(share * 100)}%`)
      break
    }
    case 'ready': {
      modelReady = true
      dom.engineStatus.classList.add('status--live')
      setStatus(`${model.name.toUpperCase()} · 48 kHz · runs on your machine`)
      break
    }
    case 'process-progress': {
      if (message.job !== currentJob) return
      setProgress(
        0.2 + message.share * 0.8,
        `Cleaning up, ${Math.round(message.share * 100)}%. ${formatSpeed(message.processedSeconds, message.elapsedMs)}`,
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
  show(result ? 'done' : 'pick')
}

function clearAlert(): void {
  dom.alert.hidden = true
  dom.alert.textContent = ''
}

async function load(file: File): Promise<void> {
  clearAlert()
  player.dispose()
  result = null
  stage.clear()
  dom.readout.replaceChildren()
  currentJob += 1
  const job = currentJob

  show('busy')
  setProgress(0.02, 'Reading the file')

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

  pendingName = file.name
  pendingDry = channels.map((channel) => channel.slice())
  setProgress(modelReady ? 0.2 : 0.05, modelReady ? 'Cleaning up' : 'Loading the model')
  send(
    { type: 'process', job, channels, sampleRate: modelSampleRate },
    channels.map((channel) => channel.buffer),
  )
}

let pendingName = ''
let pendingDry: Float32Array[] = []

function finish(wet: Float32Array[], elapsedMs: number): void {
  const dry = pendingDry
  if (dry.length === 0) return
  result = { name: pendingName, dry, wet, sampleRate: modelSampleRate }
  const seconds = dry[0].length / modelSampleRate

  // The chart plots what the pass did: the input as the filled envelope, the output as the
  // trace. What still shows through the fill is what came out.
  stage.show(
    { dry, wet, sampleRate: modelSampleRate },
    floorDifferenceDb(dry, wet, modelSampleRate),
  )

  dom.readout.replaceChildren(
    ...(
      [
        ['File', `${pendingName}, ${formatClock(seconds)}`],
        ['Channels', formatChannels(dry.length)],
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
    // Handle for the numerical comparison in scripts/; stripped from production builds.
    ;(window as unknown as { deadroom?: unknown }).deadroom = result
  }

  player.load(dry, wet, modelSampleRate)
  player.setMix(stage.amount)
  player.setBypassed(dom.bypassCheck.checked)
  show('done')
  dom.playButton.focus()
}

function reset(): void {
  player.dispose()
  dom.bypassCheck.checked = false
  result = null
  pendingDry = []
  currentJob += 1
  stage.clear()
  clearAlert()
  dom.readout.replaceChildren()
  dom.fileInput.value = ''
  show('pick')
}

// --- transport -----------------------------------------------------------

player.subscribe((state) => {
  dom.playButton.textContent = state.playing ? 'Pause' : 'Play'
  dom.transportTime.textContent = `${formatClock(state.currentTime)} / ${formatClock(state.duration)}`
  stage.setPlayhead(state.currentTime)
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
    stage.setPlayhead(player.currentTime)
    tick()
  })
}

stage.plot.onSeek((seconds) => {
  if (result) player.seek(seconds)
})

// The mix is a blend of two signals already in hand, so it is free and live.
stage.onAmount((amount) => player.setMix(amount))

// --- events --------------------------------------------------------------

dom.filepickButton.addEventListener('pointerenter', warmModel)
dom.fileInput.addEventListener('focus', warmModel)
dom.exampleButton.addEventListener('pointerenter', warmModel)
dom.fileInput.addEventListener('change', () => {
  const file = dom.fileInput.files?.[0]
  if (file) void load(file)
})

dom.exampleButton.addEventListener('click', () => void loadExample())

async function loadExample(): Promise<void> {
  clearAlert()
  try {
    const response = await fetch(`${import.meta.env.BASE_URL}example/example.mp3`)
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    const blob = await response.blob()
    await load(new File([blob], 'example.mp3', { type: 'audio/mpeg' }))
  } catch {
    fail('Could not load the example. Check your connection and try again.')
  }
}

function warmModel(): void {
  if (!modelReady) send({ type: 'load' })
}

dom.cancelButton.addEventListener('click', () => {
  send({ type: 'cancel' })
  setProgress(0, 'Stopping')
})
dom.resetButton.addEventListener('click', reset)
dom.bypassCheck.addEventListener('change', () => player.setBypassed(dom.bypassCheck.checked))
dom.playButton.addEventListener('click', () => player.toggle())

dom.downloadButton.addEventListener('click', () => {
  if (!result) return
  const wav = encodeWav(blend(result.dry, result.wet, stage.amount), {
    sampleRate: result.sampleRate,
    bitDepth: Number(dom.depthSelect.value) as WavBitDepth,
  })
  const url = URL.createObjectURL(new Blob([wav], { type: 'audio/wav' }))
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = `${result.name.replace(/\.[^.]+$/, '')} cleaned.wav`
  anchor.click()
  URL.revokeObjectURL(url)
})

document.addEventListener('keydown', (event) => {
  if (event.code !== 'Space' || !result) return
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
