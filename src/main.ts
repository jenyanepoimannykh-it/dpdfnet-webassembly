import './styles.css'
import { decodeFile, DecodeError, downmixToMono } from './audio/decode'
import { blend, floorDifferenceDb } from './audio/blend'
import { encodeWav, type WavBitDepth } from './audio/wav'
import { Player } from './audio/player'
import { model } from './models'
import { Stage } from './ui/stage'
import { demoClean, demoNoisy } from './ui/demo-trace'
import { formatChannels, formatClock, formatSpeed } from './ui/format'
import { pageMode, renderPage } from './ui/page'
import type { WorkerRequest, WorkerResponse } from './worker/protocol'

const toolMode = pageMode()
renderPage(toolMode)

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
  effectToggle: element<HTMLInputElement>('effect-toggle'),
  effectValue: element<HTMLOutputElement>('effect-value'),
}

const stage = new Stage(
  {
    canvas: 'stage-canvas',
    stat: 'stage-stat',
    value: 'stage-value',
    unit: 'stage-unit',
  },
  // Dereverb has no canned trace: its idle chart is measured from the example clip itself
  // as soon as that decodes, so the picture is always the audio the play button plays.
  toolMode === 'dereverb'
    ? { before: new Float32Array(0), after: new Float32Array(0) }
    : { before: demoNoisy, after: demoClean },
)

const player = new Player()
const examplePlayer = new Player()

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
let exampleFrameHandle = 0
let exampleReady = false
let exampleLoading: Promise<void> | null = null
let exampleSource: Float32Array[] | null = null
let exampleProcessed: Float32Array[] | null = null
/** Job number of an example run in flight, or 0. Denoise has to run the model to get one. */
let exampleJob = 0
let exampleOutput: Promise<void> | null = null
let examplePending: { resolve: () => void; reject: (reason: Error) => void } | null = null

/** The clip behind the idle play button. Dereverb ships the counterpart it renders in a
 *  fraction of a second; the denoise model is 15 MB and runs on the clip when asked. */
const exampleNames: Record<typeof toolMode, { source: string; processed?: string }> = {
  dereverb: { source: 'dereverb.mp3', processed: 'dereverb-processed.mp3' },
  denoise: { source: 'example.mp3' },
}

const progressVerb = toolMode === 'dereverb' ? 'Taking the room out' : 'Removing noise'
const exampleVerb =
  toolMode === 'dereverb' ? 'Taking the room out of the example' : 'Removing noise from the example'
const downloadSuffix = toolMode === 'dereverb' ? 'dereverbed' : 'denoised'
// Both tools download the same weights first, and that download is the first fifth of the bar.
const progressBase = 0.2

function outputAmount(): number {
  return Number(dom.effectToggle.checked)
}

function readyStatus(): string {
  return `${model.name.toUpperCase()} · 48 kHz · runs on your machine`
}

/** A file run owns the progress bar; an example run has only the status line to talk in. */
function report(share: number, label: string): void {
  if (dom.stageBusy.hidden) setStatus(label)
  else setProgress(share, label)
}

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
      report(share * 0.2, `Downloading the model, ${Math.round(share * 100)}%`)
      break
    }
    case 'ready': {
      modelReady = true
      dom.engineStatus.classList.add('status--live')
      if (exampleJob === 0) setStatus(readyStatus())
      break
    }
    case 'process-progress': {
      if (message.job === exampleJob) {
        setStatus(`${exampleVerb}, ${Math.round(message.share * 100)}%`)
        return
      }
      if (message.job !== currentJob) return
      setProgress(
        progressBase + message.share * (1 - progressBase),
        `${progressVerb}, ${Math.round(message.share * 100)}%. ${formatSpeed(message.processedSeconds, message.elapsedMs)}`,
      )
      break
    }
    case 'processed': {
      if (message.job === exampleJob) {
        finishExample(message.channels)
        return
      }
      if (message.job !== currentJob) return
      finish(message.channels, message.elapsedMs)
      break
    }
    case 'cancelled': {
      if (message.job === exampleJob) {
        abandonExample(new Error('cancelled'))
        return
      }
      if (message.job === currentJob) reset()
      break
    }
    case 'error': {
      abandonExample(new Error(message.message))
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
  stopExample()
  // The worker takes one run at a time; a half-finished example must not hold up the file.
  if (exampleJob !== 0) {
    send({ type: 'cancel' })
    abandonExample(new Error('cancelled'))
  }
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
  setProgress(modelReady ? progressBase : 0.05, modelReady ? progressVerb : 'Loading the model')
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

  refreshResultView()

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
  player.setMix(outputAmount())
  player.setBypassed(dom.bypassCheck.checked)
  show('done')
  dom.playButton.focus()
}

/** The idle chart before a file is loaded: the example clip, drawn in the state the
 *  toggle is in, so the picture always matches what pressing play produces. Until the
 *  denoise model has run on the example there is nothing measured to draw, and the canned
 *  reference trace of the same clip stands in. */
function refreshIdleView(): void {
  if (!exampleSource || !exampleProcessed) return
  stage.setIdleAudio(
    exampleSource,
    dom.effectToggle.checked ? exampleProcessed : exampleSource,
  )
}

function refreshResultView(): void {
  if (!result) return
  const visibleOutput = dom.effectToggle.checked ? result.wet : result.dry
  stage.show(
    { dry: result.dry, wet: visibleOutput, sampleRate: result.sampleRate },
    floorDifferenceDb(result.dry, visibleOutput, result.sampleRate),
  )
  stage.setPlayhead(player.currentTime)
}

function reset(): void {
  player.dispose()
  dom.bypassCheck.checked = false
  result = null
  pendingDry = []
  currentJob += 1
  stage.clear()
  if (exampleReady) stage.setIdleDuration(examplePlayer.duration)
  clearAlert()
  dom.readout.replaceChildren()
  dom.fileInput.value = ''
  show('pick')
}

// --- transport -----------------------------------------------------------

player.subscribe((state) => {
  dom.playButton.innerHTML = state.playing
    ? '<span class="button__icon" aria-hidden="true">Ⅱ</span><span>Pause</span>'
    : '<span class="button__icon" aria-hidden="true">▶</span><span>Play</span>'
  dom.playButton.setAttribute('aria-label', state.playing ? 'Pause' : 'Play')
  dom.transportTime.textContent = `${formatClock(state.currentTime)} / ${formatClock(state.duration)}`
  stage.setPlayhead(state.currentTime)
  if (state.playing && frameHandle === 0) tick()
  if (!state.playing && frameHandle !== 0) {
    cancelAnimationFrame(frameHandle)
    frameHandle = 0
  }
})

examplePlayer.subscribe((state) => {
  setExamplePlaying(state.playing)
  if (!result) {
    stage.setIdleDuration(state.duration)
    stage.setPlayhead(state.currentTime)
  }
  if (state.playing && exampleFrameHandle === 0) tickExample()
  if (!state.playing && exampleFrameHandle !== 0) {
    cancelAnimationFrame(exampleFrameHandle)
    exampleFrameHandle = 0
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

function tickExample(): void {
  exampleFrameHandle = requestAnimationFrame(() => {
    exampleFrameHandle = 0
    if (!examplePlayer.isPlaying) return
    if (!result) stage.setPlayhead(examplePlayer.currentTime)
    tickExample()
  })
}

stage.plot.onSeek((seconds) => {
  if (result) player.seek(seconds)
  else if (exampleReady) examplePlayer.seek(seconds)
})

// --- events --------------------------------------------------------------

dom.filepickButton.addEventListener('pointerenter', warmModel)
dom.fileInput.addEventListener('focus', warmModel)
dom.exampleButton.addEventListener('pointerenter', () => void prepareExample().catch(() => {}))
dom.exampleButton.addEventListener('focus', () => void prepareExample().catch(() => {}))
dom.fileInput.addEventListener('change', () => {
  const file = dom.fileInput.files?.[0]
  if (file) void load(file)
})

dom.exampleButton.addEventListener('click', () => void toggleExample())

async function prepareExample(): Promise<void> {
  if (exampleReady) return
  if (exampleLoading) return exampleLoading
  exampleLoading = (async () => {
    const names = exampleNames[toolMode]
    const decoded = await Promise.all(
      [names.source, names.processed ?? names.source].map(async (name) => {
        const response = await fetch(`${import.meta.env.BASE_URL}example/${name}`)
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        const blob = await response.blob()
        return decodeFile(new File([blob], name, { type: 'audio/mpeg' }), modelSampleRate)
      }),
    )
    // Two separately encoded files can decode a frame or two apart; the player mixes them
    // sample for sample, so trim both to the length they share.
    const length = Math.min(...decoded.map((clip) => clip.channels[0].length))
    const [source, processed] = decoded.map((clip) =>
      clip.channels.map((channel) => channel.subarray(0, length)),
    )
    exampleSource = source
    exampleProcessed = names.processed ? processed : null
    examplePlayer.load(source, exampleProcessed ?? source, modelSampleRate)
    examplePlayer.setMix(outputAmount())
    exampleReady = true
    refreshIdleView()
  })()
  try {
    await exampleLoading
  } finally {
    exampleLoading = null
  }
}

/**
 * Denoise ships no rendered counterpart to the example — it would be a second copy of a
 * clip the model can produce itself — so the first press of play downloads the weights and
 * runs the example through the same pass a picked file gets. After that the toggle A/Bs
 * the real thing, and the idle chart is a measurement rather than a reference trace.
 */
function ensureExampleOutput(): Promise<void> {
  if (exampleProcessed || !exampleSource) return Promise.resolve()
  if (exampleOutput) return exampleOutput
  const channels = exampleSource.map((channel) => Float32Array.from(channel))
  currentJob += 1
  exampleJob = currentJob
  setStatus(modelReady ? exampleVerb : 'Loading the model')
  setExampleWorking(true)
  exampleOutput = new Promise<void>((resolve, reject) => {
    examplePending = { resolve, reject }
    send(
      { type: 'process', job: exampleJob, channels, sampleRate: modelSampleRate },
      channels.map((channel) => channel.buffer),
    )
  })
  return exampleOutput
}

function finishExample(wet: Float32Array[]): void {
  exampleJob = 0
  setExampleWorking(false)
  setStatus(readyStatus())
  if (!exampleSource) return
  exampleProcessed = wet
  examplePlayer.load(exampleSource, wet, modelSampleRate)
  examplePlayer.setMix(outputAmount())
  refreshIdleView()
  examplePending?.resolve()
  examplePending = null
}

/** Clears an example run that was cancelled or failed, so the next press can try again. */
function abandonExample(reason: Error): void {
  if (exampleJob === 0) return
  exampleJob = 0
  exampleOutput = null
  setExampleWorking(false)
  setStatus(readyStatus())
  examplePending?.reject(reason)
  examplePending = null
}

function setExampleWorking(working: boolean): void {
  dom.exampleButton.disabled = working
  if (working) {
    const icon = dom.exampleButton.querySelector('span')
    if (icon) icon.textContent = '\u22EF'
    dom.exampleButton.setAttribute('aria-label', 'Preparing the example')
  } else {
    setExamplePlaying(examplePlayer.isPlaying)
  }
}

function setExamplePlaying(playing: boolean): void {
  const icon = dom.exampleButton.querySelector('span')
  if (icon) icon.textContent = playing ? 'Ⅱ' : '▶'
  dom.exampleButton.setAttribute(
    'aria-label',
    `${playing ? 'Pause' : 'Play'} ${toolMode === 'dereverb' ? 'reverbed' : 'noisy'} example`,
  )
}

function stopExample(): void {
  if (examplePlayer.isPlaying) examplePlayer.pause()
  if (examplePlayer.currentTime !== 0) examplePlayer.seek(0)
}

async function toggleExample(): Promise<void> {
  clearAlert()
  if (examplePlayer.isPlaying) {
    stopExample()
    return
  }

  try {
    await prepareExample()
    await ensureExampleOutput()
    examplePlayer.play()
  } catch {
    stopExample()
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
// Either state is whole: on is the processed signal alone, off is the original alone.
function refreshEffectLabel(): void {
  dom.effectValue.value = dom.effectToggle.checked ? `100% ${downloadSuffix}` : '100% original'
}

dom.effectToggle.addEventListener('change', () => {
  refreshEffectLabel()
  dom.bypassCheck.checked = false
  player.setBypassed(false)
  player.setMix(outputAmount())
  examplePlayer.setMix(outputAmount())
  refreshResultView()
  refreshIdleView()
})

dom.downloadButton.addEventListener('click', () => {
  if (!result) return
  const wav = encodeWav(blend(result.dry, result.wet, outputAmount()), {
    sampleRate: result.sampleRate,
    bitDepth: Number(dom.depthSelect.value) as WavBitDepth,
  })
  const url = URL.createObjectURL(new Blob([wav], { type: 'audio/wav' }))
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = `${result.name.replace(/\.[^.]+$/, '')} ${downloadSuffix}.wav`
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

// A reload can restore the toggle to where it was left, so say what it actually reads.
refreshEffectLabel()

// Start fetching the tiny built-in example so clicking Play can start immediately.
void prepareExample().catch(() => {})
