export type ToolMode = 'denoise' | 'dereverb'

interface ToolCopy {
  readonly name: string
  readonly tag: string
  /** What the processed side of the toggle is called: "100% denoised". */
  readonly pastTense: string
  readonly moduleReading: string
  readonly moduleNote: string
  readonly pickerNote: string
  readonly exampleLabel: string
  readonly howItWorks: readonly string[]
}

const copy: Record<ToolMode, ToolCopy> = {
  denoise: {
    name: 'Denoise',
    tag: 'Neural noise removal for voice, in the browser',
    pastTense: 'denoised',
    moduleReading: 'DPDFNet-8, 48 kHz',
    moduleNote:
      'One pass of the network recovers 8 to 15 dB on speech buried in noise, with most of the improvement where the input is worst.',
    pickerNote:
      'Or drop one anywhere on this page. Video files are read for their sound and come back as a WAV. Stereo is summed to mono unless you ask otherwise — it halves the work and speech loses nothing. The example is nine seconds of speech recorded under room noise.',
    exampleLabel: 'Play noisy example',
    howItWorks: [
      'Hush runs <strong>DPDFNet</strong>, a dual-path speech enhancement network published by CEVA, through ONNX Runtime compiled to WebAssembly. Your file is decoded by the browser, resampled to 48&nbsp;kHz, and processed 20&nbsp;milliseconds at a time.',
      `The model is strongest on steady and environmental noise. Room sound comes out in the same pass — 5 to 6 dB off the tail between words — because both are part of one learned output rather than separate controls. If the room is the problem rather than the noise, <a href="${route('dereverb')}">Dereverb</a> is the same pass framed around it.`,
      "The first visit downloads about 15&nbsp;MB of model weights and keeps them in the browser's cache, so later visits start straight away.",
    ],
  },
  dereverb: {
    name: 'Dereverb',
    tag: 'Neural room removal for voice, in the browser',
    pastTense: 'dereverbed',
    moduleReading: 'DPDFNet-8, 48 kHz',
    moduleNote:
      'The network shortens a room by 5 to 6 dB measured on the tail between words, and takes the steady noise in the same recording out with it. It is the engine the desktop plug-in runs.',
    pickerNote:
      'Or drop one anywhere on this page. Video files are read for their sound and come back as a WAV. Stereo is summed to mono unless you ask otherwise — it halves the work and speech loses nothing. The example is spoken voice recorded in a live room.',
    exampleLabel: 'Play reverbed example',
    howItWorks: [
      'Dereverb runs <strong>DPDFNet</strong>, a dual-path speech enhancement network published by CEVA, through ONNX Runtime compiled to WebAssembly — the same engine, and the same weights, as the JenyaDereverb2 VST3 and AU plug-in. Your file is decoded by the browser, resampled to 48&nbsp;kHz, and processed 20&nbsp;milliseconds at a time.',
      `The network takes room and noise out in one learned pass rather than as two controls, so a recording with a fan running under the voice comes back with both gone; <a href="${route('denoise')}">Denoise</a> is the same pass framed around that job. It attenuates the reflections it recognises; de-reverberation cannot restore what was never captured, so a very distant or clipped recording may still come back with artifacts. Keep the original.`,
      "The first visit downloads about 15&nbsp;MB of model weights and keeps them in the browser's cache, so later visits start straight away. The result stays sample-aligned with the input and can go straight back onto the video it came from.",
    ],
  },
}

function route(path: string): string {
  return `${import.meta.env.BASE_URL}${path}/`
}

export function pageMode(): ToolMode {
  return document.body.dataset.tool === 'dereverb' ? 'dereverb' : 'denoise'
}

export function renderPage(mode: ToolMode): void {
  const content = copy[mode]
  const denoiseCurrent = mode === 'denoise' ? ' aria-current="page"' : ''
  const dereverbCurrent = mode === 'dereverb' ? ' aria-current="page"' : ''
  const effectControl = `<div class="binary-effect">
          <label class="binary" for="effect-toggle">
            <span>Off</span>
            <input type="checkbox" id="effect-toggle" checked />
            <span class="binary__track" aria-hidden="true"><span></span></span>
            <span>On</span>
          </label>
          <output class="binary-effect__value" id="effect-value" for="effect-toggle">100% ${content.pastTense}</output>
        </div>`
  const engineStatus =
    '<p class="status" id="engine-status"><span class="status__lamp"></span>Model not loaded</p>'
  const moduleReading = content.moduleReading
    ? `<p class="module__reading">${content.moduleReading}</p>`
    : ''
  const bypassControl = '<input type="checkbox" id="bypass-check" hidden />'

  document.body.innerHTML = `
    <a class="skip-link" href="#panel">Skip to the file picker</a>

    <div class="shell">
      <header class="masthead">
        <div class="masthead__row">
          <a class="masthead__brand" href="${route('dereverb')}"><img src="${import.meta.env.BASE_URL}favicon.svg" alt="" />Hush</a>
          <nav class="toolnav" aria-label="Voice cleanup tools">
            <a href="${route('dereverb')}"${dereverbCurrent}>Dereverb</a>
            <a href="${route('denoise')}"${denoiseCurrent}>Denoise</a>
          </nav>
        </div>
        <h1 class="masthead__name">${content.name}</h1>
        <p class="masthead__tag">${content.tag}</p>
      </header>

      <section class="rack" id="panel">
        <article class="module">
          <header class="module__head">
            <h2 class="module__title">${content.name}</h2>
            ${moduleReading}
          </header>
          <div class="module__frame">
            <canvas class="module__canvas" id="stage-canvas"></canvas>
          </div>
          <div class="module__gauges">
            <div class="stat stat--idle" id="stage-stat">
              <button class="idle-play" id="example-button" type="button" aria-label="${content.exampleLabel}"><span aria-hidden="true">▶</span></button>
              <span class="stat__value" id="stage-value"></span>
              <span class="stat__unit" id="stage-unit"></span>
            </div>
            ${effectControl}
          </div>
          <p class="module__note">${content.moduleNote}</p>
        </article>

        ${engineStatus}
      </section>

      <section class="controls" aria-live="polite">
        <div class="stage" id="stage-pick">
          <div class="stage__row">
            <label class="filepick">
              <input type="file" id="file-input" accept="audio/*,video/*" />
              <span class="filepick__button">Choose a file</span>
            </label>
            <label class="check">
              <input type="checkbox" id="stereo-check" />
              <span>Keep both channels</span>
            </label>
          </div>
          <p class="stage__note">${content.pickerNote}</p>
        </div>

        <div class="stage" id="stage-busy" hidden>
          <div class="meter"><div class="meter__fill" id="progress-fill"></div></div>
          <p class="stage__status" id="progress-label">Loading the model</p>
          <button class="button" id="cancel-button" type="button">Cancel</button>
        </div>

        <div class="stage" id="stage-done" hidden>
          <div class="stage__row">
            <button class="button button--primary" id="play-button" type="button"><span class="button__icon" aria-hidden="true">▶</span><span>Play</span></button>
            <span class="clock" id="transport-time">0:00 / 0:00</span>
            ${bypassControl}
          </div>
          <div class="stage__row stage__row--split">
            <button class="button button--primary" id="download-button" type="button"><span class="button__icon button__icon--large" aria-hidden="true">↓</span><span>Download WAV</span></button>
            <label class="check">
              <span>Depth</span>
              <select id="depth-select">
                <option value="24" selected>24-bit</option>
                <option value="16">16-bit</option>
                <option value="32">32-bit float</option>
              </select>
            </label>
            <button class="button" id="reset-button" type="button"><span class="button__icon button__icon--large" aria-hidden="true">↻</span><span>Process another file</span></button>
          </div>
          <dl class="readout" id="file-readout"></dl>
        </div>

        <p class="alert" id="alert" hidden role="alert"></p>
      </section>

      <section class="notes">
        <h2 class="notes__title">How it works</h2>
        ${content.howItWorks.map((paragraph) => `<p>${paragraph}</p>`).join('')}
        <p class="notes__credit">DPDFNet is © CEVA,&nbsp;Inc., released under the Apache&nbsp;License&nbsp;2.0. ONNX Runtime is © Microsoft, MIT licensed.</p>
      </section>

      <footer class="footer">
        <p class="footer__plugin">Also available as a <a href="https://github.com/jenyanepoimannykh-it/dereverb-vst3" target="_blank" rel="noreferrer">VST3 and AU plug-in</a>.</p>
        <div class="footer__socials">
          <a class="social-link" href="https://github.com/jenyanepoimannykh-it/dpdfnet-webassembly" target="_blank" rel="noreferrer" aria-label="Hush on GitHub">
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2a10 10 0 0 0-3.16 19.49c.5.09.68-.22.68-.48v-1.87c-2.78.6-3.37-1.18-3.37-1.18-.45-1.16-1.11-1.47-1.11-1.47-.91-.62.07-.61.07-.61 1 .07 1.53 1.03 1.53 1.03.9 1.53 2.35 1.09 2.92.83.09-.65.35-1.09.64-1.34-2.22-.25-4.56-1.11-4.56-4.94 0-1.09.39-1.98 1.03-2.68-.1-.25-.45-1.27.1-2.64 0 0 .84-.27 2.75 1.02A9.56 9.56 0 0 1 12 6.82a9.55 9.55 0 0 1 2.5.34c1.91-1.29 2.75-1.02 2.75-1.02.55 1.37.2 2.39.1 2.64.64.7 1.03 1.59 1.03 2.68 0 3.84-2.34 4.68-4.57 4.93.36.31.68.92.68 1.85v2.77c0 .27.18.58.69.48A10 10 0 0 0 12 2Z" /></svg>
          </a>
          <a class="social-link" href="https://www.linkedin.com/in/zunso" target="_blank" rel="noreferrer" aria-label="Zunso on LinkedIn">
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6.5 8.3H3.2V19h3.3V8.3ZM4.85 3A1.93 1.93 0 1 0 4.85 6.86 1.93 1.93 0 0 0 4.85 3ZM19.8 12.87c0-3.22-1.72-4.72-4.02-4.72a3.46 3.46 0 0 0-3.13 1.72V8.3H9.33V19h3.32v-5.3c0-1.4.27-2.77 2.02-2.77 1.73 0 1.75 1.62 1.75 2.86V19h3.33l.05-6.13Z" /></svg>
          </a>
        </div>
      </footer>
    </div>

    <div class="dropveil" id="dropveil" hidden><span>Drop to load</span></div>
  `
}
