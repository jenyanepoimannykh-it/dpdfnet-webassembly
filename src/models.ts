// The network this build ships: DPDFNet-8 at 48 kHz, the largest of the family CEVA
// publish. Its name lives here rather than being read from the weights, because the file
// carries the smaller model's `profile` string in its own metadata upstream.

export const model = {
  id: 'dpdfnet8_48khz_hr',
  name: 'DPDFNet-8',
  sampleRate: 48000,
  megabytes: 15,
} as const

/** Weights live in public/ so the browser can cache them across deploys. */
export function modelUrls(): { modelUrl: string; metadataUrl: string } {
  const base = `${import.meta.env.BASE_URL}models/${model.id}`
  return { modelUrl: `${base}.onnx`, metadataUrl: `${base}.meta.json` }
}
