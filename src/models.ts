// The two 48 kHz networks CEVA publish. They share an ONNX signature exactly — same STFT,
// same tensor names, only the recurrent state length differs — so switching is a matter of
// pointing at a different file. Names live here rather than in the weights: the larger
// model carries the smaller one's `profile` string in its own metadata upstream.

export interface ModelChoice {
  readonly id: string
  /** Shown in the picker. */
  readonly label: string
  /** Shown in the status line once loaded. */
  readonly name: string
  readonly megabytes: number
}

export const modelChoices: readonly ModelChoice[] = [
  {
    id: 'dpdfnet2_48khz_hr',
    label: 'Standard — 10 MB',
    name: 'DPDFNet-2',
    megabytes: 10,
  },
  {
    id: 'dpdfnet8_48khz_hr',
    label: 'Larger — 15 MB, about 3× slower',
    name: 'DPDFNet-8',
    megabytes: 15,
  },
]

export const defaultModelId = modelChoices[0].id

export function findModel(id: string): ModelChoice {
  const found = modelChoices.find((choice) => choice.id === id)
  if (!found) throw new Error(`unknown model ${id}`)
  return found
}

/** Weights live in public/ so the browser can cache them across deploys. */
export function modelUrls(id: string): { modelUrl: string; metadataUrl: string } {
  const base = `${import.meta.env.BASE_URL}models/${findModel(id).id}`
  return { modelUrl: `${base}.onnx`, metadataUrl: `${base}.meta.json` }
}
