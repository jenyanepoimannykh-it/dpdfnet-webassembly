// Reads the custom metadata map out of the DPDFNet ONNX file and writes the pieces the
// browser needs as plain JSON. onnxruntime-web does not expose a model's metadata map, so
// the recurrent state seed has to travel beside the weights rather than inside them.
//
// Only the top level of ModelProto is walked, and only field 14 (metadata_props, a repeated
// StringStringEntryProto with key = field 1 and value = field 2). Everything else is skipped
// by length, so no protobuf schema or dependency is required.
import { readdirSync, readFileSync, writeFileSync } from 'node:fs'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const modelDir = resolve(root, 'public/models')

function extract(modelPath) {
const bytes = readFileSync(modelPath)

function readVarint(buf, pos) {
  let result = 0n
  let shift = 0n
  for (;;) {
    const byte = buf[pos++]
    result |= BigInt(byte & 0x7f) << shift
    if ((byte & 0x80) === 0) return [result, pos]
    shift += 7n
  }
}

function* fields(buf, start, end) {
  let pos = start
  while (pos < end) {
    let tag
    ;[tag, pos] = readVarint(buf, pos)
    const fieldNumber = Number(tag >> 3n)
    const wireType = Number(tag & 7n)
    if (wireType === 0) {
      ;[, pos] = readVarint(buf, pos)
      yield { fieldNumber, wireType }
    } else if (wireType === 1) {
      pos += 8
      yield { fieldNumber, wireType }
    } else if (wireType === 2) {
      let length
      ;[length, pos] = readVarint(buf, pos)
      const from = pos
      pos += Number(length)
      yield { fieldNumber, wireType, from, to: pos }
    } else if (wireType === 5) {
      pos += 4
      yield { fieldNumber, wireType }
    } else {
      throw new Error(`unsupported wire type ${wireType} at ${pos}`)
    }
  }
}

const metadata = new Map()
for (const field of fields(bytes, 0, bytes.length)) {
  if (field.fieldNumber !== 14 || field.wireType !== 2) continue
  let key = null
  let value = null
  for (const entry of fields(bytes, field.from, field.to)) {
    if (entry.wireType !== 2) continue
    const text = bytes.toString('utf8', entry.from, entry.to)
    if (entry.fieldNumber === 1) key = text
    else if (entry.fieldNumber === 2) value = text
  }
  if (key !== null && value !== null) metadata.set(key, value)
}

const required = [
  'sample_rate', 'n_fft', 'hop_length', 'freq_bins', 'state_size',
  'erb_norm_state_size', 'spec_norm_state_size', 'erb_norm_init', 'spec_norm_init',
]
const missing = required.filter((key) => !metadata.has(key))
if (missing.length > 0) throw new Error(`model metadata is missing ${missing.join(', ')}`)

const number = (key) => {
  const parsed = Number(metadata.get(key))
  if (!Number.isFinite(parsed)) throw new Error(`model metadata ${key} is not a number`)
  return parsed
}
const floats = (key, expectedLength) => {
  const values = metadata.get(key).split(',').map(Number)
  if (values.length !== expectedLength || values.some((value) => !Number.isFinite(value)))
    throw new Error(`model metadata ${key} is not ${expectedLength} finite floats`)
  return values
}

// The C++ reference seeds the ERB normalisation history at offset 0 and the spectral
// normalisation history immediately after it; the rest of the state starts at zero.
const erbLength = number('erb_norm_state_size')
const meta = {
  modelType: metadata.get('model_type'),
  profile: metadata.get('profile'),
  sampleRate: number('sample_rate'),
  fftSize: number('n_fft'),
  hopSize: number('hop_length'),
  bins: number('freq_bins'),
  windowType: metadata.get('window_type'),
  stateSize: number('state_size'),
  stateInit: [
    { offset: 0, values: floats('erb_norm_init', erbLength) },
    { offset: erbLength, values: floats('spec_norm_init', number('spec_norm_state_size')) },
  ],
}

const outPath = modelPath.replace(/\.onnx$/, '.meta.json')
writeFileSync(outPath, JSON.stringify(meta))
const seeded = meta.stateInit.reduce((total, segment) => total + segment.values.length, 0)
// `profile` is not trustworthy: dpdfnet8_48khz_hr carries dpdfnet2_48khz_hr's string
// upstream, which is why the site names models itself rather than reading them from here.
console.log(
  `${modelPath.split('/').pop()}: ${meta.sampleRate} Hz, fft ${meta.fftSize}/${meta.hopSize}, ` +
  `${meta.bins} bins, state ${meta.stateSize} (${seeded} seeded), profile says "${meta.profile}"`,
)
}

for (const name of readdirSync(modelDir).filter((file) => file.endsWith('.onnx')).sort()) {
  extract(resolve(modelDir, name))
}
