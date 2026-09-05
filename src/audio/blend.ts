/** dry + mix * (wet - dry), the same law the plug-in applies per sample and the player uses. */
export function blend(
  dry: readonly Float32Array[],
  wet: readonly Float32Array[],
  mix: number,
): Float32Array[] {
  if (mix >= 1) return wet.map((channel) => channel)
  if (mix <= 0) return dry.map((channel) => channel)
  return dry.map((dryChannel, index) => {
    const wetChannel = wet[index]
    const out = new Float32Array(dryChannel.length)
    for (let i = 0; i < dryChannel.length; i += 1) {
      out[i] = dryChannel[i] + mix * (wetChannel[i] - dryChannel[i])
    }
    return out
  })
}

/** Broadband level change in dB, used to say how much room came out. */
export function levelDifferenceDb(
  dry: readonly Float32Array[],
  wet: readonly Float32Array[],
): number {
  let dryEnergy = 1e-12
  let wetEnergy = 1e-12
  for (let channel = 0; channel < dry.length; channel += 1) {
    const a = dry[channel]
    const b = wet[channel]
    for (let i = 0; i < a.length; i += 1) {
      dryEnergy += a[i] * a[i]
      wetEnergy += b[i] * b[i]
    }
  }
  return 10 * Math.log10(wetEnergy / dryEnergy)
}
