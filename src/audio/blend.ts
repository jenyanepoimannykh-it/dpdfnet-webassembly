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

/**
 * Change in the quiet floor, in dB.
 *
 * Broadband level barely moves when a stage works well — speech dominates the average and
 * speech is what both stages are trying to keep. What actually changes is the floor between
 * words, which is where reverberation tails and steady noise both live. This measures the
 * 20th-percentile 20 ms frame level before and after, so the number on each module is the
 * thing a listener notices.
 *
 * Both readings are clamped at -90 dBFS, under the noise floor of any 16-bit delivery. The
 * network takes the gaps to digital silence, and without the clamp the ratio runs off to
 * 80 dB and more, which is arithmetically true and tells a listener nothing.
 */
export function floorDifferenceDb(
  before: readonly Float32Array[],
  after: readonly Float32Array[],
  sampleRate: number,
): number {
  // -90 dBFS in energy.
  const silenceEnergy = 1e-9
  const quiet = (channels: readonly Float32Array[]) => {
    const window = Math.max(1, Math.round(0.02 * sampleRate))
    const levels: number[] = []
    for (const channel of channels) {
      for (let start = 0; start + window <= channel.length; start += window) {
        let energy = 0
        for (let i = start; i < start + window; i += 1) energy += channel[i] * channel[i]
        levels.push(energy / window)
      }
    }
    if (levels.length === 0) return silenceEnergy
    levels.sort((a, b) => a - b)
    return Math.max(levels[Math.floor(levels.length * 0.2)], silenceEnergy)
  }
  return 10 * Math.log10(quiet(after) / quiet(before))
}
