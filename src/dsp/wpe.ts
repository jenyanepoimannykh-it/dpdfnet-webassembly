// Stage one: de-reverberation by weighted prediction error, the offline formulation
// published by NARA-WPE (MIT, Paderborn University).
//
// Late reverberation at frame t is largely predictable from the same frequency band a few
// frames earlier. Per bin, a linear filter over `taps` past frames — starting `delay`
// frames back so the direct sound and early reflections are left alone — is fitted to the
// observation, weighted by the inverse of the current speech power estimate. Subtracting
// the prediction leaves the residual. Two passes: the first weights by the observed power,
// the second by the power of the first pass's residual.
//
// This is not the OnlineWpe.cpp that sits unused in the plug-in repository. That is a
// recursive least-squares variant which was never added to the build and, ported and
// measured, made material worse. The batch form below is the one WPE is known for, and it
// is the right shape for an offline tool anyway: nothing here has to be causal.
//
// `amount` deliberately plays no part in the fit. The residual returned is always the full
// one, so the interface can re-blend the stage without re-running it.
import { Dft } from './fft'

export interface WpeOptions {
  readonly sampleRate: number
  readonly onProgress?: (done: number, total: number) => void
  readonly shouldCancel?: () => boolean
}

export interface WpeResult {
  /** The full residual, sample-aligned with the input and the same length. */
  readonly residual: Float32Array
  /** Per-frame level change in dB, negative where the filter removed energy. */
  readonly reductionDb: Float32Array
  readonly hopSize: number
}

const fftSize = 4096
const hopSize = fftSize / 4
const bins = fftSize / 2 + 1
const taps = 12
const delayFrames = 2
const iterations = 2

/**
 * Reverberation energy is concentrated low, and the cost of the fit is linear in the number
 * of bins. Fitting only up to 8 kHz is eight times cheaper than the full band and keeps
 * 89% of the measured benefit; above it the observation passes through untouched.
 */
const fitCeilingHz = 8000

/** Hann applied on both analysis and synthesis at 75% overlap sums to 1.5. */
const overlapScale = 2 / 3

/** Frames fitted as one block, and the extra context each block sees on either side. */
const blockFrames = Math.round((20 * 48000) / hopSize)
const contextFrames = Math.round((2 * 48000) / hopSize)

export class WpeDereverb {
  private readonly dft = new Dft(fftSize)
  private readonly window = new Float64Array(fftSize)

  constructor() {
    for (let i = 0; i < fftSize; i += 1) {
      this.window[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / fftSize)
    }
  }

  process(input: Float32Array, options: WpeOptions): WpeResult {
    const fitBins = Math.min(bins, Math.round((fitCeilingHz / (options.sampleRate / 2)) * (fftSize / 2)) + 1)
    const totalFrames = Math.max(1, Math.ceil((input.length + fftSize) / hopSize))
    const output = new Float32Array(fftSize + totalFrames * hopSize)
    const reductionDb = new Float32Array(totalFrames)

    for (let blockStart = 0; blockStart < totalFrames; blockStart += blockFrames) {
      if (options.shouldCancel?.()) throw new Error('cancelled')
      const emitFrom = blockStart
      const emitTo = Math.min(totalFrames, blockStart + blockFrames)
      // Context frames are fitted with the block but not emitted, so the filter either
      // side of a boundary is estimated from nearly the same data.
      const from = Math.max(0, emitFrom - contextFrames)
      const to = Math.min(totalFrames, emitTo + contextFrames)
      this.processBlock(input, output, reductionDb, from, to, emitFrom, emitTo, fitBins)
      options.onProgress?.(emitTo, totalFrames)
    }

    options.onProgress?.(totalFrames, totalFrames)
    return {
      residual: output.subarray(fftSize, fftSize + input.length),
      reductionDb,
      hopSize,
    }
  }

  private processBlock(
    input: Float32Array,
    output: Float32Array,
    reductionDb: Float32Array,
    from: number,
    to: number,
    emitFrom: number,
    emitTo: number,
    fitBins: number,
  ): void {
    const count = to - from
    // Spectra for the block, as interleaved re/im per (frame, bin).
    const observed = new Float64Array(count * bins * 2)
    const real = new Float64Array(fftSize)
    const imag = new Float64Array(fftSize)

    for (let f = 0; f < count; f += 1) {
      const start = (from + f + 1) * hopSize - fftSize
      for (let i = 0; i < fftSize; i += 1) {
        const index = start + i
        const sample = index >= 0 && index < input.length ? input[index] : 0
        real[i] = Number.isFinite(sample) ? sample * this.window[i] : 0
        imag[i] = 0
      }
      this.dft.forward(real, imag)
      for (let bin = 0; bin < bins; bin += 1) {
        const at = (f * bins + bin) * 2
        observed[at] = real[bin]
        observed[at + 1] = imag[bin]
      }
    }

    const estimate = observed.slice()
    const power = new Float64Array(count)
    const matrix = new Float64Array(taps * taps * 2)
    const vector = new Float64Array(taps * 2)
    const gains = new Float64Array(taps * 2)

    for (let pass = 0; pass < iterations; pass += 1) {
      for (let bin = 0; bin < fitBins; bin += 1) {
        // Weight each frame by the inverse of the current speech power estimate, lightly
        // smoothed across time; a raw per-frame estimate is too spiky to fit against.
        for (let f = 0; f < count; f += 1) {
          const at = (f * bins + bin) * 2
          power[f] = estimate[at] * estimate[at] + estimate[at + 1] * estimate[at + 1]
        }
        smoothInPlace(power)

        matrix.fill(0)
        vector.fill(0)
        for (let f = delayFrames + taps; f < count; f += 1) {
          const weight = 1 / Math.max(power[f], 1e-12)
          const yAt = (f * bins + bin) * 2
          const yRe = observed[yAt]
          const yIm = observed[yAt + 1]
          for (let row = 0; row < taps; row += 1) {
            const rowAt = ((f - delayFrames - row) * bins + bin) * 2
            const xrRe = observed[rowAt]
            const xrIm = observed[rowAt + 1]
            const wRe = xrRe * weight
            const wIm = xrIm * weight
            // R[row][col] += w(x_row) * conj(x_col)
            for (let column = row; column < taps; column += 1) {
              const colAt = ((f - delayFrames - column) * bins + bin) * 2
              const xcRe = observed[colAt]
              const xcIm = observed[colAt + 1]
              const cell = (row * taps + column) * 2
              matrix[cell] += wRe * xcRe + wIm * xcIm
              matrix[cell + 1] += wIm * xcRe - wRe * xcIm
            }
            // r[row] += w(x_row) * conj(y)
            vector[row * 2] += wRe * yRe + wIm * yIm
            vector[row * 2 + 1] += wIm * yRe - wRe * yIm
          }
        }
        // Only the upper triangle was accumulated; R is Hermitian, so mirror it.
        for (let row = 1; row < taps; row += 1) {
          for (let column = 0; column < row; column += 1) {
            const source = (column * taps + row) * 2
            const target = (row * taps + column) * 2
            matrix[target] = matrix[source]
            matrix[target + 1] = -matrix[source + 1]
          }
        }

        let trace = 0
        for (let row = 0; row < taps; row += 1) trace += matrix[(row * taps + row) * 2]
        const ridge = (1e-6 * trace) / taps + 1e-12
        for (let row = 0; row < taps; row += 1) matrix[(row * taps + row) * 2] += ridge

        if (!solveHermitian(matrix, vector, gains)) continue

        for (let f = 0; f < count; f += 1) {
          const at = (f * bins + bin) * 2
          let predRe = 0
          let predIm = 0
          if (f >= delayFrames + taps) {
            for (let row = 0; row < taps; row += 1) {
              const rowAt = ((f - delayFrames - row) * bins + bin) * 2
              const gRe = gains[row * 2]
              const gIm = gains[row * 2 + 1]
              const xRe = observed[rowAt]
              const xIm = observed[rowAt + 1]
              // conj(g) * x
              predRe += gRe * xRe + gIm * xIm
              predIm += gRe * xIm - gIm * xRe
            }
          }
          estimate[at] = observed[at] - predRe
          estimate[at + 1] = observed[at + 1] - predIm
        }
      }
    }

    for (let f = 0; f < count; f += 1) {
      const frame = from + f
      if (frame < emitFrom || frame >= emitTo) continue
      let before = 1e-12
      let after = 1e-12
      for (let bin = 0; bin < bins; bin += 1) {
        const at = (f * bins + bin) * 2
        const eRe = estimate[at]
        const eIm = estimate[at + 1]
        real[bin] = eRe
        imag[bin] = eIm
        before += observed[at] * observed[at] + observed[at + 1] * observed[at + 1]
        after += eRe * eRe + eIm * eIm
      }
      for (let bin = bins; bin < fftSize; bin += 1) {
        const mirror = fftSize - bin
        real[bin] = real[mirror]
        imag[bin] = -imag[mirror]
      }
      this.dft.inverse(real, imag)

      const scale = overlapScale / fftSize
      const base = (frame + 1) * hopSize - fftSize + fftSize
      for (let i = 0; i < fftSize; i += 1) {
        output[base + i] += real[i] * this.window[i] * scale
      }
      reductionDb[frame] = Math.min(0, 10 * Math.log10(after / before))
    }
  }
}

/** Three-tap smoothing of the power envelope, in place. */
function smoothInPlace(values: Float64Array): void {
  let previous = values[0]
  for (let i = 0; i < values.length; i += 1) {
    const next = i + 1 < values.length ? values[i + 1] : values[i]
    const current = values[i]
    values[i] = 0.25 * previous + 0.5 * current + 0.25 * next
    previous = current
  }
}

/**
 * Solves the complex system `matrix * result = vector` by Gaussian elimination with partial
 * pivoting. Returns false if the system is singular, in which case the caller leaves the
 * band alone rather than emitting whatever a division by zero produced.
 */
function solveHermitian(matrix: Float64Array, vector: Float64Array, result: Float64Array): boolean {
  const size = taps
  const a = matrix.slice()
  const b = vector.slice()

  for (let column = 0; column < size; column += 1) {
    let pivot = column
    let best = 0
    for (let row = column; row < size; row += 1) {
      const at = (row * size + column) * 2
      const magnitude = Math.hypot(a[at], a[at + 1])
      if (magnitude > best) {
        best = magnitude
        pivot = row
      }
    }
    if (!(best > 0) || !Number.isFinite(best)) return false

    if (pivot !== column) {
      for (let k = 0; k < size; k += 1) {
        const from = (pivot * size + k) * 2
        const to = (column * size + k) * 2
        const re = a[from]
        const im = a[from + 1]
        a[from] = a[to]
        a[from + 1] = a[to + 1]
        a[to] = re
        a[to + 1] = im
      }
      const re = b[pivot * 2]
      const im = b[pivot * 2 + 1]
      b[pivot * 2] = b[column * 2]
      b[pivot * 2 + 1] = b[column * 2 + 1]
      b[column * 2] = re
      b[column * 2 + 1] = im
    }

    const dAt = (column * size + column) * 2
    const dRe = a[dAt]
    const dIm = a[dAt + 1]
    const dMag = dRe * dRe + dIm * dIm
    for (let row = column + 1; row < size; row += 1) {
      const lAt = (row * size + column) * 2
      const nRe = a[lAt]
      const nIm = a[lAt + 1]
      const fRe = (nRe * dRe + nIm * dIm) / dMag
      const fIm = (nIm * dRe - nRe * dIm) / dMag
      if (fRe === 0 && fIm === 0) continue
      for (let k = column; k < size; k += 1) {
        const src = (column * size + k) * 2
        const dst = (row * size + k) * 2
        a[dst] -= fRe * a[src] - fIm * a[src + 1]
        a[dst + 1] -= fRe * a[src + 1] + fIm * a[src]
      }
      b[row * 2] -= fRe * b[column * 2] - fIm * b[column * 2 + 1]
      b[row * 2 + 1] -= fRe * b[column * 2 + 1] + fIm * b[column * 2]
    }
  }

  for (let row = size - 1; row >= 0; row -= 1) {
    let sumRe = b[row * 2]
    let sumIm = b[row * 2 + 1]
    for (let column = row + 1; column < size; column += 1) {
      const at = (row * size + column) * 2
      const xRe = result[column * 2]
      const xIm = result[column * 2 + 1]
      sumRe -= a[at] * xRe - a[at + 1] * xIm
      sumIm -= a[at] * xIm + a[at + 1] * xRe
    }
    const dAt = (row * size + row) * 2
    const dRe = a[dAt]
    const dIm = a[dAt + 1]
    const dMag = dRe * dRe + dIm * dIm
    if (!(dMag > 0)) return false
    result[row * 2] = (sumRe * dRe + sumIm * dIm) / dMag
    result[row * 2 + 1] = (sumIm * dRe - sumRe * dIm) / dMag
  }

  for (let i = 0; i < size * 2; i += 1) {
    if (!Number.isFinite(result[i])) return false
  }
  return true
}
