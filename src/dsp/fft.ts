// The model's STFT is 960 points, which is 2^6 * 3 * 5 and so out of reach of a plain
// radix-2 FFT. The C++ reference gets a mixed-radix transform for free from Accelerate;
// in the browser the cheapest equivalent is Bluestein's algorithm, which expresses the
// 960-point DFT as a 2048-point cyclic convolution. Six power-of-two transforms per audio
// frame is negligible beside a 493-node inference, and it stays exact for any size.

/** In-place, unnormalised radix-2 Cooley-Tukey FFT over split real/imaginary arrays. */
class PowerOfTwoFft {
  private readonly size: number
  private readonly reversed: Uint32Array
  private readonly cos: Float64Array
  private readonly sin: Float64Array

  constructor(size: number) {
    if (size < 2 || (size & (size - 1)) !== 0) throw new Error(`FFT size ${size} is not a power of two`)
    this.size = size
    const bits = Math.log2(size)
    this.reversed = new Uint32Array(size)
    for (let i = 0; i < size; i += 1) {
      let value = 0
      for (let bit = 0; bit < bits; bit += 1) value |= ((i >>> bit) & 1) << (bits - 1 - bit)
      this.reversed[i] = value
    }
    // Half a turn of twiddles is enough: stage k reads every (size / span)-th entry.
    this.cos = new Float64Array(size / 2)
    this.sin = new Float64Array(size / 2)
    for (let i = 0; i < size / 2; i += 1) {
      const angle = (-2 * Math.PI * i) / size
      this.cos[i] = Math.cos(angle)
      this.sin[i] = Math.sin(angle)
    }
  }

  transform(re: Float64Array, im: Float64Array): void {
    const { size, reversed, cos, sin } = this
    for (let i = 0; i < size; i += 1) {
      const j = reversed[i]
      if (j > i) {
        const tr = re[i]
        re[i] = re[j]
        re[j] = tr
        const ti = im[i]
        im[i] = im[j]
        im[j] = ti
      }
    }
    for (let span = 1; span < size; span <<= 1) {
      const step = size / (span << 1)
      for (let start = 0; start < size; start += span << 1) {
        for (let offset = 0, twiddle = 0; offset < span; offset += 1, twiddle += step) {
          const a = start + offset
          const b = a + span
          const wr = cos[twiddle]
          const wi = sin[twiddle]
          const xr = re[b] * wr - im[b] * wi
          const xi = re[b] * wi + im[b] * wr
          re[b] = re[a] - xr
          im[b] = im[a] - xi
          re[a] += xr
          im[a] += xi
        }
      }
    }
  }
}

/** Unnormalised complex DFT of arbitrary size, in place over split real/imaginary arrays. */
export class Dft {
  readonly size: number
  private readonly convolutionSize: number
  private readonly fft: PowerOfTwoFft
  private readonly chirpRe: Float64Array
  private readonly chirpIm: Float64Array
  private readonly kernelRe: Float64Array
  private readonly kernelIm: Float64Array
  private readonly workRe: Float64Array
  private readonly workIm: Float64Array

  constructor(size: number) {
    this.size = size
    let convolutionSize = 1
    while (convolutionSize < 2 * size - 1) convolutionSize <<= 1
    this.convolutionSize = convolutionSize
    this.fft = new PowerOfTwoFft(convolutionSize)

    // exp(-i*pi*n^2/size). Reducing n^2 modulo 2*size before the multiply keeps the angle
    // small; at n = 959 the raw argument is already ~2.9e6 radians, where a double has
    // shed most of its fractional precision and the twiddles visibly drift.
    this.chirpRe = new Float64Array(size)
    this.chirpIm = new Float64Array(size)
    for (let n = 0; n < size; n += 1) {
      const angle = (-Math.PI * ((n * n) % (2 * size))) / size
      this.chirpRe[n] = Math.cos(angle)
      this.chirpIm[n] = Math.sin(angle)
    }

    this.kernelRe = new Float64Array(convolutionSize)
    this.kernelIm = new Float64Array(convolutionSize)
    this.kernelRe[0] = this.chirpRe[0]
    this.kernelIm[0] = -this.chirpIm[0]
    for (let n = 1; n < size; n += 1) {
      this.kernelRe[n] = this.chirpRe[n]
      this.kernelIm[n] = -this.chirpIm[n]
      this.kernelRe[convolutionSize - n] = this.chirpRe[n]
      this.kernelIm[convolutionSize - n] = -this.chirpIm[n]
    }
    this.fft.transform(this.kernelRe, this.kernelIm)

    this.workRe = new Float64Array(convolutionSize)
    this.workIm = new Float64Array(convolutionSize)
  }

  forward(re: Float64Array, im: Float64Array): void {
    const { size, convolutionSize, chirpRe, chirpIm, kernelRe, kernelIm, workRe, workIm } = this
    for (let n = 0; n < size; n += 1) {
      workRe[n] = re[n] * chirpRe[n] - im[n] * chirpIm[n]
      workIm[n] = re[n] * chirpIm[n] + im[n] * chirpRe[n]
    }
    workRe.fill(0, size)
    workIm.fill(0, size)
    this.fft.transform(workRe, workIm)
    for (let i = 0; i < convolutionSize; i += 1) {
      const ar = workRe[i]
      const ai = workIm[i]
      workRe[i] = ar * kernelRe[i] - ai * kernelIm[i]
      workIm[i] = ar * kernelIm[i] + ai * kernelRe[i]
    }
    // An inverse transform via conjugation, folding the 1/convolutionSize into the chirp.
    for (let i = 0; i < convolutionSize; i += 1) workIm[i] = -workIm[i]
    this.fft.transform(workRe, workIm)
    const scale = 1 / convolutionSize
    for (let k = 0; k < size; k += 1) {
      const cr = workRe[k] * scale
      const ci = -workIm[k] * scale
      re[k] = cr * chirpRe[k] - ci * chirpIm[k]
      im[k] = cr * chirpIm[k] + ci * chirpRe[k]
    }
  }

  /** Unnormalised inverse: the caller divides by `size`, matching the C++ reference. */
  inverse(re: Float64Array, im: Float64Array): void {
    for (let i = 0; i < this.size; i += 1) im[i] = -im[i]
    this.forward(re, im)
    for (let i = 0; i < this.size; i += 1) im[i] = -im[i]
  }
}
