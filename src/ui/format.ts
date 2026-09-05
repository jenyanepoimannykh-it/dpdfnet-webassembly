export function formatClock(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds))
  const minutes = Math.floor(total / 60)
  const rest = total % 60
  return `${minutes}:${rest.toString().padStart(2, '0')}`
}

export function formatChannels(count: number): string {
  if (count === 1) return 'Mono'
  if (count === 2) return 'Stereo'
  return `${count} channels`
}

export function formatSpeed(audioSeconds: number, elapsedMs: number): string {
  if (elapsedMs <= 0) return ''
  const ratio = audioSeconds / (elapsedMs / 1000)
  return ratio >= 1
    ? `${ratio.toFixed(1)}× faster than real time`
    : `${(1 / ratio).toFixed(1)}× slower than real time`
}
