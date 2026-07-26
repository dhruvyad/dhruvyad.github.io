/**
 * The visual language for this series, in one file.
 *
 * Every figure builds on the same primitives — a number is a coloured cell, a
 * vector is a row of them, a matrix is a grid — so that once you have read one
 * figure you can read all of them. Keeping the language here means restyling it
 * is a single edit rather than a sweep through every component.
 */

/** Signed values: red for positive, blue for negative, pale for near-zero. */
export const POS = '#c0392b'
export const NEG = '#2c7fb8'
export const NEUTRAL = '#e9e7e4'
export const INK = 'rgba(0,0,0,0.78)'
export const MUTED = 'rgba(0,0,0,0.45)'

/** Geometry, shared so matrices line up across figures. */
export const CELL = 30
export const GAP = 3

export const matrixSize = (rows, cols, cell = CELL, gap = GAP) => ({
  width: cols * cell + (cols - 1) * gap,
  height: rows * cell + (rows - 1) * gap,
})

export const cellXY = (i, j, cell = CELL, gap = GAP) => ({
  x: j * (cell + gap),
  y: i * (cell + gap),
})

/**
 * Colour for a signed value. `scale` is the magnitude that saturates, so all the
 * figures agree on how strong a given number looks.
 */
export function signedFill(v, scale = 1) {
  const t = Math.max(0, Math.min(1, Math.abs(v) / scale))
  if (t < 0.02) return NEUTRAL
  return mix(NEUTRAL, v >= 0 ? POS : NEG, 0.15 + 0.85 * t)
}

/** Colour for a value in [0,1], used for probabilities after a softmax. */
export function unitFill(v) {
  return mix('#f4f2f0', POS, Math.max(0, Math.min(1, v)))
}

/** Text that stays readable on whatever the cell turned out to be. */
export const textOn = (v, scale = 1) => (Math.abs(v) / scale > 0.55 ? '#fff' : INK)
export const textOnUnit = (v) => (v > 0.55 ? '#fff' : INK)

function mix(a, b, t) {
  const pa = parse(a)
  const pb = parse(b)
  const c = pa.map((x, i) => Math.round(x + (pb[i] - x) * t))
  return `rgb(${c[0]},${c[1]},${c[2]})`
}

function parse(c) {
  if (c.startsWith('rgb')) return c.match(/\d+/g).map(Number)
  const h = c.replace('#', '')
  const n = h.length === 3 ? h.split('').map((x) => x + x) : h.match(/../g)
  return n.map((x) => parseInt(x, 16))
}

/** Deterministic pseudo-random values, so a figure looks the same every visit. */
export function seeded(seed) {
  let s = (seed * 2654435761) >>> 0
  return () => {
    s ^= s << 13
    s >>>= 0
    s ^= s >>> 17
    s ^= s << 5
    s >>>= 0
    return s / 4294967296
  }
}

/** A vector of values in [-max, max], rounded for legibility. */
export function randomVector(n, seed, max = 1.6) {
  const rnd = seeded(seed)
  return Array.from({ length: n }, () => Math.round((rnd() * 2 - 1) * max * 10) / 10)
}

export function randomMatrix(rows, cols, seed, max = 1.4) {
  const rnd = seeded(seed)
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => Math.round((rnd() * 2 - 1) * max * 10) / 10),
  )
}

export const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0)

export const matmul = (A, B) =>
  A.map((row) => B[0].map((_, j) => dot(row, B.map((r) => r[j]))))

export function softmax(xs) {
  const m = Math.max(...xs)
  const e = xs.map((x) => Math.exp(x - m))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((x) => x / s)
}

export const fmt = (v, d = 1) => (Math.abs(v) < 0.05 ? '0' : v.toFixed(d))
