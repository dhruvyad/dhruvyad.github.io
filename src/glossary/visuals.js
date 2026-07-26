/**
 * Tiny inline SVGs for glossary tooltips. Each returns an HTML string, so a
 * definition can show the thing rather than only describe it.
 */

const RED = '#c0392b'
const GREY = '#d8d8d8'
const INK = 'rgba(0,0,0,0.55)'

/** Labelled horizontal bars, for comparing magnitudes. */
export function bars(rows, { width = 250, unit = '' } = {}) {
  const max = Math.max(...rows.map((r) => r.value))
  const h = 17
  const labelW = 92
  // Leave room for the longest value label, e.g. "8.0 TB/s", or it gets clipped.
  const barW = width - labelW - 60
  const body = rows
    .map((r, i) => {
      const w = Math.max(1, (r.value / max) * barW)
      const y = i * h
      return `
      <text x="0" y="${y + 11}" font-size="9.5" fill="${INK}">${r.label}</text>
      <rect x="${labelW}" y="${y + 3}" width="${w}" height="9" fill="${r.hi ? RED : GREY}"/>
      <text x="${labelW + w + 5}" y="${y + 11}" font-size="9.5" fill="${INK}"
        font-family="ui-monospace,Menlo,monospace">${r.display ?? r.value}${unit}</text>`
    })
    .join('')
  return `<svg viewBox="0 0 ${width} ${rows.length * h}" width="${width}">${body}</svg>`
}

/** The expert-sparsity motif: a grid with `lit` of `total` cells active. */
export function grid(lit, total = 256, { cols = 32, cell = 6, gap = 1.4 } = {}) {
  const rows = Math.ceil(total / cols)
  const cells = Array.from({ length: total }, (_, i) => {
    const x = (i % cols) * (cell + gap)
    const y = Math.floor(i / cols) * (cell + gap)
    return `<rect x="${x}" y="${y}" width="${cell}" height="${cell}" fill="${i < lit ? RED : GREY}"/>`
  }).join('')
  const w = cols * (cell + gap)
  return `<svg viewBox="0 0 ${w} ${rows * (cell + gap)}" width="${Math.min(250, w)}">${cells}</svg>`
}

/** A memory hierarchy, widest and slowest at the bottom. */
export function hierarchy(levels) {
  const w = 250
  const h = 19
  const body = levels
    .map((l, i) => {
      const bw = 60 + (i / (levels.length - 1)) * 120
      return `
      <rect x="0" y="${i * h + 2}" width="${bw}" height="${h - 5}" fill="${l.hi ? RED : GREY}"/>
      <text x="${bw + 6}" y="${i * h + 12}" font-size="9.5" fill="${INK}">${l.label}</text>`
    })
    .join('')
  return `<svg viewBox="0 0 ${w} ${levels.length * h + 2}" width="${w}">${body}</svg>`
}

/** A miniature roofline, with an optional operating point. */
export function roofline({ point = null } = {}) {
  const w = 250
  const h = 92
  const pad = 22
  // log-x from 0.5 to 512 FLOP/byte, ridge at 206
  const X = (ai) => pad + ((Math.log2(ai) - Math.log2(0.5)) / (Math.log2(512) - Math.log2(0.5))) * (w - pad - 10)
  const Y = (v) => h - 16 - v * (h - 30)
  const ridge = 206
  const path = `M${X(0.5)},${Y(0.5 / ridge)} L${X(ridge)},${Y(1)} L${X(512)},${Y(1)}`
  const dot = point
    ? `<circle cx="${X(point.ai)}" cy="${Y(Math.min(1, point.ai / ridge))}" r="3.5" fill="${RED}"/>
       <text x="${X(point.ai) + 6}" y="${Y(Math.min(1, point.ai / ridge)) + 3}" font-size="9" fill="${RED}">${point.label}</text>`
    : ''
  return `<svg viewBox="0 0 ${w} ${h}" width="${w}">
    <line x1="${pad}" y1="${h - 16}" x2="${w - 8}" y2="${h - 16}" stroke="rgba(0,0,0,.25)"/>
    <line x1="${pad}" y1="14" x2="${pad}" y2="${h - 16}" stroke="rgba(0,0,0,.25)"/>
    <path d="${path}" fill="none" stroke="rgba(0,0,0,.55)" stroke-width="1.6"/>
    <text x="${X(ridge)}" y="10" font-size="8.5" fill="${INK}" text-anchor="middle">ridge · 206</text>
    <text x="${w - 8}" y="${h - 5}" font-size="8.5" fill="${INK}" text-anchor="end">FLOP/byte →</text>
    ${dot}
  </svg>`
}

/** Draft/verify strip for speculative decoding. */
export function draftStrip(k = 5, accepted = 4) {
  const w = 250
  const cw = 26
  const cells = Array.from({ length: k + 1 }, (_, i) => {
    const fill = i === 0 ? '#7f8c8d' : i <= accepted ? RED : GREY
    const label = i === 0 ? 'v' : i <= accepted ? '✓' : '✗'
    return `<rect x="${i * (cw + 3)}" y="0" width="${cw}" height="18" fill="${fill}"/>
      <text x="${i * (cw + 3) + cw / 2}" y="13" font-size="10" fill="#fff" text-anchor="middle">${label}</text>`
  }).join('')
  return `<svg viewBox="0 0 ${w} 32" width="${w}">${cells}
    <text x="0" y="30" font-size="9" fill="${INK}">1 verify pass · ${accepted} of ${k} drafts kept</text>
  </svg>`
}
