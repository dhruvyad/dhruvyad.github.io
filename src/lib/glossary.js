/**
 * Glossary tooltips.
 *
 * Design goal: define a term once, and have every single mention of it anywhere in
 * an article carry the tooltip — without hand-wrapping anything in the markup.
 *
 * So this walks the rendered text and wraps matches itself, and all those matches
 * share exactly one tooltip element and one definition. Adding a term to the
 * glossary annotates every existing occurrence of it across every article; nobody
 * has to remember to tag the seventh mention of HBM.
 *
 * Tooltip bodies are HTML, so a definition can carry a diagram, a bar chart or an
 * image rather than only prose.
 */

const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

/** Containers whose text must never be rewritten. */
const SKIP = new Set(['CODE', 'PRE', 'SCRIPT', 'STYLE', 'TEXTAREA', 'SVG', 'D-MATH', 'D-CODE'])
const SKIP_CLASS = 'gl-tip'

function buildMatcher(entries) {
  /** lowercased pattern -> { entry, canonical } */
  const lookup = new Map()
  const patterns = []
  for (const entry of entries) {
    for (const p of entry.terms) {
      lookup.set(p.toLowerCase(), { entry, canonical: p })
      patterns.push(p)
    }
  }
  // Longest first so "FlashAttention-4" beats "FlashAttention" and "SMEM" beats "SM".
  patterns.sort((a, b) => b.length - a.length)
  const re = new RegExp(`(?<![\\w-])(${patterns.map(escapeRe).join('|')})(?![\\w-])`, 'gi')
  return { re, lookup }
}

function shouldSkip(node) {
  for (let el = node.parentElement; el; el = el.parentElement) {
    // SVG elements report a lower-case tagName, so normalise before comparing.
    if (SKIP.has(el.tagName.toUpperCase())) return true
    if (el.classList?.contains(SKIP_CLASS)) return true
    if (el.dataset?.glTerm) return true // already annotated
    if (el.tagName.toUpperCase() === 'A') return true // don't nest interactive things
  }
  return false
}

/** Wraps every glossary hit inside `root` in a <span data-gl-term="id">. */
function annotate(root, { re, lookup }) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      if (!node.nodeValue || !node.nodeValue.trim()) return NodeFilter.FILTER_REJECT
      if (shouldSkip(node)) return NodeFilter.FILTER_REJECT
      return NodeFilter.FILTER_ACCEPT
    },
  })

  const targets = []
  for (let n = walker.nextNode(); n; n = walker.nextNode()) targets.push(n)

  let count = 0
  for (const node of targets) {
    const text = node.nodeValue
    re.lastIndex = 0
    if (!re.test(text)) continue
    re.lastIndex = 0

    const frag = document.createDocumentFragment()
    let last = 0
    let m
    while ((m = re.exec(text)) !== null) {
      const hit = lookup.get(m[0].toLowerCase())
      if (!hit) continue
      // Acronyms are case-sensitive: "SM" is a unit, "sm" is not.
      if (!hit.entry.ci && m[0] !== hit.canonical) continue

      if (m.index > last) frag.appendChild(document.createTextNode(text.slice(last, m.index)))
      const span = document.createElement('span')
      span.className = 'gl-term'
      span.dataset.glTerm = hit.entry.id
      span.tabIndex = 0
      span.setAttribute('role', 'button')
      span.setAttribute('aria-label', `${m[0]} — glossary`)
      span.textContent = m[0]
      frag.appendChild(span)
      last = m.index + m[0].length
      count++
    }
    if (!last) continue
    if (last < text.length) frag.appendChild(document.createTextNode(text.slice(last)))
    node.parentNode.replaceChild(frag, node)
  }
  return count
}

/** The single shared tooltip, reused by every term on the page. */
function createTooltip() {
  const tip = document.createElement('div')
  tip.className = 'gl-tip'
  tip.setAttribute('role', 'tooltip')
  tip.hidden = true
  document.body.appendChild(tip)
  return tip
}

function render(tip, entry) {
  const links = (entry.links ?? [])
    .map((l) => `<a href="${l.url}" target="_blank" rel="noopener">${l.label} ↗</a>`)
    .join('')
  tip.innerHTML = `
    <div class="gl-head">
      <span class="gl-title">${entry.title}</span>
      ${entry.expand ? `<span class="gl-expand">${entry.expand}</span>` : ''}
    </div>
    ${entry.short ? `<p class="gl-short">${entry.short}</p>` : ''}
    ${entry.visual ? `<div class="gl-visual">${entry.visual}</div>` : ''}
    ${entry.body ? `<div class="gl-body">${entry.body}</div>` : ''}
    ${links ? `<div class="gl-links">${links}</div>` : ''}
  `
}

function place(tip, target) {
  // Measure first, then decide which side has room.
  tip.hidden = false
  tip.style.left = '0px'
  tip.style.top = '0px'
  const r = target.getBoundingClientRect()
  const t = tip.getBoundingClientRect()
  const margin = 8

  let left = r.left + r.width / 2 - t.width / 2
  left = Math.max(margin, Math.min(left, window.innerWidth - t.width - margin))

  const below = r.bottom + 10
  const above = r.top - t.height - 10
  const top = below + t.height + margin < window.innerHeight || above < margin ? below : above

  tip.style.left = `${left + window.scrollX}px`
  tip.style.top = `${top + window.scrollY}px`
  tip.dataset.side = top === below ? 'below' : 'above'
}

/**
 * @param {object}   opts
 * @param {Element[]} opts.roots    subtrees to annotate
 * @param {object[]} opts.terms     glossary entries
 * @param {Element}  [opts.observe] container to watch for late-rendered content
 *                                  (distill builds the footnote and reference
 *                                  lists after this script runs)
 */
export function installGlossary({ roots, terms, observe }) {
  const byId = new Map(terms.map((t) => [t.id, t]))
  const matcher = buildMatcher(terms)
  const tip = createTooltip()

  let current = null
  let hideTimer = null
  let overTip = false

  const show = (target) => {
    clearTimeout(hideTimer)
    const entry = byId.get(target.dataset.glTerm)
    if (!entry) return
    if (current === target && !tip.hidden) return
    current = target
    render(tip, entry)
    place(tip, target)
    tip.classList.add('visible')
    target.setAttribute('aria-describedby', 'gl-tip')
    tip.id = 'gl-tip'
  }

  const hide = (immediate = false) => {
    clearTimeout(hideTimer)
    hideTimer = setTimeout(
      () => {
        if (overTip) return
        tip.classList.remove('visible')
        tip.hidden = true
        current?.removeAttribute('aria-describedby')
        current = null
      },
      immediate ? 0 : 120,
    )
  }

  const termAt = (node) => (node instanceof Element ? node.closest('.gl-term') : null)

  document.addEventListener('mouseover', (e) => {
    const t = termAt(e.target)
    if (t) show(t)
    else if (!tip.contains(e.target)) hide()
  })
  document.addEventListener('focusin', (e) => {
    const t = termAt(e.target)
    if (t) show(t)
  })
  document.addEventListener('focusout', (e) => {
    if (termAt(e.target)) hide()
  })
  // Tapping a term on touch devices, where there is no hover.
  document.addEventListener('click', (e) => {
    const t = termAt(e.target)
    if (t) {
      e.preventDefault()
      current === t && !tip.hidden ? hide(true) : show(t)
    } else if (!tip.contains(e.target)) {
      hide(true)
    }
  })
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') hide(true)
  })
  tip.addEventListener('mouseenter', () => {
    overTip = true
    clearTimeout(hideTimer)
  })
  tip.addEventListener('mouseleave', () => {
    overTip = false
    hide()
  })
  window.addEventListener('scroll', () => current && place(tip, current), { passive: true })

  let total = 0
  for (const root of roots) if (root) total += annotate(root, matcher)

  // distill populates the appendix lists after this module runs, so annotate
  // whatever appears there later.
  if (observe) {
    const obs = new MutationObserver((records) => {
      for (const rec of records) {
        for (const node of rec.addedNodes) {
          if (node.nodeType === 1) annotate(node, matcher)
        }
      }
    })
    obs.observe(observe, { childList: true, subtree: true })
  }

  return { annotated: total }
}
