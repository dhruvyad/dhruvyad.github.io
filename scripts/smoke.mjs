/**
 * Loads every built page in headless Chrome and asserts the things that would
 * silently break.
 *
 * This matters more than usual here: distill's template is unmaintained, its
 * components render into shadow DOM (so a broken page still looks fine to curl),
 * and the interesting parts — math, citations, lazily-mounted figures — all
 * happen at runtime. A build that "succeeds" tells you almost nothing.
 *
 * Usage: node scripts/smoke.mjs [baseUrl]   (default http://localhost:4173/)
 * Assumes `npm run build` has run and a server is serving dist/.
 */
import { spawn } from 'node:child_process'
import { existsSync, readdirSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const baseUrl = (process.argv[2] ?? 'http://localhost:4173/').replace(/\/?$/, '/')
const DEBUG_PORT = 9333

function findChrome() {
  const candidates = [
    process.env.CHROME_PATH,
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Chromium.app/Contents/MacOS/Chromium',
    '/usr/bin/google-chrome',
    '/usr/bin/google-chrome-stable',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
  ].filter(Boolean)
  const found = candidates.find((p) => existsSync(p))
  if (!found) throw new Error('No Chrome found. Set CHROME_PATH.')
  return found
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

async function waitForPort(url, attempts = 60) {
  for (let i = 0; i < attempts; i++) {
    try {
      const res = await fetch(url)
      if (res.ok) return await res.json()
    } catch {}
    await sleep(250)
  }
  throw new Error(`Timed out waiting for ${url}`)
}

/** Minimal CDP client over Node's built-in WebSocket. */
class CDP {
  constructor(ws) {
    this.ws = ws
    this.id = 0
    this.pending = new Map()
    this.listeners = []
    ws.addEventListener('message', (event) => {
      const msg = JSON.parse(event.data)
      if (msg.id && this.pending.has(msg.id)) {
        const { resolve: res, reject } = this.pending.get(msg.id)
        this.pending.delete(msg.id)
        msg.error ? reject(new Error(JSON.stringify(msg.error))) : res(msg.result)
      } else if (msg.method) {
        for (const fn of this.listeners) fn(msg)
      }
    })
  }

  static async connect(wsUrl) {
    const ws = new WebSocket(wsUrl)
    await new Promise((res, rej) => {
      ws.addEventListener('open', res, { once: true })
      ws.addEventListener('error', rej, { once: true })
    })
    return new CDP(ws)
  }

  on(fn) {
    this.listeners.push(fn)
  }

  send(method, params = {}, sessionId) {
    const id = ++this.id
    return new Promise((res, rej) => {
      this.pending.set(id, { resolve: res, reject: rej })
      this.ws.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }))
    })
  }
}

/** Runs in the page. Returns a report of what actually rendered. */
const PAGE_CHECKS = `(() => {
  const deep = (sel) => {
    const els = [...document.querySelectorAll(sel)];
    return els.map((el) => ({ el, shadow: el.shadowRoot }));
  };
  const mathEls = [...document.querySelectorAll('d-math')];
  const mathRendered = mathEls.filter(
    (m) => m.shadowRoot && m.shadowRoot.querySelector('.katex')
  ).length;
  const mathErrors = mathEls.filter(
    (m) => m.shadowRoot && /KaTeX|ParseError/i.test(m.shadowRoot.textContent || '')
      && !m.shadowRoot.querySelector('.katex')
  ).length;

  const cites = [...document.querySelectorAll('d-cite')];
  // A resolved citation renders a bracketed number in its shadow root; an
  // unresolved key renders as '?'.
  const citeText = cites.map((c) => (c.shadowRoot ? c.shadowRoot.textContent.trim() : ''));
  const citesResolved = citeText.filter((t) => /\\d/.test(t)).length;
  const citesUnresolved = citeText.filter((t) => t.includes('?')).length;

  // d-citation-list and d-footnote-list both render into their *light* DOM
  // (unlike d-math and d-cite, which use shadow roots), so count both places.
  const countItems = (el) => {
    if (!el) return 0;
    const light = el.querySelectorAll('li').length;
    const shadow = el.shadowRoot ? el.shadowRoot.querySelectorAll('li').length : 0;
    return light + shadow;
  };

  const citationEntries = countItems(document.querySelector('d-citation-list'));
  const footnotes = document.querySelectorAll('d-footnote').length;
  const footnoteEntries = countItems(document.querySelector('d-footnote-list'));

  const figures = [...document.querySelectorAll('d-figure')].map((f) => ({
    id: f.id,
    mounted: !!f.querySelector('svg, canvas'),
  }));

  const title = document.querySelector('d-title h1');
  const byline = document.querySelector('d-byline');

  return {
    mathTotal: mathEls.length,
    mathRendered,
    mathErrors,
    citesTotal: cites.length,
    citesResolved,
    citesUnresolved,
    citationEntries,
    footnotes,
    footnoteEntries,
    figures,
    hasTitle: !!title && title.textContent.trim().length > 0,
    bylineText: byline ? byline.textContent.replace(/\\s+/g, ' ').trim().slice(0, 120) : '',
    postLinks: [...document.querySelectorAll('#post-list a')].map((a) => a.getAttribute('href')),
    toolLinks: [...document.querySelectorAll('#tool-list a')].map((a) => a.getAttribute('href')),
    toolMounted: !!document.querySelector('#app')?.childElementCount,
    stages: document.querySelectorAll('.stage canvas').length,
    triangles: Math.max(
      0,
      ...[...document.querySelectorAll('[data-triangles]')].map((e) => +e.dataset.triangles || 0),
    ),
    blocks: Math.max(
      0,
      ...[...document.querySelectorAll('[data-blocks]')].map((e) => +e.dataset.blocks || 0),
    ),
  };
})()`

function slugsIn(kind) {
  const dir = resolve(root, kind)
  if (!existsSync(dir)) return []
  return readdirSync(dir, { withFileTypes: true })
    .filter((e) => e.isDirectory() && existsSync(resolve(dir, e.name, 'index.html')))
    .map((e) => e.name)
    .sort()
}

const chrome = spawn(
  findChrome(),
  [
    '--headless=new',
    `--remote-debugging-port=${DEBUG_PORT}`,
    '--no-first-run',
    '--no-default-browser-check',
    // Software WebGL, so the 3D figures actually render headlessly. Without
    // these the canvas exists but three.js can't get a context, and the figure
    // silently falls back.
    '--use-angle=swiftshader',
    '--enable-unsafe-swiftshader',
    '--hide-scrollbars',
    '--window-size=1280,900',
    // CI runners lack the namespaces Chrome's sandbox needs.
    ...(process.env.CI ? ['--no-sandbox', '--disable-dev-shm-usage'] : []),
    '--user-data-dir=' + resolve(root, 'node_modules/.cache/smoke-profile'),
    'about:blank',
  ],
  { stdio: 'ignore' },
)

let failures = []
const note = (page, msg) => failures.push(`${page}: ${msg}`)

try {
  const version = await waitForPort(`http://127.0.0.1:${DEBUG_PORT}/json/version`)
  const cdp = await CDP.connect(version.webSocketDebuggerUrl)

  const pages = [
    { name: 'index', url: baseUrl, kind: 'index' },
    ...slugsIn('posts').map((slug) => ({
      name: `posts/${slug}`,
      url: `${baseUrl}posts/${slug}/`,
      kind: 'post',
    })),
    ...slugsIn('tools').map((slug) => ({
      name: `tools/${slug}`,
      url: `${baseUrl}tools/${slug}/`,
      kind: 'tool',
    })),
  ]

  for (const page of pages) {
    const { targetId } = await cdp.send('Target.createTarget', { url: 'about:blank' })
    const { sessionId } = await cdp.send('Target.attachToTarget', { targetId, flatten: true })

    const consoleErrors = []
    const networkFailures = []
    const listener = (msg) => {
      if (msg.sessionId !== sessionId) return
      if (msg.method === 'Runtime.exceptionThrown') {
        const d = msg.params.exceptionDetails
        consoleErrors.push(d.exception?.description ?? d.text)
      }
      if (msg.method === 'Runtime.consoleAPICalled' && msg.params.type === 'error') {
        consoleErrors.push(msg.params.args.map((a) => a.value ?? a.description ?? '').join(' '))
      }
      if (msg.method === 'Log.entryAdded' && msg.params.entry.level === 'error') {
        const e = msg.params.entry
        ;(e.source === 'network' ? networkFailures : consoleErrors).push(`${e.text} ${e.url ?? ''}`)
      }
    }
    cdp.on(listener)

    await cdp.send('Runtime.enable', {}, sessionId)
    await cdp.send('Log.enable', {}, sessionId)
    await cdp.send('Page.enable', {}, sessionId)
    await cdp.send('Page.navigate', { url: page.url }, sessionId)
    await sleep(1200)

    // Walk down the page so every lazily-mounted figure passes through
    // d-figure's trigger zone. Jumping straight to the bottom does NOT work:
    // the observer only fires within ~2 viewport heights, so on a long article
    // every figure in the middle would be skipped and silently report as
    // unmounted.
    const { result: pageHeight } = await cdp.send(
      'Runtime.evaluate',
      { expression: `document.body.scrollHeight`, returnByValue: true },
      sessionId,
    )
    const step = 600
    for (let y = 0; y < pageHeight.value + step; y += step) {
      await cdp.send('Runtime.evaluate', { expression: `window.scrollTo(0, ${y})` }, sessionId)
      await sleep(90)
    }
    await sleep(1200)

    const { result } = await cdp.send(
      'Runtime.evaluate',
      { expression: PAGE_CHECKS, returnByValue: true, awaitPromise: false },
      sessionId,
    )
    const r = result.value

    // --- assertions ---
    for (const err of consoleErrors) note(page.name, `console error: ${err.slice(0, 200)}`)
    for (const err of networkFailures) note(page.name, `failed request: ${err.slice(0, 200)}`)

    if (page.kind === 'post') {
      if (!r.hasTitle) note(page.name, 'no rendered <d-title> heading')
      if (r.mathTotal === 0) note(page.name, 'no <d-math> elements found')
      if (r.mathRendered < r.mathTotal)
        note(page.name, `only ${r.mathRendered}/${r.mathTotal} <d-math> rendered by KaTeX`)
      if (r.mathErrors > 0) note(page.name, `${r.mathErrors} <d-math> reported a KaTeX error`)
      if (r.citesTotal > 0 && r.citesResolved < r.citesTotal)
        note(page.name, `only ${r.citesResolved}/${r.citesTotal} <d-cite> resolved to a number`)
      if (r.citesUnresolved > 0)
        note(page.name, `${r.citesUnresolved} <d-cite> unresolved (missing .bib key?)`)
      if (r.citesTotal > 0 && r.citationEntries === 0)
        note(page.name, 'citation list rendered empty (bibliography.bib not loaded?)')
      if (r.footnotes > 0 && r.footnoteEntries === 0) note(page.name, 'footnote list rendered empty')
      if (r.figures.length === 0) note(page.name, 'no <d-figure> elements found')
      for (const fig of r.figures) {
        if (!fig.mounted) note(page.name, `figure #${fig.id} never mounted a visualisation`)
      }
      console.log(
        `  ${page.name}\n` +
          `    math ${r.mathRendered}/${r.mathTotal} · cites ${r.citesResolved}/${r.citesTotal} · ` +
          `refs ${r.citationEntries} · footnotes ${r.footnoteEntries}/${r.footnotes} · ` +
          `figures ${r.figures.filter((f) => f.mounted).length}/${r.figures.length}`,
      )
    } else if (page.kind === 'tool') {
      if (!r.toolMounted) note(page.name, 'tool app rendered nothing')
      if (r.stages === 0) note(page.name, 'no 3D stage on the page')
      if (r.triangles === 0) note(page.name, '3D view drew no geometry')
      console.log(
        `  ${page.name}\n    ${r.stages} 3D stage(s) · ${r.triangles.toLocaleString()} triangles · ` +
          `${r.blocks.toLocaleString()} blocks`,
      )
    } else {
      if (r.postLinks.length === 0) note(page.name, 'post list is empty')
      if (r.toolLinks.length === 0) note(page.name, 'tool list is empty')
      console.log(
        `  ${page.name}\n    ${r.postLinks.length} post link(s), ${r.toolLinks.length} tool link(s)`,
      )
    }

    cdp.listeners = cdp.listeners.filter((l) => l !== listener)
    await cdp.send('Target.closeTarget', { targetId })
  }
} finally {
  chrome.kill()
}

if (failures.length) {
  console.error(`\n✗ ${failures.length} problem(s):`)
  for (const f of failures) console.error(`  - ${f}`)
  process.exit(1)
}
console.log('\n✓ smoke test passed')
