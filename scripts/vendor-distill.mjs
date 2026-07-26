/**
 * Vendors distill's web-component framework into public/vendor/.
 *
 * Why this exists: distill.pub has been dormant since 2021 and the framework
 * unmaintained since 2022, but template.v2.js hardcodes absolute distill.pub URLs
 * for KaTeX and its polyfills. Loading it from their CDN means every article on
 * this site silently depends on someone else's server staying up. So we fetch it
 * once, rewrite those URLs to resolve relative to the script's own location, and
 * commit the result.
 *
 * Run `npm run vendor-distill` to re-vendor (e.g. to pick up an upstream change).
 * The output is committed, so normal builds never touch the network.
 */
import { createHash } from 'node:crypto'
import { mkdirSync, writeFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const vendorDir = resolve(root, 'public/vendor')

const TEMPLATE_URL = 'https://distill.pub/template.v2.js'
const POLYFILLS = [
  'https://distill.pub/third-party/polyfills/webcomponents-lite.js',
  'https://distill.pub/third-party/polyfills/intersection-observer.js',
]

/**
 * Resolves sibling files against the vendored script's own URL, so the bundle
 * works under any base path with no config — a user site serves from /, a
 * project site from /<repo>/.
 */
const BASE_SHIM = `/* Vendored for github.com/dhruvyad/dhruvyad.github.io — see scripts/vendor-distill.mjs.
   Upstream: ${TEMPLATE_URL} (Apache-2.0, The Distill Template Authors).
   Local modification: KaTeX and polyfill URLs now resolve relative to this file
   instead of pointing at distill.pub. */
var __distillVendorBase = (function () {
  if (typeof document === 'undefined') return '/vendor/';
  var s = document.currentScript;
  if (!s) {
    var all = document.getElementsByTagName('script');
    s = all[all.length - 1];
  }
  var src = (s && s.src) || '';
  return src ? src.slice(0, src.lastIndexOf('/') + 1) : '/vendor/';
})();
`

async function fetchText(url) {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`GET ${url} failed: ${res.status} ${res.statusText}`)
  return res.text()
}

function patch(source) {
  const edits = [
    [
      "const katexJSURL = 'https://distill.pub/third-party/katex/katex.min.js';",
      "const katexJSURL = __distillVendorBase + 'katex/katex.min.js';",
    ],
    [
      `const katexCSSTag = '<link rel="stylesheet" href="https://distill.pub/third-party/katex/katex.min.css" crossorigin="anonymous">';`,
      `const katexCSSTag = '<link rel="stylesheet" href="' + __distillVendorBase + 'katex/katex.min.css">';`,
    ],
    [
      "url: 'https://distill.pub/third-party/polyfills/webcomponents-lite.js'",
      "url: __distillVendorBase + 'polyfills/webcomponents-lite.js'",
    ],
    [
      "url: 'https://distill.pub/third-party/polyfills/intersection-observer.js'",
      "url: __distillVendorBase + 'polyfills/intersection-observer.js'",
    ],
  ]

  let out = source
  for (const [from, to] of edits) {
    if (!out.includes(from)) {
      throw new Error(
        `Upstream template.v2.js changed — could not find:\n  ${from}\n` +
          `Update the edit list in scripts/vendor-distill.mjs.`,
      )
    }
    out = out.replaceAll(from, to)
  }

  const leftover = out.match(/https:\/\/distill\.pub\/(third-party|template)[^'"]*/g)
  if (leftover) {
    throw new Error(`Unpatched distill.pub asset URLs remain: ${[...new Set(leftover)].join(', ')}`)
  }
  return BASE_SHIM + out
}

mkdirSync(resolve(vendorDir, 'polyfills'), { recursive: true })

const upstream = await fetchText(TEMPLATE_URL)
const sha = createHash('sha256').update(upstream).digest('hex')
writeFileSync(resolve(vendorDir, 'template.v2.js'), patch(upstream))
console.log(`template.v2.js  ${upstream.length} bytes  upstream sha256=${sha.slice(0, 16)}…`)

for (const url of POLYFILLS) {
  const name = url.split('/').pop()
  const body = await fetchText(url)
  writeFileSync(resolve(vendorDir, 'polyfills', name), body)
  console.log(`polyfills/${name}  ${body.length} bytes`)
}

console.log('\nVendored into public/vendor/. Commit these files.')
