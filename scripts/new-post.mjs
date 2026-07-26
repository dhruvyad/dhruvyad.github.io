/**
 * Scaffolds a new article: npm run new-post -- my-slug "My Title"
 *
 * Writes posts/<slug>/{index.html, index.js, bibliography.bib, figures/Example.svelte}
 * with the boilerplate already wired up — front matter, KaTeX delimiters, an
 * appendix, and one lazily-mounted Svelte figure to edit or delete.
 */
import { existsSync, mkdirSync, writeFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')

// npm inserts a bare `--` when forwarding args; drop any that reach us.
const args = process.argv.slice(2).filter((a) => a !== '--')
const [rawSlug, ...titleParts] = args

if (!rawSlug) {
  console.error('Usage: npm run new-post -- <slug> "Optional Title"')
  process.exit(1)
}

const slug = rawSlug
  .toLowerCase()
  .replace(/[^a-z0-9]+/g, '-')
  .replace(/^-|-$/g, '')

if (!slug) {
  console.error(`"${rawSlug}" does not reduce to a usable slug.`)
  process.exit(1)
}

const title =
  titleParts.join(' ').trim() ||
  slug
    .split('-')
    .map((w) => w[0].toUpperCase() + w.slice(1))
    .join(' ')

const dir = resolve(root, 'posts', slug)
if (existsSync(dir)) {
  console.error(`posts/${slug} already exists.`)
  process.exit(1)
}

const today = new Date().toISOString().slice(0, 10)

const indexHtml = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <link rel="icon" href="/favicon.svg" type="image/svg+xml" />
    <title>${title} — Notes</title>

    <script src="/vendor/template.v2.js"></script>
    <link rel="stylesheet" href="/vendor/katex/katex.min.css" />
    <link rel="stylesheet" href="/src/styles/article.css" />
  </head>

  <body>
    <d-front-matter>
      <script type="text/json">
        {
          "title": "${title}",
          "description": "One sentence on what this article is about.",
          "published": "${today}",
          "tags": [],
          "authors": [
            {
              "author": "Dhruv Yadav",
              "authorURL": "https://github.com/dhruvyad"
            }
          ],
          "katex": {
            "delimiters": [
              { "left": "$", "right": "$", "display": false },
              { "left": "$$", "right": "$$", "display": true }
            ]
          }
        }
      </script>
    </d-front-matter>

    <d-title>
      <a class="home-link" href="../../">← Notes</a>
      <h1>${title}</h1>
      <p>The standfirst: what question does this article answer, and why bother?</p>
    </d-title>

    <d-article>
      <p>
        Opening paragraph. Inline math looks like $e^{i\\pi} = -1$, and display math gets its own
        block:
      </p>

      <d-math block>
        \\int_{-\\infty}^{\\infty} e^{-x^2} \\, dx = \\sqrt{\\pi}
      </d-math>

      <p>
        Here is a footnote<d-footnote>Footnotes appear in the margin on wide screens and collect
        themselves into the appendix.</d-footnote> and a citation<d-cite key="example2024"></d-cite>
        — add real keys to bibliography.bib.
      </p>

      <h2>A figure</h2>

      <p>
        Figures mount only once they near the viewport. Add <code>class="wide"</code> or
        <code>class="full-bleed"</code> to give one more room.
      </p>

      <d-figure id="example">
        <div id="example-target" class="figure-target"></div>
        <figcaption>Describe what the reader should notice, not what the figure contains.</figcaption>
      </d-figure>

      <h2>What to remember</h2>

      <p>The one or two things worth carrying away.</p>
    </d-article>

    <d-appendix>
      <d-footnote-list></d-footnote-list>
      <d-citation-list></d-citation-list>
    </d-appendix>

    <d-bibliography src="bibliography.bib"></d-bibliography>

    <script type="module" src="./index.js"></script>
  </body>
</html>
`

const indexJs = `import { mountFigure } from '../../src/lib/figure.js'
import Example from './figures/Example.svelte'

mountFigure('example', Example)
`

const exampleSvelte = `<script>
  import { scaleLinear } from 'd3-scale'

  let amplitude = $state(1)

  const width = 620
  const height = 220
  const margin = { top: 16, right: 16, bottom: 28, left: 36 }
  const innerWidth = width - margin.left - margin.right
  const innerHeight = height - margin.top - margin.bottom

  const x = scaleLinear().domain([0, 2 * Math.PI]).range([0, innerWidth])
  const y = scaleLinear().domain([-2, 2]).range([innerHeight, 0])

  const path = $derived(
    Array.from({ length: 120 }, (_, i) => {
      const t = (i / 119) * 2 * Math.PI
      return \`\${i === 0 ? 'M' : 'L'}\${x(t).toFixed(2)},\${y(amplitude * Math.sin(t)).toFixed(2)}\`
    }).join(' '),
  )
</script>

<div class="figure-controls">
  <label>
    Amplitude <span class="value">{amplitude.toFixed(2)}</span>
    <input type="range" min="0" max="2" step="0.05" bind:value={amplitude} />
  </label>
</div>

<svg viewBox="0 0 {width} {height}" {width} style="max-width:100%;height:auto">
  <g transform="translate({margin.left},{margin.top})">
    {#each y.ticks(5) as tick}
      <g transform="translate(0,{y(tick)})">
        <line x2={innerWidth} class="grid" />
        <text x="-8" dy="0.32em" text-anchor="end" class="tick">{tick}</text>
      </g>
    {/each}
    <path d={path} class="curve" />
  </g>
</svg>

<style>
  .grid {
    stroke: rgba(0, 0, 0, 0.07);
  }

  .tick {
    font-size: 11px;
    fill: rgba(0, 0, 0, 0.5);
  }

  .curve {
    fill: none;
    stroke: #c0392b;
    stroke-width: 2;
  }
</style>
`

const bib = `@article{example2024,
  title = {Replace this entry with something you actually cite},
  author = {Author, Some},
  journal = {A Journal},
  year = {2024},
  url = {https://example.com}
}
`

mkdirSync(resolve(dir, 'figures'), { recursive: true })
writeFileSync(resolve(dir, 'index.html'), indexHtml)
writeFileSync(resolve(dir, 'index.js'), indexJs)
writeFileSync(resolve(dir, 'bibliography.bib'), bib)
writeFileSync(resolve(dir, 'figures/Example.svelte'), exampleSvelte)

console.log(`Created posts/${slug}/
  index.html            the article — front matter, prose, figures
  index.js              mounts the figures
  figures/Example.svelte  a starter interactive figure
  bibliography.bib      citation keys for <d-cite>

Next: npm run dev, then open http://localhost:5173/posts/${slug}/
The homepage picks it up automatically from the front matter.`)
