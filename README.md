# Notes

A personal [distill.pub](https://distill.pub) — interactive explainers, written while learning
things properly.

**Live: https://dhruvyad.github.io/**

Every push to `main` rebuilds and republishes the site.

---

## Writing a post

```bash
npm install                                   # once
npm run new-post -- tokens-per-second "Where Tokens Per Second Comes From"
npm run dev                                   # http://localhost:5173/
```

That creates `posts/<slug>/` with the boilerplate wired up. Edit `index.html`, and the homepage
picks the post up automatically — the index is generated at build time from each article's
`<d-front-matter>` block, so there is no separate list to maintain.

```
posts/tokens-per-second/
├── index.html            the article
├── index.js              mounts the figures
├── bibliography.bib      citation keys for <d-cite>
└── figures/
    ├── model.js          maths shared by every figure in the post
    ├── DecodeBudget.svelte
    └── ...
```

Putting a post's maths in one `model.js` that all its figures import is worth the small extra
file: the figures then cannot disagree with each other, or with the numbers quoted in the prose.

## How this is built

distill.pub's own articles were **hand-written HTML** using a set of web components
(`<d-article>`, `<d-math>`, `<d-cite>`, `<d-figure>`…), with interactive figures as separate
Svelte/D3 modules mounted into `<d-figure>`. Each article was its own repository. This repo keeps
that authoring model — HTML-first, because "mount this component here, lazily, with these props"
is most of what a distill piece does and markdown can't express it — but makes it multi-post and
puts a modern build under it.

| | |
|---|---|
| Components | distill's `template.v2.js`, vendored into `public/vendor/` |
| Figures | Svelte 5 + D3; a shared block-diagram renderer for architecture |
| Build | Vite, multi-page (one HTML entry per post and per tool) |
| Deploy | GitHub Actions → GitHub Pages |

## Layout

```
posts/<slug>/      distill articles          -> /posts/<slug>/
tools/<slug>/      standalone tool pages     -> /tools/<slug>/
src/components/    components shared by both
src/lib/           glossary, figure mounting, architecture parsing
src/glossary/      shared term definitions and their tooltip visuals
```

Both lists on the homepage are generated at build time: articles from each post's
`<d-front-matter>`, tools from `tools/<slug>/meta.json`.

## Tools

`tools/model-explorer/` renders any Hugging Face `config.json` as a dataflow block diagram —
`src/lib/diagrams.js` turns the config into diagram specs and `src/components/BlockDiagram.svelte`
draws them, with composite boxes opening into their own diagram. The same component is embedded in
both articles; the tool adds preset models, file/paste loading, and a comparison table.

The parser handles dense and sparse architectures, MLA and GQA/MHA attention, DeepSeek-style sparse
attention indexers and MTP heads, and reproduces published parameter counts within about 1% across
GLM-5.2, DeepSeek-V3, gpt-oss-120b, Mixtral-8x7B and Qwen3-8B.

### Why the template is vendored

distill.pub has been dormant since 2021 and the framework unmaintained since 2022, and
`template.v2.js` hardcodes absolute `distill.pub` URLs for KaTeX and its polyfills. Loading it
from their CDN would make every article here depend on someone else's server staying up.
`scripts/vendor-distill.mjs` fetches it, rewrites those URLs to resolve relative to the script's
own location, and the result is committed. KaTeX comes from npm and is copied into
`public/vendor/katex/` on `prebuild` (it's gitignored — 60-odd font files).

To re-vendor after an upstream change: `npm run vendor-distill`. It fails loudly if the strings it
patches have moved.

## Writing figures

A figure is a Svelte component mounted into a `<d-figure>`:

```html
<d-figure id="decode-budget">
  <div id="decode-budget-target" class="figure-target"></div>
  <figcaption>What the reader should notice.</figcaption>
</d-figure>
```

```js
import { mountFigure } from '../../src/lib/figure.js'
import DecodeBudget from './figures/DecodeBudget.svelte'

mountFigure('decode-budget', DecodeBudget)
```

`<d-figure>` exposes a visibility state machine, and `src/lib/figure.js` wraps it:

| helper | when it runs | use for |
|---|---|---|
| `mountFigure` | once, ~2 viewports away | most figures |
| `mountFigureWhileVisible` | mounts/unmounts on scroll | `requestAnimationFrame` loops you don't want running off-screen |
| `onFigureReady` | once, ~2 viewports away | plain D3/canvas figures, no Svelte |

This laziness is the reason a page with twenty simulations still loads fast — nothing builds until
you scroll near it.

### Layout

distill puts a named grid across every block. Figures default to the prose column; widen them with
a class:

| class | column |
|---|---|
| *(default)* | `text` — aligned with prose |
| `wide` | `page` |
| `wider` | `middle` |
| `full-bleed` | `screen` |

Charts align left with their controls by default; add `center` to a `d-figure` to centre a narrow
chart in its column instead.

### Math, citations, footnotes

- Math: `$inline$`, `$$display$$`, or `<d-math block>` for a standalone equation. KaTeX.
- Citations: add a BibTeX entry to `bibliography.bib`, then `<d-cite key="dao2022flashattention">`.
- Footnotes: `<d-footnote>…</d-footnote>` inline; they collect into the appendix automatically.

`<d-footnote-list>` and `<d-citation-list>` render their own "Footnotes" / "References" headings —
don't add your own or you'll get duplicates.

### Byline

distill auto-inserts a `<d-byline>` with four fixed columns, two of which (Affiliations, DOI) render
as empty headings. `article.css` hides it; each post writes its own `.base-grid .byline` block with
authors and both dates. The scaffold generates it for you.

## Checks

```bash
npm run build
npm run preview          # then, in another shell:
npm run smoke
```

`npm run smoke` loads every built page in headless Chrome and asserts that KaTeX rendered every
`<d-math>`, that every `<d-cite>` resolved to a number, that footnotes and references populated,
that each `<d-figure>` mounted something, and that no page logged a console error or a failed
request. CI runs it and refuses to deploy if it fails — worth having when the upstream framework
is frozen and everything interesting happens at runtime.

## Credits

The web components are [distillpub/template](https://github.com/distillpub/template) by The
Distill Template Authors, Apache-2.0. The vendored copy in `public/vendor/template.v2.js` carries
a header noting the one local modification.
