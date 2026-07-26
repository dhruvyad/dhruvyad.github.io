# Working in this repo

Preferences Dhruv has stated while building this site. Keep this file updated as more come up —
when he gives feedback on an article, record the general rule here rather than only fixing the
instance.

## Grounding

- **Never invent numbers.** Every quantitative claim comes from a primary source: a model's real
  `config.json`, actual weight file sizes, vendor spec sheets, published benchmarks. If a number
  can't be verified, either leave it out or say plainly that it's an estimate.
- **Verify the subject exists before writing about it.** Ask what the current release actually is
  rather than reasoning from memory — model versions move fast and the knowledge cutoff will be
  behind. Check Hugging Face / arXiv / vendor docs directly.
- **Prefer examples grounded in a well-known open-weights model**, so a reader can inspect the same
  artefacts. Architecture-specific detail is the point; a hypothetical model teaches much less.
- **Check citations against real metadata.** Titles and author lists get misremembered. Note when a
  paper's title differs from the name a model card uses for the technique.
- **Cross-check derived figures against ground truth.** The parameter model in
  `posts/tokens-per-second/` reproduces the real safetensors byte count to 0.006%; that check is
  what makes the rest of the article trustworthy. Do the equivalent wherever possible.
- Prefer measuring something yourself over repeating a claim. Counting indexer tensors in the
  checkpoint is better evidence than quoting the model card about IndexShare.

## Article structure

- Explain the terms. These pieces are for learning, so name and define the jargon (HBM, MBU,
  arithmetic intensity, machine balance) instead of assuming it.
- Derive rather than assert. Show the equation, then the arithmetic, then the consequence.
- State assumptions and limits honestly. Where a model is an upper bound, say so and explain the
  gap to reality — that discussion is usually the most valuable part.
- Correct the prose when a figure disagrees with it. This has already happened once: the claim that
  speculative verification is nearly free is true for dense models and false for sparse MoE at low
  batch. The figure was right.
- End with what to remember.

## Terms and links

- **Every technical term and acronym gets a tooltip — every single occurrence, however many times it
  appears.** HBM, KV cache, MTP, indexer, DRAM, SRAM, SMEM, SM, NVFP4, SXM, MLA, all of them.
- Do this through the shared machinery, never by hand-tagging the markup: define the term once in
  `src/glossary/terms.js` and `src/lib/glossary.js` wraps every mention across the article at
  runtime. One definition, one tooltip element, N triggers. Adding a term retroactively annotates
  every article.
- Tooltips can hold **visualisations, images and diagrams**, not just prose — see
  `src/glossary/visuals.js` for the small bar/grid/hierarchy builders. Use one wherever a picture
  beats a sentence (bandwidth comparisons, sparsity grids, memory hierarchies).
- **Link generously.** Each glossary entry takes a `links: [{label, url}]` list, which is the
  best place for them: every mention of GLM-5.2 then routes to its Hugging Face repo, not just the
  first. Add inline prose links too for things that aren't glossary terms (companies, specific
  files, tool docs).
- New terms should go in the shared glossary rather than a per-post one, since they recur.

## Figures

- **Interactive beats static.** Where a static table would do, prefer a figure the reader can
  manipulate — clickable parts, sliders over the parameters that actually matter.
- **Attach real metrics to parts of a diagram.** If a diagram shows an architecture, clicking a
  component should surface its cost: parameters, FLOPs, bytes read, share of the whole.
- **Colour-code what's active; grey out what isn't.** Sparsity, utilisation and "how much of this
  thing is doing work" should be visible at a glance rather than described in prose.
- **One `model.js` per post**, imported by every figure in it. Figures must not be able to disagree
  with each other or with the prose. Verify it against a reference implementation.
- Figures align with the **prose column**, not distill's wider `page` column: same left and right
  edges as the body text. Charts are `width = 704` to fill it exactly. Measure the bounding boxes to
  confirm — don't eyeball it.
- Captions say what the reader should notice, not what the figure contains.
- Check for label collisions at the default state and at the extremes of every control.

## Site conventions

- **Homepage** is just the title and the article list. No tagline, no self-description, no byline.
- **Byline** on each article: authors, published date, last updated date. Authors are Dhruv Yadav
  and the model that drafted it (e.g. "Claude Opus 5"). No affiliations, no DOI — distill's built-in
  `<d-byline>` renders those as empty headings, so it's suppressed in `article.css` and each post
  writes its own block. The scaffold generates it.
- Keep the prose in British-ish spelling as already used ("quantisation", "utilisation").

## Before shipping

- `npm run build`, then `npm run smoke` against `npm run preview`. The smoke test checks what
  actually rendered in a real browser, which is the only thing that matters when the components use
  shadow DOM.
- Drive every interactive control and confirm the output changes. A figure that mounts is not a
  figure that works.
- Look at the rendered result. Screenshot it, at the default state and after interaction.
- Re-verify against the live URL after deploying, not just locally.
