<script>
  /**
   * The same grammar at every scale.
   *
   * A weight matrix is a rectangle sized by its real dimensions. Repetition is
   * collapsed — a 78-layer model is drawn as one layer with a ×78 badge — which
   * is what lets an entire model sit on one screen as a dozen shapes instead of
   * twenty thousand. Click into any shape and the structure stays the same while
   * detail appears: block, then heatmap, then cells with numbers.
   */
  import { collapseArchitecture, ROLES } from '../../../src/lib/architecture.js'
  import { signedFill, textOn, fmt, seeded } from './kit.js'
  import config from '../data/glm-5.2-config.json'

  const { meta, stages, totalBlocks } = collapseArchitecture(config)

  /** [] = whole model, [stageIndex] = one layer, [stageIndex, partIndex] = one matrix. */
  let path = $state([])
  let cells = $state(24) // sample size at the matrix level

  const stage = $derived(path.length ? stages[path[0]] : null)
  const part = $derived(
    path.length > 1 && stage?.kind === 'group' ? stage.parts[path[1]] : path.length === 1 && stage?.kind === 'tensor' ? stage : null,
  )
  const tensor = $derived(part?.kind === 'group' ? part.parts[0] : part)
  const level = $derived(tensor ? 'matrix' : stage ? 'layer' : 'model')

  const width = 704

  /** Box side from a real dimension — log-scaled so 256 and 155k both fit. */
  const side = (d, lo, hi) => {
    const t = (Math.log2(d) - Math.log2(256)) / (Math.log2(200000) - Math.log2(256))
    return lo + (hi - lo) * Math.max(0, Math.min(1, t))
  }
  const boxOf = (dims, lo, hi) => ({ w: side(dims[1], lo, hi), h: side(dims[0], lo, hi) })

  const fmtP = (n) =>
    n >= 1e9 ? `${(n / 1e9).toFixed(1)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(0)}M` : `${(n / 1e3).toFixed(0)}k`
  const colour = (role) => `#${ROLES[role].color.toString(16).padStart(6, '0')}`

  // ------------------------------------------------------------ model level
  const MODEL_LO = 12
  const MODEL_HI = 40
  const rowGap = 16

  const modelRows = $derived.by(() => {
    let y = 0
    const rows = stages.map((s, i) => {
      const parts = s.kind === 'tensor' ? [s] : s.parts
      const boxes = parts.map((p) => {
        const t = p.kind === 'group' ? p.parts[0] : p
        return { part: p, tensor: t, ...boxOf(t.dims, MODEL_LO, MODEL_HI) }
      })
      const h = Math.max(...boxes.map((b) => b.h)) + (s.kind === 'group' ? 34 : 18)
      const row = { stage: s, index: i, boxes, y, h }
      y += h + rowGap
      return row
    })
    return { rows, height: y }
  })

  // ------------------------------------------------------------ layer level
  const LAYER_LO = 26
  const LAYER_HI = 104

  /** Rough width of a caption, so labels get room and never collide. */
  const captionW = (t, p) => {
    const dims = `${t.dims[0].toLocaleString()}×${t.dims[1].toLocaleString()}`
    const extra = p.kind === 'group' ? ` · ×${p.count}, ${p.active} active` : ''
    return Math.max(t.name.length * 6.4, (dims + extra).length * 6.0)
  }

  const layerBoxes = $derived.by(() => {
    if (!stage || stage.kind !== 'group') return []
    // Lay out into rows first so every caption in a row shares one baseline —
    // ragged baselines read as collisions even when the text does not overlap.
    const rows = []
    let row = []
    let x = 0
    for (const p of stage.parts) {
      const t = p.kind === 'group' ? p.parts[0] : p
      const b = boxOf(t.dims, LAYER_LO, LAYER_HI)
      const advance = Math.max(b.w, captionW(t, p)) + 20
      if (x + advance > width - 8 && row.length) {
        rows.push(row)
        row = []
        x = 0
      }
      row.push({ part: p, tensor: t, x, advance, ...b })
      x += advance
    }
    if (row.length) rows.push(row)

    const out = []
    let y = 0
    for (const r of rows) {
      const rowH = Math.max(...r.map((b) => b.h))
      for (const b of r) out.push({ ...b, y, rowH })
      y += rowH + 58
    }
    return out
  })
  const layerHeight = $derived(
    layerBoxes.length ? Math.max(...layerBoxes.map((b) => b.y + b.rowH)) + 52 : 0,
  )

  // ----------------------------------------------------------- matrix level
  /**
   * Illustrative values. The config describes the shape of a weight matrix and
   * never its contents, so these are generated, not read — labelled as such
   * below rather than passed off as the model's own numbers.
   */
  const sample = $derived.by(() => {
    if (!tensor) return []
    const rnd = seeded(tensor.dims[0] * 31 + tensor.dims[1])
    const r = Math.min(cells, tensor.dims[0])
    const c = Math.min(cells, tensor.dims[1])
    return Array.from({ length: r }, () =>
      Array.from({ length: c }, () => Math.round((rnd() * 2 - 1) * 12) / 10),
    )
  })
  const cellPx = $derived(Math.min(30, Math.floor(560 / Math.max(1, sample[0]?.length ?? 1))))
  const showNumbers = $derived(cellPx >= 26)

  const enter = (i, j) => {
    path = j == null ? [i] : [i, j]
    cells = 24
  }
</script>

<div class="figure-controls">
  <button class="crumb" onclick={() => (path = [])} disabled={!path.length}>Model</button>
  {#if stage}
    <span class="sep">›</span>
    <button class="crumb" onclick={() => (path = [path[0]])} disabled={path.length < 2}>
      {stage.kind === 'group' ? stage.label : stage.name}
    </button>
  {/if}
  {#if tensor && level === 'matrix'}
    <span class="sep">›</span>
    <button class="crumb" disabled>{tensor.name}</button>
  {/if}
  {#if level === 'matrix'}
    <span class="spacer"></span>
    <label>
      Zoom <span class="value">{cells}²</span>
      <input type="range" min="6" max="40" step="2" bind:value={cells} />
    </label>
  {/if}
</div>

{#if level === 'model'}
  <svg viewBox="0 0 {width} {modelRows.height}" {width} style="max-width:100%;height:auto">
    {#each modelRows.rows as row}
      <g transform="translate(0,{row.y})">
        {#if row.stage.kind === 'group'}
          <rect
            x="0"
            y="0"
            width={width - 2}
            height={row.h}
            rx="4"
            class="group"
            role="button"
            tabindex="0"
            aria-label={row.stage.label}
            onclick={() => enter(row.index)}
            onkeydown={(e) => (e.key === 'Enter' || e.key === ' ') && enter(row.index)}
          />
          <text x="10" y="15" class="glabel">{row.stage.label}</text>
          <text x={width - 12} y="15" text-anchor="end" class="count">×{row.stage.count}</text>
        {/if}

        {#each row.boxes as b, k}
          {@const bx = (row.stage.kind === 'group' ? 12 : 0) + k * 0}
          <g
            transform="translate({row.stage.kind === 'group'
              ? 12 + row.boxes.slice(0, k).reduce((s, p) => s + p.w + 10, 0)
              : 0},{row.stage.kind === 'group' ? 24 : 0})"
          >
            <rect
              width={b.w}
              height={b.h}
              fill={colour(b.tensor.role)}
              opacity={b.part.kind === 'group' ? 0.32 : 0.9}
              class="box"
              role="button"
              tabindex="0"
              aria-label={b.tensor.name}
              onclick={(e) => {
                e.stopPropagation()
                enter(row.index, row.stage.kind === 'group' ? k : null)
              }}
              onkeydown={(e) =>
                (e.key === 'Enter' || e.key === ' ') &&
                enter(row.index, row.stage.kind === 'group' ? k : null)}
            />
            {#if b.part.kind === 'group'}
              <rect width={b.w} height={b.h} class="stack" />
              <text x={b.w / 2} y={b.h + 11} text-anchor="middle" class="mini">
                ×{b.part.count}
              </text>
            {/if}
            {#if row.stage.kind !== 'group'}
              <text x={b.w + 10} y={b.h / 2 + 4} class="tlabel">
                {b.tensor.name} · {b.tensor.dims[0].toLocaleString()}×{b.tensor.dims[1].toLocaleString()}
              </text>
            {/if}
          </g>
        {/each}
      </g>
    {/each}
  </svg>

  <p class="note">
    The whole model: <b>{totalBlocks.toLocaleString()} weight matrices</b> drawn as
    <b>{stages.length} stages</b>. Nothing has been left out — the repeats are folded into the ×N
    badges, because a {meta.layers}-layer model is one layer {meta.layers} times, and a
    {meta.nExperts}-expert layer is one expert {meta.nExperts} times. Every rectangle is sized by its
    real dimensions. Click any of them.
  </p>
{:else if level === 'layer'}
  <svg viewBox="0 0 {width} {layerHeight}" {width} style="max-width:100%;height:auto">
    {#each layerBoxes as b, k}
      <g transform="translate({b.x},{b.y})">
        <rect
          width={b.w}
          height={b.h}
          fill={colour(b.tensor.role)}
          opacity={b.part.kind === 'group' ? 0.32 : 0.9}
          class="box"
          role="button"
          tabindex="0"
          aria-label={b.tensor.name}
          onclick={() => enter(path[0], k)}
          onkeydown={(e) => (e.key === 'Enter' || e.key === ' ') && enter(path[0], k)}
        />
        {#if b.part.kind === 'group'}
          <rect width={b.w} height={b.h} class="stack" />
        {/if}
        <text x="0" y={b.rowH + 16} class="tlabel">{b.tensor.name}</text>
        <text x="0" y={b.rowH + 30} class="dims">
          {b.tensor.dims[0].toLocaleString()}×{b.tensor.dims[1].toLocaleString()}
          {#if b.part.kind === 'group'}· ×{b.part.count}, {b.part.active} active{/if}
        </text>
      </g>
    {/each}
  </svg>

  <p class="note">
    One <b>{stage.label}</b>, of which the model has <b>{stage.count}</b>. Box proportions are the
    real matrix shapes: the tall narrow ones project down into a smaller space, the wide ones project
    back up. Click a box to look inside it.
  </p>
{:else}
  <svg
    viewBox="0 0 {width} {(sample.length ?? 0) * cellPx + 46}"
    {width}
    style="max-width:100%;height:auto"
  >
    <text x="0" y="12" class="glabel">
      {tensor.name} · {tensor.dims[0].toLocaleString()} × {tensor.dims[1].toLocaleString()} ·
      {fmtP(tensor.params)} parameters
    </text>
    {#each sample as row, i}
      {#each row as v, j}
        <rect
          x={j * cellPx}
          y={24 + i * cellPx}
          width={cellPx - 1}
          height={cellPx - 1}
          fill={signedFill(v, 1.2)}
        />
        {#if showNumbers}
          <text x={j * cellPx + cellPx / 2 - 0.5} y={24 + i * cellPx + cellPx / 2 + 3} class="cellv" fill={textOn(v, 1.2)}>
            {fmt(v)}
          </text>
        {/if}
      {/each}
    {/each}
  </svg>

  <p class="note">
    Showing a <b>{sample.length}×{sample[0]?.length}</b> corner of a
    {tensor.dims[0].toLocaleString()}×{tensor.dims[1].toLocaleString()} matrix — about
    <b>{((100 * (sample.length * (sample[0]?.length ?? 0))) / (tensor.dims[0] * tensor.dims[1])).toExponential(1)}%</b>
    of it. Zoom in far enough and the numbers appear; this is the same object as the grid in the
    matrix-multiply figure above, just much larger. The values are illustrative: a
    <code>config.json</code> gives the shape of a weight matrix and never its contents.
  </p>
{/if}

<style>
  .group {
    fill: rgba(0, 0, 0, 0.028);
    stroke: rgba(0, 0, 0, 0.16);
    stroke-dasharray: 3 3;
    cursor: pointer;
  }

  .group:hover {
    fill: rgba(0, 0, 0, 0.055);
  }

  .box {
    stroke: rgba(0, 0, 0, 0.25);
    cursor: pointer;
  }

  .box:hover {
    stroke: #000;
    stroke-width: 1.6;
  }

  /* A stack of identical matrices, drawn as an offset outline. */
  .stack {
    fill: none;
    stroke: rgba(0, 0, 0, 0.4);
    transform: translate(3px, -3px);
    pointer-events: none;
  }

  .glabel {
    font-size: 11px;
    fill: rgba(0, 0, 0, 0.6);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .count {
    font-size: 12px;
    fill: #c0392b;
    font-family: 'SF Mono', Menlo, Consolas, monospace;
    font-weight: 600;
  }

  .tlabel {
    font-size: 11.5px;
    fill: rgba(0, 0, 0, 0.75);
  }

  .dims,
  .mini {
    font-size: 10px;
    fill: rgba(0, 0, 0, 0.45);
    font-family: 'SF Mono', Menlo, Consolas, monospace;
  }

  .cellv {
    font-size: 9.5px;
    text-anchor: middle;
    font-family: 'SF Mono', Menlo, Consolas, monospace;
    pointer-events: none;
  }

  .crumb {
    font: inherit;
    font-size: 12.5px;
    padding: 0.2em 0.6em;
    border: 1px solid rgba(0, 0, 0, 0.2);
    border-radius: 3px;
    background: #fff;
    cursor: pointer;
  }

  .crumb:disabled {
    background: rgba(0, 0, 0, 0.04);
    color: rgba(0, 0, 0, 0.55);
    cursor: default;
  }

  .sep {
    color: rgba(0, 0, 0, 0.3);
  }

  .spacer {
    flex: 1;
  }

  .note {
    font-size: 12px;
    line-height: 1.6em;
    color: rgba(0, 0, 0, 0.6);
    margin: 0.9em 0 0;
  }

  .note code {
    font-family: 'SF Mono', Menlo, Consolas, monospace;
    font-size: 0.92em;
    background: rgba(0, 0, 0, 0.05);
    padding: 0.05em 0.3em;
    border-radius: 2px;
  }
</style>
