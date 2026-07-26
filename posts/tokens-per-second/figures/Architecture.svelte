<script>
  import { partBreakdown, LAYER_MAP, GLM, expertsTouched, fmtBytes } from './model.js'

  let wBytes = $state(1)
  let logB = $state(0)
  let selectedId = $state('routed')

  const B = $derived(Math.round(2 ** logB))
  const bd = $derived(partBreakdown({ wBytes, B, S: 8192 }))
  const sel = $derived(bd.parts.find((p) => p.id === selectedId) ?? bd.parts[0])
  const touched = $derived(expertsTouched(B))

  const PRECISIONS = [
    { label: 'BF16', bytes: 2 },
    { label: 'FP8', bytes: 1 },
    { label: 'FP4', bytes: 0.5 },
  ]

  // The stack, top to bottom, as the signal flows. Dense MLP sits where the MoE
  // trio below it would otherwise go — it is the alternative on layers 0-2, not
  // a stage in between.
  const STACK = [
    { id: 'embed', label: 'Token embedding' },
    { id: 'indexer', label: 'DSA indexer' },
    { id: 'mla', label: 'MLA attention' },
    { id: 'dense', label: 'Dense MLP' },
    { id: 'router', label: 'Router' },
    { id: 'shared', label: 'Shared expert' },
    { id: 'routed', label: 'Routed experts' },
    { id: 'mtp', label: 'MTP head' },
    { id: 'lmhead', label: 'LM head' },
  ]

  const width = 704
  const height = 396
  const blockW = 208
  const blockH = 30
  const blockGap = 7
  const stackTop = 26
  // The layer body — everything between embedding and MTP — gets a bracket.
  const bodyFrom = 1
  const bodyTo = 6

  const blockY = (i) => stackTop + i * (blockH + blockGap)

  /** 16x16 grid of the 256 routed experts in one layer. */
  const gridX = 300
  const gridY = 46
  const cell = 11
  const gap = 1.6
  const experts = Array.from({ length: GLM.experts }, (_, i) => i)
  // Light up as many cells as the batch actually touches, in reading order.
  const lit = $derived(Math.round(touched))

  const mapX = 300
  const mapY = 268
  const mapW = 388
  const tickW = mapW / GLM.layers

  const barX = 300
  const barY = 338
  const barW = 388

  const fmtP = (n) => (n >= 1e9 ? `${(n / 1e9).toFixed(n >= 1e11 ? 0 : 2)}B` : `${(n / 1e6).toFixed(0)}M`)
  const fmtF = (n) => (n >= 1e9 ? `${(n / 1e9).toFixed(1)} GFLOP` : `${(n / 1e6).toFixed(0)} MFLOP`)

  const select = (id) => (selectedId = id)
  const onKey = (e, id) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      select(id)
    }
  }

  /** Which layers the selected part lives on, for the layer map. */
  const highlights = $derived(
    LAYER_MAP.map((l) => {
      if (selectedId === 'dense') return l.dense
      if (selectedId === 'indexer') return l.fullIndexer
      if (selectedId === 'routed' || selectedId === 'shared' || selectedId === 'router')
        return !l.dense
      if (selectedId === 'mla') return true
      return false
    }),
  )
</script>

<div class="figure-controls">
  <label>
    Batch <span class="value">{B}</span>
    <input type="range" min="0" max="10" step="0.05" bind:value={logB} />
  </label>
  {#each PRECISIONS as p}
    <button class:active={wBytes === p.bytes} onclick={() => (wBytes = p.bytes)}>{p.label}</button>
  {/each}
  <span class="hint">click any block</span>
</div>

<svg viewBox="0 0 {width} {height}" {width} style="max-width:100%;height:auto">
  <text x="0" y="12" class="cap">one of {GLM.layers} layers</text>

  <!-- bracket marking the repeated layer body -->
  <path
    d="M{blockW + 10},{blockY(bodyFrom) - 3} h6 v{blockY(bodyTo) + blockH - blockY(bodyFrom) + 6} h-6"
    class="bracket"
  />
  <text
    x={blockW + 22}
    y={(blockY(bodyFrom) + blockY(bodyTo) + blockH) / 2}
    class="bracket-t"
    dy="0.32em">×{GLM.layers}</text
  >

  {#each STACK as b, i}
    {@const p = bd.parts.find((q) => q.id === b.id)}
    {@const frac = Math.max(0, Math.min(1, p.activeShare))}
    <g
      role="button"
      tabindex="0"
      aria-label={b.label}
      class="block"
      class:sel={selectedId === b.id}
      onclick={() => select(b.id)}
      onkeydown={(e) => onKey(e, b.id)}
      transform="translate(0,{blockY(i)})"
    >
      <!-- dormant base, then the activated fraction painted over it -->
      <rect width={blockW} height={blockH} class="base" />
      <rect width={blockW * frac} height={blockH} class="live" />
      <rect width={blockW} height={blockH} class="outline" />
      <text x="9" y={blockH / 2} dy="0.32em" class="b-label" class:on-live={frac > 0.45}>
        {b.label}
      </text>
      <text x={blockW - 9} y={blockH / 2} dy="0.32em" text-anchor="end" class="b-num">
        {fmtP(p.params)}
      </text>
    </g>
  {/each}

  <!-- expert grid: the activation-sparsity picture -->
  <text x={gridX} y={gridY - 22} class="cap">routed experts in one layer</text>
  <text x={gridX} y={gridY - 8} class="cap dim">
    {lit} of {GLM.experts} read at batch {B}
  </text>
  {#each experts as e}
    {@const row = Math.floor(e / 16)}
    {@const col = e % 16}
    <rect
      x={gridX + col * (cell + gap)}
      y={gridY + row * (cell + gap)}
      width={cell}
      height={cell}
      class="expert"
      class:lit={e < lit}
      class:focus={selectedId === 'routed'}
    />
  {/each}

  <!-- layer map -->
  <text x={mapX} y={mapY - 8} class="cap">
    {GLM.layers} layers · 3 dense, 75 MoE, 21 with an indexer
  </text>
  {#each LAYER_MAP as l, i}
    <rect
      x={mapX + i * tickW}
      y={mapY}
      width={Math.max(1, tickW - 1)}
      height={16}
      class="tick"
      class:hl={highlights[i]}
    />
    {#if l.fullIndexer}
      <rect x={mapX + i * tickW} y={mapY + 19} width={Math.max(1, tickW - 1)} height={4} class="idx" />
    {/if}
  {/each}
  <text x={mapX} y={mapY + 38} class="cap dim">layer 0 → 77 · marks below = owns an indexer</text>

  <!-- overall activation bar -->
  <text x={barX} y={barY - 8} class="cap">
    {(bd.totalActive / 1e9).toFixed(1)}B of {(bd.totalParams / 1e9).toFixed(0)}B active per token
  </text>
  <rect x={barX} y={barY} width={barW} height={16} class="base" />
  <rect x={barX} y={barY} width={barW * (bd.totalActive / bd.totalParams)} height={16} class="live" />
  <text x={barX + barW} y={barY + 30} text-anchor="end" class="cap dim">
    {((100 * bd.totalActive) / bd.totalParams).toFixed(1)}% of the weights
  </text>
</svg>

<div class="panel">
  <div class="head">
    <span class="name">{sel.label}</span>
    <span class="det">{sel.count > 1 ? `×${sel.count.toLocaleString()} · ` : ''}{sel.detail}</span>
  </div>
  <div class="readouts">
    <div class="ro">
      <span class="k">parameters</span>
      <span class="v">{fmtP(sel.params)}</span>
      <span class="s">{(100 * sel.paramShare).toFixed(1)}% of model</span>
    </div>
    <div class="ro">
      <span class="k">active / token</span>
      <span class="v">{sel.active >= 1e6 ? fmtP(sel.active) : `${(sel.active / 1e3).toFixed(1)}k`}</span>
      <span class="s">{(100 * sel.activeShare).toFixed(1)}% of its weights</span>
    </div>
    <div class="ro">
      <span class="k">arithmetic</span>
      <span class="v">{fmtF(sel.flops)}</span>
      <span class="s">{(100 * sel.flopShare).toFixed(0)}% of per-token FLOPs</span>
    </div>
    <div class="ro">
      <span class="k">read per step</span>
      <span class="v">{fmtBytes(sel.bytes)}</span>
      <span class="s">{(100 * sel.byteShare).toFixed(0)}% of HBM traffic</span>
    </div>
  </div>
  <p class="note">{sel.note}</p>
</div>

<style>
  .cap {
    font-size: 10.5px;
    fill: rgba(0, 0, 0, 0.6);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .cap.dim {
    fill: rgba(0, 0, 0, 0.4);
    text-transform: none;
    letter-spacing: 0;
    font-size: 10.5px;
  }

  .base {
    fill: #e4e4e4;
  }

  .live {
    fill: #c0392b;
  }

  .outline {
    fill: none;
    stroke: rgba(0, 0, 0, 0.18);
  }

  .block {
    cursor: pointer;
  }

  .block .b-label {
    font-size: 12px;
    fill: rgba(0, 0, 0, 0.8);
  }

  .block .b-label.on-live {
    fill: #fff;
  }

  .block .b-num {
    font-size: 11px;
    font-family: 'SF Mono', Menlo, Consolas, monospace;
    fill: rgba(0, 0, 0, 0.5);
  }

  .block.sel .outline {
    stroke: #000;
    stroke-width: 2;
  }

  .block:hover .outline {
    stroke: rgba(0, 0, 0, 0.55);
  }

  .block:focus-visible .outline {
    stroke: #000;
    stroke-width: 2;
  }

  .bracket {
    fill: none;
    stroke: rgba(0, 0, 0, 0.25);
  }

  .bracket-t {
    font-size: 10.5px;
    fill: rgba(0, 0, 0, 0.45);
    font-variant-numeric: tabular-nums;
  }

  .expert {
    fill: #e4e4e4;
  }

  .expert.lit {
    fill: #c0392b;
  }

  .expert.focus {
    stroke: rgba(0, 0, 0, 0.12);
  }

  .tick {
    fill: #dcdcdc;
  }

  .tick.hl {
    fill: #c0392b;
  }

  .idx {
    fill: rgba(0, 0, 0, 0.55);
  }

  .panel {
    margin-top: 0.5em;
    padding-top: 0.7em;
    border-top: 1px solid rgba(0, 0, 0, 0.12);
  }

  .head {
    display: flex;
    align-items: baseline;
    gap: 0.6em;
    flex-wrap: wrap;
    margin-bottom: 0.6em;
  }

  .head .name {
    font-size: 15px;
    font-weight: 600;
  }

  .head .det {
    font-size: 12px;
    color: rgba(0, 0, 0, 0.5);
  }

  .readouts {
    display: flex;
    flex-wrap: wrap;
    gap: 0.8em 1.75em;
  }

  .ro {
    display: flex;
    flex-direction: column;
    gap: 0.1em;
  }

  .ro .k {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: rgba(0, 0, 0, 0.45);
  }

  .ro .v {
    font-size: 15px;
    font-family: 'SF Mono', Menlo, Consolas, monospace;
    font-variant-numeric: tabular-nums;
  }

  .ro .s {
    font-size: 11px;
    color: rgba(0, 0, 0, 0.5);
  }

  .note {
    font-size: 12px;
    line-height: 1.6em;
    color: rgba(0, 0, 0, 0.6);
    margin: 0.9em 0 0;
  }

  .hint {
    font-size: 12px;
    color: rgba(0, 0, 0, 0.45);
    font-style: italic;
  }

  button.active {
    background: #c0392b;
    color: #fff;
    border-color: #c0392b;
  }
</style>
