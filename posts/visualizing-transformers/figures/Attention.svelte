<script>
  /**
   * Attention, assembled from the two previous figures: Q·Kᵀ is a matrix multiply
   * whose every cell is a dot product between one token's query and one token's
   * key, softmax turns each row into weights, and the weights multiply V.
   *
   * Clicking a token follows its row through all three steps.
   */
  import {
    CELL,
    GAP,
    cellXY,
    signedFill,
    unitFill,
    textOn,
    textOnUnit,
    fmt,
    randomMatrix,
    dot,
    softmax,
    POS,
  } from './kit.js'

  const TOKENS = ['the', 'cat', 'sat', 'on', 'the', 'mat']
  const T = TOKENS.length
  const D = 4

  let Q = $state(randomMatrix(T, D, 11))
  let K = $state(randomMatrix(T, D, 29))
  let V = $state(randomMatrix(T, D, 47))
  let query = $state(2)
  let causal = $state(true)
  let scaled = $state(true)

  const scale = $derived(scaled ? Math.sqrt(D) : 1)
  const scores = $derived(
    Q.map((q, i) => K.map((k, j) => (causal && j > i ? -Infinity : dot(q, k) / scale))),
  )
  const weights = $derived(scores.map((row) => softmax(row)))
  const out = $derived(
    weights.map((w) => Array.from({ length: D }, (_, d) => w.reduce((s, wi, j) => s + wi * V[j][d], 0))),
  )

  const cw = CELL + GAP
  const colGap = 46
  const qW = D * cw - GAP
  const sW = T * cw - GAP
  const labW = 42

  const xQ = labW
  const xS = xQ + qW + colGap + 26
  const xW = xS + sW + colGap
  const xO = xW + sW + colGap

  const width = xO + D * cw - GAP + 8
  const top = 30
  const height = top + T * cw + 96

  const yOf = (i) => top + cellXY(i, 0).y
</script>

<div class="figure-controls">
  <button class:active={causal} onclick={() => (causal = !causal)}>
    Causal mask {causal ? 'on' : 'off'}
  </button>
  <button class:active={scaled} onclick={() => (scaled = !scaled)}>
    ÷ √d {scaled ? 'on' : 'off'}
  </button>
  <button
    onclick={() => {
      Q = randomMatrix(T, D, Math.floor(Math.random() * 1e6))
      K = randomMatrix(T, D, Math.floor(Math.random() * 1e6))
      V = randomMatrix(T, D, Math.floor(Math.random() * 1e6))
    }}>Randomise</button
  >
  <span class="hint">click a token on the left</span>
</div>

<svg viewBox="0 0 {width} {height}" {width} style="max-width:100%;height:auto">
  <!-- token labels, clickable -->
  {#each TOKENS as tok, i}
    <text
      x={labW - 10}
      y={yOf(i) + CELL / 2 + 4}
      text-anchor="end"
      class="tok"
      class:sel={i === query}
      role="button"
      tabindex="0"
      onclick={() => (query = i)}
      onkeydown={(e) => (e.key === 'Enter' || e.key === ' ') && (query = i)}>{tok}</text
    >
  {/each}

  <!-- Q -->
  <text x={xQ} y={top - 10} class="cap">Q · one row per token</text>
  {#each Q as row, i}
    {#each row as v, d}
      {@const p = cellXY(i, d)}
      <rect
        x={xQ + p.x}
        y={top + p.y}
        width={CELL}
        height={CELL}
        fill={signedFill(v, 1.4)}
        class="cell"
        class:dim={i !== query}
      />
    {/each}
  {/each}

  <text x={xS - 30} y={top + (T * cw) / 2} class="op">×Kᵀ</text>

  <!-- scores -->
  <text x={xS} y={top - 10} class="cap">scores = Q·Kᵀ{scaled ? ' / √d' : ''}</text>
  {#each scores as row, i}
    {#each row as v, j}
      {@const p = cellXY(i, j)}
      {@const masked = v === -Infinity}
      <rect
        x={xS + p.x}
        y={top + p.y}
        width={CELL}
        height={CELL}
        fill={masked ? '#f7f6f5' : signedFill(v, 2)}
        class="cell"
        class:dim={i !== query}
      />
      {#if !masked}
        <text
          x={xS + p.x + CELL / 2}
          y={top + p.y + CELL / 2 + 3.5}
          class="v"
          fill={textOn(v, 2)}
          opacity={i === query ? 1 : 0.35}>{fmt(v)}</text
        >
      {/if}
    {/each}
  {/each}

  <text x={xW - 26} y={top + (T * cw) / 2} class="op">softmax</text>

  <!-- weights -->
  <text x={xW} y={top - 10} class="cap">weights · each row sums to 1</text>
  {#each weights as row, i}
    {#each row as v, j}
      {@const p = cellXY(i, j)}
      <rect
        x={xW + p.x}
        y={top + p.y}
        width={CELL}
        height={CELL}
        fill={unitFill(v)}
        class="cell"
        class:dim={i !== query}
      />
      <text
        x={xW + p.x + CELL / 2}
        y={top + p.y + CELL / 2 + 3.5}
        class="v"
        fill={textOnUnit(v)}
        opacity={i === query ? 1 : 0.35}>{v < 0.005 ? '' : v.toFixed(2)}</text
      >
    {/each}
  {/each}

  <text x={xO - 28} y={top + (T * cw) / 2} class="op">×V</text>

  <!-- output -->
  <text x={xO} y={top - 10} class="cap">output</text>
  {#each out as row, i}
    {#each row as v, d}
      {@const p = cellXY(i, d)}
      <rect
        x={xO + p.x}
        y={top + p.y}
        width={CELL}
        height={CELL}
        fill={signedFill(v, 1.2)}
        class="cell"
        class:dim={i !== query}
      />
    {/each}
  {/each}

  <!-- what the selected token actually did -->
  <g transform="translate({labW},{top + T * cw + 26})">
    <text x="0" y="0" class="cap">
      “{TOKENS[query]}” builds its output as a weighted blend of every value row
    </text>
    {#each weights[query] as w, j}
      {#if w > 0.004}
        <g transform="translate({j * 108},14)">
          <rect x="0" y="0" width={Math.max(2, w * 96)} height="12" fill={POS} opacity="0.85" />
          <text x="0" y="26" class="mini">{(w * 100).toFixed(0)}% “{TOKENS[j]}”</text>
        </g>
      {/if}
    {/each}
  </g>
</svg>

<div class="readouts">
  <div class="ro">
    <span class="k">dot products</span>
    <span class="v">{T} × {T} = {T * T}</span>
  </div>
  <div class="ro">
    <span class="k">each of length</span>
    <span class="v">{D}</span>
  </div>
  <div class="ro">
    <span class="k">scores cost</span>
    <span class="v">{2 * T * T * D} FLOPs</span>
  </div>
  <div class="ro">
    <span class="k">grows as</span>
    <span class="v">T²</span>
  </div>
</div>

<p class="note">
  Nothing new has been introduced. The scores are one matrix multiply — every cell a dot product
  between a token's query and a token's key, exactly the quantity from the first figure. Softmax
  turns each row into weights that sum to one. Multiplying by V is another matrix multiply. The
  causal mask simply forbids looking right. Notice the score panel is <b>T × T</b>: double the
  tokens and it quadruples, which is the single fact behind every long-context difficulty.
</p>

<style>
  .cap {
    font-size: 11px;
    fill: rgba(0, 0, 0, 0.5);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .tok {
    font-size: 13px;
    fill: rgba(0, 0, 0, 0.55);
    cursor: pointer;
    font-family: Georgia, 'Times New Roman', serif;
  }

  .tok.sel {
    fill: #000;
    font-weight: 700;
  }

  .cell {
    stroke: rgba(0, 0, 0, 0.06);
  }

  .cell.dim {
    opacity: 0.34;
  }

  .v {
    font-size: 10px;
    text-anchor: middle;
    font-family: 'SF Mono', Menlo, Consolas, monospace;
    pointer-events: none;
  }

  .op {
    font-size: 11px;
    fill: rgba(0, 0, 0, 0.45);
    text-anchor: middle;
    font-family: 'SF Mono', Menlo, Consolas, monospace;
  }

  .mini {
    font-size: 10.5px;
    fill: rgba(0, 0, 0, 0.6);
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

  .readouts {
    display: flex;
    flex-wrap: wrap;
    gap: 0 1.75em;
    margin-top: 0.4em;
    padding-top: 0.6em;
    border-top: 1px solid rgba(0, 0, 0, 0.1);
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

  .note {
    font-size: 12px;
    line-height: 1.6em;
    color: rgba(0, 0, 0, 0.6);
    margin: 0.9em 0 0;
  }
</style>
