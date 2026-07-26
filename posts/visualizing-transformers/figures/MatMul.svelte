<script>
  /**
   * A matrix multiply as a grid of dot products. Hovering an output cell shows
   * exactly which row and which column produced it, and the expansion underneath
   * is the same arithmetic as the previous figure.
   */
  import { CELL, GAP, cellXY, signedFill, textOn, MUTED, fmt, randomMatrix, dot } from './kit.js'

  const M = 4
  const K = 6
  const N = 5

  let A = $state(randomMatrix(M, K, 7))
  let B = $state(randomMatrix(K, N, 23))
  let sel = $state({ i: 1, j: 2 })
  let playing = $state(false)

  const C = $derived(A.map((row) => Array.from({ length: N }, (_, j) => dot(row, B.map((r) => r[j])))))
  const terms = $derived(A[sel.i].map((v, k) => ({ a: v, b: B[k][sel.j], p: v * B[k][sel.j] })))

  const cw = CELL + GAP
  const aW = K * cw - GAP
  const bW = N * cw - GAP
  const gapX = 58
  const bX = aW + gapX
  const cX = bX + bW + gapX
  const width = cX + N * cw - GAP + 8
  const topB = -(K - M) * cw - 34 // B sits above so its columns line up with C

  // B is drawn above-right so its columns visually feed down into C's columns.
  const bY = -34
  const aY = 34
  const cY = 34
  // Tall enough for the matrices *and* the expansion written underneath them.
  const height = aY + M * cw + 84 - (bY - 26)

  $effect(() => {
    if (!playing) return
    const id = setInterval(() => {
      const next = sel.j + 1
      if (next < N) sel = { i: sel.i, j: next }
      else if (sel.i + 1 < M) sel = { i: sel.i + 1, j: 0 }
      else sel = { i: 0, j: 0 }
    }, 420)
    return () => clearInterval(id)
  })

  const scaleA = 1.4
  const scaleC = 3.2
</script>

<div class="figure-controls">
  <button class:active={playing} onclick={() => (playing = !playing)}>
    {playing ? 'Pause' : 'Sweep every cell'}
  </button>
  <button
    onclick={() => {
      A = randomMatrix(M, K, Math.floor(Math.random() * 1e6))
      B = randomMatrix(K, N, Math.floor(Math.random() * 1e6))
    }}>Randomise</button
  >
  <span class="hint">hover any cell of the result</span>
</div>

<svg viewBox="0 {bY - 26} {width} {height}" {width} style="max-width:100%;height:auto">
  <!-- A -->
  <text x="0" y={aY - 10} class="cap">A · {M}&times;{K}</text>
  {#each A as row, i}
    {#each row as v, k}
      {@const p = cellXY(i, k)}
      <rect
        x={p.x}
        y={aY + p.y}
        width={CELL}
        height={CELL}
        fill={signedFill(v, scaleA)}
        class="cell"
        class:lit={i === sel.i}
      />
      <text x={p.x + CELL / 2} y={aY + p.y + CELL / 2 + 3.5} class="v" fill={textOn(v, scaleA)}>
        {fmt(v)}
      </text>
    {/each}
  {/each}

  <!-- B -->
  <text x={bX} y={bY - 10} class="cap">B · {K}&times;{N}</text>
  {#each B as row, k}
    {#each row as v, j}
      {@const p = cellXY(k, j)}
      <rect
        x={bX + p.x}
        y={bY + p.y}
        width={CELL}
        height={CELL}
        fill={signedFill(v, scaleA)}
        class="cell"
        class:lit={j === sel.j}
      />
      <text x={bX + p.x + CELL / 2} y={bY + p.y + CELL / 2 + 3.5} class="v" fill={textOn(v, scaleA)}>
        {fmt(v)}
      </text>
    {/each}
  {/each}

  <!-- C -->
  <text x={cX} y={cY - 10} class="cap">C = A B · {M}&times;{N}</text>
  {#each C as row, i}
    {#each row as v, j}
      {@const p = cellXY(i, j)}
      <rect
        x={cX + p.x}
        y={cY + p.y}
        width={CELL}
        height={CELL}
        fill={signedFill(v, scaleC)}
        class="cell out"
        class:sel={i === sel.i && j === sel.j}
        role="button"
        tabindex="0"
        aria-label="result cell {i},{j}"
        onpointerenter={() => {
          playing = false
          sel = { i, j }
        }}
      />
      <text x={cX + p.x + CELL / 2} y={cY + p.y + CELL / 2 + 3.5} class="v" fill={textOn(v, scaleC)}>
        {fmt(v)}
      </text>
    {/each}
  {/each}

  <!-- the expansion for the selected cell -->
  <g transform="translate(0,{aY + M * cw + 22})">
    <text x="0" y="0" class="cap">
      C[{sel.i},{sel.j}] = row {sel.i} of A · column {sel.j} of B
    </text>
    {#each terms as t, k}
      <text x={k * 108} y="24" class="term">
        <tspan fill={t.a >= 0 ? '#c0392b' : '#2c7fb8'}>{fmt(t.a)}</tspan>
        <tspan class="op">×</tspan>
        <tspan fill={t.b >= 0 ? '#c0392b' : '#2c7fb8'}>{fmt(t.b)}</tspan>
        {#if k < terms.length - 1}<tspan class="op">&nbsp;+</tspan>{/if}
      </text>
    {/each}
    <text x="0" y="48" class="term total">
      = {C[sel.i][sel.j].toFixed(2)}
    </text>
  </g>
</svg>

<div class="readouts">
  <div class="ro">
    <span class="k">output cells</span>
    <span class="v">{M} × {N} = {M * N}</span>
  </div>
  <div class="ro">
    <span class="k">each costs</span>
    <span class="v">{K} mul + {K - 1} add</span>
  </div>
  <div class="ro">
    <span class="k">total</span>
    <span class="v">{2 * M * N * K} FLOPs</span>
  </div>
  <div class="ro">
    <span class="k">in general</span>
    <span class="v">2·M·N·K</span>
  </div>
</div>

<p class="note">
  A matrix multiply is nothing but a grid of dot products: one per output cell, each pairing a row
  of A with a column of B. Everything a transformer does in bulk is this operation — which is why
  the cost of a model is so often just <b>2 × M × N × K</b> counted up over a few dozen of these.
</p>

<style>
  .cap {
    font-size: 11px;
    fill: rgba(0, 0, 0, 0.5);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .cell {
    stroke: rgba(0, 0, 0, 0.07);
  }

  .cell.lit {
    stroke: rgba(0, 0, 0, 0.55);
    stroke-width: 1.5;
  }

  .cell.out {
    cursor: pointer;
  }

  .cell.sel {
    stroke: #000;
    stroke-width: 2.5;
  }

  .v {
    font-size: 10.5px;
    text-anchor: middle;
    font-family: 'SF Mono', Menlo, Consolas, monospace;
    pointer-events: none;
  }

  .term {
    font-size: 12px;
    font-family: 'SF Mono', Menlo, Consolas, monospace;
  }

  .term .op {
    fill: rgba(0, 0, 0, 0.35);
  }

  .term.total {
    font-size: 14px;
    fill: rgba(0, 0, 0, 0.8);
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
