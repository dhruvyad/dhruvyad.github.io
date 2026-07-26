<script>
  /**
   * The dot product, two ways: as the arithmetic (multiply pairwise, add it all
   * up) and as the geometry (how much two directions agree). Both readings matter
   * later — attention uses the second to decide what to read, and pays for the
   * first.
   */
  import { POS, NEG, MUTED, INK, fmt, dot } from './kit.js'

  let a = $state([0.9, -0.4, 1.2, 0.3, -1.1, 0.7, 0.2, -0.6])
  let b = $state([1.1, 0.5, 0.8, -0.9, -0.7, 0.4, 1.0, 0.1])
  /** How far through the running sum we have stepped; n = finished. */
  let step = $state(8)

  const n = 8
  const products = $derived(a.map((x, i) => x * b[i]))
  const partial = $derived(products.slice(0, step).reduce((s, x) => s + x, 0))
  const total = $derived(dot(a, b))

  const width = 704
  const height = 306
  const left = 74
  const colW = 52
  const unit = 26 // pixels per 1.0 of value
  const MAXV = 1.8
  const aBase = 84 // baseline for vector a
  const bBase = 192 // baseline for vector b
  const prodY = 240 // row of products

  const barX = (i) => left + i * colW
  const barW = colW - 14
  /** Bar rectangle for a value against a given baseline. */
  const bar = (v, base) => ({
    y: v >= 0 ? base - v * unit : base,
    h: Math.abs(v) * unit,
  })
  /** Value label just outside the bar, so it never sits on top of another. */
  const labelY = (v, base) => (v >= 0 ? base - v * unit - 5 : base + Math.abs(v) * unit + 12)

  /** Drag a bar to change the number it stands for. */
  let dragging = $state(null)
  function startDrag(which, i) {
    dragging = { which, i }
  }
  function onMove(event) {
    if (!dragging) return
    const rect = event.currentTarget.getBoundingClientRect()
    const localY = ((event.clientY - rect.top) / rect.height) * height
    const base = dragging.which === 'a' ? aBase : bBase
    const v = Math.max(-MAXV, Math.min(MAXV, (base - localY) / unit))
    const rounded = Math.round(v * 10) / 10
    if (dragging.which === 'a') a[dragging.i] = rounded
    else b[dragging.i] = rounded
  }
  const endDrag = () => (dragging = null)

  const randomise = () => {
    a = a.map(() => Math.round((Math.random() * 2 - 1) * 14) / 10)
    b = b.map(() => Math.round((Math.random() * 2 - 1) * 14) / 10)
    step = n
  }
  const align = () => {
    b = a.map((v) => Math.round(v * 10) / 10)
    step = n
  }
  const oppose = () => {
    b = a.map((v) => Math.round(-v * 10) / 10)
    step = n
  }
  const orthogonal = () => {
    // Rotate pairs by 90°, which zeroes the sum of products exactly.
    const nb = [...a]
    for (let i = 0; i + 1 < n; i += 2) {
      nb[i] = -a[i + 1]
      nb[i + 1] = a[i]
    }
    b = nb
    step = n
  }
</script>

<div class="figure-controls">
  <button onclick={align}>Same direction</button>
  <button onclick={orthogonal}>Unrelated</button>
  <button onclick={oppose}>Opposite</button>
  <button onclick={randomise}>Randomise</button>
  <label>
    Sum first <span class="value">{step}</span> terms
    <input type="range" min="0" max={n} step="1" bind:value={step} />
  </label>
</div>

<svg
  viewBox="0 0 {width} {height}"
  {width}
  style="max-width:100%;height:auto"
  onpointermove={onMove}
  onpointerup={endDrag}
  onpointerleave={endDrag}
  role="application"
  aria-label="Interactive dot product"
>
  <!-- one baseline per vector, so the two rows never collide -->
  <line x1={left - 12} x2={left + n * colW - 8} y1={aBase} y2={aBase} class="axis" />
  <line x1={left - 12} x2={left + n * colW - 8} y1={bBase} y2={bBase} class="axis" />

  <text x={left - 20} y={aBase + 5} text-anchor="end" class="lbl">a</text>
  <text x={left - 20} y={bBase + 5} text-anchor="end" class="lbl">b</text>
  <text x={left - 20} y={prodY + 12} text-anchor="end" class="lbl small">a·b</text>

  {#each a as v, i}
    {@const dim = i >= step}
    {@const ra = bar(v, aBase)}
    {@const rb = bar(b[i], bBase)}

    <rect
      x={barX(i)} y={ra.y} width={barW} height={ra.h}
      fill={v >= 0 ? POS : NEG} opacity={dim ? 0.2 : 0.95}
      class="grab" role="slider" tabindex="0"
      aria-label="a{i}" aria-valuenow={v}
      onpointerdown={() => startDrag('a', i)}
    />
    <text x={barX(i) + barW / 2} y={labelY(v, aBase)} class="num">{fmt(v)}</text>

    <rect
      x={barX(i)} y={rb.y} width={barW} height={rb.h}
      fill={b[i] >= 0 ? POS : NEG} opacity={dim ? 0.2 : 0.95}
      class="grab" role="slider" tabindex="0"
      aria-label="b{i}" aria-valuenow={b[i]}
      onpointerdown={() => startDrag('b', i)}
    />
    <text x={barX(i) + barW / 2} y={labelY(b[i], bBase)} class="num">{fmt(b[i])}</text>

    <!-- the product of the pair -->
    <rect
      x={barX(i)} y={prodY} width={barW} height={18}
      fill={products[i] >= 0 ? POS : NEG}
      opacity={dim ? 0.12 : 0.2 + 0.65 * Math.min(1, Math.abs(products[i]) / 1.6)}
    />
    <text x={barX(i) + barW / 2} y={prodY + 13} class="num prod" opacity={dim ? 0.3 : 1}>
      {fmt(products[i])}
    </text>
  {/each}

  <!-- running total -->
  <text x={left + n * colW + 6} y={prodY - 6} class="lbl small">sum</text>
  <text x={left + n * colW + 6} y={prodY + 16} class="total" fill={partial >= 0 ? POS : NEG}>
    {partial.toFixed(2)}
  </text>
</svg>

<div class="readouts">
  <div class="ro">
    <span class="k">a · b</span>
    <span class="v" style="color:{total >= 0 ? POS : NEG}">{total.toFixed(2)}</span>
  </div>
  <div class="ro">
    <span class="k">reading</span>
    <span class="v">
      {total > 0.8 ? 'pointing together' : total < -0.8 ? 'pointing apart' : 'largely unrelated'}
    </span>
  </div>
  <div class="ro">
    <span class="k">multiplies</span>
    <span class="v">{n}</span>
  </div>
  <div class="ro">
    <span class="k">adds</span>
    <span class="v">{n - 1}</span>
  </div>
</div>

<p class="note">
  Drag any bar to change a number. One dot product is {n} multiplies and {n - 1} adds, and it
  answers one question: do these two lists of numbers agree? Positive means they rise and fall
  together, near zero means they are unrelated, negative means one goes up where the other goes down.
  That question is the whole of attention.
</p>

<style>
  .axis {
    stroke: rgba(0, 0, 0, 0.35);
  }

  .axis.faint {
    stroke: rgba(0, 0, 0, 0.12);
  }

  .lbl {
    font-size: 15px;
    font-style: italic;
    fill: rgba(0, 0, 0, 0.75);
    font-family: Georgia, 'Times New Roman', serif;
  }

  .lbl.small {
    font-size: 11px;
    font-style: normal;
    font-family: inherit;
    fill: rgba(0, 0, 0, 0.45);
  }

  .num {
    font-size: 10.5px;
    text-anchor: middle;
    fill: rgba(0, 0, 0, 0.6);
    font-family: 'SF Mono', Menlo, Consolas, monospace;
    pointer-events: none;
  }

  .num.prod {
    fill: rgba(0, 0, 0, 0.75);
  }

  .total {
    font-size: 19px;
    font-family: 'SF Mono', Menlo, Consolas, monospace;
    font-variant-numeric: tabular-nums;
  }

  .grab {
    cursor: ns-resize;
  }

  .grab:hover {
    stroke: #000;
    stroke-width: 1.5;
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
