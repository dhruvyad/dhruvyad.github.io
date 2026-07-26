<script>
  import { scaleLinear } from 'd3-scale'
  import { line } from 'd3-shape'
  import { acceptedLength, specSpeedup } from './model.js'

  let alpha = $state(0.8)
  let k = $state(5)
  let logB = $state(6) // serving batch = 2^logB (64 — a realistic serving point)
  /** Cost of one draft pass as a fraction of a full forward pass. */
  let draftCost = $state(0.08)

  const B = $derived(Math.round(2 ** logB))
  const r = $derived(specSpeedup({ alpha, k, B, draftCost }))

  const width = 660
  const height = 290
  const margin = { top: 18, right: 96, bottom: 40, left: 62 }
  const innerWidth = width - margin.left - margin.right
  const innerHeight = height - margin.top - margin.bottom

  const x = scaleLinear().domain([0, 8]).range([0, innerWidth])
  const y = scaleLinear().domain([0.5, 4]).range([innerHeight, 0])

  const ks = Array.from({ length: 33 }, (_, i) => (i / 32) * 8)
  const clamp = (v) => Math.max(0.5, Math.min(4, v))

  const speedCurve = (a, b) =>
    line()
      .x((kk) => x(kk))
      .y((kk) => y(clamp(specSpeedup({ alpha: a, k: kk, B: b, draftCost }).speedup)))(ks)

  const live = $derived(speedCurve(alpha, B))
  const GHOSTS = $derived([0.6, 0.7, 0.9].map((a) => ({ a, d: speedCurve(a, B) })))

  // Best integer draft length at the current settings.
  const bestK = $derived(
    Array.from({ length: 8 }, (_, i) => i + 1).reduce((best, kk) =>
      specSpeedup({ alpha, k: kk, B, draftCost }).speedup >
      specSpeedup({ alpha, k: best, B, draftCost }).speedup
        ? kk
        : best,
    ),
  )
</script>

<div class="figure-controls">
  <label>
    Acceptance α <span class="value">{alpha.toFixed(2)}</span>
    <input type="range" min="0.3" max="0.97" step="0.01" bind:value={alpha} />
  </label>
  <label>
    Draft length k <span class="value">{k}</span>
    <input type="range" min="1" max="8" step="1" bind:value={k} />
  </label>
  <label>
    Serving batch <span class="value">{B}</span>
    <input type="range" min="0" max="10" step="0.05" bind:value={logB} />
  </label>
  <label>
    Draft cost <span class="value">{(draftCost * 100).toFixed(0)}%</span>
    <input type="range" min="0.01" max="0.4" step="0.01" bind:value={draftCost} />
  </label>
</div>

<svg viewBox="0 0 {width} {height}" {width} style="max-width:100%;height:auto">
  <g transform="translate({margin.left},{margin.top})">
    {#each y.ticks(6) as t}
      <g transform="translate(0,{y(t)})">
        <line x2={innerWidth} class="grid" class:unity={t === 1} />
        <text x="-8" dy="0.32em" text-anchor="end" class="tick">{t}&times;</text>
      </g>
    {/each}
    {#each x.ticks(8) as t}
      <g transform="translate({x(t)},{innerHeight})">
        <line y2="5" class="axis" />
        <text y="18" text-anchor="middle" class="tick">{t}</text>
      </g>
    {/each}
    <text x={innerWidth / 2} y={innerHeight + 34} text-anchor="middle" class="axis-label">
      draft length k
    </text>
    <text transform="rotate(-90)" x={-innerHeight / 2} y="-48" text-anchor="middle" class="axis-label">
      net speedup
    </text>

    <text x="6" y="12" class="anno live-t">α = {alpha.toFixed(2)}, batch {B}</text>

    {#each GHOSTS as gh}
      <path d={gh.d} class="ghost" />
    {/each}
    <text x="6" y="26" class="anno">grey: α = 0.6, 0.7, 0.9</text>

    <path d={live} class="live" />
    <line x1={x(k)} x2={x(k)} y1="0" y2={innerHeight} class="marker" />
    <circle cx={x(k)} cy={y(clamp(r.speedup))} r="5" class="dot" />
  </g>
</svg>

<div class="readouts">
  <div class="ro">
    <span class="k">tokens / step</span>
    <span class="v">{r.tokens.toFixed(2)}</span>
  </div>
  <div class="ro">
    <span class="k">verify cost</span>
    <span class="v">{r.verifyRatio.toFixed(2)}&times;</span>
  </div>
  <div class="ro">
    <span class="k">net speedup</span>
    <span class="v" class:good={r.speedup > 1.2} class:bad={r.speedup < 1}>
      {r.speedup.toFixed(2)}&times;
    </span>
  </div>
  <div class="ro">
    <span class="k">best k here</span>
    <span class="v">{bestK}</span>
  </div>
</div>

<p class="note">
  {#if r.verifyRatio > 2}
    At batch {B}, verifying {k + 1} positions costs <b>{r.verifyRatio.toFixed(2)}&times;</b> a
    single-position step — the extra positions route to experts nobody was reading yet, so the
    weight traffic genuinely grows. Speculation has to beat that factor to pay for itself.
  {:else}
    At batch {B} the experts are already nearly all being read, so verifying {k + 1} positions costs
    only <b>{r.verifyRatio.toFixed(2)}&times;</b> a single-position step. This is the regime where
    speculation is close to free.
  {/if}
  Drag the batch slider to watch the verify cost collapse.
</p>

<style>
  .grid {
    stroke: rgba(0, 0, 0, 0.07);
  }

  .grid.unity {
    stroke: rgba(0, 0, 0, 0.28);
  }

  .axis {
    stroke: rgba(0, 0, 0, 0.3);
  }

  .tick,
  .axis-label,
  .anno {
    font-size: 11px;
    fill: rgba(0, 0, 0, 0.5);
    font-variant-numeric: tabular-nums;
  }

  .ghost {
    fill: none;
    stroke: rgba(0, 0, 0, 0.18);
    stroke-width: 1.25;
  }

  .live {
    fill: none;
    stroke: #c0392b;
    stroke-width: 2.5;
  }

  .live-t {
    fill: #c0392b;
  }

  .marker {
    stroke: rgba(0, 0, 0, 0.25);
    stroke-dasharray: 2 3;
  }

  .dot {
    fill: #c0392b;
    stroke: #fff;
    stroke-width: 2;
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

  .ro .v.good {
    color: #c0392b;
  }

  .ro .v.bad {
    color: rgba(0, 0, 0, 0.4);
  }

  .note {
    font-size: 12px;
    line-height: 1.6em;
    color: rgba(0, 0, 0, 0.6);
    margin: 0.9em 0 0;
  }
</style>
