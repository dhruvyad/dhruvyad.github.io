<script>
  import { scaleLog, scaleLinear } from 'd3-scale'
  import { line } from 'd3-shape'
  import {
    expertsTouched,
    residentParams,
    expertParams,
    sparseLayers,
    activeParams,
    GLM,
  } from './model.js'

  let logB = $state(4) // batch = 2^logB
  const B = $derived(Math.round(2 ** logB))

  /** Parameters whose weights must be read for a step of batch size B. */
  const paramsRead = (b) => residentParams + sparseLayers * expertsTouched(b) * expertParams

  const width = 660
  const height = 300
  const margin = { top: 18, right: 132, bottom: 40, left: 56 }
  const innerWidth = width - margin.left - margin.right
  const innerHeight = height - margin.top - margin.bottom

  const x = scaleLog().domain([1, 1024]).range([0, innerWidth])
  const y = scaleLinear().domain([0, 800e9]).range([innerHeight, 0])

  const samples = Array.from({ length: 160 }, (_, i) => 1 * Math.pow(1024, i / 159))

  const moePath = line()
    .x((b) => x(b))
    .y((b) => y(paramsRead(b)))(samples)

  // A dense model with the same active parameter count reads the same bytes at
  // every batch size — that flat line is what MoE gives up.
  const densePath = line()
    .x((b) => x(b))
    .y(() => y(activeParams))(samples)

  const totalPath = line()
    .x((b) => x(b))
    .y(() => y(GLM.totalParams))(samples)

  const touched = $derived(expertsTouched(B))
  const read = $derived(paramsRead(B))
  const ticks = [1, 4, 16, 64, 256, 1024]
</script>

<div class="figure-controls">
  <label>
    Batch <span class="value">{B}</span>
    <input type="range" min="0" max="10" step="0.05" bind:value={logB} />
  </label>
  <span class="hint">
    {touched.toFixed(0)} of {GLM.experts} experts active per layer
  </span>
</div>

<svg viewBox="0 0 {width} {height}" {width} style="max-width:100%;height:auto">
  <g transform="translate({margin.left},{margin.top})">
    {#each y.ticks(5) as t}
      <g transform="translate(0,{y(t)})">
        <line x2={innerWidth} class="grid" />
        <text x="-8" dy="0.32em" text-anchor="end" class="tick">{t / 1e9}B</text>
      </g>
    {/each}
    {#each ticks as t}
      <g transform="translate({x(t)},{innerHeight})">
        <line y2="5" class="axis" />
        <text y="18" text-anchor="middle" class="tick">{t}</text>
      </g>
    {/each}
    <text x={innerWidth / 2} y={innerHeight + 34} text-anchor="middle" class="axis-label">
      batch size (concurrent sequences)
    </text>
    <text
      transform="rotate(-90)"
      x={-innerHeight / 2}
      y="-42"
      text-anchor="middle"
      class="axis-label"
    >
      parameters read per step
    </text>

    <path d={totalPath} class="ceiling" />
    <text x={innerWidth + 6} y={y(GLM.totalParams)} dy="0.32em" class="anno ceiling-t">
      753B total
    </text>

    <path d={densePath} class="dense" />
    <text x={innerWidth + 6} y={y(activeParams) + 4} dy="0.32em" class="anno dense-t">
      40B if dense
    </text>

    <path d={moePath} class="moe" />
    <!-- Sits in the empty upper-left rather than at the curve's right end, where
         it would collide with the 753B ceiling label. -->
    <text x="8" y="14" class="anno moe-t">GLM-5.2 MoE</text>

    <line x1={x(B)} x2={x(B)} y1="0" y2={innerHeight} class="marker" />
    <circle cx={x(B)} cy={y(read)} r="5" class="dot" />
  </g>
</svg>

<div class="readouts">
  <div class="ro">
    <span class="k">experts touched</span>
    <span class="v">{touched.toFixed(1)} / {GLM.experts}</span>
  </div>
  <div class="ro">
    <span class="k">params read</span>
    <span class="v">{(read / 1e9).toFixed(0)}B</span>
  </div>
  <div class="ro">
    <span class="k">vs batch 1</span>
    <span class="v">{(read / paramsRead(1)).toFixed(1)}&times;</span>
  </div>
  <div class="ro">
    <span class="k">per token</span>
    <span class="v">{(read / B / 1e9).toFixed(1)}B</span>
  </div>
</div>

<style>
  .grid {
    stroke: rgba(0, 0, 0, 0.07);
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

  .moe {
    fill: none;
    stroke: #c0392b;
    stroke-width: 2.5;
  }

  .dense {
    fill: none;
    stroke: #7f8c8d;
    stroke-width: 1.5;
    stroke-dasharray: 5 4;
  }

  .ceiling {
    fill: none;
    stroke: rgba(0, 0, 0, 0.35);
    stroke-width: 1;
    stroke-dasharray: 2 3;
  }

  .moe-t {
    fill: #c0392b;
  }

  .dense-t {
    fill: #7f8c8d;
  }

  .ceiling-t {
    fill: rgba(0, 0, 0, 0.45);
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

  .hint {
    font-size: 12px;
    color: rgba(0, 0, 0, 0.5);
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
</style>
