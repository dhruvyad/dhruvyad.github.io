<script>
  import { scaleLog } from 'd3-scale'
  import { line } from 'd3-shape'
  import { scoreMatrixBytes, flashTraffic, ON_CHIP, GPUS, GLM, fmtBytes } from './model.js'

  let logS = $state(13)
  const S = $derived(Math.round(2 ** logS))

  const width = 704
  const height = 320
  const margin = { top: 18, right: 176, bottom: 40, left: 88 }
  const innerWidth = width - margin.left - margin.right
  const innerHeight = height - margin.top - margin.bottom

  const x = scaleLog().domain([1024, 1048576]).range([0, innerWidth])
  const y = scaleLog().domain([1e4, 1e13]).range([innerHeight, 0])

  const samples = Array.from({ length: 140 }, (_, i) => 1024 * Math.pow(1024, i / 139))

  const naivePath = line()
    .x((s) => x(s))
    .y((s) => y(scoreMatrixBytes(s)))(samples)

  const flashPath = line()
    .x((s) => x(s))
    .y((s) => y(flashTraffic(s)))(samples)

  const levels = [
    { label: 'shared memory / SM (228 KiB)', bytes: ON_CHIP.smemPerSM },
    { label: 'L2 cache (50 MB)', bytes: ON_CHIP.l2 },
    { label: 'HBM on one H200 (141 GB)', bytes: GPUS['H200 SXM'].hbmGB * 1e9 },
  ]

  const naive = $derived(scoreMatrixBytes(S))
  const flash = $derived(flashTraffic(S))
  const xTicks = [1024, 8192, 65536, 524288]
  const fmtCtx = (n) => (n >= 1024 ? `${Math.round(n / 1024)}K` : `${n}`)
</script>

<div class="figure-controls">
  <label>
    Sequence length <span class="value">{fmtCtx(S)}</span>
    <input type="range" min="10" max="20" step="0.05" bind:value={logS} />
  </label>
</div>

<svg viewBox="0 0 {width} {height}" {width} style="max-width:100%;height:auto">
  <g transform="translate({margin.left},{margin.top})">
    {#each [1e4, 1e6, 1e8, 1e10, 1e12] as t}
      <g transform="translate(0,{y(t)})">
        <line x2={innerWidth} class="grid" />
        <text x="-8" dy="0.32em" text-anchor="end" class="tick">{fmtBytes(t)}</text>
      </g>
    {/each}
    {#each xTicks as t}
      <g transform="translate({x(t)},{innerHeight})">
        <line y2="5" class="axis" />
        <text y="18" text-anchor="middle" class="tick">{fmtCtx(t)}</text>
      </g>
    {/each}
    <text x={innerWidth / 2} y={innerHeight + 34} text-anchor="middle" class="axis-label">
      sequence length
    </text>
    <text transform="rotate(-90)" x={-innerHeight / 2} y="-70" text-anchor="middle" class="axis-label">
      bytes, one head, one layer
    </text>

    {#each levels as lv}
      <g transform="translate(0,{y(lv.bytes)})">
        <line x2={innerWidth} class="level" />
        <text x={innerWidth + 6} dy="0.32em" class="level-t">{lv.label}</text>
      </g>
    {/each}

    <path d={naivePath} class="naive" />
    <path d={flashPath} class="flash" />

    <line x1={x(S)} x2={x(S)} y1="0" y2={innerHeight} class="marker" />
    <circle cx={x(S)} cy={y(naive)} r="4.5" class="dot naive-d" />
    <circle cx={x(S)} cy={y(flash)} r="4.5" class="dot flash-d" />

    <text x={x(1048576) - 4} y={y(scoreMatrixBytes(1048576)) - 8} text-anchor="end" class="anno naive-t">
      score matrix, O(S²)
    </text>
    <text x={x(1048576) - 4} y={y(flashTraffic(1048576)) - 8} text-anchor="end" class="anno flash-t">
      FlashAttention, O(S)
    </text>
  </g>
</svg>

<div class="readouts">
  <div class="ro">
    <span class="k">score matrix</span>
    <span class="v">{fmtBytes(naive)}</span>
  </div>
  <div class="ro">
    <span class="k">flash traffic</span>
    <span class="v">{fmtBytes(flash)}</span>
  </div>
  <div class="ro">
    <span class="k">ratio</span>
    <span class="v">{(naive / flash).toFixed(0)}&times;</span>
  </div>
  <div class="ro">
    <span class="k">vs SMEM/SM</span>
    <span class="v">{(naive / ON_CHIP.smemPerSM).toFixed(0)}&times; too big</span>
  </div>
</div>

<p class="note">
  All {GLM.heads} heads across all {GLM.layers} layers would need
  {fmtBytes(naive * GLM.heads * GLM.layers)} of score matrices at this length. FlashAttention never
  writes them: it tiles the computation so each block lives in shared memory, and only Q, K, V and
  the output ever touch HBM.
</p>

<style>
  .grid {
    stroke: rgba(0, 0, 0, 0.06);
  }

  .axis {
    stroke: rgba(0, 0, 0, 0.3);
  }

  .tick,
  .axis-label,
  .anno,
  .level-t {
    font-size: 11px;
    fill: rgba(0, 0, 0, 0.5);
    font-variant-numeric: tabular-nums;
  }

  .level {
    stroke: rgba(0, 0, 0, 0.22);
    stroke-dasharray: 2 3;
  }

  .level-t {
    font-size: 10px;
    fill: rgba(0, 0, 0, 0.45);
  }

  .naive {
    fill: none;
    stroke: #c0392b;
    stroke-width: 2.5;
  }

  .flash {
    fill: none;
    stroke: #2c7fb8;
    stroke-width: 2.5;
  }

  .naive-t {
    fill: #c0392b;
  }

  .flash-t {
    fill: #2c7fb8;
  }

  .marker {
    stroke: rgba(0, 0, 0, 0.25);
    stroke-dasharray: 2 3;
  }

  .dot {
    stroke: #fff;
    stroke-width: 2;
  }

  .naive-d {
    fill: #c0392b;
  }

  .flash-d {
    fill: #2c7fb8;
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
