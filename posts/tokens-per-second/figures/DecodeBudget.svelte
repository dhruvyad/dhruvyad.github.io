<script>
  import { scaleLinear } from 'd3-scale'
  import { decodeStep, machineBalance, GPUS, GLM, fmtBytes } from './model.js'

  // Sliders work in log2 space so a single drag covers batch 1 -> 1024.
  let logB = $state(0) // batch = 2^logB
  let logS = $state(13) // context = 2^logS
  let wBytes = $state(1) // fp8
  let gpu = $state('H200 SXM')
  let nGpu = $state(8)
  let mbu = $state(1)

  const B = $derived(Math.round(2 ** logB))
  const S = $derived(Math.round(2 ** logS))
  const r = $derived(decodeStep({ B, S, wBytes, gpu, nGpu, mbu }))

  const balance = $derived(machineBalance(gpu))
  const hbmTotal = $derived(GPUS[gpu].hbmGB * nGpu * 1e9)
  const weightsFit = $derived(GLM.totalParams * wBytes < hbmTotal)

  const PRECISIONS = [
    { label: 'BF16', bytes: 2 },
    { label: 'FP8', bytes: 1 },
    { label: 'FP4', bytes: 0.5 },
  ]

  const width = 704
  const barHeight = 30
  const height = 118
  const margin = { top: 22, right: 8, bottom: 8, left: 8 }
  const innerWidth = width - margin.left - margin.right

  const x = $derived(scaleLinear().domain([0, r.bytes]).range([0, innerWidth]))

  const segments = $derived([
    { key: 'weights', label: 'weights', value: r.weightBytes, fill: '#c0392b' },
    { key: 'kv', label: 'KV cache', value: r.kvRead, fill: '#7f8c8d' },
    { key: 'idx', label: 'indexer', value: r.indexerRead, fill: '#bdc3c7' },
  ])

  // Cumulative offsets for the stacked bar.
  const stacked = $derived(
    segments.reduce((acc, s) => {
      const prev = acc.length ? acc[acc.length - 1] : { x1: 0 }
      acc.push({ ...s, x0: prev.x1 ?? 0, x1: (prev.x1 ?? 0) + s.value })
      return acc
    }, []),
  )

  const fmtCtx = (n) => (n >= 1024 ? `${(n / 1024).toFixed(0)}K` : `${n}`)
</script>

<div class="figure-controls">
  <label>
    Batch <span class="value">{B}</span>
    <input type="range" min="0" max="10" step="0.05" bind:value={logB} />
  </label>
  <label>
    Context <span class="value">{fmtCtx(S)}</span>
    <input type="range" min="10" max="20" step="0.05" bind:value={logS} />
  </label>
  <label>
    MBU <span class="value">{(mbu * 100).toFixed(0)}%</span>
    <input type="range" min="0.1" max="1" step="0.01" bind:value={mbu} />
  </label>
</div>

<div class="figure-controls">
  {#each PRECISIONS as p}
    <button class:active={wBytes === p.bytes} onclick={() => (wBytes = p.bytes)}>{p.label}</button>
  {/each}
  <span class="sep"></span>
  {#each Object.keys(GPUS) as g}
    <button class:active={gpu === g} onclick={() => (gpu = g)}>{g}</button>
  {/each}
  <span class="sep"></span>
  {#each [4, 8, 16] as n}
    <button class:active={nGpu === n} onclick={() => (nGpu = n)}>{n} GPU</button>
  {/each}
</div>

<svg viewBox="0 0 {width} {height}" {width} style="max-width:100%;height:auto">
  <g transform="translate({margin.left},{margin.top})">
    <text x="0" y="-8" class="axis-label">
      HBM traffic per decode step — {fmtBytes(r.bytes)}
    </text>
    {#each stacked as s}
      {#if s.value > 0}
        <rect x={x(s.x0)} width={Math.max(0, x(s.value))} height={barHeight} fill={s.fill} />
      {/if}
    {/each}
    <!-- Label segments wide enough to hold text; annotate the rest in the legend. -->
    {#each stacked as s}
      {#if x(s.value) > 62}
        <text x={x(s.x0) + 6} y={barHeight / 2} dy="0.32em" class="seg-label">
          {s.label} {((100 * s.value) / r.bytes).toFixed(0)}%
        </text>
      {/if}
    {/each}

    <g transform="translate(0,{barHeight + 24})">
      {#each stacked as s, i}
        <g transform="translate({i * 150},0)">
          <rect width="9" height="9" y="-8" fill={s.fill} />
          <text x="14" class="legend">{s.label} · {fmtBytes(s.value)}</text>
        </g>
      {/each}
    </g>
  </g>
</svg>

<div class="readouts">
  <div class="ro">
    <span class="k">step time</span>
    <span class="v">{r.t * 1e3 < 10 ? (r.t * 1e3).toFixed(2) : (r.t * 1e3).toFixed(1)} ms</span>
  </div>
  <div class="ro">
    <span class="k">per user</span>
    <span class="v">{r.perUser.toFixed(0)} tok/s</span>
  </div>
  <div class="ro">
    <span class="k">aggregate</span>
    <span class="v">{r.aggregate < 10000 ? r.aggregate.toFixed(0) : (r.aggregate / 1000).toFixed(1) + 'k'} tok/s</span>
  </div>
  <div class="ro">
    <span class="k">intensity</span>
    <span class="v">{r.intensity.toFixed(1)} F/B</span>
  </div>
  <div class="ro">
    <span class="k">bound by</span>
    <span class="v" class:mem={r.bound === 'memory'}>{r.bound}</span>
  </div>
</div>

<p class="note">
  {#if !weightsFit}
    <span class="warn">Weights alone need {fmtBytes(GLM.totalParams * wBytes)} but this group only
    has {fmtBytes(hbmTotal)} of HBM — this configuration cannot hold the model.</span>
  {:else}
    Machine balance for {gpu} is {balance.toFixed(0)} FLOP/byte. At intensity
    {r.intensity.toFixed(1)} the step is {r.bound}-bound, so the only thing that sets throughput is
    how many bytes have to cross HBM.
  {/if}
</p>

<style>
  .axis-label {
    font-size: 11px;
    fill: rgba(0, 0, 0, 0.55);
  }

  .seg-label {
    font-size: 11px;
    fill: #fff;
    font-family: 'SF Mono', Menlo, Consolas, monospace;
  }

  .legend {
    font-size: 11px;
    fill: rgba(0, 0, 0, 0.6);
  }

  .readouts {
    display: flex;
    flex-wrap: wrap;
    gap: 0 1.75em;
    margin-top: 0.25em;
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

  .ro .v.mem {
    color: #c0392b;
  }

  .note {
    font-size: 12px;
    line-height: 1.6em;
    color: rgba(0, 0, 0, 0.6);
    margin: 0.9em 0 0;
  }

  .warn {
    color: #c0392b;
  }

  .sep {
    width: 1px;
    height: 18px;
    background: rgba(0, 0, 0, 0.15);
  }

  button.active {
    background: #c0392b;
    color: #fff;
    border-color: #c0392b;
  }
</style>
