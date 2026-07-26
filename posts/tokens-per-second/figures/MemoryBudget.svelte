<script>
  import { scaleLinear } from 'd3-scale'
  import { GPUS, GLM, kvBytesPerToken, mhaBytesPerToken, fmtBytes } from './model.js'

  let wBytes = $state(1)
  let gpu = $state('H200 SXM')
  let nGpu = $state(8)
  let logS = $state(17) // context = 2^logS
  let logB = $state(3) // batch = 2^logB
  let attn = $state('MLA')

  const S = $derived(Math.round(2 ** logS))
  const B = $derived(Math.round(2 ** logB))

  const perToken = $derived(attn === 'MLA' ? kvBytesPerToken() : mhaBytesPerToken())
  const hbm = $derived(GPUS[gpu].hbmGB * nGpu * 1e9)
  const weights = $derived(GLM.totalParams * wBytes)
  const kv = $derived(perToken * S * B)
  const used = $derived(weights + kv)
  // Leave a little headroom for activations, workspace and fragmentation.
  const usable = $derived(hbm * 0.92)
  const maxBatch = $derived(Math.max(0, Math.floor((usable - weights) / (perToken * S))))
  const fits = $derived(used <= usable)

  const width = 660
  const height = 132
  const margin = { top: 26, right: 8, bottom: 8, left: 8 }
  const innerWidth = width - margin.left - margin.right
  const barHeight = 34

  // Scale to whichever is larger so overflow is visible rather than clipped.
  const domainMax = $derived(Math.max(hbm, used))
  const x = $derived(scaleLinear().domain([0, domainMax]).range([0, innerWidth]))

  const PRECISIONS = [
    { label: 'BF16', bytes: 2 },
    { label: 'FP8', bytes: 1 },
    { label: 'FP4', bytes: 0.5 },
  ]
  const fmtCtx = (n) => (n >= 1024 ? `${(n / 1024).toFixed(0)}K` : `${n}`)
</script>

<div class="figure-controls">
  <label>
    Context <span class="value">{fmtCtx(S)}</span>
    <input type="range" min="10" max="20" step="0.05" bind:value={logS} />
  </label>
  <label>
    Batch <span class="value">{B}</span>
    <input type="range" min="0" max="10" step="0.05" bind:value={logB} />
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
    <button class:active={nGpu === n} onclick={() => (nGpu = n)}>{n}&times;</button>
  {/each}
  <span class="sep"></span>
  {#each ['MLA', 'MHA'] as a}
    <button class:active={attn === a} onclick={() => (attn = a)}>{a}</button>
  {/each}
</div>

<svg viewBox="0 0 {width} {height}" {width} style="max-width:100%;height:auto">
  <g transform="translate({margin.left},{margin.top})">
    <text x="0" y="-10" class="axis-label">
      {nGpu}&times; {gpu} = {fmtBytes(hbm)} HBM
    </text>

    <!-- capacity track -->
    <rect width={x(hbm)} height={barHeight} class="capacity" />

    <rect width={Math.max(0, x(Math.min(weights, domainMax)))} height={barHeight} class="weights" />
    <rect
      x={x(weights)}
      width={Math.max(0, x(kv))}
      height={barHeight}
      class="kv"
      class:over={!fits}
    />

    <!-- capacity line -->
    <line x1={x(hbm)} x2={x(hbm)} y1="-4" y2={barHeight + 4} class="cap-line" />
    <line x1={x(usable)} x2={x(usable)} y1="0" y2={barHeight} class="usable-line" />

    <g transform="translate(0,{barHeight + 26})">
      <g>
        <rect width="9" height="9" y="-8" class="weights" />
        <text x="14" class="legend">weights · {fmtBytes(weights)}</text>
      </g>
      <g transform="translate(190,0)">
        <rect width="9" height="9" y="-8" class="kv" />
        <text x="14" class="legend">KV cache · {fmtBytes(kv)}</text>
      </g>
      <g transform="translate(390,0)">
        <text class="legend">
          {(100 * used / hbm).toFixed(0)}% of HBM
        </text>
      </g>
    </g>
  </g>
</svg>

<div class="readouts">
  <div class="ro">
    <span class="k">KV per token</span>
    <span class="v">{(perToken / 1024).toFixed(0)} KiB</span>
  </div>
  <div class="ro">
    <span class="k">KV per sequence</span>
    <span class="v">{fmtBytes(perToken * S)}</span>
  </div>
  <div class="ro">
    <span class="k">max concurrency</span>
    <span class="v" class:bad={maxBatch === 0}>{maxBatch}</span>
  </div>
  <div class="ro">
    <span class="k">fits</span>
    <span class="v" class:bad={!fits}>{fits ? 'yes' : 'no'}</span>
  </div>
</div>

<p class="note">
  {#if weights > usable}
    <span class="warn">The weights alone overflow this group.</span> Add GPUs or quantise further
    before worrying about KV.
  {:else if !fits}
    <span class="warn">Overflow.</span> {B} sequences of {fmtCtx(S)} tokens need
    {fmtBytes(kv)} of KV cache but only {fmtBytes(usable - weights)} is left after weights. The
    server would have to evict or queue.
  {:else}
    {fmtBytes(usable - weights)} remains after weights, so this configuration holds up to
    {maxBatch} concurrent sequences at {fmtCtx(S)} tokens.
    {#if attn === 'MHA'}
      Switch back to MLA to see what the compressed latent buys.
    {/if}
  {/if}
</p>

<style>
  .axis-label {
    font-size: 11px;
    fill: rgba(0, 0, 0, 0.55);
  }

  .capacity {
    fill: rgba(0, 0, 0, 0.06);
  }

  .weights {
    fill: #c0392b;
  }

  .kv {
    fill: #7f8c8d;
  }

  .kv.over {
    fill: #e67e22;
  }

  .cap-line {
    stroke: rgba(0, 0, 0, 0.6);
    stroke-width: 1.5;
  }

  .usable-line {
    stroke: rgba(0, 0, 0, 0.25);
    stroke-dasharray: 3 3;
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

  .ro .v.bad {
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
