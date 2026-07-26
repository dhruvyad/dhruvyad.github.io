<script>
  import { scaleLinear } from 'd3-scale'
  import { softmax, entropyBits } from './softmax.js'

  let {
    labels = ['cat', 'dog', 'fox', 'owl', 'ibis'],
    logits = [3.1, 2.4, 1.0, 0.2, -0.8],
  } = $props()

  let temperature = $state(1)

  const probs = $derived(softmax(logits, temperature))
  const entropy = $derived(entropyBits(probs))
  const maxEntropy = $derived(Math.log2(labels.length))

  const width = 620
  const rowHeight = 34
  const margin = { top: 8, right: 56, bottom: 8, left: 52 }
  const innerWidth = width - margin.left - margin.right
  const height = $derived(labels.length * rowHeight + margin.top + margin.bottom)

  const x = scaleLinear().domain([0, 1]).range([0, innerWidth])

  // Presets worth landing on exactly — the interesting behaviour is at the extremes.
  const presets = [
    { label: 'T → 0', value: 0.1 },
    { label: 'T = 1', value: 1 },
    { label: 'T → ∞', value: 8 },
  ]
</script>

<div class="figure-controls">
  <label>
    Temperature <span class="value">{temperature.toFixed(2)}</span>
    <input type="range" min="0.1" max="8" step="0.05" bind:value={temperature} />
  </label>
  {#each presets as preset}
    <button onclick={() => (temperature = preset.value)}>{preset.label}</button>
  {/each}
</div>

<svg viewBox="0 0 {width} {height}" {width} style="max-width:100%;height:auto">
  <g transform="translate({margin.left},{margin.top})">
    {#each probs as p, i}
      <g transform="translate(0,{i * rowHeight})">
        <text x="-10" y={rowHeight / 2} dy="0.32em" text-anchor="end" class="label">
          {labels[i]}
        </text>
        <rect y="6" width={innerWidth} height={rowHeight - 12} class="track" />
        <rect
          y="6"
          width={Math.max(0, x(p))}
          height={rowHeight - 12}
          class="bar"
          class:top={p === Math.max(...probs)}
        />
        <text x={innerWidth + 8} y={rowHeight / 2} dy="0.32em" class="pct">
          {(p * 100).toFixed(1)}%
        </text>
      </g>
    {/each}
  </g>
</svg>

<div class="readout">
  Entropy <span class="value">{entropy.toFixed(2)}</span> of
  <span class="value">{maxEntropy.toFixed(2)}</span> bits
  <span class="entropy-track">
    <span class="entropy-fill" style="width:{(entropy / maxEntropy) * 100}%"></span>
  </span>
</div>

<style>
  .label {
    font-size: 13px;
    fill: rgba(0, 0, 0, 0.75);
  }

  .pct {
    font-size: 12px;
    font-variant-numeric: tabular-nums;
    font-family: 'SF Mono', Menlo, Consolas, monospace;
    fill: rgba(0, 0, 0, 0.55);
  }

  .track {
    fill: rgba(0, 0, 0, 0.05);
  }

  .bar {
    fill: #b8b8b8;
  }

  .bar.top {
    fill: #c0392b;
  }

  .readout {
    display: flex;
    align-items: center;
    gap: 0.6em;
    margin-top: 0.5em;
    font-size: 13px;
    color: rgba(0, 0, 0, 0.6);
  }

  .value {
    font-variant-numeric: tabular-nums;
    font-family: 'SF Mono', Menlo, Consolas, monospace;
    color: #000;
  }

  .entropy-track {
    flex: 1;
    max-width: 180px;
    height: 4px;
    background: rgba(0, 0, 0, 0.08);
    border-radius: 2px;
    overflow: hidden;
  }

  .entropy-fill {
    display: block;
    height: 100%;
    background: #c0392b;
  }
</style>
