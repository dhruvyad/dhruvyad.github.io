<script>
  import { untrack } from 'svelte'
  import { scaleLinear, scalePoint } from 'd3-scale'
  import { select } from 'd3-selection'
  import { drag } from 'd3-drag'
  import { softmax, clamp } from './softmax.js'

  let {
    labels = ['cat', 'dog', 'fox', 'owl'],
    initial = [3.1, 2.4, 1.0, 0.2],
  } = $props()

  // `initial` seeds the drag state once and is deliberately not tracked after
  // that — the dots are user-owned from mount onward.
  /** The logits you actually control by dragging. */
  let base = $state(untrack(() => [...initial]))
  /** A constant added to every logit at once. */
  let shift = $state(0)

  const shifted = $derived(base.map((z) => z + shift))
  // Computed from the *shifted* logits, so the invariance below is demonstrated
  // rather than asserted: move the shift slider and these numbers do not budge.
  const probs = $derived(softmax(shifted, 1))

  const width = 620
  const height = 260
  const margin = { top: 16, right: 52, bottom: 28, left: 46 }
  const innerHeight = height - margin.top - margin.bottom
  const axisWidth = 210
  const barsLeft = 300
  const barsWidth = width - margin.left - margin.right - barsLeft

  const LOGIT_RANGE = [-6, 6]

  const y = scaleLinear().domain(LOGIT_RANGE).range([innerHeight, 0])
  const xClass = $derived(scalePoint().domain(labels).range([20, axisWidth]).padding(0.5))
  const xProb = scaleLinear().domain([0, 1]).range([0, barsWidth])
  const barHeight = 26
  const barGap = 12

  let handles = $state([])

  $effect(() => {
    handles.forEach((node, i) => {
      if (!node) return
      select(node).call(
        drag().on('drag', (event) => {
          // event.y is in the coordinate space of the handle's parent <g>, so
          // inverting the y scale gives a logit value directly.
          const value = clamp(y.invert(event.y), LOGIT_RANGE[0], LOGIT_RANGE[1])
          base[i] = value - shift
        }),
      )
    })
  })

  const reset = () => {
    base = [...initial]
    shift = 0
  }
</script>

<div class="figure-controls">
  <label>
    Shift all logits by <span class="value">{shift >= 0 ? '+' : ''}{shift.toFixed(2)}</span>
    <input type="range" min="-4" max="4" step="0.05" bind:value={shift} />
  </label>
  <button onclick={reset}>Reset</button>
  <span class="hint">Drag the dots</span>
</div>

<svg viewBox="0 0 {width} {height}" {width} style="max-width:100%;height:auto">
  <g transform="translate({margin.left},{margin.top})">
    <!-- Logit axis -->
    {#each y.ticks(5) as tick}
      <g transform="translate(0,{y(tick)})">
        <line x1="8" x2={axisWidth} class="grid" />
        <text x="0" dy="0.32em" text-anchor="end" class="tick">{tick}</text>
      </g>
    {/each}
    <line x1="8" x2="8" y1="0" y2={innerHeight} class="axis" />
    <!-- Sits above the axis so it can't collide with the class labels below. -->
    <text x="8" y="-6" class="axis-label">logit</text>

    <!-- Draggable logit handles -->
    {#each shifted as z, i}
      <g class="handle-group">
        <line x1={xClass(labels[i])} x2={xClass(labels[i])} y1={y(z)} y2={innerHeight} class="stem" />
        <circle
          bind:this={handles[i]}
          cx={xClass(labels[i])}
          cy={y(z)}
          r="8"
          class="handle"
          class:top={z === Math.max(...shifted)}
        />
        <text x={xClass(labels[i])} y={innerHeight + 22} text-anchor="middle" class="class-label">
          {labels[i]}
        </text>
      </g>
    {/each}

    <!-- Resulting probabilities -->
    <g transform="translate({barsLeft},{(innerHeight - labels.length * (barHeight + barGap)) / 2})">
      <text x="0" y="-10" class="axis-label">softmax(z)</text>
      {#each probs as p, i}
        <g transform="translate(0,{i * (barHeight + barGap)})">
          <text x="-8" y={barHeight / 2} dy="0.32em" text-anchor="end" class="class-label">
            {labels[i]}
          </text>
          <rect width={barsWidth} height={barHeight} class="track" />
          <rect
            width={Math.max(0, xProb(p))}
            height={barHeight}
            class="bar"
            class:top={p === Math.max(...probs)}
          />
          <text x={barsWidth + 8} y={barHeight / 2} dy="0.32em" class="pct">
            {(p * 100).toFixed(1)}%
          </text>
        </g>
      {/each}
    </g>
  </g>
</svg>

<style>
  .grid {
    stroke: rgba(0, 0, 0, 0.07);
  }

  .axis {
    stroke: rgba(0, 0, 0, 0.25);
  }

  .tick,
  .class-label,
  .axis-label {
    font-size: 11px;
    fill: rgba(0, 0, 0, 0.5);
    font-variant-numeric: tabular-nums;
  }

  .class-label {
    font-size: 12px;
    fill: rgba(0, 0, 0, 0.7);
  }

  .stem {
    stroke: rgba(0, 0, 0, 0.14);
    stroke-dasharray: 2 3;
  }

  .handle {
    fill: #b8b8b8;
    stroke: #fff;
    stroke-width: 2;
    cursor: ns-resize;
  }

  .handle:hover {
    fill: #8c8c8c;
  }

  .handle.top {
    fill: #c0392b;
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

  .pct {
    font-size: 12px;
    font-family: 'SF Mono', Menlo, Consolas, monospace;
    font-variant-numeric: tabular-nums;
    fill: rgba(0, 0, 0, 0.55);
  }

  .hint {
    font-size: 12px;
    color: rgba(0, 0, 0, 0.45);
    font-style: italic;
  }

  .value {
    font-variant-numeric: tabular-nums;
    font-family: 'SF Mono', Menlo, Consolas, monospace;
    color: #000;
  }
</style>
