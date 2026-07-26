<script>
  /**
   * Dataflow block diagrams in the style of the transformer paper's Figure 2 —
   * named operation boxes, arrows bottom to top, stacked cards for repetition —
   * except every box carries the real shapes, parameter counts and FLOPs, and
   * composite boxes open into their own diagram.
   *
   * Diagrams are data (see the specs passed in), not code: adding one is adding a
   * spec. A node with `expand` set drills into another spec; a node with `figure`
   * set points at one of the article's interactive figures.
   */
  let { diagrams, start = 'model', onFigure = null } = $props()

  let stack = $state([start])
  const id = $derived(stack[stack.length - 1])
  const spec = $derived(diagrams[id])

  const KINDS = {
    linear: { fill: '#e3ebe4', stroke: '#8aa48d' },
    matmul: { fill: '#d7d4ea', stroke: '#8b85bb' },
    softmax: { fill: '#d9ecd5', stroke: '#83ab7c' },
    mask: { fill: '#f4d8de', stroke: '#c4899a' },
    scale: { fill: '#fdf4c9', stroke: '#c9b96a' },
    concat: { fill: '#fdf4c9', stroke: '#c9b96a' },
    add: { fill: '#eeeeee', stroke: '#999' },
    norm: { fill: '#e8e6f3', stroke: '#9d97c0' },
    embed: { fill: '#e6dcef', stroke: '#a58bbd' },
    composite: { fill: '#cfd8ea', stroke: '#7d90b5' },
    moe: { fill: '#f6ddd2', stroke: '#c99378' },
    router: { fill: '#e4e4e4', stroke: '#8f8f8f' },
    terminal: { fill: 'none', stroke: 'none' },
  }

  const COL = 132
  const BOXW = 108
  const BOXH = 38

  const width = 704
  const rows = $derived(Math.max(...spec.nodes.map((n) => n.row)) + 1)
  /** Taller rows when any of them carries its metadata underneath. */
  const ROW = $derived(anyMetaBelow ? 98 : 74)
  const height = $derived(rows * ROW + 52)

  /** Grid → pixels. Row 0 is the bottom, so signal flows upward. */
  const cx = $derived((n) => width / 2 + (n.col ?? 0) * COL)
  const cy = $derived((n) => height - 30 - (n.row ?? 0) * ROW)
  const boxW = (n) => n.w ?? BOXW

  const byId = $derived(new Map(spec.nodes.map((n) => [n.id, n])))

  /** How many real boxes share each row — decides where metadata can go. */
  const rowCount = $derived.by(() => {
    const c = new Map()
    for (const n of spec.nodes) {
      if (n.kind === 'terminal') continue
      c.set(n.row, (c.get(n.row) ?? 0) + 1)
    }
    return c
  })
  const metaBelow = (n) => (rowCount.get(n.row) ?? 1) > 1 && (n.shape || n.cost)
  const anyMetaBelow = $derived(
    spec.nodes.some((n) => (rowCount.get(n.row) ?? 1) > 1 && (n.shape || n.cost)),
  )

  /**
   * Orthogonal route from one node's top to another's bottom: straight up when
   * they share a column, otherwise up, across, and in — which is how the paper
   * draws V bypassing into the second MatMul.
   */
  function route(a, b) {
    const x1 = cx(a)
    const y1 = cy(a) - (a.kind === 'terminal' ? 12 : BOXH / 2)
    const x2 = cx(b)
    const y2 = cy(b) + BOXH / 2
    if (Math.abs(x1 - x2) < 2) return `M${x1},${y1} L${x2},${y2}`
    // Clear the metadata band under the target, or the wire runs through the text.
    const mid = y2 + (metaBelow(b) ? 38 : 22)
    return `M${x1},${y1} L${x1},${mid} L${x2},${mid} L${x2},${y2}`
  }

  const enter = (n) => {
    if (n.figure && onFigure) return onFigure(n.figure)
    if (n.expand && diagrams[n.expand]) stack = [...stack, n.expand]
  }
  const up = (i) => (stack = stack.slice(0, i + 1))
</script>

<div class="figure-controls">
  {#each stack as s, i}
    {#if i > 0}<span class="sep">›</span>{/if}
    <button class="crumb" onclick={() => up(i)} disabled={i === stack.length - 1}>
      {diagrams[s].title}
    </button>
  {/each}
  <span class="spacer"></span>
  <span class="hint">click a shaded box to open it</span>
</div>

<svg viewBox="0 0 {width} {height}" {width} style="max-width:100%;height:auto">
  <defs>
    <marker id="bd-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,1 L9,5 L0,9 z" fill="#444" />
    </marker>
  </defs>

  <text x="0" y="14" class="title">{spec.title}</text>
  {#if spec.subtitle}<text x="0" y="30" class="subtitle">{spec.subtitle}</text>{/if}

  <!-- edges first, so boxes sit on top of them -->
  {#each spec.edges as e}
    {@const a = byId.get(e.from)}
    {@const b = byId.get(e.to)}
    {#if a && b}
      <path d={route(a, b)} class="edge" marker-end="url(#bd-arrow)" />
      {#if e.label}
        <text x={cx(a) + 7} y={(cy(a) + cy(b)) / 2 + 4} class="elabel">{e.label}</text>
      {/if}
    {/if}
  {/each}

  {#each spec.nodes as n}
    {@const x = cx(n)}
    {@const y = cy(n)}
    {#if n.kind === 'terminal'}
      <text {x} y={y + 5} class="terminal">{n.label}</text>
      {#if n.note}<text {x} y={y + 20} class="tnote">{n.note}</text>{/if}
    {:else}
      <g
        class="node"
        class:clickable={n.expand || n.figure}
        role={n.expand || n.figure ? 'button' : undefined}
        tabindex={n.expand || n.figure ? 0 : undefined}
        aria-label={n.label}
        onclick={() => enter(n)}
        onkeydown={(e) => (e.key === 'Enter' || e.key === ' ') && enter(n)}
      >
        <!-- stacked cards behind the box, for things that repeat -->
        {#if n.stack}
          {#each [3, 2, 1] as d}
            <rect
              x={x - boxW(n) / 2 + d * 5}
              y={y - BOXH / 2 - d * 5}
              width={boxW(n)}
              height={BOXH}
              rx="5"
              fill={KINDS[n.kind].fill}
              stroke={KINDS[n.kind].stroke}
              opacity={0.45 - d * 0.09}
            />
          {/each}
        {/if}
        <rect
          x={x - boxW(n) / 2}
          y={y - BOXH / 2}
          width={boxW(n)}
          height={BOXH}
          rx="5"
          fill={KINDS[n.kind].fill}
          stroke={KINDS[n.kind].stroke}
          stroke-width="1.4"
          class="boxrect"
        />
        <text {x} y={y + (n.note ? -1 : 4)} class="label">{n.label}</text>
        {#if n.note}<text {x} y={y + 12} class="note-in">{n.note}</text>{/if}

        {#if n.stack}
          <path
            d="M{x + boxW(n) / 2 + 20},{y - BOXH / 2 - 16} q10,0 10,10 q0,10 10,10"
            class="brace"
          />
          <text x={x + boxW(n) / 2 + 46} y={y - BOXH / 2 + 8} class="mult">×{n.stack}</text>
        {/if}

        <!-- shapes and cost: beside the box when it stands alone in its row,
             underneath when it shares the row and there is no space -->
        {#if n.shape || n.cost}
          {#if metaBelow(n)}
            <text {x} y={y + BOXH / 2 + 14} text-anchor="middle" class="meta">{n.shape ?? ''}</text>
            <text {x} y={y + BOXH / 2 + 25} text-anchor="middle" class="meta dim">{n.cost ?? ''}</text>
          {:else}
            <text x={x - boxW(n) / 2 - 10} y={y - 3} text-anchor="end" class="meta">{n.shape ?? ''}</text>
            <text x={x - boxW(n) / 2 - 10} y={y + 9} text-anchor="end" class="meta dim">{n.cost ?? ''}</text>
          {/if}
        {/if}
      </g>
    {/if}
  {/each}
</svg>

{#if spec.note}
  <p class="note">{@html spec.note}</p>
{/if}

<style>
  .title {
    font-size: 13px;
    font-weight: 600;
    fill: rgba(0, 0, 0, 0.8);
  }

  .subtitle {
    font-size: 11px;
    fill: rgba(0, 0, 0, 0.5);
  }

  .edge {
    fill: none;
    stroke: #444;
    stroke-width: 1.5;
  }

  .elabel {
    font-size: 10px;
    fill: rgba(0, 0, 0, 0.5);
    font-family: 'SF Mono', Menlo, Consolas, monospace;
  }

  .label {
    font-size: 12.5px;
    text-anchor: middle;
    fill: rgba(0, 0, 0, 0.85);
    pointer-events: none;
  }

  .note-in {
    font-size: 9.5px;
    text-anchor: middle;
    fill: rgba(0, 0, 0, 0.55);
    font-family: 'SF Mono', Menlo, Consolas, monospace;
    pointer-events: none;
  }

  .terminal {
    font-size: 14px;
    text-anchor: middle;
    fill: rgba(0, 0, 0, 0.75);
    font-family: Georgia, 'Times New Roman', serif;
    font-style: italic;
  }

  .tnote {
    font-size: 9.5px;
    text-anchor: middle;
    fill: rgba(0, 0, 0, 0.45);
    font-family: 'SF Mono', Menlo, Consolas, monospace;
  }

  .meta,
  .tnote {
    /* A white halo so a wire passing behind the label doesn't cut through it. */
    paint-order: stroke;
    stroke: #fff;
    stroke-width: 3px;
    stroke-linejoin: round;
  }

  .meta {
    font-size: 9.5px;
    fill: rgba(0, 0, 0, 0.6);
    font-family: 'SF Mono', Menlo, Consolas, monospace;
    pointer-events: none;
  }

  .meta.dim {
    fill: rgba(0, 0, 0, 0.4);
  }

  .brace {
    fill: none;
    stroke: rgba(0, 0, 0, 0.5);
    stroke-width: 1.2;
  }

  .mult {
    font-size: 12px;
    fill: #c0392b;
    font-family: 'SF Mono', Menlo, Consolas, monospace;
    font-weight: 600;
  }

  .node.clickable {
    cursor: pointer;
  }

  .node.clickable .boxrect {
    filter: drop-shadow(0 1px 0 rgba(0, 0, 0, 0.18));
  }

  .node.clickable:hover .boxrect {
    stroke-width: 2.4;
  }

  .crumb {
    font: inherit;
    font-size: 12.5px;
    padding: 0.2em 0.6em;
    border: 1px solid rgba(0, 0, 0, 0.2);
    border-radius: 3px;
    background: #fff;
    cursor: pointer;
  }

  .crumb:disabled {
    background: rgba(0, 0, 0, 0.04);
    color: rgba(0, 0, 0, 0.55);
    cursor: default;
  }

  .sep {
    color: rgba(0, 0, 0, 0.3);
  }

  .spacer {
    flex: 1;
  }

  .hint {
    font-size: 12px;
    color: rgba(0, 0, 0, 0.45);
    font-style: italic;
  }

  .note {
    font-size: 12px;
    line-height: 1.6em;
    color: rgba(0, 0, 0, 0.6);
    margin: 0.9em 0 0;
  }
</style>
