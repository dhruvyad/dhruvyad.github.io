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
    // Green is the non-linearity family: SoftMax and the pointwise activations
    // are the only places in a transformer where the function stops being linear,
    // so they get one colour and read as a class.
    softmax: { fill: '#d9ecd5', stroke: '#83ab7c' },
    act: { fill: '#c6e5bb', stroke: '#5f9152' },
    mask: { fill: '#f4d8de', stroke: '#c4899a' },
    scale: { fill: '#fdf4c9', stroke: '#c9b96a' },
    concat: { fill: '#fdf4c9', stroke: '#c9b96a' },
    add: { fill: '#eeeeee', stroke: '#999' },
    norm: { fill: '#e8e6f3', stroke: '#9d97c0' },
    embed: { fill: '#e6dcef', stroke: '#a58bbd' },
    rope: { fill: '#dfe7f2', stroke: '#8ba1bd' },
    composite: { fill: '#cfd8ea', stroke: '#7d90b5' },
    moe: { fill: '#f6ddd2', stroke: '#c99378' },
    router: { fill: '#e4e4e4', stroke: '#8f8f8f' },
    // leaf arithmetic — where the decomposition bottoms out
    elementwise: { fill: '#fbe6d4', stroke: '#c9a077' },
    reduce: { fill: '#d6e9ea', stroke: '#7fa6a9' },
    weight: { fill: '#f4f3f1', stroke: '#b6b2ab' },
    terminal: { fill: 'none', stroke: 'none' },
  }
  const DORMANT = { fill: '#efefee', stroke: '#c2c0bb' }

  const COL = 132
  const BOXW = 108
  const BOXH = 38

  const width = 704
  const rows = $derived(Math.max(...spec.nodes.map((n) => n.row)) + 1)
  /** Taller rows when any of them carries its metadata underneath. */
  const ROW = $derived(anyMetaBelow ? 98 : 74)
  /** Row 0 sits on the bottom margin, so metadata under it needs the room. */
  const PAD = $derived(spec.nodes.some((n) => n.row === 0 && metaBelow(n)) ? 52 : 30)
  const height = $derived(rows * ROW + 22 + PAD)

  /** Grid → pixels. Row 0 is the bottom, so signal flows upward. */
  const cx = $derived((n) => width / 2 + (n.col ?? 0) * COL)
  const cy = $derived((n) => height - PAD - (n.row ?? 0) * ROW)
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
  const metaBelow = (n) =>
    (n.shape || n.cost) && (n.meta === 'below' || (rowCount.get(n.row) ?? 1) > 1)
  const anyMetaBelow = $derived(spec.nodes.some(metaBelow))

  /** Half the drawn width of a node, so a wire can meet its edge and stop there. */
  const halfW = (n) => (n.kind === 'terminal' ? Math.max(13, n.label.length * 4.4) : boxW(n) / 2)
  const topY = (n) => cy(n) - (n.kind === 'terminal' ? 11 : BOXH / 2)
  // A terminal's shape note is drawn under its label, so an arrow arriving from
  // below has to stop under the note rather than run through it.
  const botY = (n) => cy(n) + (n.kind === 'terminal' ? (n.note ? 25 : 11) : BOXH / 2)

  /**
   * Orthogonal route from one node's top to another's bottom: straight up when
   * they share a column, otherwise up, across, and in — which is how the paper
   * draws V bypassing into the second MatMul.
   *
   * An edge may instead name a `lane`: a clear column to travel up. It leaves the
   * source through the side facing that lane (or through the top, if the lane is
   * its own column) and arrives on the side of the target rather than underneath
   * it. That is what keeps a second arrow into the same box — a residual, or a
   * router telling the experts which of them to run — from landing on top of the
   * first and looking like one wire.
   */
  function route(a, b, e) {
    const x1 = cx(a)
    const x2 = cx(b)
    if (e?.lane != null) {
      // Same row: nothing to route around, just meet edge to edge.
      if (Math.abs(cy(a) - cy(b)) < 2) {
        const d = Math.sign(x2 - x1) || 1
        return `M${x1 + d * halfW(a)},${cy(a)} L${x2 - d * halfW(b)},${cy(b)}`
      }
      const vx = width / 2 + e.lane * COL
      const out = vx - x1
      // `out: 'top'` forces the wire to leave through the top even when the lane
      // is off to one side — for a value that fans out, so it visibly branches
      // off the forward arrow instead of appearing to pass through the box.
      const fromTop = e.out === 'top' || Math.abs(out) < 2
      const sx = fromTop ? x1 : x1 + Math.sign(out) * halfW(a)
      const sy = fromTop ? topY(a) : cy(a)
      const into = x2 - vx
      const ex = Math.abs(into) < 2 ? x2 : x2 - Math.sign(into) * halfW(b)
      const ey = Math.abs(into) < 2 ? botY(b) : cy(b)
      return `M${sx},${sy} L${vx},${sy} L${vx},${ey} L${ex},${ey}`
    }
    const y1 = topY(a)
    const y2 = botY(b)
    if (Math.abs(x1 - x2) < 2) return `M${x1},${y1} L${x2},${y2}`
    // Clear the metadata band under the target, or the wire runs through the text.
    const mid = y2 + (metaBelow(b) ? 38 : 22)
    return `M${x1},${y1} L${x1},${mid} L${x2},${mid} L${x2},${y2}`
  }

  /** Edge labels sit against the lane, on the side away from the diagram's spine. */
  function labelAt(e, a, b) {
    if (e.lane == null) return { x: cx(a) + 7, y: (cy(a) + cy(b)) / 2 + 4, anchor: 'start' }
    const vx = width / 2 + e.lane * COL
    const outward = e.lane < 0 ? -1 : 1
    const y = (cy(a) + cy(b)) / 2 + 4
    // Prefer the outside of the lane, where nothing else is drawn — but fall back
    // inward rather than let a label run off the canvas.
    const w = (e.label?.length ?? 0) * 5.4
    if (outward < 0) return vx - 7 - w < 4 ? { x: vx + 7, y, anchor: 'start' } : { x: vx - 7, y, anchor: 'end' }
    return vx + 7 + w > width - 4 ? { x: vx - 7, y, anchor: 'end' } : { x: vx + 7, y, anchor: 'start' }
  }

  let hovered = $state(null)
  const explain = $derived(hovered ?? null)

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
    <marker id="bd-arrow-res" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,1 L9,5 L0,9 z" fill="#2c7fb8" />
    </marker>
    <marker id="bd-arrow-ctl" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,1 L9,5 L0,9 z" fill="#9a7b3f" />
    </marker>
  </defs>

  <text x="0" y="14" class="title">{spec.title}</text>
  {#if spec.subtitle}<text x="0" y="30" class="subtitle">{spec.subtitle}</text>{/if}

  <!-- edges first, so boxes sit on top of them -->
  {#each spec.edges as e}
    {@const a = byId.get(e.from)}
    {@const b = byId.get(e.to)}
    {#if a && b}
      <path
        d={route(a, b, e)}
        class="edge"
        class:residual={e.residual}
        class:control={e.control}
        marker-end="url({e.residual ? '#bd-arrow-res' : e.control ? '#bd-arrow-ctl' : '#bd-arrow'})"
      />
      {#if e.label}
        {@const p = labelAt(e, a, b)}
        <text
          x={p.x}
          y={p.y}
          text-anchor={p.anchor}
          class="elabel"
          class:res={e.residual}
          class:ctl={e.control}>{e.label}</text
        >
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
        onpointerenter={() => (hovered = n)}
        onpointerleave={() => (hovered = null)}
        onfocus={() => (hovered = n)}
        onblur={() => (hovered = null)}
      >
        <!-- Stacked cards behind the box, for things that repeat. When only some
             of the stack runs for a given token, the cards behind are drawn grey:
             the front card is the work, the pile behind it is the memory bill. -->
        {#if n.stack}
          {#each [3, 2, 1] as d}
            <rect
              x={x - boxW(n) / 2 + d * 5}
              y={y - BOXH / 2 - d * 5}
              width={boxW(n)}
              height={BOXH}
              rx="5"
              fill={n.active ? DORMANT.fill : KINDS[n.kind].fill}
              stroke={n.active ? DORMANT.stroke : KINDS[n.kind].stroke}
              opacity={n.active ? 0.9 - d * 0.12 : 0.45 - d * 0.09}
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
          stroke-dasharray={n.kind === 'weight' ? '4 3' : undefined}
          class="boxrect"
        />
        <text {x} y={y + (n.note ? -1 : 4)} class="label">{n.label}</text>
        {#if n.note}<text {x} y={y + 12} class="note-in">{n.note}</text>{/if}

        {#if n.stack}
          <path
            d="M{x + boxW(n) / 2 + 20},{y - BOXH / 2 - 16} q10,0 10,10 q0,10 10,10"
            class="brace"
          />
          <text x={x + boxW(n) / 2 + 46} y={y - BOXH / 2 + (n.active ? 2 : 8)} class="mult"
            >×{n.stack}</text
          >
          {#if n.active}
            <text x={x + boxW(n) / 2 + 46} y={y - BOXH / 2 + 15} class="active-note"
              >{n.active} run</text
            >
          {/if}
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

{#if explain?.why}
  <p class="why"><b>{explain.label}</b> — {explain.why}</p>
{:else if spec.note}
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

  /* Residual connections are drawn distinctly, because the point of them is
     that they route *around* the computation. */
  .edge.residual {
    stroke: #2c7fb8;
    stroke-width: 1.6;
    stroke-dasharray: 5 3;
  }

  /* Control signals — which experts to run, which keys to keep. They carry
     indices and gates, not activations, so they are drawn as a separate class. */
  .edge.control {
    stroke: #9a7b3f;
    stroke-width: 1.3;
    stroke-dasharray: 2 2.5;
  }

  .elabel.res {
    fill: #2c7fb8;
  }

  .elabel.ctl {
    fill: #9a7b3f;
  }

  .elabel {
    paint-order: stroke;
    stroke: #fff;
    stroke-width: 3px;
    stroke-linejoin: round;
  }

  .active-note {
    font-size: 9.5px;
    fill: #c0392b;
    font-family: 'SF Mono', Menlo, Consolas, monospace;
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

  .note,
  .why {
    font-size: 12px;
    line-height: 1.6em;
    color: rgba(0, 0, 0, 0.6);
    margin: 0.9em 0 0;
    min-height: 3.2em;
  }

  .why {
    border-left: 3px solid rgba(0, 0, 0, 0.18);
    padding-left: 0.8em;
    color: rgba(0, 0, 0, 0.72);
  }

  .why b {
    font-family: 'SF Mono', Menlo, Consolas, monospace;
    font-size: 0.95em;
  }
</style>
