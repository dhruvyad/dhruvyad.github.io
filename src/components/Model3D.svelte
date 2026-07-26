<script>
  /**
   * A 3D, orbitable view of a transformer architecture, built from a Hugging Face
   * config.json. Blocks are sized by their real matrix dimensions and coloured by
   * either role or activation, so you can see which parts a single token actually
   * puts to work and which merely sit in memory.
   *
   * Shared by the article and the model-explorer tool — it takes the config as a
   * prop and knows nothing about either.
   */
  import {
    Scene,
    PerspectiveCamera,
    WebGLRenderer,
    BoxGeometry,
    MeshLambertMaterial,
    InstancedMesh,
    Object3D,
    Color,
    AmbientLight,
    DirectionalLight,
    Raycaster,
    Vector2,
    Vector3,
    Fog,
    DynamicDrawUsage,
  } from 'three'
  import { untrack } from 'svelte'
  import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
  import { parseArchitecture, litExperts, ROLES } from '../lib/architecture.js'

  let { config, height = 460, wBytes = 1, label = '' } = $props()

  const arch = $derived(parseArchitecture(config))
  const meta = $derived(arch.meta)
  const layers = $derived(arch.layers)
  const blocks = $derived(arch.blocks)

  let logB = $state(0)
  let colorMode = $state('activation')
  let selected = $state(null)
  let hovered = $state(null)
  let failed = $state(false)

  const B = $derived(Math.round(2 ** logB))
  const lit = $derived(meta.isMoE ? litExperts(B, meta.nExperts, meta.topk) : 0)

  /**
   * A fixed pseudo-random ranking of experts per layer. Lighting experts 0..n in
   * index order would draw a neat stripe across every panel, which is not what
   * routing looks like; ranking them first scatters the active ones the way a
   * real router would while staying identical between renders.
   */
  const expertRank = $derived.by(() => {
    if (!meta.isMoE) return new Map()
    const out = new Map()
    for (const l of layers) {
      let seed = ((l.index + 1) * 2654435761) >>> 0
      const rnd = () => {
        seed ^= seed << 13
        seed >>>= 0
        seed ^= seed >>> 17
        seed ^= seed << 5
        seed >>>= 0
        return seed / 4294967296
      }
      const order = Array.from({ length: meta.nExperts }, (_, i) => i)
      for (let i = order.length - 1; i > 0; i--) {
        const j = Math.floor(rnd() * (i + 1))
        ;[order[i], order[j]] = [order[j], order[i]]
      }
      const rank = new Array(meta.nExperts)
      order.forEach((e, r) => (rank[e] = r))
      out.set(l.index, rank)
    }
    return out
  })

  // ---------------------------------------------------------- role aggregates
  const roleStats = $derived.by(() => {
    const moeLayers = layers.filter((l) => !l.dense).length
    const expertParams = meta.isMoE ? 3 * meta.hidden * meta.moeFfn : 0
    const stats = new Map()
    for (const b of blocks) {
      const s = stats.get(b.role) ?? { role: b.role, count: 0, params: 0, active: 0, bytes: 0 }
      s.count++
      s.params += b.params
      s.active += b.activeParams
      stats.set(b.role, s)
    }
    for (const s of stats.values()) {
      if (s.role === 'routed_expert') {
        s.active = moeLayers * meta.topk * expertParams
        // Only the experts this batch happens to touch get read.
        s.bytes = moeLayers * lit * expertParams * wBytes
      } else if (s.role === 'embedding' || s.role === 'mtp') {
        // A gather, and a head that is idle unless speculating.
        s.bytes = 0
      } else {
        s.bytes = s.params * wBytes
      }
    }
    const vals = [...stats.values()]
    const totalParams = vals.reduce((a, s) => a + s.params, 0)
    const totalActive = vals.reduce((a, s) => a + s.active, 0)
    const totalBytes = vals.reduce((a, s) => a + s.bytes, 0)
    for (const s of vals) {
      s.paramShare = s.params / totalParams
      s.byteShare = totalBytes ? s.bytes / totalBytes : 0
    }
    return { stats, totalParams, totalActive, totalBytes }
  })

  // ------------------------------------------------------------------ layout
  const PITCH = 0.92
  const CELL = 0.62

  /** Box extent from a matrix dimension, log-scaled so 256 and 200k both fit. */
  const extent = (d) => {
    const t = (Math.log2(d) - Math.log2(256)) / (Math.log2(200000) - Math.log2(256))
    return 0.3 + 1.5 * Math.max(0, Math.min(1, t))
  }

  const xOf = (layer) => (layer + 1) * PITCH

  /** Slot of each attention projection within its layer's column. */
  const attnSlot = $derived.by(() => {
    const slot = new Map()
    const seen = new Map()
    for (const b of blocks) {
      if (b.role !== 'attention') continue
      const n = seen.get(b.layer) ?? 0
      slot.set(b.id, n)
      seen.set(b.layer, n + 1)
    }
    return slot
  })

  /** Square-ish expert grid, whatever the expert count. */
  const gridCols = $derived(meta.isMoE ? Math.ceil(Math.sqrt(meta.nExperts)) : 1)
  const gridHalf = $derived(((gridCols - 1) * CELL) / 2)

  function place(b) {
    const x = xOf(b.layer)
    if (b.role === 'routed_expert') {
      const cols = gridCols
      return {
        pos: [
          x,
          gridHalf - Math.floor(b.expert / cols) * CELL,
          -gridHalf + (b.expert % cols) * CELL,
        ],
        size: [0.44, 0.42, 0.42],
      }
    }
    if (b.role === 'attention') {
      const i = attnSlot.get(b.id) ?? 0
      return {
        pos: [x, gridHalf - 0.5 - i * 1.05, -gridHalf - 1.5],
        size: [0.55, extent(b.dims[0]) * 0.7, extent(b.dims[1]) * 0.7],
      }
    }
    if (b.role === 'indexer')
      return {
        pos: [x, gridHalf + 1.2, -gridHalf - 1.5],
        size: [0.55, extent(b.dims[1]) * 0.7, extent(b.dims[0]) * 0.7],
      }
    if (b.role === 'router')
      return { pos: [x, -gridHalf - 1.3, -gridHalf + 0.6], size: [0.55, 0.45, extent(b.dims[1]) * 0.8] }
    if (b.role === 'shared_expert')
      return {
        pos: [x, -gridHalf - 1.3, gridHalf - 1.2],
        size: [0.55, extent(b.dims[0]) * 0.8, extent(b.dims[1]) * 0.8],
      }
    if (b.role === 'dense_mlp')
      return { pos: [x, 0, 0], size: [0.55, extent(b.dims[0]) * 3.0, extent(b.dims[1]) * 3.0] }
    const s = b.role === 'mtp' ? 2.2 : 3.0
    return {
      pos: [b.role === 'embedding' ? -1.8 : x + 1.4, 0, 0],
      size: [0.7, extent(b.dims[0]) * s, extent(b.dims[1]) * s],
    }
  }

  const ROLE_ORDER = Object.keys(ROLES)
  const grouped = $derived(
    ROLE_ORDER.map((role) => ({ role, items: blocks.filter((b) => b.role === role) })).filter(
      (g) => g.items.length,
    ),
  )

  // --------------------------------------------------------------- rendering
  let host
  let renderer, scene, camera, controls, raycaster
  let meshes = new Map()
  let needsRender = true
  let visible = true
  let disposed = false

  const DORMANT = new Color(0xdedbd6)
  const ACTIVE = new Color(0xc0392b)
  const HILITE = new Color(0x101010)

  function isActive(b) {
    if (b.role === 'routed_expert') return (expertRank.get(b.layer)?.[b.expert] ?? 0) < lit
    return b.activeParams > 0
  }

  const colorFor = (b) =>
    colorMode === 'role' ? new Color(ROLES[b.role].color) : isActive(b) ? ACTIVE : DORMANT

  const scratch = new Object3D()

  function paint() {
    for (const { role, items } of grouped) {
      const mesh = meshes.get(role)
      if (!mesh) continue
      const varies = role === 'routed_expert'
      items.forEach((b, i) => {
        const focus = selected?.id === b.id || hovered?.id === b.id
        mesh.setColorAt(i, focus ? HILITE : colorFor(b))
        if (!varies) return
        // A wall of 19,200 identical cubes hides its own interior. Shrinking the
        // dormant ones turns each expert panel into a lattice you can see
        // through, so the few active experts read from any angle.
        const { pos, size } = place(b)
        const k = focus || isActive(b) || colorMode === 'role' ? 1 : 0.34
        scratch.position.set(pos[0], pos[1], pos[2])
        scratch.scale.set(size[0] * k, size[1] * k, size[2] * k)
        scratch.updateMatrix()
        mesh.setMatrixAt(i, scratch.matrix)
      })
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
      if (varies) mesh.instanceMatrix.needsUpdate = true
    }
    needsRender = true
  }

  const framing = () => Math.max(24, meta.layers * 0.55)
  /** Three-quarter view from near the input end, so the stack recedes. */
  const homeCamera = () => {
    const len = meta.layers * PITCH
    return new Vector3(-len * 0.22, gridHalf * 1.5, gridHalf * 3.4 + 6)
  }

  function build() {
    disposed = false
    scene = new Scene()
    scene.fog = new Fog(0xf3f2f0, framing() * 1.5, framing() * 4.5)

    const w = host.clientWidth || 700
    camera = new PerspectiveCamera(42, w / height, 0.1, 900)
    const home = homeCamera()
    camera.position.copy(home)

    // We render on demand rather than every frame, so the drawing buffer has to
    // survive compositing — otherwise the canvas can appear blank in any frame
    // where nothing was redrawn.
    renderer = new WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true })
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2))
    renderer.setSize(w, height, false)
    Object.assign(renderer.domElement.style, {
      width: '100%',
      height: `${height}px`,
      display: 'block',
    })
    host.appendChild(renderer.domElement)

    scene.add(new AmbientLight(0xffffff, 1.3))
    const key = new DirectionalLight(0xffffff, 1.5)
    key.position.set(1, 2, 1.6)
    scene.add(key)
    const fill = new DirectionalLight(0xffffff, 0.55)
    fill.position.set(-1, -0.6, -1)
    scene.add(fill)

    const geo = new BoxGeometry(1, 1, 1)
    const dummy = new Object3D()
    meshes = new Map()

    for (const { role, items } of grouped) {
      const mesh = new InstancedMesh(geo, new MeshLambertMaterial({ color: 0xffffff }), items.length)
      mesh.instanceMatrix.setUsage(DynamicDrawUsage)
      mesh.userData.role = role
      items.forEach((b, i) => {
        const { pos, size } = place(b)
        dummy.position.set(pos[0], pos[1], pos[2])
        dummy.scale.set(size[0], size[1], size[2])
        dummy.updateMatrix()
        mesh.setMatrixAt(i, dummy.matrix)
        mesh.setColorAt(i, DORMANT)
      })
      mesh.instanceMatrix.needsUpdate = true
      mesh.computeBoundingSphere()
      scene.add(mesh)
      meshes.set(role, mesh)
    }

    controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.08
    controls.target.set(meta.layers * PITCH * 0.42, 0, 0)
    controls.minDistance = 3
    controls.maxDistance = framing() * 5
    controls.addEventListener('change', () => (needsRender = true))
    controls.update()
    // Capture this as the home state *after* setting the target, so reset() has
    // something correct to go back to.
    controls.saveState()

    raycaster = new Raycaster()
    paint()
    loop()
  }

  function loop() {
    if (disposed) return
    requestAnimationFrame(loop)
    if (!visible || !renderer) return
    const moved = controls.update()
    if (moved || needsRender) {
      renderer.render(scene, camera)
      needsRender = false
      // Verification hook: proves geometry actually made it to the GPU.
      host.dataset.triangles = String(renderer.info.render.triangles)
      host.dataset.blocks = String(blocks.length)
      // Camera state, so "reset view" is observable from outside the component.
      host.dataset.cam = [camera.position.x, camera.position.y, camera.position.z]
        .map((n) => n.toFixed(1))
        .join(',')
    }
  }

  /** Throttled — testing 19,200 instanced boxes on every pointer move is wasteful. */
  let lastPick = 0
  function pick(event, commit) {
    if (!renderer) return
    const now = performance.now()
    if (!commit && now - lastPick < 60) return
    lastPick = now

    const rect = renderer.domElement.getBoundingClientRect()
    const ndc = new Vector2(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1,
    )
    raycaster.setFromCamera(ndc, camera)
    const hit = raycaster.intersectObjects([...meshes.values()], false)[0]
    let block = null
    if (hit && hit.instanceId != null) {
      block =
        grouped.find((g) => g.role === hit.object.userData.role)?.items[hit.instanceId] ?? null
    }
    if (commit) {
      selected = block
      hovered = null
    } else {
      if (hovered?.id === block?.id) return
      hovered = block
      renderer.domElement.style.cursor = block ? 'pointer' : 'grab'
    }
    paint()
  }

  function resetView() {
    if (!controls) return
    // Order matters. OrbitControls only clears its residual rotation delta inside
    // update()'s non-damping branch, and reset() calls update() *after* restoring
    // the home position — so any leftover inertia gets applied on top of home.
    // Flush the delta first with damping off, then reset.
    controls.enableDamping = false
    controls.update()
    controls.reset()
    controls.enableDamping = true
    needsRender = true
  }

  $effect(() => {
    // Declare the dependencies explicitly, then untrack the call: paint() reads a
    // lot of derived state, and letting all of it become a dependency of this
    // effect would repaint far more often than needed.
    lit
    colorMode
    if (renderer) untrack(() => paint())
  })

  $effect(() =>
    untrack(() => {
      try {
        build()
      } catch (err) {
        console.error('3D view unavailable:', err)
        failed = true
        return
      }
      const onResize = () => {
        if (!renderer) return
        const w = host.clientWidth
        renderer.setSize(w, height, false)
        camera.aspect = w / height
        camera.updateProjectionMatrix()
        needsRender = true
      }
      window.addEventListener('resize', onResize)
      // Don't run a render loop while scrolled away.
      const io = new IntersectionObserver((e) => (visible = e[0].isIntersecting), { threshold: 0 })
      io.observe(host)

      return () => {
        disposed = true
        io.disconnect()
        window.removeEventListener('resize', onResize)
        controls?.dispose()
        for (const m of meshes.values()) {
          m.geometry.dispose()
          m.material.dispose()
        }
        renderer?.dispose()
        renderer?.domElement.remove()
      }
    }),
  )

  const shown = $derived(selected ?? hovered)
  const shownStats = $derived(shown ? roleStats.stats.get(shown.role) : null)

  const fmtBytes = (b) =>
    b >= 1e12
      ? `${(b / 1e12).toFixed(2)} TB`
      : b >= 1e9
        ? `${(b / 1e9).toFixed(1)} GB`
        : b >= 1e6
          ? `${(b / 1e6).toFixed(1)} MB`
          : b >= 1e3
            ? `${(b / 1e3).toFixed(1)} kB`
            : `${b.toFixed(0)} B`

  const fmtP = (n) =>
    n >= 1e9
      ? `${(n / 1e9).toFixed(2)}B`
      : n >= 1e6
        ? `${(n / 1e6).toFixed(1)}M`
        : `${(n / 1e3).toFixed(1)}k`
</script>

<div class="figure-controls">
  {#if meta.isMoE}
    <label>
      Batch <span class="value">{B}</span>
      <input type="range" min="0" max="10" step="0.05" bind:value={logB} />
    </label>
  {/if}
  <button class:active={colorMode === 'activation'} onclick={() => (colorMode = 'activation')}>
    Activation
  </button>
  <button class:active={colorMode === 'role'} onclick={() => (colorMode = 'role')}>Role</button>
  <button onclick={resetView}>Reset view</button>
  <span class="hint">drag to orbit · scroll to zoom · click a part</span>
</div>

<div
  class="stage"
  style="min-height:{height}px"
  bind:this={host}
  role="application"
  aria-label="3D view of the {label || meta.name} architecture"
  onpointermove={(e) => pick(e, false)}
  onpointerdown={(e) => pick(e, true)}
>
  {#if failed}
    <p class="fallback">This view needs WebGL, which this browser has not made available.</p>
  {/if}
</div>

<div class="legend">
  {#if colorMode === 'role'}
    {#each grouped as g}
      <span class="lg">
        <i style="background:#{ROLES[g.role].color.toString(16).padStart(6, '0')}"></i>
        {ROLES[g.role].label}
      </span>
    {/each}
  {:else}
    <span class="lg"><i style="background:#c0392b"></i> active for this token</span>
    <span class="lg"><i style="background:#dedbd6"></i> dormant — held in memory, not used</span>
    {#if meta.isMoE}
      <span class="lg dim">{lit} of {meta.nExperts} experts lit per layer at batch {B}</span>
    {/if}
  {/if}
</div>

<div class="panel">
  {#if shown}
    <div class="head">
      <span class="name">{shown.name}</span>
      <span class="det">
        {ROLES[shown.role].label}{shown.layer >= 0 && shown.layer < meta.layers
          ? ` · layer ${shown.layer}`
          : ''}{shown.note ? ` · ${shown.note}` : ''}
      </span>
    </div>
    <div class="readouts">
      <div class="ro">
        <span class="k">dimensions</span>
        <span class="v">{shown.dims[0].toLocaleString()} × {shown.dims[1].toLocaleString()}</span>
      </div>
      <div class="ro">
        <span class="k">parameters</span>
        <span class="v">{fmtP(shown.params)}</span>
      </div>
      <div class="ro">
        <span class="k">this token</span>
        <span class="v" class:on={isActive(shown)}>{isActive(shown) ? 'active' : 'dormant'}</span>
      </div>
      {#if shownStats}
        <div class="ro">
          <span class="k">all {shownStats.count.toLocaleString()} of these</span>
          <span class="v">{fmtP(shownStats.params)}</span>
          <span class="s">{(100 * shownStats.paramShare).toFixed(1)}% of model</span>
        </div>
        <div class="ro">
          <span class="k">read per step</span>
          <span class="v">{fmtBytes(shownStats.bytes)}</span>
          <span class="s">{(100 * shownStats.byteShare).toFixed(0)}% of weight traffic</span>
        </div>
      {/if}
    </div>
  {:else}
    <p class="prompt">
      <b>{label || meta.name}</b> · {meta.layers} layers ·
      {(roleStats.totalParams / 1e9).toFixed(0)}B parameters across
      {blocks.length.toLocaleString()} weight blocks, of which
      <b>
        {(roleStats.totalActive / 1e9).toFixed(1)}B ({(
          (100 * roleStats.totalActive) /
          roleStats.totalParams
        ).toFixed(1)}%)</b
      >
      run for any one token. Hover or click any part for its dimensions.
    </p>
  {/if}
</div>

<style>
  .stage {
    position: relative;
    margin-top: 0.4em;
    background: linear-gradient(180deg, #fbfbfa 0%, #f1f0ee 100%);
    border: 1px solid rgba(0, 0, 0, 0.1);
    cursor: grab;
    touch-action: none;
  }

  .fallback {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0;
    padding: 2em;
    text-align: center;
    font-size: 13px;
    color: rgba(0, 0, 0, 0.55);
  }

  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3em 1.1em;
    margin-top: 0.55em;
    font-size: 11px;
    color: rgba(0, 0, 0, 0.6);
  }

  .legend .lg {
    display: inline-flex;
    align-items: center;
    gap: 0.4em;
  }

  .legend .lg.dim {
    color: rgba(0, 0, 0, 0.45);
    font-style: italic;
  }

  .legend i {
    width: 9px;
    height: 9px;
    border-radius: 1px;
    display: inline-block;
  }

  .panel {
    margin-top: 0.6em;
    padding-top: 0.7em;
    border-top: 1px solid rgba(0, 0, 0, 0.12);
    min-height: 76px;
  }

  .head {
    display: flex;
    align-items: baseline;
    gap: 0.6em;
    flex-wrap: wrap;
    margin-bottom: 0.55em;
  }

  .head .name {
    font-size: 15px;
    font-weight: 600;
    font-family: 'SF Mono', Menlo, Consolas, monospace;
  }

  .head .det {
    font-size: 12px;
    color: rgba(0, 0, 0, 0.5);
  }

  .prompt {
    margin: 0;
    font-size: 12.5px;
    line-height: 1.6em;
    color: rgba(0, 0, 0, 0.6);
  }

  .readouts {
    display: flex;
    flex-wrap: wrap;
    gap: 0.8em 1.75em;
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
    font-size: 14px;
    font-family: 'SF Mono', Menlo, Consolas, monospace;
    font-variant-numeric: tabular-nums;
  }

  .ro .v.on {
    color: #c0392b;
  }

  .ro .s {
    font-size: 11px;
    color: rgba(0, 0, 0, 0.5);
  }

  .hint {
    font-size: 12px;
    color: rgba(0, 0, 0, 0.45);
    font-style: italic;
  }

  button.active {
    background: #c0392b;
    color: #fff;
    border-color: #c0392b;
  }
</style>
