<script>
  /**
   * Model explorer: load one or more Hugging Face config.json files and compare
   * their architectures, in 3D and in a diff table.
   *
   * The diagrams are the same component the articles embed — see
   * src/components/BlockDiagram.svelte. Every box carries the model's own
   * shapes, parameter counts and FLOPs, and composite boxes open into their own
   * diagram, so any architecture the parser recognises can be walked from the
   * whole network down to a single matrix multiply.
   */
  import BlockDiagram from '../../src/components/BlockDiagram.svelte'
  import { parseArchitecture, summarise } from '../../src/lib/architecture.js'
  import { buildDiagrams } from '../../src/lib/diagrams.js'

  import glm52 from './data/glm-5.2.json'
  import deepseekV3 from './data/deepseek-v3.json'
  import gptOss from './data/gpt-oss-120b.json'
  import mixtral from './data/mixtral-8x7b.json'
  import qwen3 from './data/qwen3-8b.json'

  const PRESETS = [
    { id: 'glm-5.2', label: 'GLM-5.2', note: 'MoE · MLA · sparse attention', config: glm52 },
    { id: 'deepseek-v3', label: 'DeepSeek-V3', note: 'MoE · MLA', config: deepseekV3 },
    { id: 'gpt-oss-120b', label: 'gpt-oss-120b', note: 'MoE · GQA', config: gptOss },
    { id: 'mixtral-8x7b', label: 'Mixtral-8x7B', note: 'MoE · GQA · 8 experts', config: mixtral },
    { id: 'qwen3-8b', label: 'Qwen3-8B', note: 'dense · GQA', config: qwen3 },
  ]

  let loaded = $state([PRESETS[0]])
  let pasteOpen = $state(false)
  let pasteText = $state('')
  let pasteName = $state('')
  let error = $state('')

  const isLoaded = (id) => loaded.some((m) => m.id === id)

  function toggle(preset) {
    if (isLoaded(preset.id)) loaded = loaded.filter((m) => m.id !== preset.id)
    else loaded = [...loaded, preset]
  }

  /** Accepts a config.json and checks the parser can make sense of it. */
  function addConfig(name, text) {
    error = ''
    let cfg
    try {
      cfg = JSON.parse(text)
    } catch (e) {
      error = `That is not valid JSON: ${e.message}`
      return false
    }
    try {
      const arch = parseArchitecture(cfg)
      if (!arch.meta.layers || !arch.meta.hidden) {
        error = 'Parsed, but found no num_hidden_layers / hidden_size — is this a model config?'
        return false
      }
      const id = `custom-${Date.now()}`
      loaded = [
        ...loaded,
        { id, label: name || arch.meta.name || 'custom', note: 'loaded from JSON', config: cfg },
      ]
      return true
    } catch (e) {
      error = `Could not parse that architecture: ${e.message}`
      return false
    }
  }

  function submitPaste() {
    if (addConfig(pasteName.trim(), pasteText)) {
      pasteText = ''
      pasteName = ''
      pasteOpen = false
    }
  }

  async function onFiles(event) {
    for (const file of event.target.files) {
      addConfig(file.name.replace(/\.json$/, ''), await file.text())
    }
    event.target.value = ''
  }

  const rows = $derived(
    loaded.map((m) => ({ ...m, s: summarise(parseArchitecture(m.config), m.config) })),
  )

  const fmtB = (n) => (n >= 1e9 ? `${(n / 1e9).toFixed(1)}B` : `${(n / 1e6).toFixed(0)}M`)
  const fmtK = (n) => (n >= 1024 ? `${Math.round(n / 1024)}K` : `${n}`)

  /** Rows of the comparison table: label, accessor, and whether bigger is notable. */
  const METRICS = [
    { k: 'Total parameters', f: (s) => fmtB(s.totalParams) },
    { k: 'Active per token', f: (s) => fmtB(s.activeParams) },
    { k: 'Active share', f: (s) => `${(100 * s.activeShare).toFixed(1)}%` },
    { k: 'Layers', f: (s) => `${s.layers}${s.denseLayers && s.moeLayers ? ` (${s.denseLayers} dense)` : ''}` },
    { k: 'Hidden size', f: (s) => s.hidden.toLocaleString() },
    { k: 'Attention', f: (s) => s.attnKind },
    { k: 'Heads (Q / KV)', f: (s) => `${s.heads} / ${s.kvHeads}` },
    { k: 'Experts', f: (s) => (s.isMoE ? `top-${s.topk} of ${s.nExperts}${s.nShared ? ` +${s.nShared} shared` : ''}` : '—') },
    { k: 'Expert FFN', f: (s) => (s.isMoE ? s.moeFfn.toLocaleString() : s.ffn?.toLocaleString() ?? '—') },
    { k: 'KV cache / token', f: (s) => `${(s.kvPerToken / 1024).toFixed(1)} KiB` },
    { k: 'Max context', f: (s) => fmtK(s.maxContext ?? 0) },
    { k: 'Sparse-attn indexers', f: (s) => (s.indexerCount ? `${s.indexerCount} (top-${s.idxTopk})` : '—') },
    { k: 'MTP heads', f: (s) => s.nMtp || '—' },
    { k: 'Vocabulary', f: (s) => s.vocab.toLocaleString() },
    { k: 'Weight blocks drawn', f: (s) => s.blocks.toLocaleString() },
  ]
</script>

<section class="picker">
  <h2>Models</h2>
  <div class="chips">
    {#each PRESETS as p}
      <button class="chip" class:on={isLoaded(p.id)} onclick={() => toggle(p)}>
        <span class="lb">{p.label}</span>
        <span class="nt">{p.note}</span>
      </button>
    {/each}
  </div>

  <div class="adders">
    <button class="ghost" onclick={() => (pasteOpen = !pasteOpen)}>
      {pasteOpen ? 'Cancel' : 'Paste a config.json'}
    </button>
    <label class="ghost file">
      Load file
      <input type="file" accept=".json,application/json" multiple onchange={onFiles} />
    </label>
    {#each rows.filter((r) => r.id.startsWith('custom-')) as r}
      <button class="chip on" onclick={() => (loaded = loaded.filter((m) => m.id !== r.id))}>
        <span class="lb">{r.label}</span>
        <span class="nt">remove</span>
      </button>
    {/each}
  </div>

  {#if pasteOpen}
    <div class="paste">
      <input placeholder="name (optional)" bind:value={pasteName} />
      <textarea
        rows="6"
        placeholder="Paste the contents of a Hugging Face config.json here"
        bind:value={pasteText}
      ></textarea>
      <button onclick={submitPaste}>Add model</button>
    </div>
  {/if}

  {#if error}<p class="err">{error}</p>{/if}
</section>

{#if rows.length === 0}
  <p class="empty">Pick at least one model above.</p>
{/if}

{#each rows as r (r.id)}
  <section class="viewer">
    <h3>{r.label} <span class="sub">{r.s.name}</span></h3>
    <BlockDiagram diagrams={buildDiagrams(r.config)} />
  </section>
{/each}

{#if rows.length > 1}
  <section class="compare">
    <h2>Comparison</h2>
    <div class="scroll">
      <table>
        <thead>
          <tr>
            <th></th>
            {#each rows as r}<th>{r.label}</th>{/each}
          </tr>
        </thead>
        <tbody>
          {#each METRICS as m}
            <tr>
              <th scope="row">{m.k}</th>
              {#each rows as r}<td>{m.f(r.s)}</td>{/each}
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
    <p class="foot">
      Parameter counts are derived from the config alone, so they land within about 1% of published
      totals. Active-per-token assumes every routed expert is the same size and that routing is
      uniform.
    </p>
  </section>
{/if}

<style>
  section {
    margin-bottom: 2.5rem;
  }

  h2 {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: rgba(0, 0, 0, 0.5);
    font-weight: 500;
    margin: 0 0 0.8rem;
  }

  h3 {
    font-size: 1.05rem;
    margin: 0 0 0.2rem;
    display: flex;
    align-items: baseline;
    gap: 0.6em;
    flex-wrap: wrap;
  }

  h3 .sub {
    font-size: 11px;
    font-family: 'SF Mono', Menlo, Consolas, monospace;
    color: rgba(0, 0, 0, 0.4);
    font-weight: 400;
  }

  .chips,
  .adders {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5em;
  }

  .adders {
    margin-top: 0.6em;
    align-items: center;
  }

  .chip {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 0.1em;
    padding: 0.4em 0.7em;
    border: 1px solid rgba(0, 0, 0, 0.2);
    border-radius: 3px;
    background: #fff;
    cursor: pointer;
    font: inherit;
    text-align: left;
  }

  .chip:hover {
    background: rgba(0, 0, 0, 0.03);
  }

  .chip.on {
    background: #c0392b;
    border-color: #c0392b;
    color: #fff;
  }

  .chip .lb {
    font-size: 13px;
    font-weight: 500;
  }

  .chip .nt {
    font-size: 10.5px;
    opacity: 0.7;
  }

  .ghost {
    font: inherit;
    font-size: 12.5px;
    padding: 0.45em 0.8em;
    border: 1px dashed rgba(0, 0, 0, 0.3);
    border-radius: 3px;
    background: transparent;
    cursor: pointer;
  }

  .ghost:hover {
    background: rgba(0, 0, 0, 0.03);
  }

  .file input {
    display: none;
  }

  .paste {
    display: flex;
    flex-direction: column;
    gap: 0.5em;
    margin-top: 0.8em;
    max-width: 620px;
  }

  .paste input,
  .paste textarea {
    font: inherit;
    font-size: 12.5px;
    padding: 0.5em;
    border: 1px solid rgba(0, 0, 0, 0.2);
    border-radius: 3px;
  }

  .paste textarea {
    font-family: 'SF Mono', Menlo, Consolas, monospace;
    font-size: 11.5px;
  }

  .paste button {
    align-self: flex-start;
    font: inherit;
    font-size: 12.5px;
    padding: 0.45em 0.9em;
    border: 1px solid #c0392b;
    background: #c0392b;
    color: #fff;
    border-radius: 3px;
    cursor: pointer;
  }

  .err {
    color: #c0392b;
    font-size: 12.5px;
  }

  .empty {
    color: rgba(0, 0, 0, 0.5);
  }

  .scroll {
    overflow-x: auto;
  }

  table {
    border-collapse: collapse;
    font-size: 13px;
    font-variant-numeric: tabular-nums;
    min-width: 100%;
  }

  thead th {
    text-align: left;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: rgba(0, 0, 0, 0.5);
    padding: 0 1em 0.5em 0;
    border-bottom: 1px solid rgba(0, 0, 0, 0.2);
    white-space: nowrap;
  }

  tbody th {
    text-align: left;
    font-weight: 400;
    color: rgba(0, 0, 0, 0.55);
    padding: 0.4em 1.5em 0.4em 0;
    border-bottom: 1px solid rgba(0, 0, 0, 0.06);
    white-space: nowrap;
  }

  tbody td {
    padding: 0.4em 1em 0.4em 0;
    border-bottom: 1px solid rgba(0, 0, 0, 0.06);
    font-family: 'SF Mono', Menlo, Consolas, monospace;
    font-size: 12.5px;
    white-space: nowrap;
  }

  .foot {
    font-size: 11.5px;
    color: rgba(0, 0, 0, 0.5);
    margin-top: 0.9em;
  }
</style>
