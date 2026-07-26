/**
 * Turns a Hugging Face `config.json` into a list of drawable blocks.
 *
 * The point is that this reads the config rather than hard-coding one model: give
 * it a dense Llama-shaped config and you get attention plus MLP per layer; give it
 * a sparse MoE config and you get routers and expert grids; give it an MLA config
 * and the attention block decomposes into its low-rank projections. Anything it
 * doesn't recognise it simply omits.
 *
 * Each block carries its two matrix dimensions, so a renderer can size it by
 * actual dimensionality instead of by a guess.
 */

export const ROLES = {
  embedding: { label: 'Embedding lookup', color: 0x8e6fb5 },
  attention: { label: 'Attention', color: 0x2c7fb8 },
  indexer: { label: 'Sparse-attention indexer', color: 0x00a0a0 },
  dense_mlp: { label: 'Dense MLP', color: 0xe08a1e },
  router: { label: 'Router', color: 0x6b6b6b },
  shared_expert: { label: 'Shared expert', color: 0xd4a017 },
  routed_expert: { label: 'Routed expert', color: 0xc0392b },
  mtp: { label: 'MTP head', color: 0x9b59b6 },
  lm_head: { label: 'LM head', color: 0x8e6fb5 },
}

const pick = (cfg, ...keys) => {
  for (const k of keys) if (cfg[k] != null) return cfg[k]
  return undefined
}

/**
 * @param {object} cfg a parsed config.json
 * @returns {{ meta: object, layers: object[], blocks: object[] }}
 */
export function parseArchitecture(cfg) {
  const H = pick(cfg, 'hidden_size', 'n_embd')
  const L = pick(cfg, 'num_hidden_layers', 'n_layer')
  const V = pick(cfg, 'vocab_size')
  const heads = pick(cfg, 'num_attention_heads', 'n_head')
  const kvHeads = pick(cfg, 'num_key_value_heads') ?? heads
  const ffn = pick(cfg, 'intermediate_size')

  // Multi-head latent attention, if this config uses it.
  const qLora = pick(cfg, 'q_lora_rank')
  const kvLora = pick(cfg, 'kv_lora_rank')
  const qkNope = pick(cfg, 'qk_nope_head_dim')
  const qkRope = pick(cfg, 'qk_rope_head_dim')
  const vHead = pick(cfg, 'v_head_dim')
  const headDim = pick(cfg, 'head_dim') ?? (H && heads ? H / heads : undefined)
  const isMLA = kvLora != null && vHead != null

  // Sparse mixture of experts, if present. Naming is not standardised: DeepSeek
  // and GLM use n_routed_experts + moe_intermediate_size, while Mixtral and
  // gpt-oss use num_local_experts and size the experts with intermediate_size.
  const nExperts = pick(cfg, 'n_routed_experts', 'num_experts', 'num_local_experts')
  const topk = pick(cfg, 'num_experts_per_tok', 'experts_per_token')
  const moeFfn = pick(cfg, 'moe_intermediate_size') ?? (nExperts != null ? ffn : undefined)
  const nShared = pick(cfg, 'n_shared_experts') ?? 0
  const denseFirst = pick(cfg, 'first_k_dense_replace') ?? 0
  const isMoE = nExperts != null && moeFfn != null && topk != null

  // DeepSeek-style sparse attention indexer, if present.
  const idxHeads = pick(cfg, 'index_n_heads')
  const idxDim = pick(cfg, 'index_head_dim')
  const idxTopk = pick(cfg, 'index_topk')
  const indexerTypes = pick(cfg, 'indexer_types')
  const hasIndexer = idxHeads != null && idxDim != null

  const nMtp = pick(cfg, 'num_nextn_predict_layers') ?? 0
  const mlpTypes = pick(cfg, 'mlp_layer_types')

  const layers = Array.from({ length: L }, (_, i) => ({
    index: i,
    dense: mlpTypes ? mlpTypes[i] === 'dense' : i < denseFirst,
    fullIndexer: hasIndexer ? (indexerTypes ? indexerTypes[i] === 'full' : true) : false,
  }))

  const blocks = []
  const add = (b) => blocks.push({ id: blocks.length, ...b })

  // ---- input side
  add({
    role: 'embedding',
    layer: -1,
    name: 'Token embedding',
    dims: [V, H],
    params: V * H,
    // A lookup reads one row per token, not the table.
    activeParams: H,
    note: 'Lookup table, one row per token',
  })

  for (const l of layers) {
    // ---- attention
    if (isMLA) {
      const qkHead = (qkNope ?? 0) + (qkRope ?? 0)
      add({ role: 'attention', layer: l.index, name: 'q_a_proj (down)', dims: [H, qLora], params: H * qLora, activeParams: H * qLora })
      add({ role: 'attention', layer: l.index, name: 'q_b_proj (up)', dims: [qLora, heads * qkHead], params: qLora * heads * qkHead, activeParams: qLora * heads * qkHead })
      add({ role: 'attention', layer: l.index, name: 'kv_a_proj (down)', dims: [H, kvLora + qkRope], params: H * (kvLora + qkRope), activeParams: H * (kvLora + qkRope) })
      add({ role: 'attention', layer: l.index, name: 'kv_b_proj (up)', dims: [kvLora, heads * (qkNope + vHead)], params: kvLora * heads * (qkNope + vHead), activeParams: kvLora * heads * (qkNope + vHead) })
      add({ role: 'attention', layer: l.index, name: 'o_proj', dims: [heads * vHead, H], params: heads * vHead * H, activeParams: heads * vHead * H })
    } else if (heads && headDim) {
      add({ role: 'attention', layer: l.index, name: 'q_proj', dims: [H, heads * headDim], params: H * heads * headDim, activeParams: H * heads * headDim })
      add({ role: 'attention', layer: l.index, name: 'k_proj', dims: [H, kvHeads * headDim], params: H * kvHeads * headDim, activeParams: H * kvHeads * headDim })
      add({ role: 'attention', layer: l.index, name: 'v_proj', dims: [H, kvHeads * headDim], params: H * kvHeads * headDim, activeParams: H * kvHeads * headDim })
      add({ role: 'attention', layer: l.index, name: 'o_proj', dims: [heads * headDim, H], params: heads * headDim * H, activeParams: heads * headDim * H })
    }

    // ---- indexer, only on the layers that own one
    if (l.fullIndexer) {
      add({
        role: 'indexer',
        layer: l.index,
        name: 'indexer',
        dims: [qLora ?? H, idxHeads * idxDim],
        params: (qLora ?? H) * idxHeads * idxDim + idxDim * H + idxHeads * H,
        activeParams: (qLora ?? H) * idxHeads * idxDim + idxDim * H + idxHeads * H,
        note: idxTopk ? `selects top-${idxTopk} keys` : undefined,
      })
    }

    // ---- feed-forward: dense, or a router plus experts
    if (!isMoE || l.dense) {
      add({ role: 'dense_mlp', layer: l.index, name: 'dense MLP', dims: [H, ffn], params: 3 * H * ffn, activeParams: 3 * H * ffn })
    } else {
      add({ role: 'router', layer: l.index, name: 'router', dims: [H, nExperts], params: H * nExperts, activeParams: H * nExperts })
      for (let s = 0; s < nShared; s++) {
        add({ role: 'shared_expert', layer: l.index, name: 'shared expert', dims: [H, moeFfn], params: 3 * H * moeFfn, activeParams: 3 * H * moeFfn })
      }
      for (let e = 0; e < nExperts; e++) {
        add({
          role: 'routed_expert',
          layer: l.index,
          expert: e,
          name: `expert ${e}`,
          dims: [H, moeFfn],
          params: 3 * H * moeFfn,
          // Only topk of nExperts run for any one token.
          activeParams: 0,
        })
      }
    }
  }

  // ---- MTP head and output side
  for (let m = 0; m < nMtp; m++) {
    add({
      role: 'mtp',
      layer: L + m,
      name: 'MTP head',
      dims: [H, H],
      params: H * H + (isMoE ? nExperts * 3 * H * moeFfn : 3 * H * ffn),
      activeParams: 0,
      note: 'drafts for speculative decoding; idle otherwise',
    })
  }

  add({
    role: 'lm_head',
    layer: L + nMtp,
    name: 'LM head',
    dims: [H, V],
    params: V * H,
    activeParams: V * H,
    note: 'projects onto the vocabulary',
  })

  return {
    meta: {
      name: cfg._name_or_path ?? cfg.architectures?.[0] ?? 'model',
      hidden: H,
      ffn,
      layers: L,
      vocab: V,
      heads,
      isMLA,
      isMoE,
      nExperts,
      topk,
      moeFfn,
      nShared,
      denseFirst,
      hasIndexer,
      idxTopk,
      indexerCount: layers.filter((l) => l.fullIndexer).length,
      nMtp,
      maxContext: pick(cfg, 'max_position_embeddings'),
      totalParams: blocks.reduce((a, b) => a + b.params, 0),
    },
    layers,
    blocks,
  }
}

/** Which experts are lit for a batch of B tokens, per layer. */
export function litExperts(B, nExperts, topk) {
  return Math.round(nExperts * (1 - Math.pow(1 - topk / nExperts, B)))
}

/**
 * Derived metrics for comparing architectures side by side. Everything here comes
 * from the parsed config, so two models can be judged on the same basis.
 */
export function summarise(arch, cfg) {
  const { meta, layers, blocks } = arch
  const moeLayers = meta.isMoE ? layers.filter((l) => !l.dense).length : 0

  // Parameters a single token actually puts through.
  let active = blocks.reduce((a, b) => a + b.activeParams, 0)
  if (meta.isMoE) {
    active += moeLayers * meta.topk * 3 * meta.hidden * meta.moeFfn
  }

  const heads = meta.heads
  const kvHeads = pick(cfg, 'num_key_value_heads') ?? heads
  const headDim = pick(cfg, 'head_dim') ?? meta.hidden / heads

  // KV cache per token, in bf16 — the number that decides how much context fits.
  const kvPerToken = meta.isMLA
    ? (pick(cfg, 'kv_lora_rank') + pick(cfg, 'qk_rope_head_dim')) * 2 * meta.layers
    : 2 * kvHeads * headDim * 2 * meta.layers

  const attnKind = meta.isMLA
    ? 'MLA (latent)'
    : kvHeads === heads
      ? 'MHA'
      : kvHeads === 1
        ? 'MQA'
        : `GQA ${heads}/${kvHeads}`

  return {
    name: meta.name,
    totalParams: meta.totalParams,
    activeParams: active,
    activeShare: active / meta.totalParams,
    layers: meta.layers,
    hidden: meta.hidden,
    vocab: meta.vocab,
    heads,
    kvHeads,
    headDim,
    attnKind,
    isMoE: meta.isMoE,
    nExperts: meta.nExperts ?? 0,
    topk: meta.topk ?? 0,
    nShared: meta.nShared ?? 0,
    moeLayers,
    denseLayers: meta.layers - moeLayers,
    ffn: pick(cfg, 'intermediate_size'),
    moeFfn: meta.moeFfn,
    kvPerToken,
    maxContext: meta.maxContext,
    indexerCount: meta.indexerCount,
    idxTopk: meta.idxTopk,
    nMtp: meta.nMtp,
    blocks: blocks.length,
  }
}
