/**
 * The diagram specs, built from a real config.json so every box carries the
 * model's own numbers rather than symbols.
 *
 * A spec is nodes on a grid (col across, row upward from 0) plus edges. Nodes
 * with `expand` open another spec; nodes with `figure` jump to one of the
 * article's interactive figures.
 */

const M = (n) => (n >= 1e9 ? `${(n / 1e9).toFixed(2)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : `${(n / 1e3).toFixed(0)}k`)
const F = (n) => (n >= 1e9 ? `${(n / 1e9).toFixed(1)} GFLOP` : `${(n / 1e6).toFixed(0)} MFLOP`)

export function buildDiagrams(cfg, { tokens = 512 } = {}) {
  const H = cfg.hidden_size
  const L = cfg.num_hidden_layers
  const nh = cfg.num_attention_heads
  const kv = cfg.num_key_value_heads ?? nh
  const hd = cfg.head_dim ?? H / nh
  const ffn = cfg.intermediate_size
  const V = cfg.vocab_size
  const E = cfg.n_routed_experts ?? cfg.num_experts ?? cfg.num_local_experts ?? 0
  const topk = cfg.num_experts_per_tok ?? cfg.experts_per_token ?? 0
  const moeFfn = cfg.moe_intermediate_size ?? ffn
  const nShared = cfg.n_shared_experts ?? 0
  const denseFirst = cfg.first_k_dense_replace ?? 0
  const isMoE = E > 0 && topk > 0
  const name = cfg._name_or_path ?? cfg.architectures?.[0] ?? 'model'

  // Multi-head latent attention, if this config uses it.
  const qLora = cfg.q_lora_rank
  const kvLora = cfg.kv_lora_rank
  const qkNope = cfg.qk_nope_head_dim ?? 0
  const qkRope = cfg.qk_rope_head_dim ?? 0
  const vHead = cfg.v_head_dim ?? hd
  const qkHead = qkNope + qkRope
  const isMLA = kvLora != null && cfg.v_head_dim != null
  // The softmax scale uses the query/key head dimension. Under MLA that is
  // qk_nope + qk_rope, not head_dim or hidden/heads.
  const sdpaDim = isMLA ? qkHead : hd
  const sdpaOut = isMLA ? vHead : hd

  // DeepSeek-style sparse attention indexer.
  const idxHeads = cfg.index_n_heads
  const idxDim = cfg.index_head_dim
  const idxTopk = cfg.index_topk
  const hasIndexer = idxHeads != null && idxDim != null
  const indexerLayers = Array.isArray(cfg.indexer_types)
    ? cfg.indexer_types.filter((t) => t === 'full').length
    : hasIndexer
      ? L
      : 0

  const nMtp = cfg.num_nextn_predict_layers ?? 0
  const mlpTypes = cfg.mlp_layer_types
  const denseLayers = mlpTypes
    ? mlpTypes.filter((t) => t === 'dense').length
    : isMoE
      ? denseFirst
      : L
  const moeLayers = L - denseLayers

  const lin = (a, b) => ({
    shape: `${a.toLocaleString()}→${b.toLocaleString()}`,
    cost: `${M(a * b)} params · ${F(2 * a * b)}`,
  })
  const expertParams = 3 * H * moeFfn
  const denseFfnParams = 3 * H * ffn

  const attnParams = isMLA
    ? H * qLora +
      qLora * nh * qkHead +
      H * (kvLora + qkRope) +
      kvLora * nh * (qkNope + vHead) +
      nh * vHead * H
    : H * nh * hd + 2 * H * kv * hd + nh * hd * H
  const idxParams = hasIndexer ? (qLora ?? H) * idxHeads * idxDim + idxDim * H + idxHeads * H : 0
  const moeParams = E * expertParams + nShared * expertParams + H * E
  const perLayer = attnParams + (isMoE ? moeParams : denseFfnParams)
  // The MTP head is a full extra layer plus a projection. It is drawn on the
  // diagram, so it has to be counted or the total won't match the checkpoint.
  const mtpParams = nMtp * (attnParams + (isMoE ? moeParams : denseFfnParams) + H * H)
  const total =
    2 * V * H +
    moeLayers * (attnParams + moeParams) +
    denseLayers * (attnParams + denseFfnParams) +
    indexerLayers * idxParams +
    mtpParams
  const activePerLayer = isMoE ? (topk + nShared) * expertParams + H * E : denseFfnParams
  const active =
    V * H +
    moeLayers * (attnParams + activePerLayer) +
    denseLayers * (attnParams + denseFfnParams) +
    indexerLayers * idxParams

  const rowTop = denseLayers && moeLayers ? 4 : 3

  return {
    // ---------------------------------------------------------------- model
    model: {
      title: 'The model',
      subtitle: `${name} · ${M(total)} parameters, ${M(active)} active per token`,
      nodes: [
        { id: 'tok', kind: 'terminal', label: 'tokens', col: 0, row: 0, note: `[${tokens}]` },
        { id: 'embed', kind: 'embed', label: 'Embedding', col: 0, row: 1, ...lin(V, H), cost: `${M(V * H)} params · gather` },
        ...(denseLayers && moeLayers
          ? [
              {
                id: 'dense', kind: 'composite', label: 'Dense block', col: 0, row: 2, w: 190,
                stack: denseLayers, expand: 'blockDense',
                cost: `${M(attnParams + denseFfnParams)} params`,
              },
              {
                id: 'block', kind: 'composite', label: 'MoE block', col: 0, row: 3, w: 190,
                stack: moeLayers, expand: 'block', cost: `${M(perLayer)} params`,
              },
            ]
          : [
              {
                id: 'block', kind: 'composite',
                label: isMoE ? 'MoE block' : 'Transformer block', col: 0, row: 2, w: 190,
                stack: L, expand: 'block', shape: `[${tokens}, ${H.toLocaleString()}]`,
                cost: `${M(perLayer)} params`,
              },
            ]),
        ...(nMtp
          ? [{ id: 'mtp', kind: 'composite', label: 'MTP head', col: 1.7, row: rowTop, w: 124, note: 'idle unless speculating', cost: `${M(mtpParams)} params` }]
          : []),
        { id: 'norm', kind: 'norm', label: 'Final norm', col: 0, row: rowTop, note: 'RMSNorm' },
        { id: 'head', kind: 'linear', label: 'LM head', col: 0, row: rowTop + 1, ...lin(H, V) },
        { id: 'logits', kind: 'terminal', label: 'logits', col: 0, row: rowTop + 2, note: `[${tokens}, ${V.toLocaleString()}]` },
      ],
      edges: [
        { from: 'tok', to: 'embed' },
        ...(denseLayers && moeLayers
          ? [{ from: 'embed', to: 'dense' }, { from: 'dense', to: 'block' }]
          : [{ from: 'embed', to: 'block' }]),
        { from: 'block', to: 'norm' }, { from: 'norm', to: 'head' }, { from: 'head', to: 'logits' },
      ],
      note: `The whole network in ${denseLayers && moeLayers ? 'six' : 'five'} boxes. The stack is drawn once with a ×N badge, because ${L} layers is one layer ${L} times.${
        denseLayers && moeLayers
          ? ` The first ${denseLayers} layers are dense and the remaining ${moeLayers} are sparse, so they are genuinely different blocks.`
          : ''
      } Click a block to open it.`,
    },

    // ---------------------------------------------------------------- block
    block: {
      title: 'Transformer block',
      subtitle: `one of ${moeLayers || L} · ${M(perLayer)} parameters`,
      nodes: [
        { id: 'in', kind: 'terminal', label: 'x', col: 0, row: 0, note: `[${tokens}, ${H.toLocaleString()}]` },
        { id: 'n1', kind: 'norm', label: 'RMSNorm', col: 0, row: 1 },
        { id: 'attn', kind: 'composite', label: isMLA ? 'Latent Attention' : 'Multi-Head Attention', col: 0, row: 2, w: 200, expand: 'mha', cost: `${M(attnParams)} params` },
        { id: 'add1', kind: 'add', label: '+', col: 0, row: 3, w: 44, note: 'residual' },
        { id: 'n2', kind: 'norm', label: 'RMSNorm', col: 0, row: 4 },
        {
          id: 'ffn', kind: isMoE ? 'moe' : 'linear',
          label: isMoE ? 'Mixture of Experts' : 'Feed-forward', col: 0, row: 5, w: 200,
          expand: isMoE ? 'moe' : 'ffn', cost: `${M(isMoE ? moeParams : denseFfnParams)} params`,
          note: isMoE ? `top-${topk} of ${E}` : `${H.toLocaleString()}→${ffn.toLocaleString()}→${H.toLocaleString()}`,
        },
        { id: 'add2', kind: 'add', label: '+', col: 0, row: 6, w: 44, note: 'residual' },
        { id: 'out', kind: 'terminal', label: 'x′', col: 0, row: 7, note: `[${tokens}, ${H.toLocaleString()}]` },
      ],
      edges: [
        { from: 'in', to: 'n1' }, { from: 'n1', to: 'attn' }, { from: 'attn', to: 'add1' },
        { from: 'in', to: 'add1' },
        { from: 'add1', to: 'n2' }, { from: 'n2', to: 'ffn' }, { from: 'ffn', to: 'add2' },
        { from: 'add1', to: 'add2' },
        { from: 'add2', to: 'out' },
      ],
      note: 'Both halves are written <i>into</i> a running stream rather than replacing it — the two lines that skip past the boxes are the residual connections, and they are why gradients survive a deep stack.',
    },

    // ---------------------------------------------------- dense block variant
    ...(denseLayers && moeLayers
      ? {
          blockDense: {
            title: 'Dense block',
            subtitle: `the first ${denseLayers} layers · ${M(attnParams + denseFfnParams)} parameters`,
            nodes: [
              { id: 'in', kind: 'terminal', label: 'x', col: 0, row: 0, note: `[${tokens}, ${H.toLocaleString()}]` },
              { id: 'n1', kind: 'norm', label: 'RMSNorm', col: 0, row: 1 },
              { id: 'attn', kind: 'composite', label: isMLA ? 'Latent Attention' : 'Multi-Head Attention', col: 0, row: 2, w: 200, expand: 'mha', cost: `${M(attnParams)} params` },
              { id: 'add1', kind: 'add', label: '+', col: 0, row: 3, w: 44, note: 'residual' },
              { id: 'n2', kind: 'norm', label: 'RMSNorm', col: 0, row: 4 },
              { id: 'ffn', kind: 'linear', label: 'Feed-forward', col: 0, row: 5, w: 200, expand: 'ffn', cost: `${M(denseFfnParams)} params`, note: `${H.toLocaleString()}→${ffn.toLocaleString()}→${H.toLocaleString()}` },
              { id: 'add2', kind: 'add', label: '+', col: 0, row: 6, w: 44, note: 'residual' },
              { id: 'out', kind: 'terminal', label: 'x′', col: 0, row: 7 },
            ],
            edges: [
              { from: 'in', to: 'n1' }, { from: 'n1', to: 'attn' }, { from: 'attn', to: 'add1' },
              { from: 'in', to: 'add1' }, { from: 'add1', to: 'n2' }, { from: 'n2', to: 'ffn' },
              { from: 'ffn', to: 'add2' }, { from: 'add1', to: 'add2' }, { from: 'add2', to: 'out' },
            ],
            note: `Identical to the sparse blocks except here, where an ordinary feed-forward network sits instead of a router and ${E} experts. Routing this early tends to hurt, so the first ${denseLayers} layers stay dense.`,
          },
          ffn: {
            title: 'Feed-forward',
            subtitle: `${M(denseFfnParams)} parameters`,
            nodes: [
              { id: 'in', kind: 'terminal', label: 'x', col: 0, row: 0 },
              { id: 'up', kind: 'linear', label: 'Linear', col: 0, row: 1, ...lin(H, ffn) },
              { id: 'act', kind: 'scale', label: 'SwiGLU', col: 0, row: 2 },
              { id: 'down', kind: 'linear', label: 'Linear', col: 0, row: 3, ...lin(ffn, H) },
              { id: 'out', kind: 'terminal', label: 'out', col: 0, row: 4 },
            ],
            edges: [
              { from: 'in', to: 'up' }, { from: 'up', to: 'act' },
              { from: 'act', to: 'down' }, { from: 'down', to: 'out' },
            ],
            note: 'Wide in the middle, back to the model width at the end.',
          },
        }
      : {}),

    // ------------------------------------------------------------ attention
    mha: isMLA
      ? {
          title: 'Multi-head Latent Attention',
          subtitle: `${nh} heads · queries via a ${qLora.toLocaleString()}-dim latent, keys and values via ${kvLora.toLocaleString()}`,
          nodes: [
            { id: 'x', kind: 'terminal', label: 'x', col: 0, row: 0, note: `[${tokens}, ${H.toLocaleString()}]` },
            { id: 'qa', kind: 'linear', label: 'q_a_proj', col: -0.85, row: 1, w: 100, ...lin(H, qLora) },
            { id: 'kva', kind: 'linear', label: 'kv_a_proj', col: 0.85, row: 1, w: 100, ...lin(H, kvLora + qkRope) },
            { id: 'qb', kind: 'linear', label: 'q_b_proj', col: -0.85, row: 2, w: 100, ...lin(qLora, nh * qkHead) },
            { id: 'kvb', kind: 'linear', label: 'kv_b_proj', col: 0.85, row: 2, w: 100, ...lin(kvLora, nh * (qkNope + vHead)) },
            ...(hasIndexer
              ? [
                  { id: 'idx', kind: 'router', label: 'Indexer', col: -2.1, row: 2, w: 100, note: `top-${idxTopk?.toLocaleString?.() ?? idxTopk}`, cost: `${M(idxParams)} params` },
                ]
              : []),
            { id: 'sdpa', kind: 'composite', label: 'Scaled Dot-Product Attention', col: 0, row: 3, w: 300, stack: nh, expand: 'sdpa', note: 'no parameters of its own' },
            { id: 'lo', kind: 'linear', label: 'o_proj', col: 0, row: 4, ...lin(nh * vHead, H) },
            { id: 'out', kind: 'terminal', label: 'out', col: 0, row: 5, note: `[${tokens}, ${H.toLocaleString()}]` },
          ],
          edges: [
            { from: 'x', to: 'qa' }, { from: 'x', to: 'kva' },
            { from: 'qa', to: 'qb' }, { from: 'kva', to: 'kvb' },
            { from: 'qb', to: 'sdpa' }, { from: 'kvb', to: 'sdpa' },
            ...(hasIndexer ? [{ from: 'x', to: 'idx' }, { from: 'idx', to: 'sdpa' }] : []),
            { from: 'sdpa', to: 'lo' }, { from: 'lo', to: 'out' },
          ],
          note: `Every projection goes through a bottleneck. Keys and values are squeezed to <b>${kvLora.toLocaleString()}</b> dimensions before being expanded back to ${(nh * (qkNope + vHead)).toLocaleString()} — and it is the small latent that gets cached, not the expansion, which is what makes a very long context affordable.${
            hasIndexer
              ? ` The indexer on the left is not part of classic attention: it scores every token cheaply and keeps only the top ${idxTopk?.toLocaleString?.() ?? idxTopk}, so attention never sees the whole context.`
              : ''
          }`,
        }
      : {
          title: 'Multi-Head Attention',
          subtitle: `${nh} heads of ${hd} dims${kv !== nh ? ` · ${kv} shared KV heads (GQA)` : ''}`,
          nodes: [
            { id: 'V', kind: 'terminal', label: 'V', col: -1.15, row: 0 },
            { id: 'K', kind: 'terminal', label: 'K', col: 0, row: 0 },
            { id: 'Q', kind: 'terminal', label: 'Q', col: 1.15, row: 0 },
            { id: 'lv', kind: 'linear', label: 'Linear', col: -1.15, row: 1, w: 92, stack: kv, ...lin(H, kv * hd) },
            { id: 'lk', kind: 'linear', label: 'Linear', col: 0, row: 1, w: 92, stack: kv, ...lin(H, kv * hd) },
            { id: 'lq', kind: 'linear', label: 'Linear', col: 1.15, row: 1, w: 92, stack: nh, ...lin(H, nh * hd) },
            { id: 'sdpa', kind: 'composite', label: 'Scaled Dot-Product Attention', col: 0, row: 2, w: 300, stack: nh, expand: 'sdpa', note: 'no parameters of its own' },
            { id: 'concat', kind: 'concat', label: 'Concat', col: 0, row: 3, note: `${nh} × ${hd} = ${(nh * hd).toLocaleString()}` },
            { id: 'lo', kind: 'linear', label: 'Linear', col: 0, row: 4, ...lin(nh * hd, H) },
            { id: 'out', kind: 'terminal', label: 'out', col: 0, row: 5, note: `[${tokens}, ${H.toLocaleString()}]` },
          ],
          edges: [
            { from: 'V', to: 'lv' }, { from: 'K', to: 'lk' }, { from: 'Q', to: 'lq' },
            { from: 'lv', to: 'sdpa' }, { from: 'lk', to: 'sdpa' }, { from: 'lq', to: 'sdpa' },
            { from: 'sdpa', to: 'concat' }, { from: 'concat', to: 'lo' }, { from: 'lo', to: 'out' },
          ],
          note: `Each head attends for its own reason, over its own ${hd}-dimensional slice.${
            kv !== nh
              ? ` Note the asymmetry: ${nh} query heads but only ${kv} key/value heads, shared between them — grouped-query attention, which shrinks the KV cache ${(nh / kv).toFixed(0)}× at almost no quality cost.`
              : ''
          }`,
        },

    // ----------------------------------------------------------------- SDPA
    sdpa: {
      title: 'Scaled Dot-Product Attention',
      subtitle: `one head · d = ${sdpaDim}, so the divisor is √${sdpaDim} = ${Math.sqrt(sdpaDim).toFixed(1)}`,
      nodes: [
        { id: 'Q', kind: 'terminal', label: 'Q', col: -0.55, row: 0, note: `[${tokens}, ${sdpaDim}]` },
        { id: 'K', kind: 'terminal', label: 'K', col: 0.55, row: 0, note: `[${tokens}, ${sdpaDim}]` },
        { id: 'V', kind: 'terminal', label: 'V', col: 1.9, row: 0, note: `[${tokens}, ${sdpaOut}]` },
        {
          id: 'qk', kind: 'matmul', label: 'MatMul', col: 0, row: 1, figure: 'matmul',
          note: 'Q·Kᵀ', shape: `→ [${tokens}, ${tokens}]`, cost: F(2 * tokens * tokens * sdpaDim),
        },
        { id: 'scale', kind: 'scale', label: 'Scale', col: 0, row: 2, note: `÷ ${Math.sqrt(sdpaDim).toFixed(1)}` },
        { id: 'mask', kind: 'mask', label: 'Mask (opt.)', col: 0, row: 3, note: 'causal' },
        { id: 'sm', kind: 'softmax', label: 'SoftMax', col: 0, row: 4, note: 'rows sum to 1', figure: 'attention' },
        {
          id: 'av', kind: 'matmul', label: 'MatMul', col: 0, row: 5, figure: 'matmul',
          note: 'weights·V', shape: `→ [${tokens}, ${sdpaOut}]`, cost: F(2 * tokens * tokens * sdpaOut),
        },
        { id: 'out', kind: 'terminal', label: 'head out', col: 0, row: 6, note: `[${tokens}, ${sdpaOut}]` },
      ],
      edges: [
        { from: 'Q', to: 'qk' }, { from: 'K', to: 'qk' },
        { from: 'qk', to: 'scale' }, { from: 'scale', to: 'mask' },
        { from: 'mask', to: 'sm' }, { from: 'sm', to: 'av' },
        { from: 'V', to: 'av' }, { from: 'av', to: 'out' },
      ],
      note: `The only box here with no parameters at all — attention is a way of <i>using</i> numbers, not storing them. Both MatMuls cost about ${F(2 * tokens * tokens * sdpaDim)} at ${tokens} tokens and grow with the square of context. Click either to see the operation itself.`,
    },

    // ------------------------------------------------------------------ MoE
    ...(isMoE
      ? {
          moe: {
            title: 'Mixture of Experts',
            subtitle: `${E} experts, top-${topk} per token${nShared ? ` + ${nShared} always-on` : ''} · ${M(E * expertParams)} stored, ${M((topk + nShared) * expertParams)} used`,
            nodes: [
              { id: 'in', kind: 'terminal', label: 'x', col: 0, row: 0, note: `[${tokens}, ${H.toLocaleString()}]` },
              { id: 'router', kind: 'router', label: 'Router', col: -1.3, row: 1, w: 96, ...lin(H, E), note: `scores ${E}` },
              { id: 'topk', kind: 'mask', label: `Top-${topk}`, col: -1.3, row: 2, w: 96, note: 'keep the best' },
              {
                id: 'experts', kind: 'moe', label: 'Expert FFN', col: 0.55, row: 2, w: 150, stack: E,
                shape: `${H.toLocaleString()}→${moeFfn.toLocaleString()}→${H.toLocaleString()}`,
                cost: `${M(expertParams)} params each`,
              },
              ...(nShared
                ? [{ id: 'shared', kind: 'moe', label: 'Shared expert', col: -1.3, row: 3, w: 130, note: 'always runs', cost: `${M(nShared * expertParams)} params` }]
                : []),
              { id: 'mix', kind: 'concat', label: 'Weighted sum', col: 0.4, row: 4, w: 160, note: `${topk + nShared} outputs` },
              { id: 'out', kind: 'terminal', label: 'out', col: 0.4, row: 5, note: `[${tokens}, ${H.toLocaleString()}]` },
            ],
            edges: [
              { from: 'in', to: 'router' }, { from: 'in', to: 'experts' },
              { from: 'router', to: 'topk' }, { from: 'topk', to: 'experts' },
              ...(nShared ? [{ from: 'in', to: 'shared' }, { from: 'shared', to: 'mix' }] : []),
              { from: 'experts', to: 'mix' }, { from: 'mix', to: 'out' },
            ],
            note: `The router is trivial — ${M(H * E)} parameters — but it decides which <b>${M(topk * expertParams)}</b> of the layer's <b>${M(E * expertParams)}</b> actually runs. That ratio is the whole idea: capacity is decoupled from cost, and only ${((100 * topk) / E).toFixed(0)}% of the layer participates in any given token.`,
          },
        }
      : {
          ffn: {
            title: 'Feed-forward',
            subtitle: `${M(3 * H * ffn)} parameters`,
            nodes: [
              { id: 'in', kind: 'terminal', label: 'x', col: 0, row: 0 },
              { id: 'up', kind: 'linear', label: 'Linear', col: 0, row: 1, ...lin(H, ffn) },
              { id: 'act', kind: 'scale', label: 'SwiGLU', col: 0, row: 2 },
              { id: 'down', kind: 'linear', label: 'Linear', col: 0, row: 3, ...lin(ffn, H) },
              { id: 'out', kind: 'terminal', label: 'out', col: 0, row: 4 },
            ],
            edges: [
              { from: 'in', to: 'up' }, { from: 'up', to: 'act' },
              { from: 'act', to: 'down' }, { from: 'down', to: 'out' },
            ],
            note: 'Wide in the middle, back to the model width at the end.',
          },
        }),
  }
}
