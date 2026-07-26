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
  const E = cfg.num_local_experts ?? cfg.n_routed_experts ?? 0
  const topk = cfg.num_experts_per_tok ?? 0
  const isMoE = E > 0 && topk > 0
  const name = cfg._name_or_path ?? cfg.architectures?.[0] ?? 'model'

  const lin = (a, b) => ({
    shape: `${a.toLocaleString()}→${b.toLocaleString()}`,
    cost: `${M(a * b)} params · ${F(2 * a * b)}`,
  })
  const expertParams = 3 * H * ffn

  const attnParams = H * nh * hd + 2 * H * kv * hd + nh * hd * H
  const ffnParams = isMoE ? E * expertParams + H * E : 3 * H * ffn
  const perLayer = attnParams + ffnParams
  const total = 2 * V * H + L * perLayer
  const active = V * H + L * (attnParams + (isMoE ? topk * expertParams + H * E : 3 * H * ffn))

  return {
    // ---------------------------------------------------------------- model
    model: {
      title: 'The model',
      subtitle: `${name} · ${M(total)} parameters, ${M(active)} active per token`,
      nodes: [
        { id: 'tok', kind: 'terminal', label: 'tokens', col: 0, row: 0, note: `[${tokens}]` },
        { id: 'embed', kind: 'embed', label: 'Embedding', col: 0, row: 1, note: 'lookup', ...lin(V, H), cost: `${M(V * H)} params · gather` },
        {
          id: 'block', kind: 'composite', label: 'Transformer block', col: 0, row: 2, w: 190,
          stack: L, expand: 'block', shape: `[${tokens}, ${H.toLocaleString()}]`,
          cost: `${M(perLayer)} params · ${F(2 * (attnParams + (isMoE ? topk * expertParams : 3 * H * ffn)))}`,
        },
        { id: 'norm', kind: 'norm', label: 'Final norm', col: 0, row: 3, note: 'RMSNorm' },
        { id: 'head', kind: 'linear', label: 'LM head', col: 0, row: 4, ...lin(H, V) },
        { id: 'logits', kind: 'terminal', label: 'logits', col: 0, row: 5, note: `[${tokens}, ${V.toLocaleString()}]` },
      ],
      edges: [
        { from: 'tok', to: 'embed' }, { from: 'embed', to: 'block' },
        { from: 'block', to: 'norm' }, { from: 'norm', to: 'head' }, { from: 'head', to: 'logits' },
      ],
      note: `The whole network. One block, drawn once and repeated <b>${L}</b> times — which is why a model this large fits in five boxes. Click the block to open it.`,
    },

    // ---------------------------------------------------------------- block
    block: {
      title: 'Transformer block',
      subtitle: `one of ${L} · ${M(perLayer)} parameters`,
      nodes: [
        { id: 'in', kind: 'terminal', label: 'x', col: 0, row: 0, note: `[${tokens}, ${H.toLocaleString()}]` },
        { id: 'n1', kind: 'norm', label: 'RMSNorm', col: 0, row: 1 },
        { id: 'attn', kind: 'composite', label: 'Multi-Head Attention', col: 0, row: 2, w: 200, expand: 'mha', cost: `${M(attnParams)} params` },
        { id: 'add1', kind: 'add', label: '+', col: 0, row: 3, w: 44, note: 'residual' },
        { id: 'n2', kind: 'norm', label: 'RMSNorm', col: 0, row: 4 },
        {
          id: 'ffn', kind: isMoE ? 'moe' : 'linear',
          label: isMoE ? 'Mixture of Experts' : 'Feed-forward', col: 0, row: 5, w: 200,
          expand: isMoE ? 'moe' : 'ffn', cost: `${M(ffnParams)} params`,
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

    // ------------------------------------------------------------------ MHA
    mha: {
      title: 'Multi-Head Attention',
      subtitle: `${nh} heads of ${hd} dims${kv !== nh ? ` · ${kv} shared KV heads (GQA)` : ''}`,
      nodes: [
        { id: 'V', kind: 'terminal', label: 'V', col: -1.15, row: 0 },
        { id: 'K', kind: 'terminal', label: 'K', col: 0, row: 0 },
        { id: 'Q', kind: 'terminal', label: 'Q', col: 1.15, row: 0 },
        { id: 'lv', kind: 'linear', label: 'Linear', col: -1.15, row: 1, w: 92, stack: kv, ...lin(H, kv * hd) },
        { id: 'lk', kind: 'linear', label: 'Linear', col: 0, row: 1, w: 92, stack: kv, ...lin(H, kv * hd) },
        { id: 'lq', kind: 'linear', label: 'Linear', col: 1.15, row: 1, w: 92, stack: nh, ...lin(H, nh * hd) },
        {
          id: 'sdpa', kind: 'composite', label: 'Scaled Dot-Product Attention', col: 0, row: 2,
          w: 300, stack: nh, expand: 'sdpa', note: 'no parameters of its own',
        },
        { id: 'concat', kind: 'concat', label: 'Concat', col: 0, row: 3, note: `${nh} × ${hd} = ${(nh * hd).toLocaleString()}` },
        { id: 'lo', kind: 'linear', label: 'Linear', col: 0, row: 4, ...lin(nh * hd, H) },
        { id: 'out', kind: 'terminal', label: 'out', col: 0, row: 5, note: `[${tokens}, ${H.toLocaleString()}]` },
      ],
      edges: [
        { from: 'V', to: 'lv' }, { from: 'K', to: 'lk' }, { from: 'Q', to: 'lq' },
        { from: 'lv', to: 'sdpa' }, { from: 'lk', to: 'sdpa' }, { from: 'lq', to: 'sdpa' },
        { from: 'sdpa', to: 'concat' }, { from: 'concat', to: 'lo' }, { from: 'lo', to: 'out' },
      ],
      note: `Each head attends for its own reason, over its own ${hd}-dimensional slice. ${
        kv !== nh
          ? `Note the asymmetry: ${nh} query heads but only ${kv} key/value heads, shared between them — grouped-query attention, which shrinks the KV cache ${(nh / kv).toFixed(0)}× at almost no quality cost.`
          : ''
      }`,
    },

    // ----------------------------------------------------------------- SDPA
    sdpa: {
      title: 'Scaled Dot-Product Attention',
      subtitle: `one head · d = ${hd}, so the divisor is √${hd} = ${Math.sqrt(hd).toFixed(1)}`,
      nodes: [
        { id: 'Q', kind: 'terminal', label: 'Q', col: -0.55, row: 0, note: `[${tokens}, ${hd}]` },
        { id: 'K', kind: 'terminal', label: 'K', col: 0.55, row: 0, note: `[${tokens}, ${hd}]` },
        { id: 'V', kind: 'terminal', label: 'V', col: 1.9, row: 0, note: `[${tokens}, ${hd}]` },
        {
          id: 'qk', kind: 'matmul', label: 'MatMul', col: 0, row: 1, figure: 'matmul',
          note: 'Q·Kᵀ', shape: `→ [${tokens}, ${tokens}]`, cost: F(2 * tokens * tokens * hd),
        },
        { id: 'scale', kind: 'scale', label: 'Scale', col: 0, row: 2, note: `÷ ${Math.sqrt(hd).toFixed(1)}` },
        { id: 'mask', kind: 'mask', label: 'Mask (opt.)', col: 0, row: 3, note: 'causal' },
        { id: 'sm', kind: 'softmax', label: 'SoftMax', col: 0, row: 4, note: 'rows sum to 1', figure: 'attention' },
        {
          id: 'av', kind: 'matmul', label: 'MatMul', col: 0, row: 5, figure: 'matmul',
          note: 'weights·V', shape: `→ [${tokens}, ${hd}]`, cost: F(2 * tokens * tokens * hd),
        },
        { id: 'out', kind: 'terminal', label: 'head out', col: 0, row: 6, note: `[${tokens}, ${hd}]` },
      ],
      edges: [
        { from: 'Q', to: 'qk' }, { from: 'K', to: 'qk' },
        { from: 'qk', to: 'scale' }, { from: 'scale', to: 'mask' },
        { from: 'mask', to: 'sm' }, { from: 'sm', to: 'av' },
        { from: 'V', to: 'av' }, { from: 'av', to: 'out' },
      ],
      note: `The only box here with no parameters at all — attention is a way of <i>using</i> numbers, not storing them. Both MatMuls cost ${F(2 * tokens * tokens * hd)} at ${tokens} tokens and grow with the square of context. Click either to see the operation itself.`,
    },

    // ------------------------------------------------------------------ MoE
    ...(isMoE
      ? {
          moe: {
            title: 'Mixture of Experts',
            subtitle: `${E} experts, top-${topk} per token · ${M(E * expertParams)} stored, ${M(topk * expertParams)} used`,
            nodes: [
              { id: 'in', kind: 'terminal', label: 'x', col: 0, row: 0, note: `[${tokens}, ${H.toLocaleString()}]` },
              { id: 'router', kind: 'router', label: 'Router', col: -1.3, row: 1, w: 96, ...lin(H, E), note: `scores ${E}` },
              { id: 'topk', kind: 'mask', label: `Top-${topk}`, col: -1.3, row: 2, w: 96, note: 'keep the best' },
              {
                id: 'experts', kind: 'moe', label: 'Expert FFN', col: 0.55, row: 2, w: 150, stack: E,
                shape: `${H.toLocaleString()}→${ffn.toLocaleString()}→${H.toLocaleString()}`,
                cost: `${M(expertParams)} params each`,
              },
              { id: 'mix', kind: 'concat', label: 'Weighted sum', col: 0, row: 3, w: 150, note: `${topk} outputs` },
              { id: 'out', kind: 'terminal', label: 'out', col: 0, row: 4, note: `[${tokens}, ${H.toLocaleString()}]` },
            ],
            edges: [
              { from: 'in', to: 'router' }, { from: 'in', to: 'experts' },
              { from: 'router', to: 'topk' }, { from: 'topk', to: 'experts' },
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
