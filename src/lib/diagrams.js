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

  const lin = (a, b, why) => ({
    shape: `${a.toLocaleString()}→${b.toLocaleString()}`,
    cost: `${M(a * b)} params · ${F(2 * a * b)}`,
    expand: linearPrim(a, b, why),
    why,
  })
  const expertParams = 3 * H * moeFfn
  const denseFfnParams = 3 * H * ffn

  // ------------------------------------------------------------- primitives
  /**
   * Diagrams for the leaf operations, generated on demand and keyed by shape, so
   * every composite box can be opened until nothing is left but multiplies, adds
   * and reductions with their dimensions on show.
   */
  const prims = {}
  const reg = (id, make) => {
    if (!prims[id]) prims[id] = make()
    return id
  }
  const N = (n) => Math.round(n).toLocaleString()

  /** A matrix multiply, decomposed into the multiplies and the reduction. */
  const matmulPrim = (m, k, n) =>
    reg(`mm-${m}-${k}-${n}`, () => ({
      title: `MatMul · [${N(m)}×${N(k)}] × [${N(k)}×${N(n)}]`,
      subtitle: `${N(m * n)} output cells, each one a dot product of length ${N(k)}`,
      nodes: [
        { id: 'a', kind: 'terminal', label: 'A', col: -0.7, row: 0, note: `[${N(m)}, ${N(k)}]` },
        { id: 'b', kind: 'terminal', label: 'B', col: 0.7, row: 0, note: `[${N(k)}, ${N(n)}]` },
        {
          id: 'mul', kind: 'elementwise', label: 'Multiply', col: 0, row: 1, w: 132,
          note: 'pairwise', cost: `${N(m * n * k)} multiplies`,
          why: `Every output cell pairs one row of A with one column of B and multiplies them entry by entry — ${N(k)} products per cell.`,
        },
        {
          id: 'sum', kind: 'reduce', label: 'Sum', col: 0, row: 2, w: 132,
          note: `${N(k)} terms per cell`, cost: `${N(m * n * (k - 1))} adds`,
          why: `Each cell's ${N(k)} products are added into one number. In parallel this is a tree of depth log₂(${N(k)}) ≈ ${Math.ceil(Math.log2(k))}.`,
        },
        { id: 'out', kind: 'terminal', label: 'C', col: 0, row: 3, note: `[${N(m)}, ${N(n)}]` },
      ],
      edges: [
        { from: 'a', to: 'mul' }, { from: 'b', to: 'mul' },
        { from: 'mul', to: 'sum' }, { from: 'sum', to: 'out' },
      ],
      note: `This is the bottom: multiply, then add. <b>${N(2 * m * n * k)} FLOPs</b>, which is the 2·M·N·K everything else is built from.`,
    }))

  /** x @ W — a Linear layer is one matrix multiply against a stored weight. */
  const linearPrim = (a, b, why) =>
    reg(`lin-${a}-${b}`, () => ({
      title: `Linear · ${N(a)} → ${N(b)}`,
      subtitle: `${M(a * b)} stored parameters`,
      nodes: [
        { id: 'x', kind: 'terminal', label: 'x', col: -0.75, row: 0, note: `[${tokens}, ${N(a)}]` },
        {
          id: 'w', kind: 'weight', label: 'W', col: 0.75, row: 0, w: 96,
          note: `[${N(a)}, ${N(b)}]`, cost: `${M(a * b)} params`,
          why: 'A stored weight matrix — this is where the model\'s knowledge actually lives. It does not change between tokens.',
        },
        {
          id: 'mm', kind: 'matmul', label: 'MatMul', col: 0, row: 1, expand: matmulPrim(tokens, a, b),
          cost: `${F(2 * tokens * a * b)}`, why: 'Open this to see the multiplies and adds themselves.',
        },
        { id: 'out', kind: 'terminal', label: 'out', col: 0, row: 2, note: `[${tokens}, ${N(b)}]` },
      ],
      edges: [{ from: 'x', to: 'mm' }, { from: 'w', to: 'mm' }, { from: 'mm', to: 'out' }],
      note: why ?? 'A projection: every output feature is a weighted sum of every input feature.',
    }))

  /** Numerically stable softmax over each row. */
  const softmaxPrim = (rows, cols) =>
    reg(`sm-${rows}-${cols}`, () => ({
      title: `SoftMax · over each of ${N(rows)} rows`,
      subtitle: `${N(cols)} values per row → ${N(cols)} weights that sum to 1`,
      nodes: [
        { id: 'x', kind: 'terminal', label: 'scores', col: 0, row: 0, note: `[${N(rows)}, ${N(cols)}]` },
        { id: 'max', kind: 'reduce', label: 'Row max', col: 0, row: 1, w: 124, cost: `${N(rows * (cols - 1))} comparisons`,
          why: 'Not part of the definition — subtracting the row maximum first is what stops exp() overflowing. It cannot change the result, because softmax is shift-invariant.' },
        { id: 'sub', kind: 'elementwise', label: 'Subtract', col: 0, row: 2, w: 124, cost: `${N(rows * cols)} subtracts` },
        { id: 'exp', kind: 'elementwise', label: 'exp', col: 0, row: 3, w: 124, cost: `${N(rows * cols)} exponentials`,
          why: 'The expensive part on real hardware: transcendentals run on separate units from the multiply-add pipelines, which is why attention kernels are often bounded by this and not by the matmuls.' },
        { id: 'sum', kind: 'reduce', label: 'Row sum', col: 0, row: 4, w: 124, cost: `${N(rows * (cols - 1))} adds` },
        { id: 'div', kind: 'elementwise', label: 'Divide', col: 0, row: 5, w: 124, cost: `${N(rows * cols)} divides` },
        { id: 'out', kind: 'terminal', label: 'weights', col: 0, row: 6, note: `rows sum to 1` },
      ],
      edges: [
        { from: 'x', to: 'max' }, { from: 'max', to: 'sub' }, { from: 'sub', to: 'exp' },
        { from: 'exp', to: 'sum' }, { from: 'sum', to: 'div' }, { from: 'div', to: 'out' },
      ],
      note: 'Five passes over the data and no parameters at all. Two of them are reductions along the row, which is why a fused kernel matters: done naively this reads the scores from memory five times.',
    }))

  /** Causal mask: add −∞ above the diagonal. */
  const maskPrim = (T) =>
    reg(`mask-${T}`, () => ({
      title: `Causal mask · ${N(T)} × ${N(T)}`,
      subtitle: `${N((T * (T - 1)) / 2)} of ${N(T * T)} entries forbidden`,
      nodes: [
        { id: 'x', kind: 'terminal', label: 'scores', col: -0.7, row: 0, note: `[${N(T)}, ${N(T)}]` },
        { id: 'm', kind: 'weight', label: 'mask', col: 0.7, row: 0, w: 96, note: 'upper triangle', cost: 'no parameters',
          why: 'Not learned — a fixed pattern. Row i may only see columns 0…i, so everything above the diagonal is forbidden.' },
        { id: 'add', kind: 'elementwise', label: 'Add −∞', col: 0, row: 1, w: 132, cost: `${N((T * (T - 1)) / 2)} writes`,
          why: 'Adding −∞ before the softmax makes exp() return exactly zero, so the row still normalises correctly. Masking after the softmax would leave the rows unnormalised.' },
        { id: 'out', kind: 'terminal', label: 'masked', col: 0, row: 2, note: 'lower triangular' },
      ],
      edges: [{ from: 'x', to: 'add' }, { from: 'm', to: 'add' }, { from: 'add', to: 'out' }],
      note: `Almost half the score matrix is thrown away. Kernels that know the mask is triangular simply never compute that half — which is where the factor-of-two saving in causal attention comes from.`,
    }))

  /** Divide every score by a constant. */
  const scalePrim = (rows, cols, d) =>
    reg(`scale-${rows}-${cols}-${d}`, () => ({
      title: `Scale · ÷ √${d} = ${Math.sqrt(d).toFixed(2)}`,
      subtitle: `${N(rows * cols)} values, one multiply each`,
      nodes: [
        { id: 'x', kind: 'terminal', label: 'scores', col: 0, row: 0, note: `[${N(rows)}, ${N(cols)}]` },
        { id: 'mul', kind: 'elementwise', label: 'Multiply', col: 0, row: 1, w: 150, note: `× ${(1 / Math.sqrt(d)).toFixed(4)}`, cost: `${N(rows * cols)} multiplies`,
          why: `A dot product of ${d} terms has standard deviation √${d}, so without this the scores grow with head width and push the softmax toward one-hot — where its gradient vanishes.` },
        { id: 'out', kind: 'terminal', label: 'scaled', col: 0, row: 2 },
      ],
      edges: [{ from: 'x', to: 'mul' }, { from: 'mul', to: 'out' }],
      note: 'The cheapest box in the whole diagram, and the one most often skipped in explanations. It exists to keep the softmax in a range where it still has a gradient.',
    }))

  /** RMSNorm: rescale each token vector to unit root-mean-square. */
  const rmsnormPrim = () =>
    reg(`rms-${H}`, () => ({
      title: `RMSNorm · over ${N(H)} features`,
      subtitle: `${M(H)} parameters — one gain per feature`,
      nodes: [
        { id: 'x', kind: 'terminal', label: 'x', col: 0, row: 0, note: `[${tokens}, ${N(H)}]` },
        { id: 'sq', kind: 'elementwise', label: 'Square', col: 0, row: 1, w: 124, cost: `${N(tokens * H)} multiplies` },
        { id: 'mean', kind: 'reduce', label: 'Mean', col: 0, row: 2, w: 124, cost: `${N(tokens * (H - 1))} adds`,
          why: `Averaged across the ${N(H)} features of each token — so tokens are normalised independently and nothing leaks between positions.` },
        { id: 'rs', kind: 'elementwise', label: 'rsqrt', col: 0, row: 3, w: 124, cost: `${N(tokens)} inverse square roots` },
        { id: 'mul', kind: 'elementwise', label: 'Multiply', col: 0, row: 4, w: 124, cost: `${N(tokens * H)} multiplies` },
        { id: 'g', kind: 'weight', label: 'gain γ', col: 1.1, row: 4, w: 96, note: `[${N(H)}]`, cost: `${M(H)} params`,
          why: 'A learned per-feature scale, so the layer can undo the normalisation where it needs to.' },
        { id: 'out', kind: 'terminal', label: 'out', col: 0, row: 5 },
      ],
      edges: [
        { from: 'x', to: 'sq' }, { from: 'sq', to: 'mean' }, { from: 'mean', to: 'rs' },
        { from: 'rs', to: 'mul' }, { from: 'g', to: 'mul' }, { from: 'mul', to: 'out' },
      ],
      note: 'Note what is missing compared with LayerNorm: no mean subtraction. RMSNorm keeps only the scale term, which is cheaper and works about as well.',
    }))

  /** Elementwise add — the residual connection. */
  const addPrim = () =>
    reg(`add-${H}`, () => ({
      title: 'Residual add',
      subtitle: `${N(tokens * H)} additions, no parameters`,
      nodes: [
        { id: 'a', kind: 'terminal', label: 'x', col: -0.7, row: 0, note: `[${tokens}, ${N(H)}]` },
        { id: 'b', kind: 'terminal', label: 'f(x)', col: 0.7, row: 0, note: `[${tokens}, ${N(H)}]` },
        { id: 'add', kind: 'elementwise', label: 'Add', col: 0, row: 1, w: 132, cost: `${N(tokens * H)} adds`,
          why: 'The block writes into the stream rather than replacing it. Gradients reach earlier layers through this addition unchanged, which is what makes a deep stack trainable.' },
        { id: 'out', kind: 'terminal', label: 'x′', col: 0, row: 2 },
      ],
      edges: [{ from: 'a', to: 'add' }, { from: 'b', to: 'add' }, { from: 'add', to: 'out' }],
      note: 'The single cheapest and most important operation in the architecture.',
    }))

  /** Concat is a view, not work. */
  const concatPrim = () =>
    reg('concat', () => ({
      title: 'Concat',
      subtitle: 'a relabelling, not a computation',
      nodes: [
        { id: 'x', kind: 'terminal', label: `${nh} heads`, col: 0, row: 0, note: `${nh} × [${tokens}, ${N(sdpaOut)}]` },
        { id: 'v', kind: 'elementwise', label: 'View', col: 0, row: 1, w: 150, note: 'no data moves', cost: '0 FLOPs',
          why: 'The heads are already laid out side by side in memory, so joining them is a change of interpretation rather than a copy.' },
        { id: 'out', kind: 'terminal', label: 'joined', col: 0, row: 2, note: `[${tokens}, ${N(nh * sdpaOut)}]` },
      ],
      edges: [{ from: 'x', to: 'v' }, { from: 'v', to: 'out' }],
      note: 'Worth opening once precisely because there is nothing here. Concat costs nothing; the mixing that follows it is done by the output projection.',
    }))

  /** Embedding lookup. */
  const gatherPrim = () =>
    reg('gather', () => ({
      title: 'Embedding lookup',
      subtitle: `${M(V * H)} parameters, ${N(tokens * H)} values read`,
      nodes: [
        { id: 'ids', kind: 'terminal', label: 'token ids', col: -0.7, row: 0, note: `[${tokens}]` },
        { id: 'tab', kind: 'weight', label: 'table', col: 0.7, row: 0, w: 100, note: `[${N(V)}, ${N(H)}]`, cost: `${M(V * H)} params`,
          why: `One row per vocabulary entry. Large in parameters — ${M(V * H)} — but only the rows for tokens actually present are ever touched.` },
        { id: 'g', kind: 'elementwise', label: 'Gather', col: 0, row: 1, w: 132, note: 'index select', cost: '0 FLOPs',
          why: 'No arithmetic at all: this reads rows by index. It is the clearest case of parameter count being a poor guide to cost.' },
        { id: 'out', kind: 'terminal', label: 'x', col: 0, row: 2, note: `[${tokens}, ${N(H)}]` },
      ],
      edges: [{ from: 'ids', to: 'g' }, { from: 'tab', to: 'g' }, { from: 'g', to: 'out' }],
      note: 'A lookup, not a multiply — which is why this box carries no FLOPs anywhere in the diagrams above.',
    }))

  /** A SwiGLU feed-forward: two projections up, a gate, one projection down. */
  const ffnPrim = (mid, why) =>
    reg(`ffn-${H}-${mid}`, () => ({
      title: `Feed-forward · ${N(H)} → ${N(mid)} → ${N(H)}`,
      subtitle: `${M(3 * H * mid)} parameters`,
      nodes: [
        { id: 'x', kind: 'terminal', label: 'x', col: 0, row: 0, note: `[${tokens}, ${N(H)}]` },
        { id: 'gate', kind: 'linear', label: 'W_gate', col: -0.85, row: 1, w: 100, ...lin(H, mid,
          'Produces the gate. Passed through SiLU and used to decide, per feature, how much of the other branch survives.') },
        { id: 'up', kind: 'linear', label: 'W_up', col: 0.85, row: 1, w: 100, ...lin(H, mid,
          'Projects into the wide intermediate space — the branch that carries the content, as opposed to the gate that filters it.') },
        { id: 'act', kind: 'elementwise', label: 'SiLU', col: -0.85, row: 2, w: 100, cost: `${N(tokens * mid)} ops`,
          why: 'x·σ(x). Smooth, and unlike ReLU it lets a little negative signal through, which trains better.' },
        { id: 'gmul', kind: 'elementwise', label: 'Multiply', col: 0, row: 3, w: 124, cost: `${N(tokens * mid)} multiplies`,
          why: 'The gating step: the activated gate scales the content branch feature by feature. This multiplication is what makes it *gated* rather than a plain MLP.' },
        { id: 'down', kind: 'linear', label: 'W_down', col: 0, row: 4, w: 110, ...lin(mid, H,
          `Projects the ${N(mid)}-wide intermediate result back down to the ${N(H)}-wide residual stream.`) },
        { id: 'out', kind: 'terminal', label: 'out', col: 0, row: 5, note: `[${tokens}, ${N(H)}]` },
      ],
      edges: [
        { from: 'x', to: 'gate' }, { from: 'x', to: 'up' }, { from: 'gate', to: 'act' },
        { from: 'act', to: 'gmul' }, { from: 'up', to: 'gmul' },
        { from: 'gmul', to: 'down' }, { from: 'down', to: 'out' },
      ],
      note: why ?? `Wide in the middle — ${(mid / H).toFixed(1)}× the model width — then back. Three matrices, and two thirds of the parameters in a typical dense layer.`,
    }))

  /** Top-k selection over expert scores. */
  const topkPrim = () =>
    reg(`topk-${E}-${topk}`, () => ({
      title: `Top-${topk} of ${E}`,
      subtitle: `per token, ${N(tokens)} times`,
      nodes: [
        { id: 'sc', kind: 'terminal', label: 'scores', col: 0, row: 0, note: `[${tokens}, ${E}]` },
        { id: 'sel', kind: 'reduce', label: `Select ${topk}`, col: 0, row: 1, w: 150, cost: `${N(tokens * E)} comparisons`,
          why: `A partial sort. Cheap in arithmetic, but it decides which expert weights have to be fetched from memory — so this tiny operation sets the bandwidth cost of the entire layer.` },
        { id: 'out', kind: 'terminal', label: 'indices', col: 0, row: 2, note: `[${tokens}, ${topk}]` },
      ],
      edges: [{ from: 'sc', to: 'sel' }, { from: 'sel', to: 'out' }],
      note: `This is where sparsity is created. Everything after it runs on ${((100 * topk) / E).toFixed(0)}% of the layer.`,
    }))



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
        { id: 'embed', kind: 'embed', label: 'Embedding', col: 0, row: 1, ...lin(V, H), cost: `${M(V * H)} params · gather`, expand: gatherPrim(),
          why: 'Turns token ids into vectors by reading rows from a table. Large in parameters, nearly free to run.' },
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
        { id: 'norm', kind: 'norm', label: 'Final norm', col: 0, row: rowTop, note: 'RMSNorm', expand: rmsnormPrim(),
          why: 'One last normalisation before the vocabulary projection.' },
        { id: 'head', kind: 'linear', label: 'LM head', col: 0, row: rowTop + 1, ...lin(H, V,
          `Scores every one of the ${N(V)} vocabulary entries against the final hidden state. Unlike the embedding this is a real matrix multiply, and its weights are read in full on every single step.`) },
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
        { id: 'n1', kind: 'norm', label: 'RMSNorm', col: 0, row: 1, expand: rmsnormPrim(),
          why: 'Normalises each token before attention reads it. Pre-norm — the stream itself is never normalised, only the copy handed to the block.' },
        { id: 'attn', kind: 'composite', label: isMLA ? 'Latent Attention' : 'Multi-Head Attention', col: 0, row: 2, w: 200, expand: 'mha', cost: `${M(attnParams)} params` },
        { id: 'add1', kind: 'add', label: '+', col: 0, row: 3, w: 44, expand: addPrim(),
          why: 'Attention\'s output is added back to the stream that entered the block — the dashed line on the left. The stream is never replaced, only written into.' },
        { id: 'n2', kind: 'norm', label: 'RMSNorm', col: 0, row: 4, expand: rmsnormPrim(),
          why: 'The same normalisation again, this time for the feed-forward half.' },
        {
          id: 'ffn', kind: isMoE ? 'moe' : 'linear',
          label: isMoE ? 'Mixture of Experts' : 'Feed-forward', col: 0, row: 5, w: 200,
          expand: isMoE ? 'moe' : ffnPrim(ffn), cost: `${M(isMoE ? moeParams : denseFfnParams)} params`,
          note: isMoE ? `top-${topk} of ${E}` : `${H.toLocaleString()}→${ffn.toLocaleString()}→${H.toLocaleString()}`,
        },
        { id: 'add2', kind: 'add', label: '+', col: 0, row: 6, w: 44, expand: addPrim(),
          why: 'And again after the feed-forward. Two additions per block, and every layer in the stack writes into the same stream.' },
        { id: 'out', kind: 'terminal', label: 'x′', col: 0, row: 7, note: `[${tokens}, ${H.toLocaleString()}]` },
      ],
      edges: [
        { from: 'in', to: 'n1' }, { from: 'n1', to: 'attn' }, { from: 'attn', to: 'add1' },
        // Residuals route out to the side, so it is visible where they come from.
        { from: 'in', to: 'add1', via: -1.9, residual: true, label: 'residual' },
        { from: 'add1', to: 'n2' }, { from: 'n2', to: 'ffn' }, { from: 'ffn', to: 'add2' },
        { from: 'add1', to: 'add2', via: 1.9, residual: true, label: 'residual' },
        { from: 'add2', to: 'out' },
      ],
      note: 'The two dashed blue paths are the residual connections. The left one carries the block\'s input straight past attention; the right one carries the post-attention stream straight past the feed-forward. Both halves are written <i>into</i> the stream rather than replacing it, and that is what lets gradients reach the bottom of a deep stack.',
    },

    // ---------------------------------------------------- dense block variant
    ...(denseLayers && moeLayers
      ? {
          blockDense: {
            title: 'Dense block',
            subtitle: `the first ${denseLayers} layers · ${M(attnParams + denseFfnParams)} parameters`,
            nodes: [
              { id: 'in', kind: 'terminal', label: 'x', col: 0, row: 0, note: `[${tokens}, ${H.toLocaleString()}]` },
              { id: 'n1', kind: 'norm', label: 'RMSNorm', col: 0, row: 1, expand: rmsnormPrim(),
          why: 'Normalises each token before attention reads it. Pre-norm — the stream itself is never normalised, only the copy handed to the block.' },
              { id: 'attn', kind: 'composite', label: isMLA ? 'Latent Attention' : 'Multi-Head Attention', col: 0, row: 2, w: 200, expand: 'mha', cost: `${M(attnParams)} params` },
              { id: 'add1', kind: 'add', label: '+', col: 0, row: 3, w: 44, note: 'residual' },
              { id: 'n2', kind: 'norm', label: 'RMSNorm', col: 0, row: 4 },
              { id: 'ffn', kind: 'linear', label: 'Feed-forward', col: 0, row: 5, w: 200, expand: ffnPrim(ffn), cost: `${M(denseFfnParams)} params`, note: `${H.toLocaleString()}→${ffn.toLocaleString()}→${H.toLocaleString()}` },
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
          subtitle: `${nh} query heads of ${hd} dims${kv !== nh ? ` sharing ${kv} key/value heads (GQA)` : ''}`,
          nodes: [
            { id: 'x', kind: 'terminal', label: 'x', col: 0, row: 0, note: `[${tokens}, ${N(H)}]` },
            {
              id: 'lq', kind: 'linear', label: 'W_Q', col: -1.15, row: 1, w: 96,
              note: `→ ${nh} heads`, ...lin(H, nh * hd,
                `Produces the queries — what each token is looking for. One fused matrix of ${N(H)}×${N(nh * hd)}, whose output is then read as ${nh} separate ${hd}-dim heads.`),
            },
            {
              id: 'lk', kind: 'linear', label: 'W_K', col: 0, row: 1, w: 96,
              note: `→ ${kv} heads`, ...lin(H, kv * hd,
                `Produces the keys — what each token offers.${kv !== nh ? ` Only ${kv} heads' worth, because ${nh / kv} query heads share each key head. That is grouped-query attention, and it shrinks the KV cache ${nh / kv}×.` : ''}`),
            },
            {
              id: 'lv', kind: 'linear', label: 'W_V', col: 1.15, row: 1, w: 96,
              note: `→ ${kv} heads`, ...lin(H, kv * hd,
                `Produces the values — what each token contributes once it has been selected. Same head count as the keys, and cached alongside them.`),
            },
            {
              id: 'sdpa', kind: 'composite', label: 'Scaled Dot-Product Attention', col: 0, row: 2,
              w: 300, stack: nh, expand: 'sdpa', note: 'no parameters of its own',
              why: `Genuinely ${nh} independent computations — unlike the projections above, which are one matrix each. Every head attends over its own ${hd}-dimensional slice, for its own reasons.`,
            },
            {
              id: 'concat', kind: 'concat', label: 'Concat', col: 0, row: 3, expand: concatPrim(),
              note: `${nh} × ${hd} = ${N(nh * hd)}`,
              why: 'Lays the heads side by side. Free — no data moves; open it to see why.',
            },
            {
              id: 'lo', kind: 'linear', label: 'W_O', col: 0, row: 4, ...lin(nh * hd, H,
                `The output projection, and the only place the heads get to interact. Each head has produced a ${hd}-dim answer in its own subspace; this mixes all ${N(nh * hd)} of those numbers back down to the ${N(H)}-wide residual stream. Without it the heads would write into disjoint slices and never combine.`),
            },
            { id: 'out', kind: 'terminal', label: 'out', col: 0, row: 5, note: `[${tokens}, ${N(H)}]` },
          ],
          edges: [
            { from: 'x', to: 'lq' }, { from: 'x', to: 'lk' }, { from: 'x', to: 'lv' },
            { from: 'lq', to: 'sdpa' }, { from: 'lk', to: 'sdpa' }, { from: 'lv', to: 'sdpa' },
            { from: 'sdpa', to: 'concat' }, { from: 'concat', to: 'lo' }, { from: 'lo', to: 'out' },
          ],
          note: `Three projections of the same input, then ${nh} independent attentions, then one projection back.${
            kv !== nh
              ? ` Note the asymmetry: ${nh} query heads against only ${kv} key/value heads. The queries are cheap to make and thrown away; the keys and values are cached for every token in the context, so there are fewer of them on purpose.`
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
          id: 'qk', kind: 'matmul', label: 'MatMul', col: 0, row: 1,
          note: 'Q·Kᵀ', shape: `→ [${tokens}, ${tokens}]`, cost: F(2 * tokens * tokens * sdpaDim),
          expand: matmulPrim(tokens, sdpaDim, tokens),
          why: `Every query against every key: ${N(tokens * tokens)} dot products of length ${sdpaDim}. This is the step that grows with the square of context length.`,
        },
        { id: 'scale', kind: 'scale', label: 'Scale', col: 0, row: 2, note: `÷ ${Math.sqrt(sdpaDim).toFixed(1)}`,
          expand: scalePrim(tokens, tokens, sdpaDim),
          why: `Divides by √${sdpaDim} to keep the scores at unit variance whatever the head width, so the softmax below still has a usable gradient.` },
        { id: 'mask', kind: 'mask', label: 'Mask (opt.)', col: 0, row: 3, note: 'causal',
          expand: maskPrim(tokens),
          why: 'Forbids looking to the right by adding −∞ above the diagonal. Optional in general; required whenever the model generates left to right.' },
        { id: 'sm', kind: 'softmax', label: 'SoftMax', col: 0, row: 4, note: 'rows sum to 1',
          expand: softmaxPrim(tokens, tokens),
          why: 'Turns each row of scores into weights that sum to one. No parameters, but five passes over the data — open it to see them.' },
        {
          id: 'av', kind: 'matmul', label: 'MatMul', col: 0, row: 5,
          note: 'weights·V', shape: `→ [${tokens}, ${sdpaOut}]`, cost: F(2 * tokens * tokens * sdpaOut),
          expand: matmulPrim(tokens, tokens, sdpaOut),
          why: 'Each token\'s output is a weighted blend of every value row, using the weights just computed. This is where attention actually moves information between positions.',
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
              { id: 'router', kind: 'router', label: 'Router', col: -1.3, row: 1, w: 96, note: `scores ${E}`, ...lin(H, E,
                `A tiny matrix — ${M(H * E)} parameters — that scores all ${E} experts for each token. Trivial arithmetic, but it decides which expert weights must be fetched, so it sets the memory bill for the whole layer.`) },
              { id: 'topk', kind: 'mask', label: `Top-${topk}`, col: -1.3, row: 2, w: 96, note: 'keep the best',
                expand: topkPrim(),
                why: `Keeps the ${topk} highest-scoring experts and discards the rest. This is the step that creates the sparsity.` },
              {
                id: 'experts', kind: 'moe', label: 'Expert FFN', col: 0.55, row: 2, w: 150, stack: E,
                shape: `${H.toLocaleString()}→${moeFfn.toLocaleString()}→${H.toLocaleString()}`,
                cost: `${M(expertParams)} params each`,
                expand: ffnPrim(moeFfn),
                why: `An ordinary feed-forward network. There are ${E} of them, identical in shape and different in weights, and only ${topk} run for any given token.`,
              },
              ...(nShared
                ? [{ id: 'shared', kind: 'moe', label: 'Shared expert', col: -1.3, row: 3, w: 130, note: 'always runs', cost: `${M(nShared * expertParams)} params` }]
                : []),
              { id: 'mix', kind: 'concat', label: 'Weighted sum', col: 0.4, row: 4, w: 160, note: `${topk + nShared} outputs`,
                why: `Combines the chosen experts' outputs, weighted by their router scores, so the layer's output is a blend rather than a hard pick.` },
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
      : {}),

    // Every primitive the specs above asked for, keyed by shape.
    ...prims,
  }
}
