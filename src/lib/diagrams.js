/**
 * The diagram specs, built from a real config.json so every box carries the
 * model's own numbers rather than symbols.
 *
 * A spec is nodes on a grid (col across, row upward from 0) plus edges. Nodes
 * with `expand` open another spec; nodes with `figure` jump to one of the
 * article's interactive figures. An edge may name a `lane` — a clear column to
 * route up — and may be marked `residual` or `control`.
 *
 * Where behaviour differs between architectures it is read from the config and
 * the source is named in a comment, because these are exactly the details that
 * get glossed over: routers do not all score the same way, activations are not
 * all plain SwiGLU, and RoPE does not touch every dimension.
 */

const M = (n) => (n >= 1e9 ? `${(n / 1e9).toFixed(2)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : `${(n / 1e3).toFixed(0)}k`)
const F = (n) => (n >= 1e9 ? `${(n / 1e9).toFixed(1)} GFLOP` : `${(n / 1e6).toFixed(0)} MFLOP`)

/** hidden_act as a config writes it, versus what people call it. */
const ACT_NAMES = {
  silu: 'SiLU',
  swish: 'SiLU',
  gelu: 'GELU',
  gelu_new: 'GELU',
  gelu_pytorch_tanh: 'GELU',
  relu: 'ReLU',
}
const GLU_NAMES = { SiLU: 'SwiGLU', GELU: 'GEGLU', ReLU: 'ReGLU' }

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

  // Rotary embeddings. Under MLA only the qk_rope slice is rotated; under plain
  // MHA/GQA the whole head is.
  const ropeTheta = cfg.rope_theta ?? cfg.rope_parameters?.rope_theta
  const ropeScale = cfg.rope_scaling ?? null

  // Activation. Every config here says silu, but reading it is what makes the
  // diagram generalise — and gpt-oss's swiglu_limit says the gate is clamped.
  const actName = ACT_NAMES[cfg.hidden_act] ?? cfg.hidden_act ?? 'SiLU'
  const gluName = GLU_NAMES[actName] ?? `${actName}-gated`
  const swigluLimit = cfg.swiglu_limit ?? null
  const gluAlpha = swigluLimit != null ? 1.702 : null // GptOssExperts.alpha

  // How the router turns hidden states into expert choices. Three real variants:
  //   sigmoid   DeepSeek-V3 / GLM — sigmoid, add e_score_correction_bias, top-k,
  //             gather, normalise, scale by routed_scaling_factor
  //   softmax   Mixtral — softmax over all experts, then top-k, then renormalise
  //   post      gpt-oss — top-k on the raw logits first, softmax over just those
  const scoringFunc = cfg.scoring_func ?? (cfg.model_type === 'gpt_oss' ? 'post' : 'softmax')
  const isSigmoidRouter = scoringFunc === 'sigmoid'
  const isPostSoftmax = scoringFunc === 'post'
  const nGroup = cfg.n_group ?? 1
  const topkGroup = cfg.topk_group ?? 1
  const grouped = nGroup > 1
  // gpt-oss softmaxes the k survivors, which already sum to one, so there is no
  // separate renormalisation. Mixtral has no norm_topk_prob key but its code
  // always renormalises, so `true` is the right default for everything else.
  const normTopk = isPostSoftmax ? false : (cfg.norm_topk_prob ?? true)
  const routedScaling = cfg.routed_scaling_factor ?? 1
  // The auxiliary-loss-free balancer, present only where topk_method says so.
  const hasBias = cfg.topk_method === 'noaux_tc'

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
      note: `This is the bottom: multiply, then add. <b>${N(2 * m * n * k)} FLOPs</b>, which is the 2·M·N·K everything else is built from. Note that it is entirely linear — nothing here can produce a function a single matrix could not.`,
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
  const softmaxPrim = (rows, cols, over) =>
    reg(`sm-${rows}-${cols}`, () => ({
      title: `SoftMax · over each of ${N(rows)} rows`,
      subtitle: `${N(cols)} ${over ?? 'values'} per row → ${N(cols)} weights that sum to 1`,
      nodes: [
        { id: 'x', kind: 'terminal', label: 'scores', col: 0, row: 0, note: `[${N(rows)}, ${N(cols)}]` },
        { id: 'max', kind: 'reduce', label: 'Row max', col: 0, row: 1, w: 124, cost: `${N(rows * (cols - 1))} comparisons`,
          why: 'Not part of the definition — subtracting the row maximum first is what stops exp() overflowing. It cannot change the result, because softmax is shift-invariant.' },
        { id: 'sub', kind: 'elementwise', label: 'Subtract', col: 0, row: 2, w: 124, cost: `${N(rows * cols)} subtracts` },
        { id: 'exp', kind: 'act', label: 'exp', col: 0, row: 3, w: 124, cost: `${N(rows * cols)} exponentials`,
          why: 'The non-linear step, and the expensive one on real hardware: transcendentals run on separate units from the multiply-add pipelines, which is why attention kernels are often bounded by this and not by the matmuls.' },
        { id: 'sum', kind: 'reduce', label: 'Row sum', col: 0, row: 4, w: 124, cost: `${N(rows * (cols - 1))} adds` },
        { id: 'div', kind: 'elementwise', label: 'Divide', col: 0, row: 5, w: 124, cost: `${N(rows * cols)} divides` },
        { id: 'out', kind: 'terminal', label: 'weights', col: 0, row: 6, note: `rows sum to 1` },
      ],
      edges: [
        { from: 'x', to: 'max' }, { from: 'max', to: 'sub' }, { from: 'sub', to: 'exp' },
        { from: 'exp', to: 'sum' }, { from: 'sum', to: 'div' }, { from: 'div', to: 'out' },
      ],
      note: 'Five passes over the data and no parameters at all. Two of them are reductions along the row, which is why a fused kernel matters: done naively this reads the scores from memory five times. The whole non-linearity lives in the one green box.',
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
  const rmsnormPrim = (d = H) =>
    reg(`rms-${d}`, () => ({
      title: `RMSNorm · over ${N(d)} features`,
      subtitle: `${M(d)} parameters — one gain per feature`,
      nodes: [
        { id: 'x', kind: 'terminal', label: 'x', col: 0, row: 0, note: `[${tokens}, ${N(d)}]` },
        { id: 'sq', kind: 'elementwise', label: 'Square', col: 0, row: 1, w: 124, cost: `${N(tokens * d)} multiplies` },
        { id: 'mean', kind: 'reduce', label: 'Mean', col: 0, row: 2, w: 124, cost: `${N(tokens * (d - 1))} adds`,
          why: `Averaged across the ${N(d)} features of each token — so tokens are normalised independently and nothing leaks between positions.` },
        { id: 'rs', kind: 'act', label: 'rsqrt', col: 0, row: 3, w: 124, cost: `${N(tokens)} inverse square roots`,
          why: 'Dividing by a quantity derived from the input is itself non-linear — normalisation is not just bookkeeping, it changes what functions the layer can express.' },
        { id: 'mul', kind: 'elementwise', label: 'Multiply', col: 0, row: 4, w: 124, cost: `${N(tokens * d)} multiplies` },
        { id: 'g', kind: 'weight', label: 'gain γ', col: 1.1, row: 4, w: 96, note: `[${N(d)}]`, cost: `${M(d)} params`,
          why: 'A learned per-feature scale, so the layer can undo the normalisation where it needs to.' },
        { id: 'out', kind: 'terminal', label: 'out', col: 0, row: 5 },
      ],
      edges: [
        { from: 'x', to: 'sq' }, { from: 'sq', to: 'mean' }, { from: 'mean', to: 'rs' },
        { from: 'rs', to: 'mul' }, { from: 'g', to: 'mul', lane: 1.1 }, { from: 'mul', to: 'out' },
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

  /** Rotary position embedding, on whichever slice of the head carries position. */
  const ropePrim = (dim, what) =>
    reg(`rope-${dim}`, () => ({
      title: `RoPE · ${N(dim)} dimensions`,
      subtitle: `${N(dim / 2)} pairs, each rotated by an angle proportional to position`,
      nodes: [
        { id: 'x', kind: 'terminal', label: what ?? 'q, k', col: -0.7, row: 0, note: `[${tokens}, ${N(dim)}]` },
        { id: 'tab', kind: 'weight', label: 'cos, sin', col: 0.75, row: 0, w: 104, note: `θ = ${ropeTheta?.toLocaleString?.() ?? '—'}`, cost: 'no parameters',
          why: `Not learned. Pair j at position i is rotated by i·θ^(−2j/${N(dim)}); a large base θ makes the slowest pair turn slowly, which is what lets the model be stretched to long context${ropeScale ? ` — this config also applies ${ropeScale.rope_type ?? ropeScale.type} scaling ×${ropeScale.factor}` : ''}.` },
        { id: 'rot', kind: 'elementwise', label: 'Rotate pairs', col: 0, row: 1, w: 150,
          cost: `${N(2 * tokens * dim)} multiplies · ${N(tokens * dim)} adds`,
          why: 'Each adjacent pair of features is treated as a point in a plane and turned. Because a rotation preserves length, the dot product of two rotated vectors depends only on the *difference* of their positions — which is the whole trick.' },
        { id: 'out', kind: 'terminal', label: 'rotated', col: 0, row: 2, note: `[${tokens}, ${N(dim)}]` },
      ],
      edges: [{ from: 'x', to: 'rot' }, { from: 'tab', to: 'rot' }, { from: 'rot', to: 'out' }],
      note: 'No parameters and no matrix multiply, yet this is the only place position enters the model at all. Everything else in a transformer is permutation-equivariant.',
    }))

  /** The pointwise activation — the non-linearity in the feed-forward half. */
  const actPrim = () =>
    reg(`act-${actName}${swigluLimit ? '-clamped' : ''}`, () => ({
      title: `${actName} · x·σ(${gluAlpha ? `${gluAlpha}·` : ''}x)`,
      subtitle: 'the only non-linear step in the feed-forward half',
      nodes: [
        { id: 'x', kind: 'terminal', label: 'x', col: 0, row: 0 },
        ...(swigluLimit != null
          ? [{ id: 'cl', kind: 'elementwise', label: 'Clamp', col: 0, row: 1, w: 140, note: `≤ ${swigluLimit}`,
              why: `gpt-oss clamps the gate at ${swigluLimit} and the content branch at ±${swigluLimit} before gating. Not cosmetic: it is what keeps the MXFP4-quantised experts from producing outliers the 4-bit format cannot represent.` }]
          : []),
        { id: 'sig', kind: 'act', label: 'σ(x)', col: 0, row: swigluLimit != null ? 2 : 1, w: 140,
          note: gluAlpha ? `σ(${gluAlpha}·x)` : 'logistic', cost: `1 exponential per value`,
          why: gluAlpha
            ? `The logistic sigmoid, with the input pre-scaled by ${gluAlpha}. That constant is not arbitrary — x·σ(1.702x) is the standard close approximation to GELU, so gpt-oss is computing a GELU-shaped curve through a cheaper route.`
            : 'The logistic sigmoid: squashes to (0,1), smooth everywhere. This is the only non-linear thing that happens; everything around it is multiplication.' },
        { id: 'mul', kind: 'elementwise', label: 'Multiply', col: 0, row: swigluLimit != null ? 3 : 2, w: 140, cost: '1 multiply per value',
          why: 'x·σ(x). Multiplying the input by its own squashed self is what makes the curve non-monotonic — it dips slightly negative near x ≈ −1 instead of flattening to zero like ReLU, and that small dip is most of why it trains better.' },
        { id: 'out', kind: 'terminal', label: `${actName}(x)`, col: 0, row: swigluLimit != null ? 4 : 3 },
      ],
      edges: [
        ...(swigluLimit != null
          ? [{ from: 'x', to: 'cl' }, { from: 'cl', to: 'sig' }, { from: 'cl', to: 'mul', lane: 1.3 }]
          : [{ from: 'x', to: 'sig' }, { from: 'x', to: 'mul', lane: 1.3 }]),
        { from: 'sig', to: 'mul' },
        { from: 'mul', to: 'out' },
      ],
      note: 'Without this box the whole feed-forward network would collapse: a stack of matrix multiplies with nothing non-linear between them is just one matrix multiply. Every green box in these diagrams is load-bearing for that reason.',
    }))

  /** The logistic sigmoid on its own — used by sigmoid-scored routers. */
  const sigmoidPrim = (rows, cols) =>
    reg(`sig-${rows}-${cols}`, () => ({
      title: `Sigmoid · ${N(rows * cols)} values`,
      subtitle: 'one score per expert per token, computed independently',
      nodes: [
        { id: 'x', kind: 'terminal', label: 'logits', col: 0, row: 0, note: `[${N(rows)}, ${N(cols)}]` },
        { id: 's', kind: 'act', label: 'σ(x)', col: 0, row: 1, w: 150, cost: `${N(rows * cols)} exponentials`,
          why: 'Each expert is squashed to (0,1) on its own. Unlike a softmax the scores do not compete for a fixed budget, so a per-expert bias can be added to balance load without changing what the scores mean.' },
        { id: 'out', kind: 'terminal', label: 'scores', col: 0, row: 2, note: 'each in (0, 1)' },
      ],
      edges: [{ from: 'x', to: 's' }, { from: 's', to: 'out' }],
      note: 'The difference from softmax matters: softmax scores sum to one, so raising one expert lowers every other. Sigmoid scores are independent, which is what makes DeepSeek-style auxiliary-loss-free load balancing possible — the bias shifts the ranking without distorting the weights that get used.',
    }))

  /** A gated feed-forward: two projections up, an activation, a gate, one down. */
  const ffnPrim = (mid, why) =>
    reg(`ffn-${H}-${mid}`, () => ({
      title: `Feed-forward · ${N(H)} → ${N(mid)} → ${N(H)}`,
      subtitle: `${gluName} · ${M(3 * H * mid)} parameters`,
      nodes: [
        { id: 'x', kind: 'terminal', label: 'x', col: 0, row: 0, note: `[${tokens}, ${N(H)}]` },
        { id: 'gate', kind: 'linear', label: 'W_gate', col: -0.85, row: 1, w: 100, ...lin(H, mid,
          'Produces the gate. Passed through the activation and used to decide, per feature, how much of the other branch survives.') },
        { id: 'up', kind: 'linear', label: 'W_up', col: 0.85, row: 1, w: 100, ...lin(H, mid,
          'Projects into the wide intermediate space — the branch that carries the content, as opposed to the gate that filters it.') },
        { id: 'act', kind: 'act', label: actName, col: -0.85, row: 2, w: 100, cost: `${N(tokens * mid)} ops`,
          expand: actPrim(),
          why: `The non-linearity, and the only one in this half of the block. Open it: everything else here is a matrix multiply, and without this box all three of them would collapse into one.` },
        { id: 'gmul', kind: 'elementwise', label: 'Multiply', col: 0, row: 3, w: 124,
          note: swigluLimit != null ? '(up + 1) × gate' : undefined,
          cost: `${N(tokens * mid)} multiplies`,
          why: swigluLimit != null
            ? 'The gating step. gpt-oss adds one to the content branch before multiplying, so a zero gate passes the content through unchanged rather than deleting it — a residual inside the activation.'
            : 'The gating step: the activated gate scales the content branch feature by feature. This multiplication is what makes it *gated* rather than a plain MLP.' },
        { id: 'down', kind: 'linear', label: 'W_down', col: 0, row: 4, w: 110, ...lin(mid, H,
          `Projects the ${N(mid)}-wide intermediate result back down to the ${N(H)}-wide residual stream.`) },
        { id: 'out', kind: 'terminal', label: 'out', col: 0, row: 5, note: `[${tokens}, ${N(H)}]` },
      ],
      edges: [
        { from: 'x', to: 'gate' }, { from: 'x', to: 'up' }, { from: 'gate', to: 'act' },
        { from: 'act', to: 'gmul' }, { from: 'up', to: 'gmul', lane: 0.85 },
        { from: 'gmul', to: 'down' }, { from: 'down', to: 'out' },
      ],
      note: why ?? `${
        mid >= H
          ? `Wide in the middle — ${(mid / H).toFixed(1)}× the model width — then back.`
          : `Narrower in the middle than the stream it reads: ${(mid / H).toFixed(2)}× the model width. An expert is individually small; the capacity comes from there being ${E} of them.`
      } Three matrices, and exactly one green box. ${gluName} is that green box applied to one branch and multiplied into the other — take it away and all three matrices collapse into one.`,
    }))

  /** Top-k expert selection, and what comes out of it. */
  const pickPrim = () =>
    reg(`topk-${E}-${topk}`, () => {
      // Rows are stacked in the order this architecture actually applies them,
      // which differs between families, so they are counted rather than hardcoded.
      let row = 0
      const nodes = [
        { id: 'sc', kind: 'terminal', label: isPostSoftmax ? 'logits' : 'scores', col: 0, row: row, note: `[${tokens}, ${E}]` },
        ...(hasBias
          ? [{ id: 'bias', kind: 'weight', label: 'bias b', col: 1.5, row: 0, w: 104, meta: 'below', note: `[${E}]`, cost: 'not trained by loss',
              why: 'The auxiliary-loss-free load balancer: a per-expert offset nudged up or down between steps according to how busy each expert has been. It shifts the ranking used for selection, and is deliberately not added to the weights the outputs are blended with.' }]
          : []),
      ]
      const edges = []
      let prev = 'sc'
      const push = (n) => {
        nodes.push({ ...n, row: ++row })
        edges.push({ from: prev, to: n.id })
        prev = n.id
      }
      if (grouped)
        push({ id: 'grp', kind: 'reduce', label: `Group top-${topkGroup}`, col: 0, w: 150, note: `of ${nGroup} groups`,
          cost: `${N(tokens * nGroup)} group scores`,
          why: `The ${E} experts are partitioned into ${nGroup} groups, one per device. Each group is scored by its best two experts, only the top ${topkGroup} groups survive, and the final choice is made within those — so one token's work never touches more than ${topkGroup} devices, which is what bounds the all-to-all traffic.` })
      push({ id: 'sel', kind: 'reduce', label: `Select ${topk}`, col: 0, w: 150, meta: 'below', cost: `${N(tokens * E)} comparisons`,
        why: `A partial sort. Cheap in arithmetic, but it decides which expert weights have to be fetched from memory — so this tiny operation sets the bandwidth cost of the entire layer.` })
      if (!isPostSoftmax)
        push({ id: 'gath', kind: 'elementwise', label: 'Gather scores', col: 0, w: 150, cost: `${N(tokens * topk)} reads`,
          why: hasBias
            ? 'Reads back the scores *without* the balancing bias at the chosen indices. The bias decided who runs; it does not decide how much they count.'
            : 'Reads back the probabilities at the chosen indices.' })
      if (normTopk)
        push({ id: 'nrm', kind: 'elementwise', label: 'Normalise', col: 0, w: 150,
          note: routedScaling !== 1 ? `then × ${routedScaling}` : 'sum to 1',
          cost: `${N(tokens * topk)} divides`,
          why: `The ${topk} surviving scores are divided by their sum so they form a blend rather than an arbitrary scale${routedScaling !== 1 ? `, then multiplied by routed_scaling_factor = ${routedScaling} to restore the magnitude the network was trained at` : ''}.` })
      nodes.push(
        { id: 'idx', kind: 'terminal', label: 'indices', col: -0.85, row: row + 1, note: `[${tokens}, ${topk}]` },
        { id: 'w', kind: 'terminal', label: isPostSoftmax ? 'top values' : 'gate weights', col: 0.85, row: row + 1, note: `[${tokens}, ${topk}]` },
      )
      edges.push(
        { from: prev, to: 'w' },
        { from: 'sel', to: 'idx', lane: -1.5, control: true },
        ...(hasBias ? [{ from: 'bias', to: grouped ? 'grp' : 'sel', lane: 1.5, control: true, label: 'ranking only' }] : []),
      )
      return {
        title: `Top-${topk} of ${E}`,
        subtitle: `per token, ${N(tokens)} times${grouped ? ` · restricted to ${topkGroup} of ${nGroup} groups` : ''}`,
        nodes,
        edges,
        note: `Two things come out of here, and both are used: <b>indices</b> say which ${topk} of the ${E} expert weight matrices get fetched from memory at all, and <b>${isPostSoftmax ? 'top values' : 'gate weights'}</b> ${isPostSoftmax ? 'are softmaxed into the blend weights one step later' : 'say how much each of their outputs counts in the blend'}. This is where sparsity is created — everything after it runs on ${((100 * topk) / E).toFixed(1)}% of the layer, and nothing downstream ever sees the other ${E - topk}.`,
      }
    })

  /** The DSA lightning indexer: score every key cheaply, keep the best. */
  const indexerPrim = () =>
    reg('indexer', () => ({
      title: 'Lightning indexer',
      subtitle: `${idxHeads} heads of ${idxDim} dims → keeps the top ${idxTopk?.toLocaleString?.() ?? idxTopk} keys per query`,
      nodes: [
        { id: 'ql', kind: 'terminal', label: 'q latent', col: -1.35, row: 0, note: `[${tokens}, ${N(qLora ?? H)}]` },
        { id: 'x', kind: 'terminal', label: 'x', col: 0.9, row: 0, note: `[${tokens}, ${N(H)}]` },
        { id: 'wqb', kind: 'linear', label: 'wq_b', col: -1.35, row: 1, w: 96, ...lin(qLora ?? H, idxHeads * idxDim,
          `Builds ${idxHeads} indexer query heads of ${idxDim} dims — from the same latent the real attention queries come from, so producing them costs almost nothing extra.`) },
        { id: 'wk', kind: 'linear', label: 'wk', col: 0.35, row: 1, w: 96, ...lin(H, idxDim,
          `One ${idxDim}-dim key per token, shared by all ${idxHeads} indexer heads. That is why the indexer's own cache is small enough to be worth keeping: ${idxDim} numbers per token against the attention cache's thousands.`) },
        { id: 'wp', kind: 'linear', label: 'weights_proj', col: 1.85, row: 1, w: 108, ...lin(H, idxHeads,
          `One scalar per head per query — how much each of the ${idxHeads} heads' opinion counts when their votes are combined.`) },
        { id: 'dot', kind: 'matmul', label: 'MatMul', col: -0.5, row: 2, note: 'q·kᵀ', stack: idxHeads,
          cost: F(2 * tokens * tokens * idxDim * idxHeads), expand: matmulPrim(tokens, idxDim, tokens),
          why: `Every indexer query against every key, ${idxHeads} times over. Still O(S²) — the indexer does not escape the quadratic, it just makes the quadratic part ${((idxDim * idxHeads) / (sdpaDim * nh)).toFixed(2)}× as wide as real attention would be.` },
        { id: 'relu', kind: 'act', label: 'ReLU', col: -0.5, row: 3, cost: `${N(tokens * tokens * idxHeads)} max ops`,
          why: 'ReLU, and not softmax or a smooth gate — chosen for throughput. This runs over the entire context on every layer that owns an indexer, so a single comparison per element is the whole budget.' },
        { id: 'wsum', kind: 'reduce', label: 'Weighted sum', col: -0.5, row: 4, w: 150, note: `over ${idxHeads} heads`,
          cost: `${N(tokens * tokens * idxHeads)} multiply-adds`,
          why: `The ${idxHeads} heads' opinions are combined using the per-head weights, giving one score per (query, key) pair: I = Σⱼ wⱼ · ReLU(qⱼ · k).` },
        { id: 'top', kind: 'reduce', label: `Top-${idxTopk?.toLocaleString?.() ?? idxTopk}`, col: -0.5, row: 5, w: 150, note: 'per query',
          why: `Keeps the ${idxTopk?.toLocaleString?.() ?? idxTopk} highest-scoring keys for each query and discards the rest. At the ${tokens} tokens drawn here that is everything; the saving only begins past ${idxTopk?.toLocaleString?.() ?? idxTopk} tokens of context, which is the point — this exists for the long tail.` },
        { id: 'out', kind: 'terminal', label: 'key set', col: -0.5, row: 6, note: `≤ ${idxTopk?.toLocaleString?.() ?? idxTopk} per query` },
      ],
      edges: [
        { from: 'ql', to: 'wqb' }, { from: 'x', to: 'wk' }, { from: 'x', to: 'wp' },
        { from: 'wqb', to: 'dot' }, { from: 'wk', to: 'dot', lane: 0.35 },
        { from: 'dot', to: 'relu' }, { from: 'relu', to: 'wsum' },
        { from: 'wp', to: 'wsum', lane: 1.85, control: true, label: 'head weights' },
        { from: 'wsum', to: 'top' }, { from: 'top', to: 'out' },
      ],
      note: `<b>${M(idxParams)} parameters</b> deciding what ${M(attnParams)} of attention is allowed to look at. The ReLU is the only non-linear step, and it was picked because it is the cheapest one available — the indexer's whole design constraint is that it must be much cheaper per token than the attention it replaces.`,
    }))

  const attnParams = isMLA
    ? H * qLora +
      qLora + // q_a_layernorm, on the query latent
      kvLora + // kv_a_layernorm, on the compressed half of the KV latent only
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

  /** The router chain, in the order this architecture actually applies it. */
  const scoreNode = (row) =>
    isSigmoidRouter
      ? { id: 'score', kind: 'act', label: 'Sigmoid', col: -1.5, row, w: 100, note: `${E} scores`,
          expand: sigmoidPrim(tokens, E),
          why: 'Squashes each expert\'s logit to (0,1) independently. Independence is the point — it is what lets a load-balancing bias be added to the ranking without distorting the blend weights.' }
      : { id: 'score', kind: 'act', label: 'SoftMax', col: -1.5, row, w: 100,
          note: isPostSoftmax ? `over the ${topk} kept` : `over all ${E}`,
          expand: softmaxPrim(tokens, isPostSoftmax ? topk : E, 'experts'),
          why: isPostSoftmax
            ? `Applied *after* the top-${topk} selection, over only the ${topk} survivors. The discarded experts never enter the normalisation at all, so the gates depend on which experts won but not on how close the rest were.`
            : `Applied over all ${E} experts before selection, so every expert's score depends on every other's. The ${topk} that survive are then renormalised among themselves.` }

  const pickNode = (row) => ({
    id: 'pick', kind: 'mask', label: `Top-${topk}`, col: -1.5, row, w: 100,
    note: grouped ? `${topkGroup}/${nGroup} groups` : 'keep the best',
    expand: pickPrim(),
    why: `Keeps the ${topk} highest-scoring experts and discards the other ${E - topk}. Open it — two things come out, the indices and the gate weights, and the diagram above shows where each one goes.`,
  })

  return {
    // ---------------------------------------------------------------- model
    model: {
      title: 'The model',
      subtitle: `${name} · ${M(total)} parameters, ${M(active)} active per token`,
      nodes: [
        { id: 'tok', kind: 'terminal', label: 'tokens', col: 0, row: 0, note: `[${tokens}]` },
        // Deliberately not `lin(V, H)`: an embedding is a row lookup, so it has a
        // shape but no matrix multiply to open into.
        { id: 'embed', kind: 'embed', label: 'Embedding', col: 0, row: 1,
          shape: `${V.toLocaleString()}→${H.toLocaleString()}`,
          cost: `${M(V * H)} params · gather`, expand: gatherPrim(),
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
        { id: 'attn', kind: 'composite', label: isMLA ? 'Latent Attention' : 'Multi-Head Attention', col: 0, row: 2, w: 200, expand: 'mha', cost: `${M(attnParams)} params`,
          note: `SoftMax · ${nh} heads` },
        { id: 'add1', kind: 'add', label: '+', col: 0, row: 3, w: 44, expand: addPrim(),
          why: 'Attention\'s output is added back to the stream that entered the block — the dashed blue line coming in from the left. The stream is never replaced, only written into.' },
        { id: 'n2', kind: 'norm', label: 'RMSNorm', col: 0, row: 4, expand: rmsnormPrim(),
          why: 'The same normalisation again, this time for the feed-forward half.' },
        {
          id: 'ffn', kind: isMoE ? 'moe' : 'linear',
          label: isMoE ? 'Mixture of Experts' : 'Feed-forward', col: 0, row: 5, w: 200,
          expand: isMoE ? 'moe' : ffnPrim(ffn), cost: `${M(isMoE ? moeParams : denseFfnParams)} params`,
          note: isMoE ? `${gluName} · top-${topk} of ${E}` : `${gluName} · ${H.toLocaleString()}→${ffn.toLocaleString()}→${H.toLocaleString()}`,
        },
        { id: 'add2', kind: 'add', label: '+', col: 0, row: 6, w: 44, expand: addPrim(),
          why: 'And again after the feed-forward. Two additions per block, and every layer in the stack writes into the same stream.' },
        { id: 'out', kind: 'terminal', label: 'x′', col: 0, row: 7, note: `[${tokens}, ${H.toLocaleString()}]` },
      ],
      edges: [
        { from: 'in', to: 'n1' }, { from: 'n1', to: 'attn' }, { from: 'attn', to: 'add1' },
        // Residuals leave through the side of their source and arrive on the side
        // of the add, so it is unambiguous where each one is picked up.
        { from: 'in', to: 'add1', lane: -1.9, out: 'top', residual: true, label: 'residual' },
        { from: 'add1', to: 'n2' }, { from: 'n2', to: 'ffn' }, { from: 'ffn', to: 'add2' },
        { from: 'add1', to: 'add2', lane: 1.9, out: 'top', residual: true, label: 'residual' },
        { from: 'add2', to: 'out' },
      ],
      note: `The two dashed blue paths are the residual connections, and they are drawn leaving the exact point they are taken from: the left one is picked up at <i>x</i>, before the first norm, and carried past attention; the right one is picked up at the first <b>+</b>, after attention has been added, and carried past the feed-forward. Both halves write <i>into</i> the stream rather than replacing it, and that is what lets gradients reach the bottom of a ${L}-layer stack. Note also that the stream itself never passes through a normalisation — only the copies handed to the two boxes do.`,
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
              { id: 'attn', kind: 'composite', label: isMLA ? 'Latent Attention' : 'Multi-Head Attention', col: 0, row: 2, w: 200, expand: 'mha', cost: `${M(attnParams)} params`, note: `SoftMax · ${nh} heads` },
              { id: 'add1', kind: 'add', label: '+', col: 0, row: 3, w: 44, expand: addPrim(),
                why: 'The same residual add as in the sparse blocks.' },
              { id: 'n2', kind: 'norm', label: 'RMSNorm', col: 0, row: 4, expand: rmsnormPrim(),
                why: 'The second normalisation, for the feed-forward half.' },
              { id: 'ffn', kind: 'linear', label: 'Feed-forward', col: 0, row: 5, w: 200, expand: ffnPrim(ffn), cost: `${M(denseFfnParams)} params`, note: `${gluName} · ${H.toLocaleString()}→${ffn.toLocaleString()}→${H.toLocaleString()}` },
              { id: 'add2', kind: 'add', label: '+', col: 0, row: 6, w: 44, expand: addPrim(),
                why: 'And again after the feed-forward.' },
              { id: 'out', kind: 'terminal', label: 'x′', col: 0, row: 7 },
            ],
            edges: [
              { from: 'in', to: 'n1' }, { from: 'n1', to: 'attn' }, { from: 'attn', to: 'add1' },
              { from: 'in', to: 'add1', lane: -1.9, out: 'top', residual: true, label: 'residual' },
              { from: 'add1', to: 'n2' }, { from: 'n2', to: 'ffn' }, { from: 'ffn', to: 'add2' },
              { from: 'add1', to: 'add2', lane: 1.9, out: 'top', residual: true, label: 'residual' },
              { from: 'add2', to: 'out' },
            ],
            note: `Identical to the sparse blocks except here, where an ordinary ${gluName} feed-forward sits instead of a router and ${E} experts. Routing this early tends to hurt, so the first ${denseLayers} layers stay dense.`,
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
            { id: 'qa', kind: 'linear', label: 'q_a_proj', col: -1.2, row: 1, w: 100, ...lin(H, qLora,
              `Squeezes the ${N(H)}-wide stream down to a ${N(qLora)}-dim query latent. Queries are recomputed every step and never cached, so this bottleneck buys parameters rather than memory.`) },
            { id: 'kva', kind: 'linear', label: 'kv_a_proj', col: 1.2, row: 1, w: 100, note: `${kvLora} + ${qkRope}`,
              ...lin(H, kvLora + qkRope,
                `Produces two things at once: a ${N(kvLora)}-dim latent that keys and values are both re-expanded from, and a separate ${qkRope}-dim slice that carries position. <b>This ${N(kvLora + qkRope)}-wide output is the entire KV cache</b> — everything to its left in this diagram is recomputed from it.`) },
            // Both latents are normalised before anything else touches them —
            // and the kv one covers only the compressed half, never the rope slice.
            { id: 'qn', kind: 'norm', label: 'q_a_layernorm', col: -1.2, row: 2, w: 120,
              note: `over ${qLora.toLocaleString()}`, cost: `${M(qLora)} params`,
              expand: rmsnormPrim(qLora),
              why: `RMSNorm on the query latent. A ${N(H)}-wide vector has just been squeezed into ${N(qLora)} numbers, and their scale is not guaranteed; normalising here is what keeps the expansion that follows numerically stable.` },
            { id: 'kvn', kind: 'norm', label: 'kv_a_layernorm', col: 1.2, row: 2, w: 120,
              note: `over ${kvLora.toLocaleString()}`, cost: `${M(kvLora)} params`,
              expand: rmsnormPrim(kvLora),
              why: `RMSNorm on the compressed part of the KV latent — the ${N(kvLora)} dimensions only, not the ${qkRope} rope dimensions beside them. Normalising a rotated vector would destroy the angle that encodes position, so the positional slice is routed around this box as well as around kv_b_proj.` },
            { id: 'qb', kind: 'linear', label: 'q_b_proj', col: -1.2, row: 3, w: 100, ...lin(qLora, nh * qkHead,
              `Expands the latent back out to ${nh} query heads of ${qkHead} dims each — ${qkNope} carrying content and ${qkRope} reserved for position.`) },
            { id: 'kvb', kind: 'linear', label: 'kv_b_proj', col: 1.2, row: 3, w: 100, ...lin(kvLora, nh * (qkNope + vHead),
              `Re-expands the cached ${N(kvLora)}-dim latent into ${nh} content-key heads of ${qkNope} and ${nh} value heads of ${vHead}. Run fresh on every step: MLA trades this arithmetic for a ${((nh * (qkNope + vHead)) / (kvLora + qkRope)).toFixed(0)}× smaller cache.`) },
            ...(hasIndexer
              ? [{ id: 'idx', kind: 'router', label: 'Indexer', col: -2.15, row: 1, w: 100, note: `top-${idxTopk?.toLocaleString?.() ?? idxTopk}`, cost: `${M(idxParams)} params`,
                  expand: indexerPrim(),
                  why: `Not part of classic attention. It scores every key cheaply — ${idxHeads} heads of ${idxDim} dims and a ReLU — and hands the winners to attention, so attention never sees the whole context. Open it.` }]
              : []),
            { id: 'rope', kind: 'rope', label: 'RoPE', col: 0, row: 4, w: 150, note: `${qkRope} of ${qkHead} dims`,
              expand: ropePrim(qkRope, 'rope slice'),
              why: `Position enters here, and only on part of the vector: ${qkRope} of each query head's ${qkHead} dims, and one ${qkRope}-dim key slice shared by all ${nh} heads. The other ${qkNope} dims carry no position at all. That split is not an optimisation but a requirement — the cached latent has to be position-free so it can be re-expanded later, so the positional part is kept outside it and cached separately.` },
            { id: 'sdpa', kind: 'composite', label: 'Scaled Dot-Product Attention', col: 0, row: 5, w: 240, stack: nh, expand: 'sdpa', note: 'no parameters of its own' },
            { id: 'lo', kind: 'linear', label: 'o_proj', col: 0, row: 6, ...lin(nh * vHead, H,
              `The output projection, and the only place the ${nh} heads get to interact. Each has produced a ${vHead}-dim answer in its own subspace; this mixes all ${N(nh * vHead)} of those numbers back down to the ${N(H)}-wide residual stream.`) },
            { id: 'out', kind: 'terminal', label: 'out', col: 0, row: 7, note: `[${tokens}, ${H.toLocaleString()}]` },
          ],
          edges: [
            { from: 'x', to: 'qa' }, { from: 'x', to: 'kva' },
            { from: 'qa', to: 'qn' }, { from: 'qn', to: 'qb' },
            { from: 'kva', to: 'kvn' }, { from: 'kvn', to: 'kvb' },
            ...(hasIndexer
              ? [
                  { from: 'x', to: 'idx' },
                  { from: 'qa', to: 'idx', lane: -2.15 },
                  { from: 'idx', to: 'sdpa', lane: -2.15, control: true, label: `top-${idxTopk?.toLocaleString?.() ?? idxTopk}` },
                ]
              : []),
            { from: 'qb', to: 'rope' },
            // The rope slice skips kv_b entirely — it is cached as-is and shared
            // by every head, which is the whole reason MLA splits the head.
            { from: 'kva', to: 'rope', lane: 2.05, label: `${qkRope}-dim rope` },
            { from: 'kvb', to: 'sdpa', lane: 1.2 },
            { from: 'rope', to: 'sdpa' },
            { from: 'sdpa', to: 'lo' }, { from: 'lo', to: 'out' },
          ],
          note: `Every projection goes through a bottleneck. Keys and values are squeezed to <b>${kvLora.toLocaleString()}</b> dimensions before being expanded back to ${(nh * (qkNope + vHead)).toLocaleString()} — and it is the small latent that gets cached, not the expansion. Follow the line that leaves <code>kv_a_proj</code> on the right: those ${qkRope} dimensions bypass both the layernorm and <code>kv_b_proj</code>, and are cached exactly as they are. They have to be — a normalisation would rescale them and a re-expansion would have to be position-aware, and neither survives contact with a rotated vector. Splitting the head into a compressible half and a positional half is the whole idea.${
            hasIndexer
              ? ` The indexer on the far left is separate again: it decides which ${idxTopk?.toLocaleString?.() ?? idxTopk} keys attention is even allowed to see.`
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
              id: 'rope', kind: 'rope', label: 'RoPE', col: -0.575, row: 2, w: 150, note: 'Q and K only',
              expand: ropePrim(hd, 'q, k'),
              why: `Position enters the model here and nowhere else. Queries and keys are rotated by an angle proportional to their position, so their dot product depends on how far apart they are; values are left alone, because a value is *what* a token contributes, not *where* it sits.`,
            },
            {
              id: 'sdpa', kind: 'composite', label: 'Scaled Dot-Product Attention', col: 0, row: 3,
              w: 240, stack: nh, expand: 'sdpa', note: 'no parameters of its own',
              why: `Genuinely ${nh} independent computations — unlike the projections below, which are one matrix each. Every head attends over its own ${hd}-dimensional slice, for its own reasons.`,
            },
            {
              id: 'concat', kind: 'concat', label: 'Concat', col: 0, row: 4, expand: concatPrim(),
              note: `${nh} × ${hd} = ${N(nh * hd)}`,
              why: 'Lays the heads side by side. Free — no data moves; open it to see why.',
            },
            {
              id: 'lo', kind: 'linear', label: 'W_O', col: 0, row: 5, ...lin(nh * hd, H,
                `The output projection, and the only place the heads get to interact. Each head has produced a ${hd}-dim answer in its own subspace; this mixes all ${N(nh * hd)} of those numbers back down to the ${N(H)}-wide residual stream. Without it the heads would write into disjoint slices and never combine.`),
            },
            { id: 'out', kind: 'terminal', label: 'out', col: 0, row: 6, note: `[${tokens}, ${N(H)}]` },
          ],
          edges: [
            { from: 'x', to: 'lq' }, { from: 'x', to: 'lk' }, { from: 'x', to: 'lv' },
            { from: 'lq', to: 'rope' }, { from: 'lk', to: 'rope' },
            { from: 'rope', to: 'sdpa' },
            // V skips the rotation entirely, and the diagram should show that.
            { from: 'lv', to: 'sdpa', lane: 1.15, label: 'V — no rotation' },
            { from: 'sdpa', to: 'concat' }, { from: 'concat', to: 'lo' }, { from: 'lo', to: 'out' },
          ],
          note: `Three projections of the same input, a rotation applied to two of the three, then ${nh} independent attentions, then one projection back. The line on the right is <b>V</b> bypassing RoPE: position belongs to the matching, not to the content.${
            kv !== nh
              ? ` Note the asymmetry too: ${nh} query heads against only ${kv} key/value heads. The queries are cheap to make and thrown away; the keys and values are cached for every token in the context, so there are fewer of them on purpose.`
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
          expand: softmaxPrim(tokens, tokens, 'scores'),
          why: 'The non-linearity of the attention half, and the reason attention is not just a product of matrices. It turns each row of scores into weights that sum to one — and because those weights depend on the input, attention computes a *different* linear map for every sequence it sees.' },
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
        { from: 'V', to: 'av', lane: 1.9, label: 'V bypasses' }, { from: 'av', to: 'out' },
      ],
      note: `The only box here with no parameters at all — attention is a way of <i>using</i> numbers, not storing them. Both MatMuls cost about ${F(2 * tokens * tokens * sdpaDim)} at ${tokens} tokens and grow with the square of context. The single green box is doing all the non-linear work: strip the SoftMax out and Q·Kᵀ·V collapses into one fixed linear map, the same one for every input. Click any box to see the operation itself.`,
    },

    // ------------------------------------------------------------------ MoE
    ...(isMoE
      ? {
          moe: {
            title: 'Mixture of Experts',
            subtitle: `${E} experts, top-${topk} per token${nShared ? ` + ${nShared} always-on` : ''} · ${M(E * expertParams)} stored, ${M((topk + nShared) * expertParams)} used`,
            nodes: [
              { id: 'in', kind: 'terminal', label: 'x', col: 0, row: 0, note: `[${tokens}, ${H.toLocaleString()}]` },
              { id: 'router', kind: 'router', label: 'Router', col: -1.5, row: 1, w: 100, note: `scores ${E}`, meta: 'below', ...lin(H, E,
                `A tiny matrix — ${M(H * E)} parameters — that scores all ${E} experts for each token. Trivial arithmetic, but it decides which expert weights must be fetched, so it sets the memory bill for the whole layer.`) },
              // gpt-oss selects on raw logits and softmaxes the survivors; every
              // other family here scores first and selects second.
              ...(isPostSoftmax ? [pickNode(2), scoreNode(3)] : [scoreNode(2), pickNode(3)]),
              {
                id: 'experts', kind: 'moe', label: 'Expert FFN', col: 0.4, row: 4, w: 140,
                stack: E, active: topk, note: gluName, meta: 'below',
                shape: `${H.toLocaleString()}→${moeFfn.toLocaleString()}→${H.toLocaleString()}`,
                cost: `${M(expertParams)} params each`,
                expand: ffnPrim(moeFfn),
                why: `An ordinary ${gluName} feed-forward network. There are ${E} of them, identical in shape and different in weights, and only the ${topk} named by the router run for any given token — which is why the pile behind the front card is drawn grey.`,
              },
              ...(nShared
                ? [{ id: 'shared', kind: 'moe', label: 'Shared expert', col: 2.15, row: 4, w: 104, note: 'always runs', cost: `${M(nShared * expertParams)} params`,
                    expand: ffnPrim(moeFfn),
                    why: 'Runs for every token regardless of routing, and is not gated. It absorbs whatever is common to all inputs, so the routed experts are free to specialise instead of each relearning the same basics.' }]
                : []),
              { id: 'mix', kind: 'concat', label: 'Weighted sum', col: 0.4, row: 5, w: 150, note: `${topk + nShared} outputs`,
                why: `Blends the chosen experts' outputs using the gate weights coming in from the left${nShared ? `, then adds the shared expert's output, which is not gated at all` : ''}. A blend rather than a hard pick is what makes the router differentiable — the gradient reaches it through these weights.` },
              { id: 'out', kind: 'terminal', label: 'out', col: 0.4, row: 6, note: `[${tokens}, ${H.toLocaleString()}]` },
            ],
            edges: [
              { from: 'in', to: 'router' },
              ...(isPostSoftmax
                ? [{ from: 'router', to: 'pick' }, { from: 'pick', to: 'score' }]
                : [{ from: 'router', to: 'score' }, { from: 'score', to: 'pick' }]),
              { from: 'in', to: 'experts' },
              // The two things the router produces, and where each one goes.
              // Out of the right, not the top: gpt-oss puts the SoftMax directly
              // above the Top-k, and a lane up its own column would run through it.
              { from: 'pick', to: 'experts', lane: -0.7, control: true, label: `which ${topk}` },
              { from: isPostSoftmax ? 'score' : 'pick', to: 'mix', lane: -2.55, control: true, label: 'gate weights' },
              ...(nShared ? [{ from: 'in', to: 'shared' }, { from: 'shared', to: 'mix', lane: 2.15 }] : []),
              { from: 'experts', to: 'mix' }, { from: 'mix', to: 'out' },
            ],
            note: `Follow the two dashed gold lines, because they are the whole mechanism: <b>Top-${topk}</b> emits indices, which say which ${topk} of the ${E} expert weight matrices get fetched from memory at all, and gate weights, which say how much each of their outputs counts in the blend. Nothing else in the layer reads the router. ${M(H * E)} parameters decide which <b>${M(topk * expertParams)}</b> of the layer's <b>${M(E * expertParams)}</b> does any work — capacity decoupled from cost, with only ${((100 * topk) / E).toFixed(1)}% of the layer participating in any given token.`,
          },
        }
      : {}),

    // Every primitive the specs above asked for, keyed by shape.
    ...prims,
  }
}
