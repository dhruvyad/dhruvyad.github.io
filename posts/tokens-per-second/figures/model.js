/**
 * The roofline decode model for GLM-5.2, shared by every figure in this article
 * so they can't disagree with each other or with the prose.
 *
 * Every architectural constant below is read from the published config.json of
 * zai-org/GLM-5.2. The parameter accounting reproduces the model's real
 * safetensors byte count (1,506,667,387,408 bytes / 2 = 753.33B) to within
 * 0.006%, so the derived traffic numbers are grounded rather than guessed.
 */

// ---------------------------------------------------------------- architecture
export const GLM = {
  name: 'GLM-5.2',
  hidden: 6144,
  layers: 78,
  heads: 64,
  qkHeadDim: 256, //  = 192 nope + 64 rope
  qkNopeHeadDim: 192,
  vHeadDim: 256,
  ropeHeadDim: 64,
  qLoraRank: 2048,
  kvLoraRank: 512,
  ffn: 12288, // dense layers
  moeFfn: 2048, // per expert
  experts: 256,
  topk: 8,
  sharedExperts: 1,
  denseLayers: 3,
  vocab: 154880,
  indexHeads: 32,
  indexHeadDim: 128,
  indexTopk: 2048,
  // Only 22 of the 79 checkpoint layers carry indexer weights — this is
  // IndexShare / cross-layer index reuse, and it is visible in the safetensors
  // index, not just claimed in the README.
  indexerLayers: 22,
  totalParams: 753.33e9,
}

const G = GLM
export const sparseLayers = G.layers - G.denseLayers // 75

/** Parameters in one MLA attention block. */
export const attnParams =
  G.hidden * G.qLoraRank + // q_a_proj
  G.qLoraRank * G.heads * G.qkHeadDim + // q_b_proj
  G.hidden * (G.kvLoraRank + G.ropeHeadDim) + // kv_a_proj_with_mqa
  G.kvLoraRank * G.heads * (G.qkNopeHeadDim + G.vHeadDim) + // kv_b_proj
  G.heads * G.vHeadDim * G.hidden // o_proj

/** Parameters in one DSA lightning indexer. */
export const indexerParams =
  G.qLoraRank * G.indexHeads * G.indexHeadDim + // wq_b, reusing the MLA q latent
  G.indexHeadDim * G.hidden + // wk
  G.indexHeads * G.hidden // weights_proj

export const denseMlpParams = 3 * G.hidden * G.ffn
export const expertParams = 3 * G.hidden * G.moeFfn
export const routerParams = G.hidden * G.experts

/**
 * Weights every token must read no matter the batch size: attention, indexers,
 * the three dense MLPs, the always-on shared expert, routers, and the
 * unembedding. Everything else depends on which experts the batch happens to hit.
 */
export const residentParams =
  G.layers * attnParams +
  G.indexerLayers * indexerParams +
  G.denseLayers * denseMlpParams +
  G.vocab * G.hidden +
  sparseLayers * (G.sharedExperts * expertParams + routerParams)

export const activeParams = residentParams + sparseLayers * G.topk * expertParams

/**
 * Distinct experts a batch of B tokens activates in one MoE layer.
 *
 * Each token routes to `topk` of `experts`, so a given expert is missed by a
 * given token with probability (1 - topk/experts). This assumes independent,
 * uniform routing; real routers are skewed toward popular experts, which makes
 * this an optimistic estimate of how little traffic you can get away with.
 */
export function expertsTouched(B, experts = G.experts, topk = G.topk) {
  return experts * (1 - Math.pow(1 - topk / experts, B))
}

/** KV cache bytes per token: MLA stores one compressed latent per layer. */
export function kvBytesPerToken(bytesPerElem = 2) {
  return (G.kvLoraRank + G.ropeHeadDim) * bytesPerElem * G.layers
}

/** What the KV cache would cost without MLA, as plain multi-head attention. */
export function mhaBytesPerToken(bytesPerElem = 2) {
  return 2 * G.heads * G.vHeadDim * bytesPerElem * G.layers
}

/**
 * Per-layer structure, straight from config.json's `mlp_layer_types` and
 * `indexer_types` arrays: the first 3 layers are dense, the remaining 75 are MoE,
 * and only 21 of the 78 own an indexer (layers 0-2, then every fourth from 6).
 * A 22nd indexer lives on the MTP layer.
 */
export const LAYER_MAP = Array.from({ length: G.layers }, (_, i) => ({
  index: i,
  dense: i < G.denseLayers,
  fullIndexer: i < 3 || (i >= 6 && (i - 6) % 4 === 0),
}))

const mtpParams = attnParams + G.experts * expertParams + expertParams + routerParams + G.hidden ** 2

/**
 * Cost of each structural piece of the model, so a diagram can attach real
 * numbers to the thing you click on.
 *
 * `active` is what a single token actually touches — the whole point of a sparse
 * MoE is that this is a small fraction of `params`.
 */
export function partBreakdown({ wBytes = 1, B = 1, S = 8192 } = {}) {
  const selected = Math.min(S, G.indexTopk)
  const touched = expertsTouched(B)
  const attnScoreFlops = 2 * G.heads * (G.qkHeadDim + G.vHeadDim) * selected
  const indexScoreFlops = 2 * G.indexHeads * G.indexHeadDim * S

  const parts = [
    {
      id: 'embed',
      label: 'Token embedding',
      count: 1,
      detail: `${G.vocab.toLocaleString()} × ${G.hidden}`,
      params: G.vocab * G.hidden,
      // A token touches exactly one row of the table, not the table.
      active: G.hidden,
      flops: 0,
      bytes: B * G.hidden * wBytes,
      note: 'A lookup, not a matrix multiply. It holds a lot of parameters but costs almost no arithmetic and almost no bandwidth — only the rows for the tokens in flight get read.',
    },
    {
      id: 'indexer',
      label: 'DSA indexer',
      count: G.indexerLayers,
      detail: '22 of 79 layers',
      params: G.indexerLayers * indexerParams,
      active: G.indexerLayers * indexerParams,
      flops: G.indexerLayers * (2 * indexerParams + indexScoreFlops),
      bytes: G.indexerLayers * indexerParams * wBytes,
      note: 'Picks the top-2048 keys each query will attend to. Tiny in parameters but its scoring pass is O(S) per query per layer, which is why sharing one indexer across four layers matters at long context.',
    },
    {
      id: 'mla',
      label: 'MLA attention',
      count: G.layers,
      detail: `${G.heads} heads, 512-dim latent`,
      params: G.layers * attnParams,
      active: G.layers * attnParams,
      flops: G.layers * (2 * attnParams + attnScoreFlops),
      bytes: G.layers * attnParams * wBytes,
      note: `Every token uses all of it. Also the piece that owns the KV cache: ${(kvBytesPerToken() / 1024).toFixed(2)} KiB per token, compressed to one latent per layer instead of keys and values per head.`,
    },
    {
      id: 'dense',
      label: 'Dense MLP',
      count: G.denseLayers,
      detail: 'layers 0–2 only',
      params: G.denseLayers * denseMlpParams,
      active: G.denseLayers * denseMlpParams,
      flops: G.denseLayers * 2 * denseMlpParams,
      bytes: G.denseLayers * denseMlpParams * wBytes,
      note: 'The first three layers are ordinary feed-forward blocks. Routing very early in the network tends to hurt, so those layers stay dense.',
    },
    {
      id: 'router',
      label: 'Router',
      count: sparseLayers,
      detail: `${G.hidden} → ${G.experts} scores`,
      params: sparseLayers * routerParams,
      active: sparseLayers * routerParams,
      flops: sparseLayers * 2 * routerParams,
      bytes: sparseLayers * routerParams * wBytes,
      note: 'Scores all 256 experts and keeps the top 8. Negligible in cost, but it decides which 9.7B of expert weights get read — so it sets the bandwidth bill for the whole layer.',
    },
    {
      id: 'shared',
      label: 'Shared expert',
      count: sparseLayers,
      detail: '1 per MoE layer, always on',
      params: sparseLayers * expertParams,
      active: sparseLayers * expertParams,
      flops: sparseLayers * 2 * expertParams,
      bytes: sparseLayers * expertParams * wBytes,
      note: 'Runs for every token regardless of routing, so the routed experts only have to learn what is special about a token rather than what is common to all of them.',
    },
    {
      id: 'routed',
      label: 'Routed experts',
      count: sparseLayers * G.experts,
      detail: `top-${G.topk} of ${G.experts}, × ${sparseLayers} layers`,
      params: sparseLayers * G.experts * expertParams,
      active: sparseLayers * G.topk * expertParams,
      flops: sparseLayers * 2 * G.topk * expertParams,
      bytes: sparseLayers * touched * expertParams * wBytes,
      note: `Where the model lives: 96% of all parameters. One token activates 8 of 256 per layer, but a batch of ${B} collectively touches ${touched.toFixed(0)} — and every one of those has to be read from HBM.`,
    },
    {
      id: 'mtp',
      label: 'MTP head',
      count: 1,
      detail: 'one extra layer',
      params: mtpParams,
      active: 0,
      flops: 0,
      bytes: 0,
      note: 'A spare transformer layer that predicts the token after next, used to draft candidates for speculative decoding. Idle unless speculation is switched on, which is why it costs nothing here.',
    },
    {
      id: 'lmhead',
      label: 'LM head',
      count: 1,
      detail: `${G.hidden} → ${G.vocab.toLocaleString()}`,
      params: G.vocab * G.hidden,
      active: G.vocab * G.hidden,
      flops: 2 * G.vocab * G.hidden,
      bytes: G.vocab * G.hidden * wBytes,
      note: 'Projects the final hidden state onto the vocabulary. Unlike the embedding this is a real matrix multiply, and its weights are streamed in full on every step.',
    },
  ]

  const totalParams = parts.reduce((a, p) => a + p.params, 0)
  const totalActive = parts.reduce((a, p) => a + p.active, 0)
  const totalFlops = parts.reduce((a, p) => a + p.flops, 0)
  const totalBytes = parts.reduce((a, p) => a + p.bytes, 0)

  return {
    parts: parts.map((p) => ({
      ...p,
      paramShare: p.params / totalParams,
      activeShare: p.active / p.params,
      flopShare: totalFlops ? p.flops / totalFlops : 0,
      byteShare: totalBytes ? p.bytes / totalBytes : 0,
    })),
    totalParams,
    totalActive,
    totalFlops,
    totalBytes,
    touched,
  }
}

// ------------------------------------------------------------------- hardware
export const GPUS = {
  'H100 SXM': { hbmGB: 80, bw: 3.35e12, bf16: 989e12 },
  'H200 SXM': { hbmGB: 141, bw: 4.8e12, bf16: 989e12 },
  B200: { hbmGB: 192, bw: 8.0e12, bf16: 2250e12 },
}

/** FLOP per byte at which a GPU stops being bandwidth-limited. */
export const machineBalance = (gpu) => GPUS[gpu].bf16 / GPUS[gpu].bw

// ---------------------------------------------------------------- decode step
/**
 * One decode step for the whole tensor-parallel group.
 *
 * @param B            batch size (concurrent sequences)
 * @param S            context length in tokens
 * @param wBytes       bytes per weight (2 = bf16, 1 = fp8, 0.5 = fp4)
 * @param gpu          key into GPUS
 * @param nGpu         GPUs in the group
 * @param mbu          achieved fraction of peak bandwidth/compute (1 = roofline)
 * @param tokensPerSeq token positions pushed through per sequence. 1 for ordinary
 *                     decode; k+1 when verifying a speculative draft.
 *
 * The distinction between B and tokensPerSeq matters and is easy to get wrong.
 * Both multiply the number of token positions, so both pull in more experts and
 * more arithmetic. But KV traffic scales only with B: the k+1 positions of one
 * sequence share a single KV cache, read once and reused across all of them.
 * That asymmetry is exactly what makes speculation cheaper than batching.
 */
export function decodeStep({
  B,
  S,
  wBytes = 1,
  gpu = 'H200 SXM',
  nGpu = 8,
  mbu = 1,
  kvBytes = 2,
  tokensPerSeq = 1,
}) {
  const g = GPUS[gpu]
  const aggBw = g.bw * nGpu * mbu
  const aggFlops = g.bf16 * nGpu * mbu

  const positions = B * tokensPerSeq

  // Every token position routes independently, so the union of experts the step
  // touches grows with the total number of positions, not the sequence count.
  const touched = expertsTouched(positions)
  const routedParams = sparseLayers * touched * expertParams
  const weightBytes = (residentParams + routedParams) * wBytes

  // DSA: attention reads only the top-k selected tokens, not the whole context.
  // Read once per sequence — all of that sequence's query positions reuse it.
  const selected = Math.min(S, G.indexTopk)
  const kvRead = B * selected * (G.kvLoraRank + G.ropeHeadDim) * kvBytes * G.layers

  // The indexer still has to score every token — but only on the layers that
  // own an indexer. That is precisely what IndexShare saves.
  const indexerRead = B * S * G.indexHeadDim * kvBytes * G.indexerLayers

  const bytes = weightBytes + kvRead + indexerRead

  let flops = 2 * activeParams * positions
  flops += 2 * positions * G.layers * G.heads * (G.qkHeadDim + G.vHeadDim) * selected
  flops += 2 * positions * G.indexerLayers * G.indexHeads * G.indexHeadDim * S

  const tMem = bytes / aggBw
  const tCmp = flops / aggFlops
  const t = Math.max(tMem, tCmp)

  return {
    bytes,
    weightBytes,
    kvRead,
    indexerRead,
    flops,
    touched,
    positions,
    tMem,
    tCmp,
    t,
    bound: tMem >= tCmp ? 'memory' : 'compute',
    perUser: 1 / t,
    aggregate: B / t,
    intensity: flops / bytes,
  }
}

// -------------------------------------------------------- speculative decoding
/**
 * Expected tokens emitted per verification step when drafting k tokens that are
 * each accepted with probability alpha (Leviathan et al. 2023). The +1 is the
 * bonus token the verifier produces even when every draft is rejected.
 *
 *   E[tokens] = (1 - alpha^(k+1)) / (1 - alpha)
 */
export function acceptedLength(alpha, k) {
  if (alpha >= 1) return k + 1
  return (1 - Math.pow(alpha, k + 1)) / (1 - alpha)
}

/**
 * Net speedup from speculation, in wall-clock terms.
 *
 * The verify step's cost is not assumed — it comes from the roofline model, by
 * asking what a step with k+1 positions per sequence actually costs relative to
 * one with a single position. For a dense model that ratio is ~1 and speculation
 * is nearly free. For a sparse MoE it is greater than 1 at small batch, because
 * the extra positions drag in extra experts, and it falls back toward 1 at large
 * batch where the experts were all being read anyway.
 */
export function specSpeedup({ alpha, k, B = 1, S = 8192, draftCost = 0.08, ...rest }) {
  const one = decodeStep({ B, S, tokensPerSeq: 1, ...rest })
  const many = decodeStep({ B, S, tokensPerSeq: k + 1, ...rest })
  const verifyRatio = many.t / one.t
  const tokens = acceptedLength(alpha, k)
  const cost = verifyRatio + k * draftCost
  return { tokens, verifyRatio, cost, speedup: tokens / cost }
}

// ------------------------------------------------------------------ attention
/**
 * Bytes in the full S x S attention score matrix for one head, one layer — the
 * thing FlashAttention refuses to write to HBM.
 */
export function scoreMatrixBytes(S, bytesPerElem = 2) {
  return S * S * bytesPerElem
}

/** Bytes FlashAttention actually moves for one head: Q, K, V in and O out. */
export function flashTraffic(S, bytesPerElem = 2) {
  return (2 * S * G.qkHeadDim + 2 * S * G.vHeadDim) * bytesPerElem
}

/** On-chip memory, for scale. H100/H200: 132 SMs, 228 KiB shared each, 50 MB L2. */
export const ON_CHIP = {
  smemPerSM: 228 * 1024,
  sms: 132,
  l2: 50e6,
}

export const fmtBytes = (b) => {
  if (b >= 1e12) return `${(b / 1e12).toFixed(2)} TB`
  if (b >= 1e9) return `${(b / 1e9).toFixed(1)} GB`
  if (b >= 1e6) return `${(b / 1e6).toFixed(1)} MB`
  if (b >= 1e3) return `${(b / 1e3).toFixed(1)} kB`
  return `${b.toFixed(0)} B`
}

export const fmtNum = (n) => {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`
  return n.toFixed(0)
}
