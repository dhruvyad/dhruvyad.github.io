/**
 * Shared glossary. Every mention of any of these terms, in any article, gets a
 * tooltip carrying this definition and these links — see src/lib/glossary.js.
 *
 * Fields:
 *   id      stable key
 *   terms   every spelling to match. Case-sensitive unless `ci: true`, so that
 *           "SM" the streaming multiprocessor does not match "sm" in prose.
 *   title   heading shown in the tooltip
 *   expand  what an acronym stands for
 *   short   one-line definition
 *   body    optional extra HTML
 *   visual  optional inline SVG (see visuals.js)
 *   links   optional [{ label, url }] — reused across every occurrence
 */
import { bars, grid, hierarchy, roofline, draftStrip } from './visuals.js'

const NV = 'https://www.nvidia.com/en-us/data-center/'

/** Shared by the acronym and spelled-out forms of "mixture of experts". */
const MOE_DEF = {
  title: 'Mixture of Experts (MoE)',
  short:
    'A layer holding many parallel feed-forward "experts", of which a router selects a few per token.',
  visual: grid(8, 256),
  body: 'It decouples capacity from cost: GLM-5.2 stores 753B parameters but runs 40.3B per token. The catch is that different tokens choose different experts, so batching drags the whole model back into memory.',
}

export const TERMS = [
  // ---------------------------------------------------------------- the model
  {
    id: 'glm52',
    terms: ['GLM-5.2', 'GLM 5.2'],
    title: 'GLM-5.2',
    short:
      "Zhipu AI's flagship open-weights model, released 13 June 2026 under MIT: 753B total parameters, 40.3B active per token, 1M context.",
    body: 'A sparse mixture-of-experts model using MLA attention and DeepSeek Sparse Attention. Every architectural number in this article comes from its published config.',
    links: [
      { label: 'Weights on Hugging Face', url: 'https://huggingface.co/zai-org/GLM-5.2' },
      { label: 'GLM-5 technical report', url: 'https://arxiv.org/abs/2602.15763' },
      { label: 'GitHub', url: 'https://github.com/zai-org/GLM-5' },
    ],
  },
  {
    id: 'safetensors',
    terms: ['safetensors'],
    title: 'safetensors',
    short: 'A weight file format that stores tensors as a flat buffer with a JSON header.',
    body: 'Because the header is plain JSON and the payload is contiguous, you can read a model’s exact parameter count from file sizes alone — which is how the 753.33B figure here was checked.',
    links: [{ label: 'Format spec', url: 'https://github.com/huggingface/safetensors' }],
  },

  // ------------------------------------------------------------------ memory
  {
    id: 'hbm',
    terms: ['HBM', 'HBM3', 'HBM3e'],
    title: 'HBM',
    expand: 'High Bandwidth Memory',
    short:
      'Stacked DRAM sitting beside the GPU die. It holds the weights and the KV cache, and its bandwidth is what sets decode speed.',
    visual: bars(
      [
        { label: 'H100 · 80 GB', value: 3.35, display: '3.35' },
        { label: 'H200 · 141 GB', value: 4.8, display: '4.8', hi: true },
        { label: 'B200 · 192 GB', value: 8.0, display: '8.0' },
      ],
      { unit: ' TB/s' },
    ),
    body: 'Fast compared to ordinary system DRAM, but still roughly an order of magnitude slower than the SRAM on the die — which is the whole reason FlashAttention exists.',
  },
  {
    id: 'dram',
    terms: ['DRAM'],
    title: 'DRAM',
    expand: 'Dynamic Random-Access Memory',
    short:
      'Memory that stores each bit as charge in a capacitor, so it is dense and cheap but comparatively slow and must be refreshed.',
    body: 'A GPU’s HBM is DRAM. Capacity is measured in tens or hundreds of gigabytes; the trade for that capacity is latency and bandwidth.',
  },
  {
    id: 'sram',
    terms: ['SRAM'],
    title: 'SRAM',
    expand: 'Static Random-Access Memory',
    short:
      'On-die memory built from latches. Far faster than DRAM and needs no refresh, but each bit costs several transistors, so there is very little of it.',
    visual: hierarchy([
      { label: 'registers — fastest, smallest', hi: true },
      { label: 'shared memory · 228 KiB per SM', hi: true },
      { label: 'L2 cache · 50 MB' },
      { label: 'HBM · 141 GB, slowest' },
    ]),
    body: 'Registers, shared memory and the L2 cache are all SRAM. Keeping a computation inside it, instead of round-tripping through HBM, is the central trick of a fast attention kernel.',
  },
  {
    id: 'smem',
    terms: ['SMEM', 'shared memory'],
    ci: false,
    title: 'Shared memory (SMEM)',
    short:
      'A software-managed scratchpad of SRAM inside each SM — 228 KiB on Hopper — that all threads in a block can read and write.',
    body: 'FlashAttention tiles attention so each tile of the score matrix fits here. For scale: one 8192×8192 score matrix is 134 MB, about 575× too large.',
  },
  {
    id: 'sm',
    terms: ['SM', 'SMs'],
    title: 'SM',
    expand: 'Streaming Multiprocessor',
    short:
      'The GPU’s core scheduling unit: its own registers, shared memory, warp schedulers and tensor cores. An H100 has 132 of them.',
    body: 'A kernel’s work is split into blocks, each assigned to one SM. Per-SM resources therefore bound how much data a kernel can keep on-chip at once.',
  },
  {
    id: 'l2',
    terms: ['L2 cache', 'L2'],
    title: 'L2 cache',
    short: 'A ~50 MB SRAM cache shared by every SM, sitting between them and HBM.',
    body: 'Large by cache standards and tiny by model standards — it cannot hold even one attention score matrix at 8K context.',
  },
  {
    id: 'bandwidth',
    terms: ['memory bandwidth', 'bandwidth'],
    ci: true,
    title: 'Memory bandwidth',
    short: 'Bytes per second movable between HBM and the GPU cores.',
    body: 'For a memory-bound kernel this is the only hardware number that matters: step time is simply bytes moved divided by bandwidth.',
  },
  {
    id: 'nvlink',
    terms: ['NVLink'],
    title: 'NVLink',
    short: 'NVIDIA’s GPU-to-GPU interconnect — 1.8 TB/s bidirectional per GPU on Blackwell.',
    body: 'Tensor parallelism needs a collective reduction at every layer, so interconnect latency lands directly on the critical path of every decode step.',
  },

  // ----------------------------------------------------------------- compute
  {
    id: 'flops',
    terms: ['FLOPs', 'FLOP', 'FLOP/s', 'TFLOP/s', 'TFLOPS', 'GFLOP', 'PFLOPS', 'petaFLOPS'],
    title: 'FLOPs',
    expand: 'Floating-point operations',
    short:
      'A count of arithmetic operations. A matrix multiply of an N-parameter weight matrix costs about 2N FLOPs per token — one multiply and one add each.',
    body: 'Note the distinction: FLOPs is a quantity of work, FLOP/s is a rate. An H200 does about 989 TFLOP/s in BF16.',
  },
  {
    id: 'bf16',
    terms: ['BF16', 'bf16', 'bfloat16'],
    ci: true,
    title: 'BF16',
    expand: 'Brain floating point, 16-bit',
    short:
      'A 16-bit float with the same exponent range as FP32 but fewer mantissa bits — the default training and reference format.',
    body: 'Two bytes per parameter, so GLM-5.2 in BF16 is 1403 GiB and does not fit on eight H200s.',
  },
  {
    id: 'fp8',
    terms: ['FP8'],
    title: 'FP8',
    short: 'An 8-bit float. One byte per parameter, halving both storage and bandwidth versus BF16.',
    body: 'Nearly universal for serving large models: it is the difference between GLM-5.2 fitting on an 8-GPU node and not.',
  },
  {
    id: 'fp4',
    terms: ['FP4'],
    title: 'FP4',
    short: 'A 4-bit float — half a byte per parameter.',
    body: 'Quarter the bytes of BF16, so quarter the decode traffic. Blackwell added hardware support, which is why 4-bit serving became practical there.',
  },
  {
    id: 'nvfp4',
    terms: ['NVFP4'],
    title: 'NVFP4',
    short:
      'NVIDIA’s 4-bit format for Blackwell: FP4 values with a small per-block scale factor to preserve accuracy.',
    body: 'The block scaling is what makes 4 bits survive real models. Published GLM-5.2 numbers on B200/B300 use NVFP4 checkpoints.',
  },
  {
    id: 'quantisation',
    terms: ['quantisation', 'quantization', 'quantise', 'quantize', 'quantised', 'quantized'],
    ci: true,
    title: 'Quantisation',
    short: 'Storing weights or activations in fewer bits than they were trained in.',
    body: 'For memory-bound inference it buys twice: fewer bytes to hold and fewer bytes to read. It is the single most effective lever on decode throughput.',
  },
  {
    id: 'sxm',
    terms: ['SXM'],
    title: 'SXM',
    short:
      'NVIDIA’s socketed board format for datacentre GPUs, as opposed to a PCIe card.',
    body: 'SXM modules get higher power limits and full NVLink connectivity, so they run faster than the PCIe variant of the same chip. "H200 SXM" names that package.',
  },
  {
    id: 'hopper',
    terms: ['Hopper'],
    title: 'Hopper',
    short: 'NVIDIA’s GPU architecture behind the H100 and H200.',
    body: 'Introduced the Transformer Engine and FP8. FlashAttention-3 was written specifically for its asynchronous execution model.',
  },
  {
    id: 'blackwell',
    terms: ['Blackwell'],
    title: 'Blackwell',
    short: 'NVIDIA’s architecture behind the B200 and B300.',
    body: 'Scales asymmetrically over Hopper: tensor core throughput roughly doubles while shared-memory bandwidth and the exponential units do not. FlashAttention-4 is a response to exactly that.',
  },
  {
    id: 'gpu-h200',
    terms: ['H200'],
    title: 'H200',
    short: '141 GB of HBM3e at 4.8 TB/s, with the same compute die as the H100.',
    body: 'Its advantage over the H100 is entirely memory: 76% more capacity and 43% more bandwidth, both of which are what decode is limited by.',
    links: [{ label: 'Specs', url: `${NV}h200/` }],
  },
  {
    id: 'gpu-h100',
    terms: ['H100'],
    title: 'H100',
    short: '80 GB of HBM3 at 3.35 TB/s, about 989 BF16 TFLOP/s, 132 SMs.',
    links: [{ label: 'Specs', url: `${NV}h100/` }],
  },
  {
    id: 'gpu-b200',
    terms: ['B200'],
    title: 'B200',
    short: '192 GB of HBM3e at 8 TB/s and about 2.25 BF16 PFLOP/s — the first Blackwell datacentre part.',
    links: [{ label: 'Specs', url: `${NV}dgx-b200/` }],
  },
  {
    id: 'gpu-b300',
    terms: ['B300'],
    title: 'B300',
    short: 'Blackwell Ultra: 288 GB of HBM3e at 8 TB/s, up to 15 dense FP4 PFLOP/s.',
  },
  {
    id: 'sfu',
    terms: ['SFU'],
    title: 'SFU',
    expand: 'Special Function Unit',
    short:
      'Hardware for transcendentals — exponentials, reciprocals, square roots — separate from the main arithmetic pipelines.',
    body: 'Softmax needs an exponential per element, so on Blackwell the SFUs became the attention bottleneck. FlashAttention-4 sidesteps them by approximating exp() on the FMA units instead.',
  },
  {
    id: 'fma',
    terms: ['FMA'],
    title: 'FMA',
    expand: 'Fused Multiply-Add',
    short: 'A single instruction computing a×b+c, the workhorse of numeric code.',
    body: 'Blackwell has FMA throughput to spare, so it can be cheaper to evaluate a polynomial approximation of exp() there than to queue for the special-function units.',
  },
  {
    id: 'cute-dsl',
    terms: ['CuTe-DSL', 'CuTe'],
    title: 'CuTe-DSL',
    short: 'A Python-embedded DSL in NVIDIA’s CUTLASS library for writing GPU kernels.',
    body: 'FlashAttention-4 is written entirely in it, which cut compile times 20–30× versus C++ templates.',
    links: [{ label: 'CUTLASS', url: 'https://github.com/NVIDIA/cutlass' }],
  },

  // ---------------------------------------------------------------- roofline
  {
    id: 'roofline',
    terms: ['roofline'],
    ci: true,
    title: 'Roofline model',
    short:
      'A performance model where a kernel’s ceiling is the lower of two limits: bandwidth × arithmetic intensity, or peak compute.',
    visual: roofline({ point: { ai: 2.3, label: 'decode' } }),
    body: 'Plotted against arithmetic intensity it looks like a roof: a sloped bandwidth-limited region, then a flat compute-limited one. Decode sits far down the slope.',
    links: [{ label: 'Williams et al. 2009', url: 'https://doi.org/10.1145/1498765.1498785' }],
  },
  {
    id: 'arith-intensity',
    terms: ['arithmetic intensity', 'operational intensity'],
    ci: true,
    title: 'Arithmetic intensity',
    short: 'FLOPs performed per byte moved from memory. Written I = FLOPs / bytes.',
    body: 'A property of the algorithm, not the hardware. Compare it against machine balance to learn which resource you are actually waiting on. GLM-5.2 decode is about 2.3 FLOP/byte at batch 1.',
  },
  {
    id: 'machine-balance',
    terms: ['machine balance'],
    ci: true,
    title: 'Machine balance',
    short:
      'The arithmetic intensity at which a machine stops being bandwidth-limited: peak compute ÷ peak bandwidth.',
    body: 'For an H200, 989 TFLOP/s ÷ 4.8 TB/s ≈ 206 FLOP/byte. Below that the tensor cores idle waiting for memory.',
  },
  {
    id: 'memory-bound',
    terms: ['memory-bound', 'bandwidth-bound'],
    ci: true,
    title: 'Memory-bound',
    short: 'The kernel finishes when the bytes arrive; the arithmetic units are partly idle.',
    body: 'Decoding one token at a time is severely memory-bound: tens of gigabytes of weights are read to do a few tens of gigaflops of work.',
  },
  {
    id: 'compute-bound',
    terms: ['compute-bound'],
    ci: true,
    title: 'Compute-bound',
    short: 'The kernel finishes when the arithmetic finishes; memory keeps up.',
    body: 'Prefill is compute-bound because one pass over the weights serves the whole prompt at once.',
  },
  {
    id: 'mbu',
    terms: ['MBU', 'memory-bandwidth utilisation', 'memory bandwidth utilization'],
    title: 'MBU',
    expand: 'Memory-Bandwidth Utilisation',
    short: 'The fraction of theoretical peak bandwidth a kernel actually achieves.',
    body: 'The gap between roofline arithmetic and a real server. At batch 1 it is often under 20%, eaten by kernel launch overhead, scattered expert reads and per-layer collectives.',
  },

  // ----------------------------------------------------------------- serving
  {
    id: 'tps',
    terms: ['tokens per second', 'tok/s', 'TPS'],
    ci: true,
    title: 'Tokens per second',
    short: 'The throughput of generation — but ambiguous unless you say per user or aggregate.',
    body: 'Per-user TPS is 1 ÷ step time. Aggregate TPS is batch ÷ step time. Batching pushes one down while pushing the other up, so a single unqualified number says very little.',
  },
  {
    id: 'prefill',
    terms: ['prefill'],
    ci: true,
    title: 'Prefill',
    short: 'The pass that processes the whole prompt and builds its KV cache before any token is emitted.',
    body: 'One pass over the weights serves every prompt token, so intensity scales with prompt length and prefill is compute-bound. It determines TTFT.',
  },
  {
    id: 'decode',
    terms: ['decode'],
    ci: true,
    title: 'Decode',
    short: 'The token-at-a-time generation loop that follows prefill.',
    body: 'Each step reads all active weights to produce one token per sequence, making it memory-bound. It determines TPOT — what a user experiences as speed.',
  },
  {
    id: 'ttft',
    terms: ['TTFT', 'time-to-first-token'],
    title: 'TTFT',
    expand: 'Time To First Token',
    short: 'Latency from request to the first emitted token — essentially prefill time.',
    body: 'Grows with prompt length. In the published GLM-5.2 numbers it ranges from 196 ms at concurrency 1 to 6.4 s at concurrency 1024.',
  },
  {
    id: 'tpot',
    terms: ['TPOT', 'time-per-output-token'],
    title: 'TPOT',
    expand: 'Time Per Output Token',
    short: 'The steady-state gap between tokens during decode; the reciprocal of per-user tokens per second.',
    body: 'The number a reader actually feels. 1.86 ms is a fast stream; 280 ms is a server rationing bandwidth across a thousand requests.',
  },
  {
    id: 'pd-disagg',
    terms: ['prefill/decode disaggregation', 'PD disaggregation', 'disaggregation'],
    ci: true,
    title: 'Prefill/decode disaggregation',
    short: 'Running prefill and decode on separate GPU pools.',
    body: 'They have opposite bottlenecks — one compute-bound, one memory-bound — so mixing them on the same devices means neither runs at its best. Splitting them lets each pool be sized and batched independently.',
  },
  {
    id: 'batch',
    terms: ['batch size', 'batching', 'batch'],
    ci: true,
    title: 'Batch size',
    short: 'How many sequences are decoded in the same step, sharing one pass over the weights.',
    visual: grid(8, 256),
    body: 'The standard fix for a memory-bound kernel. It works far less well for a mixture of experts, because each extra sequence pulls in experts nobody was reading — the grid above is 8 of 256 experts at batch 1.',
  },
  {
    id: 'concurrency',
    terms: ['concurrency'],
    ci: true,
    title: 'Concurrency',
    short: 'The number of requests a server is serving at once.',
    body: 'Raising it raises aggregate throughput and per-user latency together. There is no setting that is simply fast — only a choice about who waits.',
  },
  {
    id: 'tensor-parallel',
    terms: ['tensor parallelism', 'tensor-parallel', 'tensor parallel'],
    ci: true,
    title: 'Tensor parallelism',
    short: 'Splitting each weight matrix across GPUs so they compute one layer together.',
    body: 'It multiplies the aggregate bandwidth available to a step, which is why a 753B model can decode quickly at all. The cost is a collective reduction at every layer — 78 layers, twice each, on the critical path.',
  },
  {
    id: 'all-reduce',
    terms: ['all-reduce', 'collective reduction', 'collective'],
    ci: true,
    title: 'All-reduce',
    short: 'A collective in which every GPU contributes a partial result and all receive the sum.',
    body: 'Tensor parallelism needs one after each split matrix multiply. At batch 1 the latency of these, not the arithmetic, can dominate the step.',
  },
  {
    id: 'continuous-batching',
    terms: ['continuous batching', 'chunked prefill'],
    ci: true,
    title: 'Continuous batching',
    short: 'Admitting and retiring sequences every step instead of running fixed batches to completion.',
    body: 'Keeps the batch full as requests of different lengths come and go, which is what makes high aggregate throughput achievable in practice.',
  },

  // ------------------------------------------------------------ architecture
  // One definition, two entries: the acronym must match case-sensitively while
  // the spelled-out phrase should not. Pattern sets stay disjoint so neither
  // shadows the other in the lookup table.
  {
    id: 'moe',
    terms: ['MoE'],
    ci: false,
    ...MOE_DEF,
  },
  {
    id: 'moe-phrase',
    terms: ['mixture of experts', 'mixture-of-experts'],
    ci: true,
    ...MOE_DEF,
  },
  {
    id: 'expert',
    terms: ['expert', 'experts'],
    ci: true,
    title: 'Expert',
    short: 'One feed-forward sub-network inside a mixture-of-experts layer.',
    body: 'In GLM-5.2 each is 3 × 6144 × 2048 ≈ 37.7M parameters, and there are 256 routed plus 1 shared in each of the 75 MoE layers.',
  },
  {
    id: 'routed-experts',
    terms: ['routed expert', 'routed experts'],
    ci: true,
    title: 'Routed experts',
    short: 'The experts a router picks per token — 8 of 256 per layer in GLM-5.2.',
    body: 'They hold 96% of the model’s parameters and 3% of them run for any given token. This is simultaneously why the model is cheap to run and why it is expensive to batch.',
  },
  {
    id: 'shared-expert',
    terms: ['shared expert', 'shared experts'],
    ci: true,
    title: 'Shared expert',
    short: 'An expert that runs for every token regardless of routing.',
    body: 'It absorbs whatever is common to all tokens, freeing the routed experts to specialise. Always read, so it is part of the fixed cost of a step.',
  },
  {
    id: 'router',
    terms: ['router', 'routing', 'top-k routing'],
    ci: true,
    title: 'Router',
    short: 'A small linear layer that scores every expert and keeps the top few.',
    body: 'Negligible in parameters and FLOPs, but it decides which ~9.7B of expert weights get read per layer — so it sets the bandwidth bill for everything downstream.',
  },
  {
    id: 'active-params',
    terms: ['active parameters', 'active parameter', 'active per token'],
    ci: true,
    title: 'Active parameters',
    short: 'The subset of weights that participate in producing one token.',
    body: 'Total parameters set what you must store; active parameters set what you must read. Different constraints, different hardware limits — conflating them is the commonest estimation error.',
  },
  {
    id: 'mla',
    terms: ['MLA', 'multi-head latent attention'],
    ci: false,
    title: 'MLA',
    expand: 'Multi-head Latent Attention',
    short:
      'Attention that caches one low-rank latent per token per layer instead of keys and values per head.',
    visual: bars(
      [
        { label: 'MLA latent', value: 87.75, display: '87.75 KiB', hi: true },
        { label: 'plain MHA', value: 4997, display: '4.88 MiB' },
      ],
      { width: 250 },
    ),
    body: 'GLM-5.2 stores a 512-dimensional latent plus a 64-dimensional RoPE key — 87.75 KiB per token across 78 layers, 57× less than the same shapes as multi-head attention. This is what makes a million-token context affordable.',
    links: [{ label: 'DeepSeek-V2 paper', url: 'https://arxiv.org/abs/2405.04434' }],
  },
  {
    id: 'mha',
    terms: ['MHA', 'multi-head attention'],
    ci: false,
    title: 'MHA',
    expand: 'Multi-Head Attention',
    short: 'Standard attention: every head keeps its own keys and values, all cached per token.',
    body: 'Simple and expensive. For GLM-5.2’s shapes it would need 4.88 MiB of cache per token, which at 1M context is 4.9 TiB for a single sequence.',
  },
  {
    id: 'kv-cache',
    terms: ['KV cache', 'KV-cache', 'key-value cache'],
    ci: true,
    title: 'KV cache',
    short:
      'The stored keys and values of all previous tokens, so each new token can attend to them without recomputing.',
    body: 'It converts attention from quadratic recomputation into a memory cost that grows linearly with context × batch, and it competes with the weights for HBM capacity.',
  },
  {
    id: 'rope',
    terms: ['RoPE'],
    title: 'RoPE',
    expand: 'Rotary Position Embedding',
    short: 'Positional information injected by rotating query and key vectors by an angle set by position.',
    body: 'Relative by construction, and it extrapolates to longer contexts better than learned absolute embeddings. GLM-5.2 keeps a 64-dimensional RoPE component outside the compressed MLA latent.',
  },
  {
    id: 'dsa',
    terms: ['DSA', 'DeepSeek Sparse Attention'],
    ci: false,
    title: 'DSA',
    expand: 'DeepSeek Sparse Attention',
    short:
      'Sparse attention where a lightweight indexer selects the top-k most relevant tokens per query, cutting core attention from O(S²) to O(Sk).',
    body: 'GLM-5.2 uses k = 2048, so attention reads 2048 tokens whether the context is 8K or 1M. The indexer that makes the choice is itself O(S²), which is the cost sparse attention adds back.',
  },
  {
    id: 'indexer',
    terms: ['indexer', 'lightning indexer'],
    ci: true,
    title: 'Indexer',
    short:
      'The small scoring network in front of sparse attention that decides which keys each query will attend to.',
    body: 'Tiny in parameters — 9.4M per copy — but it must score every token in the context, so its cost grows with context length. GLM-5.2 carries only 22 of them across 79 layers.',
  },
  {
    id: 'indexshare',
    terms: ['IndexShare', 'IndexCache', 'cross-layer index reuse'],
    ci: false,
    title: 'IndexShare',
    short:
      'Running an indexer on only a minority of layers and having the rest reuse the nearest one’s selection.',
    body: 'Consecutive layers pick nearly the same tokens, so most layers need not choose for themselves. GLM-5.2 keeps 21 of 78 — visible in the checkpoint as indexer weights on 22 layers — for a reported 2.9× cut in per-token FLOPs at 1M context.',
    links: [{ label: 'IndexCache paper', url: 'https://arxiv.org/abs/2603.12201' }],
  },
  {
    id: 'lm-head',
    terms: ['LM head', 'unembedding'],
    ci: true,
    title: 'LM head',
    short: 'The final matrix projecting the hidden state onto vocabulary logits.',
    body: 'For GLM-5.2 that is 6144 × 154,880 ≈ 952M parameters, streamed in full on every step. Unlike the embedding table it is a real matrix multiply.',
  },
  {
    id: 'embedding',
    terms: ['token embedding', 'embedding table', 'embedding'],
    ci: true,
    title: 'Token embedding',
    short: 'The lookup table mapping token ids to vectors.',
    body: 'Large in parameters, nearly free to run: a lookup reads one row, not the matrix. A good reminder that parameter counts are a poor proxy for cost.',
  },
  {
    id: 'context',
    terms: ['context length', 'context window', 'context'],
    ci: true,
    title: 'Context length',
    short: 'How many tokens the model can attend over. GLM-5.2 supports 1,048,576.',
    body: 'Long context costs memory before it costs arithmetic: KV cache grows linearly with it, and at 1M tokens one sequence needs 87.75 GiB even with MLA.',
  },

  // ------------------------------------------------------------- attention kernels
  {
    id: 'flashattention',
    terms: [
      'FlashAttention',
      'FlashAttention-2',
      'FlashAttention-3',
      'FlashAttention-4',
      'FA2',
      'FA3',
      'FA4',
    ],
    ci: false,
    title: 'FlashAttention',
    short:
      'An exact attention algorithm that never writes the S×S score matrix to memory, tiling the computation so each block stays in on-chip SRAM.',
    visual: bars(
      [
        { label: 'score matrix', value: 134.2, display: '134 MB' },
        { label: 'FlashAttention', value: 16.8, display: '16.8 MB', hi: true },
      ],
      { width: 250 },
    ),
    body: 'Same arithmetic and same numerics as standard attention, asymptotically less memory traffic. Version 2 improved work partitioning, 3 targeted Hopper’s asynchrony, 4 targeted Blackwell’s asymmetric scaling.',
    links: [
      { label: 'FlashAttention (2022)', url: 'https://arxiv.org/abs/2205.14135' },
      { label: 'FlashAttention-4 (2026)', url: 'https://arxiv.org/abs/2603.05451' },
      { label: 'Code', url: 'https://github.com/Dao-AILab/flash-attention' },
    ],
  },
  {
    id: 'tiling',
    terms: ['tiling', 'tiles', 'tile'],
    ci: true,
    title: 'Tiling',
    short: 'Restructuring a computation into blocks small enough to fit in fast memory.',
    body: 'The general technique behind most fast GPU kernels, and the specific reason FlashAttention avoids materialising the score matrix.',
  },
  {
    id: 'softmax',
    terms: ['softmax'],
    ci: true,
    title: 'Softmax',
    short: 'Turns a vector of scores into a probability distribution by exponentiating and normalising.',
    body: 'Attention applies it across each query’s scores. The exponential is why attention kernels care about special-function throughput, and the normalisation is what FlashAttention must compute incrementally.',
  },
  {
    id: 'pagedattention',
    terms: ['PagedAttention', 'paged attention'],
    ci: false,
    title: 'PagedAttention',
    short: 'Managing the KV cache as fixed-size pages, like virtual memory, instead of contiguous blocks.',
    body: 'Removes the fragmentation and over-reservation that naive allocation causes, so a server can hold far more concurrent sequences in the same HBM. The basis of vLLM.',
    links: [
      { label: 'Paper', url: 'https://arxiv.org/abs/2309.06180' },
      { label: 'vLLM', url: 'https://github.com/vllm-project/vllm' },
    ],
  },

  // --------------------------------------------------------------- speculation
  {
    id: 'spec-decoding',
    terms: ['speculative decoding', 'speculative sampling', 'speculation'],
    ci: true,
    title: 'Speculative decoding',
    short:
      'Draft several tokens cheaply, then verify them all in one pass of the real model, keeping the prefix it agrees with.',
    visual: draftStrip(5, 4),
    body: 'Exact, not approximate: the output distribution is unchanged. It works because verification of k+1 positions costs little more than one position when you are memory-bound — though on a sparse MoE the extra positions do pull in extra experts.',
    links: [{ label: 'Leviathan et al. 2023', url: 'https://arxiv.org/abs/2211.17192' }],
  },
  {
    id: 'mtp',
    terms: ['MTP', 'MTP head', 'MTP heads', 'multi-token prediction'],
    ci: false,
    title: 'MTP',
    expand: 'Multi-Token Prediction',
    short:
      'An extra layer trained to predict the token after next from the main model’s hidden state, used to draft speculative candidates.',
    body: 'Because it shares the backbone’s representations its drafts are accepted often. GLM-5.2 ships one such layer, ~9.9B parameters, idle unless speculation is enabled; serving stacks iterate it to draft several tokens.',
    links: [{ label: 'DeepSeek-V3 report', url: 'https://arxiv.org/abs/2412.19437' }],
  },
  {
    id: 'acceptance',
    terms: ['acceptance rate', 'acceptance length', 'accepted length'],
    ci: true,
    title: 'Acceptance rate',
    short: 'The probability that the target model keeps a drafted token, written α.',
    body: 'Expected tokens per verification step is (1 − α^(k+1)) / (1 − α) for a draft of length k. Returns diminish sharply in k, so a high α is worth much more than a long draft.',
  },
  {
    id: 'eagle',
    terms: ['EAGLE'],
    title: 'EAGLE',
    short: 'A speculative decoding method that drafts from the target model’s own hidden states.',
    body: 'The name serving frameworks use for this family of draft heads — SGLang enables GLM-5.2’s MTP layer through its EAGLE implementation.',
  },

  // ----------------------------------------------------------------- serving stacks
  {
    id: 'sglang',
    terms: ['SGLang'],
    title: 'SGLang',
    short: 'An open-source LLM serving framework, and one of the reference deployments for GLM-5.2.',
    links: [
      { label: 'GLM-5.2 cookbook', url: 'https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.2' },
      { label: 'GitHub', url: 'https://github.com/sgl-project/sglang' },
    ],
  },
  {
    id: 'vllm',
    terms: ['vLLM'],
    title: 'vLLM',
    short: 'A widely used LLM serving engine, originally built around PagedAttention.',
    links: [{ label: 'GitHub', url: 'https://github.com/vllm-project/vllm' }],
  },
]
