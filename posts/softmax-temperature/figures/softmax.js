/**
 * Shared math for this post's figures.
 */

/**
 * Numerically stable softmax with temperature.
 *
 * Subtracting the max before exponentiating is the standard trick: it can't
 * change the result (softmax is shift-invariant) but it guarantees the largest
 * exponent is exp(0) = 1, so nothing overflows.
 */
export function softmax(logits, temperature = 1) {
  const scaled = logits.map((z) => z / temperature)
  const max = Math.max(...scaled)
  const exps = scaled.map((z) => Math.exp(z - max))
  const total = exps.reduce((a, b) => a + b, 0)
  return exps.map((e) => e / total)
}

/** Shannon entropy in bits — how undecided the distribution is. */
export function entropyBits(probs) {
  return -probs.reduce((sum, p) => (p > 0 ? sum + p * Math.log2(p) : sum), 0)
}

export const clamp = (value, min, max) => Math.min(max, Math.max(min, value))
