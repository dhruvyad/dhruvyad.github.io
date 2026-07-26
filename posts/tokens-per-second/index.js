import { mountFigure } from '../../src/lib/figure.js'
import { installGlossary } from '../../src/lib/glossary.js'
import { TERMS } from '../../src/glossary/terms.js'
import Architecture from './figures/Architecture.svelte'
import DecodeBudget from './figures/DecodeBudget.svelte'
import ExpertTraffic from './figures/ExpertTraffic.svelte'
import MemoryBudget from './figures/MemoryBudget.svelte'
import AttentionTraffic from './figures/AttentionTraffic.svelte'
import Speculative from './figures/Speculative.svelte'

// Wrap every mention of every glossary term. Runs before the figures mount so it
// only ever sees prose, never generated SVG. The appendix is watched separately
// because distill builds the footnote and reference lists after this module runs.
installGlossary({
  roots: [document.querySelector('d-title'), document.querySelector('d-article')],
  terms: TERMS,
  observe: document.querySelector('d-appendix'),
})

mountFigure('architecture', Architecture)
mountFigure('decode-budget', DecodeBudget)
mountFigure('expert-traffic', ExpertTraffic)
mountFigure('memory-budget', MemoryBudget)
mountFigure('attention-traffic', AttentionTraffic)
mountFigure('speculative', Speculative)
