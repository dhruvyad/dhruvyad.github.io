import { mountFigure } from '../../src/lib/figure.js'
import { installGlossary } from '../../src/lib/glossary.js'
import { TERMS } from '../../src/glossary/terms.js'
import BlockDiagram from '../../src/components/BlockDiagram.svelte'
import { buildDiagrams } from '../../src/lib/diagrams.js'
import glmConfig from './data/glm-5.2-config.json'
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

// The same diagram component the model-explorer tool uses, fed this article's
// subject. Walks from the whole network down to a single matrix multiply.
mountFigure('architecture', BlockDiagram, { diagrams: buildDiagrams(glmConfig) })
mountFigure('decode-budget', DecodeBudget)
mountFigure('expert-traffic', ExpertTraffic)
mountFigure('memory-budget', MemoryBudget)
mountFigure('attention-traffic', AttentionTraffic)
mountFigure('speculative', Speculative)
