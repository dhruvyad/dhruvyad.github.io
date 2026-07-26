import { mountFigure } from '../../src/lib/figure.js'
import DecodeBudget from './figures/DecodeBudget.svelte'
import ExpertTraffic from './figures/ExpertTraffic.svelte'
import MemoryBudget from './figures/MemoryBudget.svelte'
import AttentionTraffic from './figures/AttentionTraffic.svelte'
import Speculative from './figures/Speculative.svelte'

mountFigure('decode-budget', DecodeBudget)
mountFigure('expert-traffic', ExpertTraffic)
mountFigure('memory-budget', MemoryBudget)
mountFigure('attention-traffic', AttentionTraffic)
mountFigure('speculative', Speculative)
