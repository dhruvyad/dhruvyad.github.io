import { mountFigure } from '../../src/lib/figure.js'
import TemperatureBars from './figures/TemperatureBars.svelte'
import ShiftInvariance from './figures/ShiftInvariance.svelte'

// Each figure builds itself only once it nears the viewport — see src/lib/figure.js.
mountFigure('temperature-bars', TemperatureBars)
mountFigure('shift-invariance', ShiftInvariance)
