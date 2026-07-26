import { mountFigure } from '../../src/lib/figure.js'
import { installGlossary } from '../../src/lib/glossary.js'
import { TERMS } from '../../src/glossary/terms.js'
import DotProduct from './figures/DotProduct.svelte'
import MatMul from './figures/MatMul.svelte'
import Attention from './figures/Attention.svelte'
import ModelMap from './figures/ModelMap.svelte'
import Diagram from './figures/Diagram.svelte'

installGlossary({
  roots: [document.querySelector('d-title'), document.querySelector('d-article')],
  terms: TERMS,
  observe: document.querySelector('d-appendix'),
})

mountFigure('dot-product', DotProduct)
mountFigure('matmul', MatMul)
mountFigure('attention', Attention)
mountFigure('model-map', ModelMap)
mountFigure('diagram', Diagram)
