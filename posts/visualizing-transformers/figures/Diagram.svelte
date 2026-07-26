<script>
  /** Thin wrapper: build the specs from a real config and hand them to the renderer. */
  import BlockDiagram from '../../../src/components/BlockDiagram.svelte'
  import { buildDiagrams } from './diagrams.js'
  import config from '../data/mixtral-8x7b.json'

  let { start = 'model' } = $props()
  const diagrams = buildDiagrams(config, { tokens: 512 })

  /** Clicking a primitive box scrolls to the figure that teaches it. */
  const onFigure = (id) => {
    const el = document.getElementById(id)
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }
</script>

<BlockDiagram {diagrams} {start} {onFigure} />
