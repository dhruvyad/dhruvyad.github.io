import { mount as svelteMount, unmount } from 'svelte'

/**
 * Helpers around <d-figure>'s visibility state machine.
 *
 * <d-figure> gives three events (see distill's d-figure.js):
 *
 *   ready      fired once, when the figure comes within ~2 viewport heights.
 *              This is where expensive setup belongs — the whole point is that a
 *              page with 20 simulations doesn't build all 20 on load.
 *   onscreen   fired every time the figure scrolls into actual view.
 *   offscreen  fired every time it leaves.
 *
 * d-figure also replays `ready` for listeners that attach late, so there's no
 * race between the module loading and the figure scrolling past.
 */

function requireFigure(id) {
  const figure = document.getElementById(id)
  if (!figure) throw new Error(`No <d-figure id="${id}"> on the page`)
  return figure
}

/**
 * Mounts a Svelte component into <d-figure id="{id}"> the first time it nears the
 * viewport. Returns a promise resolving to the component instance, mostly so
 * callers can await it in tests or the console.
 *
 *   mountFigure('softmax', SoftmaxTemperature, { logits: [2, 1, 0.5] })
 *
 * The component mounts into an inner `#{id}-target` element if the markup has
 * one, otherwise into the figure itself.
 */
export function mountFigure(id, Component, props = {}) {
  const figure = requireFigure(id)
  return new Promise((resolve) => {
    figure.addEventListener('ready', () => {
      const target = figure.querySelector(`#${id}-target`) ?? figure
      resolve(svelteMount(Component, { target, props }))
    })
  })
}

/**
 * Like mountFigure, but takes a `() => import('./Heavy.svelte')` thunk and only
 * downloads the module once the figure nears the viewport.
 *
 * Worth it for figures with a big dependency — the 3D view pulls in three.js,
 * ~157 kB gzipped, which no reader should pay for before scrolling to it.
 */
export function mountFigureLazy(id, loader, props = {}) {
  const figure = requireFigure(id)
  return new Promise((resolve, reject) => {
    figure.addEventListener('ready', async () => {
      try {
        const mod = await loader()
        const target = figure.querySelector(`#${id}-target`) ?? figure
        resolve(svelteMount(mod.default, { target, props }))
      } catch (err) {
        console.error(`Figure "${id}" failed to load:`, err)
        reject(err)
      }
    })
  })
}

/**
 * Same as mountFigure, but unmounts when the figure scrolls away and remounts
 * when it returns. For figures running a requestAnimationFrame loop that you
 * don't want burning cycles off-screen and don't need to preserve state for.
 */
export function mountFigureWhileVisible(id, Component, props = {}) {
  const figure = requireFigure(id)
  let instance = null
  const target = () => figure.querySelector(`#${id}-target`) ?? figure

  figure.addEventListener('ready', () => {
    if (!instance) instance = svelteMount(Component, { target: target(), props })
  })
  figure.addEventListener('onscreen', () => {
    if (!instance) instance = svelteMount(Component, { target: target(), props })
  })
  figure.addEventListener('offscreen', () => {
    if (instance) {
      unmount(instance)
      instance = null
    }
  })
}

/**
 * Escape hatch for figures written as plain D3/canvas rather than Svelte.
 * `setup(element)` runs once when the figure nears the viewport; if it returns a
 * function, that runs on offscreen (handy for stopping timers).
 *
 *   onFigureReady('grid', (el) => { const t = d3.timer(...); return () => t.stop() })
 */
export function onFigureReady(id, setup) {
  const figure = requireFigure(id)
  figure.addEventListener('ready', () => {
    const teardown = setup(figure.querySelector(`#${id}-target`) ?? figure)
    if (typeof teardown === 'function') {
      figure.addEventListener('offscreen', teardown)
    }
  })
}
