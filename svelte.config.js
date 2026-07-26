/**
 * Svelte 5 defaults are what we want (runes mode on, no SSR). This file exists
 * mainly so the toolchain stops guessing and so compiler options have a home.
 */
export default {
  compilerOptions: {
    // Figures live inside <d-figure>, which is a custom element in the light DOM,
    // so ordinary scoped styles apply normally. Nothing exotic needed here.
  },
}
