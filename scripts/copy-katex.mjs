/**
 * Copies KaTeX's dist into public/vendor/katex/ so the vendored distill template
 * can load it as a sibling file.
 *
 * KaTeX comes from npm (pinned in package.json) rather than being committed,
 * because dist/fonts is ~60 binary files. This runs on predev and prebuild.
 */
import { cpSync, existsSync, mkdirSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const from = resolve(root, 'node_modules/katex/dist')
const to = resolve(root, 'public/vendor/katex')

if (!existsSync(from)) {
  console.error('node_modules/katex/dist not found — run `npm install` first.')
  process.exit(1)
}

mkdirSync(to, { recursive: true })
for (const entry of ['katex.min.js', 'katex.min.css', 'fonts']) {
  cpSync(resolve(from, entry), resolve(to, entry), { recursive: true })
}
console.log('Copied KaTeX into public/vendor/katex/')
