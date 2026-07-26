import { readdirSync, existsSync, readFileSync } from 'node:fs'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'

const root = dirname(fileURLToPath(import.meta.url))
const postsDir = resolve(root, 'posts')

/** Every directory under posts/ that has an index.html is a post. */
function postSlugs() {
  if (!existsSync(postsDir)) return []
  return readdirSync(postsDir, { withFileTypes: true })
    .filter((e) => e.isDirectory() && existsSync(resolve(postsDir, e.name, 'index.html')))
    .map((e) => e.name)
    .sort()
}

/**
 * Reads the <d-front-matter> JSON out of a post's index.html. That block is the
 * single source of truth for a post's metadata: distill's template renders the
 * byline from it, and the site index below reads the same bytes.
 */
function readFrontMatter(slug) {
  const html = readFileSync(resolve(postsDir, slug, 'index.html'), 'utf8')
  const match = html.match(/<d-front-matter>\s*<script type="text\/json">([\s\S]*?)<\/script>/)
  if (!match) throw new Error(`posts/${slug}/index.html has no <d-front-matter> JSON block`)
  try {
    return JSON.parse(match[1])
  } catch (err) {
    throw new Error(`posts/${slug}/index.html has invalid front-matter JSON: ${err.message}`)
  }
}

/**
 * Exposes the post list to the homepage as `virtual:posts`, newest first, so
 * adding a post never means editing an index by hand.
 */
function postsIndex() {
  const virtualId = 'virtual:posts'
  const resolvedId = '\0' + virtualId
  return {
    name: 'notes:posts-index',
    resolveId(id) {
      if (id === virtualId) return resolvedId
    },
    load(id) {
      if (id !== resolvedId) return
      const posts = postSlugs()
        .map((slug) => {
          const fm = readFrontMatter(slug)
          return {
            slug,
            title: fm.title ?? slug,
            description: fm.description ?? '',
            published: fm.published ?? null,
            tags: fm.tags ?? [],
          }
        })
        .sort((a, b) => String(b.published ?? '').localeCompare(String(a.published ?? '')))
      return `export default ${JSON.stringify(posts, null, 2)}`
    },
    configureServer(server) {
      // Adding or retitling a post should show up without restarting the dev server.
      server.watcher.add(resolve(postsDir, '**/index.html'))
      const invalidate = (file) => {
        if (!file.startsWith(postsDir)) return
        const mod = server.moduleGraph.getModuleById(resolvedId)
        if (mod) server.moduleGraph.invalidateModule(mod)
        server.ws.send({ type: 'full-reload' })
      }
      server.watcher.on('add', invalidate)
      server.watcher.on('change', invalidate)
      server.watcher.on('unlink', invalidate)
    },
  }
}

/**
 * <d-bibliography src="bibliography.bib"> is a custom-element attribute, so Vite's
 * HTML pipeline never sees it. Copy each post's .bib next to its page so the
 * relative URL in the markup resolves in dev and in the build alike.
 */
function copyBibliographies() {
  return {
    name: 'notes:copy-bibliographies',
    generateBundle() {
      for (const slug of postSlugs()) {
        const dir = resolve(postsDir, slug)
        for (const file of readdirSync(dir)) {
          if (!file.endsWith('.bib')) continue
          this.emitFile({
            type: 'asset',
            fileName: `posts/${slug}/${file}`,
            source: readFileSync(resolve(dir, file)),
          })
        }
      }
    },
  }
}

export default defineConfig({
  // User site: served from the root of https://dhruvyad.github.io/
  base: '/',
  appType: 'mpa',
  plugins: [svelte(), postsIndex(), copyBibliographies()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        index: resolve(root, 'index.html'),
        ...Object.fromEntries(
          postSlugs().map((slug) => [slug, resolve(postsDir, slug, 'index.html')]),
        ),
      },
    },
  },
})
