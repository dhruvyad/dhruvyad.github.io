import posts from 'virtual:posts'
import tools from 'virtual:tools'

/**
 * `virtual:posts` is generated at build time from each post's <d-front-matter>
 * block (see vite.config.js), so this list can never drift from the articles.
 */

const formatDate = (iso) => {
  if (!iso) return ''
  const date = new Date(iso)
  if (Number.isNaN(date.valueOf())) return ''
  return date.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    timeZone: 'UTC',
  })
}

const entry = (item, base) => {
  const tags = (item.tags ?? []).map((t) => `<span class="tag">${t}</span>`).join('')
  const date = formatDate(item.published)
  const meta = [date && `<span>${date}</span>`, tags].filter(Boolean).join('')
  return `<li>
    <h3><a href="${base}/${item.slug}/">${item.title}</a></h3>
    <p class="description">${item.description}</p>
    ${meta ? `<div class="meta">${meta}</div>` : ''}
  </li>`
}

const render = (id, items, base, emptyHtml) => {
  const list = document.getElementById(id)
  if (!list) return
  if (items.length === 0) list.outerHTML = `<div class="empty">${emptyHtml}</div>`
  else list.innerHTML = items.map((i) => entry(i, base)).join('')
}

render(
  'post-list',
  posts,
  'posts',
  'No articles yet. Start one with <code>npm run new-post my-slug</code>.',
)
render('tool-list', tools, 'tools', 'No tools yet.')
