import posts from 'virtual:posts'

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

const list = document.getElementById('post-list')

if (posts.length === 0) {
  list.outerHTML = `<div class="empty">
    No articles yet. Start one with <code>npm run new-post my-slug</code>.
  </div>`
} else {
  list.innerHTML = posts
    .map((post) => {
      const tags = post.tags.map((t) => `<span class="tag">${t}</span>`).join('')
      const date = formatDate(post.published)
      const meta = [date && `<span>${date}</span>`, tags].filter(Boolean).join('')
      return `<li>
      <h3><a href="posts/${post.slug}/">${post.title}</a></h3>
      <p class="description">${post.description}</p>
      ${meta ? `<div class="meta">${meta}</div>` : ''}
    </li>`
    })
    .join('')
}
