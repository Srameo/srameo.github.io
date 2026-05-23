(function () {
    const POSTS_MANIFEST = '/blog/posts.json';
    const POSTS_DIR = '/blog/posts/';
    const BLOG_CONFIG = '/blog/config.json';
    const DEFAULT_POST = 'template';
    const DEFAULT_CONFIG = {
        title: 'Notes and research writeups',
        summary: 'A small Markdown-powered space for technical notes, experiments, and longer-form thoughts.'
    };

    function escapeHtml(value) {
        return String(value)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function escapeAttr(value) {
        return escapeHtml(value).replace(/`/g, '&#96;');
    }

    function sanitizeUrl(value) {
        const url = String(value || '').trim();
        if (!url) return '#';
        if (/^(https?:|mailto:|\/|#|\.\.?\/)/i.test(url)) return url;
        return '#';
    }

    function slugify(value) {
        return String(value)
            .trim()
            .toLowerCase()
            .replace(/<[^>]+>/g, '')
            .replace(/[^a-z0-9\u4e00-\u9fa5]+/g, '-')
            .replace(/^-+|-+$/g, '') || 'section';
    }

    function parseFrontMatter(markdown) {
        const source = markdown.replace(/\r\n?/g, '\n');
        if (!source.startsWith('---\n')) {
            return { meta: {}, body: source };
        }

        const end = source.indexOf('\n---', 4);
        if (end === -1) {
            return { meta: {}, body: source };
        }

        const rawMeta = source.slice(4, end).trim();
        const body = source.slice(end + 4).replace(/^\n/, '');
        const meta = {};

        rawMeta.split('\n').forEach(function (line) {
            const match = line.match(/^([A-Za-z0-9_-]+):\s*(.*)$/);
            if (!match) return;
            let value = match[2].trim();
            if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
                value = value.slice(1, -1);
            }
            meta[match[1]] = value;
        });

        return { meta, body };
    }

    function extractFootnotes(markdown) {
        const lines = markdown.replace(/\r\n?/g, '\n').split('\n');
        const bodyLines = [];
        const footnotes = new Map();
        let index = 0;

        while (index < lines.length) {
            const match = lines[index].match(/^\[\^([^\]]+)\]:\s*(.*)$/);
            if (!match) {
                bodyLines.push(lines[index]);
                index += 1;
                continue;
            }

            const id = match[1].trim();
            const content = [match[2].trim()];
            index += 1;

            while (index < lines.length && /^( {2,}|\t)/.test(lines[index])) {
                content.push(lines[index].replace(/^( {2,}|\t)/, '').trim());
                index += 1;
            }

            footnotes.set(id, content.join(' ').trim());
        }

        return {
            body: bodyLines.join('\n'),
            footnotes
        };
    }

    function createMarkdownContext(footnotes) {
        return {
            footnotes,
            footnoteNumbers: new Map(),
            footnoteOrder: [],
            footnoteRefCounts: new Map()
        };
    }

    function getFootnoteTargetId(id) {
        return `fn-${slugify(id)}`;
    }

    function registerFootnote(id, context) {
        if (!context || !context.footnotes || !context.footnotes.has(id)) {
            return null;
        }

        if (!context.footnoteNumbers.has(id)) {
            context.footnoteOrder.push(id);
            context.footnoteNumbers.set(id, context.footnoteOrder.length);
        }

        const refCount = (context.footnoteRefCounts.get(id) || 0) + 1;
        context.footnoteRefCounts.set(id, refCount);

        return {
            number: context.footnoteNumbers.get(id),
            refId: `${getFootnoteTargetId(id)}-ref-${refCount}`,
            targetId: getFootnoteTargetId(id)
        };
    }

    function renderInline(source, context) {
        const placeholders = [];

        function hold(html) {
            const key = `%%BLOG_HTML_${placeholders.length}%%`;
            placeholders.push({ key, html });
            return key;
        }

        let text = String(source);

        text = text.replace(/`([^`]+)`/g, function (_, code) {
            return hold(`<code>${escapeHtml(code)}</code>`);
        });

        text = text.replace(/!\[([^\]]*)\]\(([^)\s]+)(?:\s+"([^"]+)")?\)/g, function (_, alt, url) {
            return hold(`<img src="${escapeAttr(sanitizeUrl(url))}" alt="${escapeAttr(alt)}">`);
        });

        text = text.replace(/\[([^\]]+)\]\(([^)\s]+)(?:\s+"([^"]+)")?\)/g, function (_, label, url) {
            const href = sanitizeUrl(url);
            const external = /^https?:\/\//i.test(href);
            const attrs = external ? ' target="_blank" rel="noopener"' : '';
            return hold(`<a href="${escapeAttr(href)}"${attrs}>${renderInline(label, context)}</a>`);
        });

        text = text.replace(/\[\^([^\]]+)\]/g, function (match, id) {
            const footnote = registerFootnote(id.trim(), context);
            if (!footnote) return match;
            return hold(`<sup class="blog-footnote-ref" id="${escapeAttr(footnote.refId)}"><a href="#${escapeAttr(footnote.targetId)}" aria-label="Footnote ${footnote.number}">${footnote.number}</a></sup>`);
        });

        let html = escapeHtml(text)
            .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
            .replace(/\*([^*]+)\*/g, '<em>$1</em>');

        placeholders.forEach(function (item) {
            html = html.replace(item.key, item.html);
        });

        return html;
    }

    function isTableDivider(line) {
        return /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line);
    }

    function splitTableRow(line) {
        return line
            .trim()
            .replace(/^\|/, '')
            .replace(/\|$/, '')
            .split('|')
            .map(function (cell) {
                return cell.trim();
            });
    }

    function renderTable(lines, startIndex, context) {
        const header = splitTableRow(lines[startIndex]);
        const rows = [];
        let index = startIndex + 2;

        while (index < lines.length && /\|/.test(lines[index]) && lines[index].trim()) {
            rows.push(splitTableRow(lines[index]));
            index += 1;
        }

        const headHtml = header.map(function (cell) {
            return `<th>${renderInline(cell, context)}</th>`;
        }).join('');

        const bodyHtml = rows.map(function (row) {
            return `<tr>${row.map(function (cell) {
                return `<td>${renderInline(cell, context)}</td>`;
            }).join('')}</tr>`;
        }).join('');

        return {
            html: `<table><thead><tr>${headHtml}</tr></thead><tbody>${bodyHtml}</tbody></table>`,
            nextIndex: index
        };
    }

    function renderList(lines, startIndex, ordered, context) {
        const tag = ordered ? 'ol' : 'ul';
        const pattern = ordered ? /^\s*\d+\.\s+(.*)$/ : /^\s*[-*+]\s+(.*)$/;
        const items = [];
        let index = startIndex;

        while (index < lines.length) {
            const match = lines[index].match(pattern);
            if (!match) break;
            items.push(`<li>${renderInline(match[1], context)}</li>`);
            index += 1;
        }

        return {
            html: `<${tag}>${items.join('')}</${tag}>`,
            nextIndex: index
        };
    }

    function renderMarkdown(markdown) {
        const extracted = extractFootnotes(markdown);
        const context = createMarkdownContext(extracted.footnotes);
        const lines = extracted.body.replace(/\r\n?/g, '\n').split('\n');
        const html = [];
        const toc = [];
        const usedIds = new Map();
        let index = 0;

        function uniqueId(text) {
            const base = slugify(text);
            const count = usedIds.get(base) || 0;
            usedIds.set(base, count + 1);
            return count ? `${base}-${count + 1}` : base;
        }

        while (index < lines.length) {
            const line = lines[index];

            if (!line.trim()) {
                index += 1;
                continue;
            }

            const fence = line.match(/^```(\w*)\s*$/);
            if (fence) {
                const code = [];
                index += 1;
                while (index < lines.length && !/^```\s*$/.test(lines[index])) {
                    code.push(lines[index]);
                    index += 1;
                }
                index += 1;
                const language = fence[1] ? ` class="language-${escapeAttr(fence[1])}"` : '';
                html.push(`<pre><code${language}>${escapeHtml(code.join('\n'))}</code></pre>`);
                continue;
            }

            const heading = line.match(/^(#{1,6})\s+(.+)$/);
            if (heading) {
                const level = heading[1].length;
                const text = heading[2].replace(/\s+#+$/, '').trim();
                const id = uniqueId(text);
                if (level === 2 || level === 3) {
                    toc.push({ id, level, text });
                }
                html.push(`<h${level} id="${escapeAttr(id)}">${renderInline(text, context)}</h${level}>`);
                index += 1;
                continue;
            }

            if (/^>\s?/.test(line)) {
                const quote = [];
                while (index < lines.length && /^>\s?/.test(lines[index])) {
                    quote.push(lines[index].replace(/^>\s?/, ''));
                    index += 1;
                }
                html.push(`<blockquote>${renderInline(quote.join(' '), context)}</blockquote>`);
                continue;
            }

            if (/^\s*[-*+]\s+/.test(line)) {
                const rendered = renderList(lines, index, false, context);
                html.push(rendered.html);
                index = rendered.nextIndex;
                continue;
            }

            if (/^\s*\d+\.\s+/.test(line)) {
                const rendered = renderList(lines, index, true, context);
                html.push(rendered.html);
                index = rendered.nextIndex;
                continue;
            }

            if (index + 1 < lines.length && /\|/.test(line) && isTableDivider(lines[index + 1])) {
                const rendered = renderTable(lines, index, context);
                html.push(rendered.html);
                index = rendered.nextIndex;
                continue;
            }

            const image = line.match(/^!\[([^\]]*)\]\(([^)\s]+)(?:\s+"([^"]+)")?\)\s*$/);
            if (image) {
                const caption = image[3] ? `<figcaption>${renderInline(image[3], context)}</figcaption>` : '';
                html.push(`<figure class="blog-figure"><img src="${escapeAttr(sanitizeUrl(image[2]))}" alt="${escapeAttr(image[1])}">${caption}</figure>`);
                index += 1;
                continue;
            }

            const paragraph = [line.trim()];
            index += 1;
            while (
                index < lines.length &&
                lines[index].trim() &&
                !/^#{1,6}\s+/.test(lines[index]) &&
                !/^```/.test(lines[index]) &&
                !/^>\s?/.test(lines[index]) &&
                !/^\s*[-*+]\s+/.test(lines[index]) &&
                !/^\s*\d+\.\s+/.test(lines[index])
            ) {
                paragraph.push(lines[index].trim());
                index += 1;
            }
            html.push(`<p>${renderInline(paragraph.join(' '), context)}</p>`);
        }

        return {
            html: html.join('\n'),
            toc,
            footnotes: renderFootnotes(context)
        };
    }

    function renderFootnotes(context) {
        if (!context.footnoteOrder.length) return '';

        return `
          <ol>
            ${context.footnoteOrder.map(function (id) {
                const number = context.footnoteNumbers.get(id);
                const targetId = getFootnoteTargetId(id);
                const note = context.footnotes.get(id);
                return `<li id="${escapeAttr(targetId)}" data-footnote-ref="${escapeAttr(targetId)}-ref-1"><a class="blog-note-number" href="#${escapeAttr(targetId)}-ref-1" aria-label="Back to reference ${number}">${number}</a><span>${renderInline(note)}</span></li>`;
            }).join('')}
          </ol>`;
    }

    function getPostSlug() {
        const params = new URLSearchParams(window.location.search);
        return document.body.dataset.blogPost || params.get('post') || params.get('slug') || DEFAULT_POST;
    }

    function setPageMetadata(meta) {
        const title = meta.title || 'Blog';
        document.title = `${title} - Xin Jin`;
        const description = document.querySelector('meta[name="description"]');
        if (description && (meta.summary || meta.description)) {
            description.setAttribute('content', meta.summary || meta.description);
        }
    }

    function getPostHref(slug) {
        return `/blog/post.html?post=${encodeURIComponent(slug)}`;
    }

    function renderPostNav(navigation) {
        const prev = navigation && navigation.prev;
        const next = navigation && navigation.next;

        return `
      <nav class="blog-bottom-nav" aria-label="Post navigation">
        ${prev ? `<a class="blog-bottom-link" href="${escapeAttr(getPostHref(prev.slug))}">Previous</a>` : '<span class="blog-bottom-link blog-bottom-link-disabled">Previous</span>'}
        <a class="blog-bottom-link blog-bottom-top" href="#top">Back to top</a>
        ${next ? `<a class="blog-bottom-link" href="${escapeAttr(getPostHref(next.slug))}">Next</a>` : '<span class="blog-bottom-link blog-bottom-link-disabled">Next</span>'}
      </nav>`;
    }

    function renderPostShell(meta, rendered, navigation) {
        const author = meta.author || 'Xin Jin';
        const date = meta.date || '';
        const readTime = meta.readTime || meta.readtime || '';
        const summary = meta.summary || meta.description || '';
        const kicker = meta.kicker || 'Notes';
        const hero = meta.hero || '';
        const heroCaption = meta.heroCaption || meta.herocaption || '';
        const heroAlt = meta.heroAlt || meta.heroalt || meta.title || '';

        const tocHtml = rendered.toc.length ? `
        <aside class="blog-toc" aria-label="Table of contents">
          <div class="blog-toc-title">Contents</div>
          <ol>
            ${rendered.toc.map(function (item) {
                const className = item.level === 3 ? ' class="blog-toc-subitem"' : '';
                return `<li${className}><a href="#${escapeAttr(item.id)}">${renderInline(item.text)}</a></li>`;
            }).join('')}
          </ol>
        </aside>` : '<aside class="blog-toc blog-sidebar-empty" aria-hidden="true"></aside>';

        const footnotesHtml = rendered.footnotes ? `
        <aside class="blog-notes" aria-label="Footnotes">
          ${rendered.footnotes}
        </aside>` : '<aside class="blog-notes blog-sidebar-empty" aria-hidden="true"></aside>';

        const metaItems = [date, readTime].filter(Boolean).map(function (item) {
            return `<span>${escapeHtml(item)}</span>`;
        }).join('');

        const heroHtml = hero ? `
      <figure class="blog-hero">
        <img src="${escapeAttr(sanitizeUrl(hero))}" alt="${escapeAttr(heroAlt)}">
        ${heroCaption ? `<figcaption>${renderInline(heroCaption)}</figcaption>` : ''}
      </figure>` : '';

        return `
    <article class="blog-article" id="top">
      <header class="blog-header">
        <div class="blog-kicker">${escapeHtml(kicker)}</div>
        <h1 class="blog-title">${renderInline(meta.title || 'Untitled')}</h1>
        <div class="blog-meta">
          <a href="/">${escapeHtml(author)}</a>
          ${metaItems}
        </div>
        ${summary ? `<p class="blog-summary">${renderInline(summary)}</p>` : ''}
      </header>
      ${heroHtml}
      <div class="blog-layout">
        ${tocHtml}
        <div class="blog-content">
          ${rendered.html}
        </div>
        ${footnotesHtml}
      </div>
      ${renderPostNav(navigation)}
    </article>`;
    }

    function showError(main, message) {
        main.innerHTML = `
    <section class="blog-error">
      <h1>Blog post not found</h1>
      <p>${escapeHtml(message)}</p>
      <p><a href="/blog/">Back to blog index</a></p>
    </section>`;
    }

    async function loadMarkdownPost(slug) {
        const response = await fetch(`${POSTS_DIR}${encodeURIComponent(slug)}.md`);
        if (!response.ok) {
            throw new Error(`Could not load /blog/posts/${slug}.md`);
        }
        return response.text();
    }

    async function renderPost() {
        const main = document.querySelector('[data-blog-main]');
        if (!main) return;

        const slug = getPostSlug();
        try {
            const markdown = await loadMarkdownPost(slug);
            const parsed = parseFrontMatter(markdown);
            const rendered = renderMarkdown(parsed.body);
            const navigation = await getPostNavigation(slug);
            setPageMetadata(parsed.meta);
            main.innerHTML = renderPostShell(parsed.meta, rendered, navigation);
            initRenderedPost();
        } catch (error) {
            showError(main, error.message);
        }
    }

    function normalizePostEntry(entry) {
        if (typeof entry === 'string') {
            return { slug: entry };
        }
        return entry || {};
    }

    async function getManifest() {
        const response = await fetch(POSTS_MANIFEST);
        if (!response.ok) {
            throw new Error('Could not load /blog/posts.json');
        }
        return response.json();
    }

    async function getBlogConfig() {
        try {
            const response = await fetch(BLOG_CONFIG);
            if (!response.ok) return DEFAULT_CONFIG;
            return Object.assign({}, DEFAULT_CONFIG, await response.json());
        } catch (error) {
            return DEFAULT_CONFIG;
        }
    }

    async function getPostSummary(entry) {
        const post = normalizePostEntry(entry);
        if (!post.slug) return null;

        try {
            const markdown = await loadMarkdownPost(post.slug);
            const parsed = parseFrontMatter(markdown);
            return Object.assign({}, parsed.meta, post);
        } catch (error) {
            return Object.assign({ title: post.slug, summary: error.message }, post);
        }
    }

    async function getPostNavigation(currentSlug) {
        try {
            const manifest = await getManifest();
            const posts = manifest.map(normalizePostEntry).filter(function (post) {
                return post.slug;
            });
            const index = posts.findIndex(function (post) {
                return post.slug === currentSlug;
            });
            if (index === -1) return {};
            return {
                prev: posts[index - 1] || null,
                next: posts[index + 1] || null
            };
        } catch (error) {
            return {};
        }
    }

    function renderIndex(posts, config) {
        const rows = posts.map(function (post) {
            const href = getPostHref(post.slug);
            return `
        <a class="blog-post-row" href="${escapeAttr(href)}">
          <span class="blog-post-date">${escapeHtml(post.date || '')}</span>
          <span class="blog-post-body">
            <span class="blog-post-title">${renderInline(post.title || post.slug)}</span>
            ${post.summary ? `<span class="blog-post-summary">${renderInline(post.summary)}</span>` : ''}
          </span>
        </a>`;
        }).join('');

        return `
    <section class="blog-index-header">
      <div class="blog-kicker">Blog</div>
      <h1 class="blog-title">${renderInline(config.title)}</h1>
      <p class="blog-summary">${renderInline(config.summary)}</p>
    </section>
    <section class="blog-post-list" aria-label="Blog posts">
      ${rows || '<p class="blog-empty">No posts yet.</p>'}
    </section>`;
    }

    async function renderBlogIndex() {
        const main = document.querySelector('[data-blog-main]');
        if (!main) return;

        try {
            const [manifest, config] = await Promise.all([getManifest(), getBlogConfig()]);
            const posts = (await Promise.all(manifest.map(getPostSummary))).filter(Boolean);
            main.innerHTML = renderIndex(posts, config);
        } catch (error) {
            showError(main, error.message);
        }
    }

    function initBlog() {
        const view = document.body.dataset.blogView || 'post';
        if (view === 'index') {
            renderBlogIndex();
        } else {
            renderPost();
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initBlog);
    } else {
        initBlog();
    }

    function initRenderedPost() {
        scheduleFootnotePositioning();
        document.querySelectorAll('.blog-content img, .blog-hero img').forEach(function (image) {
            if (!image.complete) {
                image.addEventListener('load', scheduleFootnotePositioning, { once: true });
            }
        });
    }

    let footnotePositionFrame = null;

    function scheduleFootnotePositioning() {
        if (footnotePositionFrame) {
            cancelAnimationFrame(footnotePositionFrame);
        }
        footnotePositionFrame = requestAnimationFrame(positionFootnotes);
    }

    function positionFootnotes() {
        footnotePositionFrame = null;

        const layout = document.querySelector('.blog-layout');
        const notes = document.querySelector('.blog-notes');
        if (!layout || !notes || notes.classList.contains('blog-sidebar-empty')) return;

        const items = Array.from(notes.querySelectorAll('li[data-footnote-ref]'));
        const isStacked = window.matchMedia('(max-width: 880px)').matches;

        notes.style.minHeight = '';
        items.forEach(function (item) {
            item.style.position = '';
            item.style.top = '';
        });

        if (isStacked) return;

        const layoutTop = layout.getBoundingClientRect().top + window.scrollY;
        let previousBottom = 0;

        items.forEach(function (item) {
            const reference = document.getElementById(item.dataset.footnoteRef);
            if (!reference) return;

            item.style.position = 'absolute';
            const referenceTop = reference.getBoundingClientRect().top + window.scrollY - layoutTop;
            const top = Math.max(referenceTop - 2, previousBottom + 12);
            item.style.top = `${top}px`;
            previousBottom = top + item.offsetHeight;
        });

        notes.style.minHeight = `${Math.max(previousBottom, 1)}px`;
    }

    window.addEventListener('resize', scheduleFootnotePositioning);
    window.addEventListener('load', scheduleFootnotePositioning);
}());
