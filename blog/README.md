# Markdown Blog

Write posts as Markdown files in `blog/posts/`.

## Blog Index

Edit `/blog/` title and summary in `blog/config.json`.

The post list comes from `blog/posts.json`. Each listed slug loads the matching Markdown file from `blog/posts/`.

## New Post

1. Copy `blog/posts/template.md` to a new lowercase slug, for example `blog/posts/my-note.md`.
2. Edit the front matter at the top of the file:

```md
---
title: My Note
date: May 17, 2026
author: Xin Jin
kicker: Notes
readTime: 5 min read
summary: One sentence summary for the blog index and post header.
hero: /assets/paper_teaser/s3_dit.png
heroAlt: Short description of the hero image
heroCaption: Optional caption shown under the hero image.
---
```

3. Add the slug to `blog/posts.json`:

```json
[
  "my-note",
  "template"
]
```

4. Preview it at `/blog/post.html?post=my-note`.

Supported Markdown includes headings, paragraphs, links, inline code, fenced code blocks, blockquotes, unordered/ordered lists, images, simple tables, and footnotes.

Footnotes use standard Markdown syntax:

```md
This sentence has a note.[^note]

[^note]: Footnotes render in the right rail on desktop and below the post on mobile.
```

The post page automatically adds bottom navigation:

- previous post from `blog/posts.json`
- back to top
- next post from `blog/posts.json`
