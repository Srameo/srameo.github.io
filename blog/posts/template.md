---
title: Blog Post Title
date: May 17, 2026
author: Xin Jin
kicker: Notes
readTime: 5 min read
summary: Write the short version here: what question the post answers, what changed your mind, and what the reader should remember.
hero: /assets/paper_teaser/s3_dit.png
heroAlt: Replace with a descriptive image alt text
heroCaption: Replace this with a figure, result, diagram, or teaser that anchors the article.
---

## Motivation

Start with the concrete context. A good opening paragraph explains the situation, the surprising part, and why this note is worth writing now.[^opening]

Keep paragraphs short and direct. This template is designed for research notes, experiment logs, paper commentary, or implementation writeups.

## Method

Use ordinary Markdown for prose, [links](https://srameo.github.io/), lists, equations written as text, footnotes,[^footnotes] and figures. Inline code looks like `compute_loss`, while longer snippets sit in a fenced block.

| Approach | Signal | Cost |
| --- | --- | --- |
| Baseline | Weak | Low |
| Improved | Dense | Medium |

## Implementation

Code blocks are intentionally plain and readable, with enough contrast in both light and dark modes.

```python
def train_step(batch, model):
    loss = model(batch).loss
    loss.backward()
    return loss.item()
```

![Replace with an experiment result or supporting visual](/assets/paper_teaser/le3d.gif "Use figures sparingly. When a visual matters, let it span the article column.")

> A useful quote, observation, or design note can live here without becoming a big decorated card.

## Takeaways

- First useful takeaway.
- Second useful takeaway.
- One thing to revisit later.

[^opening]: Footnotes use the standard Markdown shape. Put the reference inline, then define it near the end of the file.
[^footnotes]: Referenced notes render in the right rail on desktop and below the article on mobile.
