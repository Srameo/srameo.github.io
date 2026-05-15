# Repository Guidelines

## Project Structure & Module Organization

This repository is a static personal homepage. The root `index.html` is the main page, with shared styles in `css/` and shared browser scripts in `js/`. Static images, PDFs, videos, and paper teasers live in `assets/`. Publication BibTeX snippets are stored as HTML files in `bibtex/`. Project pages live under `projects/`, often as self-contained subdirectories with their own `index.html` and `assets/`. `my-soarxiv/` contains a small Python-generated paper galaxy; edit `my-soarxiv/papers.json` and regenerate its output when papers change. `twentytwenty/` is a vendored comparison-slider demo/plugin area; avoid broad refactors there unless specifically required.

## Build, Test, and Development Commands

- `python3 -m http.server 8080`: serve the repository root locally at `http://localhost:8080`.
- `npm run fetch-stars`: update GitHub star counts using `scripts/fetch-stars.js`.
- `npm test`: currently a placeholder and exits with an error; do not rely on it as validation.
- `UV_CACHE_DIR=.uv-cache uv venv .venv`: create the local Python environment used by `my-soarxiv`.
- `UV_CACHE_DIR=.uv-cache uv pip install -r my-soarxiv/requirements.txt`: install galaxy build dependencies.
- `HF_HOME=.hf-cache UV_CACHE_DIR=.uv-cache .venv/bin/python my-soarxiv/build.py`: regenerate `my-soarxiv/galaxy.json`.

## Coding Style & Naming Conventions

Use 2-space indentation in HTML, CSS, and JavaScript where nearby files do. Keep changes minimal and consistent with existing static-page patterns. Prefer semantic HTML and simple vanilla JavaScript over new frameworks. Use lowercase, hyphenated names for new project folders and assets, for example `projects/new-paper/` or `assets/paper_teaser/new-paper.jpg`. Keep generated or heavyweight assets out of source edits unless the page needs them.

## Testing Guidelines

There is no formal test suite. Validate static changes by serving locally and checking the affected pages in a browser. For data updates, confirm JSON is valid with commands such as `python3 -m json.tool projects/galaxy.json >/dev/null`. For `my-soarxiv`, rerun `build.py` and verify it completes without errors.

## Commit & Pull Request Guidelines

Recent history primarily uses concise imperative messages, with automated updates following `chore: update GitHub star counts`. Use short, scoped commits such as `update project teaser`, `fix navbar link`, or `chore: update GitHub star counts`. Pull requests should describe the visible change, list touched pages, mention generated assets or data files, and include screenshots for layout or visual updates.

## Agent-Specific Instructions

Do not overwrite unrelated local changes. Check `git status --short` before editing and keep patches focused. Do not commit, branch, or deploy unless explicitly asked. Keep local build caches such as `.venv/`, `.uv-cache/`, and `.hf-cache/` untracked.
