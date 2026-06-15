# InferenceKey SDK — documentation site

The docs-as-code project for **docs.inferencekey.com**. An [Astro
Starlight](https://starlight.astro.build/) static site: cacheable HTML, no SSR,
built-in search (Pagefind), dark mode, copy buttons, SEO, and i18n (EN source +
ES/FR, untranslated pages fall back to EN).

## Develop

```bash
npm install
npm run dev          # local preview at http://localhost:4321
npm run docs:build   # static output → dist/
npm run preview      # serve the built dist/
npm run linkcheck    # lychee over dist/ (run after docs:build)
```

## Content

- `src/content/docs/en/` — English (source of truth).
- `src/content/docs/es/`, `src/content/docs/fr/` — translations; any page not
  present here falls back to its English version automatically.
- Page structure (sidebar) is defined in `astro.config.mjs`.

Examples use the **real SDK API** (Python `inferencekey`, Node
`@inferencekey/sdk`). The per-language API reference is auto-generated in CI
(sphinx/autodoc for Python, typedoc for TypeScript) and slotted under `api/`;
Go and Java appear as "Coming soon" until their C-ABI bindings land.

## Deploy (Docker → registry → Swarm)

`Dockerfile` builds the static site and serves it with nginx. The Swarm compose
file lives in the **monorepo** at `infrastructure/docker-compose-docs.yml`
(mirrors the landing's pipeline), so deployment reuses the existing
Jenkins → private registry → Swarm flow and never blocks an SDK release.
