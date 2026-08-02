# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project shape

This is a small, two-process Python service — not a framework project.
Everything lives in two files at the repo root:

- `app.py` — FastAPI web app. Defines config loading (`AppConfig`,
  `FeedConfig`, `load_config`), the SQLite cache helpers, IOC/marker
  protection used around translation, RSS rendering (`build_translated_feed_xml`),
  and the HTTP routes. It never talks to a translation API directly.
- `worker.py` — polling loop that does the actual work: fetch source feeds,
  optionally pull full article content, call the translation provider, write
  to the SQLite cache, and render feed XML. Imports shared helpers from `app.py`.

Keep this split: `app.py` only serves what's cached; `worker.py` is the only
place that fetches/translates.

## Dependencies

Follow the "less is safer" posture in the user's global CLAUDE.md:

- Don't add a new dependency for something that's a few lines of stdlib code.
- Current deps are deliberately few: `fastapi`, `uvicorn`, `PyYAML`, `httpx`,
  `feedparser`, `feedgen`. Any addition should be justified.
- Pin exact versions; `uv.lock` is the source of truth for reproducible
  builds — regenerate it with `just lock` / `just lock-update`, never hand-edit.
- No postinstall/lifecycle scripts in added packages.

## Config changes

- `config.yaml` (per-deployment, gitignored) is the runtime source of truth;
  `config.yaml.example` must be kept in sync with any new fields added to
  `FeedConfig` or `AppConfig` in `app.py`, with comments explaining defaults.
- Same for `.env` / `.env.example` for secrets and endpoints.
- If a change alters how feed XML is rendered (not just translation logic),
  bump `cache.render_version` (or remind the user to run
  `just bump-render-version`) — the feed cache is keyed on it, so stale XML
  won't otherwise be rebuilt.

## Translation-sensitive code

- IOC preservation (`protect_iocs`/`restore_iocs`) and marker protection
  (`protect_markers`/`restore_markers`, `protect_breaks`/`restore_breaks`) in
  `app.py` exist to stop translation engines from mangling URLs, hashes,
  CVE IDs, line breaks, and embedded image markers. If you touch text
  processing in the translate path, verify these round-trip correctly —
  a broken marker regex silently corrupts feed content rather than erroring.
- `worker.py` chunks text before sending to translation providers
  (`_chunk_text` for LibreTranslate, `_chunk_text_bytes` for DeepL's 128KiB
  request limit). Preserve chunk-then-rejoin behavior if you change
  provider calls.
- DeepL quota exhaustion (`_DeepLQuotaExceeded`) triggers an automatic
  fallback to LibreTranslate in `translate_sync` — don't remove this without
  discussing it, it's a deliberate resilience feature.

## Testing changes

There is no test suite. Verify changes by running the app and worker
locally or via `docker compose up`, and checking `/feeds/<id>.xml` output
and worker logs. Run `ruff check .` before considering Python changes done.

## Security notes

- `/images/{name}` validates the filename against a strict
  `[A-Fa-f0-9]{64}\.[A-Za-z0-9]+` pattern and resolves the path within
  `CFG.image_dir` before serving — preserve this if touching that route,
  it's the only user-controlled path in the app.
- Feed content is fetched from external, sometimes untrusted sources. Don't
  add functionality that executes or evaluates fetched content — treat it as
  data only (text/HTML extraction, never `eval`/template execution).
