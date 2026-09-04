# Chronatello

Chronatello converts the public DATA 385 Fall 2026 schedule into an automatically updated iCalendar subscription.

Subscribe at:

```text
https://lomzem.github.io/chronatello/calendar.ics
```

## How it works

1. A synchronous Rust program fetches the semantic Quarto schedule table.
2. It reads all `Assignments` items and explicit deadline phrases from `In Class`.
3. New or changed items go to `gemini-3.5-flash-lite` for a clear title and body.
4. Rust validates the source date, weekday, time, classification, and link IDs.
5. Rust generates a complete RFC 5545 calendar and public state file.
6. GitHub Actions publishes the files through GitHub Pages.

All-day deadlines are the default. Explicit source times use `America/Los_Angeles`; “end of class” means 1:45 PM. Each event keeps the professor's original text and all source links. Stable source identities preserve event UIDs when a deadline changes.

The LLM cannot set UIDs, revision numbers, arbitrary URLs, or unsupported dates. A failed fetch, invalid response, or unsafe change stops publication and leaves the last Pages deployment available.

## Local checks

The normal test suite does not use the network or Gemini:

```sh
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --locked --all-features
```

For a live run, create an ignored `.env` file containing `GEMINI_API_KEY`. Load it without putting the value in command history:

```sh
set -a
. ./.env
set +a
cargo run --locked --release
```

The live run writes `public/index.html`, `public/calendar.ics`, and `public/_state.json`. The `public` directory is generated and ignored by Git.

## Manual identity overrides

`overrides.json` maps exact normalized professor text to an identity. Use it only when an unlinked assignment is renamed and moved, so the source no longer provides enough evidence to match it automatically.

For a new identity, use a readable value:

```json
{
  "Submit the final analysis by Mon 8/31": "final-analysis"
}
```

To connect renamed text to an event that was already published, copy its `source_key` from the public `_state.json` file:

```json
{
  "Submit the final analysis by Mon 8/31": "source_key:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
}
```

## Automation

- `.github/workflows/ci.yml` checks formatting, Clippy, and tests on pushes and pull requests.
- `.github/workflows/pages.yml` runs daily and supports manual dispatch.
- The Pages workflow reads `GEMINI_API_KEY` from the repository Actions secret.
- Generated state is public and contains no API key.

GitHub and calendar clients control refresh timing. Updates are eventual, not immediate.

## License

MIT
