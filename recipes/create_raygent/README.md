# Create-Raygent Preset Recipes

These recipes show copyable `create_raygent(...)` construction patterns. They
use a no-network fake provider so they can run in development without API keys.

Run from the repository root with `python -m`, for example:

```bash
uv run python -m recipes.create_raygent.minimal
uv run python -m recipes.create_raygent.project_reader
uv run python -m recipes.create_raygent.full_developer
```

The recipes intentionally stay outside `src/`. Runtime library behavior belongs
under `src/raygent_harness/`; this folder is for product-facing construction
examples.
