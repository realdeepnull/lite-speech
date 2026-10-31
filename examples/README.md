# Examples

Private, unpublished workspace packages. They are never published (`private:
true`, no `files` field) and consume the packages the way a user would.

- `demo/` — bundler-free browser demo against the built ESM output (M3).

No example defines `build`/`test`/`size`, so the root `--if-present` scripts skip
it until it does.
