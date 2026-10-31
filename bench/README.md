# Benchmarks

Private, unpublished workspace packages. Both suites write JSON reports under
`bench/report/`, which is git-ignored.

- `benchmark/` — cold start, time-to-first-audio (p50/p95), real-time factor,
  peak RAM, cancellation latency, and download size per dtype (M5).
- `quality/` — fixed corpus, chunk-consistency check (streamed vs. full
  utterance), and listening-test outcomes (M5).

No threshold gating happens here. Numbers published in the README and docs must
trace back to a committed report; see [docs/ROADMAP.md](../docs/ROADMAP.md) §M5.
