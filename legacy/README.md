# Legacy material

This directory preserves earlier scripts, notebooks, exploratory studies, plots,
and their outputs. It is retained for provenance and must not be used to
reproduce or substantiate the article's final comparison.

The only canonical implementation is
`src/salt_experiments/unified_comparison/`. The article's final numerical archive
is `data/output/unifiedComparisonFinal/`; a new reproducible run writes to
`data/output/unifiedComparison/` by default.

Contents are organized by their former top-level location:

- `ablations/`: historical Set12 GNLM and Set50 hybrid ablation scripts and
  outputs, grouped separately from the canonical implementation;
- `src/`: earlier experiment runners and helper implementations;
- `data/output/`: historical results, plots, and intermediate artifacts;
- `docs/`: superseded analyses and notes;
- `logs/`: historical run logs.
