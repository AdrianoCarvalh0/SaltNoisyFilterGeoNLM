# Ablation studies

This directory contains targeted sensitivity and ablation runners that extend
the canonical experiment. They do not replace `../../run.py` and must state
their scope explicitly in their output protocol.

Generated results are stored below `data/output/studies/ablations/`. To keep
the repository compact, a study stores its protocol, hashes, calibration curves,
per-configuration metrics, and CSV summaries, but not generated image or array
outputs. The final comparison archive in `data/output/unifiedComparisonFinal/`
is separate and must be retained unchanged.
