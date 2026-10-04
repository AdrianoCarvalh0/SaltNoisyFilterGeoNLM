# Auxiliary sensitivity and ablation studies

This is the single index for exploratory and configuration-selection studies.
They complement, but never replace, the canonical comparison in `../../run.py`.
The official result archive is separate from these studies.

| Study | Scope and purpose | Runner | Results |
|---|---|---|---|
| GNLM structural pilot | Three Set12 images, medium density, `tau={0,4}`; local variants around `(f,t,k)=(1,3,7)`. Descriptive only. | `gnlm_structural_sensitivity.py` | `data/output/studies/ablations/gnlmStructuralSensitivitySet12MediumV1/` |
| NLMedians structural sensitivity | All 11 Set12 images, medium density, `tau={0,4}`; full grid `f={1,2,3}`, `t={2,3,4}`. Set50 is excluded from configuration selection. | `nlmedians_structural_sensitivity.py` | `data/output/studies/ablations/nlmediansStructuralSensitivitySet12MediumV1/` |
| NLMedians pilot | Earlier three-image pilot retained for provenance; superseded for selection by the full Set12 study. | `nlmedians_structural_sensitivity.py` (historical run) | `data/output/studies/ablations/nlmediansStructuralSensitivityPilotSet12MediumV1/` |

Each result directory contains a `protocol.json`, complete calibration curves,
per-configuration `results.csv`, and `summary.csv`, but no restored images or
arrays. This keeps studies reproducible without duplicating the final archive.

The NLMedians full study selects `(f,t)=(2,2)` as the fixed V4 baseline by the
PSNR/runtime compromise: `(2,2)` dominates the previous `(2,3)` setting in the
tested Set12-medium conditions. The alternative `(3,2)` has higher mean SSIM
and auxiliary score, so the study does not claim a universal optimum.

Run a study from the repository root, for example:

```bash
python src/salt_experiments/unified_comparison/studies/ablations/nlmedians_structural_sensitivity.py
```
