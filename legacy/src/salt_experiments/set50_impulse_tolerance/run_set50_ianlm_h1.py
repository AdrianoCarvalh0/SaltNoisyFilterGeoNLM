"""Run the Set50 tolerance sweep with a fixed, Set12-selected IANLM h=1."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from set12_impulse_tolerance import run_set12_impulse_tolerance as experiment


experiment.SOURCE_BASE = Path("/workspace/data/output/set50")
experiment.OUTPUT_BASE = Path("/workspace/data/output/set50ImpulseToleranceH1")
experiment.DATASET_NAME = "set50"
experiment.RUN_GHNLM = False
experiment.IANLM_FIXED_H = 1.0
experiment.REUSE_SOURCE_NLM_H = True


if __name__ == "__main__":
    experiment.run_experiment()
