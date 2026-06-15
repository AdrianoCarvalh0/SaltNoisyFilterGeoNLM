import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from functions.aswmf_nlm_h001_experiment import run_experiment


CONFIG = {
    "dataset_name": "set50",
    "dataset_label": "Set50",
    "env_prefix": "SET50_ASWMF_NLM_H001",
    "source_base": Path("/workspace/data/output/set50"),
    "fallback_source_base": Path("/workspace/data/output/set50Hibrid"),
    "hybrid_results_base": Path("/workspace/data/output/set50Hibrid"),
    "output_base": Path("/workspace/data/output/set50Hibrid/aswmf_nlm_h001"),
}


if __name__ == "__main__":
    run_experiment(CONFIG)
