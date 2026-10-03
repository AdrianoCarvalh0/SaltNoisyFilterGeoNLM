import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from functions.aswmf_nlm_h001_experiment import run_experiment


CONFIG = {
    "dataset_name": "set12",
    "dataset_label": "Set12",
    "env_prefix": "SET12_ASWMF_NLM_H001",
    "source_base": Path("/workspace/data/output/set12"),
    "fallback_source_base": Path("/workspace/data/output/set12Hibrid"),
    "hybrid_results_base": Path("/workspace/data/output/set12Hibrid"),
    "output_base": Path("/workspace/data/output/set12Hibrid/aswmf_nlm_h001"),
}


if __name__ == "__main__":
    run_experiment(CONFIG)
