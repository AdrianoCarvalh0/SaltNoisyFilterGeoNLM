from pathlib import Path
import pickle
import time

import cupy as cp
import numpy as np
import pandas as pd

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from functions.nlm_functions import NLM_fast_cuda_global


OUTPUT_DIR = Path("/workspace/data/output/hibrid_runtime_tables")
LEVELS = ["low", "moderate", "medium", "high", "extreme"]
DATASETS = [("set12", "Set12"), ("set50", "Set50")]
F = 4
T = 7


def load_nlm_vector(dataset_name, level):
    path = (
        Path(f"/workspace/data/output/{dataset_name}Hibrid")
        / f"salt_pepper_{level}"
        / "full_512"
        / "results"
        / f"array_nlm_salt_pepper_{level}_filtereds.pkl"
    )
    with path.open("rb") as file:
        return pickle.load(file)


def existing_details(path):
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".pkl":
        with path.open("rb") as file:
            return pd.DataFrame(pickle.load(file))
    return pd.read_excel(path)


def measure_one(noisy, h):
    image_gpu = cp.asarray(noisy.astype(np.float32))
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    output = NLM_fast_cuda_global(image_gpu, h=float(h), f=F, t=T)
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start
    del output
    del image_gpu
    cp.get_default_memory_pool().free_all_blocks()
    return elapsed


def save_details(records):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pkl_path = OUTPUT_DIR / "nlm_cuda_runtime_details.pkl"
    xlsx_path = OUTPUT_DIR / "nlm_cuda_runtime_details.xlsx"
    with pkl_path.open("wb") as file:
        pickle.dump(records, file, protocol=pickle.HIGHEST_PROTOCOL)
    pd.DataFrame(records).to_excel(xlsx_path, index=False)
    return pkl_path, xlsx_path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pkl_path = OUTPUT_DIR / "nlm_cuda_runtime_details.pkl"
    current = existing_details(pkl_path)
    records = current.to_dict("records") if not current.empty else []
    seen = {
        (row["dataset"], row["level"], row["file_name"])
        for row in records
    }

    warm = load_nlm_vector("set12", "low")[0]
    measure_one(warm["img_noisy_salt_pepper_np"][:64, :64], h=float(warm["nlm_h"]))

    total = sum(len(load_nlm_vector(dataset_name, level)) for dataset_name, _ in DATASETS for level in LEVELS)
    done = len(seen)

    for dataset_name, dataset_label in DATASETS:
        for level in LEVELS:
            vector = load_nlm_vector(dataset_name, level)
            for item in vector:
                key = (dataset_label, level, item["file_name"])
                if key in seen:
                    continue
                elapsed = measure_one(
                    item["img_noisy_salt_pepper_np"],
                    h=float(item["nlm_h"]),
                )
                record = {
                    "dataset": dataset_label,
                    "level": level,
                    "file_name": item["file_name"],
                    "nlm_h": float(item["nlm_h"]),
                    "nlm_f": F,
                    "nlm_t": T,
                    "runtime_s": elapsed,
                }
                records.append(record)
                seen.add(key)
                done += 1
                print(
                    f"{done}/{total} {dataset_label} {level} {item['file_name']}: "
                    f"NLM CUDA {elapsed:.4f}s",
                    flush=True,
                )
            save_details(records)

    df = pd.DataFrame(records)
    summary = (
        df.groupby(["dataset", "level"])
        .agg(
            mean_runtime_s=("runtime_s", "mean"),
            n=("runtime_s", "count"),
        )
        .reset_index()
    )
    summary.to_excel(OUTPUT_DIR / "nlm_cuda_runtime_summary.xlsx", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
