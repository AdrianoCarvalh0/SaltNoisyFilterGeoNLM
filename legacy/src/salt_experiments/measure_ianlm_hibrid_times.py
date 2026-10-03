import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from functions.anlm_functions import run_anlm_pipeline


LEVELS = ["low", "medium", "moderate", "high", "extreme"]
RESOLUTION = "full_512"
OUTPUT_DIR = Path("/workspace/data/output/ianlm_timing")

DATASETS = [
    {
        "dataset": "set12Hibrid",
        "base": Path("/workspace/data/output/set12Hibrid"),
    },
    {
        "dataset": "set50Hibrid",
        "base": Path("/workspace/data/output/set50Hibrid"),
    },
]

IANLM_CONFIG = {
    "f": 1,
    "t": 3,
    "h_multiplier": 0.001,
    "switch_impulse_only": True,
    "reject_impulse_candidates": True,
    "use_aswmf_spatial_weights": True,
    "aswmf_weight_diag_1": 1.0,
    "aswmf_weight_diag_2": 1.0,
    "aswmf_weight_other": 10.0,
}


def natural_key(value):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


def load_pickle(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def save_pickle(path, value):
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def warmup_ianlm():
    dummy = np.zeros((16, 16), dtype=np.float32)
    dummy[8, 8] = 255.0
    run_anlm_pipeline(
        img_original=dummy,
        h_base=100.0,
        img_noisy=dummy,
        f=IANLM_CONFIG["f"],
        t=IANLM_CONFIG["t"],
        mult=IANLM_CONFIG["h_multiplier"],
        switch_impulse_only=IANLM_CONFIG["switch_impulse_only"],
        reject_impulse_candidates=IANLM_CONFIG["reject_impulse_candidates"],
        use_aswmf_spatial_weights=IANLM_CONFIG["use_aswmf_spatial_weights"],
        aswmf_weight_diag_1=IANLM_CONFIG["aswmf_weight_diag_1"],
        aswmf_weight_diag_2=IANLM_CONFIG["aswmf_weight_diag_2"],
        aswmf_weight_other=IANLM_CONFIG["aswmf_weight_other"],
    )


def measure_item(dataset_name, level, index, item):
    start = time.perf_counter()
    _, h_used, psnr, ssim, method_score = run_anlm_pipeline(
        img_original=item["img_reference_np"],
        h_base=float(item["nlm_h"]),
        img_noisy=item["img_noisy_salt_pepper_np"],
        f=IANLM_CONFIG["f"],
        t=IANLM_CONFIG["t"],
        mult=IANLM_CONFIG["h_multiplier"],
        switch_impulse_only=IANLM_CONFIG["switch_impulse_only"],
        reject_impulse_candidates=IANLM_CONFIG["reject_impulse_candidates"],
        use_aswmf_spatial_weights=IANLM_CONFIG["use_aswmf_spatial_weights"],
        aswmf_weight_diag_1=IANLM_CONFIG["aswmf_weight_diag_1"],
        aswmf_weight_diag_2=IANLM_CONFIG["aswmf_weight_diag_2"],
        aswmf_weight_other=IANLM_CONFIG["aswmf_weight_other"],
    )
    elapsed = time.perf_counter() - start
    return {
        "dataset": dataset_name,
        "level": level,
        "order": index,
        "image": item["file_name"],
        "time_ianlm_seconds": elapsed,
        "time_ianlm_ms": elapsed * 1000.0,
        "h_ianlm": h_used,
        "f": IANLM_CONFIG["f"],
        "t": IANLM_CONFIG["t"],
        "h_multiplier": IANLM_CONFIG["h_multiplier"],
        "psnr_ianlm": psnr,
        "ssim_ianlm": ssim,
        "score_ianlm": method_score,
    }


def measure_dataset(dataset):
    records = []
    for level in LEVELS:
        path = (
            dataset["base"]
            / f"salt_pepper_{level}"
            / RESOLUTION
            / "results"
            / f"array_nlm_salt_pepper_{level}_filtereds.pkl"
        )
        items = sorted(load_pickle(path), key=lambda row: natural_key(row["file_name"]))
        for index, item in enumerate(items, start=1):
            record = measure_item(dataset["dataset"], level, index, item)
            records.append(record)
            print(
                f"{record['dataset']} {level} {record['image']}: "
                f"{record['time_ianlm_seconds']:.6f}s"
            )
    return records


def write_xlsx(records, path):
    df = pd.DataFrame(records)
    summary = (
        df.groupby(["dataset", "level"], sort=False)
        .agg(
            images=("image", "count"),
            mean_time_ianlm_seconds=("time_ianlm_seconds", "mean"),
            median_time_ianlm_seconds=("time_ianlm_seconds", "median"),
            min_time_ianlm_seconds=("time_ianlm_seconds", "min"),
            max_time_ianlm_seconds=("time_ianlm_seconds", "max"),
            total_time_ianlm_seconds=("time_ianlm_seconds", "sum"),
        )
        .reset_index()
    )
    overall = (
        df.groupby("dataset", sort=False)
        .agg(
            images=("image", "count"),
            mean_time_ianlm_seconds=("time_ianlm_seconds", "mean"),
            median_time_ianlm_seconds=("time_ianlm_seconds", "median"),
            total_time_ianlm_seconds=("time_ianlm_seconds", "sum"),
        )
        .reset_index()
    )

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="times_all", index=False)
        summary.to_excel(writer, sheet_name="summary_by_level", index=False)
        overall.to_excel(writer, sheet_name="summary_by_dataset", index=False)
        for dataset_name, dataset_df in df.groupby("dataset", sort=False):
            dataset_df.to_excel(writer, sheet_name=dataset_name[:31], index=False)

        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            for column_cells in ws.columns:
                max_len = max(len(str(cell.value or "")) for cell in column_cells)
                ws.column_dimensions[column_cells[0].column_letter].width = min(
                    max(max_len + 2, 12),
                    28,
                )
            for row in ws.iter_rows(min_row=2):
                for cell in row:
                    if isinstance(cell.value, float):
                        cell.number_format = "0.000000"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    warmup_ianlm()

    records = []
    for dataset in DATASETS:
        records.extend(measure_dataset(dataset))

    pickle_path = OUTPUT_DIR / "ianlm_times_set12_set50_hibrid.pkl"
    xlsx_path = OUTPUT_DIR / "ianlm_times_set12_set50_hibrid.xlsx"
    save_pickle(pickle_path, records)
    write_xlsx(records, xlsx_path)
    print(f"Pickle saved to: {pickle_path}")
    print(f"XLSX saved to: {xlsx_path}")


if __name__ == "__main__":
    main()
