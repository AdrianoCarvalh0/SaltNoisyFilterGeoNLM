import pickle
from pathlib import Path

import pandas as pd


INPUTS = [
    (
        "Set12",
        Path("/workspace/data/output/set12Hibrid/summary/results_set12_hibrid_all.pkl"),
    ),
    (
        "Set50",
        Path("/workspace/data/output/set50Hibrid/summary/results_set50_hibrid_all.pkl"),
    ),
]
OUTPUT_DIR = Path("/workspace/data/output/ianlm_vs_ghnlm")
LEVELS = ["low", "moderate", "medium", "high", "extreme"]


def load_pickle(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for dataset, path in INPUTS:
        for row in load_pickle(path):
            records.append(
                {
                    "dataset": dataset,
                    "level": row["level"],
                    "image": row["file_name"],
                    "psnr_ianlm": row["psnr_anlm"],
                    "ssim_ianlm": row["ssim_anlm"],
                    "score_ianlm": row["score_anlm"],
                    "psnr_ghnlm": row["psnr_geonlm_hibrid"],
                    "ssim_ghnlm": row["ssim_geonlm_hibrid"],
                    "score_ghnlm": row["score_geonlm_hibrid"],
                    "delta_psnr_ianlm_minus_ghnlm": row["psnr_anlm"] - row["psnr_geonlm_hibrid"],
                    "delta_ssim_ianlm_minus_ghnlm": row["ssim_anlm"] - row["ssim_geonlm_hibrid"],
                    "delta_score_ianlm_minus_ghnlm": row["score_anlm"] - row["score_geonlm_hibrid"],
                }
            )

    df = pd.DataFrame(records)
    df["_level_order"] = df["level"].map({level: index for index, level in enumerate(LEVELS)})
    df = df.sort_values(["dataset", "_level_order", "image"]).drop(columns=["_level_order"])

    summary = (
        df.groupby(["dataset", "level"], sort=False)
        .agg(
            n_images=("image", "count"),
            mean_psnr_ianlm=("psnr_ianlm", "mean"),
            mean_psnr_ghnlm=("psnr_ghnlm", "mean"),
            mean_delta_psnr_ianlm_minus_ghnlm=("delta_psnr_ianlm_minus_ghnlm", "mean"),
            mean_ssim_ianlm=("ssim_ianlm", "mean"),
            mean_ssim_ghnlm=("ssim_ghnlm", "mean"),
            mean_delta_ssim_ianlm_minus_ghnlm=("delta_ssim_ianlm_minus_ghnlm", "mean"),
            mean_score_ianlm=("score_ianlm", "mean"),
            mean_score_ghnlm=("score_ghnlm", "mean"),
            mean_delta_score_ianlm_minus_ghnlm=("delta_score_ianlm_minus_ghnlm", "mean"),
            wins_ianlm_vs_ghnlm=("delta_score_ianlm_minus_ghnlm", lambda s: int((s > 0).sum())),
            ties_ianlm_vs_ghnlm=("delta_score_ianlm_minus_ghnlm", lambda s: int((s == 0).sum())),
            wins_ghnlm_vs_ianlm=("delta_score_ianlm_minus_ghnlm", lambda s: int((s < 0).sum())),
        )
        .reset_index()
    )
    overall = (
        df.groupby("dataset", sort=False)
        .agg(
            n_images=("image", "count"),
            mean_psnr_ianlm=("psnr_ianlm", "mean"),
            mean_psnr_ghnlm=("psnr_ghnlm", "mean"),
            mean_delta_psnr_ianlm_minus_ghnlm=("delta_psnr_ianlm_minus_ghnlm", "mean"),
            mean_ssim_ianlm=("ssim_ianlm", "mean"),
            mean_ssim_ghnlm=("ssim_ghnlm", "mean"),
            mean_delta_ssim_ianlm_minus_ghnlm=("delta_ssim_ianlm_minus_ghnlm", "mean"),
            mean_score_ianlm=("score_ianlm", "mean"),
            mean_score_ghnlm=("score_ghnlm", "mean"),
            mean_delta_score_ianlm_minus_ghnlm=("delta_score_ianlm_minus_ghnlm", "mean"),
            wins_ianlm_vs_ghnlm=("delta_score_ianlm_minus_ghnlm", lambda s: int((s > 0).sum())),
            ties_ianlm_vs_ghnlm=("delta_score_ianlm_minus_ghnlm", lambda s: int((s == 0).sum())),
            wins_ghnlm_vs_ianlm=("delta_score_ianlm_minus_ghnlm", lambda s: int((s < 0).sum())),
        )
        .reset_index()
    )

    pkl_path = OUTPUT_DIR / "ianlm_vs_ghnlm_comparison.pkl"
    with pkl_path.open("wb") as handle:
        pickle.dump(
            {
                "records": df.to_dict("records"),
                "summary_by_level": summary.to_dict("records"),
                "summary_by_dataset": overall.to_dict("records"),
            },
            handle,
        )

    xlsx_path = OUTPUT_DIR / "ianlm_vs_ghnlm_comparison.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="all_images", index=False)
        summary.to_excel(writer, sheet_name="summary_by_level", index=False)
        overall.to_excel(writer, sheet_name="summary_by_dataset", index=False)
        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            for column_cells in ws.columns:
                max_len = max(len(str(cell.value or "")) for cell in column_cells)
                ws.column_dimensions[column_cells[0].column_letter].width = min(
                    max(max_len + 2, 12),
                    34,
                )
            for row in ws.iter_rows(min_row=2):
                for cell in row:
                    if isinstance(cell.value, float):
                        cell.number_format = "0.000000"

    print(f"Pickle saved to: {pkl_path}")
    print(f"XLSX saved to: {xlsx_path}")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
