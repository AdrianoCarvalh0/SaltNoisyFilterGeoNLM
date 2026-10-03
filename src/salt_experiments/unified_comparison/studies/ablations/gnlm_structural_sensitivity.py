"""Small, reproducible GNLM structural-sensitivity pilot for Set12.

The pilot holds the noise density at ``medium`` (5%) and compares a local
neighbourhood of the final GNLM configuration, ``(f, t, k) = (1, 3, 7)``.
By default it evaluates three fixed Set12 images at both final tolerances
(``tau=0`` and ``tau=4``). It is descriptive: it does not establish a
universal optimum or replace a full sensitivity study.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

# Allow direct execution from the repository root while keeping all ablations
# outside the canonical comparison runner's top-level namespace.
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))

from run import LEVELS, PROTOCOL, calibrate, load_reference, metrics
from lib.geonlm_functions import run_geonlm_pipeline
from lib.noisy_functions import add_near_extreme_impulse_noise


ROOT = PACKAGE_ROOT.parents[2]
DEFAULT_OUTPUT = ROOT / "data/output/studies/ablations/gnlmStructuralSensitivitySet12MediumV1"
DEFAULT_IMAGES = ("01", "06", "11")
DEFAULT_TOLERANCES = (0, 4)

# This intentionally local grid changes one structural dimension at a time
# around the reported configuration. Every value of k is valid at image edges
# for t >= 3 in the current GNLM implementation.
VARIANTS = (
    {"name": "target_f1_t3_k7", "description": "Reported configuration", "f": 1, "t": 3, "k": 7},
    {"name": "lower_k_f1_t3_k3", "description": "Lower graph connectivity", "f": 1, "t": 3, "k": 3},
    {"name": "mid_k_f1_t3_k5", "description": "Intermediate graph connectivity", "f": 1, "t": 3, "k": 5},
    {"name": "wider_search_f1_t5_k7", "description": "Wider search radius", "f": 1, "t": 5, "k": 7},
    {"name": "larger_patch_f2_t3_k7", "description": "Larger patch radius", "f": 2, "t": 3, "k": 7},
)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def digest(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def gamma_for(h_nlm, sigma):
    return 1.40 if h_nlm < 60 or sigma < 10 else 1.55


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="+", default=DEFAULT_IMAGES,
                        help="Set12 image basenames (default: 01 06 11).")
    parser.add_argument("--tolerances", nargs="+", type=int,
                        choices=DEFAULT_TOLERANCES, default=DEFAULT_TOLERANCES)
    parser.add_argument("--variants", nargs="+", choices=[item["name"] for item in VARIANTS],
                        help="Variants to execute (default: all local variants).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="Recompute completed variant records.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned cases without writing files.")
    return parser.parse_args()


def selected_variants(names):
    return list(VARIANTS if not names else [item for item in VARIANTS if item["name"] in names])


def case_path(output, tolerance, image_name):
    return output / "set12" / f"tolerance_{tolerance}" / "medium" / image_name


def load_or_calibrate(reference, noisy, destination, level, force):
    record_path = destination / "nlm.json"
    if record_path.exists() and not force:
        return json.loads(record_path.read_text())

    _, record = calibrate(reference, noisy, destination, level)
    atomic_json(record_path, record)
    return record


def run_case(output, image_name, tolerance, variants, force):
    reference = load_reference("set12", image_name)
    noisy, mask = add_near_extreme_impulse_noise(
        reference,
        salt_prob=LEVELS["medium"] / 2,
        pepper_prob=LEVELS["medium"] / 2,
        impulse_tolerance=tolerance,
        seed=PROTOCOL["seed"],
        return_mask=True,
    )
    noisy = np.asarray(noisy, dtype=np.float32)
    destination = case_path(output, tolerance, image_name)
    destination.mkdir(parents=True, exist_ok=True)
    case = {
        "dataset": "set12",
        "file_name": image_name,
        "level": "medium",
        "density": LEVELS["medium"],
        "tolerance": tolerance,
        "seed": PROTOCOL["seed"],
        "reference_sha256": digest(reference),
        "noisy_sha256": digest(noisy),
        "corruption_sha256": digest(mask),
    }
    atomic_json(destination / "case.json", case)
    nlm = load_or_calibrate(reference, noisy, destination, "medium", force)
    h_nlm, sigma = float(nlm["h"]), float(nlm["sigma"])
    gamma = gamma_for(h_nlm, sigma)
    rows = []

    for variant in variants:
        record_path = destination / f"gnlm_{variant['name']}.json"
        if record_path.exists() and not force:
            rows.append(json.loads(record_path.read_text()))
            continue

        started = time.perf_counter()
        filtered, h_gnlm, _, _, _ = run_geonlm_pipeline(
            reference, h_nlm, noisy,
            f=variant["f"], t=variant["t"], mult=gamma, nn=variant["k"],
        )
        elapsed = time.perf_counter() - started
        row = {
            **case,
            "method": "gnlm",
            "variant": variant["name"],
            "variant_description": variant["description"],
            "f": variant["f"], "t": variant["t"], "k": variant["k"],
            "h_nlm": h_nlm, "sigma": sigma, "gamma": gamma,
            "h_gnlm": float(h_gnlm),
            **metrics(reference, filtered),
            "elapsed_seconds": elapsed,
        }
        atomic_json(record_path, row)
        rows.append(row)
        print(
            f"DONE tau={tolerance} image={image_name} {variant['name']} "
            f"score={row['score']:.6f} time={elapsed:.1f}s",
            flush=True,
        )
    return rows


def write_summaries(output, rows):
    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(output / "results.csv", index=False)
    summary = (
        dataframe.groupby(["tolerance", "variant", "variant_description", "f", "t", "k"], as_index=False)
        .agg(
            images=("file_name", "nunique"),
            mean_psnr=("psnr", "mean"), mean_ssim=("ssim", "mean"),
            mean_score=("score", "mean"), mean_elapsed_seconds=("elapsed_seconds", "mean"),
        )
        .sort_values(["tolerance", "mean_score"], ascending=[True, False])
    )
    summary.to_csv(output / "summary.csv", index=False)
    return summary


def main():
    args = parse_args()
    variants = selected_variants(args.variants)
    if args.dry_run:
        for tolerance in args.tolerances:
            for image_name in args.images:
                for variant in variants:
                    print(f"tau={tolerance} image={image_name} {variant['name']}")
        return

    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "protocol.json", {
        "study": "Set12 medium-density GNLM structural-sensitivity pilot",
        "dataset": "set12", "images": args.images, "density": LEVELS["medium"],
        "tolerances": args.tolerances, "seed": PROTOCOL["seed"],
        "variants": variants,
        "nlm_calibration": PROTOCOL["nlm"],
        "gnlm_smoothing": "h_nlm*(1.40 if h_nlm<60 or sigma<10 else 1.55)",
        "scope": "Descriptive local pilot; not a universal optimum claim.",
    })
    rows = []
    for tolerance in args.tolerances:
        for image_name in args.images:
            rows.extend(run_case(args.output, image_name, tolerance, variants, args.force))
    summary = write_summaries(args.output, rows)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
