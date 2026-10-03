from pathlib import Path
import pickle
import sys

PROJECT_ROOT = Path("/workspace")
sys.path.append(str(PROJECT_ROOT / "src" / "salt_experiments"))

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import networkx as nx
import numpy as np
from scipy.ndimage import generic_filter
import sklearn.neighbors as sknn
from sklearn.decomposition import PCA

from functions.geonlm_medians_functions import (
    _aswmf_local_fallback,
    _aswmf_spatial_weight,
    _robust_center_value,
)
from functions.nlm_functions import mirror_cpu


PICKLE_PATH = (
    PROJECT_ROOT
    / "data/output/set12Hibrid/salt_pepper_medium/full_512/results/array_nlm_salt_pepper_medium_filtereds.pkl"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/output/set12Hibrid/salt_pepper_medium/full_512/results/path_graphs_07_medium"
)
FILE_NAME = "07.png"

F = 1
T = 3
NN_COUNT = 7
H_MULTIPLIER = 0.001

Z_ALPHA = 1.96
OUTLIER_PIXEL_ALPHA = 0.0
REJECT_IMPULSE_CANDIDATES = True
USE_ASWMF_SPATIAL_WEIGHTS = True
ASWMF_WEIGHT_DIAG_1 = 1.0
ASWMF_WEIGHT_DIAG_2 = 1.0
ASWMF_WEIGHT_OTHER = 10.0

SELECTION_MARGIN = 55
MIN_CENTER_DISTANCE = 58
VARIANCE_WINDOW = 31
PATCH_CONTEXT_RADIUS = 18

REGION_TARGETS = [
    ("smooth_1", "Smooth", 0.03),
    ("smooth_2", "Smooth", 0.08),
    ("mixed_1", "Mixed", 0.45),
    ("mixed_2", "Mixed", 0.55),
    ("dense_1", "Dense", 0.93),
    ("dense_2", "Dense", 0.98),
]

COLORS = {
    "Smooth": "#2ca02c",
    "Mixed": "#ff7f0e",
    "Dense": "#d62728",
}


def load_item(pickle_path=PICKLE_PATH, file_name=FILE_NAME):
    with pickle_path.open("rb") as fobj:
        records = pickle.load(fobj)
    item = next(row for row in records if row["file_name"] == file_name)
    noisy = item["img_noisy_salt_pepper_np"].astype(np.float32)
    ref = item["img_reference_np"].astype(np.float32)
    nlm_h = float(item["nlm_h"])
    return item, noisy, ref, nlm_h


def Extract_patches_local(img, i, j, f, t):
    """Notebook-style local patch extraction around row i, column j."""
    m, n = img.shape
    patch_size = (2 * f + 1) * (2 * f + 1)
    img_n = mirror_cpu(img.astype(np.float32), f)

    im = i + f
    jn = j + f
    patch_central = img_n[im - f:im + f + 1, jn - f:jn + f + 1].copy()
    central = patch_central.reshape((1, patch_central.size))[-1]

    rmin = max(im - t, f)
    rmax = min(im + t, m + f)
    smin = max(jn - t, f)
    smax = min(jn + t, n + f)

    num_elem = (rmax - rmin + 1) * (smax - smin + 1)
    dataset = np.zeros((num_elem, patch_size), dtype=np.float32)
    coords = []
    source = -1
    k = 0

    for r in range(rmin, rmax + 1):
        for s in range(smin, smax + 1):
            patch = img_n[r - f:r + f + 1, s - f:s + f + 1].copy()
            neighbor = patch.reshape((1, patch.size))[-1]
            dataset[k, :] = neighbor.copy()
            coords.append((r - f, s - f))
            if np.array_equal(central, neighbor):
                source = k
            k += 1

    if source < 0:
        source = 0
    return dataset, source, coords


def build_KNN_Graph(data, k):
    n_neighbors = min(k, max(1, len(data) - 1))
    knn_graph = sknn.kneighbors_graph(data, n_neighbors=n_neighbors, mode="distance")
    return nx.from_scipy_sparse_array(knn_graph)


def compute_patchspace_metrics(patches):
    unique_rows = np.unique(patches, axis=0).shape[0]
    energy = np.mean(np.var(patches, axis=1))
    _, singular_values, _ = np.linalg.svd(patches, full_matrices=False)
    rank = np.sum(singular_values > 1e-6)
    spectral_ratio = singular_values[0] / (np.sum(singular_values) + 1e-12)
    return {
        "unique_rows": int(unique_rows),
        "rank": int(rank),
        "energy": float(energy),
        "spectral_ratio": float(spectral_ratio),
    }


def compute_graph_metrics(graph, embedding, patch_center):
    degrees = np.array([degree for _, degree in graph.degree()])
    center_coords = embedding[patch_center]
    dists = np.linalg.norm(embedding - center_coords, axis=1)
    return {
        "avg_degree": float(np.mean(degrees)),
        "degree_var": float(np.var(degrees)),
        "density": float(nx.density(graph)),
        "clustering": float(nx.average_clustering(graph)),
        "avg_dist_center": float(np.mean(dists)),
    }


def select_impulse_regions(
    noisy,
    ref,
    margin=SELECTION_MARGIN,
    min_distance=MIN_CENTER_DISTANCE,
    variance_window=VARIANCE_WINDOW,
):
    var_map = generic_filter(ref, np.var, size=variance_window, mode="reflect")
    impulse_mask = (noisy == 0.0) | (noisy == 255.0)

    valid_mask = impulse_mask.copy()
    valid_mask[:margin, :] = False
    valid_mask[-margin:, :] = False
    valid_mask[:, :margin] = False
    valid_mask[:, -margin:] = False

    ys, xs = np.where(valid_mask)
    vals = var_map[ys, xs]
    selected = []

    for key, label, quantile in REGION_TARGETS:
        target = np.quantile(vals, quantile)
        order = np.argsort(np.abs(vals - target))
        chosen = None

        for idx in order:
            y, x = int(ys[idx]), int(xs[idx])
            far_enough = all(
                (x - sx) ** 2 + (y - sy) ** 2 >= min_distance**2
                for _, _, sx, sy, _ in selected
            )
            if far_enough:
                chosen = (key, label, x, y, float(var_map[y, x]))
                break

        if chosen is None:
            idx = int(order[0])
            chosen = (
                key,
                label,
                int(xs[idx]),
                int(ys[idx]),
                float(var_map[ys[idx], xs[idx]]),
            )

        selected.append(chosen)

    return selected, var_map


def graph_record_for_pixel(noisy, ref, row, col, key, label, variance, h):
    patches, source, coords = Extract_patches_local(noisy, row, col, F, T)
    graph = build_KNN_Graph(patches, NN_COUNT)
    lengths, paths = nx.single_source_dijkstra(graph, source)

    img_n = mirror_cpu(noisy.astype(np.float32), F)
    im = row + F
    jn = col + F

    pixels_search = np.empty(len(coords), dtype=np.float32)
    valid_candidates = np.ones(len(coords), dtype=bool)
    spatial_weights = np.ones(len(coords), dtype=np.float32)

    for idx, (candidate_row, candidate_col) in enumerate(coords):
        r = candidate_row + F
        s = candidate_col + F
        patch = img_n[r - F:r + F + 1, s - F:s + F + 1]
        candidate_center = img_n[r, s]
        pixels_search[idx] = _robust_center_value(
            patch,
            candidate_center,
            z_alpha=Z_ALPHA,
            outlier_pixel_alpha=OUTLIER_PIXEL_ALPHA,
        )
        if REJECT_IMPULSE_CANDIDATES and (
            candidate_center == 0.0 or candidate_center == 255.0
        ):
            valid_candidates[idx] = False
        if USE_ASWMF_SPATIAL_WEIGHTS:
            spatial_weights[idx] = _aswmf_spatial_weight(
                r - im,
                s - jn,
                weight_diag_1=ASWMF_WEIGHT_DIAG_1,
                weight_diag_2=ASWMF_WEIGHT_DIAG_2,
                weight_other=ASWMF_WEIGHT_OTHER,
            )

    points = list(lengths.keys())
    distances = np.array(list(lengths.values()), dtype=np.float32)
    similarity_weights = np.exp(-(distances**2) / (h**2))
    similarity_weights *= valid_candidates[points].astype(np.float32)
    if USE_ASWMF_SPATIAL_WEIGHTS:
        similarity_weights *= spatial_weights[points]

    z_value = float(similarity_weights.sum())
    if z_value <= 0:
        filtered = _aswmf_local_fallback(
            img_n,
            im,
            jn,
            radius=F,
            weight_diag_1=ASWMF_WEIGHT_DIAG_1,
            weight_diag_2=ASWMF_WEIGHT_DIAG_2,
            weight_other=ASWMF_WEIGHT_OTHER,
        )
    else:
        filtered = float(np.sum(similarity_weights * pixels_search[points]) / z_value)

    tree_edges = set()
    active_edges = set()
    for point, weight in zip(points, similarity_weights):
        path = paths[point]
        for a, b in zip(path[:-1], path[1:]):
            edge = tuple(sorted((a, b)))
            tree_edges.add(edge)
            if weight > 0:
                active_edges.add(edge)

    embedding = PCA(n_components=2).fit_transform(patches)
    if np.allclose(embedding.std(axis=0), 0):
        embedding = np.array([(col, row) for row, col in coords], dtype=np.float32)
    pos = {idx: embedding[idx] for idx in range(len(embedding))}

    return {
        "key": key,
        "label": label,
        "row": int(row),
        "col": int(col),
        "variance": float(variance),
        "patches": patches,
        "source": int(source),
        "coords": coords,
        "graph": graph,
        "paths": paths,
        "path_lengths": lengths,
        "tree_edges": tree_edges,
        "active_edges": active_edges,
        "valid_candidates": valid_candidates,
        "similarity_weights": dict(zip(points, similarity_weights)),
        "z": z_value,
        "filtered": float(filtered),
        "center_noisy": float(noisy[row, col]),
        "center_reference": float(ref[row, col]),
        "embedding": embedding,
        "pos": pos,
        "patch_metrics": compute_patchspace_metrics(patches),
        "graph_metrics": compute_graph_metrics(graph, embedding, source),
    }


def build_graph_records(noisy, ref, selected, h):
    return [
        graph_record_for_pixel(noisy, ref, y, x, key, label, variance, h)
        for key, label, x, y, variance in selected
    ]


def draw_graph(ax, record, show_title=True, detailed=True):
    graph = record["graph"]
    pos = record["pos"]
    color = COLORS[record["label"]]

    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        edge_color="#c9c9c9",
        width=0.55,
        alpha=0.38,
    )
    if record["tree_edges"]:
        nx.draw_networkx_edges(
            graph,
            pos,
            ax=ax,
            edgelist=list(record["tree_edges"]),
            edge_color="#6baed6",
            width=1.05,
            alpha=0.72,
        )
    if record["active_edges"]:
        nx.draw_networkx_edges(
            graph,
            pos,
            ax=ax,
            edgelist=list(record["active_edges"]),
            edge_color=color,
            width=2.4,
            alpha=0.95,
        )

    node_colors = []
    node_sizes = []
    for node in graph.nodes:
        if node == record["source"]:
            node_colors.append("#e41a1c")
            node_sizes.append(100)
        elif not record["valid_candidates"][node]:
            node_colors.append("#111111")
            node_sizes.append(38)
        else:
            weight = record["similarity_weights"].get(node, 0.0)
            node_colors.append(color if weight > 0 else "#f2f2f2")
            node_sizes.append(56 if weight > 0 else 30)

    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="#333333",
        linewidths=0.35,
    )

    if show_title:
        if detailed:
            metrics = record["graph_metrics"]
            patch_metrics = record["patch_metrics"]
            ax.set_title(
                f"{record['label']} patch ({record['col']}, {record['row']})\n"
                f"nodes={graph.number_of_nodes()} edges={graph.number_of_edges()} "
                f"Z={record['z']:.2e}\n"
                f"density={metrics['density']:.3f} clustering={metrics['clustering']:.3f} "
                f"rank={patch_metrics['rank']}\n"
                f"noisy={record['center_noisy']:.0f} reference={record['center_reference']:.1f} "
                f"filtered={record['filtered']:.1f}",
                fontsize=10,
            )
        else:
            ax.set_title(record["label"], fontsize=10)
    ax.axis("off")


def save_region_overview(noisy, records, output_dir):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(noisy, cmap="gray", vmin=0, vmax=255)

    for idx, record in enumerate(records, start=1):
        color = COLORS[record["label"]]
        x = record["col"]
        y = record["row"]
        ax.add_patch(
            Rectangle(
                (x - T - 0.5, y - T - 0.5),
                2 * T + 1,
                2 * T + 1,
                fill=False,
                lw=2.1,
                ec=color,
            )
        )
        ax.scatter([x], [y], s=42, c=color, edgecolors="white", linewidths=0.8)
        ax.text(
            x + 7,
            y - 7,
            f"{record['label']} {idx}",
            color="white",
            fontsize=9,
            weight="bold",
            bbox=dict(facecolor=color, edgecolor="none", alpha=0.9, pad=2),
        )

    ax.axis("off")
    fig.savefig(output_dir / "selected_regions.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "selected_regions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_individual_outputs(noisy, records, output_dir):
    patch_dir = output_dir / "patches"
    graph_dir = output_dir / "graphs"
    patch_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    for idx, record in enumerate(records, start=1):
        x = record["col"]
        y = record["row"]
        crop = noisy[
            y - PATCH_CONTEXT_RADIUS:y + PATCH_CONTEXT_RADIUS + 1,
            x - PATCH_CONTEXT_RADIUS:x + PATCH_CONTEXT_RADIUS + 1,
        ]

        fig_patch, ax_patch = plt.subplots(figsize=(3.4, 3.4))
        ax_patch.imshow(crop, cmap="gray", vmin=0, vmax=255)
        ax_patch.add_patch(
            Rectangle(
                (
                    PATCH_CONTEXT_RADIUS - T - 0.5,
                    PATCH_CONTEXT_RADIUS - T - 0.5,
                ),
                2 * T + 1,
                2 * T + 1,
                fill=False,
                lw=1.6,
                ec=COLORS[record["label"]],
            )
        )
        ax_patch.scatter(
            [PATCH_CONTEXT_RADIUS],
            [PATCH_CONTEXT_RADIUS],
            s=28,
            c=COLORS[record["label"]],
            edgecolors="white",
            linewidths=0.7,
        )
        ax_patch.axis("off")
        fig_patch.savefig(
            patch_dir / f"{idx:02d}_{record['key']}_patch.pdf",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.01,
        )
        fig_patch.savefig(
            patch_dir / f"{idx:02d}_{record['key']}_patch.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.01,
        )
        plt.close(fig_patch)

        fig_graph, ax_graph = plt.subplots(figsize=(5.2, 5.2))
        draw_graph(ax_graph, record, show_title=False)
        fig_graph.savefig(
            graph_dir / f"{idx:02d}_{record['key']}_graph.pdf",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02,
        )
        fig_graph.savefig(
            graph_dir / f"{idx:02d}_{record['key']}_graph.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02,
        )
        plt.close(fig_graph)


def save_combined_figure(noisy, records, nlm_h, output_dir):
    h = nlm_h * H_MULTIPLIER
    fig = plt.figure(figsize=(24, 17), constrained_layout=True)
    gs = fig.add_gridspec(3, 7, width_ratios=[1.55, 1, 1, 1, 1, 1, 1])

    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.imshow(noisy, cmap="gray", vmin=0, vmax=255)
    for idx, record in enumerate(records, start=1):
        color = COLORS[record["label"]]
        x = record["col"]
        y = record["row"]
        ax_img.add_patch(
            Rectangle(
                (x - T - 0.5, y - T - 0.5),
                2 * T + 1,
                2 * T + 1,
                fill=False,
                lw=2.0,
                ec=color,
            )
        )
        ax_img.scatter([x], [y], s=42, c=color, edgecolors="white", linewidths=0.8)
        ax_img.text(
            x + 7,
            y - 7,
            f"{record['label']} {idx}",
            color="white",
            fontsize=9,
            weight="bold",
            bbox=dict(facecolor=color, edgecolor="none", alpha=0.9, pad=2),
        )
    ax_img.set_title("Selected impulse-centered patches", fontsize=13)
    ax_img.axis("off")

    for col, record in enumerate(records, start=1):
        x = record["col"]
        y = record["row"]
        color = COLORS[record["label"]]

        ax_patch = fig.add_subplot(gs[0, col])
        crop = noisy[
            y - PATCH_CONTEXT_RADIUS:y + PATCH_CONTEXT_RADIUS + 1,
            x - PATCH_CONTEXT_RADIUS:x + PATCH_CONTEXT_RADIUS + 1,
        ]
        ax_patch.imshow(crop, cmap="gray", vmin=0, vmax=255)
        ax_patch.add_patch(
            Rectangle(
                (
                    PATCH_CONTEXT_RADIUS - T - 0.5,
                    PATCH_CONTEXT_RADIUS - T - 0.5,
                ),
                2 * T + 1,
                2 * T + 1,
                fill=False,
                lw=1.8,
                ec=color,
            )
        )
        ax_patch.scatter(
            [PATCH_CONTEXT_RADIUS],
            [PATCH_CONTEXT_RADIUS],
            s=30,
            c=color,
            edgecolors="white",
            linewidths=0.7,
        )
        ax_patch.set_title(
            f"{record['label']} {col}\n"
            f"center=({x}, {y}) variance={record['variance']:.1f}",
            fontsize=10,
        )
        ax_patch.axis("off")

        ax_graph = fig.add_subplot(gs[1:, col])
        draw_graph(ax_graph, record, show_title=True, detailed=True)

    fig.suptitle(
        "Set12 07.png | salt-and-pepper medium | "
        f"f={F}, t={T}, nn={NN_COUNT}, h={h:.3f} "
        f"(nlm_h={nlm_h:.0f} x {H_MULTIPLIER})",
        fontsize=16,
    )
    fig.savefig(output_dir / "set12_07_medium_path_graphs_combined.pdf", dpi=300)
    fig.savefig(output_dir / "set12_07_medium_path_graphs_combined.png", dpi=300)
    plt.close(fig)


def save_graph_vector(records, output_dir):
    with (output_dir / "set12_07_medium_graph_records.pkl").open("wb") as fobj:
        pickle.dump(records, fobj)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    item, noisy, ref, nlm_h = load_item()
    h = nlm_h * H_MULTIPLIER
    selected, _ = select_impulse_regions(noisy, ref)
    records = build_graph_records(noisy, ref, selected, h)

    save_graph_vector(records, OUTPUT_DIR)
    save_region_overview(noisy, records, OUTPUT_DIR)
    save_individual_outputs(noisy, records, OUTPUT_DIR)
    save_combined_figure(noisy, records, nlm_h, OUTPUT_DIR)

    print(f"File: {FILE_NAME}")
    print(f"Pickle: {PICKLE_PATH}")
    print(f"Salt-and-pepper density: {item['salt_pepper_density']}")
    print(f"nlm_h={nlm_h:.0f}; h={h:.3f}; f={F}; t={T}; nn={NN_COUNT}")
    print(f"Output directory: {OUTPUT_DIR}")
    for idx, record in enumerate(records, start=1):
        print(
            f"{idx:02d} {record['label']:6s}: center=({record['col']:3d}, {record['row']:3d}), "
            f"variance={record['variance']:8.2f}, noisy={record['center_noisy']:6.1f}, "
            f"reference={record['center_reference']:6.1f}, filtered={record['filtered']:6.1f}, "
            f"Z={record['z']:.2e}"
        )

    return records


if __name__ == "__main__":
    main()
