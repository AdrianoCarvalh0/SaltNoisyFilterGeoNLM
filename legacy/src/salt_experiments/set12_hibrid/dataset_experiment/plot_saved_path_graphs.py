from pathlib import Path
import argparse
import pickle

import matplotlib.pyplot as plt
import networkx as nx


DEFAULT_GRAPH_RECORDS = Path(
    "/workspace/data/output/set12Hibrid/salt_pepper_medium/full_512/results/"
    "path_graphs_07_medium/set12_07_medium_graph_records.pkl"
)
DEFAULT_OUTPUT_DIR = Path(
    "/workspace/data/output/set12Hibrid/salt_pepper_medium/full_512/results/"
    "path_graphs_07_medium/networkx_style"
)


def load_graph_records(path):
    with Path(path).open("rb") as fobj:
        return pickle.load(fobj)


def graph_layout(graph, layout_name="spring", seed=42):
    if layout_name == "kamada":
        return nx.kamada_kawai_layout(graph, weight="weight")
    if layout_name == "spectral":
        return nx.spectral_layout(graph, weight="weight")
    return nx.spring_layout(
        graph,
        seed=seed,
        weight="weight",
        iterations=300,
        k=None,
    )


def plot_graph_record(
    record,
    ax=None,
    layout_name="spring",
    seed=42,
    node_size=42,
    edge_width=0.25,
    show_title=False,
):
    graph = record["graph"]
    source = record["source"]
    pos = graph_layout(graph, layout_name=layout_name, seed=seed)

    if ax is None:
        _, ax = plt.subplots(figsize=(8.5, 6.0))

    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        edge_color="#8f8f8f",
        width=edge_width,
        alpha=0.42,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color="#3158b6",
        node_size=node_size,
        linewidths=0.0,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        nodelist=[source],
        node_color="#e41a1c",
        node_size=node_size * 1.18,
        linewidths=0.0,
    )

    if show_title:
        ax.set_title(
            f"{record['label']} patch ({record['col']}, {record['row']})",
            fontsize=11,
        )

    ax.set_axis_off()
    return ax


def save_individual_graphs(
    records,
    output_dir,
    layout_name="spring",
    seed=42,
    node_size=42,
    edge_width=0.25,
    dpi=300,
):
    individual_dir = output_dir / "individual"
    individual_dir.mkdir(parents=True, exist_ok=True)

    for idx, record in enumerate(records, start=1):
        fig, ax = plt.subplots(figsize=(8.5, 6.0))
        plot_graph_record(
            record,
            ax=ax,
            layout_name=layout_name,
            seed=seed + idx,
            node_size=node_size,
            edge_width=edge_width,
            show_title=False,
        )
        base_name = f"{idx:02d}_{record['key']}_{layout_name}"
        fig.savefig(individual_dir / f"{base_name}.pdf", dpi=dpi, bbox_inches="tight")
        fig.savefig(individual_dir / f"{base_name}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def save_combined_graphs(
    records,
    output_dir,
    layout_name="spring",
    seed=42,
    node_size=28,
    edge_width=0.2,
    dpi=300,
):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    for idx, (ax, record) in enumerate(zip(axes.ravel(), records), start=1):
        plot_graph_record(
            record,
            ax=ax,
            layout_name=layout_name,
            seed=seed + idx,
            node_size=node_size,
            edge_width=edge_width,
            show_title=True,
        )

    fig.savefig(output_dir / f"all_graphs_{layout_name}.pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / f"all_graphs_{layout_name}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot saved path graphs using the original NetworkX-style graph view."
    )
    parser.add_argument("--records", type=Path, default=DEFAULT_GRAPH_RECORDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--layout",
        choices=["spring", "kamada", "spectral"],
        default="spring",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--node-size", type=float, default=42)
    parser.add_argument("--edge-width", type=float, default=0.25)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    records = load_graph_records(args.records)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    save_individual_graphs(
        records,
        args.output_dir,
        layout_name=args.layout,
        seed=args.seed,
        node_size=args.node_size,
        edge_width=args.edge_width,
        dpi=args.dpi,
    )
    save_combined_graphs(
        records,
        args.output_dir,
        layout_name=args.layout,
        seed=args.seed,
        node_size=args.node_size * 0.75,
        edge_width=args.edge_width * 0.8,
        dpi=args.dpi,
    )

    print(f"Loaded records: {len(records)}")
    print(f"Layout: {args.layout}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
