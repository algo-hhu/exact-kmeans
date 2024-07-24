import argparse
import json
from pathlib import Path
from typing import List, Optional

import graphviz
import numpy as np


def plot(
    nodes: List,
    filename: str,
    plot_folder: Path,
    optimal_objective: Optional[float] = None,
    branch_color: str = "gray",
    penwidth: str = "2",
) -> None:
    plot_folder = Path(plot_folder)
    plot_folder.mkdir(exist_ok=True)

    g = graphviz.Digraph("G", format="png")
    g.attr(rankdir="LR")
    g.node("root", "[]", style="filled", fillcolor=branch_color, penwidth=penwidth)
    nodes.sort(key=lambda x: x[1])
    for obj, time, sizes in nodes:
        if obj == "branch":
            color = branch_color
        elif obj == "sum_bound_greater":
            color = "yellow"
        elif obj == "ilp_sum_bound_greater":
            color = "orange"
        elif obj == "infeasible":
            color = "red"
        elif obj == "k-1":
            color = "cyan"
        elif isinstance(obj, float):
            color = "green"
        else:
            raise ValueError(f"Unknown object {obj}")

        if (
            optimal_objective is not None
            and isinstance(obj, float)
            and np.isclose(obj, optimal_objective)
        ):
            border_color = "gold"
        else:
            border_color = "black"

        if isinstance(obj, float):
            node_content = f"{sizes}\n{time:.3f}s\n({obj:.2f})"
            # edge_content = f"{sizes}"
        else:
            node_content = f"{sizes}\n {time:.3f}s"

        edge_content = ""

        g.node(
            f"{sizes}",
            node_content,
            style="filled",
            fillcolor=color,
            color=border_color,
            penwidth=penwidth,
        )
        if len(sizes) == 1:
            g.edge("root", f"{sizes}", edge_content)
        else:
            g.edge(f"{sizes[:-1]}", f"{sizes}", edge_content)

    g.render(filename=plot_folder / f"{filename.replace('.json', '')}", format="png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--plot-folder", type=Path, required=True)

    args = parser.parse_args()

    with open(args.output_json, "r") as f:
        jj = json.load(f)
    plot(
        nodes=jj["processed_cluster_sizes"],
        optimal_objective=jj["objective"] if "objective" in jj else None,
        filename=args.output_json.name,
        plot_folder=args.plot_folder,
    )
