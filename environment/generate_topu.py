import pickle
import networkx as nx


def generate_topology(
    num_nodes,
    min_degree,
    max_degree,
    delay_range,
    bandwidth_range,
    rng,
    target_avg_degree=None,
    target_degrees=None,
    max_attempts=50,
):
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    if target_degrees is not None or target_avg_degree is not None:
        for _ in range(max_attempts):
            degrees = _prepare_target_degrees(
                num_nodes,
                min_degree,
                max_degree,
                rng,
                target_avg_degree=target_avg_degree,
                target_degrees=target_degrees,
            )
            if _connect_with_targets(graph, degrees, delay_range, bandwidth_range, rng):
                return graph
            graph.clear()
            graph.add_nodes_from(range(num_nodes))
    _ensure_min_degree(
        graph,
        min_degree,
        max_degree,
        delay_range,
        bandwidth_range,
        rng,
    )
    return graph


def visualize_topology(
    graph,
    pos=None,
    drawer=None,
    show=True,
    save_path=None,
    layout="spring",
    show_edge_labels=False,
    edge_curvature=0.15,
    edge_width=1.6,
    edge_alpha=0.8,
):
    if drawer is not None:
        return drawer(graph, pos=pos, show=show, save_path=save_path)
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib 不可用，请安装或提供 drawer") from exc
    if pos is None:
        if layout == "circular":
            layout = nx.circular_layout(graph)
        elif layout == "shell":
            layout = nx.shell_layout(graph)
        elif layout == "kamada_kawai":
            layout = nx.kamada_kawai_layout(graph)
        else:
            layout = nx.spring_layout(graph, seed=0, k=1.2, iterations=200)
    else:
        layout = pos
    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_nodes(
        graph,
        layout,
        ax=ax,
        node_color="#86b6f6",
        edgecolors="#2f4f7f",
        linewidths=1.0,
        node_size=700,
    )
    edge_kwargs = {}
    if edge_curvature:
        edge_kwargs["connectionstyle"] = f"arc3,rad={edge_curvature}"
    nx.draw_networkx_edges(
        graph,
        layout,
        ax=ax,
        edge_color="#4d4d4d",
        width=edge_width,
        alpha=edge_alpha,
        **edge_kwargs,
    )
    nx.draw_networkx_labels(
        graph,
        layout,
        ax=ax,
        font_size=9,
        font_color="#1a1a1a",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
    )
    if show_edge_labels:
        edge_labels = {
            (u, v): f"d={data.get('delay', 0):.1f}, b={data.get('bandwidth', 0):.1f}"
            for u, v, data in graph.edges(data=True)
        }
        nx.draw_networkx_edge_labels(
            graph,
            layout,
            edge_labels=edge_labels,
            ax=ax,
            font_size=7,
            rotate=False,
        )
    ax.axis("off")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def _prepare_target_degrees(
    num_nodes,
    min_degree,
    max_degree,
    rng,
    target_avg_degree=None,
    target_degrees=None,
):
    if target_degrees is not None:
        if len(target_degrees) != num_nodes:
            raise ValueError("target_degrees 长度必须等于 num_nodes")
        degrees = [int(d) for d in target_degrees]
    else:
        degrees = rng.integers(min_degree, max_degree + 1, size=num_nodes).tolist()
        if target_avg_degree is not None:
            target_avg_degree = float(target_avg_degree)
            degrees = _adjust_average_degree(degrees, min_degree, max_degree, target_avg_degree, rng)
    degrees = [max(min_degree, min(max_degree, int(d))) for d in degrees]
    if sum(degrees) % 2 == 1:
        idx = int(rng.integers(0, num_nodes))
        if degrees[idx] < max_degree:
            degrees[idx] += 1
        elif degrees[idx] > min_degree:
            degrees[idx] -= 1
    return degrees


def _adjust_average_degree(degrees, min_degree, max_degree, target_avg_degree, rng, max_steps=5000):
    current_avg = sum(degrees) / len(degrees)
    steps = 0
    while abs(current_avg - target_avg_degree) > 0.2 and steps < max_steps:
        if current_avg > target_avg_degree:
            candidates = [i for i, d in enumerate(degrees) if d > min_degree]
            if not candidates:
                break
            idx = int(rng.choice(candidates))
            degrees[idx] -= 1
        else:
            candidates = [i for i, d in enumerate(degrees) if d < max_degree]
            if not candidates:
                break
            idx = int(rng.choice(candidates))
            degrees[idx] += 1
        current_avg = sum(degrees) / len(degrees)
        steps += 1
    return degrees


def _connect_with_targets(graph, target_degrees, delay_range, bandwidth_range, rng, max_steps=10000):
    remaining = {node: int(target_degrees[node]) for node in graph.nodes()}
    steps = 0
    while steps < max_steps and sum(remaining.values()) > 0:
        nodes = [n for n, r in remaining.items() if r > 0]
        if len(nodes) < 2:
            break
        node = int(rng.choice(nodes))
        partner_candidates = [
            n for n in nodes
            if n != node and remaining[n] > 0 and not graph.has_edge(node, n)
        ]
        if not partner_candidates:
            steps += 1
            continue
        partner = int(rng.choice(partner_candidates))
        _add_edge_with_attrs(graph, node, partner, delay_range, bandwidth_range, rng)
        remaining[node] -= 1
        remaining[partner] -= 1
        steps += 1
    return sum(remaining.values()) == 0


def _add_edge_with_attrs(graph, node, neighbor, delay_range, bandwidth_range, rng):
    graph.add_edge(
        node,
        neighbor,
        delay=rng.uniform(*delay_range),
        bandwidth=rng.uniform(*bandwidth_range),
    )


def save_topology(graph, path):
    with open(path, "wb") as file_handle:
        pickle.dump(graph, file_handle)


def load_topology(path):
    with open(path, "rb") as file_handle:
        return pickle.load(file_handle)


def _ensure_min_degree(graph, min_degree, max_degree, delay_range, bandwidth_range, rng, max_tries=1000):
    tries = 0
    while tries < max_tries:
        low_degree_nodes = [n for n, d in graph.degree() if d < min_degree]
        if not low_degree_nodes:
            break
        node = low_degree_nodes[0]
        candidate_nodes = [
            n for n, d in graph.degree()
            if n != node and d < max_degree and not graph.has_edge(node, n)
        ]
        if not candidate_nodes:
            break
        neighbor = rng.choice(candidate_nodes)
        _add_edge_with_attrs(graph, node, neighbor, delay_range, bandwidth_range, rng)
        tries += 1
    _fill_remaining_edges(graph, max_degree, delay_range, bandwidth_range, rng)


def _fill_remaining_edges(graph, max_degree, delay_range, bandwidth_range, rng, max_edges=2000):
    edges_added = 0
    while edges_added < max_edges:
        candidates = [n for n, d in graph.degree() if d < max_degree]
        if len(candidates) < 2:
            break
        node = rng.choice(candidates)
        neighbor_candidates = [
            n for n, d in graph.degree()
            if n != node and d < max_degree and not graph.has_edge(node, n)
        ]
        if not neighbor_candidates:
            break
        neighbor = rng.choice(neighbor_candidates)
        _add_edge_with_attrs(graph, node, neighbor, delay_range, bandwidth_range, rng)
        edges_added += 1


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(0)
    graph = generate_topology(18, 2, 7, (1.0, 10.0), (10.0, 100.0), rng, target_degrees=[2, 2, 2, 2, 3, 6, 6, 5, 5, 2, 4, 5, 7, 5, 3, 5, 5, 3])
    save_topology(graph, "topology.pkl")
    graph = load_topology("topology.pkl")
    visualize_topology(graph, save_path="topology.png", show=False, edge_curvature=0.25)