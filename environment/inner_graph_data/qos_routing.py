#!/usr/bin/env python3
"""QoS-aware routing helpers for II-class NetworkX graphs.

Algorithm:
  1. Keep only online nodes and links.
  2. Filter links whose available bandwidth is below the requested minimum.
  3. Generate K candidate paths ordered by total link_latency.
  4. Return the first candidate whose end-to-end loss is within the limit.
"""

from __future__ import annotations

from itertools import islice
from math import prod
from typing import Any, Iterable

import networkx as nx


class NoQoSPath(RuntimeError):
    """Raised when no path satisfies the requested QoS constraints."""


def build_online_routing_graph(graph: nx.Graph, offline_status: int = 0) -> nx.Graph:
    """Keep only online nodes and links; collapse parallel links by min latency."""
    routing_graph = nx.Graph()

    for node, attrs in graph.nodes(data=True):
        if int(attrs.get("node_status", 0)) != offline_status:
            routing_graph.add_node(node, **attrs)

    edge_iter = (
        graph.edges(keys=True, data=True)
        if graph.is_multigraph()
        else ((u, v, None, attrs) for u, v, attrs in graph.edges(data=True))
    )
    for src, dst, _, attrs in edge_iter:
        if src not in routing_graph or dst not in routing_graph:
            continue
        if int(attrs.get("link_status", 0)) == offline_status:
            continue
        _add_best_parallel_edge(routing_graph, src, dst, attrs)

    return routing_graph


def choose_reachable_pair(graph: nx.Graph) -> tuple[str, str]:
    """Pick two nodes from the largest connected component."""
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    for component in components:
        if len(component) >= 2:
            nodes = sorted(component)
            return nodes[0], nodes[-1]
    raise NoQoSPath("no reachable online node pair found")


def find_delay_shortest_qos_path(
    graph: nx.Graph,
    src: str,
    dst: str,
    *,
    min_bandwidth: float = 0.0,
    max_loss_rate: float = 1.0,
    k_paths: int = 20,
) -> dict[str, Any]:
    """Find the lowest-delay path that satisfies bandwidth and loss constraints."""
    candidate_graph = _filter_by_min_bandwidth(graph, min_bandwidth)

    checked = 0
    try:
        candidates = nx.shortest_simple_paths(
            candidate_graph,
            source=src,
            target=dst,
            weight="link_latency",
        )
        for path in islice(candidates, max(1, k_paths)):
            checked += 1
            metrics = path_metrics(candidate_graph, path)
            if metrics["path_loss_rate"] <= max_loss_rate:
                return {
                    "path": path,
                    "metrics": metrics,
                    "checked_paths": checked,
                    "candidate_graph": candidate_graph,
                }
    except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
        raise NoQoSPath(str(exc)) from exc

    raise NoQoSPath(
        f"no path satisfies min_bandwidth={min_bandwidth}Mbps, "
        f"max_loss_rate={max_loss_rate}, k_paths={k_paths}"
    )


def path_metrics(graph: nx.Graph, path: list[str]) -> dict[str, float]:
    edges = list(iter_path_edges(graph, path))
    if not edges:
        return {
            "total_latency": 0.0,
            "hop_count": 0,
            "bottleneck_bandwidth": 0.0,
            "path_loss_rate": 0.0,
            "max_link_utilization": 0.0,
            "avg_link_utilization": 0.0,
        }

    losses = [loss_rate(edge) for edge in edges]
    utilizations = [link_utilization(edge) for edge in edges]
    return {
        "total_latency": sum(link_latency(edge) for edge in edges),
        "hop_count": float(len(edges)),
        "bottleneck_bandwidth": min(available_bandwidth(edge) for edge in edges),
        "path_loss_rate": 1.0 - prod(1.0 - loss for loss in losses),
        "max_link_utilization": max(utilizations) if utilizations else 0.0,
        "avg_link_utilization": sum(utilizations) / len(utilizations) if utilizations else 0.0,
    }


def path_hop_details(graph: nx.Graph, path: list[str]) -> list[dict[str, object]]:
    hops = []
    for src, dst in zip(path, path[1:]):
        edge = graph[src][dst]
        src_port, src_port_ip, dst_port, dst_port_ip = _oriented_ports(edge, src, dst)
        hops.append(
            {
                "src_node": src,
                "src_manage_ip": graph.nodes[src].get("node_manage_ip_addr", ""),
                "src_port": src_port,
                "src_port_ip": src_port_ip,
                "dst_node": dst,
                "dst_manage_ip": graph.nodes[dst].get("node_manage_ip_addr", ""),
                "dst_port": dst_port,
                "dst_port_ip": dst_port_ip,
                "link_id": edge.get("link_id", ""),
                "link_latency": link_latency(edge),
                "link_bandwidth": safe_float(edge.get("link_bandwidth"), 0.0),
                "link_utilization": edge.get("link_utilization"),
                "bandwidth_capacity_available": edge.get("bandwidth_capacity_available"),
                "available_bandwidth": available_bandwidth(edge),
                "link_loss_rate": edge.get("link_loss_rate"),
            }
        )
    return hops


def iter_path_edges(graph: nx.Graph, path: list[str]) -> Iterable[dict[str, Any]]:
    for src, dst in zip(path, path[1:]):
        yield graph[src][dst]


def available_bandwidth(edge: dict[str, Any]) -> float:
    value = edge.get("bandwidth_capacity_available")
    if value in (None, ""):
        value = edge.get("link_bandwidth")
    return safe_float(value, 0.0)


def loss_rate(edge: dict[str, Any]) -> float:
    return min(max(safe_float(edge.get("link_loss_rate"), 0.0), 0.0), 1.0)


def link_utilization(edge: dict[str, Any]) -> float:
    return min(max(safe_float(edge.get("link_utilization"), 0.0), 0.0), 1.0)


def link_latency(edge: dict[str, Any]) -> float:
    return max(safe_float(edge.get("link_latency"), 0.0), 0.0)


def safe_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _filter_by_min_bandwidth(graph: nx.Graph, min_bandwidth: float) -> nx.Graph:
    filtered = nx.Graph()
    filtered.add_nodes_from(graph.nodes(data=True))
    for src, dst, attrs in graph.edges(data=True):
        if available_bandwidth(attrs) < min_bandwidth:
            continue
        filtered.add_edge(src, dst, **attrs)
    return filtered


def _add_best_parallel_edge(graph: nx.Graph, src: str, dst: str, attrs: dict[str, Any]) -> None:
    if not graph.has_edge(src, dst):
        graph.add_edge(src, dst, **attrs)
        return

    old = graph[src][dst]
    old_key = (link_latency(old), -available_bandwidth(old), loss_rate(old))
    new_key = (link_latency(attrs), -available_bandwidth(attrs), loss_rate(attrs))
    if new_key < old_key:
        graph.remove_edge(src, dst)
        graph.add_edge(src, dst, **attrs)


def _oriented_ports(
    edge: dict[str, Any],
    src: str,
    dst: str,
) -> tuple[str, str, str, str]:
    edge_src = edge.get("src_node", src)
    edge_dst = edge.get("dst_node", dst)

    if edge_src == src and edge_dst == dst:
        return (
            edge.get("src_port", ""),
            edge.get("src_port_ip", ""),
            edge.get("dst_port", ""),
            edge.get("dst_port_ip", ""),
        )
    if edge_src == dst and edge_dst == src:
        return (
            edge.get("dst_port", ""),
            edge.get("dst_port_ip", ""),
            edge.get("src_port", ""),
            edge.get("src_port_ip", ""),
        )
    return (
        edge.get("src_port", ""),
        edge.get("src_port_ip", ""),
        edge.get("dst_port", ""),
        edge.get("dst_port_ip", ""),
    )
