#!/usr/bin/env python3
"""生成内场业务流及其当前路径。

默认随机生成 20 对源目的节点业务，路径格式为：
  node_id（manage_ip） -> node_id（manage_ip）

运行方式：
  # 默认在线获取拓扑和链路指标后生成 20 条随机业务。
  python generate_inner_business_flows.py

  # 指定随机业务数量。
  python generate_inner_business_flows.py --count 50

  # 使用本地拓扑和链路指标 JSON。
  python generate_inner_business_flows.py --net_offline
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx

from environment.inner_graph_data import qos_routing as qos
from environment.inner_graph_data import topology_to_networkx as topo


ROOT = Path(__file__).resolve().parent
GRAPH_DIR = ROOT / "environment" / "inner_graph_data"
DEFAULT_OUTPUT = GRAPH_DIR / "json-data" / "inner_business_flows.json"
DEFAULT_TOPOLOGY_JSON = GRAPH_DIR / "json-data" / "network_topology_state.json"
DEFAULT_LINK_METRIC_JSON = GRAPH_DIR / "json-data" / "link_metric.json"
DEFAULT_BASE_GRAPHML = GRAPH_DIR / "base_ii_topology.graphml"

# 在这里直接添加指定源目的节点业务；不需要从命令行传参。
# 示例：CUSTOM_FLOWS = [("asu0n0", "eru1n5"), ("asu0n1", "eru1n4")]
CUSTOM_FLOWS: list[tuple[str, str]] = []

# 在这里直接添加指定源目的 IP 业务，支持节点管理 IP 或端口 IP。
# 示例：CUSTOM_FLOW_IPS = [("10.104.0.254", "10.103.21.254")]
CUSTOM_FLOW_IPS: list[tuple[str, str]] = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="随机生成内场业务流路径文件。")
    parser.add_argument("--count", type=int, default=20, help="随机业务数量，默认 20。")
    parser.add_argument("--seed", type=int, help="随机种子；不填则每次随机。")
    parser.add_argument("--net_offline", action="store_true", help="使用本地拓扑和链路指标 JSON；默认在线获取。")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="业务流输出 JSON 文件。")
    parser.add_argument("--topology-json", type=Path, default=DEFAULT_TOPOLOGY_JSON, help="拓扑 JSON 路径。")
    parser.add_argument("--link-metric-json", type=Path, default=DEFAULT_LINK_METRIC_JSON, help="链路指标 JSON 路径。")
    parser.add_argument("--base-graphml", type=Path, default=DEFAULT_BASE_GRAPHML, help="base GraphML 路径。")
    return parser.parse_args()


def load_routing_graph(args: argparse.Namespace) -> tuple[nx.Graph, nx.Graph]:
    fetch_online = not args.net_offline
    graph = topo.build_updated_graph(
        latest_json=args.topology_json,
        base_graphml=args.base_graphml,
        fetch_latest=fetch_online,
        rebuild_base=False,
    )
    if fetch_online:
        topo.fetch_latest_link_metrics_json(args.link_metric_json)
    if args.link_metric_json.exists():
        topo.update_graph_from_link_metrics(graph, args.link_metric_json)

    routing_graph = qos.build_online_routing_graph(graph, offline_status=topo.OFFLINE_STATUS)
    return graph, routing_graph


def resolve_node_by_ip(graph: nx.Graph, ip: str) -> str:
    matches = []
    for node, attrs in graph.nodes(data=True):
        if attrs.get("node_manage_ip_addr") == ip:
            matches.append((node, "manage"))
            continue
        for port in attrs.get("node_ports", []):
            if isinstance(port, dict) and port.get("ip_address") == ip:
                matches.append((node, "port"))
                break

    if not matches:
        raise ValueError(f"IP not found in online graph: {ip}")

    manage_matches = [node for node, match_type in matches if match_type == "manage"]
    if manage_matches:
        return manage_matches[0]

    unique_nodes = sorted({node for node, _ in matches})
    if len(unique_nodes) > 1:
        raise ValueError(f"IP maps to multiple nodes: {ip} -> {unique_nodes}")
    return unique_nodes[0]


def path_text(graph: nx.Graph, path: list[str]) -> str:
    return " -> ".join(f"{node}（{graph.nodes[node].get('node_manage_ip_addr') or '-'}）" for node in path)


def flow_record(graph: nx.Graph, src: str, dst: str, flow_id: str, source: str) -> dict[str, Any]:
    path = nx.shortest_path(graph, source=src, target=dst, weight="link_latency")
    metrics = qos.path_metrics(graph, path)
    return {
        "flow_id": flow_id,
        "source": source,
        "src": src,
        "dst": dst,
        "path_nodes": path,
        "path_info": path_text(graph, path),
        "hop_count": int(metrics["hop_count"]),
        "total_latency": metrics["total_latency"],
        "bottleneck_bandwidth": metrics["bottleneck_bandwidth"],
        "path_loss_rate": metrics["path_loss_rate"],
    }


def connected_ordered_pairs(graph: nx.Graph) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for component in nx.connected_components(graph):
        nodes = sorted(component)
        for src in nodes:
            for dst in nodes:
                if src != dst:
                    pairs.append((src, dst))
    return pairs


def main() -> int:
    args = parse_args()
    if args.count < 0:
        print(f"[ERR] --count 不能为负数: {args.count}", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    graph, routing_graph = load_routing_graph(args)
    if routing_graph.number_of_nodes() < 2:
        summary = topo.summarize_graph(graph)
        print(
            "[ERR] 路由图在线节点不足，无法生成业务。"
            f" 当前节点数={routing_graph.number_of_nodes()}, 边数={routing_graph.number_of_edges()}。",
            file=sys.stderr,
        )
        print(
            "[ERR] 请检查拓扑 JSON 是否为空，或先在线刷新/重建 base。"
            f" graph_summary={json.dumps(summary, ensure_ascii=False)}",
            file=sys.stderr,
        )
        return 1

    flows: list[dict[str, Any]] = []
    used_pairs: set[tuple[str, str]] = set()

    for idx, (src, dst) in enumerate(CUSTOM_FLOWS, start=1):
        if src not in routing_graph or dst not in routing_graph:
            print(f"[ERR] 指定业务节点不在线: {src} -> {dst}", file=sys.stderr)
            return 1
        if not nx.has_path(routing_graph, src, dst):
            print(f"[ERR] 指定业务不可达: {src} -> {dst}", file=sys.stderr)
            return 1
        flows.append(flow_record(routing_graph, src, dst, f"custom_{idx:03d}", "custom"))
        used_pairs.add((src, dst))

    offset = len(flows)
    for idx, (src_ip, dst_ip) in enumerate(CUSTOM_FLOW_IPS, start=1):
        src = resolve_node_by_ip(routing_graph, src_ip)
        dst = resolve_node_by_ip(routing_graph, dst_ip)
        if not nx.has_path(routing_graph, src, dst):
            print(f"[ERR] 指定 IP 业务不可达: {src_ip}({src}) -> {dst_ip}({dst})", file=sys.stderr)
            return 1
        flows.append(flow_record(routing_graph, src, dst, f"custom_{offset + idx:03d}", "custom_ip"))
        used_pairs.add((src, dst))

    candidates = [pair for pair in connected_ordered_pairs(routing_graph) if pair not in used_pairs]
    rng.shuffle(candidates)
    if args.count > len(candidates):
        print(f"[WARN] 可用随机业务只有 {len(candidates)} 条，将少于请求数量 {args.count}。")

    for idx, (src, dst) in enumerate(candidates[: args.count], start=1):
        flows.append(flow_record(routing_graph, src, dst, f"random_{idx:03d}", "random"))

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "net_mode": "offline_json" if args.net_offline else "online_api",
        "random_count": min(args.count, len(candidates)),
        "custom_count": len(CUSTOM_FLOWS) + len(CUSTOM_FLOW_IPS),
        "flow_count": len(flows),
        "flows": flows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] saved flows: {args.output}")
    print(f"[INFO] flow_count={len(flows)}")
    for flow in flows[: min(10, len(flows))]:
        print(f"  - {flow['flow_id']}: {flow['path_info']}")
    if len(flows) > 10:
        print(f"  ... 其余 {len(flows) - 10} 条见输出文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
