#!/usr/bin/env python3
"""离线生成内场业务 JSON，结构与 /api/v1/task/all 返回一致。

默认随机生成 20 对源目的业务，路径字段使用链路序列：
  nodeA_nodeB -> nodeB_nodeC

运行方式：
  # 默认在线获取拓扑和链路指标后，离线生成 20 条业务。
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
import re
import sys
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
    parser = argparse.ArgumentParser(description="离线随机生成内场业务 JSON。")
    parser.add_argument("--count", type=int, default=20, help="随机业务数量，默认 20。")
    parser.add_argument("--seed", type=int, help="随机种子；不填则每次随机。")
    parser.add_argument("--net_offline", action="store_true", help="使用本地拓扑和链路指标 JSON；默认在线获取。")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="业务 JSON 输出文件。")
    parser.add_argument("--topology-json", type=Path, default=DEFAULT_TOPOLOGY_JSON, help="拓扑 JSON 路径。")
    parser.add_argument("--link-metric-json", type=Path, default=DEFAULT_LINK_METRIC_JSON, help="链路指标 JSON 路径。")
    parser.add_argument("--base-graphml", type=Path, default=DEFAULT_BASE_GRAPHML, help="保留参数，当前离线业务生成不依赖 base GraphML。")
    return parser.parse_args()


def load_routing_graph(args: argparse.Namespace) -> tuple[nx.Graph, nx.Graph]:
    fetch_online = not args.net_offline
    if fetch_online:
        topo.fetch_latest_topology_json(args.topology_json)
        topo.fetch_latest_link_metrics_json(args.link_metric_json)

    # 业务接口中的 start/end 包含 hu 等网关节点，因此这里加载全量节点类型。
    graph = topo.load_topology_graph(args.topology_json, multigraph=False, node_types=None)
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


def host_ip(graph: nx.Graph, node: str) -> str:
    ports = graph.nodes[node].get("node_ports", [])
    for port in ports:
        if isinstance(port, dict) and port.get("ip_address"):
            return str(port["ip_address"])
    return str(graph.nodes[node].get("node_manage_ip_addr") or "")


def reported_cluster_id(node: str) -> str:
    match = re.search(r"([a-z]+)(\d+)n\d+", node)
    if not match:
        return "1"
    return str(int(match.group(2)) + 1)


def link_label(src: str, dst: str) -> str:
    return f"{src}_{dst}"


def path_link_labels(path: list[str]) -> list[str]:
    return [link_label(src, dst) for src, dst in zip(path, path[1:])]


def edge_for_hop(graph: nx.Graph, src: str, dst: str) -> dict[str, Any]:
    edge = graph[src][dst]
    return dict(edge)


def task_record(graph: nx.Graph, src: str, dst: str, idx: int, rng: random.Random) -> dict[str, Any]:
    path = nx.shortest_path(graph, source=src, target=dst, weight="link_latency")
    links = path_link_labels(path)
    bandwidth_demand = round(rng.uniform(10.0, 45.0), 2)
    task_type = 2 if rng.random() < 0.75 else 1
    computing_allocated = (
        rng.uniform(3.0, 4.5) if task_type == 1 else rng.uniform(8.0, 10.8)
    )
    start_host_ip = host_ip(graph, src)
    end_host_ip = host_ip(graph, dst)
    start_port = str(rng.randint(30000, 65000))
    end_port = str(15000 + idx)

    bandwidth_allocated: dict[str, float] = {}
    bandwidth_sla: dict[str, float] = {}
    for link_id, hop_src, hop_dst in zip(links, path, path[1:]):
        edge = edge_for_hop(graph, hop_src, hop_dst)
        allocated = qos.available_bandwidth(edge)
        bandwidth_allocated[link_id] = allocated
        bandwidth_sla[link_id] = allocated / bandwidth_demand if bandwidth_demand else 0.0

    return {
        "task_id": f"{dst}:{end_host_ip}:{end_port}_{src}:{start_host_ip}:{start_port}",
        "type": task_type,
        "bandwidth_demand": bandwidth_demand,
        "computing_demand": 1.0,
        "start": src,
        "domain": "domain-1",
        "reported_cluster_id": reported_cluster_id(src),
        "end": dst,
        "start_host_ip": start_host_ip,
        "end_host_ip": end_host_ip,
        "start_port": start_port,
        "end_port": end_port,
        "computing_allocated": computing_allocated,
        "computing_sla": computing_allocated,
        "computing_allocated_tra": 0.0,
        "bandwidth_sla": bandwidth_sla,
        "bandwidth_sla_tra": {},
        "bandwidth_allocated": bandwidth_allocated,
        "bandwidth_allocated_tra": {},
        "path": links,
    }


def connected_ordered_pairs(graph: nx.Graph, endpoint_nodes: set[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for component in nx.connected_components(graph):
        nodes = sorted(set(component) & endpoint_nodes)
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

    tasks: list[dict[str, Any]] = []
    used_pairs: set[tuple[str, str]] = set()

    for idx, (src, dst) in enumerate(CUSTOM_FLOWS, start=1):
        if src not in routing_graph or dst not in routing_graph:
            print(f"[ERR] 指定业务节点不在线: {src} -> {dst}", file=sys.stderr)
            return 1
        if not nx.has_path(routing_graph, src, dst):
            print(f"[ERR] 指定业务不可达: {src} -> {dst}", file=sys.stderr)
            return 1
        tasks.append(task_record(routing_graph, src, dst, idx, rng))
        used_pairs.add((src, dst))

    offset = len(tasks)
    for idx, (src_ip, dst_ip) in enumerate(CUSTOM_FLOW_IPS, start=1):
        src = resolve_node_by_ip(routing_graph, src_ip)
        dst = resolve_node_by_ip(routing_graph, dst_ip)
        if not nx.has_path(routing_graph, src, dst):
            print(f"[ERR] 指定 IP 业务不可达: {src_ip}({src}) -> {dst_ip}({dst})", file=sys.stderr)
            return 1
        tasks.append(task_record(routing_graph, src, dst, offset + idx, rng))
        used_pairs.add((src, dst))

    endpoint_nodes = {node for node in routing_graph.nodes if str(node).startswith("hu")}
    if len(endpoint_nodes) < 2:
        endpoint_nodes = set(routing_graph.nodes)

    candidates = [
        pair for pair in connected_ordered_pairs(routing_graph, endpoint_nodes) if pair not in used_pairs
    ]
    rng.shuffle(candidates)
    if args.count > len(candidates):
        print(f"[WARN] 可用随机业务只有 {len(candidates)} 条，将少于请求数量 {args.count}。")

    for idx, (src, dst) in enumerate(candidates[: args.count], start=1):
        tasks.append(task_record(routing_graph, src, dst, len(tasks) + 1, rng))

    payload = {
        "code": 0,
        "msg": "离线生成业务列表成功",
        "data": {
            "tasks": tasks,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] saved tasks: {args.output}")
    print(f"[INFO] task_count={len(tasks)}")
    for task in tasks[: min(10, len(tasks))]:
        print(
            f"  - {task['task_id']}: "
            f"{task['start']}({task['start_host_ip']}) -> "
            f"{task['end']}({task['end_host_ip']}), hops={len(task['path'])}"
        )
    if len(tasks) > 10:
        print(f"  ... 其余 {len(tasks) - 10} 条见输出文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
