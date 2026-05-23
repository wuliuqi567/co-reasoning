#!/usr/bin/env python3
"""QoS 路由示例。

运行方式：
  # 离线模式：使用 environment/inner_graph_data/json-data 下已有的 JSON 数据。
  python inner-rl-reroute_III.py

  # 指定源宿节点 ID。
  python inner-rl-reroute_III.py --src asu0n0 --dst eru1n5

  # 指定源宿 IP，支持节点管理 IP 或端口 IP。
  python inner-rl-reroute_III.py --src-ip 10.104.0.254 --dst-ip 10.103.21.254

  # 在线模式：先获取最新拓扑，再更新图。
  python inner-rl-reroute_III.py --fetch-online

  # 同时在线获取链路指标。
  python inner-rl-reroute_III.py --fetch-online --fetch-link-metrics

参数说明：
  --fetch-online：从拓扑接口获取最新拓扑；不加时使用本地 JSON。
  --fetch-link-metrics：从链路指标接口获取最新链路指标；不加时使用本地 JSON。
  --src/--dst：按节点 ID 指定源宿，必须成对使用。
  --src-ip/--dst-ip：按管理 IP 或端口 IP 指定源宿，必须成对使用。

固定默认值：
  拓扑 JSON：environment/inner_graph_data/json-data/network_topology_state.json
  base 图：environment/inner_graph_data/base_ii_topology.graphml
  链路指标 JSON：environment/inner_graph_data/json-data/link_metric.json
  最小瓶颈可用带宽：100 Mbps
  最大端到端丢包率：0.01
  最多检查候选路径数：20

流程：
  1. 加载或创建全离线 base GraphML。
  2. 用最新拓扑 JSON 更新节点和链路状态；本项目中 status=0 表示离线/故障。
  3. 用 link_metric.json 或链路指标接口更新链路利用率、可用带宽、丢包率等属性。
  4. 只保留在线节点和在线链路构建路由图。
  5. 选择源宿节点。
  6. 生成按 link_latency 排序的候选路径，返回第一条满足带宽和丢包率约束的路径。

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from environment.inner_graph_data import qos_routing as qos
from environment.inner_graph_data import topology_to_networkx as topo


ROOT = Path(__file__).resolve().parent
GRAPH_DIR = ROOT / "environment" / "inner_graph_data"
TOPOLOGY_JSON = GRAPH_DIR / "json-data" / "network_topology_state.json"
BASE_GRAPHML = GRAPH_DIR / "base_ii_topology.graphml"
LINK_METRIC_JSON = GRAPH_DIR / "json-data" / "link_metric.json"
DEFAULT_MIN_BANDWIDTH = 100.0
DEFAULT_MAX_LOSS_RATE = 0.01
DEFAULT_K_PATHS = 20


def node_label(graph, node: str) -> str:
    manage_ip = graph.nodes[node].get("node_manage_ip_addr") or "-"
    return f"{node}({manage_ip})"


def path_with_manage_ips(graph, path: list[str]) -> list[dict[str, str]]:
    return [
        {
            "node_id": node,
            "manage_ip": graph.nodes[node].get("node_manage_ip_addr", ""),
        }
        for node in path
    ]


def path_hop_details(graph, path: list[str]) -> list[dict[str, object]]:
    return qos.path_hop_details(graph, path)


def print_path_details(graph, path: list[str]) -> None:
    def fmt(value: object) -> object:
        return "-" if value is None else value

    print("[PATH] node ids:")
    print("  " + " -> ".join(path))
    print("[PATH] manage IPs:")
    print("  " + " -> ".join(node_label(graph, node) for node in path))
    print("[PATH] hop port details:")
    for idx, hop in enumerate(path_hop_details(graph, path), start=1):
        print(
            f"  {idx}. "
            f"{hop['src_node']}[{hop['src_manage_ip']}] "
            f"{hop['src_port']}({hop['src_port_ip']}) -> "
            f"{hop['dst_node']}[{hop['dst_manage_ip']}] "
            f"{hop['dst_port']}({hop['dst_port_ip']}) | "
            f"latency={hop['link_latency']}ms, bandwidth={hop['link_bandwidth']}Mbps"
            f", utilization={fmt(hop['link_utilization'])}, "
            f"available_bw={fmt(hop['available_bandwidth'])}, "
            f"loss={fmt(hop['link_loss_rate'])}"
        )


def resolve_node_by_ip(graph, ip: str) -> str:
    """Resolve a manage IP or any node port IP to a node id."""
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


def resolve_endpoints(args: argparse.Namespace, graph) -> tuple[str, str]:
    if (args.src or args.dst) and (args.src_ip or args.dst_ip):
        raise ValueError("Use either --src/--dst or --src-ip/--dst-ip, not both.")

    if args.src_ip or args.dst_ip:
        if not args.src_ip or not args.dst_ip:
            raise ValueError("--src-ip and --dst-ip must be provided together.")
        return resolve_node_by_ip(graph, args.src_ip), resolve_node_by_ip(graph, args.dst_ip)

    if args.src or args.dst:
        if not args.src or not args.dst:
            raise ValueError("--src and --dst must be provided together.")
        return args.src, args.dst

    return qos.choose_reachable_pair(graph)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="刷新网络图并执行 QoS 路由。")
    parser.add_argument(
        "--fetch-online",
        action="store_true",
        help="从拓扑接口获取最新拓扑；默认使用本地拓扑 JSON。",
    )
    parser.add_argument(
        "--fetch-link-metrics",
        action="store_true",
        help="从链路指标接口获取最新指标；默认使用本地 link_metric.json。",
    )
    parser.add_argument("--src", help="源节点 ID，需与 --dst 成对使用。")
    parser.add_argument("--dst", help="目的节点 ID，需与 --src 成对使用。")
    parser.add_argument("--src-ip", help="源节点管理 IP 或端口 IP，需与 --dst-ip 成对使用。")
    parser.add_argument("--dst-ip", help="目的节点管理 IP 或端口 IP，需与 --src-ip 成对使用。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    graph = topo.build_updated_graph(
        latest_json=TOPOLOGY_JSON,
        base_graphml=BASE_GRAPHML,
        fetch_latest=args.fetch_online,
        rebuild_base=False,
    )
    if args.fetch_link_metrics:
        topo.fetch_latest_link_metrics_json(LINK_METRIC_JSON)
    if LINK_METRIC_JSON.exists():
        topo.update_graph_from_link_metrics(graph, LINK_METRIC_JSON)

    print(json.dumps(topo.summarize_graph(graph), ensure_ascii=False, indent=2))
    topo.print_fault_links(graph)

    # QoS routing should ignore every offline/fault node or link.
    routing_graph = qos.build_online_routing_graph(graph, offline_status=topo.OFFLINE_STATUS)
    print(
        f"[INFO] routing graph: nodes={routing_graph.number_of_nodes()}, "
        f"edges={routing_graph.number_of_edges()}"
    )
    print(
        "[INFO] QoS defaults: "
        f"min_bandwidth={DEFAULT_MIN_BANDWIDTH}Mbps, "
        f"max_loss_rate={DEFAULT_MAX_LOSS_RATE}, "
        f"k_paths={DEFAULT_K_PATHS}"
    )

    try:
        src, dst = resolve_endpoints(args, routing_graph)
    except ValueError as exc:
        print(f"[ERR] {exc}", file=sys.stderr)
        return 1

    if src not in routing_graph or dst not in routing_graph:
        print(f"[ERR] src or dst is not online: src={src}, dst={dst}", file=sys.stderr)
        return 1

    try:
        route = qos.find_delay_shortest_qos_path(
            routing_graph,
            src,
            dst,
            min_bandwidth=DEFAULT_MIN_BANDWIDTH,
            max_loss_rate=DEFAULT_MAX_LOSS_RATE,
            k_paths=DEFAULT_K_PATHS,
        )
    except qos.NoQoSPath as exc:
        print(f"[ERR] no QoS path: {src} -> {dst}; {exc}", file=sys.stderr)
        return 1

    path = route["path"]
    metrics = route["metrics"]
    routing_graph = route["candidate_graph"]

    print(f"[PATH] {src} -> {dst}")
    print_path_details(routing_graph, path)
    print("[PATH] structured node IP info:")
    print(json.dumps(path_with_manage_ips(routing_graph, path), ensure_ascii=False, indent=2))
    print(
        "[PATH] metrics: "
        f"total_latency={metrics['total_latency']:.3f}ms, "
        f"hops={int(metrics['hop_count'])}, "
        f"bottleneck_available_bandwidth={metrics['bottleneck_bandwidth']:.3f}Mbps, "
        f"path_loss_rate={metrics['path_loss_rate']:.6f}, "
        f"max_link_utilization={metrics['max_link_utilization']:.6f}, "
        f"checked_candidate_paths={route['checked_paths']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
