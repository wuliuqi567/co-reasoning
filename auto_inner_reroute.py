#!/usr/bin/env python3
"""自动检测内场业务受故障影响并触发重路由。

运行方式：
  # 默认每 5 秒在线获取拓扑/链路状态，知识库在线。
  python auto_inner_reroute.py

  # 后台运行，关闭终端不影响；输出写入日志。
  nohup python auto_inner_reroute.py --kg_offline > logs/auto_inner_reroute.log 2>&1 &

  # 当前知识库不可用但网络数据在线。
  python auto_inner_reroute.py --kg_offline

  # 本地验证：使用本地拓扑/链路 JSON，只检查一次。
  python auto_inner_reroute.py --kg_offline --net_offline --once
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx

from environment.inner_graph_data import topology_to_networkx as topo
from generate_inner_business_flows import (
    DEFAULT_BASE_GRAPHML,
    DEFAULT_LINK_METRIC_JSON,
    DEFAULT_OUTPUT,
    DEFAULT_TOPOLOGY_JSON,
    load_routing_graph,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_REROUTE_SCRIPT = ROOT / "inner_rl_reroute.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="自动检测内场业务故障影响并触发重路由。")
    parser.add_argument("--flows", type=Path, default=DEFAULT_OUTPUT, help="业务流 JSON 文件。")
    parser.add_argument("--interval", type=float, default=5.0, help="检测间隔秒数，默认 5 秒。")
    parser.add_argument("--once", action="store_true", help="只检测一次后退出。")
    parser.add_argument("--max-iterations", type=int, help="最多检测轮数；不填则持续运行。")
    parser.add_argument("--kg_offline", action="store_true", help="调用 inner_rl_reroute.py 时使用离线知识库。")
    parser.add_argument("--net_offline", action="store_true", help="使用本地拓扑和链路指标 JSON；默认在线获取。")
    parser.add_argument("--topology-json", type=Path, default=DEFAULT_TOPOLOGY_JSON, help="拓扑 JSON 路径。")
    parser.add_argument("--link-metric-json", type=Path, default=DEFAULT_LINK_METRIC_JSON, help="链路指标 JSON 路径。")
    parser.add_argument("--base-graphml", type=Path, default=DEFAULT_BASE_GRAPHML, help="base GraphML 路径。")
    parser.add_argument("--reroute-script", type=Path, default=DEFAULT_REROUTE_SCRIPT, help="重路由脚本路径。")
    parser.add_argument("--print-child-output", action="store_true", help="打印 inner_rl_reroute.py 完整输出。")
    return parser.parse_args()


def load_flows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"业务流文件不存在: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    flows = data.get("flows", [])
    if not isinstance(flows, list):
        raise ValueError(f"业务流文件格式错误，缺少 flows 列表: {path}")
    return [flow for flow in flows if isinstance(flow, dict)]


def fault_nodes(graph: nx.Graph) -> list[str]:
    return [
        node
        for node, attrs in graph.nodes(data=True)
        if int(attrs.get("node_status", topo.OFFLINE_STATUS)) == topo.OFFLINE_STATUS
    ]


def flow_failure_reasons(flow: dict[str, Any], routing_graph: nx.Graph) -> list[str]:
    path_nodes = flow.get("path_nodes", [])
    if not isinstance(path_nodes, list) or not path_nodes:
        return ["flow has no saved path_nodes"]

    reasons = []
    for node in path_nodes:
        if node not in routing_graph:
            reasons.append(f"node offline: {node}")

    for src, dst in zip(path_nodes, path_nodes[1:]):
        if src not in routing_graph or dst not in routing_graph:
            continue
        if not routing_graph.has_edge(src, dst):
            reasons.append(f"link offline: {src} -> {dst}")

    return reasons


def affected_flows(flows: list[dict[str, Any]], routing_graph: nx.Graph) -> list[tuple[dict[str, Any], list[str]]]:
    affected = []
    for flow in flows:
        reasons = flow_failure_reasons(flow, routing_graph)
        if reasons:
            affected.append((flow, reasons))
    return affected


def run_reroute(flow: dict[str, Any], args: argparse.Namespace) -> tuple[int, str, str]:
    src = flow.get("src")
    dst = flow.get("dst")
    if not src or not dst:
        return 1, "", "flow missing src or dst"

    command = [sys.executable, str(args.reroute_script), "--src", str(src), "--dst", str(dst)]
    if args.kg_offline:
        command.append("--kg_offline")
    if args.net_offline:
        command.append("--net_offline")

    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if args.print_child_output:
        print(completed.stdout)
        if completed.stderr:
            print(completed.stderr, file=sys.stderr)
    return completed.returncode, completed.stdout, completed.stderr


def extract_final_policy_path(output: str) -> list[dict[str, Any]]:
    marker = "final_policy"
    marker_idx = output.find(marker)
    if marker_idx == -1:
        return []

    json_start = output.find("{", marker_idx)
    if json_start == -1:
        return []

    try:
        policy, _ = json.JSONDecoder().raw_decode(output[json_start:])
    except json.JSONDecodeError:
        return []

    path = policy.get("path", []) if isinstance(policy, dict) else []
    return path if isinstance(path, list) else []


def format_policy_path(path: list[dict[str, Any]]) -> str:
    parts = []
    for item in path:
        if not isinstance(item, dict):
            continue
        node_id = item.get("node_id", "")
        ip = item.get("manage_ip", "") or "-"
        if node_id:
            parts.append(f"{node_id}（{ip}）")
    return " -> ".join(parts)


def print_fault_summary(graph: nx.Graph) -> None:
    nodes = fault_nodes(graph)
    links = topo.get_fault_links(graph)
    print(f"[INFO] fault_nodes={len(nodes)}, fault_links={len(links)}")
    for node in nodes[:10]:
        print(f"  - node offline: {node}")
    for link in links[:10]:
        print(f"  - link offline: {link['link_id']} | {link['src_node']} -> {link['dst_node']}")
    if len(nodes) > 10 or len(links) > 10:
        print("  ... 故障项较多，仅显示前 10 个节点和前 10 条链路")


def check_once(args: argparse.Namespace, flows: list[dict[str, Any]]) -> None:
    print(f"\n[{datetime.now().isoformat(timespec='seconds')}] 获取拓扑和链路状态...")
    graph, routing_graph = load_routing_graph(args)
    print_fault_summary(graph)

    if not fault_nodes(graph) and not topo.get_fault_links(graph):
        print("[INFO] 未检测到节点或链路故障。")
        return

    impacted = affected_flows(flows, routing_graph)
    if not impacted:
        print("[INFO] 检测到故障，但当前业务未受影响。")
        return

    print(f"[WARN] 受影响业务数: {len(impacted)}")
    for flow, reasons in impacted:
        print("\n-------------------------------受影响业务---------------------------------")
        print(f"flow_id={flow.get('flow_id')}, src={flow.get('src')}, dst={flow.get('dst')}")
        print(f"原路径: {flow.get('path_info', '')}")
        print("影响原因:")
        for reason in reasons:
            print(f"  - {reason}")

        print(f"[INFO] 执行重路由: {flow.get('src')} -> {flow.get('dst')}")
        code, stdout, stderr = run_reroute(flow, args)
        if code != 0:
            print(f"[ERR] 重路由失败: flow_id={flow.get('flow_id')}, code={code}")
            if stderr:
                print(stderr, file=sys.stderr)
            continue

        new_path = extract_final_policy_path(stdout)
        new_path_text = format_policy_path(new_path)
        if new_path_text:
            print(f"重路由后路径: {new_path_text}")
        else:
            print("[WARN] 未能解析重路由后的最终路径。")


def main() -> int:
    args = parse_args()
    try:
        flows = load_flows(args.flows)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[ERR] {exc}", file=sys.stderr)
        return 1

    print(f"[INFO] loaded flows: {args.flows}, count={len(flows)}")
    iteration = 0
    while True:
        iteration += 1
        check_once(args, flows)
        if args.once:
            break
        if args.max_iterations is not None and iteration >= args.max_iterations:
            break
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
