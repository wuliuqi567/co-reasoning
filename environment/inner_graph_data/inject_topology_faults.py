#!/usr/bin/env python3
"""自定义注入节点或链路故障到拓扑 JSON。

默认不会覆盖原始拓扑文件，而是生成一份新的故障拓扑 JSON。

运行方式：
  # 先在脚本顶部的 FAULT_NODES / FAULT_LINK_IDS / FAULT_LINK_ENDPOINTS 中设置故障。
  python environment/inner_graph_data/inject_topology_faults.py

  # 生成后配合自动检测脚本使用。
  python auto_inner_reroute.py --kg_offline --net_offline --once \
    --topology-json environment/inner_graph_data/json-data/network_topology_state_fault.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(__file__).resolve().parent / "json-data" / "network_topology_state.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "json-data" / "network_topology_state_fault.json"
OFFLINE_STATUS = 0
ONLINE_STATUS = 1

# 在这里直接设置节点故障。
# 示例：FAULT_NODES = ["asu0n0", "eru1n5"]
FAULT_NODES: list[str] = []

# 在这里直接设置链路故障，使用 link_id。
# 示例：FAULT_LINK_IDS = ["asu0n0:3_bsu0n0:3"]
FAULT_LINK_IDS: list[str] = []

# 在这里直接设置链路故障，使用两端节点；会匹配两节点之间的所有链路。
# 示例：FAULT_LINK_ENDPOINTS = [("asu0n0", "bsu0n0")]
FAULT_LINK_ENDPOINTS: list[tuple[str, str]] = [("asu0n0", "bsu0n0")]

# 节点故障时，是否同时把与该节点相连的链路置为故障。
FAIL_INCIDENT_LINKS = True

# 注入前是否先把所有节点、端口和链路状态置为在线。
CLEAR_EXISTING = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="向拓扑 JSON 注入自定义节点/链路故障。")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="输入拓扑 JSON。")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出故障拓扑 JSON。")
    parser.add_argument("--in-place", action="store_true", help="直接覆盖输入文件。")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def extract_topology(data: dict[str, Any]) -> dict[str, Any]:
    payload = data.get("data", data)
    if not isinstance(payload, dict):
        return {}
    topo = payload.get("topo") or payload.get("topology") or data.get("topo")
    return topo if isinstance(topo, dict) else {}


def topology_items(topo: dict[str, Any], singular: str, plural: str) -> list[dict[str, Any]]:
    items = topo.get(singular)
    if items is None:
        items = topo.get(plural)
    return items if isinstance(items, list) else []


def link_endpoints(link: dict[str, Any]) -> tuple[str, str]:
    src = link.get("src") if isinstance(link.get("src"), dict) else {}
    dst = link.get("dst") if isinstance(link.get("dst"), dict) else {}
    src_node = src.get("src_node") or link.get("src_node") or ""
    dst_node = dst.get("dst_node") or link.get("dst_node") or ""
    if (not src_node or not dst_node) and link.get("link_id"):
        parts = str(link["link_id"]).split("_")
        if len(parts) == 2:
            src_node = parts[0].split(":")[0]
            dst_node = parts[1].split(":")[0]
    return src_node, dst_node


def set_node_status(node: dict[str, Any], status: int) -> None:
    node["node_status"] = status
    ports = node.get("node_ports")
    if isinstance(ports, list):
        for port in ports:
            if isinstance(port, dict):
                port["status"] = status


def set_all_online(nodes: list[dict[str, Any]], links: list[dict[str, Any]]) -> None:
    for node in nodes:
        set_node_status(node, ONLINE_STATUS)
    for link in links:
        link["link_status"] = ONLINE_STATUS


def inject_node_faults(
    nodes: list[dict[str, Any]],
    links: list[dict[str, Any]],
    node_ids: list[str],
    *,
    fail_incident_links: bool,
) -> tuple[int, list[str]]:
    targets = set(node_ids)
    updated = 0
    missing = []

    for node_id in targets:
        matched = False
        for node in nodes:
            if node.get("node_id") != node_id:
                continue
            set_node_status(node, OFFLINE_STATUS)
            matched = True
            updated += 1
            break
        if not matched:
            missing.append(node_id)

    if fail_incident_links and targets:
        for link in links:
            src_node, dst_node = link_endpoints(link)
            if src_node in targets or dst_node in targets:
                link["link_status"] = OFFLINE_STATUS

    return updated, missing


def inject_link_faults_by_id(links: list[dict[str, Any]], link_ids: list[str]) -> tuple[int, list[str]]:
    targets = set(link_ids)
    updated = 0
    missing = []

    for link_id in targets:
        matched = False
        for link in links:
            if link.get("link_id") != link_id:
                continue
            link["link_status"] = OFFLINE_STATUS
            matched = True
            updated += 1
        if not matched:
            missing.append(link_id)

    return updated, missing


def inject_link_faults_by_endpoints(links: list[dict[str, Any]], endpoint_pairs: list[tuple[str, str]]) -> tuple[int, list[str]]:
    updated = 0
    missing = []

    for src, dst in endpoint_pairs:
        target = {src, dst}
        matched = False
        for link in links:
            link_src, link_dst = link_endpoints(link)
            if {link_src, link_dst} != target:
                continue
            link["link_status"] = OFFLINE_STATUS
            matched = True
            updated += 1
        if not matched:
            missing.append(f"{src}<->{dst}")

    return updated, missing


def update_summary(data: dict[str, Any], nodes: list[dict[str, Any]], links: list[dict[str, Any]]) -> None:
    summary = data.get("summary")
    if not isinstance(summary, dict):
        summary = {}
        data["summary"] = summary

    summary["node_count"] = len(nodes)
    summary["link_count"] = len(links)
    summary["fault_node_count"] = sum(1 for node in nodes if int(node.get("node_status", 0)) == OFFLINE_STATUS)
    summary["fault_link_count"] = sum(1 for link in links if int(link.get("link_status", 0)) == OFFLINE_STATUS)
    summary["timestamp"] = datetime.now().astimezone().isoformat()


def main() -> int:
    args = parse_args()
    output_path = args.input if args.in_place else args.output

    data = load_json(args.input)
    topo = extract_topology(data)
    nodes = topology_items(topo, "node", "nodes")
    links = topology_items(topo, "link", "links")
    if not nodes and not links:
        print(f"[ERR] 拓扑为空，无法注入故障: {args.input}", file=sys.stderr)
        return 1

    if CLEAR_EXISTING:
        set_all_online(nodes, links)

    node_updated, missing_nodes = inject_node_faults(
        nodes,
        links,
        FAULT_NODES,
        fail_incident_links=FAIL_INCIDENT_LINKS,
    )
    link_updated, missing_links = inject_link_faults_by_id(links, FAULT_LINK_IDS)
    endpoint_updated, missing_endpoint_links = inject_link_faults_by_endpoints(
        links,
        FAULT_LINK_ENDPOINTS,
    )
    update_summary(data, nodes, links)
    save_json(data, output_path)

    print(f"[OK] saved fault topology: {output_path}")
    print(
        "[INFO] injected: "
        f"nodes={node_updated}, links_by_id={link_updated}, links_by_endpoints={endpoint_updated}"
    )
    if missing_nodes:
        print(f"[WARN] 未找到节点: {missing_nodes}", file=sys.stderr)
    if missing_links:
        print(f"[WARN] 未找到 link_id: {missing_links}", file=sys.stderr)
    if missing_endpoint_links:
        print(f"[WARN] 未找到端点链路: {missing_endpoint_links}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
