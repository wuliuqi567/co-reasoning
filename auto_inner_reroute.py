#!/usr/bin/env python3
"""自动检测内场业务受故障影响并触发重路由。

运行方式：
  # 默认每 5 秒在线获取拓扑、链路状态和业务列表，知识库在线。
  python auto_inner_reroute.py

  # 后台运行，关闭终端不影响；输出写入日志。
  nohup conda run -n co-reasoning python auto_inner_reroute.py > logs/auto_inner_reroute.log 2>&1 &

  # 当前知识库不可用但网络数据在线。
  python auto_inner_reroute.py --kg_offline

  # 本地验证：使用本地拓扑/链路 JSON，只检查一次。
  python auto_inner_reroute.py --kg_offline --net_offline --once
"""

from __future__ import annotations

import argparse
import importlib.util
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
DEFAULT_TASK_JSON = ROOT / "environment" / "inner_graph_data" / "json-data" / "task_all.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="自动检测内场业务故障影响并触发重路由。")
    parser.add_argument(
        "--tasks",
        "--flows",
        dest="tasks",
        type=Path,
        help="业务 JSON 文件；在线默认 task_all.json，离线默认 inner_business_flows.json。",
    )
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


def resolve_tasks_path(args: argparse.Namespace) -> Path:
    if args.tasks is not None:
        return args.tasks
    return DEFAULT_OUTPUT if args.net_offline else DEFAULT_TASK_JSON


def fetch_latest_tasks_json(output_path: Path) -> None:
    module_path = ROOT / "environment" / "inner_graph_data" / "get-task-data.py"
    spec = importlib.util.spec_from_file_location("get_task_data", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load task fetcher: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = module.fetch_task_data(module.DEFAULT_URL, timeout=10.0, retries=2)
    module.validate_api_result(data)
    module.save_json(data, output_path)


def split_link_nodes(link_id: str) -> tuple[str, str] | None:
    parts = str(link_id).split("_", 1)
    if len(parts) != 2:
        return None
    return parts[0].split(":", 1)[0], parts[1].split(":", 1)[0]


def path_nodes_from_links(path_links: list[str]) -> list[str]:
    nodes: list[str] = []
    for link_id in path_links:
        endpoints = split_link_nodes(link_id)
        if endpoints is None:
            continue
        src, dst = endpoints
        if not nodes:
            nodes.extend([src, dst])
        elif nodes[-1] == src:
            nodes.append(dst)
        elif nodes[-1] == dst:
            nodes.append(src)
        else:
            nodes.extend([src, dst])
    return nodes


def normalize_task(raw: dict[str, Any]) -> dict[str, Any]:
    task = dict(raw)

    # 兼容旧版 generate_inner_business_flows.py 输出的 flows[].path_nodes。
    if "task_id" not in task and task.get("flow_id"):
        task["task_id"] = task.get("flow_id")
    if "start" not in task and task.get("src"):
        task["start"] = task.get("src")
    if "end" not in task and task.get("dst"):
        task["end"] = task.get("dst")
    if "path" not in task and isinstance(task.get("path_nodes"), list):
        task["path"] = [
            f"{src}_{dst}" for src, dst in zip(task["path_nodes"], task["path_nodes"][1:])
        ]

    path_links = task.get("path") if isinstance(task.get("path"), list) else []
    if "path_nodes" not in task:
        task["path_nodes"] = path_nodes_from_links([str(link) for link in path_links])

    if "src" not in task and task.get("start"):
        task["src"] = task.get("start")
    if "dst" not in task and task.get("end"):
        task["dst"] = task.get("end")

    return task


def load_tasks(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"业务 JSON 文件不存在: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))

    tasks = []
    payload = data.get("data") if isinstance(data, dict) else None
    if isinstance(payload, dict) and isinstance(payload.get("tasks"), list):
        tasks = payload["tasks"]
    elif isinstance(data, dict) and isinstance(data.get("tasks"), list):
        tasks = data["tasks"]
    elif isinstance(data, dict) and isinstance(data.get("flows"), list):
        tasks = data["flows"]

    if not isinstance(tasks, list):
        raise ValueError(f"业务 JSON 格式错误，缺少 data.tasks/tasks/flows 列表: {path}")
    return [normalize_task(task) for task in tasks if isinstance(task, dict)]


def fault_nodes(graph: nx.Graph) -> list[str]:
    return [
        node
        for node, attrs in graph.nodes(data=True)
        if int(attrs.get("node_status", topo.OFFLINE_STATUS)) == topo.OFFLINE_STATUS
    ]


def task_failure_reasons(task: dict[str, Any], routing_graph: nx.Graph) -> list[str]:
    path_nodes = task.get("path_nodes", [])
    if not isinstance(path_nodes, list) or not path_nodes:
        return ["task has no saved path nodes"]

    reasons = []
    for node in path_nodes:
        if node not in routing_graph:
            reasons.append(f"node offline: {node}")

    path_links = task.get("path") if isinstance(task.get("path"), list) else []
    if path_links:
        link_pairs = [split_link_nodes(str(link_id)) for link_id in path_links]
        pairs = [pair for pair in link_pairs if pair is not None]
    else:
        pairs = list(zip(path_nodes, path_nodes[1:]))

    for src, dst in pairs:
        if src not in routing_graph or dst not in routing_graph:
            continue
        if not routing_graph.has_edge(src, dst):
            reasons.append(f"link offline: {src} -> {dst}")

    return reasons


def affected_tasks(tasks: list[dict[str, Any]], routing_graph: nx.Graph) -> list[tuple[dict[str, Any], list[str]]]:
    affected = []
    for task in tasks:
        reasons = task_failure_reasons(task, routing_graph)
        if reasons:
            affected.append((task, reasons))
    return affected


def reroute_endpoints(task: dict[str, Any]) -> tuple[str | None, str | None]:
    src = task.get("start") or task.get("src")
    dst = task.get("end") or task.get("dst")
    path_nodes = task.get("path_nodes", [])

    # 业务接口的 start/end 经常是 hu 网关节点；当前内场路由脚本计算 II 类图，
    # 因此重路由时使用 hu 接入后的第一个在线网络节点。
    if isinstance(path_nodes, list) and len(path_nodes) >= 2:
        if src and str(src).startswith("hu") and path_nodes[0] == src:
            src = path_nodes[1]
        if dst and str(dst).startswith("hu") and path_nodes[-1] == dst:
            dst = path_nodes[-2]

    return str(src) if src else None, str(dst) if dst else None


def run_reroute(task: dict[str, Any], args: argparse.Namespace) -> tuple[int, str, str]:
    src, dst = reroute_endpoints(task)
    if not src or not dst:
        return 1, "", "task missing start/end"

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


def format_task_path(graph: nx.Graph, task: dict[str, Any]) -> str:
    path_nodes = task.get("path_nodes", [])
    if not isinstance(path_nodes, list) or not path_nodes:
        path_links = task.get("path", [])
        return " -> ".join(str(link) for link in path_links) if isinstance(path_links, list) else ""

    parts = []
    for node in path_nodes:
        manage_ip = graph.nodes[node].get("node_manage_ip_addr", "-") if node in graph else "-"
        parts.append(f"{node}（{manage_ip or '-'}）")
    return " -> ".join(parts)


def load_tasks_for_iteration(args: argparse.Namespace) -> tuple[list[dict[str, Any]], Path]:
    tasks_path = resolve_tasks_path(args)
    if not args.net_offline:
        fetch_latest_tasks_json(tasks_path)
    return load_tasks(tasks_path), tasks_path


def check_once(args: argparse.Namespace) -> None:
    print(f"\n[{datetime.now().isoformat(timespec='seconds')}] 获取拓扑、链路状态和业务列表...")
    graph, routing_graph = load_routing_graph(args)
    tasks, tasks_path = load_tasks_for_iteration(args)
    print(f"[INFO] loaded tasks: {tasks_path}, count={len(tasks)}")
    print_fault_summary(graph)

    if not fault_nodes(graph) and not topo.get_fault_links(graph):
        print("[INFO] 未检测到节点或链路故障。")
        return

    impacted = affected_tasks(tasks, routing_graph)
    if not impacted:
        print("[INFO] 检测到故障，但当前业务未受影响。")
        return

    print(f"[WARN] 受影响业务数: {len(impacted)}")
    for task, reasons in impacted:
        print("\n-------------------------------受影响业务---------------------------------")
        print(
            f"task_id={task.get('task_id')}, "
            f"start={task.get('start')}({task.get('start_host_ip', '-')}) -> "
            f"end={task.get('end')}({task.get('end_host_ip', '-')})"
        )
        print(f"原路径: {format_task_path(graph, task)}")
        print("影响原因:")
        for reason in reasons:
            print(f"  - {reason}")

        reroute_src, reroute_dst = reroute_endpoints(task)
        print(f"[INFO] 执行重路由: {reroute_src} -> {reroute_dst}")
        code, stdout, stderr = run_reroute(task, args)
        if code != 0:
            print(f"[ERR] 重路由失败: task_id={task.get('task_id')}, code={code}")
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

    print(
        "[INFO] task mode: "
        f"{'offline_json' if args.net_offline else 'online_api'}, "
        f"path={resolve_tasks_path(args)}"
    )
    iteration = 0
    while True:
        iteration += 1
        try:
            check_once(args)
        except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
            print(f"[ERR] {exc}", file=sys.stderr)
            return 1
        if args.once:
            break
        if args.max_iterations is not None and iteration >= args.max_iterations:
            break
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
