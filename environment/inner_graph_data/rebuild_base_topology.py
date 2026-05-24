#!/usr/bin/env python3
"""拓扑结构变化后重建内场 base GraphML。

默认行为：
  1. 在线获取最新拓扑和链路指标 JSON。
  2. 基于最新拓扑重建 base_ii_topology.graphml。
  3. 用最新拓扑和链路指标回放一次在线图更新，并打印摘要用于检查。

运行方式：
  # 拓扑/链路接口在线时使用。
  python environment/inner_graph_data/rebuild_base_topology.py

  # 拓扑/链路接口不可用时，使用本地 JSON 重建。
  python environment/inner_graph_data/rebuild_base_topology.py --net_offline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment.inner_graph_data import topology_to_networkx as topo  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="重建内场 II 类 base 拓扑 GraphML。")
    parser.add_argument(
        "--net_offline",
        action="store_true",
        help="使用本地拓扑和链路指标 JSON；默认在线获取两类数据。",
    )
    parser.add_argument(
        "--topology-json",
        type=Path,
        default=topo.DEFAULT_INPUT,
        help="拓扑 JSON 路径。",
    )
    parser.add_argument(
        "--link-metric-json",
        type=Path,
        default=topo.DEFAULT_LINK_METRIC_INPUT,
        help="链路指标 JSON 路径。",
    )
    parser.add_argument(
        "--base-graphml",
        type=Path,
        default=topo.DEFAULT_BASE_GRAPHML,
        help="要重建的 base GraphML 路径。",
    )
    parser.add_argument("--timeout", type=float, default=10.0, help="在线请求超时时间（秒）。")
    parser.add_argument("--retries", type=int, default=2, help="在线请求重试次数。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fetch_online = not args.net_offline

    if fetch_online:
        print("[INFO] 在线获取最新拓扑 JSON...")
        topo.fetch_latest_topology_json(
            args.topology_json,
            timeout=args.timeout,
            retries=args.retries,
        )
        print("[INFO] 在线获取最新链路指标 JSON...")
        topo.fetch_latest_link_metrics_json(
            args.link_metric_json,
            timeout=args.timeout,
            retries=args.retries,
        )
    else:
        print("[INFO] 网络数据离线，使用本地拓扑和链路指标 JSON。")

    base_graph = topo.build_base_graph(args.topology_json)
    saved_path = topo.save_base_graphml(base_graph, args.base_graphml)
    print(f"[OK] 已重建 base GraphML: {saved_path}")
    print("[INFO] base 图摘要（所有节点/链路状态应为 0）：")
    print(json.dumps(topo.summarize_graph(base_graph), ensure_ascii=False, indent=2))

    updated_graph = topo.load_or_create_base_graph(
        args.base_graphml,
        args.topology_json,
        rebuild=False,
    )
    topo.update_graph_from_latest_topology(updated_graph, args.topology_json)
    if args.link_metric_json.exists():
        topo.update_graph_from_link_metrics(updated_graph, args.link_metric_json)

    print("[INFO] 使用最新拓扑和链路指标回放更新后的图摘要：")
    print(json.dumps(topo.summarize_graph(updated_graph), ensure_ascii=False, indent=2))
    topo.print_fault_links(updated_graph)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
