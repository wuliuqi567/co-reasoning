#!/usr/bin/env python3
"""协同重路由编排入口。

流程：
  1. 运行 inner_rl_reroute_II.py，获得 II 类本地最短路径策略。
  2. 参考 rl_reroute_II.py 的上报方式，将 II 策略写入知识库。
  3. 运行 inner_rl_reroute_III.py，获得 III 类 QoS 重路由策略。
  4. 从知识库读取 II 策略。
  5. 参考 rl_reroute.py 的方式比较策略并组织协同推理日志。

默认调用真实知识库接口，并在线获取拓扑和链路指标。
添加 --kg_offline 时使用本地 JSON 模拟知识库。
添加 --net_offline 时使用本地拓扑和链路指标 JSON。
python inner_rl_reroute.py --kg_offline --net_offline

默认源宿节点：asu0n0 -> eru1n5。
默认日志路径：知识库离线写 logs/access.log，知识库在线写 /home/ict/projects/kg_network/semprotocol/log/access.log。
当前脚本不下发真实流表。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from inner_post_table_flow import policy_compare


ROOT = Path(__file__).resolve().parent
II_SCRIPT = ROOT / "inner_rl_reroute_II.py"
III_SCRIPT = ROOT / "inner_rl_reroute_III.py"
DEFAULT_LOG_PATH = ROOT / "logs" / "access.log"
DEFAULT_ONLINE_LOG_PATH = Path("/home/ict/projects/kg_network/semprotocol/log/access.log")
DEFAULT_OFFLINE_KB_PATH = ROOT / "logs" / "offline_ii_policy.json"
II_POLICY_NAME = "co_reasoning_II_1_policy"
DEFAULT_SRC = "asu0n0"
DEFAULT_DST = "eru1n5"

# 或者 python inner_rl_reroute.py --src-ip 10.104.0.254 --dst-ip 10.103.21.254


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="执行 II/III 协同重路由流程，默认知识库和网络数据都在线。")
    parser.add_argument("--kg_offline", action="store_true", help="知识库离线：使用本地 JSON 模拟 II 策略上报/读取。")
    parser.add_argument("--net_offline", action="store_true", help="网络数据离线：使用本地拓扑和链路指标 JSON。")
    parser.add_argument("--src", default=DEFAULT_SRC, help=f"源节点 ID，默认 {DEFAULT_SRC}。")
    parser.add_argument("--dst", default=DEFAULT_DST, help=f"目的节点 ID，默认 {DEFAULT_DST}。")
    parser.add_argument("--src-ip", help="源节点管理 IP 或端口 IP，需与 --dst-ip 成对使用。")
    parser.add_argument("--dst-ip", help="目的节点管理 IP 或端口 IP，需与 --src-ip 成对使用。")
    parser.add_argument("--log-path", type=Path, help="协同推理日志输出路径；不填时离线/在线自动选择默认路径。")
    parser.add_argument("--kg_offline_path", type=Path, default=DEFAULT_OFFLINE_KB_PATH, help="知识库离线模拟 JSON 路径。")
    parser.add_argument("--print-child-output", action="store_true", help="打印 II/III 子脚本完整输出。")
    return parser.parse_args()


def configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    try:
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    except OSError as exc:
        print(f"[WARN] 无法创建文件日志处理器: {exc}", file=sys.stderr)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )


def resolve_log_path(args: argparse.Namespace) -> Path:
    if args.log_path is not None:
        return args.log_path
    if use_online_kb(args):
        return DEFAULT_ONLINE_LOG_PATH
    return DEFAULT_LOG_PATH


def use_online_kb(args: argparse.Namespace) -> bool:
    return not args.kg_offline


def get_time_str() -> str:
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S") + f":{now.microsecond // 1000:03d}"


def build_child_args(args: argparse.Namespace) -> list[str]:
    child_args: list[str] = []
    if args.net_offline:
        child_args.append("--net_offline")
    if args.src_ip or args.dst_ip:
        if not args.src_ip or not args.dst_ip:
            raise ValueError("--src-ip and --dst-ip must be provided together.")
        child_args.extend(["--src-ip", args.src_ip, "--dst-ip", args.dst_ip])
    else:
        child_args.extend(["--src", args.src, "--dst", args.dst])
    return child_args


def run_route_script(script_path: Path, args: argparse.Namespace, label: str) -> tuple[str, int]:
    command = [sys.executable, str(script_path), *build_child_args(args)]
    start = time.time()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    response_time_ms = int((time.time() - start) * 1000)

    if args.print_child_output:
        print(f"\n-------------------------------{label} 脚本输出---------------------------------\n")
        print(completed.stdout)
        if completed.stderr:
            print(completed.stderr, file=sys.stderr)

    if completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout)
        if completed.stderr:
            print(completed.stderr, file=sys.stderr)
        raise RuntimeError(f"{label} 路由脚本执行失败: {script_path}")

    return completed.stdout, response_time_ms


def parse_route_output(output: str, response_time_ms: int, source: str) -> dict[str, Any]:
    path = _extract_structured_path(output)
    metrics = _extract_metrics(output)
    node_path = _extract_node_path(output)

    return {
        "source": source,
        "path": path,
        "node_path": node_path,
        "delay": metrics.get("total_latency"),
        "bandwidth": metrics.get("bottleneck_available_bandwidth"),
        "loss_rate": metrics.get("path_loss_rate"),
        "max_link_utilization": metrics.get("max_link_utilization"),
        "hop_num": int(metrics.get("hops", len(path) - 1 if path else 0)),
        "checked_candidate_paths": metrics.get("checked_candidate_paths"),
        "response_time": response_time_ms,
    }


def _extract_structured_path(output: str) -> list[dict[str, Any]]:
    marker = "[PATH] structured node IP info:"
    start = output.find(marker)
    if start == -1:
        return []

    after_marker = output[start + len(marker):]
    next_marker = after_marker.find("\n[PATH] metrics:")
    json_text = after_marker[:next_marker if next_marker != -1 else None].strip()
    if not json_text:
        return []
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _extract_metrics(output: str) -> dict[str, float]:
    metrics_line = ""
    for line in output.splitlines():
        if line.startswith("[PATH] metrics:"):
            metrics_line = line
            break
    if not metrics_line:
        return {}

    metrics: dict[str, float] = {}
    for key, value in re.findall(r"([a-zA-Z_]+)=([0-9.]+)", metrics_line):
        parsed_value = float(value)
        metrics[key] = int(parsed_value) if key in {"hops", "checked_candidate_paths"} else parsed_value
    return metrics


def _extract_node_path(output: str) -> list[str]:
    for idx, line in enumerate(output.splitlines()):
        if line.startswith("[PATH] node ids:"):
            lines = output.splitlines()
            if idx + 1 < len(lines):
                return [item.strip() for item in lines[idx + 1].split("->") if item.strip()]
    return []


def make_ii_log(time1: str, time2: str, time3: str) -> str:
    content1 = "II类运行本地重路由模型"
    content2 = "生成本地重路由策略，并下发给II类智能体执行"
    content3 = "将本地重路由策略更新到本地知识单元，并通过上报给III类知识单元"
    return (
        f"time1={time1}, content1={content1}; "
        f"time2={time2}, content2={content2}; "
        f"time3={time3}, content3={content3}"
    )


def post_ii_policy_offline(policy: dict[str, Any], kb_path: Path) -> None:
    kb_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": II_POLICY_NAME,
        "domain": "网络状态域",
        "meaning": "II类本地重路由策略上报给III类知识单元",
        "update_time": datetime.now().isoformat(timespec="milliseconds"),
        "output_decision": policy,
    }
    kb_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def get_ii_policy_offline(kb_path: Path) -> dict[str, Any]:
    data = json.loads(kb_path.read_text(encoding="utf-8"))
    policy = data.get("output_decision", {})
    return policy if isinstance(policy, dict) else {}


def post_ii_policy(policy: dict[str, Any], args: argparse.Namespace) -> None:
    if use_online_kb(args):
        from inner_post_II_info import post_II_info

        post_II_info(policy)
        print(f"[INFO] II 策略已上报真实知识库: {II_POLICY_NAME}")
        return

    post_ii_policy_offline(policy, args.kg_offline_path)
    print(f"[INFO] II 策略已写入离线知识库: {args.kg_offline_path}")


def get_ii_policy(args: argparse.Namespace) -> dict[str, Any]:
    if use_online_kb(args):
        from inner_post_II_info import get_II_info

        print(f"[INFO] 从真实知识库读取 II 策略: {II_POLICY_NAME}")
        return get_II_info(II_POLICY_NAME)

    print(f"[INFO] 从离线知识库读取 II 策略: {args.kg_offline_path}")
    return get_ii_policy_offline(args.kg_offline_path)


def build_result_string(local_policy: dict[str, Any], global_policy: dict[str, Any]) -> str:
    result1 = f"[{local_policy.get('delay', 'N/A')} {global_policy.get('delay', 'N/A')}"
    local_path = local_policy.get("path", [])
    global_path = global_policy.get("path", [])
    result2 = f"{len(local_path) if isinstance(local_path, list) else '0'} {len(global_path) if isinstance(global_path, list) else '0'}"
    result3 = f"{local_policy.get('bandwidth', '0')} {global_policy.get('bandwidth', '0')}"
    result4 = f"{local_policy.get('response_time', '0')} {global_policy.get('response_time', '0')}]"
    return result1 + " " + result2 + " " + result3 + " " + result4


def main() -> int:
    args = parse_args()
    log_path = resolve_log_path(args)
    configure_logging(log_path)
    kg_mode = "知识库在线" if use_online_kb(args) else "知识库离线"
    net_mode = "网络数据离线" if args.net_offline else "网络数据在线"
    print(f"[INFO] 运行模式: {kg_mode}, {net_mode}")
    print(f"[INFO] 协同推理日志路径: {log_path}")

    content4 = "III类检测到某II类节点/链路失效，III类触发协同推理功能"
    content5 = "查询网络状态知识"
    content6 = "运行全局重路由模型"
    content7 = "推理生成全局重路由策略"
    content8 = "获取本地重路由策略并执行协同优化机制"
    content9 = "下发全局重路由策略,并交由II类智能体执行"
    content10 = "协同推理结束"

    time1 = get_time_str()
    ii_output, ii_response_time = run_route_script(II_SCRIPT, args, "II")
    time2 = get_time_str()
    local_policy = parse_route_output(ii_output, ii_response_time, "II")
    time3 = get_time_str()
    local_policy["log_II_info"] = make_ii_log(time1, time2, time3)

    print("-------------------------------II 本地路由策略---------------------------------\n")
    print("local_policy", json.dumps(local_policy, ensure_ascii=False, indent=2))
    print("\n")

    post_ii_policy(local_policy, args)

    time4 = get_time_str()
    time5 = get_time_str()
    time6 = get_time_str()
    iii_output, iii_response_time = run_route_script(III_SCRIPT, args, "III")
    time7 = get_time_str()
    global_policy = parse_route_output(iii_output, iii_response_time, "III")

    print("-------------------------------III 全局重路由策略---------------------------------\n")
    print("global_policy", json.dumps(global_policy, ensure_ascii=False, indent=2))
    print("\n")

    local_policy = get_ii_policy(args)
    kb_source = "真实知识库" if use_online_kb(args) else "离线知识库"
    print(f"-------------------------------从{kb_source}获取 II policy---------------------------------\n")
    print("local_policy", json.dumps(local_policy, ensure_ascii=False, indent=2))
    print("\n")

    final_policy = policy_compare(global_policy, local_policy)
    if final_policy is None:
        print("[WARN] policy_compare 返回 None，使用 III 全局策略")
        final_policy = global_policy

    time8 = get_time_str()
    time9 = get_time_str()
    time10 = get_time_str()

    result = build_result_string(local_policy, global_policy)
    print(f"性能指标结果: {result}")
    print("-------------------------------最终协同策略---------------------------------\n")
    print("final_policy", json.dumps(final_policy, ensure_ascii=False, indent=2))
    print("\n")

    local_policy_log = str(local_policy.get("log_II_info") or make_ii_log(time1, time2, time3)).strip()
    log_prefix = f"[collaborative_reasoning] {local_policy_log}"
    full_log = (
        f"{log_prefix}; "
        f"time4={time4}, content4={content4}; "
        f"time5={time5}, content5={content5}; "
        f"time6={time6}, content6={content6}; "
        f"time7={time7}, content7={content7}; "
        f"time8={time8}, content8={content8}; "
        f"time9={time9}, content9={content9}; "
        f"time10={time10}, content10={content10}; "
        f"result={result}; status=1; cor_node=II_node_2 III_node_1"
    )

    logging.info("%s", full_log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
