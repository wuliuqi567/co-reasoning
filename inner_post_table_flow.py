#!/usr/bin/env python3
"""Inner routing flow-table helpers.

This module is dedicated to inner_rl_reroute.py. It keeps post_table_flow.py
unchanged and uses 192.168.1.24 as the default reporting/downlink host.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from urllib.parse import urlsplit, urlunsplit

import requests

from post_table_flow import build_payload, policy_compare


INNER_FLOW_BASE_IP = "192.168.1.24"
DEFAULT_URL = f"http://{INNER_FLOW_BASE_IP}:12590/api/flow/sflowtblCfg"


def _replace_url_ip(base_url: str, target_ip: str) -> str:
    parts = urlsplit(base_url)
    host = parts.hostname or ""
    port = parts.port
    if host == "":
        return base_url
    netloc = f"{target_ip}:{port}" if port else target_ip
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _send_payload(url: str, payload: dict, timeout: float, retries: int, verbose: bool, dry_run: bool = False) -> int:
    headers = {"Content-Type": "application/json"}

    if verbose:
        print("[INFO] Payload JSON:")
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    if dry_run:
        print(f"[DRY-RUN] POST {url}")
        print("[DRY-RUN] 已跳过真实下发")
        return 0

    last_exc = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            print(f"[HTTP] POST {url}")
            print(f"[HTTP] Status: {resp.status_code}")

            if resp.text and resp.text.strip():
                try:
                    print("[HTTP] Response JSON:")
                    print(json.dumps(resp.json(), ensure_ascii=False, indent=2))
                except Exception:
                    print("[HTTP] Response Text:")
                    print(resp.text)

            if 200 <= resp.status_code < 300:
                print("[OK] 下发成功")
                return 0
            print("[ERR] 下发失败（非 2xx）")
            return 2
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                backoff = 0.8 * (2 ** attempt)
                print(f"[WARN] 请求异常：{exc}，{backoff:.1f}s 后重试...", file=sys.stderr)
                time.sleep(backoff)
            else:
                print(f"[ERR] 请求失败（重试耗尽）：{exc}", file=sys.stderr)
                return 1

    print(f"[ERR] Unexpected fallthrough: {last_exc}", file=sys.stderr)
    return 1


def send_flow_table(
    flow_table: list,
    timeout: float,
    retries: int,
    verbose: bool,
    dry_run: bool = False,
    route_by_node_ip: bool = False,
) -> int:
    """Send bidirectional flow-table entries for an inner routing policy.

    By default all requests are sent to 192.168.1.24. Set route_by_node_ip=True
    only when each path item must be posted to its own node management IP.
    """
    if not flow_table:
        print("[ERR] flow_table 为空", file=sys.stderr)
        return 1

    base = flow_table[0]
    nw_src = base.get("src_dev_ip")
    nw_dst = base.get("dst_dev_ip")
    if not nw_src or not nw_dst:
        print("[ERR] flow_table[0] 缺少 src_dev_ip 或 dst_dev_ip", file=sys.stderr)
        return 1

    results = []
    for item in flow_table[1:]:
        target_ip = item.get("ip")
        next_hop_forward = item.get("next_node_ip")
        next_hop_reverse = item.get("in_port_ip")
        if not next_hop_forward:
            print(f"[WARN] 跳过缺失 next_node_ip 的项：{item}", file=sys.stderr)
            continue

        url = _replace_url_ip(DEFAULT_URL, target_ip) if route_by_node_ip and target_ip else DEFAULT_URL

        forward_payload = build_payload(
            nw_src=nw_src,
            nw_dst=nw_dst,
            nextHop=next_hop_forward,
        )
        results.append(_send_payload(url, forward_payload, timeout, retries, verbose, dry_run))

        if not next_hop_reverse:
            print(f"[WARN] 无法下发反向流表（缺少 in_port_ip）：{item}", file=sys.stderr)
            continue

        reverse_payload = build_payload(
            nw_src=nw_dst,
            nw_dst=nw_src,
            nextHop=next_hop_reverse,
        )
        results.append(_send_payload(url, reverse_payload, timeout, retries, verbose, dry_run))

    for code in results:
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="inner 双向流表下发工具")
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP 超时时间（秒）")
    parser.add_argument("--retries", type=int, default=2, help="请求重试次数")
    parser.add_argument("--verbose", action="store_true", help="打印 payload 与响应详情")
    parser.add_argument("--dry-run", action="store_true", help="仅打印将要下发的数据，不发送 HTTP 请求")
    parser.add_argument(
        "--route-by-node-ip",
        action="store_true",
        help="按路径项中的节点 IP 下发；默认统一发往 192.168.1.24。",
    )
    args = parser.parse_args()

    print(f"[INFO] inner flow-table default URL: {DEFAULT_URL}")
