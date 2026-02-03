#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
import time
from urllib.parse import urlsplit, urlunsplit

import requests

DEFAULT_URL = "http://192.168.2.26:12590/api/flow/sflowtblCfg"


def build_payload(nw_src: str, nw_dst: str, nextHop: str) -> dict:
    return {
        "flowtable": [
            {
                "op_cmd": "upsert",
                "id": "flow-00012500166530323400004400000000-00012500166530323400004400000000:192.168.20.100:8554_00012500163431326600004400000000:192.168.40.100:8000-00012500163431326600004400000000",
                "idle_timeout": 30,
                "hard_timeout": 3600,
                "table_id": 0,
                "priority": 5000,
                "match": {
                    "nw_src": nw_src,
                    "nw_dst": nw_dst,
                    "ethernet_match": {
                        "ethernet_type": {
                            "type": 2048
                        }
                    }
                },
                "instructions": {
                    "instruction": [
                        {
                            "order": 1,
                            "apply_actions": {
                                "action": [
                                    {
                                        "order": 0,
                                        "output": "8000",
                                        "nextHop": nextHop
                                    }
                                ]
                            }
                        }
                    ]
                }
            }
        ],
        "origin": 2
    }


def _replace_url_ip(base_url: str, target_ip: str) -> str:
    parts = urlsplit(base_url)
    host = parts.hostname or ""
    port = parts.port

    if host == "":
        return base_url

    netloc = f"{target_ip}:{port}" if port else target_ip
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))

def _send_payload(url: str, payload: dict, timeout: float, retries: int, verbose: bool) -> int:
    headers = {"Content-Type": "application/json"}

    if verbose:
        print("[INFO] Payload JSON:")
        print(json.dumps(payload, ensure_ascii=False, indent=2))

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
            else:
                print("[ERR] 下发失败（非 2xx）")
                return 2

        except Exception as e:
            last_exc = e
            if attempt < retries:
                backoff = 0.8 * (2 ** attempt)
                print(f"[WARN] 请求异常：{e}，{backoff:.1f}s 后重试...", file=sys.stderr)
                time.sleep(backoff)
            else:
                print(f"[ERR] 请求失败（重试耗尽）：{e}", file=sys.stderr)
                return 1

    print(f"[ERR] Unexpected fallthrough: {last_exc}", file=sys.stderr)
    return 1


def send(url: str, timeout: float, retries: int, verbose: bool) -> int:
    payload = build_payload(
        nw_src="192.168.20.100/29",
        nw_dst="192.168.40.100/29",
        nextHop="192.168.22.2",
    )
    return _send_payload(url, payload, timeout, retries, verbose)


def send_flow_table(flow_table: list, timeout: float, retries: int, verbose: bool) -> int:
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
        nextHop = item.get("next_node_ip")
        if not target_ip or not nextHop:
            print(f"[WARN] 跳过缺失字段的项：{item}", file=sys.stderr)
            continue

        url = _replace_url_ip(DEFAULT_URL, target_ip)
        payload = build_payload(nw_src=nw_src, nw_dst=nw_dst, nextHop=nextHop)
        results.append(_send_payload(url, payload, timeout, retries, verbose))

    for code in results:
        if code != 0:
            return code
    return 0

flow_table = [
  {
    "src_dev_ip": "192.168.10.2/24",
    "dst_dev_ip": "192.168.40.2/24"
  },
  {
    "node_idx": 15,
    "ip": "192.168.2.10",
    "in_port": "00012500163030653a00004400000000:1",
    "in_port_ip": "192.168.10.1",
    "out_port": "00012500163030653a00004400000000:4",
    "out_port_ip": "192.168.13.1",
    "next_node_ip": "192.168.13.2"
  },
  {
    "node_idx": 16,
    "ip": "192.168.2.26",
    "in_port": "00012500163431326600004400000000:2",
    "in_port_ip": "192.168.13.2",
    "out_port": "00012500163431326600004400000000:1",
    "out_port_ip": "192.168.40.1",
    "next_node_ip": "192.168.40.2"
  }
]


def policy_compare(global_policy: dict, local_policy: dict):
    global_path = global_policy['path']
    local_path = local_policy['path']
    global_hop_num = len(global_path)
    local_hop_num = len(local_path)
    global_delay = global_policy['delay']
    local_delay = local_policy['delay']
    global_bandwidth = global_policy['bandwidth']
    local_bandwidth = local_policy['bandwidth']
    global_response_time = global_policy['response_time']
    local_response_time = local_policy['response_time']

    global_final_policy = None
    if global_delay <= local_delay:
        print(f"[INFO] 全局重路由时延：{global_delay}ms，本地重路由时延：{local_delay}ms")
        print(f"[INFO] 全局重路由时延更短")
        global_final_policy = global_policy
    elif global_delay > local_delay:
        print(f"[INFO] 全局重路由时延：{global_delay}ms，本地重路由时延：{local_delay}ms")
        print(f"[INFO] 本地重路由时延更短")
        global_final_policy = local_policy

    if global_bandwidth <= local_bandwidth:
        print(f"[INFO] 全局重路由带宽：{global_bandwidth}MHz，本地重路由带宽：{local_bandwidth}MHz")
        print(f"[INFO] 全局重路由带宽更窄")
    elif global_bandwidth > local_bandwidth:
        print(f"[INFO] 全局重路由带宽：{global_bandwidth}MHz，本地重路由带宽：{local_bandwidth}MHz")
        print(f"[INFO] 本地重路由带宽更窄")

    if global_response_time <= local_response_time:
        print(f"[INFO] 全局重路由响应时间：{global_response_time}ms，本地重路由响应时间：{local_response_time}ms")
        print(f"[INFO] 全局重路由响应时间更短")
    elif global_response_time > local_response_time:
        print(f"[INFO] 全局重路由响应时间：{global_response_time}ms，本地重路由响应时间：{local_response_time}ms")
        print(f"[INFO] 本地重路由响应时间更短")

    if global_hop_num <= local_hop_num:
        print(f"[INFO] 全局重路由跳数：{global_hop_num}，本地重路由跳数：{local_hop_num}")
        print(f"[INFO] 全局重路由跳数更短")
    elif global_hop_num > local_hop_num:
        print(f"[INFO] 全局重路由跳数：{global_hop_num}，本地重路由跳数：{local_hop_num}")
        print(f"[INFO] 本地重路由跳数更短")

    return global_final_policy


if __name__ == "__main__":
    
    send_flow_table(flow_table, timeout=10.0, retries=2, verbose=True)