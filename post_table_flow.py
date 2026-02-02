#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
import time

import requests

DEFAULT_URL = "http://192.168.2.26:12590/api/flow/sflowtblCfg"


def build_payload() -> dict:
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
                "nw_src": "192.168.10.2/24",
                "nw_dst": "192.168.40.2/24",
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
                                    "next_hop": "192.168.22.2"
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

def send(url: str, timeout: float, retries: int, verbose: bool) -> int:
    payload = build_payload()
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


def main() -> int:
    ap = argparse.ArgumentParser(description="POST flow table JSON to 192.168.2.12")
    ap.add_argument("--url", default=DEFAULT_URL, help="目标 URL")
    ap.add_argument("--timeout", type=float, default=10.0, help="超时（秒）")
    ap.add_argument("--retries", type=int, default=2, help="失败重试次数")
    ap.add_argument("--verbose", action="store_true", help="打印下发 JSON")
    args = ap.parse_args()
    return send(args.url, args.timeout, args.retries, args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
