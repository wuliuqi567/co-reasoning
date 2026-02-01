# flow_sender.py
from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional, Union

import requests

# JSON 类型（dict/list 等均可被 requests 序列化为 JSON）
JsonType = Union[Dict[str, Any], list, str, int, float, bool, None]


def build_url(ip: str, port: int, path: str) -> str:
    """
    根据 ip/port/path 拼接目标 URL
    例如：ip=192.168.2.24, port=12590, path=/api/flow/sflowtblCfg
    -> http://192.168.2.24:12590/api/flow/sflowtblCfg
    """
    path = path if path.startswith("/") else ("/" + path)
    return f"http://{ip}:{port}{path}"


def strip_newlines_in_strings(obj: JsonType) -> JsonType:
    """
    递归去掉 payload 中所有字符串里的 \r 和 \n

    作用：
      - 复制粘贴超长 id（meter-id / flow-id）时，可能被自动换行
      - JSON 字符串里出现真实换行会导致设备端解析失败或格式不符合预期
      - 因此默认开启 strip_newlines=True
    """
    if isinstance(obj, str):
        return obj.replace("\r", "").replace("\n", "")
    if isinstance(obj, list):
        return [strip_newlines_in_strings(x) for x in obj]
    if isinstance(obj, dict):
        return {k: strip_newlines_in_strings(v) for k, v in obj.items()}
    return obj


def send_flowtable(
    ip: str,
    flow_payload: JsonType,
    *,
    port: int = 12590,
    path: str = "/api/flow/sflowtblCfg",
    method: str = "POST",
    timeout: float = 10.0,
    retries: int = 2,
    backoff_base: float = 0.8,
    headers: Optional[Dict[str, str]] = None,
    strip_newlines: bool = True,
    verbose: bool = False,
    session: Optional[requests.Session] = None,
) -> requests.Response:
    """
    向指定 IP 下发“流表 JSON”（传入的是 Python 变量，而非文件）。

    参数说明：
      - ip: 目标设备 IP，例如 "192.168.2.24"
      - flow_payload: 你的流表变量（dict / list 等 JSON 可序列化对象）
      - port/path: 目标设备 API 的端口与路径（默认 12590 + /api/flow/sflowtblCfg）
      - method: HTTP 方法（POST/PUT）
      - timeout: 单次请求超时时间（秒）
      - retries: 失败重试次数（例如 2 表示最多尝试 3 次：1 次 + 2 次重试）
      - backoff_base: 指数退避基数（秒），重试等待：0.8, 1.6, 3.2 ...
      - headers: 额外请求头（例如鉴权 Authorization）
      - strip_newlines: 是否自动去掉字符串里的换行（建议默认 True）
      - verbose: 是否打印中文调试信息（建议调试时 True）
      - session: 可复用的 requests.Session（批量下发多台设备时更高效）

    返回：
      - requests.Response（你可以通过 resp.status_code / resp.json() / resp.text 读取结果）
    """
    method = method.upper().strip()
    if method not in ("POST", "PUT"):
        raise ValueError("method 参数只能是 POST 或 PUT")

    url = build_url(ip, port, path)

    # 对流表变量做一次“去换行清洗”（默认开启）
    payload = strip_newlines_in_strings(flow_payload) if strip_newlines else flow_payload

    # 请求头：默认 application/json；可叠加用户自定义 headers
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)

    # 允许传入 session 做连接复用；否则内部新建一个
    sess = session or requests.Session()

    if verbose:
        print("【信息】准备下发流表")
        print(f"【信息】请求方法：{method}")
        print(f"【信息】目标地址：{url}")
        print(f"【信息】超时：{timeout}s，重试次数：{retries}")
        try:
            print("【信息】下发的 JSON 内容：")
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        except Exception:
            print("【警告】JSON 打印失败：payload 可能含不可序列化对象（请检查流表变量）")

    # 带重试的请求
    for attempt in range(retries + 1):
        try:
            resp = sess.request(method, url, json=payload, headers=hdrs, timeout=timeout)

            if verbose:
                print(f"【结果】HTTP 状态码：{resp.status_code}")
                if (resp.text or "").strip():
                    try:
                        print("【结果】响应 JSON：")
                        print(json.dumps(resp.json(), ensure_ascii=False, indent=2))
                    except Exception:
                        print("【结果】响应文本：")
                        print(resp.text)
                else:
                    print("【结果】响应内容为空（可能是 204 No Content，也可能设备不返回 body）")

            return resp

        except Exception as e:
            if attempt < retries:
                sleep_t = backoff_base * (2 ** attempt)
                if verbose:
                    print(f"【警告】请求失败：{e}")
                    print(f"【警告】将在 {sleep_t:.1f}s 后重试（第 {attempt + 1}/{retries} 次重试）...")
                time.sleep(sleep_t)
            else:
                raise RuntimeError(f"请求失败（已重试 {retries} 次仍失败）：{e}") from e

    raise RuntimeError("未知错误：请求流程异常结束")


# =============================
# 如何使用（示例）
# =============================
if __name__ == "__main__":
    # 1) 假设你通过其他代码得到一个“流表变量”（dict）
    flow_payload = {
        "origin": 1,
        "flowtable": [
            {
                "id": "Flow_Path0_Node17_to_Node16",
                "table_id": 0,
                "priority": 5000,
                "match": {
                    "nw_src": "192.168.2.12/32",
                    "nw_dst": "192.168.2.30/32",
                    "ethernet-match": {"ethernet-type": {"type": 2048}},
                },
                "instructions": {
                    "instruction": [
                        {
                            "order": 1,
                            "apply-actions": {
                                "action": [
                                    {
                                        "order": 0,
                                        "output": "00012500163431326600004400000000:3",
                                        "nextHop": "192.168.2.26",
                                    }
                                ]
                            },
                        }
                    ]
                },
            }
        ],
    }

    # 2) 调用 send_flowtable：把目标 IP 和流表变量传进去即可
    target_ip = "192.168.2.24"
    resp = send_flowtable(target_ip, flow_payload, verbose=True)

    # 3) 根据返回状态码判断是否成功（一般 2xx 算成功）
    if 200 <= resp.status_code < 300:
        print("【成功】流表下发成功")
    else:
        print("【失败】流表下发失败")
        print("状态码：", resp.status_code)
        print("响应：", resp.text)
