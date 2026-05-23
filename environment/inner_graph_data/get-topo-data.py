#!/usr/bin/env python3
"""Fetch current network topology/state data from the topology service."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

try:
    import requests
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    requests = None

REQUEST_ERRORS = (HTTPError, URLError, OSError)
if requests is not None:
    REQUEST_ERRORS = REQUEST_ERRORS + (requests.RequestException,)


DEFAULT_URL = (
    "http://192.168.1.20:8000/api/v1/network/topology/query"
    "?online_nodes_only=true"
    "&active_links_only=false"
    "&exclude_class_I=true"
    "&exclude_class_III=false"
    "&with_summary=true"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent / "json-data" / "network_topology_state.json"
)


def _str_bool(value: bool) -> str:
    return "true" if value else "false"


def _merge_url_query(url: str, params: dict[str, str]) -> str:
    parts = urlsplit(url)
    query_params = dict(parse_qsl(parts.query, keep_blank_values=True))
    query_params.update(params)
    query = urlencode(query_params)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))


def build_query_url(args: argparse.Namespace) -> str:
    params = {
        "online_nodes_only": _str_bool(args.online_nodes_only),
        "active_links_only": _str_bool(args.active_links_only),
        "exclude_class_I": _str_bool(args.exclude_class_i),
        "exclude_class_III": _str_bool(args.exclude_class_iii),
        "with_summary": _str_bool(args.with_summary),
    }
    return _merge_url_query(args.url, params)


def fetch_network_state(url: str, timeout: float, retries: int) -> dict[str, Any]:
    last_exc: Optional[Exception] = None

    for attempt in range(retries + 1):
        try:
            if requests is not None:
                response = requests.get(url, timeout=timeout)
                response.raise_for_status()
                return response.json()

            request = Request(url, headers={"Accept": "application/json"})
            with urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
            return json.loads(body)
        except REQUEST_ERRORS as exc:
            last_exc = exc
            if attempt < retries:
                sleep_s = 0.8 * (2**attempt)
                print(
                    f"[WARN] request failed: {exc}; retry in {sleep_s:.1f}s",
                    file=sys.stderr,
                )
                time.sleep(sleep_s)
                continue
            raise RuntimeError(f"request failed after {retries + 1} attempts: {exc}") from exc
        except ValueError as exc:
            raise RuntimeError("response is not valid JSON") from exc

    raise RuntimeError(f"unexpected request failure: {last_exc}")


def save_json(data: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def validate_api_result(data: dict[str, Any]) -> None:
    code = data.get("code")
    if code is not None and code not in (0, 200, "0", "200"):
        msg = data.get("msg", "")
        raise RuntimeError(f"API returned code={code}, msg={msg}")


def _extract_topology(data: dict[str, Any]) -> dict[str, Any]:
    payload = data.get("data", data)
    if not isinstance(payload, dict):
        return {}

    topo = payload.get("topo") or payload.get("topology") or data.get("topo")
    return topo if isinstance(topo, dict) else {}


def _extract_items(topo: dict[str, Any], singular: str, plural: str) -> list[Any]:
    items = topo.get(singular)
    if items is None:
        items = topo.get(plural)
    return items if isinstance(items, list) else []


def print_summary(data: dict[str, Any], output_path: Path) -> None:
    topo = _extract_topology(data)
    nodes = _extract_items(topo, "node", "nodes")
    links = _extract_items(topo, "link", "links")

    payload = data.get("data", data)
    summary = data.get("summary")
    if summary is None and isinstance(payload, dict):
        summary = payload.get("summary")

    print(f"[OK] saved: {output_path}")
    if topo:
        print(f"[INFO] nodes: {len(nodes)}, links: {len(links)}")
    if summary is not None:
        print("[INFO] summary:")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch network topology/state data from the topology query API."
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="Topology query API URL.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to save the raw network state JSON.",
    )
    parser.add_argument("--timeout", type=float, default=10.0, help="Request timeout in seconds.")
    parser.add_argument("--retries", type=int, default=2, help="Retry count after the first try.")
    parser.add_argument(
        "--online-nodes-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Query online nodes only.",
    )
    parser.add_argument(
        "--active-links-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Query active links only.",
    )
    parser.add_argument(
        "--exclude-class-i",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude class-I nodes.",
    )
    parser.add_argument(
        "--exclude-class-iii",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exclude class-III nodes.",
    )
    parser.add_argument(
        "--with-summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ask the API to return summary data.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the final request URL without sending it.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    url = build_query_url(args)

    if args.dry_run:
        print(url)
        return 0

    try:
        data = fetch_network_state(url, timeout=args.timeout, retries=args.retries)
        validate_api_result(data)
        save_json(data, args.output)
        print_summary(data, args.output)
    except RuntimeError as exc:
        print(f"[ERR] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
