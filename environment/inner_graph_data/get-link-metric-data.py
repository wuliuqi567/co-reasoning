#!/usr/bin/env python3
"""Fetch current network link metrics from the metrics service."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    import requests
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    requests = None

REQUEST_ERRORS = (HTTPError, URLError, OSError)
if requests is not None:
    REQUEST_ERRORS = REQUEST_ERRORS + (requests.RequestException,)


DEFAULT_URL = "http://192.168.1.20:8000/api/v1/network/link/metrics/query"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "json-data" / "link_metric.json"


def fetch_link_metrics(url: str, timeout: float, retries: int) -> dict[str, Any]:
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


def _extract_link_metrics(data: dict[str, Any]) -> list[Any]:
    payload = data.get("data", data)
    if not isinstance(payload, dict):
        return []
    metrics = payload.get("link_metrics") or payload.get("metrics") or data.get("link_metrics")
    return metrics if isinstance(metrics, list) else []


def print_summary(data: dict[str, Any], output_path: Path) -> None:
    metrics = _extract_link_metrics(data)
    print(f"[OK] saved: {output_path}")
    print(f"[INFO] link_metrics: {len(metrics)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch network link metrics.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Link metrics API URL.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to save the raw link metrics JSON.",
    )
    parser.add_argument("--timeout", type=float, default=10.0, help="Request timeout in seconds.")
    parser.add_argument("--retries", type=int, default=2, help="Retry count after the first try.")
    parser.add_argument("--dry-run", action="store_true", help="Only print the request URL.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dry_run:
        print(args.url)
        return 0

    try:
        data = fetch_link_metrics(args.url, timeout=args.timeout, retries=args.retries)
        validate_api_result(data)
        save_json(data, args.output)
        print_summary(data, args.output)
    except RuntimeError as exc:
        print(f"[ERR] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
