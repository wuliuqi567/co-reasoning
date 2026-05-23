#!/usr/bin/env python3
"""Parse fetched topology JSON and build a NetworkX graph."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Iterable, Optional, Union

try:
    import networkx as nx
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment
    raise ModuleNotFoundError(
        "Missing dependency: networkx. Install project dependencies with "
        "`pip install -r requirements.txt`."
    ) from exc


DEFAULT_INPUT = Path(__file__).resolve().parent / "json-data" / "network_topology_state.json"
DEFAULT_LINK_METRIC_INPUT = Path(__file__).resolve().parent / "json-data" / "link_metric.json"
DEFAULT_GRAPHML = Path(__file__).resolve().parent / "graph-data" / "ii_network_topology.graphml"
DEFAULT_BASE_GRAPHML = Path(__file__).resolve().parent / "base_ii_topology.graphml"
DEFAULT_II_NODE_TYPES = frozenset({3, 4, 5})
OFFLINE_STATUS = 0
ONLINE_STATUS = 1
FAULT_LINK_STATUS = 0

NODE_TYPE_NAMES = {
    1: "I类终端",
    2: "I类簇头",
    3: "II类车载",
    4: "II类接入",
    5: "II类骨干",
    6: "IV类网关",
    7: "未知类型7",
    8: "III类网管",
}


def load_topology_graph(
    source: Union[str, Path, dict[str, Any]] = DEFAULT_INPUT,
    *,
    multigraph: bool = True,
    node_types: Optional[Iterable[int]] = DEFAULT_II_NODE_TYPES,
    include_missing_endpoints: bool = False,
) -> Union[nx.Graph, nx.MultiGraph]:
    """Load topology JSON and return a NetworkX graph.

    Args:
        source: JSON file path or an already parsed response dictionary.
        multigraph: Build ``nx.MultiGraph`` when true, preserving parallel links.
        node_types: Optional node type filter. Defaults to II-class nodes:
            ``{3, 4, 5}``.
        include_missing_endpoints: Add placeholder nodes for link endpoints that
            are referenced by links but absent from the node list.

    Returns:
        A populated NetworkX graph object.
    """
    data = _load_json(source)
    topo = _extract_topology(data)
    nodes = _extract_items(topo, "node", "nodes")
    links = _extract_items(topo, "link", "links")
    node_type_filter = set(node_types) if node_types is not None else None

    graph: Union[nx.Graph, nx.MultiGraph]
    graph = nx.MultiGraph() if multigraph else nx.Graph()
    graph.graph["source"] = str(source) if isinstance(source, (str, Path)) else "<dict>"
    graph.graph["summary"] = data.get("summary", {})
    graph.graph["api_code"] = data.get("code")
    graph.graph["api_msg"] = data.get("msg")
    graph.graph["raw_node_count"] = len(nodes)
    graph.graph["raw_link_count"] = len(links)

    for idx, node in enumerate(nodes):
        node_id = node.get("node_id")
        if not node_id:
            continue

        node_type = _safe_int(node.get("node_type"))
        if node_type_filter is not None and node_type not in node_type_filter:
            continue

        attrs = _build_node_attrs(node, idx)
        graph.add_node(node_id, **attrs)

    skipped_links = 0
    for idx, link in enumerate(links):
        src_node, dst_node = _get_link_endpoints(link)
        if not src_node or not dst_node:
            skipped_links += 1
            continue

        if src_node not in graph:
            if include_missing_endpoints:
                _add_missing_endpoint_node(graph, src_node)
            else:
                skipped_links += 1
                continue
        if dst_node not in graph:
            if include_missing_endpoints:
                _add_missing_endpoint_node(graph, dst_node)
            else:
                skipped_links += 1
                continue

        attrs = _build_edge_attrs(graph, link, idx)
        if multigraph:
            edge_key = attrs.get("link_id") or idx
            if graph.has_edge(src_node, dst_node, key=edge_key):
                edge_key = f"{edge_key}#{idx}"
            graph.add_edge(src_node, dst_node, key=edge_key, **attrs)
        else:
            graph.add_edge(src_node, dst_node, **attrs)

    graph.graph["skipped_link_count"] = skipped_links
    return graph


def build_base_graph(
    source: Union[str, Path, dict[str, Any]] = DEFAULT_INPUT,
    *,
    multigraph: bool = True,
    node_types: Optional[Iterable[int]] = DEFAULT_II_NODE_TYPES,
) -> Union[nx.Graph, nx.MultiGraph]:
    """Build a static II-class topology graph with all status fields set to 0."""
    graph = load_topology_graph(
        source,
        multigraph=multigraph,
        node_types=node_types,
        include_missing_endpoints=False,
    )
    set_all_status_offline(graph)
    graph.graph["is_base_graph"] = True
    graph.graph["status_semantics"] = "0=offline/fault, 1=online"
    return graph


def set_all_status_offline(graph: Union[nx.Graph, nx.MultiGraph]) -> None:
    """Set every node, port, and link status in the graph to 0."""
    for _, attrs in graph.nodes(data=True):
        attrs["node_status"] = OFFLINE_STATUS
        _set_ports_status(attrs, OFFLINE_STATUS)

    for attrs in _iter_edge_attrs(graph):
        attrs["link_status"] = OFFLINE_STATUS


def save_base_graphml(
    graph: Optional[Union[nx.Graph, nx.MultiGraph]] = None,
    output_path: Union[str, Path] = DEFAULT_BASE_GRAPHML,
    source: Union[str, Path, dict[str, Any]] = DEFAULT_INPUT,
) -> Path:
    if graph is None:
        graph = build_base_graph(source)
    return save_graphml(graph, output_path)


def load_graphml(path: Union[str, Path]) -> Union[nx.Graph, nx.MultiGraph]:
    graph = nx.read_graphml(str(path))
    _restore_graph_attrs(graph)
    return graph


def load_or_create_base_graph(
    base_graphml: Union[str, Path] = DEFAULT_BASE_GRAPHML,
    source: Union[str, Path, dict[str, Any]] = DEFAULT_INPUT,
    *,
    rebuild: bool = False,
) -> Union[nx.Graph, nx.MultiGraph]:
    base_path = Path(base_graphml)
    if base_path.exists() and not rebuild:
        graph = load_graphml(base_path)
        set_all_status_offline(graph)
        return graph

    graph = build_base_graph(source)
    save_base_graphml(graph, base_path)
    return graph


def update_graph_from_latest_topology(
    graph: Union[nx.Graph, nx.MultiGraph],
    latest_source: Union[str, Path, dict[str, Any]] = DEFAULT_INPUT,
    *,
    node_types: Optional[Iterable[int]] = DEFAULT_II_NODE_TYPES,
) -> Union[nx.Graph, nx.MultiGraph]:
    """Update a base graph in-place using latest topology JSON status/metrics."""
    set_all_status_offline(graph)
    latest_graph = load_topology_graph(
        latest_source,
        multigraph=graph.is_multigraph(),
        node_types=node_types,
        include_missing_endpoints=False,
    )

    updated_nodes = 0
    for node_id, latest_attrs in latest_graph.nodes(data=True):
        if node_id not in graph:
            continue
        graph.nodes[node_id].update(latest_attrs)
        updated_nodes += 1

    edge_index = _build_edge_index(graph)
    updated_edges = 0
    for _, _, latest_attrs in _iter_edges(latest_graph):
        link_id = latest_attrs.get("link_id")
        if not link_id or link_id not in edge_index:
            continue
        src, dst, key = edge_index[link_id]
        if graph.is_multigraph():
            graph.edges[src, dst, key].update(latest_attrs)
        else:
            graph.edges[src, dst].update(latest_attrs)
        updated_edges += 1

    graph.graph["latest_source"] = (
        str(latest_source) if isinstance(latest_source, (str, Path)) else "<dict>"
    )
    graph.graph["updated_node_count"] = updated_nodes
    graph.graph["updated_link_count"] = updated_edges
    graph.graph["status_semantics"] = "0=offline/fault, 1=online"
    return graph


def build_updated_graph(
    latest_json: Union[str, Path] = DEFAULT_INPUT,
    base_graphml: Union[str, Path] = DEFAULT_BASE_GRAPHML,
    *,
    fetch_latest: bool = False,
    rebuild_base: bool = False,
    timeout: float = 10.0,
    retries: int = 2,
) -> Union[nx.Graph, nx.MultiGraph]:
    """Load/create base graph, optionally fetch latest JSON, then update statuses."""
    latest_path = Path(latest_json)
    if fetch_latest:
        fetch_latest_topology_json(latest_path, timeout=timeout, retries=retries)

    base_graph = load_or_create_base_graph(
        base_graphml,
        latest_path,
        rebuild=rebuild_base,
    )
    return update_graph_from_latest_topology(base_graph, latest_path)


def update_graph_from_link_metrics(
    graph: Union[nx.Graph, nx.MultiGraph],
    metric_source: Union[str, Path, dict[str, Any]] = DEFAULT_LINK_METRIC_INPUT,
) -> Union[nx.Graph, nx.MultiGraph]:
    """Update edge metrics by matching link_metrics[].link_id to graph edge link_id."""
    metric_data = _load_json(metric_source)
    metrics = _extract_link_metrics(metric_data)
    edge_index = _build_edge_index(graph)

    updated = 0
    skipped_no_id = 0
    skipped_no_match = 0
    for metric in metrics:
        if not isinstance(metric, dict):
            continue

        link_id = metric.get("link_id")
        if not link_id:
            skipped_no_id += 1
            continue
        if link_id not in edge_index:
            skipped_no_match += 1
            continue

        src, dst, key = edge_index[link_id]
        edge_attrs = graph.edges[src, dst, key] if graph.is_multigraph() else graph.edges[src, dst]
        _update_edge_metric_attrs(edge_attrs, metric)
        updated += 1

    graph.graph["link_metric_source"] = (
        str(metric_source) if isinstance(metric_source, (str, Path)) else "<dict>"
    )
    graph.graph["link_metric_count"] = len(metrics)
    graph.graph["link_metric_updated_count"] = updated
    graph.graph["link_metric_skipped_no_id"] = skipped_no_id
    graph.graph["link_metric_skipped_no_match"] = skipped_no_match
    return graph


def fetch_latest_link_metrics_json(
    output_path: Union[str, Path] = DEFAULT_LINK_METRIC_INPUT,
    *,
    timeout: float = 10.0,
    retries: int = 2,
) -> dict[str, Any]:
    """Fetch latest link metrics JSON by reusing get-link-metric-data.py."""
    module_path = Path(__file__).resolve().parent / "get-link-metric-data.py"
    spec = importlib.util.spec_from_file_location("get_link_metric_data", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load link metrics fetcher: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    data = module.fetch_link_metrics(module.DEFAULT_URL, timeout=timeout, retries=retries)
    module.validate_api_result(data)
    module.save_json(data, Path(output_path))
    return data


def fetch_latest_topology_json(
    output_path: Union[str, Path] = DEFAULT_INPUT,
    *,
    timeout: float = 10.0,
    retries: int = 2,
) -> dict[str, Any]:
    """Fetch latest topology JSON by reusing get-topo-data.py."""
    module_path = Path(__file__).resolve().parent / "get-topo-data.py"
    spec = importlib.util.spec_from_file_location("get_topo_data", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load topology fetcher: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    data = module.fetch_network_state(module.DEFAULT_URL, timeout=timeout, retries=retries)
    module.validate_api_result(data)
    module.save_json(data, Path(output_path))
    return data


def _extract_link_metrics(data: dict[str, Any]) -> list[dict[str, Any]]:
    payload = data.get("data", data)
    if not isinstance(payload, dict):
        return []
    metrics = payload.get("link_metrics") or payload.get("metrics") or data.get("link_metrics")
    return metrics if isinstance(metrics, list) else []


def _update_edge_metric_attrs(edge_attrs: dict[str, Any], metric: dict[str, Any]) -> None:
    field_defaults = {
        "link_latency": edge_attrs.get("link_latency", 0.0),
        "link_utilization": 0.0,
        "bandwidth_capacity_available": edge_attrs.get("link_bandwidth", 0.0),
        "link_loss_rate": 0.0,
    }

    for field, default in field_defaults.items():
        value = metric.get(field)
        if value in (None, ""):
            continue
        value = _safe_float(value, default=default)
        if field == "link_latency" and value < 0:
            continue
        edge_attrs[field] = value

    flow_status = metric.get("flow_table_status")
    if flow_status not in (None, ""):
        edge_attrs["flow_table_status"] = _safe_int(flow_status, default=0)


def _add_missing_endpoint_node(
    graph: Union[nx.Graph, nx.MultiGraph],
    node_id: str,
) -> None:
    graph.add_node(
        node_id,
        idx=graph.number_of_nodes(),
        node_id=node_id,
        node_type=0,
        node_type_name="缺失节点信息",
        node_status=1,
        node_location="",
        longitude=None,
        latitude=None,
        node_manage_ip_addr="",
        node_ports=[],
        port_count=0,
        port_ids=[],
        port_ips=[],
        port_info={},
        missing_from_node_list=True,
    )


def _set_ports_status(node_attrs: dict[str, Any], status: int) -> None:
    ports = _coerce_json_value(node_attrs.get("node_ports"), default=[])
    if not isinstance(ports, list):
        ports = []

    for port in ports:
        if isinstance(port, dict):
            port["status"] = status
    node_attrs["node_ports"] = ports

    port_info = _coerce_json_value(node_attrs.get("port_info"), default={})
    if not isinstance(port_info, dict):
        port_info = {}
    for port in port_info.values():
        if isinstance(port, dict):
            port["status"] = status
    node_attrs["port_info"] = port_info


def _load_json(source: Union[str, Path, dict[str, Any]]) -> dict[str, Any]:
    if isinstance(source, dict):
        return source

    path = Path(source)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_topology(data: dict[str, Any]) -> dict[str, Any]:
    payload = data.get("data", data)
    if not isinstance(payload, dict):
        return {}

    topo = payload.get("topo") or payload.get("topology") or data.get("topo")
    return topo if isinstance(topo, dict) else {}


def _extract_items(topo: dict[str, Any], singular: str, plural: str) -> list[dict[str, Any]]:
    items = topo.get(singular)
    if items is None:
        items = topo.get(plural)
    return items if isinstance(items, list) else []


def _build_node_attrs(node: dict[str, Any], idx: int) -> dict[str, Any]:
    node_id = node.get("node_id")
    node_type = _safe_int(node.get("node_type"))
    location = node.get("node_location", "") or ""
    longitude, latitude = _parse_location(location)

    ports = node.get("node_ports") or node.get("ports") or []
    if not isinstance(ports, list):
        ports = []

    port_info = {}
    port_ids = []
    port_ips = []
    for port in ports:
        if not isinstance(port, dict):
            continue
        port_id = port.get("port_id")
        if not port_id:
            continue
        port_ids.append(port_id)
        port_ips.append(port.get("ip_address", "") or "")
        port_info[port_id] = _build_port_attrs(port)

    return {
        "idx": idx,
        "node_id": node_id,
        "node_type": node_type,
        "node_type_name": NODE_TYPE_NAMES.get(node_type, f"类型{node_type}"),
        "node_status": _safe_int(node.get("node_status"), default=1),
        "node_location": location,
        "longitude": longitude,
        "latitude": latitude,
        "node_manage_ip_addr": node.get("node_manage_ip_addr", "") or "",
        "node_ports": list(port_info.values()),
        "port_count": len(ports),
        "port_ids": port_ids,
        "port_ips": port_ips,
        "port_info": port_info,
    }


def _build_port_attrs(port: dict[str, Any]) -> dict[str, Any]:
    return {
        "port_id": port.get("port_id", "") or "",
        "status": _safe_int(port.get("status"), default=0),
        "nid": _safe_int(port.get("nid"), default=0),
        "teid": _safe_int(port.get("teid"), default=0),
        "ip_address": port.get("ip_address", "") or "",
        "mac_address": port.get("mac_address", "") or "",
    }


def _get_link_endpoints(link: dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    src = link.get("src") if isinstance(link.get("src"), dict) else {}
    dst = link.get("dst") if isinstance(link.get("dst"), dict) else {}

    src_node = src.get("src_node") or link.get("src_node") or link.get("source")
    dst_node = dst.get("dst_node") or link.get("dst_node") or link.get("target")

    if (not src_node or not dst_node) and link.get("link_id"):
        parsed = _parse_link_id_endpoints(str(link["link_id"]))
        if parsed:
            src_node, dst_node = parsed

    return src_node, dst_node


def _parse_link_id_endpoints(link_id: str) -> Optional[tuple[str, str]]:
    parts = link_id.split("_")
    if len(parts) != 2:
        return None
    return parts[0].split(":")[0], parts[1].split(":")[0]


def _build_edge_attrs(
    graph: Union[nx.Graph, nx.MultiGraph],
    link: dict[str, Any],
    idx: int,
) -> dict[str, Any]:
    src = link.get("src") if isinstance(link.get("src"), dict) else {}
    dst = link.get("dst") if isinstance(link.get("dst"), dict) else {}
    src_node, dst_node = _get_link_endpoints(link)
    src_port = src.get("src_port") or link.get("src_port") or ""
    dst_port = dst.get("dst_port") or link.get("dst_port") or ""

    return {
        "idx": idx,
        "link_id": link.get("link_id", ""),
        "link_status": _safe_int(link.get("link_status"), default=1),
        "link_bandwidth": _safe_float(link.get("link_bandwidth"), default=0.0),
        "link_latency": _safe_float(link.get("link_latency"), default=0.0),
        "src_node": src_node or "",
        "src_port": src_port,
        "dst_node": dst_node or "",
        "dst_port": dst_port,
        "src_port_ip": _lookup_port_ip(graph, src_node, src_port),
        "dst_port_ip": _lookup_port_ip(graph, dst_node, dst_port),
    }


def _lookup_port_ip(
    graph: Union[nx.Graph, nx.MultiGraph],
    node_id: Optional[str],
    port_id: str,
) -> str:
    if not node_id or not port_id or node_id not in graph:
        return ""
    port_info = graph.nodes[node_id].get("port_info", {})
    if not isinstance(port_info, dict):
        return ""
    port = port_info.get(port_id, {})
    return port.get("ip_address", "") if isinstance(port, dict) else ""


def _parse_location(location: str) -> tuple[Optional[float], Optional[float]]:
    if not location or "," not in location:
        return None, None
    try:
        longitude, latitude = location.split(",", 1)
        return float(longitude.strip()), float(latitude.strip())
    except ValueError:
        return None, None


def _safe_int(value: Any, default: int = 0) -> int:
    if value in ("", None):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_node_types(value: Optional[str]) -> Optional[set[int]]:
    if not value:
        return set(DEFAULT_II_NODE_TYPES)
    if value.lower() == "all":
        return None
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def _graphml_safe_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False)


def _coerce_json_value(value: Any, default: Any) -> Any:
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return default
    if text[0] not in "[{":
        return value

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def _restore_graph_attrs(graph: Union[nx.Graph, nx.MultiGraph]) -> None:
    for _, attrs in graph.nodes(data=True):
        attrs["idx"] = _safe_int(attrs.get("idx"))
        attrs["node_type"] = _safe_int(attrs.get("node_type"))
        attrs["node_status"] = _safe_int(attrs.get("node_status"), default=OFFLINE_STATUS)
        attrs["longitude"] = _safe_float(attrs.get("longitude")) if attrs.get("longitude") != "" else None
        attrs["latitude"] = _safe_float(attrs.get("latitude")) if attrs.get("latitude") != "" else None
        attrs["node_ports"] = _coerce_json_value(attrs.get("node_ports"), default=[])
        attrs["port_ids"] = _coerce_json_value(attrs.get("port_ids"), default=[])
        attrs["port_ips"] = _coerce_json_value(attrs.get("port_ips"), default=[])
        attrs["port_info"] = _coerce_json_value(attrs.get("port_info"), default={})
        attrs["port_count"] = _safe_int(attrs.get("port_count"), default=0)

    for attrs in _iter_edge_attrs(graph):
        attrs["idx"] = _safe_int(attrs.get("idx"))
        attrs["link_status"] = _safe_int(attrs.get("link_status"), default=OFFLINE_STATUS)
        attrs["link_bandwidth"] = _safe_float(attrs.get("link_bandwidth"), default=0.0)
        attrs["link_latency"] = _safe_float(attrs.get("link_latency"), default=0.0)
        for field in ("link_utilization", "bandwidth_capacity_available", "link_loss_rate"):
            if field in attrs and attrs[field] != "":
                attrs[field] = _safe_float(attrs.get(field), default=0.0)
        if "flow_table_status" in attrs and attrs["flow_table_status"] != "":
            attrs["flow_table_status"] = _safe_int(attrs.get("flow_table_status"), default=0)


def _copy_for_graphml(graph: Union[nx.Graph, nx.MultiGraph]) -> Union[nx.Graph, nx.MultiGraph]:
    copied = graph.__class__()
    copied.graph.update({k: _graphml_safe_value(v) for k, v in graph.graph.items()})

    for node_id, attrs in graph.nodes(data=True):
        copied.add_node(
            node_id,
            **{key: _graphml_safe_value(value) for key, value in attrs.items()},
        )

    if graph.is_multigraph():
        for src, dst, key, attrs in graph.edges(keys=True, data=True):
            copied.add_edge(
                src,
                dst,
                key=key,
                **{name: _graphml_safe_value(value) for name, value in attrs.items()},
            )
    else:
        for src, dst, attrs in graph.edges(data=True):
            copied.add_edge(
                src,
                dst,
                **{name: _graphml_safe_value(value) for name, value in attrs.items()},
            )
    return copied


def save_graphml(graph: Union[nx.Graph, nx.MultiGraph], output_path: Union[str, Path]) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    nx.write_graphml(_copy_for_graphml(graph), tmp_path)
    tmp_path.replace(path)
    return path


def _build_edge_index(
    graph: Union[nx.Graph, nx.MultiGraph],
) -> dict[str, tuple[str, str, Optional[str]]]:
    index = {}
    if graph.is_multigraph():
        for src, dst, key, attrs in graph.edges(keys=True, data=True):
            link_id = attrs.get("link_id")
            if link_id:
                index[str(link_id)] = (src, dst, key)
    else:
        for src, dst, attrs in graph.edges(data=True):
            link_id = attrs.get("link_id")
            if link_id:
                index[str(link_id)] = (src, dst, None)
    return index


def summarize_graph(graph: Union[nx.Graph, nx.MultiGraph]) -> dict[str, Any]:
    fault_links = get_fault_links(graph)
    node_status_count = _count_node_status(graph)
    link_status_count = _count_link_status(graph)
    summary = {
        "graph_type": graph.__class__.__name__,
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "raw_nodes": graph.graph.get("raw_node_count", 0),
        "raw_links": graph.graph.get("raw_link_count", 0),
        "skipped_links": graph.graph.get("skipped_link_count", 0),
        "node_type_count": _count_node_types(graph.nodes(data=True)),
        "node_status_count": node_status_count,
        "link_status_count": link_status_count,
        "online_node_count": sum(
            count for status, count in node_status_count.items() if status != OFFLINE_STATUS
        ),
        "online_link_count": sum(
            count for status, count in link_status_count.items() if status != OFFLINE_STATUS
        ),
        "fault_link_count": len(fault_links),
    }
    if "link_metric_count" in graph.graph:
        summary.update(
            {
                "link_metric_count": graph.graph.get("link_metric_count", 0),
                "link_metric_updated_count": graph.graph.get("link_metric_updated_count", 0),
                "link_metric_skipped_no_match": graph.graph.get(
                    "link_metric_skipped_no_match", 0
                ),
            }
        )
    return summary


def _count_node_types(nodes: Iterable[tuple[str, dict[str, Any]]]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for _, attrs in nodes:
        node_type = _safe_int(attrs.get("node_type"))
        counts[node_type] = counts.get(node_type, 0) + 1
    return dict(sorted(counts.items()))


def _count_link_status(graph: Union[nx.Graph, nx.MultiGraph]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for attrs in _iter_edge_attrs(graph):
        status = _safe_int(attrs.get("link_status"))
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _count_node_status(graph: Union[nx.Graph, nx.MultiGraph]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for _, attrs in graph.nodes(data=True):
        status = _safe_int(attrs.get("node_status"), default=OFFLINE_STATUS)
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def get_fault_links(graph: Union[nx.Graph, nx.MultiGraph]) -> list[dict[str, Any]]:
    """Return links whose link_status is 0, which indicates a failed link."""
    fault_links = []
    for src, dst, attrs in _iter_edges(graph):
        if _safe_int(attrs.get("link_status")) != FAULT_LINK_STATUS:
            continue
        fault_links.append(
            {
                "link_id": attrs.get("link_id", ""),
                "link_status": attrs.get("link_status"),
                "src_node": attrs.get("src_node", src),
                "src_port": attrs.get("src_port", ""),
                "dst_node": attrs.get("dst_node", dst),
                "dst_port": attrs.get("dst_port", ""),
                "link_bandwidth": attrs.get("link_bandwidth", 0.0),
                "link_latency": attrs.get("link_latency", 0.0),
            }
        )
    return fault_links


def print_fault_links(graph: Union[nx.Graph, nx.MultiGraph]) -> None:
    fault_links = get_fault_links(graph)
    if not fault_links:
        print("[INFO] fault links(link_status=0): 0")
        return

    print(f"[INFO] fault links(link_status=0): {len(fault_links)}")
    for link in fault_links:
        print(
            "  - "
            f"{link['link_id']} | "
            f"{link['src_node']}({link['src_port']}) -> "
            f"{link['dst_node']}({link['dst_port']}) | "
            f"bandwidth={link['link_bandwidth']}Mbps, "
            f"latency={link['link_latency']}ms"
        )


def _iter_edges(
    graph: Union[nx.Graph, nx.MultiGraph],
) -> Iterable[tuple[str, str, dict[str, Any]]]:
    if graph.is_multigraph():
        for src, dst, _, attrs in graph.edges(keys=True, data=True):
            yield src, dst, attrs
    else:
        for src, dst, attrs in graph.edges(data=True):
            yield src, dst, attrs


def _iter_edge_attrs(graph: Union[nx.Graph, nx.MultiGraph]) -> Iterable[dict[str, Any]]:
    for _, _, attrs in _iter_edges(graph):
        yield attrs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse topology JSON and build a NetworkX graph."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Fetched topology JSON path.",
    )
    parser.add_argument(
        "--base-graphml",
        type=Path,
        default=DEFAULT_BASE_GRAPHML,
        help="Static base GraphML path.",
    )
    parser.add_argument(
        "--build-base",
        action="store_true",
        help="Build an all-offline static base GraphML from input JSON.",
    )
    parser.add_argument(
        "--update-from-base",
        action="store_true",
        help="Load/create base GraphML and update it from input JSON.",
    )
    parser.add_argument(
        "--fetch-latest",
        action="store_true",
        help="Fetch latest topology JSON before updating from base.",
    )
    parser.add_argument(
        "--rebuild-base",
        action="store_true",
        help="Rebuild base GraphML even when it already exists.",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Build nx.Graph instead of nx.MultiGraph.",
    )
    parser.add_argument(
        "--node-types",
        default="3,4,5",
        help='Comma-separated node type filter. Defaults to "3,4,5" for II-class nodes. Use "all" to disable filtering.',
    )
    parser.add_argument(
        "--include-missing-endpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add placeholder nodes for link endpoints missing from the node list.",
    )
    parser.add_argument(
        "--output-graphml",
        type=Path,
        help=f"Optional GraphML output path. Default suggestion: {DEFAULT_GRAPHML}",
    )
    return parser.parse_args()


def main() -> Union[nx.Graph, nx.MultiGraph]:
    args = parse_args()
    node_types = _parse_node_types(args.node_types)

    if args.build_base:
        graph = build_base_graph(
            args.input,
            multigraph=not args.simple,
            node_types=node_types,
        )
        output_path = save_base_graphml(graph, args.base_graphml)
        print(f"[OK] saved base GraphML: {output_path}")
    elif args.update_from_base or args.fetch_latest or args.rebuild_base:
        graph = build_updated_graph(
            latest_json=args.input,
            base_graphml=args.base_graphml,
            fetch_latest=args.fetch_latest,
            rebuild_base=args.rebuild_base,
        )
    else:
        graph = load_topology_graph(
            args.input,
            multigraph=not args.simple,
            node_types=node_types,
            include_missing_endpoints=args.include_missing_endpoints,
        )

    print(json.dumps(summarize_graph(graph), ensure_ascii=False, indent=2))
    if not args.build_base:
        print_fault_links(graph)
    if args.output_graphml:
        output_path = save_graphml(graph, args.output_graphml)
        print(f"[OK] saved GraphML: {output_path}")

    return graph


if __name__ == "__main__":
    main()
