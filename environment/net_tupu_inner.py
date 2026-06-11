"""Inner-field NetTupu environment adapter.

This module reuses the DDQN training environment in ``net_tupu_iii.py`` and
only changes how topology data is loaded. Inner-field topology data lives under
``environment/inner_graph_data`` and is stored as online JSON plus link metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import networkx as nx

from environment.net_tupu_iii import NetTupu
from environment.tools import _coerce_float, _coerce_int
from environment.inner_graph_data import topology_to_networkx as inner_topo


class NetTupuInner(NetTupu):
    """DDQN routing environment backed by ``environment/inner_graph_data``."""

    def __init__(self, env_config):
        self.inner_env_config = env_config
        self.inner_graph_data_dir = Path(__file__).resolve().parent / "inner_graph_data"
        super().__init__(env_config)

    def _load_graph_by_source(self, source: str) -> nx.Graph | None:
        source_text = str(source)
        if source_text in {"inner_latest", "inner_graph_data", "inner_json"}:
            return self._load_inner_latest_graph()

        path_obj = Path(source_text)
        if path_obj.suffix.lower() == ".json":
            return self._load_inner_json_graph(path_obj)

        return super()._load_graph_by_source(source_text)

    def _load_inner_latest_graph(self) -> nx.Graph:
        cfg = self.inner_env_config
        topology_json = Path(
            getattr(
                cfg,
                "inner_topology_json",
                self.inner_graph_data_dir / "json-data" / "network_topology_state.json",
            )
        )
        link_metric_json = Path(
            getattr(
                cfg,
                "inner_link_metric_json",
                self.inner_graph_data_dir / "json-data" / "link_metric.json",
            )
        )

        if bool(getattr(cfg, "inner_fetch_latest", False)):
            inner_topo.fetch_latest_topology_json(topology_json)
            inner_topo.fetch_latest_link_metrics_json(link_metric_json)

        graph = inner_topo.load_topology_graph(
            topology_json,
            multigraph=False,
            node_types=self._parse_inner_node_types(
                getattr(cfg, "inner_node_types", "3,4,5")
            ),
            include_missing_endpoints=False,
        )
        if link_metric_json.exists():
            inner_topo.update_graph_from_link_metrics(graph, link_metric_json)

        return self._prepare_inner_graph(graph)

    def _load_inner_json_graph(self, path_obj: Path) -> nx.Graph:
        graph = inner_topo.load_topology_graph(
            path_obj,
            multigraph=False,
            node_types=self._parse_inner_node_types(
                getattr(self.inner_env_config, "inner_node_types", "3,4,5")
            ),
            include_missing_endpoints=False,
        )
        link_metric_json = Path(
            getattr(
                self.inner_env_config,
                "inner_link_metric_json",
                self.inner_graph_data_dir / "json-data" / "link_metric.json",
            )
        )
        if link_metric_json.exists():
            inner_topo.update_graph_from_link_metrics(graph, link_metric_json)
        return self._prepare_inner_graph(graph)

    def _prepare_inner_graph(self, graph: nx.Graph) -> nx.Graph:
        graph = self._collapse_multigraph(graph)
        self._normalize_inner_attributes(graph)
        return self._relabel_graph_nodes(graph)

    @staticmethod
    def _parse_inner_node_types(raw: Any) -> Iterable[int] | None:
        if raw is None:
            return {3, 4, 5}
        if isinstance(raw, str):
            text = raw.strip()
            if not text or text.lower() == "all":
                return None
            return {int(part.strip()) for part in text.split(",") if part.strip()}
        return {int(item) for item in raw}

    @staticmethod
    def _collapse_multigraph(graph: nx.Graph) -> nx.Graph:
        if not graph.is_multigraph():
            return graph

        simple_graph = nx.Graph()
        simple_graph.graph.update(graph.graph)
        simple_graph.add_nodes_from((node, dict(attrs)) for node, attrs in graph.nodes(data=True))

        for src, dst, attrs in graph.edges(data=True):
            attrs = dict(attrs)
            if not simple_graph.has_edge(src, dst):
                simple_graph.add_edge(src, dst, **attrs)
                continue

            current = simple_graph[src][dst]
            current_latency = _coerce_float(current.get("link_latency", 0.0))
            new_latency = _coerce_float(attrs.get("link_latency", 0.0))
            if new_latency < current_latency:
                simple_graph[src][dst].update(attrs)

        return simple_graph

    @staticmethod
    def _normalize_inner_attributes(graph: nx.Graph) -> None:
        for _, attrs in graph.nodes(data=True):
            attrs["node_status"] = _coerce_int(attrs.get("node_status", 1), 1)

        for _, _, attrs in graph.edges(data=True):
            attrs["link_status"] = _coerce_int(attrs.get("link_status", 1), 1)
            latency = _coerce_float(attrs.get("link_latency", attrs.get("delay", 0.0)))
            attrs["link_latency"] = latency
            attrs["delay"] = latency

            bandwidth = _coerce_float(
                attrs.get(
                    "link_bandwidth",
                    attrs.get("bandwidth_capacity_available", attrs.get("bandwidth", 0.0)),
                )
            )
            attrs["link_bandwidth"] = bandwidth
            attrs["bandwidth"] = bandwidth

            if "bandwidth_capacity_available" in attrs:
                attrs["bandwidth_capacity_available"] = _coerce_float(
                    attrs.get("bandwidth_capacity_available"), bandwidth
                )
            attrs["link_utilization"] = _coerce_float(attrs.get("link_utilization", 0.0))
            attrs["link_loss_rate"] = _coerce_float(attrs.get("link_loss_rate", 0.0))
