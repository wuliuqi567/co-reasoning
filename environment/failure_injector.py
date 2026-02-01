"""
故障注入相关功能。

支持的故障模式:
    - edge: 随机边故障
    - node: 随机节点故障
    - specified_nodes: 指定节点故障
    - specified_edges: 指定边故障
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import numpy as np
import networkx as nx

from environment.tools import _get_edge_loss_rate, _get_edge_utilization, _is_failed_status


@dataclass
class FailureConfig:
    """故障注入配置
    
    属性:
        enable_failure: 是否启用故障注入
        failure_mode: 故障模式 ("edge" | "node" | "specified_nodes" | "specified_edges")
        fail_num: 随机故障时的故障数量
        fail_step: 故障注入时机 (-1: reset时注入; >=0: 指定步数时注入)
        failure_prob: 故障发生概率 (0.0-1.0)
        ensure_reachable: 是否确保故障后 src-dst 仍可达
        max_failure_tries: 最大重试次数
        utilization_threshold: 链路利用率阈值
        loss_rate_threshold: 链路丢包率阈值
        fail_nodes: 指定要故障的节点列表 (用于 specified_nodes 模式)
        fail_edges: 指定要故障的边列表 (用于 specified_edges 模式)
    """
    enable_failure: bool = False
    failure_mode: str = "edge"  # "edge" | "node" | "specified_nodes" | "specified_edges"
    fail_num: int = 2
    fail_step: int = -1  # -1: reset时注入; >=0: 指定步数时注入
    failure_prob: float = 0.2  # 故障发生概率 (0.0-1.0)，1.0 表示每次都发生
    ensure_reachable: bool = True
    max_failure_tries: int = 30
    utilization_threshold: float = 0.85  # 链路利用率阈值
    loss_rate_threshold: float = 0.1  # 链路丢包率阈值 (默认10%)
    fail_nodes: List[int] = field(default_factory=list)  # 指定故障节点列表
    fail_edges: List[Tuple[int, int]] = field(default_factory=list)  # 指定故障边列表

    @classmethod
    def from_env_config(cls, env_config) -> "FailureConfig":
        # 解析 fail_nodes
        fail_nodes_raw = getattr(env_config, "fail_nodes", [])
        if isinstance(fail_nodes_raw, str):
            # 支持逗号分隔的字符串格式，如 "1,2,3"
            fail_nodes = [int(x.strip()) for x in fail_nodes_raw.split(",") if x.strip()]
        elif isinstance(fail_nodes_raw, (list, tuple)):
            fail_nodes = [int(x) for x in fail_nodes_raw]
        else:
            fail_nodes = []
        
        # 解析 fail_edges
        fail_edges_raw = getattr(env_config, "fail_edges", [])
        if isinstance(fail_edges_raw, str):
            # 支持格式如 "1-2,3-4" 或 "1,2;3,4"
            fail_edges = []
            for pair in fail_edges_raw.replace(";", ",").split(","):
                if "-" in pair:
                    parts = pair.split("-")
                    if len(parts) == 2:
                        fail_edges.append((int(parts[0].strip()), int(parts[1].strip())))
        elif isinstance(fail_edges_raw, (list, tuple)):
            fail_edges = [(int(e[0]), int(e[1])) for e in fail_edges_raw if len(e) >= 2]
        else:
            fail_edges = []
        
        return cls(
            enable_failure=bool(getattr(env_config, "enable_failure", False)),
            failure_mode=getattr(env_config, "failure_mode", "edge"),
            fail_num=int(getattr(env_config, "fail_num", 2)),
            fail_step=int(getattr(env_config, "fail_step", -1)),
            failure_prob=float(getattr(env_config, "failure_prob", 0.2)),
            ensure_reachable=bool(getattr(env_config, "ensure_reachable", True)),
            max_failure_tries=int(getattr(env_config, "max_failure_tries", 30)),
            utilization_threshold=float(getattr(env_config, "utilization_threshold", 0.85)),
            loss_rate_threshold=float(getattr(env_config, "loss_rate_threshold", 0.1)),
            fail_nodes=fail_nodes,
            fail_edges=fail_edges,
        )


class FailureInjector:
    """故障注入器 - 通过修改状态标记模拟故障
    
    支持的故障模式:
        - edge: 随机边故障
        - node: 随机节点故障
        - specified_nodes: 指定节点故障
        - specified_edges: 指定边故障
    """

    def __init__(self, config: FailureConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng

    def inject(
        self,
        base_graph: nx.Graph,
        src: int,
        dst: int
    ) -> Tuple[nx.Graph, List[Tuple[int, int]], List[int]]:
        """
        注入故障 (仅修改状态值, 不删除节点/边)。

        返回: (damaged_graph, dead_edges, dead_nodes)
        """
        # 检查是否启用故障
        if not self.config.enable_failure:
            return base_graph.copy(), [], []
        
        # 指定节点/边模式不需要 fail_num > 0
        is_specified_mode = self.config.failure_mode in ("specified_nodes", "specified_edges")
        if not is_specified_mode and self.config.fail_num <= 0:
            return base_graph.copy(), [], []

        for _ in range(max(1, self.config.max_failure_tries)):
            g = base_graph.copy()
            dead_edges, dead_nodes, node_affected_edges = [], [], []

            if self.config.failure_mode == "edge":
                dead_edges = self._fail_random_edges(g, self.config.fail_num)
            elif self.config.failure_mode == "node":
                dead_nodes, node_affected_edges = self._fail_random_nodes(g, self.config.fail_num, exclude={src, dst})
                dead_edges.extend(node_affected_edges)
            elif self.config.failure_mode == "specified_nodes":
                dead_nodes, node_affected_edges = self._fail_specified_nodes(g, self.config.fail_nodes)
                dead_edges.extend(node_affected_edges)
            elif self.config.failure_mode == "specified_edges":
                dead_edges = self._fail_specified_edges(g, self.config.fail_edges)
            else:
                raise ValueError(f"Unknown failure_mode: {self.config.failure_mode}")

            if not self.config.ensure_reachable:
                return g, dead_edges, dead_nodes

            if self._has_path_without_failed(g, src, dst):
                return g, dead_edges, dead_nodes

        return g, dead_edges, dead_nodes

    def inject_nodes(
        self,
        base_graph: nx.Graph,
        node_ids: List[int],
        src: Optional[int] = None,
        dst: Optional[int] = None
    ) -> Tuple[nx.Graph, List[Tuple[int, int]], List[int]]:
        """
        直接注入指定节点故障 (便捷方法)。
        
        参数:
            base_graph: 基础图
            node_ids: 要故障的节点 ID 列表
            src: 源节点 (用于可达性检查，可选)
            dst: 目标节点 (用于可达性检查，可选)
        
        返回: (damaged_graph, dead_edges, dead_nodes)
        """
        g = base_graph.copy()
        dead_nodes, dead_edges = self._fail_specified_nodes(g, node_ids)
        
        # 可选的可达性检查
        if self.config.ensure_reachable and src is not None and dst is not None:
            if not self._has_path_without_failed(g, src, dst):
                # 如果不可达，返回原图
                return base_graph.copy(), [], []
        
        return g, dead_edges, dead_nodes

    def inject_edges(
        self,
        base_graph: nx.Graph,
        edge_list: List[Tuple[int, int]],
        src: Optional[int] = None,
        dst: Optional[int] = None
    ) -> Tuple[nx.Graph, List[Tuple[int, int]], List[int]]:
        """
        直接注入指定边故障 (便捷方法)。
        
        参数:
            base_graph: 基础图
            edge_list: 要故障的边列表 [(u, v), ...]
            src: 源节点 (用于可达性检查，可选)
            dst: 目标节点 (用于可达性检查，可选)
        
        返回: (damaged_graph, dead_edges, dead_nodes)
        """
        g = base_graph.copy()
        dead_edges = self._fail_specified_edges(g, edge_list)
        
        # 可选的可达性检查
        if self.config.ensure_reachable and src is not None and dst is not None:
            if not self._has_path_without_failed(g, src, dst):
                # 如果不可达，返回原图
                return base_graph.copy(), [], []
        
        return g, dead_edges, []

    def _fail_random_edges(self, g: nx.Graph, k: int) -> List[Tuple[int, int]]:
        """随机标记 k 条边为故障 (link_status = 0)。"""
        edges = [(u, v) for u, v in g.edges() if not _is_failed_status(g[u][v].get("link_status"))]
        if not edges or k <= 0:
            return []
        self.rng.shuffle(edges)
        removed = []
        for u, v in edges[:k]:
            g[u][v]["link_status"] = 0
            removed.append((int(u), int(v)))
        return removed

    def _fail_random_nodes(self, g: nx.Graph, k: int, exclude: Set[int] = None) -> Tuple[List[int], List[Tuple[int, int]]]:
        """随机标记 k 个节点为故障 (node_status = 0)，并同时标记相连的边为故障。

        返回: (故障节点列表, 受影响的边列表)
        """
        exclude = exclude or set()
        nodes = [n for n in g.nodes() if n not in exclude and not _is_failed_status(g.nodes[n].get("node_status"))]
        if not nodes or k <= 0:
            return [], []
        self.rng.shuffle(nodes)
        removed_nodes = []
        affected_edges = []
        for n in nodes[:k]:
            g.nodes[n]["node_status"] = 0
            # 标记与该节点相连的所有边为故障
            for neighbor in list(g.neighbors(n)):
                if not _is_failed_status(g[n][neighbor].get("link_status")):
                    g[n][neighbor]["link_status"] = 0
                    affected_edges.append((int(n), int(neighbor)))
            removed_nodes.append(int(n))
        return removed_nodes, affected_edges

    def _fail_specified_nodes(self, g: nx.Graph, node_ids: List[int]) -> Tuple[List[int], List[Tuple[int, int]]]:
        """标记指定节点为故障 (node_status = 0)，并同时标记相连的边为故障。
        
        参数:
            g: 图对象
            node_ids: 要故障的节点 ID 列表
        
        返回: (故障节点列表, 受影响的边列表)
        """
        if not node_ids:
            return [], []
        
        removed_nodes = []
        affected_edges = []
        
        for n in node_ids:
            # 检查节点是否存在于图中
            if n not in g.nodes():
                continue
            # 检查节点是否已经故障
            if _is_failed_status(g.nodes[n].get("node_status")):
                continue
            
            g.nodes[n]["node_status"] = 0
            # 标记与该节点相连的所有边为故障
            for neighbor in list(g.neighbors(n)):
                if not _is_failed_status(g[n][neighbor].get("link_status")):
                    g[n][neighbor]["link_status"] = 0
                    affected_edges.append((int(n), int(neighbor)))
            removed_nodes.append(int(n))
        
        return removed_nodes, affected_edges

    def _fail_specified_edges(self, g: nx.Graph, edge_list: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """标记指定边为故障 (link_status = 0)。
        
        参数:
            g: 图对象
            edge_list: 要故障的边列表 [(u, v), ...]
        
        返回: 故障边列表
        """
        if not edge_list:
            return []
        
        removed = []
        for u, v in edge_list:
            # 检查边是否存在（考虑无向图）
            if g.has_edge(u, v):
                if not _is_failed_status(g[u][v].get("link_status")):
                    g[u][v]["link_status"] = 0
                    removed.append((int(u), int(v)))
            elif g.has_edge(v, u):
                if not _is_failed_status(g[v][u].get("link_status")):
                    g[v][u]["link_status"] = 0
                    removed.append((int(v), int(u)))
        
        return removed

    def _has_path_without_failed(self, g: nx.Graph, src: int, dst: int) -> bool:
        """判断在过滤故障后是否可达（同时检查利用率和丢包率阈值）。"""
        util_threshold = self.config.utilization_threshold
        loss_threshold = self.config.loss_rate_threshold

        def _node_ok(n):
            return not _is_failed_status(g.nodes[n].get("node_status"))

        def _edge_ok(u, v):
            if _is_failed_status(g[u][v].get("link_status")):
                return False
            if _get_edge_utilization(g[u][v]) > util_threshold:
                return False
            if _get_edge_loss_rate(g[u][v]) > loss_threshold:
                return False
            return True

        view = nx.subgraph_view(g, filter_node=_node_ok, filter_edge=_edge_ok)
        return view.has_node(src) and view.has_node(dst) and nx.has_path(view, src, dst)


# ============================================================================
# 便捷函数
# ============================================================================

def fail_nodes(
    graph: nx.Graph,
    node_ids: List[int],
    rng: Optional[np.random.Generator] = None
) -> Tuple[nx.Graph, List[Tuple[int, int]], List[int]]:
    """
    便捷函数：对指定节点注入故障。
    
    参数:
        graph: 原始图
        node_ids: 要故障的节点 ID 列表
        rng: 随机数生成器 (可选)
    
    返回: (damaged_graph, dead_edges, dead_nodes)
    
    示例:
        >>> damaged_graph, dead_edges, dead_nodes = fail_nodes(graph, [1, 2, 3])
    """
    if rng is None:
        rng = np.random.default_rng()
    
    config = FailureConfig(
        enable_failure=True,
        failure_mode="specified_nodes",
        fail_nodes=list(node_ids),
        ensure_reachable=False,
    )
    injector = FailureInjector(config, rng)
    return injector.inject(graph, src=-1, dst=-1)


def fail_edges(
    graph: nx.Graph,
    edge_list: List[Tuple[int, int]],
    rng: Optional[np.random.Generator] = None
) -> Tuple[nx.Graph, List[Tuple[int, int]], List[int]]:
    """
    便捷函数：对指定边注入故障。
    
    参数:
        graph: 原始图
        edge_list: 要故障的边列表 [(u, v), ...]
        rng: 随机数生成器 (可选)
    
    返回: (damaged_graph, dead_edges, dead_nodes)
    
    示例:
        >>> damaged_graph, dead_edges, dead_nodes = fail_edges(graph, [(0, 1), (2, 3)])
    """
    if rng is None:
        rng = np.random.default_rng()
    
    config = FailureConfig(
        enable_failure=True,
        failure_mode="specified_edges",
        fail_edges=list(edge_list),
        ensure_reachable=False,
    )
    injector = FailureInjector(config, rng)
    return injector.inject(graph, src=-1, dst=-1)
