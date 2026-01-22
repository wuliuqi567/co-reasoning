"""
网络路由环境 (NetTupu) - 重构版

特性:
    1. 支持多种观察模式 (neighbor / matrix / hybrid)
    2. 支持故障注入 (边/节点损毁)
    3. 低耦合设计，易于扩展

观察模式:
    - neighbor: 原始邻居列表模式 [current, dst, (nbr_id, delay, bw) * max_degree]
    - matrix: 矩阵模式 [adj_matrix, delay_matrix, bw_matrix, current_onehot, dst_onehot]
    - hybrid: 两种模式的组合
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Set, Dict, Any

import numpy as np
import networkx as nx
from gymnasium.spaces import Box, Discrete, Dict as DictSpace

from xuance.environment import RawEnvironment

try:
    from environment.generate_topu import generate_topology, load_topology, visualize_topology
except ModuleNotFoundError:
    from generate_topu import generate_topology, load_topology, visualize_topology


# ============================================================================
# 配置数据类
# ============================================================================

@dataclass
class ObservationConfig:
    """观察空间配置"""
    # 观察模式: "neighbor" | "matrix" | "compact" | "hybrid"
    obs_type: str = "compact"
    
    # 矩阵模式配置 (用于 "matrix" 模式)
    include_adjacency: bool = True       # 是否包含邻接矩阵
    include_delay_matrix: bool = True    # 是否包含时延矩阵
    include_bandwidth_matrix: bool = True  # 是否包含带宽矩阵
    use_onehot_nodes: bool = True        # 是否使用 one-hot 编码节点
    
    # 压缩矩阵模式配置 (用于 "compact" 模式)
    # compact 模式：单个 n×n 矩阵 + one-hot 编码，大幅降低维度
    # 矩阵值 = 链路质量指标，编码方式由 compact_encoding 决定
    compact_encoding: str = "delay"  # "quality" | "delay" | "bandwidth" | "combined"
    use_upper_triangle: bool = True   # 是否只用上三角（进一步压缩，无向图适用）
    
    # 矩阵归一化
    normalize_delay: bool = True
    normalize_bandwidth: bool = True
    
    @classmethod
    def from_env_config(cls, env_config) -> "ObservationConfig":
        return cls(
            obs_type=getattr(env_config, "obs_type", "neighbor"),
            include_adjacency=getattr(env_config, "include_adjacency", True),
            include_delay_matrix=getattr(env_config, "include_delay_matrix", True),
            include_bandwidth_matrix=getattr(env_config, "include_bandwidth_matrix", True),
            use_onehot_nodes=getattr(env_config, "use_onehot_nodes", True),
            compact_encoding=getattr(env_config, "compact_encoding", "quality"),
            use_upper_triangle=getattr(env_config, "use_upper_triangle", False),
            normalize_delay=getattr(env_config, "normalize_delay", True),
            normalize_bandwidth=getattr(env_config, "normalize_bandwidth", True),
        )


@dataclass
class FailureConfig:
    """故障注入配置"""
    enable_failure: bool = True
    failure_mode: str = "edge"  # "edge" | "node"
    fail_num: int = 2
    fail_step: int = -1  # -1: reset时注入; >=0: 指定步数时注入
    ensure_reachable: bool = True
    max_failure_tries: int = 30
    
    @classmethod
    def from_env_config(cls, env_config) -> "FailureConfig":
        return cls(
            enable_failure=bool(getattr(env_config, "enable_failure", True)),
            failure_mode=getattr(env_config, "failure_mode", "edge"),
            fail_num=int(getattr(env_config, "fail_num", 2)),
            fail_step=int(getattr(env_config, "fail_step", -1)),
            ensure_reachable=bool(getattr(env_config, "ensure_reachable", True)),
            max_failure_tries=int(getattr(env_config, "max_failure_tries", 30)),
        )


@dataclass
class RewardConfig:
    """奖励配置"""
    loop_penalty: float = -0.5
    timeout_penalty: float = -5.0
    invalid_action_penalty: float = -1.0
    disconnect_penalty: float = -5.0
    step_penalty: float = -0.01
    progress_scale: float = 0.02
    success_base: float = 1.0
    success_scale: float = 9.0
    
    @classmethod
    def from_env_config(cls, env_config) -> "RewardConfig":
        return cls(
            loop_penalty=float(getattr(env_config, "loop_penalty", -0.5)),
            timeout_penalty=float(getattr(env_config, "timeout_penalty", -5.0)),
            invalid_action_penalty=float(getattr(env_config, "invalid_action_penalty", -1.0)),
            disconnect_penalty=float(getattr(env_config, "disconnect_penalty", -5.0)),
            step_penalty=float(getattr(env_config, "step_penalty", -0.01)),
            progress_scale=float(getattr(env_config, "progress_scale", 0.02)),
            success_base=float(getattr(env_config, "success_base", 1.0)),
            success_scale=float(getattr(env_config, "success_scale", 9.0)),
        )


# ============================================================================
# 观察空间构建器
# ============================================================================

class ObservationBuilder:
    """观察空间构建器 - 负责构建不同类型的观察"""
    
    def __init__(
        self,
        num_nodes: int,
        max_degree: int,
        delay_range: Tuple[float, float],
        bandwidth_range: Tuple[float, float],
        config: ObservationConfig
    ):
        self.num_nodes = num_nodes
        self.max_degree = max_degree
        self.delay_range = delay_range
        self.bandwidth_range = bandwidth_range
        self.config = config
        
        # 预计算观察空间维度
        self._compute_observation_spaces()
    
    def _compute_observation_spaces(self):
        """计算各种观察模式的空间"""
        n = self.num_nodes
        
        # 邻居模式维度: [current, dst, (nbr_id, delay, bw) * max_degree]
        self.neighbor_obs_dim = 2 + self.max_degree * 3
        
        # 完整矩阵模式维度 (3 个 n×n 矩阵)
        self.matrix_obs_dim = 0
        if self.config.include_adjacency:
            self.matrix_obs_dim += n * n  # 邻接矩阵
        if self.config.include_delay_matrix:
            self.matrix_obs_dim += n * n  # 时延矩阵
        if self.config.include_bandwidth_matrix:
            self.matrix_obs_dim += n * n  # 带宽矩阵
        if self.config.use_onehot_nodes:
            self.matrix_obs_dim += n * 2  # 当前节点 + 目标节点 one-hot
        
        # 压缩矩阵模式维度 (单个矩阵 + one-hot)
        if self.config.use_upper_triangle:
            # 上三角: n*(n+1)/2 (包含对角线)
            self.compact_matrix_size = n * (n + 1) // 2
        else:
            # 完整矩阵: n*n
            self.compact_matrix_size = n * n
        
        self.compact_obs_dim = self.compact_matrix_size
        if self.config.use_onehot_nodes:
            self.compact_obs_dim += n * 2  # 当前节点 + 目标节点 one-hot
    
    def get_observation_space(self) -> Box:
        """获取观察空间"""
        obs_type = self.config.obs_type
        
        if obs_type == "neighbor":
            return self._get_neighbor_space()
        elif obs_type == "matrix":
            return self._get_matrix_space()
        elif obs_type == "compact":
            return self._get_compact_space()
        elif obs_type == "hybrid":
            return self._get_hybrid_space()
        else:
            raise ValueError(f"Unknown obs_type: {obs_type}")
    
    def _get_neighbor_space(self) -> Box:
        """邻居模式观察空间"""
        obs_low = np.concatenate([
            np.array([0.0, 0.0], dtype=np.float32),
            np.array([-1.0, 0.0, 0.0] * self.max_degree, dtype=np.float32)
        ])
        obs_high = np.concatenate([
            np.array([float(self.num_nodes - 1), float(self.num_nodes - 1)], dtype=np.float32),
            np.array(
                [float(self.num_nodes - 1), float(self.delay_range[1]), float(self.bandwidth_range[1])]
                * self.max_degree,
                dtype=np.float32,
            )
        ])
        return Box(obs_low, obs_high, shape=(self.neighbor_obs_dim,), dtype=np.float32)
    
    def _get_matrix_space(self) -> Box:
        """矩阵模式观察空间"""
        return Box(
            low=0.0,
            high=1.0,  # 归一化后的范围
            shape=(self.matrix_obs_dim,),
            dtype=np.float32
        )
    
    def _get_compact_space(self) -> Box:
        """压缩矩阵模式观察空间 (单个质量矩阵 + one-hot)"""
        return Box(
            low=0.0,
            high=1.0,  # 归一化后的范围
            shape=(self.compact_obs_dim,),
            dtype=np.float32
        )
    
    def _get_hybrid_space(self) -> Box:
        """混合模式观察空间 (neighbor + compact)"""
        total_dim = self.neighbor_obs_dim + self.compact_obs_dim
        return Box(
            low=-1.0,
            high=max(float(self.num_nodes - 1), float(self.delay_range[1]), float(self.bandwidth_range[1]), 1.0),
            shape=(total_dim,),
            dtype=np.float32
        )
    
    def build_observation(
        self,
        graph: nx.Graph,
        current_node: int,
        dst_node: int,
        neighbors: List[int]
    ) -> np.ndarray:
        """构建观察"""
        obs_type = self.config.obs_type
        
        if obs_type == "neighbor":
            return self._build_neighbor_obs(graph, current_node, dst_node, neighbors)
        elif obs_type == "matrix":
            return self._build_matrix_obs(graph, current_node, dst_node)
        elif obs_type == "compact":
            return self._build_compact_obs(graph, current_node, dst_node)
        elif obs_type == "hybrid":
            neighbor_obs = self._build_neighbor_obs(graph, current_node, dst_node, neighbors)
            compact_obs = self._build_compact_obs(graph, current_node, dst_node)
            return np.concatenate([neighbor_obs, compact_obs])
        else:
            raise ValueError(f"Unknown obs_type: {obs_type}")
    
    def _build_neighbor_obs(
        self,
        graph: nx.Graph,
        current_node: int,
        dst_node: int,
        neighbors: List[int]
    ) -> np.ndarray:
        """构建邻居模式观察"""
        obs = np.zeros(self.neighbor_obs_dim, dtype=np.float32)
        obs[0] = float(current_node)
        obs[1] = float(dst_node)
        
        # 填充邻居槽
        for i, node in enumerate(neighbors[:self.max_degree]):
            if graph.has_edge(current_node, node):
                edge_data = graph[current_node][node]
                base = 2 + i * 3
                obs[base] = float(node)
                obs[base + 1] = float(edge_data.get("delay", 0.0))
                obs[base + 2] = float(edge_data.get("bandwidth", 0.0))
        
        # 无效邻居槽
        for j in range(len(neighbors), self.max_degree):
            base = 2 + j * 3
            obs[base] = -1.0
        
        return obs
    
    def _build_matrix_obs(
        self,
        graph: nx.Graph,
        current_node: int,
        dst_node: int
    ) -> np.ndarray:
        """构建矩阵模式观察"""
        n = self.num_nodes
        obs_parts = []
        
        # 邻接矩阵
        if self.config.include_adjacency:
            adj_matrix = self._build_adjacency_matrix(graph)
            obs_parts.append(adj_matrix.flatten())
        
        # 时延矩阵
        if self.config.include_delay_matrix:
            delay_matrix = self._build_delay_matrix(graph)
            if self.config.normalize_delay:
                delay_matrix = self._normalize_matrix(delay_matrix, self.delay_range)
            obs_parts.append(delay_matrix.flatten())
        
        # 带宽矩阵
        if self.config.include_bandwidth_matrix:
            bw_matrix = self._build_bandwidth_matrix(graph)
            if self.config.normalize_bandwidth:
                bw_matrix = self._normalize_matrix(bw_matrix, self.bandwidth_range)
            obs_parts.append(bw_matrix.flatten())
        
        # One-hot 编码
        if self.config.use_onehot_nodes:
            current_onehot = self._build_onehot(current_node, n)
            dst_onehot = self._build_onehot(dst_node, n)
            obs_parts.append(current_onehot)
            obs_parts.append(dst_onehot)
        
        return np.concatenate(obs_parts).astype(np.float32)
    
    def _build_adjacency_matrix(self, graph: nx.Graph) -> np.ndarray:
        """构建邻接矩阵"""
        n = self.num_nodes
        adj = np.zeros((n, n), dtype=np.float32)
        for u, v in graph.edges():
            if u < n and v < n:
                adj[u, v] = 1.0
                adj[v, u] = 1.0
        return adj
    
    def _build_delay_matrix(self, graph: nx.Graph) -> np.ndarray:
        """构建时延矩阵"""
        n = self.num_nodes
        delay = np.zeros((n, n), dtype=np.float32)
        for u, v, data in graph.edges(data=True):
            if u < n and v < n:
                d = float(data.get("delay", 0.0))
                delay[u, v] = d
                delay[v, u] = d
        return delay
    
    def _build_bandwidth_matrix(self, graph: nx.Graph) -> np.ndarray:
        """构建带宽矩阵"""
        n = self.num_nodes
        bw = np.zeros((n, n), dtype=np.float32)
        for u, v, data in graph.edges(data=True):
            if u < n and v < n:
                b = float(data.get("bandwidth", 0.0))
                bw[u, v] = b
                bw[v, u] = b
        return bw
    
    def _build_onehot(self, node: int, n: int) -> np.ndarray:
        """构建 one-hot 编码"""
        onehot = np.zeros(n, dtype=np.float32)
        if 0 <= node < n:
            onehot[node] = 1.0
        return onehot
    
    def _normalize_matrix(self, matrix: np.ndarray, value_range: Tuple[float, float]) -> np.ndarray:
        """归一化矩阵到 [0, 1]"""
        min_val, max_val = value_range
        if max_val - min_val > 1e-6:
            return (matrix - min_val) / (max_val - min_val)
        return matrix
    
    def _build_compact_obs(
        self,
        graph: nx.Graph,
        current_node: int,
        dst_node: int
    ) -> np.ndarray:
        """
        构建压缩矩阵模式观察
        
        将邻接、时延、带宽信息压缩到单个矩阵中，大幅降低维度。
        
        编码方式 (compact_encoding):
            - "quality": 链路质量 = bw_norm / (1 + delay_norm)，越高越好
            - "delay": 仅使用归一化时延
            - "bandwidth": 仅使用归一化带宽
            - "combined": 组合编码 = bw_norm * 0.5 + (1 - delay_norm) * 0.5
        """
        n = self.num_nodes
        obs_parts = []
        
        # 构建压缩质量矩阵
        quality_matrix = self._build_quality_matrix(graph)
        
        # 是否只用上三角
        if self.config.use_upper_triangle:
            # 提取上三角（包含对角线）
            matrix_flat = quality_matrix[np.triu_indices(n)]
        else:
            matrix_flat = quality_matrix.flatten()
        
        obs_parts.append(matrix_flat)
        
        # One-hot 编码
        if self.config.use_onehot_nodes:
            current_onehot = self._build_onehot(current_node, n)
            dst_onehot = self._build_onehot(dst_node, n)
            obs_parts.append(current_onehot)
            obs_parts.append(dst_onehot)
        
        return np.concatenate(obs_parts).astype(np.float32)
    
    def _build_quality_matrix(self, graph: nx.Graph) -> np.ndarray:
        """
        构建链路质量矩阵 - 将邻接、时延、带宽压缩到单个矩阵
        
        矩阵值含义:
            - 0: 无连接
            - >0: 链路质量指标（归一化到 (0, 1]）
        """
        n = self.num_nodes
        quality = np.zeros((n, n), dtype=np.float32)
        
        encoding = self.config.compact_encoding
        
        for u, v, data in graph.edges(data=True):
            if u >= n or v >= n:
                continue
            
            delay = float(data.get("delay", 0.0))
            bandwidth = float(data.get("bandwidth", 0.0))
            
            # 归一化
            delay_norm = self._normalize_value(delay, self.delay_range)
            bw_norm = self._normalize_value(bandwidth, self.bandwidth_range)
            
            # 根据编码方式计算质量值
            if encoding == "quality":
                # 高带宽低时延 = 高质量
                # 公式: bw_norm / (1 + delay_norm)，范围约 (0, 1]
                q = bw_norm / (1.0 + delay_norm)
            elif encoding == "delay":
                # 仅时延（反转，低时延 = 高值）
                q = 1.0 - delay_norm if delay_norm < 1.0 else 0.01
            elif encoding == "bandwidth":
                # 仅带宽
                q = bw_norm if bw_norm > 0 else 0.01
            elif encoding == "combined":
                # 组合：带宽和时延各占一半权重
                q = bw_norm * 0.5 + (1.0 - delay_norm) * 0.5
            else:
                # 默认：仅表示连通性
                q = 1.0
            
            # 确保有连接的边质量 > 0
            q = max(q, 0.01)
            
            quality[u, v] = q
            quality[v, u] = q
        
        return quality
    
    def _normalize_value(self, value: float, value_range: Tuple[float, float]) -> float:
        """归一化单个值到 [0, 1]"""
        min_val, max_val = value_range
        if max_val - min_val > 1e-6:
            return (value - min_val) / (max_val - min_val)
        return 0.0


# ============================================================================
# 故障注入器
# ============================================================================

class FailureInjector:
    """故障注入器 - 负责图的损毁操作"""
    
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
        注入故障
        
        返回:
            (damaged_graph, dead_edges, dead_nodes)
        """
        if not self.config.enable_failure or self.config.fail_num <= 0:
            return base_graph.copy(), [], []
        
        for _ in range(max(1, self.config.max_failure_tries)):
            g = base_graph.copy()
            dead_edges = []
            dead_nodes = []
            
            if self.config.failure_mode == "edge":
                dead_edges = self._remove_random_edges(g, self.config.fail_num)
            elif self.config.failure_mode == "node":
                dead_nodes = self._remove_random_nodes(g, self.config.fail_num, exclude={src, dst})
            else:
                raise ValueError(f"Unknown failure_mode: {self.config.failure_mode}")
            
            # 不要求可达性，直接返回
            if not self.config.ensure_reachable:
                return g, dead_edges, dead_nodes
            
            # 检查可达性
            if g.has_node(src) and g.has_node(dst) and nx.has_path(g, src, dst):
                return g, dead_edges, dead_nodes
        
        # 重试失败，返回最后一次结果
        return g, dead_edges, dead_nodes
    
    def _remove_random_edges(self, g: nx.Graph, k: int) -> List[Tuple[int, int]]:
        """随机移除 k 条边"""
        edges = list(g.edges())
        if len(edges) == 0 or k <= 0:
            return []
        
        self.rng.shuffle(edges)
        removed = []
        
        for u, v in edges:
            if len(removed) >= k:
                break
            if g.has_edge(u, v):
                g.remove_edge(u, v)
                removed.append((int(u), int(v)))
        
        return removed
    
    def _remove_random_nodes(self, g: nx.Graph, k: int, exclude: Set[int] = None) -> List[int]:
        """随机移除 k 个节点"""
        exclude = exclude or set()
        nodes = [n for n in g.nodes() if n not in exclude]
        
        if len(nodes) == 0 or k <= 0:
            return []
        
        self.rng.shuffle(nodes)
        removed = []
        
        for n in nodes:
            if len(removed) >= k:
                break
            if g.has_node(n):
                g.remove_node(n)
                removed.append(int(n))
        
        return removed


# ============================================================================
# 奖励计算器
# ============================================================================

class RewardCalculator:
    """奖励计算器"""
    
    def __init__(self, config: RewardConfig):
        self.config = config
    
    def compute_reward(
        self,
        action: int,
        neighbors: List[int],
        current_node: int,
        next_node: int,
        dst: int,
        path: List[int],
        path_delay: float,
        step_delay: float,
        shortest_path_delay: float,
        dist_to_dst: Dict[int, float],
        is_reachable_current: bool,
        is_reachable_next: bool
    ) -> Tuple[float, bool, str]:
        """
        计算奖励
        
        返回:
            (reward, terminated, reason)
        """
        cfg = self.config
        
        # 无效动作
        if not (0 <= action < len(neighbors)):
            return cfg.invalid_action_penalty, False, "invalid_action"
        
        # 当前节点不可达目标
        if not is_reachable_current:
            return cfg.disconnect_penalty, True, "disconnected_current"
        
        # 下一节点不可达目标（且不是目标本身）
        if not is_reachable_next and next_node != dst:
            return cfg.disconnect_penalty, True, "disconnected_next"
        
        # 到达目标
        if next_node == dst:
            total_delay = path_delay + step_delay
            if not np.isfinite(shortest_path_delay) or shortest_path_delay <= 0.0:
                quality_ratio = 0.0
            else:
                quality_ratio = shortest_path_delay / max(total_delay, 1e-6)
                quality_ratio = float(np.clip(quality_ratio, 0.0, 2.0))
            reward = cfg.success_base + cfg.success_scale * quality_ratio
            return reward, True, "arrive"
        
        # 环路惩罚
        visit_count = path.count(next_node)
        if visit_count > 0:
            reward = cfg.loop_penalty * float(visit_count)
            return reward, False, "loop"
        
        # 正常步进 + 进展奖励
        d_cur = dist_to_dst.get(current_node, 0.0)
        d_next = dist_to_dst.get(next_node, d_cur)
        progress = d_cur - d_next
        reward = cfg.step_penalty + cfg.progress_scale * float(progress)
        
        return float(reward), False, "step"


# ============================================================================
# 主环境类
# ============================================================================

class NetTupu(RawEnvironment):
    """
    网络路由环境
    
    支持多种观察模式:
        - neighbor: 原始邻居列表模式
        - matrix: 矩阵模式（邻接矩阵 + 时延/带宽矩阵 + one-hot 编码）
        - hybrid: 混合模式
    """
    
    def __init__(self, env_config):
        super(NetTupu, self).__init__()
        self.env_id = env_config.env_id
        self.rng = np.random.default_rng()
        
        # ---------------------------------------------------------------------
        # 加载配置
        # ---------------------------------------------------------------------
        self.obs_config = ObservationConfig.from_env_config(env_config)
        self.failure_config = FailureConfig.from_env_config(env_config)
        self.reward_config = RewardConfig.from_env_config(env_config)
        
        # ---------------------------------------------------------------------
        # 加载/生成拓扑
        # ---------------------------------------------------------------------
        default_graph_path = os.path.join(os.path.dirname(__file__), "topology.pkl")
        self.graph_path = getattr(env_config, "graph_path", default_graph_path)
        
        if self.graph_path is not None and os.path.exists(self.graph_path):
            print(f"Loading topology from {self.graph_path}...")
            graph = load_topology(self.graph_path)
            self._sync_graph_attributes(graph)
        else:
            print("Generating new topology...")
            self.num_nodes = getattr(env_config, "num_nodes", 18)
            self.min_degree = getattr(env_config, "min_degree", 2)
            self.max_degree = getattr(env_config, "max_degree", 7)
            self.delay_range = getattr(env_config, "delay_range", (1.0, 10.0))
            self.bandwidth_range = getattr(env_config, "bandwidth_range", (10.0, 100.0))
            self.target_avg_degree = getattr(env_config, "target_avg_degree", None)
            self.target_degrees = getattr(
                env_config, "target_degrees",
                [2, 2, 2, 2, 3, 6, 6, 5, 5, 2, 4, 5, 7, 5, 3, 5, 5, 3],
            )
            graph = self._generate_graph()
        
        # base_graph 永不修改；active_graph 每回合可被损毁
        self.base_graph = graph
        self.active_graph = self.base_graph.copy()
        
        # ---------------------------------------------------------------------
        # 初始化组件
        # ---------------------------------------------------------------------
        self.obs_builder = ObservationBuilder(
            num_nodes=self.num_nodes,
            max_degree=self.max_degree,
            delay_range=self.delay_range,
            bandwidth_range=self.bandwidth_range,
            config=self.obs_config
        )
        
        self.failure_injector = FailureInjector(
            config=self.failure_config,
            rng=self.rng
        )
        
        self.reward_calculator = RewardCalculator(
            config=self.reward_config
        )
        
        # ---------------------------------------------------------------------
        # Episode 状态
        # ---------------------------------------------------------------------
        self.src = None
        self.dst = None
        self.current_node = None
        self.path = []
        self.path_delay = 0.0
        
        self.shortest_path = None
        self.shortest_path_delay = np.inf
        self.dist_to_dst = {}
        
        self.failure_happened = False
        self.dead_edges = []
        self.dead_nodes = []
        
        # ---------------------------------------------------------------------
        # 空间定义
        # ---------------------------------------------------------------------
        self.observation_space = self.obs_builder.get_observation_space()
        self.action_space = Discrete(n=self.max_degree)
        
        self.max_episode_steps = int(getattr(env_config, "max_episode_steps", 64))
        self._current_step = 0
    
    # =========================================================================
    # Core API
    # =========================================================================
    
    def reset(self, **kwargs):
        self._current_step = 0
        self.path = []
        self.path_delay = 0.0
        
        # 每回合恢复活跃图
        self.active_graph = self.base_graph.copy()
        self.failure_happened = False
        self.dead_edges = []
        self.dead_nodes = []
        
        if kwargs.get("regenerate_graph", False):
            self.regenerate_topology()
        
        # 采样 src/dst
        self._sample_src_dst()
        
        # reset 时注入故障
        if self.failure_config.enable_failure and self.failure_config.fail_step < 0:
            self._inject_failure()
        
        # 初始化 episode
        self.current_node = self.src
        self.path.append(self.current_node)
        
        self._recompute_shortest_and_dists()
        
        observation = self._build_observation()
        info = self._build_info(extra={"reset": True})
        return observation, info
    
    def step(self, action):
        self._current_step += 1
        
        # 动态损毁
        if (self.failure_config.enable_failure
            and not self.failure_happened
            and self.failure_config.fail_step >= 0
            and self._current_step == self.failure_config.fail_step):
            self._inject_failure()
            self._recompute_shortest_and_dists()
        
        neighbors = self._get_neighbor_list(self.current_node)
        
        # 当前节点无邻居
        if len(neighbors) == 0:
            reward = self.reward_config.disconnect_penalty
            terminated = True
            truncated = False
            observation = self._build_observation()
            info = self._build_info(extra={
                "action_idx": int(action),
                "chosen_node": -1,
                "step_delay": 0.0,
                "is_loop": False,
                "visit_count": 0,
                "terminated_reason": "no_neighbors",
            })
            return observation, reward, terminated, truncated, info
        
        # 选择的节点
        chosen_node = neighbors[int(action)] if 0 <= int(action) < len(neighbors) else -1
        
        # 计算奖励和转移
        if chosen_node != -1:
            edge_data = self.active_graph[self.current_node][chosen_node]
            step_delay = float(edge_data.get("delay", 0.0))
            is_reachable_current = self._is_reachable_from(self.current_node)
            is_reachable_next = self._is_reachable_from(chosen_node)
        else:
            step_delay = 0.0
            is_reachable_current = True
            is_reachable_next = True
        
        reward, terminated, reason = self.reward_calculator.compute_reward(
            action=int(action),
            neighbors=neighbors,
            current_node=self.current_node,
            next_node=chosen_node if chosen_node != -1 else self.current_node,
            dst=self.dst,
            path=self.path,
            path_delay=self.path_delay,
            step_delay=step_delay,
            shortest_path_delay=self.shortest_path_delay,
            dist_to_dst=self.dist_to_dst,
            is_reachable_current=is_reachable_current,
            is_reachable_next=is_reachable_next
        )
        
        # 状态推进
        if chosen_node != -1 and reason != "invalid_action":
            self.current_node = chosen_node
            self.path_delay += step_delay
        self.path.append(self.current_node)
        
        observation = self._build_observation()
        truncated = self._current_step >= self.max_episode_steps
        
        # 超时惩罚
        if truncated and not terminated:
            reward += self.reward_config.timeout_penalty
        
        visit_count = self.path[:-1].count(chosen_node) if chosen_node != -1 else 0
        
        info = self._build_info(extra={
            "action_idx": int(action),
            "chosen_node": int(chosen_node) if chosen_node != -1 else -1,
            "step_delay": float(step_delay),
            "is_loop": visit_count > 0,
            "visit_count": int(visit_count),
            "terminated_reason": reason,
        })
        
        return observation, reward, terminated, truncated, info
    
    def render(self, *args, **kwargs):
        return self._build_info(extra={})
    
    def close(self):
        pass
    
    # =========================================================================
    # Topology Operations
    # =========================================================================
    
    def visualize(self, **kwargs):
        return visualize_topology(self.active_graph, **kwargs)
    
    def print_topology(self):
        for node in sorted(self.active_graph.nodes()):
            neighbors = sorted(self.active_graph.neighbors(node))
            print(f"node {node}: degree={len(neighbors)}, neighbors={neighbors}")
    
    def regenerate_topology(self):
        self.base_graph = self._generate_graph()
        self.active_graph = self.base_graph.copy()
        self._sync_graph_attributes(self.base_graph)
        
        # 重建观察构建器
        self.obs_builder = ObservationBuilder(
            num_nodes=self.num_nodes,
            max_degree=self.max_degree,
            delay_range=self.delay_range,
            bandwidth_range=self.bandwidth_range,
            config=self.obs_config
        )
        self.observation_space = self.obs_builder.get_observation_space()
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _sample_src_dst(self):
        nodes = list(self.base_graph.nodes())
        selected = self.rng.choice(nodes, size=2, replace=False)
        self.src, self.dst = int(selected[0]), int(selected[1])
    
    def _get_neighbor_list(self, node: int) -> List[int]:
        if not self.active_graph.has_node(node):
            return []
        return sorted(self.active_graph.neighbors(node))
    
    def _get_action_mask(self) -> np.ndarray:
        neighbors = self._get_neighbor_list(self.current_node)
        num_neighbors = len(neighbors)
        mask = np.zeros(self.max_degree, dtype=bool)
        mask[:min(num_neighbors, self.max_degree)] = True
        return mask
    
    def _build_observation(self) -> np.ndarray:
        neighbors = self._get_neighbor_list(self.current_node)
        return self.obs_builder.build_observation(
            graph=self.active_graph,
            current_node=self.current_node,
            dst_node=self.dst,
            neighbors=neighbors
        )
    
    def _inject_failure(self):
        self.failure_happened = True
        self.active_graph, self.dead_edges, self.dead_nodes = self.failure_injector.inject(
            base_graph=self.base_graph,
            src=self.src,
            dst=self.dst
        )
    
    def _recompute_shortest_and_dists(self):
        self.shortest_path = None
        self.shortest_path_delay = np.inf
        self.dist_to_dst = {}
        
        if not self.active_graph.has_node(self.dst):
            return
        
        try:
            self.dist_to_dst = nx.single_source_dijkstra_path_length(
                self.active_graph, self.dst, weight="delay"
            )
        except Exception:
            self.dist_to_dst = {}
        
        if self.active_graph.has_node(self.src) and (self.src in self.dist_to_dst):
            try:
                self.shortest_path = nx.shortest_path(
                    self.active_graph, self.src, self.dst, weight="delay"
                )
                self.shortest_path_delay = float(
                    self._calculate_path_delay(self.shortest_path, graph=self.active_graph)
                )
            except Exception:
                self.shortest_path = None
                self.shortest_path_delay = float(self.dist_to_dst.get(self.src, np.inf))
    
    def _is_reachable_from(self, node: int) -> bool:
        return node in self.dist_to_dst
    
    def _build_info(self, extra: Dict[str, Any] = None) -> Dict[str, Any]:
        if extra is None:
            extra = {}
        
        is_connected = (
            self.active_graph.has_node(self.src)
            and self.active_graph.has_node(self.dst)
            and nx.has_path(self.active_graph, self.src, self.dst)
        ) if (self.active_graph.has_node(self.src) and self.active_graph.has_node(self.dst)) else False
        
        info = {
            "src": int(self.src) if self.src is not None else None,
            "dst": int(self.dst) if self.dst is not None else None,
            "current_node": int(self.current_node) if self.current_node is not None else None,
            "path": self.path.copy(),
            "shortest_path": self.shortest_path,
            "shortest_path_delay": float(self.shortest_path_delay) if np.isfinite(self.shortest_path_delay) else None,
            "path_delay": float(self.path_delay),
            "action_mask": self._get_action_mask(),
            # 故障信息
            "failure_happened": bool(self.failure_happened),
            "failure_mode": self.failure_config.failure_mode,
            "fail_step": int(self.failure_config.fail_step),
            "fail_num": int(self.failure_config.fail_num),
            "dead_edges": self.dead_edges.copy(),
            "dead_nodes": self.dead_nodes.copy(),
            "is_connected_src_dst": bool(is_connected),
        }
        info.update(extra)
        return info
    
    # =========================================================================
    # Graph Helpers
    # =========================================================================
    
    def _generate_graph(self) -> nx.Graph:
        return generate_topology(
            self.num_nodes,
            self.min_degree,
            self.max_degree,
            self.delay_range,
            self.bandwidth_range,
            self.rng,
            target_avg_degree=self.target_avg_degree,
            target_degrees=self.target_degrees,
        )
    
    def _sync_graph_attributes(self, graph: nx.Graph):
        self.num_nodes = graph.number_of_nodes()
        degrees = [degree for _, degree in graph.degree()]
        self.min_degree = min(degrees) if degrees else 0
        self.max_degree = max(degrees) if degrees else 0
        
        delays = [data.get("delay") for _, _, data in graph.edges(data=True) if "delay" in data]
        bandwidths = [data.get("bandwidth") for _, _, data in graph.edges(data=True) if "bandwidth" in data]
        self.delay_range = (
            float(min(delays)) if delays else 0.0,
            float(max(delays)) if delays else 0.0
        )
        self.bandwidth_range = (
            float(min(bandwidths)) if bandwidths else 0.0,
            float(max(bandwidths)) if bandwidths else 0.0
        )
    
    def _calculate_path_delay(self, path: List[int], graph: nx.Graph) -> float:
        if path is None or len(path) < 2:
            return 0.0
        total = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if graph.has_edge(u, v):
                total += float(graph[u][v].get("delay", 0.0))
        return total


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    class _Config:
        env_id = "NetTupu-Debug"
        delay_range = (1.0, 10.0)
        bandwidth_range = (10.0, 100.0)
        seed = 0
        graph_path = "topology.pkl"
        
        # 观察配置
        obs_type = "compact"  # "neighbor" | "matrix" | "compact" | "hybrid"
        include_adjacency = True
        include_delay_matrix = True
        include_bandwidth_matrix = True
        use_onehot_nodes = True
        normalize_delay = True
        normalize_bandwidth = True
        
        # 压缩模式配置
        compact_encoding = "quality"  # "quality" | "delay" | "bandwidth" | "combined"
        use_upper_triangle = False    # True 可进一步压缩维度
        
        # 故障配置
        enable_failure = True
        failure_mode = "edge"
        fail_num = 2
        fail_step = -1
        ensure_reachable = True
    
    print("=" * 60)
    print("观察模式维度对比 (num_nodes=18, max_degree=7)")
    print("=" * 60)
    
    # 测试不同观察模式的维度
    for obs_type in ["neighbor", "matrix", "compact", "hybrid"]:
        _Config.obs_type = obs_type
        _Config.use_upper_triangle = False
        env = NetTupu(env_config=_Config())
        obs, _ = env.reset()
        print(f"{obs_type:10} 模式: 维度 = {obs.shape[0]}")
    
    # 测试压缩模式 + 上三角
    _Config.obs_type = "compact"
    _Config.use_upper_triangle = True
    env = NetTupu(env_config=_Config())
    obs, _ = env.reset()
    print(f"{'compact+tri':10} 模式: 维度 = {obs.shape[0]} (上三角压缩)")
    
    print()
    print("=" * 60)
    print("压缩编码方式测试")
    print("=" * 60)
    
    _Config.obs_type = "compact"
    _Config.use_upper_triangle = False
    
    for encoding in ["quality", "delay", "bandwidth", "combined"]:
        _Config.compact_encoding = encoding
        env = NetTupu(env_config=_Config())
        obs, info = env.reset()
        print(f"编码方式: {encoding:10} | src={info['src']}, dst={info['dst']}")
