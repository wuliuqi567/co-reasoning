"""
网络路由环境 (NetTupu) - 重构版

特性:
    1. 支持两种观察模式 (neighbor / state)
    2. 支持故障注入 (边/节点损毁)
    3. 支持链路利用率阈值过滤

观察模式:
    - neighbor: 邻居列表模式 [current, dst, (nbr_id, delay, bw) * max_degree]
    - state: 状态模式 [节点特征 + 邻居槽位 + one-hot]

状态模式设计:
    对每个节点 i:
        - node_status: 在线=1, 故障=0
        - visited: 已访问标记
        - 邻居槽位 (max_degree 个, 每槽 5 维):
            [mask, delay_norm, utilization, link_on, loss_rate]
    拼接后追加:
        - 当前节点 one-hot
        - 目的节点 one-hot
"""

import os
import sys
# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List, Set, Dict, Any

import numpy as np
import networkx as nx
from gymnasium.spaces import Box, Discrete

from xuance.environment import RawEnvironment


# ============================================================================
# 辅助函数
# ============================================================================

def _coerce_float(value: Any, default: float = 0.0) -> float:
    """将输入值转换为浮点数。"""
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_int(value: Any, default: int = 0) -> int:
    """将输入值转换为整数。"""
    if value is None or value == "":
        return int(default)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _get_edge_latency(edge_data: Dict[str, Any]) -> float:
    """从边属性获取链路时延 (优先 link_latency, 其次 delay)。"""
    return _coerce_float(edge_data.get("link_latency", edge_data.get("delay", 0.0)))


def _get_edge_bandwidth(edge_data: Dict[str, Any]) -> float:
    """从边属性获取链路带宽 (优先 link_bandwidth, 其次 bandwidth)。"""
    return _coerce_float(edge_data.get("link_bandwidth", edge_data.get("bandwidth", 0.0)))


def _get_edge_utilization(edge_data: Dict[str, Any]) -> float:
    """从边属性获取链路利用率。"""
    return _coerce_float(edge_data.get("link_utilization", 0.0))


def _get_edge_loss_rate(edge_data: Dict[str, Any]) -> float:
    """从边属性获取链路丢包率。"""
    return _coerce_float(edge_data.get("link_loss_rate", 0.0))


def _is_failed_status(value: Any) -> bool:
    """判断状态值是否为故障标记 (-1)。"""
    try:
        return int(float(value)) == -1
    except (TypeError, ValueError):
        return False


# ============================================================================
# 配置数据类
# ============================================================================

@dataclass
class ObservationConfig:
    """观察空间配置"""
    obs_type: str = "state"  # "state" | "neighbor"
    normalize_delay: bool = True
    normalize_bandwidth: bool = True

    @classmethod
    def from_env_config(cls, env_config) -> "ObservationConfig":
        return cls(
            obs_type=getattr(env_config, "obs_type", "state"),
            normalize_delay=getattr(env_config, "normalize_delay", True),
            normalize_bandwidth=getattr(env_config, "normalize_bandwidth", True),
        )


@dataclass
class FailureConfig:
    """故障注入配置"""
    enable_failure: bool = False
    failure_mode: str = "edge"  # "edge" | "node"
    fail_num: int = 2
    fail_step: int = -1  # -1: reset时注入; >=0: 指定步数时注入
    failure_prob: float = 0.2  # 故障发生概率 (0.0-1.0)，1.0 表示每次都发生
    ensure_reachable: bool = True
    max_failure_tries: int = 30
    utilization_threshold: float = 0.85  # 链路利用率阈值
    loss_rate_threshold: float = 0.1  # 链路丢包率阈值 (默认10%)

    @classmethod
    def from_env_config(cls, env_config) -> "FailureConfig":
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
    suboptimal_penalty: float = -2.0  # 最后一跳绕路的惩罚

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
            suboptimal_penalty=float(getattr(env_config, "suboptimal_penalty", -2.0)),
        )


# ============================================================================
# 观察空间构建器
# ============================================================================

class ObservationBuilder:
    """观察空间构建器 - 支持 neighbor 和 state 两种模式"""

    def __init__(
        self,
        num_nodes: int,
        max_degree: int,
        delay_range: Tuple[float, float],
        bandwidth_range: Tuple[float, float],
        config: ObservationConfig
    ):
        """
        初始化观察构建器。

        参数:
            num_nodes: 节点总数
            max_degree: 最大度
            delay_range: 时延范围 (min, max)
            bandwidth_range: 带宽范围 (min, max)
            config: 观察配置
        """
        self.num_nodes = num_nodes
        self.max_degree = max_degree
        self.delay_range = delay_range
        self.bandwidth_range = bandwidth_range
        self.config = config
        self._compute_dimensions()

    def _compute_dimensions(self):
        """计算观测维度。"""
        n = self.num_nodes
        d = self.max_degree

        # neighbor 模式: [current, dst, (nbr_id, delay, bw) * max_degree]
        self.neighbor_obs_dim = 2 + d * 3

        # state 模式:
        # 每节点: node_status(1) + visited(1) + 邻居槽位(max_degree * 5)
        # 全局: current_onehot(n) + dst_onehot(n)
        self.state_obs_dim = n * (2 + d * 5) + 2 * n

    def get_observation_space(self) -> Box:
        """获取观测空间。"""
        if self.config.obs_type == "neighbor":
            return self._get_neighbor_space()
        elif self.config.obs_type == "state":
            return self._get_state_space()
        else:
            raise ValueError(f"Unsupported obs_type: {self.config.obs_type}")

    def _get_neighbor_space(self) -> Box:
        """neighbor 模式观测空间。"""
        obs_low = np.concatenate([
            np.array([0.0, 0.0], dtype=np.float32),
            np.array([-1.0, 0.0, 0.0] * self.max_degree, dtype=np.float32)
        ])
        obs_high = np.concatenate([
            np.array([float(self.num_nodes - 1), float(self.num_nodes - 1)], dtype=np.float32),
            np.array([float(self.num_nodes - 1), float(self.delay_range[1]), float(self.bandwidth_range[1])] * self.max_degree, dtype=np.float32)
        ])
        return Box(obs_low, obs_high, shape=(self.neighbor_obs_dim,), dtype=np.float32)

    def _get_state_space(self) -> Box:
        """state 模式观测空间。"""
        return Box(low=0.0, high=1.0, shape=(self.state_obs_dim,), dtype=np.float32)

    def build_observation(
        self,
        graph: nx.Graph,
        current_node: int,
        dst_node: int,
        neighbors: List[int],
        path: Optional[List[int]] = None
    ) -> np.ndarray:
        """构建观测向量。"""
        if self.config.obs_type == "neighbor":
            return self._build_neighbor_obs(graph, current_node, dst_node, neighbors)
        elif self.config.obs_type == "state":
            return self._build_state_obs(graph, current_node, dst_node, path or [])
        else:
            raise ValueError(f"Unsupported obs_type: {self.config.obs_type}")

    def _build_neighbor_obs(
        self,
        graph: nx.Graph,
        current_node: int,
        dst_node: int,
        neighbors: List[int]
    ) -> np.ndarray:
        """
        构建 neighbor 模式观测。

        格式: [current, dst, (nbr_id, delay, bw) * max_degree]
        """
        obs = np.zeros(self.neighbor_obs_dim, dtype=np.float32)
        obs[0] = float(current_node)
        obs[1] = float(dst_node)

        for i, nbr in enumerate(neighbors[:self.max_degree]):
            if graph.has_edge(current_node, nbr):
                data = graph[current_node][nbr]
                base = 2 + i * 3
                obs[base] = float(nbr)
                obs[base + 1] = _get_edge_latency(data)
                obs[base + 2] = _get_edge_bandwidth(data)

        for j in range(len(neighbors), self.max_degree):
            obs[2 + j * 3] = -1.0

        return obs

    def _build_state_obs(
        self,
        graph: nx.Graph,
        current_node: int,
        dst_node: int,
        path: List[int]
    ) -> np.ndarray:
        """
        构建 state 模式观测。

        格式:
            [节点特征拼接] + [当前节点 one-hot] + [目的节点 one-hot]

        每节点特征:
            - node_status: 在线=1, 故障=0
            - visited: 已访问=1, 未访问=0
            - 邻居槽位 (max_degree 个, 每槽 5 维):
                [mask, delay_norm, utilization, link_on, loss_rate]
        """
        n = self.num_nodes
        d = self.max_degree

        # 节点在线状态
        node_online = np.zeros(n, dtype=np.float32)
        for i in range(n):
            if graph.has_node(i):
                status = graph.nodes[i].get("node_status", 1)
                node_online[i] = 0.0 if _is_failed_status(status) else 1.0

        # 已访问标记
        visited = np.zeros(n, dtype=np.float32)
        for node in path:
            if 0 <= int(node) < n:
                visited[int(node)] = 1.0

        # 邻居槽位: [mask, delay_norm, utilization, link_on, loss_rate]
        neighbor_slots = np.zeros((n, d, 5), dtype=np.float32)
        for i in range(n):
            if not graph.has_node(i):
                continue
            nbrs = sorted(graph.neighbors(i))
            for k in range(min(len(nbrs), d)):
                nbr = nbrs[k]
                data = graph[i][nbr]
                delay = _get_edge_latency(data)
                delay_norm = self._normalize_value(delay, self.delay_range)
                utilization = float(np.clip(_get_edge_utilization(data), 0.0, 1.0))
                link_on = 0.0 if _is_failed_status(data.get("link_status")) else 1.0
                loss_rate = float(np.clip(_coerce_float(data.get("link_loss_rate", 0.0)), 0.0, 1.0))
                neighbor_slots[i, k, :] = [1.0, delay_norm, utilization, link_on, loss_rate]

        # one-hot 编码
        current_onehot = np.zeros(n, dtype=np.float32)
        dst_onehot = np.zeros(n, dtype=np.float32)
        if 0 <= current_node < n:
            current_onehot[current_node] = 1.0
        if 0 <= dst_node < n:
            dst_onehot[dst_node] = 1.0
        
        # 先构建每个节点的特征向量，再整体拼接
        # 单节点特征: [node_online, visited] + [邻居槽位 (max_degree * 5)]
        per_node = []
        for i in range(n):
            per_node.append(np.concatenate([
                np.array([node_online[i], visited[i]], dtype=np.float32),
                neighbor_slots[i].flatten(),
            ]))
        node_features = np.concatenate(per_node) if per_node else np.array([], dtype=np.float32)

        return np.concatenate([
            node_features,
            current_onehot,
            dst_onehot,
        ]).astype(np.float32)

    def _normalize_value(self, value: float, value_range: Tuple[float, float]) -> float:
        """归一化值到 [0, 1]。"""
        min_val, max_val = value_range
        if max_val - min_val > 1e-6:
            return float(np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0))
        return 0.0


# ============================================================================
# 故障注入器
# ============================================================================

class FailureInjector:
    """故障注入器 - 通过修改状态标记模拟故障"""

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
        if not self.config.enable_failure or self.config.fail_num <= 0:
            return base_graph.copy(), [], []

        for _ in range(max(1, self.config.max_failure_tries)):
            g = base_graph.copy()
            dead_edges, dead_nodes, node_affected_edges = [], [], []

            if self.config.failure_mode == "edge":
                dead_edges = self._fail_random_edges(g, self.config.fail_num)
            elif self.config.failure_mode == "node":
                dead_nodes, node_affected_edges = self._fail_random_nodes(g, self.config.fail_num, exclude={src, dst})
                # 将节点故障导致的边故障也添加到 dead_edges
                dead_edges.extend(node_affected_edges)
            else:
                raise ValueError(f"Unknown failure_mode: {self.config.failure_mode}")

            if not self.config.ensure_reachable:
                return g, dead_edges, dead_nodes

            if self._has_path_without_failed(g, src, dst):
                return g, dead_edges, dead_nodes

        return g, dead_edges, dead_nodes

    def _fail_random_edges(self, g: nx.Graph, k: int) -> List[Tuple[int, int]]:
        """随机标记 k 条边为故障 (link_status = -1)。"""
        edges = [(u, v) for u, v in g.edges() if not _is_failed_status(g[u][v].get("link_status"))]
        if not edges or k <= 0:
            return []
        self.rng.shuffle(edges)
        removed = []
        for u, v in edges[:k]:
            g[u][v]["link_status"] = -1
            removed.append((int(u), int(v)))
        return removed

    def _fail_random_nodes(self, g: nx.Graph, k: int, exclude: Set[int] = None) -> Tuple[List[int], List[Tuple[int, int]]]:
        """随机标记 k 个节点为故障 (node_status = -1)，并同时标记相连的边为故障。
        
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
            g.nodes[n]["node_status"] = -1
            # 标记与该节点相连的所有边为故障
            for neighbor in list(g.neighbors(n)):
                if not _is_failed_status(g[n][neighbor].get("link_status")):
                    g[n][neighbor]["link_status"] = -1
                    affected_edges.append((int(n), int(neighbor)))
            removed_nodes.append(int(n))
        return removed_nodes, affected_edges

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
        计算奖励。

        返回: (reward, terminated, reason)
        """
        cfg = self.config

        # 无效动作
        if not (0 <= action < len(neighbors)):
            return cfg.invalid_action_penalty, False, "invalid_action"

        # 断连检查
        if not is_reachable_current:
            return cfg.disconnect_penalty, True, "disconnected_current"
        if not is_reachable_next:
            return cfg.disconnect_penalty, True, "disconnected_next"

        # 到达目标
        if next_node == dst:
            total_delay = path_delay + step_delay
            
            # 无法计算最短路径延迟时，给基础奖励
            if not np.isfinite(shortest_path_delay) or shortest_path_delay <= 0.0:
                return cfg.success_base, True, "arrive"
            
            # 获取当前节点到目的地的最短距离
            d_cur = dist_to_dst.get(current_node, np.inf)
            
            # 判断最后一跳是否是最优的
            # d_cur 是从 current_node 到 dst 的最短路径延迟
            # step_delay 是实际走的这一跳的延迟
            # 如果 d_cur ≈ step_delay，说明直接走就是最短的（最后一跳最优）
            # 如果 d_cur < step_delay，说明有更短的路径但没选（最后一跳绕路）
            is_last_hop_optimal = np.isfinite(d_cur) and np.isclose(d_cur, step_delay, rtol=0.01)
            
            # 判断整条路径是否是最优的
            is_path_optimal = np.isclose(total_delay, shortest_path_delay, rtol=0.01)
            
            if is_path_optimal:
                # 整条路径都是最优的，给最高奖励
                quality_ratio = 1.0
                return cfg.success_base + cfg.success_scale * quality_ratio, True, "arrive_optimal"
            elif is_last_hop_optimal:
                # 最后一跳是最优的，但整条路径不是（之前绕路了）
                # 根据路径质量给奖励
                quality_ratio = float(np.clip(shortest_path_delay / max(total_delay, 1e-6), 0.0, 1.0))
                return cfg.success_base + cfg.success_scale * quality_ratio, True, "arrive"
            else:
                # 最后一跳也绕路了，给惩罚
                # 惩罚程度与绕路比例相关
                detour_ratio = (step_delay - d_cur) / max(d_cur, 1e-6) if np.isfinite(d_cur) and d_cur > 0 else 1.0
                penalty = cfg.suboptimal_penalty * min(detour_ratio, 1.0)
                return cfg.success_base + penalty, True, "arrive_suboptimal"

        # 环路惩罚
        visit_count = path.count(next_node)
        if visit_count > 0:
            return cfg.loop_penalty * float(visit_count), False, "loop"

        # 正常步进
        d_cur = dist_to_dst.get(current_node, np.inf)
        d_next = dist_to_dst.get(next_node, np.inf)
        if not np.isfinite(d_cur) or not np.isfinite(d_next):
            # 节点不可达时无进度奖励
            progress = 0.0
        else:
            progress = d_cur - d_next
        return cfg.step_penalty + cfg.progress_scale * float(progress), False, "step"


# ============================================================================
# 主环境类
# ============================================================================

class NetTupu(RawEnvironment):
    """网络路由环境"""

    def __init__(self, env_config):
        super(NetTupu, self).__init__()
        self.env_id = env_config.env_id
        self.rng = np.random.default_rng()

        # 加载配置
        self.obs_config = ObservationConfig.from_env_config(env_config)
        self.failure_config = FailureConfig.from_env_config(env_config)
        self.reward_config = RewardConfig.from_env_config(env_config)

        # 加载/生成拓扑
        # graph_source: "random_example" | "history" | "random" | 自定义路径

        # env_id = "NetTupu"
        # graph_source = "random_example"  # 使用固定图
        # graph_source = "history"       # 从history随机选择
        # graph_source = "random"        # 从random随机选择
        # graph_source = "/path/to/custom.graphml"  # 自定义路径
        
        self.graph_source = getattr(env_config, "graph_source", "random_example")
        self.graph_data_dir = Path(os.path.dirname(__file__)) / "graph_data"
        
        graph = self._load_graph_by_source(self.graph_source)

        if graph is not None:
            self._normalize_graph_attributes(graph)
            graph, self.status_dead_edges, self.status_dead_nodes = self._apply_status_failures(graph)
            self._sync_graph_attributes(graph)


        self.base_graph = graph
        self.active_graph = self.base_graph.copy()

        # 初始化组件
        self.obs_builder = ObservationBuilder(
            num_nodes=self.num_nodes,
            max_degree=self.max_degree,
            delay_range=self.delay_range,
            bandwidth_range=self.bandwidth_range,
            config=self.obs_config
        )
        self.failure_injector = FailureInjector(config=self.failure_config, rng=self.rng)
        self.reward_calculator = RewardCalculator(config=self.reward_config)

        # Episode 状态
        self.test_or_train = getattr(env_config, "test", False) and getattr(env_config, "execute_reroute", False)
        if self.test_or_train:
            self.src = env_config.src
            self.dst = env_config.dst
        else:
            self.src, self.dst, self.current_node = None, None, None
        self.path, self.path_delay = [], 0.0
        self.shortest_path, self.shortest_path_delay = None, np.inf
        self.dist_to_dst = {}
        self.failure_happened, self.dead_edges, self.dead_nodes = False, [], []
        # 故障前的路径信息（用于对比）
        self.path_before_failure = None
        self.shortest_path_before_failure = None

        # 空间定义
        self.observation_space = self.obs_builder.get_observation_space()
        self.action_space = Discrete(n=self.max_degree)
        self.max_episode_steps = int(getattr(env_config, "max_episode_steps", 64))
        self._current_step = 0

    # =========================================================================
    # Core API
    # =========================================================================

    def reset(self, **kwargs):
        """重置环境。"""
        self._current_step = 0
        self.path, self.path_delay = [], 0.0
        self.active_graph = self.base_graph.copy()
        self.failure_happened, self.dead_edges, self.dead_nodes = False, [], []
        self.path_before_failure = None
        self.shortest_path_before_failure = None

        if kwargs.get("regenerate_graph", False):
            self.regenerate_topology()

        
        if self.test_or_train:
            if self.src is None or self.dst is None:
                raise ValueError("test_or_train=True 需要传入 src 和 dst")
            self.src = int(self.src)
            self.dst = int(self.dst)
        else:
            self._sample_src_dst()

        # 根据概率决定是否注入故障
        if self.failure_config.enable_failure and self.failure_config.fail_step < 0:
            if self.rng.random() < self.failure_config.failure_prob:
                self._inject_failure()

        self.current_node = self.src
        self.path.append(self.current_node)
        self._recompute_shortest_and_dists()

        extra = {"reset": True, "test_or_train": self.test_or_train} if not self.test_or_train else {"reset": True, "src": self.src, "dst": self.dst, "test_or_train": self.test_or_train}
        return self._build_observation(), self._build_info(extra=extra)

    def step(self, action):
        """执行动作。"""
        self._current_step += 1

        # 动态损毁（根据概率决定是否注入）
        if (self.failure_config.enable_failure
            and not self.failure_happened
            and self.failure_config.fail_step >= 0
            and self._current_step == self.failure_config.fail_step):
            if self.rng.random() < self.failure_config.failure_prob:
                self._inject_failure()
                self._recompute_shortest_and_dists()

        neighbors = self._get_neighbor_list(self.current_node)

        if not neighbors:
            return (
                self._build_observation(),
                self.reward_config.disconnect_penalty,
                True, False,
                self._build_info(extra={"action_idx": int(action), "chosen_node": -1, "terminated_reason": "no_neighbors"})
            )

        chosen_node = neighbors[int(action)] if 0 <= int(action) < len(neighbors) else -1

        # 先计算当前节点的可达性
        is_reachable_current = self._is_reachable_from(self.current_node)

        if chosen_node != -1:
            step_delay = _get_edge_latency(self.active_graph[self.current_node][chosen_node])
            is_reachable_next = self._is_reachable_from(chosen_node)
        else:
            # 无效动作：不移动，next就是current
            step_delay = 0.0
            is_reachable_next = is_reachable_current

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

        # 只有在有效移动时才更新状态（排除无效动作和断连情况）
        if chosen_node != -1 and reason not in ("invalid_action", "disconnected_current", "disconnected_next"):
            self.current_node = chosen_node
            self.path_delay += step_delay
        self.path.append(self.current_node)

        truncated = self._current_step >= self.max_episode_steps
        if truncated and not terminated:
            reward += self.reward_config.timeout_penalty

        info = self._build_info(extra={
            "action_idx": int(action),
            "chosen_node": int(chosen_node) if chosen_node != -1 else -1,
            "step_delay": float(step_delay),
            "terminated_reason": reason,
        })
        return self._build_observation(), reward, terminated, truncated, info

    def render(self, *args, **kwargs):
        return self._build_info()

    def close(self):
        pass

    # =========================================================================
    # Topology Operations
    # =========================================================================

    def visualize(self, **kwargs):
        visualize_fn = self._resolve_visualize()
        return visualize_fn(self.active_graph, **kwargs)

    def regenerate_topology(self):
        """重新生成拓扑。"""
        self.base_graph = self._generate_graph()
        self._normalize_graph_attributes(self.base_graph)
        self.status_dead_edges, self.status_dead_nodes = [], []
        self.active_graph = self.base_graph.copy()
        self._sync_graph_attributes(self.base_graph)
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
        """采样源/目标节点。"""
        nodes = [n for n in self.base_graph.nodes() if not _is_failed_status(self.base_graph.nodes[n].get("node_status"))]
        if len(nodes) < 2:
            nodes = list(self.base_graph.nodes())
        selected = self.rng.choice(nodes, size=2, replace=False)
        self.src, self.dst = int(selected[0]), int(selected[1])
    
    def set_src_dst_idx(self, src: int, dst: int):
        self.src = src
        self.dst = dst

    def set_src_dst_ip(self, src: str, dst: str):
        map = self.get_manage_ip_to_node_id_map()
        self.src = map[src]
        self.dst = map[dst]

    def get_manage_ip_to_node_id_map(self, graph: Optional[nx.Graph] = None) -> Dict[str, str]:
        """构建节点 node_manage_ip_addr -> node_id 映射。"""
        g = graph or self.base_graph
        mapping: Dict[str, str] = {}
        if g is None:
            return mapping
        for _, attrs in g.nodes(data=True):
            node_id = attrs.get("node_id")
            ip_addr = attrs.get("node_manage_ip_addr")
            if node_id and ip_addr:
                mapping[str(ip_addr)] = str(node_id)
        return mapping

    def get_idx_to_ip_map(self, graph: Optional[nx.Graph] = None) -> Dict[int, str]:
        """构建节点 idx -> node_manage_ip_addr 映射。"""
        g = graph or self.base_graph
        mapping: Dict[int, str] = {}
        if g is None:
            return mapping
        for node, attrs in g.nodes(data=True):
            idx = attrs.get("idx", node)
            ip_addr = attrs.get("node_manage_ip_addr", "")
            mapping[int(idx)] = str(ip_addr)
        return mapping

    def _path_to_ip_port(self, path: List[int], graph: Optional[nx.Graph] = None) -> List[Dict[str, str]]:
        """
        将节点路径转换为 IP:port 格式。
        
        返回格式: [
            {"node_idx": 0, "ip": "192.168.1.1", "out_port": "node_id:1"},
            {"node_idx": 1, "ip": "192.168.1.2", "in_port": "node_id:2", "out_port": "node_id:3"},
            ...
            {"node_idx": n, "ip": "192.168.1.n", "in_port": "node_id:x"}
        ]
        """
        g = graph or self.active_graph
        if g is None or len(path) == 0:
            return []

        result = []
        idx_to_ip = self.get_idx_to_ip_map(g)

        for i, node_idx in enumerate(path):
            node_attrs = g.nodes.get(node_idx, {})
            ip = idx_to_ip.get(node_idx, node_attrs.get("node_manage_ip_addr", ""))
            
            entry = {
                "node_idx": node_idx,
                "ip": ip,
            }

            # 获取入端口 (来自前一跳)
            if i > 0:
                prev_node = path[i - 1]
                if g.has_edge(prev_node, node_idx):
                    edge_data = g[prev_node][node_idx]
                    # 对于无向图，需要判断方向
                    entry["in_port"] = edge_data.get("dst_port", "")

            # 获取出端口 (到下一跳)
            if i < len(path) - 1:
                next_node = path[i + 1]
                if g.has_edge(node_idx, next_node):
                    edge_data = g[node_idx][next_node]
                    entry["out_port"] = edge_data.get("src_port", "")

            result.append(entry)

        return result

    

    def _is_node_failed(self, node: int) -> bool:
        if not self.active_graph.has_node(node):
            return True
        return _is_failed_status(self.active_graph.nodes[node].get("node_status"))

    def _is_edge_unusable(self, u: int, v: int) -> bool:
        """判断边是否不可用 (故障、利用率超阈值或丢包率超阈值)。"""
        if not self.active_graph.has_edge(u, v):
            return True
        data = self.active_graph[u][v]
        if _is_failed_status(data.get("link_status")):
            return True
        if _get_edge_utilization(data) > self.failure_config.utilization_threshold:
            return True
        if _get_edge_loss_rate(data) > self.failure_config.loss_rate_threshold:
            return True
        return False

    def _get_routing_graph(self) -> nx.Graph:
        """获取过滤故障/拥塞后的图视图。"""
        def _node_ok(n):
            return not self._is_node_failed(n)

        def _edge_ok(u, v):
            return not self._is_edge_unusable(u, v)

        return nx.subgraph_view(self.active_graph, filter_node=_node_ok, filter_edge=_edge_ok)

    def _get_neighbor_list(self, node: int) -> List[int]:
        if self._is_node_failed(node):
            return []
        routing_graph = self._get_routing_graph()
        if not routing_graph.has_node(node):
            return []
        return sorted(routing_graph.neighbors(node))

    def _get_action_mask(self) -> np.ndarray:
        neighbors = self._get_neighbor_list(self.current_node)
        mask = np.zeros(self.max_degree, dtype=bool)
        mask[:min(len(neighbors), self.max_degree)] = True
        return mask

    def _build_observation(self) -> np.ndarray:
        neighbors = self._get_neighbor_list(self.current_node)
        graph_for_obs = self.active_graph if self.obs_config.obs_type == "state" else self._get_routing_graph()
        return self.obs_builder.build_observation(
            graph=graph_for_obs,
            current_node=self.current_node,
            dst_node=self.dst,
            neighbors=neighbors,
            path=self.path,
        )

    def _inject_failure(self):
        # 保存故障前的路径信息
        self.path_before_failure = self.path.copy() if self.path else None
        # 计算故障前的最短路径
        try:
            routing_graph_before = self._get_routing_graph()
            if (routing_graph_before.has_node(self.src) and routing_graph_before.has_node(self.dst)
                and nx.has_path(routing_graph_before, self.src, self.dst)):
                self.shortest_path_before_failure = nx.shortest_path(
                    routing_graph_before, self.src, self.dst, weight="link_latency"
                )
            else:
                self.shortest_path_before_failure = None
        except Exception:
            self.shortest_path_before_failure = None
        
        self.failure_happened = True
        self.active_graph, self.dead_edges, self.dead_nodes = self.failure_injector.inject(
            base_graph=self.base_graph, src=self.src, dst=self.dst
        )

    def _recompute_shortest_and_dists(self):
        self.shortest_path, self.shortest_path_delay, self.dist_to_dst = None, np.inf, {}
        routing_graph = self._get_routing_graph()
        if not routing_graph.has_node(self.dst):
            return
        try:
            self.dist_to_dst = nx.single_source_dijkstra_path_length(routing_graph, self.dst, weight="link_latency")
        except Exception:
            self.dist_to_dst = {}
        if routing_graph.has_node(self.src) and self.src in self.dist_to_dst:
            try:
                self.shortest_path = nx.shortest_path(routing_graph, self.src, self.dst, weight="link_latency")
                self.shortest_path_delay = float(self._calculate_path_delay(self.shortest_path))
            except Exception:
                self.shortest_path_delay = float(self.dist_to_dst.get(self.src, np.inf))

    def _is_reachable_from(self, node: int) -> bool:
        return node in self.dist_to_dst

    def _build_info(self, extra: Dict[str, Any] = None) -> Dict[str, Any]:
        routing_graph = self._get_routing_graph()
        is_connected = (
            routing_graph.has_node(self.src) and routing_graph.has_node(self.dst)
            and nx.has_path(routing_graph, self.src, self.dst)
        ) if (self.src is not None and self.dst is not None) else False

        # 计算当前路径的指标
        path_metrics = self._get_path_metrics(self.path) if self.path else {"delay": 0.0, "bandwidth": 0.0, "loss_rate": 0.0}
        
        # 计算最短路径的指标
        shortest_path_metrics = self._get_path_metrics(self.shortest_path) if self.shortest_path else {"delay": 0.0, "bandwidth": 0.0, "loss_rate": 0.0}

        info = {
            "src": self.src,
            "dst": self.dst,
            "current_node": self.current_node,
            # 当前路径信息
            "path": self.path.copy(),
            "path_ip_port": self._path_to_ip_port(self.path),
            "path_delay": float(self.path_delay),
            "path_bandwidth": float(path_metrics["bandwidth"]),
            "path_loss_rate": float(path_metrics["loss_rate"]),
            # 最短路径信息
            "shortest_path": self.shortest_path,
            "shortest_path_ip_port": self._path_to_ip_port(self.shortest_path) if self.shortest_path else [],
            "shortest_path_delay": float(self.shortest_path_delay) if np.isfinite(self.shortest_path_delay) else None,
            "shortest_path_bandwidth": float(shortest_path_metrics["bandwidth"]),
            "shortest_path_loss_rate": float(shortest_path_metrics["loss_rate"]),
            # 动作掩码
            "action_mask": self._get_action_mask(),
            # 故障信息
            "failure_happened": self.failure_happened,
            "failure_mode": self.failure_config.failure_mode,
            "fail_step": self.failure_config.fail_step,
            "fail_num": self.failure_config.fail_num,
            "dead_edges": self.status_dead_edges + self.dead_edges,
            "dead_nodes": self.status_dead_nodes + self.dead_nodes,
            "is_connected_src_dst": is_connected,
        }

        # 如果发生故障，添加故障前的路径信息
        if self.failure_happened:
            info["path_before_failure"] = self.path_before_failure
            info["shortest_path_before_failure"] = self.shortest_path_before_failure
            
            # 计算故障前最短路径的指标（在 base_graph 上计算）
            if self.shortest_path_before_failure:
                before_metrics = {
                    "delay": self._calculate_path_delay_on_graph(self.shortest_path_before_failure, self.base_graph),
                    "bandwidth": self._calculate_path_bandwidth(self.shortest_path_before_failure, self.base_graph),
                    "loss_rate": self._calculate_path_loss_rate(self.shortest_path_before_failure, self.base_graph),
                }
                info["shortest_path_before_failure_delay"] = float(before_metrics["delay"])
                info["shortest_path_before_failure_bandwidth"] = float(before_metrics["bandwidth"])
                info["shortest_path_before_failure_loss_rate"] = float(before_metrics["loss_rate"])
            
            # 检查故障是否影响了之前的最短路径
            if self.shortest_path_before_failure:
                affected = self._is_path_affected_by_failure(self.shortest_path_before_failure)
                info["failure_affected_original_path"] = affected

        if extra:
            info.update(extra)
        return info

    def _is_path_affected_by_failure(self, path: List[int]) -> bool:
        """检查路径是否受故障影响。"""
        if not path or len(path) < 2:
            return False
        
        all_dead_edges = set()
        for e in (self.status_dead_edges + self.dead_edges):
            all_dead_edges.add((e[0], e[1]))
            all_dead_edges.add((e[1], e[0]))  # 无向图
        
        all_dead_nodes = set(self.status_dead_nodes + self.dead_nodes)
        
        # 检查路径上的节点是否故障
        for node in path:
            if node in all_dead_nodes:
                return True
        
        # 检查路径上的边是否故障
        for i in range(len(path) - 1):
            if (path[i], path[i + 1]) in all_dead_edges:
                return True
        
        return False

    # =========================================================================
    # Graph Helpers
    # =========================================================================

    def _generate_graph(self) -> nx.Graph:
        generate_random_topology = self._resolve_generate_random_topology()
        seed = int(self.rng.integers(0, 1_000_000_000))
        return generate_random_topology(
            num_nodes=self.num_nodes,
            min_nodes=self.num_nodes,
            max_nodes=self.num_nodes,
            min_degree=self.min_degree,
            max_degree=self.max_degree,
            seed=seed,
        )

    @staticmethod
    def _resolve_visualize():
        """延迟导入可视化函数，兼容脚本/包两种运行方式。"""
        try:
            from environment.topo_parse.topo_parser import visualize  # type: ignore
            return visualize
        except (ModuleNotFoundError, ImportError):
            try:
                from topo_parse.topo_parser import visualize  # type: ignore
                return visualize
            except (ModuleNotFoundError, ImportError):
                from .topo_parse.topo_parser import visualize  # type: ignore
                return visualize

    @staticmethod
    def _resolve_generate_random_topology():
        """延迟导入随机拓扑生成器，兼容脚本/包两种运行方式。"""
        try:
            from environment.topo_parse.gen_graph import generate_random_topology  # type: ignore
            return generate_random_topology
        except (ModuleNotFoundError, ImportError):
            try:
                from topo_parse.gen_graph import generate_random_topology  # type: ignore
                return generate_random_topology
            except (ModuleNotFoundError, ImportError):
                from .topo_parse.gen_graph import generate_random_topology  # type: ignore
                return generate_random_topology

    def _load_graph_by_source(self, source: str) -> Optional[nx.Graph]:
        """
        根据 graph_source 配置加载图。
        
        参数:
            source: 图来源
                - "random_example": 固定加载 random/random_example.graphml
                - "history": 从 history 目录随机选择一个图
                - "random": 从 random 目录随机选择一个图
                - 其他: 视为自定义文件/目录路径
        
        返回:
            nx.Graph 或 None
        """
        if source == "random_example":
            # 固定加载 random/random_example.graphml
            target = self.graph_data_dir / "random" / "random_example.graphml"
            if target.exists():
                print(f"Loading fixed topology: {target}")
                return self._load_graph_file(target)
            else:
                print(f"Warning: random_example.graphml not found at {target}")
                return None
        
        elif source == "random":
            # 从 random 目录随机选择一个图
            random_dir = self.graph_data_dir / "random"
            if not random_dir.exists():
                print(f"Warning: random directory not found: {random_dir}")
                return None
            return self._load_random_from_dir(random_dir)

        elif source == "latest_II_class":
            # 加载 II 类网络拓扑图
            return self._load_graph_file(self.graph_data_dir / "latest_II_class.graphml")

        elif source == "II_class_history_random":
            # 从 II 类网络拓扑图随机目录随机选择一个图
            history_dir = self.graph_data_dir / "II_class_history"
            if not history_dir.exists():
                print(f"Warning: II_class_history directory not found: {history_dir}")
                return None
            return self._load_random_from_dir(history_dir)

        else:
            # 自定义路径
            return self._load_graph_from_path(source)
    
    def _load_random_from_dir(self, dir_path: Path) -> Optional[nx.Graph]:
        """从目录中随机选择一个图文件加载。"""
        files = sorted(dir_path.glob("*.graphml")) + sorted(dir_path.glob("*.pkl"))
        if not files:
            print(f"Warning: no graph files found in {dir_path}")
            return None
        selected = files[int(self.rng.integers(0, len(files)))]
        print(f"Loading random topology from {selected}...")
        return self._load_graph_file(selected)

    def _load_graph_from_path(self, path: str) -> Optional[nx.Graph]:
        """从指定路径加载图（支持文件或目录）。"""
        path_obj = Path(path)
        if path_obj.is_dir():
            return self._load_random_from_dir(path_obj)
        if path_obj.is_file():
            print(f"Loading topology from {path_obj}...")
            return self._load_graph_file(path_obj)
        return None

    def _load_graph_file(self, path_obj: Path) -> Optional[nx.Graph]:
        suffix = path_obj.suffix.lower()
        if suffix == ".graphml":
            graph = nx.read_graphml(str(path_obj))
        elif suffix in (".pkl", ".pickle"):
            with open(path_obj, "rb") as f:
                graph = pickle.load(f)
        else:
            return None
        return self._relabel_graph_nodes(graph)

    def _relabel_graph_nodes(self, graph: nx.Graph) -> nx.Graph:
        """将节点重编号为连续整数，并设置 idx 属性。"""
        nodes = list(graph.nodes())
        if not nodes:
            return graph

        # 保存原始 ID
        for n in nodes:
            graph.nodes[n].setdefault("orig_id", str(n))

        # 按数值或字符串排序
        try:
            sorted_nodes = sorted(nodes, key=lambda n: int(float(n)))
        except (ValueError, TypeError):
            sorted_nodes = sorted(nodes, key=str)

        # 重编号
        mapping = {node: idx for idx, node in enumerate(sorted_nodes)}
        new_graph = nx.relabel_nodes(graph, mapping, copy=True)

        # 设置 idx 属性 (整数索引)
        for node in new_graph.nodes():
            new_graph.nodes[node]["idx"] = int(node)

        return new_graph

    def _normalize_graph_attributes(self, graph: nx.Graph) -> None:
        """规范化图属性，确保所有必需字段存在。"""
        for node, attrs in graph.nodes(data=True):
            # 确保 idx 属性存在
            if "idx" not in attrs:
                attrs["idx"] = int(node)
            attrs["node_status"] = _coerce_int(attrs.get("node_status", 1), 1)
        for _, _, data in graph.edges(data=True):
            data["link_status"] = _coerce_int(data.get("link_status", 1), 1)
            latency = _coerce_float(data.get("link_latency", data.get("delay", 0.0)))
            data["link_latency"] = latency
            data["delay"] = latency
            bandwidth = _coerce_float(data.get("link_bandwidth", data.get("bandwidth", 0.0)))
            data["link_bandwidth"] = bandwidth
            data["bandwidth"] = bandwidth

    def _apply_status_failures(self, graph: nx.Graph) -> Tuple[nx.Graph, List[Tuple[int, int]], List[int]]:
        dead_nodes = [int(n) for n, attrs in graph.nodes(data=True) if _is_failed_status(attrs.get("node_status"))]
        dead_edges = [(int(u), int(v)) for u, v, data in graph.edges(data=True) if _is_failed_status(data.get("link_status"))]
        return graph, dead_edges, dead_nodes

    def _sync_graph_attributes(self, graph: nx.Graph):
        self.num_nodes = graph.number_of_nodes()
        degrees = [d for _, d in graph.degree()]
        self.min_degree = min(degrees) if degrees else 0
        self.max_degree = max(degrees) if degrees else 0
        delays = [_get_edge_latency(data) for _, _, data in graph.edges(data=True)]
        bandwidths = [_get_edge_bandwidth(data) for _, _, data in graph.edges(data=True)]
        self.delay_range = (min(delays) if delays else 0.0, max(delays) if delays else 0.0)
        self.bandwidth_range = (min(bandwidths) if bandwidths else 0.0, max(bandwidths) if bandwidths else 0.0)

    def _calculate_path_delay(self, path: List[int]) -> float:
        if not path or len(path) < 2:
            return 0.0
        total = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.active_graph.has_edge(u, v):
                total += _get_edge_latency(self.active_graph[u][v])
        return total

    def _calculate_path_bandwidth(self, path: List[int], graph: nx.Graph = None) -> float:
        """计算路径的瓶颈带宽（最小带宽）。"""
        g = graph or self.active_graph
        if not path or len(path) < 2:
            return 0.0
        min_bandwidth = float('inf')
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if g.has_edge(u, v):
                bw = _get_edge_bandwidth(g[u][v])
                if bw < min_bandwidth:
                    min_bandwidth = bw
            else:
                return 0.0  # 边不存在
        return min_bandwidth if np.isfinite(min_bandwidth) else 0.0

    def _calculate_path_loss_rate(self, path: List[int], graph: nx.Graph = None) -> float:
        """计算路径的总丢包率（1 - 各段成功率的乘积）。"""
        g = graph or self.active_graph
        if not path or len(path) < 2:
            return 0.0
        success_rate = 1.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if g.has_edge(u, v):
                edge_loss = _get_edge_loss_rate(g[u][v])
                success_rate *= (1.0 - edge_loss)
            else:
                return 1.0  # 边不存在，完全丢包
        return 1.0 - success_rate

    def _get_path_metrics(self, path: List[int], graph: nx.Graph = None) -> Dict[str, float]:
        """获取路径的综合指标。"""
        g = graph or self.active_graph
        return {
            "delay": self._calculate_path_delay(path) if g == self.active_graph else self._calculate_path_delay_on_graph(path, g),
            "bandwidth": self._calculate_path_bandwidth(path, g),
            "loss_rate": self._calculate_path_loss_rate(path, g),
        }

    def _calculate_path_delay_on_graph(self, path: List[int], graph: nx.Graph) -> float:
        """在指定图上计算路径延迟。"""
        if not path or len(path) < 2:
            return 0.0
        total = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if graph.has_edge(u, v):
                total += _get_edge_latency(graph[u][v])
        return total


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    class _Config:
        env_id = "NetTupu-Debug"
        obs_type = "state"
        enable_failure = False
        failure_mode = "edge"
        fail_num = 2
        fail_step = -1

    env = NetTupu(env_config=_Config())
    obs, info = env.reset()
    # 结构化输出，便于核对维度与取值
    if _Config.obs_type == "state":
        n = env.num_nodes
        d = env.max_degree
        node_feat_len = n * (2 + d * 5)
        node_features = obs[:node_feat_len].reshape(n, 2 + d * 5)
        current_onehot = obs[node_feat_len: node_feat_len + n]
        dst_onehot = obs[node_feat_len + n: node_feat_len + 2 * n]

        print("obs segments:")
        print(f"  node_features: {node_features.shape}")
        print(f"  current_onehot: {current_onehot.shape}")
        print(f"  dst_onehot: {dst_onehot.shape}")

        cur_idx = int(np.argmax(current_onehot)) if current_onehot.sum() > 0 else -1
        dst_idx = int(np.argmax(dst_onehot)) if dst_onehot.sum() > 0 else -1
        print(f"  current idx: {cur_idx}, dst idx: {dst_idx}")

        show_nodes = min(7, n)
        show_slots = min(7, d)
        for i in range(show_nodes):
            feat = node_features[i]
            node_online = feat[0]
            visited = feat[1]
            slots = feat[2:].reshape(d, 5)
            active = np.where(slots[:, 0] > 0.0)[0]
            print(f"node {i}: online={node_online:.0f}, visited={visited:.0f}, active_slots={len(active)}")
            for s in active[:show_slots]:
                mask, delay, util, link_on, loss = slots[s]
                print(f"  slot{s}: mask={mask:.0f}, delay={delay:.3f}, util={util:.3f}, link_on={link_on:.0f}, loss={loss:.3f}")
            if len(active) > show_slots:
                print(f"  ... {len(active) - show_slots} more slots")
    else:
        print(f"obs: {obs}")
    print(f"obs_type: {_Config.obs_type}")
    print(f"obs shape: {obs.shape}")
    print(f"obs dim: {obs.shape[0]}")
    print(f"num_nodes: {env.num_nodes}, max_degree: {env.max_degree}")
    print(f"expected state dim: {env.num_nodes * (2 + env.max_degree * 5) + 2 * env.num_nodes}")
    print(f"src: {info['src']}, dst: {info['dst']}")
