import os
import numpy as np
import networkx as nx
from gymnasium.spaces import Box, Discrete
from xuance.environment import RawEnvironment

try:
    from environment.generate_topu import generate_topology, load_topology, visualize_topology
except ModuleNotFoundError:
    from generate_topu import generate_topology, load_topology, visualize_topology


class NetTupu(RawEnvironment):
    """
    NetTupu with resilience (anti-damage) support.

    New env_config fields (all optional):
      - failure_mode: "none" | "link" | "node"
      - fail_num: int (how many links/nodes to fail)
      - fail_num_range: (min, max) -> random int each episode (overrides fail_num)
      - fail_step: int (0 means fail at reset; >0 means fail during episode)
      - fail_step_range: (min, max) -> random int each episode (overrides fail_step)
      - protect_src_dst: bool (default True) do not remove src/dst as failed nodes
      - protect_current_node: bool (default True) when node failure happens mid-episode, do not remove current node
      - disconnect_penalty: float (default -10.0) when graph becomes unreachable (from current to dst), end early and penalize
      - max_reset_tries: int (default 50) when fail_step==0 and damage may make episode unsolvable
    """

    def __init__(self, env_config):
        super(NetTupu, self).__init__()
        self.env_id = env_config.env_id  # The environment id.

        # 默认路径相对于项目根目录
        default_graph_path = os.path.join(os.path.dirname(__file__), "topology.pkl")
        self.graph_path = getattr(env_config, "graph_path", default_graph_path)

        self.rng = np.random.default_rng()

        if self.graph_path is not None:
            print(f"Loading topology from {self.graph_path}...")
            self.graph = load_topology(self.graph_path)
            self._sync_graph_attributes(self.graph)
        else:
            print("Generating new topology...")
            self.num_nodes = getattr(env_config, "num_nodes", 18)
            self.min_degree = getattr(env_config, "min_degree", 2)
            self.max_degree = getattr(env_config, "max_degree", 7)
            self.delay_range = getattr(env_config, "delay_range", (1.0, 10.0))
            self.bandwidth_range = getattr(env_config, "bandwidth_range", (10.0, 100.0))
            self.target_avg_degree = getattr(env_config, "target_avg_degree", None)
            self.target_degrees = getattr(
                env_config,
                "target_degrees",
                [2, 2, 2, 2, 3, 6, 6, 5, 5, 2, 4, 5, 7, 5, 3, 5, 5, 3],
            )
            self.graph = self._generate_graph()

        # ---------- Resilience / damage configs ----------
        self.failure_mode = getattr(env_config, "failure_mode", "none")  # "none" | "link" | "node"
        self.fail_num = int(getattr(env_config, "fail_num", 0))
        self.fail_num_range = getattr(env_config, "fail_num_range", None)  # tuple/list (min,max)
        self.fail_step = int(getattr(env_config, "fail_step", 0))  # 0 at reset; >0 during episode
        self.fail_step_range = getattr(env_config, "fail_step_range", None)  # tuple/list (min,max)
        self.protect_src_dst = bool(getattr(env_config, "protect_src_dst", True))
        self.protect_current_node = bool(getattr(env_config, "protect_current_node", True))
        self.disconnect_penalty = float(getattr(env_config, "disconnect_penalty", -10.0))
        self.max_reset_tries = int(getattr(env_config, "max_reset_tries", 50))

        # base_graph 永远不变；active_graph 每个 episode / 中途损毁会变化
        self.base_graph = self.graph
        self.active_graph = self.base_graph.copy()

        # ---------- Episode states ----------
        self.src = None
        self.dst = None
        self.current_node = None
        self.path = []
        self.path_delay = 0.0  # 累积路径时延
        self.shortest_path_delay = None  # 最短路径时延（基于 active_graph）
        self.shortest_path = None

        # damage bookkeeping
        self.dead_edges = set()  # set of (u,v) with u<v
        self.dead_nodes = set()  # set of node ids
        self.failure_happened = False
        self._episode_fail_step = None
        self._episode_fail_num = 0
        self.is_connected_src_dst = True  # src->dst on active_graph (for baseline existence)

        # 奖励函数配置
        self.loop_penalty = getattr(env_config, "loop_penalty", -0.5)  # 环路惩罚
        self.timeout_penalty = getattr(env_config, "timeout_penalty", -5.0)  # 超时惩罚

        # 观测空间: [current_node, dst_node, neighbor_info × max_degree]
        # 每个邻居: [node_id, delay, bandwidth]
        obs_dim = 2 + self.max_degree * 3
        obs_low = np.concatenate([
            np.array([0.0, 0.0], dtype=np.float32),  # current_node, dst_node
            np.array([-1.0, 0.0, 0.0] * self.max_degree, dtype=np.float32)  # neighbors
        ])
        obs_high = np.concatenate([
            np.array([float(self.num_nodes - 1), float(self.num_nodes - 1)], dtype=np.float32),
            np.array(
                [float(self.num_nodes - 1), float(self.delay_range[1]), float(self.bandwidth_range[1])]
                * self.max_degree,
                dtype=np.float32,
            )
        ])

        self.observation_space = Box(obs_low, obs_high, shape=(obs_dim,))
        self.action_space = Discrete(n=self.max_degree)

        self.max_episode_steps = 64
        self._current_step = 0

    # --------------------------- Core APIs ---------------------------

    def reset(self, **kwargs):
        self._current_step = 0
        self.path = []
        self.path_delay = 0.0

        if kwargs.get("regenerate_graph", False):
            self.regenerate_topology()

        # 这一回合随机决定 fail_num / fail_step（若给了 range）
        self._episode_fail_num = self._sample_episode_fail_num()
        self._episode_fail_step = self._sample_episode_fail_step()

        # 为了避免 fail_step==0 时一上来就断开导致“无解episode”，做重试采样
        for _ in range(self.max_reset_tries):
            # 每次尝试都从 base_graph 重新采 src/dst
            self._sample_src_dst(graph=self.base_graph)
            self.current_node = self.src
            self.path = [self.current_node]
            self.path_delay = 0.0

            # 每回合重置 active_graph
            self.active_graph = self.base_graph.copy()
            self.dead_edges, self.dead_nodes = set(), set()
            self.failure_happened = False

            # 若 fail_step==0，在 reset 就施加损毁
            if self.failure_mode != "none" and self._episode_fail_step == 0 and self._episode_fail_num > 0:
                self._apply_failure(now_step=0)

            # 计算 shortest baseline（基于 active_graph）
            self._recompute_shortest_baseline()

            # 若此时 src->dst 不可达且损毁发生在 reset，就再试一次（重采 src/dst）
            if self.is_connected_src_dst:
                break
        else:
            # 重试仍失败：保底（不损毁）让 episode 可用
            self.active_graph = self.base_graph.copy()
            self.dead_edges, self.dead_nodes = set(), set()
            self.failure_happened = False
            self._recompute_shortest_baseline()

        observation = self._build_observation()
        info = self._build_info(extra={
            "reset": True
        })
        return observation, info

    def step(self, action):
        self._current_step += 1

        # 中途施加损毁（若设置 fail_step>0）
        if (
            (not self.failure_happened)
            and self.failure_mode != "none"
            and (self._episode_fail_num > 0)
            and (self._episode_fail_step is not None)
            and (self._episode_fail_step > 0)
            and (self._current_step == self._episode_fail_step)
        ):
            self._apply_failure(now_step=self._current_step)
            self._recompute_shortest_baseline()

        # 若当前节点在 active_graph 中已不存在（极端情况：允许移除当前节点）
        if self.current_node not in self.active_graph:
            reward = self.disconnect_penalty
            terminated = True
            truncated = False
            observation = self._build_observation()
            info = self._build_info(extra={
                "failure_note": "current_node_removed"
            })
            return observation, reward, terminated, truncated, info

        neighbors = self._get_neighbor_list(self.current_node, graph=self.active_graph)

        # 记录实际选择的节点（-1 表示无效动作）
        chosen_node = neighbors[action] if (0 <= action < len(neighbors)) else -1

        # 先判断：损毁后从当前节点到 dst 是否仍可达；不可达就提前结束（抗毁伤训练关键）
        if not self._is_reachable(self.current_node, self.dst, graph=self.active_graph):
            reward = self.disconnect_penalty
            terminated = True
            truncated = False
            observation = self._build_observation()
            info = self._build_info(extra={
                "failure_note": "unreachable_after_failure",
                "chosen_node": chosen_node,
                "action_idx": int(action),
                "step_delay": 0.0,
            })
            return observation, reward, terminated, truncated, info

        # 计算奖励 + 下一节点
        reward, terminated, next_node, step_delay = self._compute_reward(action, neighbors)

        # 状态推进
        self.current_node = next_node
        self.path.append(self.current_node)
        self.path_delay += step_delay

        observation = self._build_observation()

        truncated = self._current_step >= self.max_episode_steps
        if truncated and not terminated:
            reward += self.timeout_penalty

        # 环路统计（不包括刚刚添加的 current_node）
        visit_count = self.path[:-1].count(chosen_node) if chosen_node != -1 else 0

        info = self._build_info(extra={
            "action_idx": int(action),
            "chosen_node": chosen_node,
            "step_delay": step_delay,
            "is_loop": visit_count > 0,
            "visit_count": visit_count,
        })

        return observation, reward, terminated, truncated, info

    def render(self, *args, **kwargs):
        return self._build_info()

    def visualize(self, **kwargs):
        # 默认可视化 active_graph；如需基础图可传 use_base=True
        use_base = bool(kwargs.pop("use_base", False))
        G = self.base_graph if use_base else self.active_graph
        return visualize_topology(G, **kwargs)

    def print_topology(self, use_base=False):
        G = self.base_graph if use_base else self.active_graph
        for node in sorted(G.nodes()):
            neighbors = sorted(G.neighbors(node))
            degree = len(neighbors)
            neighbor_str = ", ".join(str(n) for n in neighbors)
            print(f"node {node}: degree={degree}, neighbors=[{neighbor_str}]")

    def regenerate_topology(self):
        self.graph = self._generate_graph()
        self._sync_graph_attributes(self.graph)
        self.base_graph = self.graph
        self.active_graph = self.base_graph.copy()

    def close(self):
        return

    # --------------------------- Helpers ---------------------------

    def _build_info(self, extra=None):
        if extra is None:
            extra = {}
        info = {
            "src": self.src,
            "dst": self.dst,
            "current_node": self.current_node,
            "path": self.path.copy(),
            "path_delay": float(self.path_delay),
            "shortest_path": self.shortest_path,
            "shortest_path_delay": self.shortest_path_delay if self.shortest_path_delay is not None else -1.0,
            "action_mask": self._get_action_mask(),
            # failure metadata
            "failure_mode": self.failure_mode,
            "fail_step": int(self._episode_fail_step) if self._episode_fail_step is not None else -1,
            "fail_num": int(self._episode_fail_num),
            "failure_happened": bool(self.failure_happened),
            "dead_edges": sorted(list(self.dead_edges)),
            "dead_nodes": sorted(list(self.dead_nodes)),
            "is_connected_src_dst": bool(self.is_connected_src_dst),
        }
        info.update(extra)
        return info

    def _sample_src_dst(self, graph):
        nodes = list(graph.nodes())
        selected = self.rng.choice(nodes, size=2, replace=False)
        self.src, self.dst = int(selected[0]), int(selected[1])

    def _build_observation(self):
        """
        构建观测向量: [current_node, dst_node, neighbor_info × max_degree]
        每个邻居信息: [node_id, delay, bandwidth]
        无效邻居位置 node_id 填充为 -1
        (neighbors 来自 active_graph)
        """
        neighbors = self._get_neighbor_list(self.current_node, graph=self.active_graph)
        obs_dim = 2 + self.max_degree * 3
        observation = np.zeros(obs_dim, dtype=np.float32)

        observation[0] = float(self.current_node) if self.current_node is not None else 0.0
        observation[1] = float(self.dst) if self.dst is not None else 0.0

        # 邻居信息
        for i, node in enumerate(neighbors[: self.max_degree]):
            try:
                edge_data = self.active_graph[self.current_node][node]
                base = 2 + i * 3
                observation[base] = float(node)
                observation[base + 1] = float(edge_data.get("delay", 0.0))
                observation[base + 2] = float(edge_data.get("bandwidth", 0.0))
            except Exception:
                # 极端情况下边不存在，按无效邻居处理
                base = 2 + i * 3
                observation[base] = -1.0
                observation[base + 1] = 0.0
                observation[base + 2] = 0.0

        # 无效邻居位置填充 -1
        for j in range(len(neighbors), self.max_degree):
            base = 2 + j * 3
            observation[base] = -1.0

        return observation

    def _get_neighbor_list(self, node, graph):
        if node is None:
            return []
        if node not in graph:
            return []
        return sorted(graph.neighbors(node))

    def _get_action_mask(self):
        """
        返回当前节点的动作掩码（基于 active_graph）。
        True 表示该动作有效（对应的邻居存在），False 表示无效。
        长度为 max_degree。
        """
        neighbors = self._get_neighbor_list(self.current_node, graph=self.active_graph)
        num_neighbors = len(neighbors)
        mask = np.zeros(self.max_degree, dtype=bool)
        mask[: min(num_neighbors, self.max_degree)] = True
        return mask

    def _generate_graph(self):
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

    def _sync_graph_attributes(self, graph):
        self.num_nodes = graph.number_of_nodes()
        degrees = [degree for _, degree in graph.degree()]
        self.min_degree = min(degrees) if degrees else 0
        self.max_degree = max(degrees) if degrees else 0
        delays = [data.get("delay") for _, _, data in graph.edges(data=True) if "delay" in data]
        bandwidths = [data.get("bandwidth") for _, _, data in graph.edges(data=True) if "bandwidth" in data]
        self.delay_range = (
            float(min(delays)) if delays else 0.0,
            float(max(delays)) if delays else 0.0,
        )
        self.bandwidth_range = (
            float(min(bandwidths)) if bandwidths else 0.0,
            float(max(bandwidths)) if bandwidths else 0.0,
        )

    # --------------------------- Failure / Resilience ---------------------------

    def _sample_episode_fail_num(self):
        if self.failure_mode == "none":
            return 0
        r = self.fail_num_range
        if isinstance(r, (list, tuple)) and len(r) == 2:
            lo, hi = int(r[0]), int(r[1])
            if hi < lo:
                lo, hi = hi, lo
            return int(self.rng.integers(lo, hi + 1))
        return int(max(0, self.fail_num))

    def _sample_episode_fail_step(self):
        if self.failure_mode == "none":
            return None
        r = self.fail_step_range
        if isinstance(r, (list, tuple)) and len(r) == 2:
            lo, hi = int(r[0]), int(r[1])
            if hi < lo:
                lo, hi = hi, lo
            lo = max(0, lo)
            hi = min(self.max_episode_steps, hi)
            return int(self.rng.integers(lo, hi + 1))
        # 特殊：fail_step=-1 表示随机中途发生（1..max_episode_steps-1）
        if int(self.fail_step) == -1:
            return int(self.rng.integers(1, max(2, self.max_episode_steps)))
        return int(max(0, self.fail_step))

    def _apply_failure(self, now_step):
        """Apply link/node failures to active_graph."""
        if self.failure_mode == "none" or self._episode_fail_num <= 0:
            return

        G = self.active_graph

        if self.failure_mode == "link":
            edges = list(G.edges())
            if len(edges) == 0:
                self.failure_happened = True
                return
            k = min(self._episode_fail_num, len(edges))
            # sample edges
            idx = self.rng.choice(len(edges), size=k, replace=False)
            dead = []
            for i in np.atleast_1d(idx):
                u, v = edges[int(i)]
                dead.append((u, v))
            G.remove_edges_from(dead)
            for u, v in dead:
                a, b = (int(u), int(v))
                if a > b:
                    a, b = b, a
                self.dead_edges.add((a, b))

        elif self.failure_mode == "node":
            nodes = list(G.nodes())
            # protect src/dst (and optionally current node) from removal
            protected = set()
            if self.protect_src_dst and (self.src is not None) and (self.dst is not None):
                protected.update([int(self.src), int(self.dst)])
            if self.protect_current_node and (self.current_node is not None):
                protected.add(int(self.current_node))
            candidates = [int(n) for n in nodes if int(n) not in protected]
            if len(candidates) == 0:
                self.failure_happened = True
                return
            k = min(self._episode_fail_num, len(candidates))
            idx = self.rng.choice(len(candidates), size=k, replace=False)
            dead_nodes = [candidates[int(i)] for i in np.atleast_1d(idx)]
            G.remove_nodes_from(dead_nodes)
            self.dead_nodes.update(dead_nodes)

        else:
            # unknown mode -> no-op
            pass

        self.failure_happened = True

    def _recompute_shortest_baseline(self):
        """
        Recompute shortest path baseline on active_graph from src to dst.
        If no path, mark is_connected_src_dst False.
        """
        self.shortest_path = None
        self.shortest_path_delay = None
        self.is_connected_src_dst = True

        if (self.src is None) or (self.dst is None):
            self.is_connected_src_dst = False
            return

        try:
            sp = nx.shortest_path(self.active_graph, self.src, self.dst, weight="delay")
            self.shortest_path = sp
            self.shortest_path_delay = self._calculate_path_delay(sp, graph=self.active_graph)
            self.is_connected_src_dst = True
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            self.is_connected_src_dst = False
            self.shortest_path = None
            self.shortest_path_delay = None

    def _is_reachable(self, u, v, graph):
        if u is None or v is None:
            return False
        if u not in graph or v not in graph:
            return False
        try:
            return nx.has_path(graph, u, v)
        except Exception:
            return False

    # --------------------------- Reward / Delay ---------------------------

    def _compute_reward(self, action, neighbors):
        """
        奖励设计（基于 active_graph）：
        1) 无效动作：reward = -1.0
        2) 环路：reward = loop_penalty * visit_count
        3) 中间步骤：reward = -0.01
        4) 到达终点：reward = 1.0 + 9.0 * (shortest_delay / total_delay)，并将 ratio clamp 到 <=1.0
        5) 若 shortest_path_delay 不存在（极端）：当作 ratio=1.0
        """
        if 0 <= action < len(neighbors):
            next_node = neighbors[action]

            # edge exists by construction (neighbors from active_graph), but be robust
            try:
                edge_data = self.active_graph[self.current_node][next_node]
                step_delay = float(edge_data.get("delay", 0.0))
            except Exception:
                return -1.0, False, self.current_node, 0.0

            terminated = (next_node == self.dst)

            if terminated:
                total_delay = float(self.path_delay + step_delay)
                if total_delay <= 1e-9:
                    total_delay = 1e-9
                if self.shortest_path_delay is None or self.shortest_path_delay <= 1e-9:
                    quality_ratio = 1.0
                else:
                    quality_ratio = float(self.shortest_path_delay) / total_delay
                    # clamp: avoid >1.0 due to mid-episode failure changing baseline
                    quality_ratio = min(1.0, max(0.0, quality_ratio))
                reward = 1.0 + 9.0 * quality_ratio
            else:
                visit_count = self.path.count(next_node)
                if visit_count > 0:
                    reward = float(self.loop_penalty) * float(visit_count)
                else:
                    reward = -0.01

            return float(reward), bool(terminated), int(next_node), float(step_delay)
        else:
            return -1.0, False, self.current_node, 0.0

    def _calculate_path_delay(self, path, graph):
        if path is None or len(path) < 2:
            return 0.0
        total_delay = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            try:
                edge_data = graph[u][v]
                total_delay += float(edge_data.get("delay", 0.0))
            except Exception:
                # path invalid under this graph
                return float("inf")
        return float(total_delay)

    @staticmethod
    def _normalize(value, min_value, max_value):
        if max_value <= min_value:
            return 0.0
        return (value - min_value) / (max_value - min_value)


if __name__ == "__main__":
    class _Config:
        env_id = "NetTupu-Debug"
        delay_range = (1.0, 10.0)
        bandwidth_range = (10.0, 100.0)
        seed = 0
        graph_path = "topology.pkl"

        # Resilience config demo:
        failure_mode = "link"   # "link" or "node" or "none"
        fail_num = 2
        fail_step = 5           # 0 at reset, or >0 mid-episode
        disconnect_penalty = -10.0
        protect_src_dst = True
        protect_current_node = True

    env = NetTupu(env_config=_Config())
    obs, info = env.reset()
    env.visualize(save_path="topology_active.png", show=False, edge_curvature=0.25)
    env.print_topology()
