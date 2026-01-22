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
    Net routing environment with resilience (anti-damage) mechanism.

    Resilience = Failure Injection + Dynamic valid action set + Masked action selection (agent side) +
                 Reachability-aware termination/reward (env side).

    Default (if not provided in env_config):
      enable_failure=True
      failure_mode="edge"
      fail_num=2
      fail_step=-1 (inject at reset)
      ensure_reachable=True
    """

    def __init__(self, env_config):
        super(NetTupu, self).__init__()
        self.env_id = env_config.env_id

        # ---------------------------------------------------------------------
        # Load / generate topology
        # ---------------------------------------------------------------------
        default_graph_path = os.path.join(os.path.dirname(__file__), "topology.pkl")
        self.graph_path = getattr(env_config, "graph_path", default_graph_path)

        self.rng = np.random.default_rng()

        if self.graph_path is not None:
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
                env_config,
                "target_degrees",
                [2, 2, 2, 2, 3, 6, 6, 5, 5, 2, 4, 5, 7, 5, 3, 5, 5, 3],
            )
            graph = self._generate_graph()

        # base_graph 永不修改；active_graph 每回合可被损毁
        self.base_graph = graph
        self.active_graph = self.base_graph.copy()

        # ---------------------------------------------------------------------
        # Episode state
        # ---------------------------------------------------------------------
        self.src = None
        self.dst = None
        self.current_node = None
        self.path = []
        self.path_delay = 0.0

        # shortest path (under active_graph)
        self.shortest_path = None
        self.shortest_path_delay = np.inf

        # dist_to_dst: shortest delay from each node -> dst (under active_graph)
        self.dist_to_dst = {}

        # ---------------------------------------------------------------------
        # Reward / termination config
        # ---------------------------------------------------------------------
        self.loop_penalty = float(getattr(env_config, "loop_penalty", -0.5))
        self.timeout_penalty = float(getattr(env_config, "timeout_penalty", -5.0))
        self.invalid_action_penalty = float(getattr(env_config, "invalid_action_penalty", -1.0))
        self.disconnect_penalty = float(getattr(env_config, "disconnect_penalty", -5.0))
        self.step_penalty = float(getattr(env_config, "step_penalty", -0.01))
        self.progress_scale = float(getattr(env_config, "progress_scale", 0.02))  # 进展 shaping（默认很小）
        # 终点奖励：1 + 9*(shortest/actual)（与原逻辑兼容）
        self.success_base = float(getattr(env_config, "success_base", 1.0))
        self.success_scale = float(getattr(env_config, "success_scale", 9.0))

        # ---------------------------------------------------------------------
        # Failure injection (resilience) config: 默认开启 + reset 时边损毁2条
        # ---------------------------------------------------------------------
        self.enable_failure = bool(getattr(env_config, "enable_failure", True))
        self.failure_mode = getattr(env_config, "failure_mode", "edge")  # "edge" / "node"
        self.fail_num = int(getattr(env_config, "fail_num", 2))
        self.fail_step = int(getattr(env_config, "fail_step", -1))  # -1 reset inject; >=0 step inject
        self.ensure_reachable = bool(getattr(env_config, "ensure_reachable", True))
        self.max_failure_tries = int(getattr(env_config, "max_failure_tries", 30))

        self.failure_happened = False
        self.dead_edges = []
        self.dead_nodes = []

        # ---------------------------------------------------------------------
        # Observation / action space (same layout as your original)
        # obs = [current_node, dst_node, (nbr_id, delay, bw)*max_degree]
        # invalid neighbor slot: nbr_id = -1
        # ---------------------------------------------------------------------
        obs_dim = 2 + self.max_degree * 3
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
        self.observation_space = Box(obs_low, obs_high, shape=(obs_dim,))
        self.action_space = Discrete(n=self.max_degree)

        self.max_episode_steps = int(getattr(env_config, "max_episode_steps", 64))
        self._current_step = 0

    # =====================================================================
    # Core API
    # =====================================================================
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
            # regenerate will also reset base_graph/active_graph
            self.regenerate_topology()

        # 采样 src/dst（从 base_graph 节点里选）
        self._sample_src_dst()

        # reset 时就注入损毁（默认）
        if self.enable_failure and self.fail_num > 0 and self.fail_step < 0:
            self._inject_failure_with_retry()

        # 初始化 episode
        self.current_node = self.src
        self.path.append(self.current_node)

        self._recompute_shortest_and_dists()

        observation = self._build_observation()
        info = self._build_info(extra={
            "reset": True,
        })
        return observation, info

    def step(self, action):
        self._current_step += 1

        # 动态损毁：到达 fail_step 时触发一次（如果你将 fail_step>=0）
        if (self.enable_failure and (not self.failure_happened)
                and self.fail_num > 0 and self.fail_step >= 0
                and self._current_step == self.fail_step):
            self._inject_failure_with_retry()
            self._recompute_shortest_and_dists()

        neighbors = self._get_neighbor_list(self.current_node)

        # 若当前节点已无可用邻居：直接判定断连
        if len(neighbors) == 0:
            reward = self.disconnect_penalty
            terminated = True
            truncated = False
            step_delay = 0.0
            chosen_node = -1
            observation = self._build_observation()
            info = self._build_info(extra={
                "action_idx": int(action),
                "chosen_node": chosen_node,
                "step_delay": step_delay,
                "is_loop": False,
                "visit_count": 0,
                "terminated_reason": "no_neighbors",
            })
            return observation, reward, terminated, truncated, info

        # 记录实际选择的节点（-1 表示无效动作）
        if 0 <= int(action) < len(neighbors):
            chosen_node = neighbors[int(action)]
        else:
            chosen_node = -1

        reward, terminated, next_node, step_delay, reason = self._compute_reward_and_transition(int(action), neighbors)

        # 状态推进
        self.current_node = next_node
        self.path.append(self.current_node)
        self.path_delay += step_delay

        observation = self._build_observation()
        truncated = self._current_step >= self.max_episode_steps

        # 超时惩罚：超过最大步数还未到达终点
        if truncated and not terminated:
            reward += self.timeout_penalty

        # 统计 visit_count（不包括刚刚添加的 current_node）
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
        return

    # =====================================================================
    # Topology ops
    # =====================================================================
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

    # =====================================================================
    # Helpers
    # =====================================================================
    def _sample_src_dst(self):
        nodes = list(self.base_graph.nodes())
        selected = self.rng.choice(nodes, size=2, replace=False)
        self.src, self.dst = int(selected[0]), int(selected[1])

    def _get_neighbor_list(self, node):
        # neighbors based on active_graph (critical for resilience)
        if not self.active_graph.has_node(node):
            return []
        return sorted(self.active_graph.neighbors(node))

    def _get_action_mask(self):
        neighbors = self._get_neighbor_list(self.current_node)
        num_neighbors = len(neighbors)
        mask = np.zeros(self.max_degree, dtype=bool)
        mask[:min(num_neighbors, self.max_degree)] = True
        return mask

    def _build_observation(self):
        """
        obs = [current_node, dst_node, (nbr_id, delay, bw)*max_degree]
        invalid slot: nbr_id=-1
        """
        neighbors = self._get_neighbor_list(self.current_node)
        obs_dim = 2 + self.max_degree * 3
        obs = np.zeros(obs_dim, dtype=np.float32)

        obs[0] = float(self.current_node)
        obs[1] = float(self.dst)

        # fill neighbor slots
        for i, node in enumerate(neighbors[:self.max_degree]):
            edge_data = self.active_graph[self.current_node][node]
            base = 2 + i * 3
            obs[base] = float(node)
            obs[base + 1] = float(edge_data.get("delay", 0.0))
            obs[base + 2] = float(edge_data.get("bandwidth", 0.0))

        # invalid neighbor slots
        for j in range(len(neighbors), self.max_degree):
            base = 2 + j * 3
            obs[base] = -1.0

        return obs

    def _recompute_shortest_and_dists(self):
        """
        recompute shortest_path, shortest_path_delay and dist_to_dst under active_graph.
        """
        self.shortest_path = None
        self.shortest_path_delay = np.inf
        self.dist_to_dst = {}

        # If dst missing (node failure), treat as unreachable
        if not self.active_graph.has_node(self.dst):
            return

        # dist_to_dst: shortest delay to dst for each node
        # For undirected graphs, single_source from dst works
        try:
            self.dist_to_dst = nx.single_source_dijkstra_path_length(
                self.active_graph, self.dst, weight="delay"
            )
        except Exception:
            self.dist_to_dst = {}

        # shortest path from src to dst if reachable
        if self.active_graph.has_node(self.src) and (self.src in self.dist_to_dst):
            try:
                self.shortest_path = nx.shortest_path(self.active_graph, self.src, self.dst, weight="delay")
                self.shortest_path_delay = float(self._calculate_path_delay(self.shortest_path, graph=self.active_graph))
            except Exception:
                self.shortest_path = None
                self.shortest_path_delay = float(self.dist_to_dst.get(self.src, np.inf))

    def _is_reachable_from(self, node: int) -> bool:
        return (node in self.dist_to_dst)

    def _compute_reward_and_transition(self, action: int, neighbors):
        """
        Return (reward, terminated, next_node, step_delay, reason)
        """
        # invalid action index
        if not (0 <= action < len(neighbors)):
            return self.invalid_action_penalty, False, self.current_node, 0.0, "invalid_action"

        next_node = neighbors[action]
        edge_data = self.active_graph[self.current_node][next_node]
        step_delay = float(edge_data.get("delay", 0.0))

        # If dst unreachable already from current (after failure), terminate
        if not self._is_reachable_from(self.current_node):
            return self.disconnect_penalty, True, self.current_node, 0.0, "disconnected_current"

        # If choosing next leads to unreachable region, terminate (fast fail to reduce variance)
        # (This is important for resilience: after damage, agent must avoid dead-ends.)
        if not self._is_reachable_from(next_node) and next_node != self.dst:
            return self.disconnect_penalty, True, next_node, step_delay, "disconnected_next"

        # reach dst
        if next_node == self.dst:
            total_delay = float(self.path_delay + step_delay)
            if not np.isfinite(self.shortest_path_delay) or self.shortest_path_delay <= 0.0:
                quality_ratio = 0.0
            else:
                quality_ratio = float(self.shortest_path_delay / max(total_delay, 1e-6))
                # clip to avoid extremely large spikes if total_delay is tiny
                quality_ratio = float(np.clip(quality_ratio, 0.0, 2.0))
            reward = self.success_base + self.success_scale * quality_ratio
            return reward, True, next_node, step_delay, "arrive"

        # loop penalty
        visit_count = self.path.count(next_node)
        if visit_count > 0:
            reward = self.loop_penalty * float(visit_count)
            return reward, False, next_node, step_delay, "loop"

        # progress shaping (small): encourage moving closer to dst under current damaged graph
        d_cur = float(self.dist_to_dst.get(self.current_node, 0.0))
        d_next = float(self.dist_to_dst.get(next_node, d_cur))
        progress = d_cur - d_next  # positive means closer
        reward = self.step_penalty + self.progress_scale * float(progress)

        return float(reward), False, next_node, step_delay, "step"

    def _build_info(self, extra=None):
        if extra is None:
            extra = {}
        info = {
            "src": int(self.src) if self.src is not None else None,
            "dst": int(self.dst) if self.dst is not None else None,
            "current_node": int(self.current_node) if self.current_node is not None else None,
            "path": self.path.copy(),
            "shortest_path": self.shortest_path,
            "shortest_path_delay": float(self.shortest_path_delay) if np.isfinite(self.shortest_path_delay) else None,
            "path_delay": float(self.path_delay),
            "action_mask": self._get_action_mask(),
            # resilience fields
            "failure_happened": bool(self.failure_happened),
            "failure_mode": self.failure_mode,
            "fail_step": int(self.fail_step),
            "fail_num": int(self.fail_num),
            "dead_edges": self.dead_edges.copy(),
            "dead_nodes": self.dead_nodes.copy(),
            "is_connected_src_dst": bool(self.active_graph.has_node(self.src))
                                    and bool(self.active_graph.has_node(self.dst))
                                    and bool(nx.has_path(self.active_graph, self.src, self.dst))
                                    if (self.active_graph.has_node(self.src) and self.active_graph.has_node(self.dst))
                                    else False,
        }
        info.update(extra)
        return info

    # =====================================================================
    # Failure Injection (Resilience)
    # =====================================================================
    def _inject_failure_with_retry(self):
        """
        Inject failures into active_graph.
        If ensure_reachable=True, retry a few times to keep src->dst reachable.
        """
        self.failure_happened = True

        if not self.enable_failure or self.fail_num <= 0:
            return

        # try multiple times to satisfy reachability (if required)
        for _ in range(max(1, self.max_failure_tries)):
            g = self.base_graph.copy()

            dead_edges = []
            dead_nodes = []

            if self.failure_mode == "edge":
                dead_edges = self._remove_random_edges(g, self.fail_num)
            elif self.failure_mode == "node":
                dead_nodes = self._remove_random_nodes(g, self.fail_num, exclude={self.src, self.dst})
            else:
                raise ValueError(f"Unknown failure_mode: {self.failure_mode}")

            # if not enforcing reachability, accept immediately
            if not self.ensure_reachable:
                self.active_graph = g
                self.dead_edges = dead_edges
                self.dead_nodes = dead_nodes
                return

            # enforce reachability
            if g.has_node(self.src) and g.has_node(self.dst) and nx.has_path(g, self.src, self.dst):
                self.active_graph = g
                self.dead_edges = dead_edges
                self.dead_nodes = dead_nodes
                return

        # fallback: if cannot keep reachable, accept the last injected graph
        self.active_graph = g
        self.dead_edges = dead_edges
        self.dead_nodes = dead_nodes

    def _remove_random_edges(self, g: nx.Graph, k: int):
        edges = list(g.edges())
        if len(edges) == 0 or k <= 0:
            return []
        self.rng.shuffle(edges)

        removed = []
        for (u, v) in edges:
            if len(removed) >= k:
                break
            if not g.has_edge(u, v):
                continue
            # optional: avoid deleting self-loop (usually none)
            g.remove_edge(u, v)
            removed.append((int(u), int(v)))
        return removed

    def _remove_random_nodes(self, g: nx.Graph, k: int, exclude=None):
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

    # =====================================================================
    # Graph helpers
    # =====================================================================
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
        self.delay_range = (float(min(delays)) if delays else 0.0, float(max(delays)) if delays else 0.0)
        self.bandwidth_range = (float(min(bandwidths)) if bandwidths else 0.0, float(max(bandwidths)) if bandwidths else 0.0)

    def _calculate_path_delay(self, path, graph: nx.Graph):
        if path is None or len(path) < 2:
            return 0.0
        total = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if graph.has_edge(u, v):
                total += float(graph[u][v].get("delay", 0.0))
            else:
                # edge missing (can happen if path is stale)
                total += 0.0
        return total


if __name__ == "__main__":
    class _Config:
        env_id = "NetTupu-Debug"
        delay_range = (1.0, 10.0)
        bandwidth_range = (10.0, 100.0)
        seed = 0
        graph_path = "topology.pkl"
        # default resilience
        enable_failure = True
        failure_mode = "edge"
        fail_num = 2
        fail_step = -1
        ensure_reachable = True

    env = NetTupu(env_config=_Config())
    obs, info = env.reset()
    print("reset info:", {k: info[k] for k in ["src", "dst", "failure_happened", "dead_edges", "is_connected_src_dst"]})
    env.visualize(save_path="topology_active.png", show=False, edge_curvature=0.25)
    env.print_topology()
