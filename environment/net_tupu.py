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
    def __init__(self, env_config):
        super(NetTupu, self).__init__()
        self.env_id = env_config.env_id  # The environment id.
        # 默认路径相对于项目根目录
        default_graph_path = os.path.join(os.path.dirname(__file__), "topology.pkl")
        self.graph_path = getattr(env_config, "graph_path", default_graph_path)
        seed = getattr(env_config, "seed", None)
        self.rng = np.random.default_rng(seed)
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
        self.src = None
        self.dst = None
        self.current_node = None
        self.path = []
        obs_low = np.array([-1.0, 0.0, 0.0] * self.max_degree, dtype=np.float32)
        obs_high = np.array(
            [float(self.num_nodes - 1), float(self.delay_range[1]), float(self.bandwidth_range[1])]
            * self.max_degree,
            dtype=np.float32,
        )
        self.observation_space = Box(obs_low, obs_high, shape=[self.max_degree * 3])  # Define observation space.
        self.action_space = Discrete(n=self.max_degree)  # Choose neighbor index.
        self.max_episode_steps = 64  # The max episode length.
        self._current_step = 0  # The count of steps of current episode.
        

    def reset(self, **kwargs):  # Reset your environment.
        self._current_step = 0
        self.path = []  # 清空路径记录
        if kwargs.get("regenerate_graph", False):
            self.regenerate_topology()
        self._sample_src_dst()
        self.current_node = self.src
        self.path.append(self.current_node)  # 记录起始节点
        self.shortest_path = nx.shortest_path(self.graph, self.src, self.dst)
        observation = self._build_observation()
        info = {
            "src": self.src, 
            "dst": self.dst, 
            "current_node": self.current_node,
            "path": self.path.copy(),
            "shortest_path": self.shortest_path,
            "action_mask": self._get_action_mask()  # 动作掩码
        }
        return observation, info

    def step(self, action):  # Run a step with an action.
        self._current_step += 1
        neighbors = self._get_neighbor_list(self.current_node)
        
        # 记录实际选择的节点（-1 表示无效动作）
        if 0 <= action < len(neighbors):
            chosen_node = neighbors[action]
        else:
            chosen_node = -1
        
        reward, terminated, next_node = self._compute_reward(action, neighbors)
        self.current_node = next_node
        self.path.append(self.current_node)  # 记录每步决策后的节点
        observation = self._build_observation()
        truncated = self._current_step >= self.max_episode_steps
        info = {
            "src": self.src, 
            "dst": self.dst, 
            "current_node": self.current_node,
            "path": self.path.copy(),
            "action_idx": int(action),  # 邻居索引
            "chosen_node": chosen_node,  # 实际选择的节点
            "shortest_path": self.shortest_path,
            "action_mask": self._get_action_mask()  # 下一步的动作掩码
        }
        return observation, reward, terminated, truncated, info

    def render(self, *args, **kwargs):  # Render your environment and return an image if the render_mode is "rgb_array".
        return {
            "src": self.src, 
            "dst": self.dst, 
            "current_node": self.current_node,
            "path": self.path.copy(),
            "shortest_path": self.shortest_path,
            "action_mask": self._get_action_mask()
        }

    def visualize(self, **kwargs):
        return visualize_topology(self.graph, **kwargs)

    def print_topology(self):
        for node in sorted(self.graph.nodes()):
            neighbors = sorted(self.graph.neighbors(node))
            degree = len(neighbors)
            neighbor_str = ", ".join(str(n) for n in neighbors)
            print(f"node {node}: degree={degree}, neighbors=[{neighbor_str}]")

    def regenerate_topology(self):
        self.graph = self._generate_graph()

    def close(self):  # Close your environment.
        return


    def _sample_src_dst(self):
        nodes = list(self.graph.nodes())
        selected = self.rng.choice(nodes, size=2, replace=False)
        self.src, self.dst = int(selected[0]), int(selected[1])  # 转为普通 int

    def _build_observation(self):
        neighbors = self._get_neighbor_list(self.current_node)
        observation = np.zeros(self.max_degree * 3, dtype=np.float32)
        for i, node in enumerate(neighbors[: self.max_degree]):
            edge_data = self.graph[self.current_node][node]
            base = i * 3
            observation[base] = float(node)
            observation[base + 1] = float(edge_data["delay"])
            observation[base + 2] = float(edge_data["bandwidth"])
        for j in range(len(neighbors), self.max_degree):
            base = j * 3
            observation[base] = -1.0
        return observation

    def _get_neighbor_list(self, node):
        return sorted(self.graph.neighbors(node))

    def _get_action_mask(self):
        """
        返回当前节点的动作掩码。
        True 表示该动作有效（对应的邻居存在），False 表示无效。
        长度为 max_degree。
        """
        neighbors = self._get_neighbor_list(self.current_node)
        num_neighbors = len(neighbors)
        mask = np.zeros(self.max_degree, dtype=bool)
        mask[:num_neighbors] = True  # 前 num_neighbors 个动作有效
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

    def _compute_reward(self, action, neighbors):
        """
        计算奖励并更新当前节点。
        返回: (reward, terminated, next_node)
        - 有效动作: 奖励 = -时延归一化 + 带宽归一化，到达目标额外 +1
        - 无效动作: 奖励 = -1，保持原地不动，不终止
        """
        if 0 <= action < len(neighbors):
            next_node = neighbors[action]
            edge_data = self.graph[self.current_node][next_node]
            delay_norm = self._normalize(edge_data["delay"], *self.delay_range)
            bandwidth_norm = self._normalize(edge_data["bandwidth"], *self.bandwidth_range)
            reward = -delay_norm + bandwidth_norm  # 低时延高带宽更好
            terminated = (next_node == self.dst)
            if terminated:
                reward += 1.0  # 到达目标额外奖励
            return reward, terminated, next_node
        else:
            # 无效动作：惩罚但不终止，保持原地
            return -1.0, False, self.current_node

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

    env = NetTupu(env_config=_Config())
    env.reset()
    env.visualize(save_path="topology.png", show=False, edge_curvature=0.25)
    env.print_topology()
