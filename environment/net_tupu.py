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
        self.path_delay = 0.0  # 累积路径时延
        self.shortest_path_delay = 0.0  # 最短路径时延
        
        # 奖励函数配置
        # 可选: "v1" (原始), "v2", "v3", "v4" (推荐，带环路和超时惩罚)
        self.reward_version = getattr(env_config, "reward_version", "v4")
        self.loop_penalty = getattr(env_config, "loop_penalty", -0.5)  # 环路惩罚
        self.timeout_penalty = getattr(env_config, "timeout_penalty", -5.0)  # 超时惩罚
        # 观测空间: [current_node, dst_node, neighbor_info × max_degree]
        # 每个邻居: [node_id, delay, bandwidth]
        # 总维度: 2 + max_degree * 3
        obs_dim = 2 + self.max_degree * 3
        obs_low = np.concatenate([
            np.array([0.0, 0.0], dtype=np.float32),  # current_node, dst_node
            np.array([-1.0, 0.0, 0.0] * self.max_degree, dtype=np.float32)  # neighbors
        ])
        obs_high = np.concatenate([
            np.array([float(self.num_nodes - 1), float(self.num_nodes - 1)], dtype=np.float32),  # current_node, dst_node
            np.array(
                [float(self.num_nodes - 1), float(self.delay_range[1]), float(self.bandwidth_range[1])]
                * self.max_degree,
                dtype=np.float32,
            )  # neighbors
        ])
        self.observation_space = Box(obs_low, obs_high, shape=(obs_dim,))  # Define observation space.
        self.action_space = Discrete(n=self.max_degree)  # Choose neighbor index.
        self.max_episode_steps = 64  # The max episode length.
        self._current_step = 0  # The count of steps of current episode.
        

    def reset(self, **kwargs):  # Reset your environment.
        self._current_step = 0
        self.path = []  # 清空路径记录
        self.path_delay = 0.0  # 重置路径时延
        if kwargs.get("regenerate_graph", False):
            self.regenerate_topology()
        self._sample_src_dst()
        self.current_node = self.src
        self.path.append(self.current_node)  # 记录起始节点
        self.shortest_path = nx.shortest_path(self.graph, self.src, self.dst)
        self.shortest_path_delay = self._calculate_path_delay(self.shortest_path)
        observation = self._build_observation()
        info = {
            "src": self.src, 
            "dst": self.dst, 
            "current_node": self.current_node,
            "path": self.path.copy(),
            "shortest_path": self.shortest_path,
            "shortest_path_delay": self.shortest_path_delay,
            "path_delay": self.path_delay,
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
        
        # 根据配置选择奖励函数
        reward, terminated, next_node, step_delay = self._get_reward(action, neighbors)
        
        self.current_node = next_node
        self.path.append(self.current_node)  # 记录每步决策后的节点
        self.path_delay += step_delay  # 累积路径时延
        observation = self._build_observation()
        truncated = self._current_step >= self.max_episode_steps
        
        # 超时惩罚：如果超过最大步数还未到达终点
        if truncated and not terminated:
            reward += self.timeout_penalty
        
        # 统计访问次数（不包括刚刚添加的当前节点）
        visit_count = self.path[:-1].count(chosen_node) if chosen_node != -1 else 0
        
        info = {
            "src": self.src, 
            "dst": self.dst, 
            "current_node": self.current_node,
            "path": self.path.copy(),
            "action_idx": int(action),  # 邻居索引
            "chosen_node": chosen_node,  # 实际选择的节点
            "shortest_path": self.shortest_path,
            "shortest_path_delay": self.shortest_path_delay,
            "path_delay": self.path_delay,
            "step_delay": step_delay,
            "is_loop": visit_count > 0,  # 是否形成环路
            "visit_count": visit_count,  # 该节点之前被访问的次数
            "action_mask": self._get_action_mask()  # 下一步的动作掩码
        }
        return observation, reward, terminated, truncated, info
    
    def _get_reward(self, action, neighbors):
        """根据配置选择奖励函数"""
        if self.reward_version == "v1":
            return self._compute_reward(action, neighbors)
        elif self.reward_version == "v2":
            return self._compute_reward_v2(action, neighbors)
        elif self.reward_version == "v3":
            return self._compute_reward_v3(action, neighbors)
        elif self.reward_version == "v4":
            return self._compute_reward_v4(action, neighbors)
        else:
            return self._compute_reward(action, neighbors)

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
        """
        构建观测向量: [current_node, dst_node, neighbor_info × max_degree]
        每个邻居信息: [node_id, delay, bandwidth]
        无效邻居位置 node_id 填充为 -1
        """
        neighbors = self._get_neighbor_list(self.current_node)
        obs_dim = 2 + self.max_degree * 3
        observation = np.zeros(obs_dim, dtype=np.float32)
        
        # 前两维: 当前节点和目的地节点
        observation[0] = float(self.current_node)
        observation[1] = float(self.dst)
        
        # 后面是邻居信息
        for i, node in enumerate(neighbors[: self.max_degree]):
            edge_data = self.graph[self.current_node][node]
            base = 2 + i * 3  # 偏移2个位置
            observation[base] = float(node)
            observation[base + 1] = float(edge_data["delay"])
            observation[base + 2] = float(edge_data["bandwidth"])
        
        # 无效邻居位置填充 -1
        for j in range(len(neighbors), self.max_degree):
            base = 2 + j * 3
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
        原始奖励函数：
        计算奖励并更新当前节点。
        返回: (reward, terminated, next_node, step_delay)
        - 有效动作: 奖励 = -时延归一化 + 带宽归一化，到达目标额外 +1
        - 无效动作: 奖励 = -1，保持原地不动，不终止
        """
        if 0 <= action < len(neighbors):
            next_node = neighbors[action]
            edge_data = self.graph[self.current_node][next_node]
            step_delay = edge_data["delay"]
            delay_norm = self._normalize(edge_data["delay"], *self.delay_range)
            bandwidth_norm = self._normalize(edge_data["bandwidth"], *self.bandwidth_range)
            reward = -delay_norm + 0 * bandwidth_norm  # 低时延高带宽更好
            terminated = (next_node == self.dst)
            if terminated:
                reward += 1.0  # 到达目标额外奖励
            return reward, terminated, next_node, step_delay
        else:
            # 无效动作：惩罚但不终止，保持原地
            return -1.0, False, self.current_node, 0.0

    def _compute_reward_v2(self, action, neighbors):
        """
        改进的奖励函数 V2：
        
        1. 无效动作：reward = -1.0，保持原地不动
        2. 有效动作（未到达终点）：reward = 0.0
        3. 到达终点：reward = 10.0 * (最短时延 / 实际时延)
        
        返回: (reward, terminated, next_node, step_delay)
        """
        if 0 <= action < len(neighbors):
            next_node = neighbors[action]
            edge_data = self.graph[self.current_node][next_node]
            step_delay = edge_data["delay"]
            
            # 检查是否到达终点
            terminated = (next_node == self.dst)
            
            if terminated:
                # 到达终点：计算路径时延比值作为奖励
                total_delay = self.path_delay + step_delay
                if total_delay > 0:
                    # 性能奖励：最短时延/实际时延，值域 (0, 1]，最优为1
                    delay_ratio = self.shortest_path_delay / total_delay
                    reward = 10.0 * delay_ratio  # 放大奖励信号
                else:
                    reward = 10.0  # 边界情况
            else:
                # 中间步骤：reward = 0.0
                reward = 0.0
                
            return reward, terminated, next_node, step_delay
        else:
            # 无效动作：惩罚，保持原地，时延为0
            return -1.0, False, self.current_node, 0.0

    def _compute_reward_v3(self, action, neighbors):
        """
        改进的奖励函数 V3：
        
        解决源目的变化导致的奖励不一致问题：
        1. 使用相对指标（比值）而非绝对值
        2. 中间步骤提供引导信号
        3. 终点奖励反映路径质量
        
        奖励设计：
        1. 无效动作：reward = -1.0
        2. 有效动作（中间步骤）：reward = 小的负值（引导快速到达）
        3. 到达终点：reward = 基础奖励 + 质量奖励
           - 基础奖励：1.0（鼓励完成任务）
           - 质量奖励：根据 (最短时延/实际时延) 计算
        
        返回: (reward, terminated, next_node, step_delay)
        """
        if 0 <= action < len(neighbors):
            next_node = neighbors[action]
            edge_data = self.graph[self.current_node][next_node]
            step_delay = edge_data["delay"]
            
            # 检查是否到达终点
            terminated = (next_node == self.dst)
            
            if terminated:
                # 到达终点：基础奖励 + 质量奖励
                total_delay = self.path_delay + step_delay
                if total_delay > 0 and self.shortest_path_delay > 0:
                    # 质量比值：值域 (0, 1]，最优为1
                    quality_ratio = self.shortest_path_delay / total_delay
                    # 基础奖励 + 质量奖励（按比例）
                    # 最优路径: 1.0 + 9.0 * 1.0 = 10.0
                    # 2倍时延: 1.0 + 9.0 * 0.5 = 5.5
                    # 3倍时延: 1.0 + 9.0 * 0.33 = 4.0
                    reward = 1.0 + 9.0 * quality_ratio
                else:
                    reward = 10.0
            else:
                # 中间步骤：小的负奖励，引导快速到达终点
                # 使用归一化的步进惩罚，与源目的无关
                reward = -0.01  # 每步小惩罚，鼓励找短路径
                
            return reward, terminated, next_node, step_delay
        else:
            # 无效动作：较大惩罚
            return -1.0, False, self.current_node, 0.0

    def _compute_reward_v4(self, action, neighbors):
        """
        改进的奖励函数 V4（推荐）：
        
        在 V3 基础上增加：
        1. 环路惩罚：访问已访问过的节点，惩罚与访问次数成正比
        2. 超时惩罚：超过最大跳数未到达终点
        
        奖励设计：
        1. 无效动作：reward = -1.0
        2. 访问已访问节点（环路）：reward = loop_penalty * 访问次数
           - 第1次重复访问: -0.5 × 1 = -0.5
           - 第2次重复访问: -0.5 × 2 = -1.0
           - 第3次重复访问: -0.5 × 3 = -1.5
        3. 有效动作（中间步骤）：reward = -0.01（引导快速到达）
        4. 到达终点：reward = 1.0 + 9.0 * (最短时延/实际时延)
        5. 超时未到达：在 step 中额外惩罚 -5.0
        
        返回: (reward, terminated, next_node, step_delay)
        """
        if 0 <= action < len(neighbors):
            next_node = neighbors[action]
            edge_data = self.graph[self.current_node][next_node]
            step_delay = edge_data["delay"]
            
            # 检查是否到达终点
            terminated = (next_node == self.dst)
            
            if terminated:
                # 到达终点：基础奖励 + 质量奖励
                total_delay = self.path_delay + step_delay

                quality_ratio = self.shortest_path_delay / total_delay
                reward = 1.0 + 9.0 * quality_ratio

            else:
                # 统计该节点在路径中已被访问的次数
                visit_count = self.path.count(next_node)
                
                if visit_count > 0:
                    # 环路惩罚：惩罚与访问次数成正比
                    # 访问次数越多，惩罚越大
                    reward = self.loop_penalty * visit_count
                else:
                    # 正常中间步骤：小的负奖励
                    reward = -0.01
                
            return reward, terminated, next_node, step_delay
        else:
            # 无效动作：较大惩罚
            return -1.0, False, self.current_node, 0.0
    
    def _calculate_path_delay(self, path):
        """
        计算给定路径的总时延
        """
        if len(path) < 2:
            return 0.0
        total_delay = 0.0
        for i in range(len(path) - 1):
            edge_data = self.graph[path[i]][path[i + 1]]
            total_delay += edge_data["delay"]
        return total_delay

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
