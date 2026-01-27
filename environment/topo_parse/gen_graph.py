#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
随机拓扑图生成器

功能:
    - 生成节点数量在 7-20 之间的随机连通图
    - 节点的度在 1-10 之间
    - 节点包含 idx (整数索引) 和 node_id (字符串标识)
    - 边包含链路属性 (时延、带宽、利用率、状态等)

使用示例:
    from gen_graph import generate_random_topology, batch_generate
    G = generate_random_topology(num_nodes=10, seed=42)
    batch_generate(num_graphs=10)
"""

import random
import uuid
from pathlib import Path
from typing import Optional, List

import networkx as nx

from topo_parser import save_to_graphml


# ============================================================================
# 配置常量
# ============================================================================

NODE_TYPES = {3: "II类车载", 4: "II类接入", 5: "II类骨干"}
LATENCY_RANGE = (0.05, 10.0)  # 时延范围 (ms)
BANDWIDTH_OPTIONS = [60, 100, 1000]  # 带宽选项 (Mbps)


# ============================================================================
# 核心生成函数
# ============================================================================

def generate_random_topology(
    num_nodes: Optional[int] = None,
    min_nodes: int = 7,
    max_nodes: int = 20,
    min_degree: int = 1,
    max_degree: int = 10,
    seed: Optional[int] = None,
) -> nx.Graph:
    """
    生成随机网络拓扑图。

    参数:
        num_nodes: 节点数量 (None 则随机)
        min_nodes/max_nodes: 节点数范围
        min_degree/max_degree: 度数范围
        seed: 随机种子

    返回:
        nx.Graph: 带有节点/边属性的图
    """
    if seed is not None:
        random.seed(seed)

    if num_nodes is None:
        num_nodes = random.randint(min_nodes, max_nodes)
    num_nodes = max(min_nodes, min(max_nodes, num_nodes))

    G = _generate_connected_graph(num_nodes, min_degree, max_degree)
    _add_node_attributes(G)
    _add_edge_attributes(G)

    return G


def _generate_connected_graph(num_nodes: int, min_degree: int, max_degree: int) -> nx.Graph:
    """生成连通的随机图结构。"""
    for _ in range(100):
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))

        # 生成树确保连通
        if num_nodes > 1:
            nodes = list(range(num_nodes))
            random.shuffle(nodes)
            for i in range(1, num_nodes):
                G.add_edge(nodes[i], random.choice(nodes[:i]))

        # 随机添加边
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if not G.has_edge(i, j) and G.degree(i) < max_degree and G.degree(j) < max_degree:
                    if random.random() < 0.3:
                        G.add_edge(i, j)

        # 验证约束
        if all(min_degree <= G.degree(n) <= max_degree for n in G.nodes()) and nx.is_connected(G):
            return G

    # 备用: BA 模型
    m = max(1, min(min_degree, num_nodes - 1))
    return nx.barabasi_albert_graph(num_nodes, m)


def _add_node_attributes(G: nx.Graph) -> None:
    """添加节点属性: idx, node_id, node_type, node_status, port_count, port_ids 等。"""
    num_nodes = G.number_of_nodes()

    # 节点类型分布
    types = [5] + random.choices([3, 4, 5], weights=[0.3, 0.5, 0.2], k=num_nodes - 1)
    random.shuffle(types)

    subnet = random.randint(1, 254)
    used_hosts = set()

    for i, node in enumerate(sorted(G.nodes())):
        node_type = types[i]

        # 唯一 IP
        while True:
            host = random.randint(1, 254)
            if host not in used_hosts:
                used_hosts.add(host)
                break

        # 生成字符串 node_id
        node_id = f"0001{node_type:02x}{uuid.uuid4().hex[:16]}00004400000000"

        # 生成经纬度
        longitude = round(random.uniform(112.9, 113.1), 6)
        latitude = round(random.uniform(28.1, 28.3), 6)
        node_location = f"{longitude},{latitude}"

        # 生成端口信息 (根据节点度数生成足够的端口)
        degree = G.degree(node)
        port_count = max(degree, random.randint(4, 8))  # 至少4个端口，或与度数匹配
        port_ids = [f"{node_id}:{p+1}" for p in range(port_count)]

        G.nodes[node].update({
            "idx": i,  # 整数索引 (用于观测空间)
            "node_id": node_id,  # 字符串标识
            "node_type": node_type,
            "node_type_name": NODE_TYPES.get(node_type, f"类型{node_type}"),
            "node_manage_ip_addr": f"192.168.{subnet}.{host}",
            "node_location": node_location,
            "longitude": longitude,
            "latitude": latitude,
            "node_status": 1,
            "port_count": port_count,
            "port_ids": port_ids,
        })


def _add_edge_attributes(G: nx.Graph) -> None:
    """添加边属性: link_id, link_latency, link_bandwidth, link_status, src_port, dst_port 等。"""
    # 记录每个节点已使用的端口索引
    node_port_idx = {n: 0 for n in G.nodes()}

    for u, v in G.edges():
        u_type = G.nodes[u].get("node_type", 4)
        v_type = G.nodes[v].get("node_type", 4)
        u_node_id = G.nodes[u].get("node_id", str(u))
        v_node_id = G.nodes[v].get("node_id", str(v))

        # 根据节点类型确定带宽
        if u_type == 5 or v_type == 5:
            bandwidth = 1000 if u_type in (4, 5) and v_type in (4, 5) else random.choice(BANDWIDTH_OPTIONS)
        elif u_type == 3 or v_type == 3:
            bandwidth = 60
        else:
            bandwidth = random.choice(BANDWIDTH_OPTIONS)

        # 分配端口 (使用端口索引递增)
        src_port_idx = node_port_idx[u]
        dst_port_idx = node_port_idx[v]
        node_port_idx[u] += 1
        node_port_idx[v] += 1

        src_port = f"{u_node_id}:{src_port_idx + 1}"
        dst_port = f"{v_node_id}:{dst_port_idx + 1}"

        # 生成链路 ID: u_node_id:src_port_v_node_id:dst_port
        link_id = f"{src_port}_{dst_port}"

        G[u][v].update({
            "link_id": link_id,
            "link_status": 1,
            "link_type": 2 if bandwidth == 60 else 1,  # 1=有线, 2=无线
            "link_bandwidth": bandwidth,
            "link_latency": round(random.uniform(*LATENCY_RANGE), 3),
            "link_utilization": 0.0,
            "link_loss_rate": 0.0,
            "src_port": src_port,
            "dst_port": dst_port,
        })


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    print("=== 生成单个随机图 ===")
    G = generate_random_topology(num_nodes=18, seed=42)
    print(f"节点: {G.number_of_nodes()}, 边: {G.number_of_edges()}")
    for n in list(G.nodes())[:3]:
        print(f"  节点 {n}: idx={G.nodes[n].get('idx')}, node_id={G.nodes[n].get('node_id')[:20]}...")

    output_dir = Path(__file__).parent.parent / "graph_data" / "random"
    save_to_graphml(G, "random_example.graphml", output_dir)
