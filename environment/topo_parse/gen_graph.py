#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
随机拓扑图生成器

基于 II 类节点图的属性结构，随机生成新的网络拓扑图。

功能:
    - 生成节点数量在 7-20 之间的随机图
    - 节点的度在 1-10 之间
    - 保持 II 类节点的属性结构（node_type, link_bandwidth 等）
    - 保证生成的图是连通的

使用示例:
    from gen_graph import generate_random_topology, batch_generate
    
    # 生成单个图
    G = generate_random_topology(num_nodes=10, seed=42)
    
    # 批量生成并保存
    batch_generate(num_graphs=10, output_dir="graph_data/random")
"""

import random
import uuid
from pathlib import Path
from typing import Optional, List, Tuple

import networkx as nx

# 导入保存函数
from topo_parser import save_to_graphml


# ============================================================================
# 配置常量
# ============================================================================

# II 类节点类型
II_NODE_TYPES = {
    3: "II类车载",
    4: "II类接入",
    5: "II类骨干",
}

# 默认节点属性模板
DEFAULT_NODE_ATTRS = {
    "node_status": 1,
    "port_count": 4,
}

# 默认边属性模板
DEFAULT_EDGE_ATTRS = {
    "link_status": 1,
    "link_utilization": 0,
    "flow_table_status": 1,
    "link_loss_rate": 0.0,
}

# 链路带宽选项 (Mbps)
BANDWIDTH_OPTIONS = [60, 100, 1000]

# 链路延迟范围 (ms)
LATENCY_RANGE = (0.05, 10.0)

# 经纬度范围（默认在长沙附近）
LONGITUDE_RANGE = (112.90, 113.10)
LATITUDE_RANGE = (28.10, 28.30)

# IP 地址基础
IP_BASE = "192.168.{subnet}.{host}"


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
    ensure_connected: bool = True
) -> nx.Graph:
    """
    生成随机 II 类网络拓扑图
    
    参数:
        num_nodes: 节点数量，如果为 None 则随机生成
        min_nodes: 最小节点数 (默认 7)
        max_nodes: 最大节点数 (默认 20)
        min_degree: 最小节点度 (默认 1)
        max_degree: 最大节点度 (默认 10)
        seed: 随机种子
        ensure_connected: 是否确保图连通 (默认 True)
    
    返回:
        nx.Graph: 生成的拓扑图
    """
    if seed is not None:
        random.seed(seed)
    
    # 确定节点数量
    if num_nodes is None:
        num_nodes = random.randint(min_nodes, max_nodes)
    num_nodes = max(min_nodes, min(max_nodes, num_nodes))
    
    # 生成基础图结构
    G = _generate_graph_structure(num_nodes, min_degree, max_degree, ensure_connected)
    
    # 添加节点属性
    _add_node_attributes(G)
    
    # 添加边属性
    _add_edge_attributes(G)
    
    return G


def _generate_graph_structure(
    num_nodes: int,
    min_degree: int,
    max_degree: int,
    ensure_connected: bool
) -> nx.Graph:
    """
    生成满足度约束的随机图结构
    """
    max_attempts = 100
    
    for attempt in range(max_attempts):
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        
        # 首先生成一棵生成树，确保连通性
        if ensure_connected and num_nodes > 1:
            nodes = list(range(num_nodes))
            random.shuffle(nodes)
            for i in range(1, num_nodes):
                # 连接到之前的某个节点
                target = random.choice(nodes[:i])
                G.add_edge(nodes[i], target)
        
        # 随机添加更多边，但要满足度约束
        all_possible_edges = [
            (i, j) for i in range(num_nodes) 
            for j in range(i + 1, num_nodes) 
            if not G.has_edge(i, j)
        ]
        random.shuffle(all_possible_edges)
        
        for u, v in all_possible_edges:
            deg_u = G.degree(u)
            deg_v = G.degree(v)
            
            # 检查是否可以添加边
            if deg_u < max_degree and deg_v < max_degree:
                # 随机决定是否添加（控制边的密度）
                if random.random() < 0.3:  # 30% 概率添加边
                    G.add_edge(u, v)
        
        # 验证度约束
        degrees = [G.degree(n) for n in G.nodes()]
        if all(min_degree <= d <= max_degree for d in degrees):
            # 验证连通性
            if not ensure_connected or nx.is_connected(G):
                return G
    
    # 如果多次尝试失败，使用备用策略
    print(f"警告: 使用备用策略生成图 (节点数={num_nodes})")
    return _fallback_graph_generation(num_nodes, min_degree, max_degree)


def _fallback_graph_generation(
    num_nodes: int,
    min_degree: int,
    max_degree: int
) -> nx.Graph:
    """
    备用图生成策略：使用 Barabási-Albert 模型并调整
    """
    # 使用 BA 模型生成
    m = max(1, min(min_degree, num_nodes - 1))
    G = nx.barabasi_albert_graph(num_nodes, m)
    
    # 移除超过 max_degree 的边
    for node in list(G.nodes()):
        while G.degree(node) > max_degree:
            neighbors = list(G.neighbors(node))
            # 移除一条边，优先移除度高的邻居
            neighbor_to_remove = max(neighbors, key=lambda n: G.degree(n))
            if G.degree(neighbor_to_remove) > min_degree:
                G.remove_edge(node, neighbor_to_remove)
            else:
                break
    
    return G


def _add_node_attributes(G: nx.Graph) -> None:
    """
    为图中的节点添加 II 类节点属性
    """
    num_nodes = G.number_of_nodes()
    
    # 决定节点类型分布：至少 1 个骨干节点
    node_types = []
    node_types.append(5)  # 至少 1 个骨干节点
    
    # 随机分配其余节点类型
    type_choices = [3, 4, 5]  # 车载、接入、骨干
    type_weights = [0.3, 0.5, 0.2]  # 权重
    
    for _ in range(num_nodes - 1):
        node_type = random.choices(type_choices, weights=type_weights, k=1)[0]
        node_types.append(node_type)
    
    random.shuffle(node_types)
    
    # 生成 IP 地址
    subnet = random.randint(1, 254)
    used_hosts = set()
    
    for i, node in enumerate(G.nodes()):
        node_type = node_types[i]
        
        # 生成唯一 IP
        while True:
            host = random.randint(1, 254)
            if host not in used_hosts:
                used_hosts.add(host)
                break
        
        ip_addr = IP_BASE.format(subnet=subnet, host=host)
        
        # 生成经纬度
        longitude = random.uniform(*LONGITUDE_RANGE)
        latitude = random.uniform(*LATITUDE_RANGE)
        
        # 生成节点 ID
        node_id = _generate_node_id(node_type)
        
        # 生成端口 ID
        port_count = DEFAULT_NODE_ATTRS["port_count"]
        port_ids = [f"{node_id}:{p}" for p in range(1, port_count + 1)]
        
        # 设置节点属性
        G.nodes[node].update({
            "node_id": node_id,
            "node_type": node_type,
            "node_type_name": II_NODE_TYPES[node_type],
            "node_manage_ip_addr": ip_addr,
            "node_location": f"{longitude:.6f},{latitude:.6f}",
            "longitude": longitude,
            "latitude": latitude,
            "node_status": DEFAULT_NODE_ATTRS["node_status"],
            "port_count": port_count,
            "port_ids": ",".join(port_ids),
        })


def _add_edge_attributes(G: nx.Graph) -> None:
    """
    为图中的边添加链路属性
    """
    for u, v in G.edges():
        u_id = G.nodes[u].get("node_id", str(u))
        v_id = G.nodes[v].get("node_id", str(v))
        u_type = G.nodes[u].get("node_type", 4)
        v_type = G.nodes[v].get("node_type", 4)
        
        # 根据节点类型决定带宽
        # 骨干节点之间或骨干与接入之间使用高带宽
        if u_type == 5 or v_type == 5:
            if u_type in (4, 5) and v_type in (4, 5):
                bandwidth = 1000  # 骨干/接入之间 1000Mbps
            else:
                bandwidth = random.choice([60, 100, 1000])
        elif u_type == 3 or v_type == 3:
            bandwidth = 60  # 车载节点使用低带宽
        else:
            bandwidth = random.choice(BANDWIDTH_OPTIONS)
        
        # 随机延迟
        latency = round(random.uniform(*LATENCY_RANGE), 3)
        
        # 生成端口号
        src_port = f"{u_id}:{random.randint(1, 4)}"
        dst_port = f"{v_id}:{random.randint(1, 4)}"
        
        # 生成链路 ID
        link_id = f"{src_port}_{dst_port}"
        
        # 设置边属性
        G[u][v].update({
            "link_id": link_id,
            "link_status": DEFAULT_EDGE_ATTRS["link_status"],
            "link_bandwidth": bandwidth,
            "link_latency": latency,
            "src_port": src_port,
            "dst_port": dst_port,
            "link_utilization": DEFAULT_EDGE_ATTRS["link_utilization"],
            "bandwidth_capacity_available": bandwidth,
            "link_loss_rate": DEFAULT_EDGE_ATTRS["link_loss_rate"],
            "flow_table_status": DEFAULT_EDGE_ATTRS["flow_table_status"],
        })


def _generate_node_id(node_type: int) -> str:
    """
    生成节点 ID (模拟真实格式)
    格式: 0001{type:02x}{random_hex}00004400000000
    """
    type_hex = f"{node_type:02x}"
    random_hex = uuid.uuid4().hex[:16]
    return f"0001{type_hex}{random_hex}00004400000000"


# ============================================================================
# 批量生成函数
# ============================================================================

def batch_generate(
    num_graphs: int = 10,
    output_dir: Optional[str] = None,
    min_nodes: int = 7,
    max_nodes: int = 20,
    min_degree: int = 1,
    max_degree: int = 10,
    seed: Optional[int] = None,
    filename_prefix: str = "random_topo"
) -> List[Path]:
    """
    批量生成随机拓扑图并保存
    
    参数:
        num_graphs: 生成图的数量
        output_dir: 输出目录，默认为 graph_data/random
        min_nodes: 最小节点数
        max_nodes: 最大节点数
        min_degree: 最小节点度
        max_degree: 最大节点度
        seed: 随机种子（如果指定，每个图使用 seed + i 作为种子）
        filename_prefix: 文件名前缀
    
    返回:
        List[Path]: 保存的文件路径列表
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "graph_data" / "random"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    print(f"开始批量生成 {num_graphs} 个随机拓扑图...")
    print(f"参数: 节点数 [{min_nodes}, {max_nodes}], 度 [{min_degree}, {max_degree}]")
    print(f"输出目录: {output_dir}")
    print("-" * 60)
    
    for i in range(num_graphs):
        # 设置种子
        graph_seed = (seed + i) if seed is not None else None
        
        # 生成图
        G = generate_random_topology(
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            min_degree=min_degree,
            max_degree=max_degree,
            seed=graph_seed
        )
        
        # 保存
        filename = f"{filename_prefix}_{i:03d}.graphml"
        saved_path = save_to_graphml(G, filename, output_dir)
        saved_paths.append(saved_path)
        
        # 打印统计信息
        degrees = [G.degree(n) for n in G.nodes()]
        print(f"  [{i+1}/{num_graphs}] 节点: {G.number_of_nodes()}, "
              f"边: {G.number_of_edges()}, "
              f"度范围: [{min(degrees)}, {max(degrees)}]")
    
    print("-" * 60)
    print(f"完成! 共生成 {len(saved_paths)} 个图")
    
    return saved_paths


def generate_with_specific_structure(
    num_nodes: int,
    edge_list: List[Tuple[int, int]],
    seed: Optional[int] = None
) -> nx.Graph:
    """
    基于指定的边列表生成图（用于生成特定结构）
    
    参数:
        num_nodes: 节点数量
        edge_list: 边列表 [(u, v), ...]
        seed: 随机种子
    
    返回:
        nx.Graph: 生成的图
    """
    if seed is not None:
        random.seed(seed)
    
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edge_list)
    
    _add_node_attributes(G)
    _add_edge_attributes(G)
    
    return G


# ============================================================================
# 统计和验证函数
# ============================================================================

def validate_graph(G: nx.Graph) -> dict:
    """
    验证生成的图是否满足约束
    
    返回:
        dict: 验证结果
    """
    degrees = [G.degree(n) for n in G.nodes()]
    
    result = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "is_connected": nx.is_connected(G),
        "min_degree": min(degrees) if degrees else 0,
        "max_degree": max(degrees) if degrees else 0,
        "avg_degree": sum(degrees) / len(degrees) if degrees else 0,
        "degree_distribution": {d: degrees.count(d) for d in set(degrees)},
    }
    
    return result


def print_graph_stats(G: nx.Graph) -> None:
    """
    打印图的统计信息
    """
    stats = validate_graph(G)
    
    print("=" * 50)
    print("图统计信息")
    print("=" * 50)
    print(f"节点数: {stats['num_nodes']}")
    print(f"边数: {stats['num_edges']}")
    print(f"是否连通: {stats['is_connected']}")
    print(f"最小度: {stats['min_degree']}")
    print(f"最大度: {stats['max_degree']}")
    print(f"平均度: {stats['avg_degree']:.2f}")
    print(f"度分布: {stats['degree_distribution']}")
    print("=" * 50)


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    # 示例 1: 生成单个随机图
    print("\n=== 示例 1: 生成单个随机图 ===")
    G = generate_random_topology(num_nodes=12, seed=42)
    print_graph_stats(G)
    
    # 保存
    output_dir = Path(__file__).parent.parent / "graph_data"
    save_to_graphml(G, "random_example.graphml", output_dir)
    
    # 示例 2: 批量生成
    print("\n=== 示例 2: 批量生成 10 个随机图 ===")
    batch_generate(
        num_graphs=10,
        min_nodes=7,
        max_nodes=20,
        min_degree=1,
        max_degree=10,
        seed=123
    )
