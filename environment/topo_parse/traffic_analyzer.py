#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
网络流量分析器

功能:
    1. parse_traffic_metrics() -> 解析节点流量统计
    2. build_traffic_vector() -> 构建流量向量
    3. estimate_traffic_matrix() -> 基于启发式估算流量矩阵（近似）
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Dict, Tuple
import networkx as nx


def parse_traffic_metrics(json_path: Union[str, Path]) -> pd.DataFrame:
    """
    解析流量统计数据，返回DataFrame
    
    参数:
        json_path: NM_traffic_metrics.json 文件路径
    
    返回:
        DataFrame with columns: node_id, receive_traffic_total, send_traffic_total, 
                                traffic_rate, packet_loss_rate
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    metrics = data.get('data', {}).get('traffic_metrics', [])
    
    # 转换为 DataFrame
    df = pd.DataFrame(metrics)
    
    # 处理 null 值
    df['receive_traffic_total'] = df['receive_traffic_total'].fillna(0)
    df['send_traffic_total'] = df['send_traffic_total'].fillna(0)
    df['traffic_rate'] = df['traffic_rate'].fillna(0)
    df['packet_loss_rate'] = df['packet_loss_rate'].fillna(0)
    
    # 添加计算字段
    df['total_traffic'] = df['receive_traffic_total'] + df['send_traffic_total']
    df['traffic_balance'] = df['receive_traffic_total'] - df['send_traffic_total']
    
    return df


def build_traffic_vector(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    构建节点流量特征向量
    
    返回:
        {
            node_id: {
                'receive': 接收流量,
                'send': 发送流量,
                'rate': 流量速率,
                'loss': 丢包率,
                'total': 总流量,
                'balance': 流量平衡（正值=接收多，负值=发送多）
            }
        }
    """
    traffic_vector = {}
    
    for _, row in df.iterrows():
        traffic_vector[row['node_id']] = {
            'receive': float(row['receive_traffic_total']),
            'send': float(row['send_traffic_total']),
            'rate': float(row['traffic_rate']),
            'loss': float(row['packet_loss_rate']),
            'total': float(row['total_traffic']),
            'balance': float(row['traffic_balance'])
        }
    
    return traffic_vector


def estimate_traffic_matrix_gravity(
    G: nx.Graph, 
    traffic_vector: Dict[str, Dict[str, float]],
    method: str = 'gravity'
) -> Tuple[np.ndarray, list]:
    """
    基于重力模型估算流量矩阵（启发式方法，仅为近似）
    
    警告：这是基于假设的估算，不是真实的流量矩阵！
    
    参数:
        G: networkx 图
        traffic_vector: 节点流量向量
        method: 估算方法
            - 'gravity': 重力模型（流量与节点发送/接收量成正比）
            - 'uniform': 均匀分配
    
    返回:
        (traffic_matrix, node_list)
        traffic_matrix[i][j] = 从节点i到节点j的估算流量
    """
    # 获取所有节点列表
    nodes = list(G.nodes())
    n = len(nodes)
    
    # 初始化流量矩阵
    traffic_matrix = np.zeros((n, n))
    
    if method == 'gravity':
        # 重力模型：TM[i][j] = send[i] * receive[j] / sum(receive[all])
        total_receive = sum(traffic_vector.get(node, {}).get('receive', 0) for node in nodes)
        
        if total_receive == 0:
            print("警告: 总接收流量为0，无法估算流量矩阵")
            return traffic_matrix, nodes
        
        for i, src in enumerate(nodes):
            src_send = traffic_vector.get(src, {}).get('send', 0)
            
            if src_send == 0:
                continue
            
            for j, dst in enumerate(nodes):
                if i == j:
                    continue  # 不考虑自环
                
                dst_receive = traffic_vector.get(dst, {}).get('receive', 0)
                
                # 重力模型公式
                traffic_matrix[i][j] = (src_send * dst_receive) / total_receive
    
    elif method == 'uniform':
        # 均匀分配：每个节点的发送流量平均分配给所有其他节点
        for i, src in enumerate(nodes):
            src_send = traffic_vector.get(src, {}).get('send', 0)
            
            if src_send == 0:
                continue
            
            # 统计有多少个目标节点（排除自己）
            num_targets = n - 1
            if num_targets == 0:
                continue
            
            avg_traffic = src_send / num_targets
            
            for j in range(n):
                if i != j:
                    traffic_matrix[i][j] = avg_traffic
    
    return traffic_matrix, nodes


def print_traffic_matrix(traffic_matrix: np.ndarray, nodes: list, top_k: int = 10):
    """
    打印流量矩阵（显示前 top_k 个最大流量对）
    """
    print("\n" + "="*80)
    print("流量矩阵估算结果（仅供参考，非真实数据）")
    print("="*80)
    
    # 找出最大的K个流量对
    flow_pairs = []
    n = len(nodes)
    
    for i in range(n):
        for j in range(n):
            if i != j and traffic_matrix[i][j] > 0:
                flow_pairs.append((
                    nodes[i][:20],
                    nodes[j][:20],
                    traffic_matrix[i][j]
                ))
    
    # 按流量大小排序
    flow_pairs.sort(key=lambda x: -x[2])
    
    print(f"\nTop {top_k} 流量对 (源 -> 目的):")
    print(f"{'源节点':<22} -> {'目的节点':<22}   {'估算流量(MB)':>15}")
    print("-" * 80)
    
    for src, dst, traffic in flow_pairs[:top_k]:
        print(f"{src:<22} -> {dst:<22}   {traffic:>15.2f}")
    
    # 统计信息
    total_traffic = traffic_matrix.sum()
    print(f"\n总流量（估算）: {total_traffic:.2f} MB")
    print(f"平均流量对: {traffic_matrix.mean():.2f} MB")
    print(f"非零流量对数量: {np.count_nonzero(traffic_matrix)}")


def analyze_traffic(
    topo_path: Union[str, Path],
    traffic_path: Union[str, Path]
):
    """
    完整的流量分析流程
    
    参数:
        topo_path: topo.json 路径
        traffic_path: NM_traffic_metrics.json 路径
    """
    from topo_parser import parse_topology
    
    print("="*80)
    print("网络流量分析")
    print("="*80)
    
    # 1. 解析拓扑
    print("\n[1/4] 解析拓扑图...")
    G = parse_topology(topo_path)
    print(f"✓ 节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")
    
    # 2. 解析流量数据
    print("\n[2/4] 解析流量统计...")
    df = parse_traffic_metrics(traffic_path)
    print(f"✓ 流量记录数: {len(df)}")
    
    # 3. 构建流量向量
    print("\n[3/4] 构建流量向量...")
    traffic_vector = build_traffic_vector(df)
    
    # 显示流量统计
    print("\n【节点流量统计 Top 10】")
    df_sorted = df.sort_values('total_traffic', ascending=False)
    print(df_sorted[['node_id', 'receive_traffic_total', 'send_traffic_total', 
                     'total_traffic', 'traffic_rate']].head(10).to_string(index=False))
    
    # 4. 估算流量矩阵（基于重力模型）
    print("\n[4/4] 估算流量矩阵（重力模型）...")
    print("⚠️  警告：这是基于启发式假设的估算，不是真实的流量矩阵！")
    print("    真实流量矩阵需要流级监控数据（记录每个流的源、目的地）")
    
    traffic_matrix, nodes = estimate_traffic_matrix_gravity(G, traffic_vector, method='gravity')
    print_traffic_matrix(traffic_matrix, nodes, top_k=15)
    
    return G, df, traffic_vector, traffic_matrix, nodes


if __name__ == "__main__":
    # 示例用法
    topo_file = Path(__file__).parent.parent / "jsondata/topo.json"
    traffic_file = Path(__file__).parent.parent / "jsondata/NM_traffic_metrics.json"
    
    if topo_file.exists() and traffic_file.exists():
        G, df, traffic_vector, traffic_matrix, nodes = analyze_traffic(topo_file, traffic_file)
    else:
        print(f"文件不存在:")
        if not topo_file.exists():
            print(f"  - {topo_file}")
        if not traffic_file.exists():
            print(f"  - {traffic_file}")
