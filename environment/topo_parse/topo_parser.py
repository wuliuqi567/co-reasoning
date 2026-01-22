#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
网络拓扑解析器

功能:
    1. parse_topology(json_path) -> 从JSON文件解析得到networkx图
    2. update_link_metrics(G, json_path) -> 从link_metric.json更新图的边属性
    3. visualize(G, output_path)  -> 可视化拓扑图

使用示例:
    from topo_parser import parse_topology, update_link_metrics, visualize

    # 解析JSON得到图
    G = parse_topology("response_1768554636080.json")
    
    # 更新链路属性
    G = update_link_metrics(G, "link_metric.json")

    # 可视化
    visualize(G, "topo.png")
"""

import json
from pathlib import Path
from typing import Union, Optional

import networkx as nx

# ============================================================================
# 配置常量
# ============================================================================

NODE_TYPE_NAMES = {
    1: "I类终端",
    2: "I类簇头",
    3: "II类车载",
    4: "II类接入",
    5: "II类骨干",
    6: "IV类网关",
    8: "III类网管",
}

NODE_COLORS = {
    3: "#95E1D3",   # II类车载 - 浅绿
    4: "#95E1D3",   # II类接入 - 青色
    5: "#95E1D3",   # II类骨干 - 红色
    6: "#F38181",   # IV类网关 - 粉红
}
DEFAULT_COLOR = "#C7CEEA"  # 默认 - 浅蓝


# ============================================================================
# 核心解析函数
# ============================================================================

def parse_topology(source: Union[str, Path, dict]) -> nx.Graph:
    """
    解析拓扑数据，返回networkx图

    参数:
        source: JSON文件路径(str/Path) 或 已解析的字典

    返回:
        nx.Graph: 节点为网络设备，边为链路连接

    节点属性:
        - node_id, node_type, node_type_name
        - node_manage_ip_addr, node_location, node_status
        - longitude, latitude (从node_location解析的经纬度)

    边属性:
        - link_id, link_status
        - link_bandwidth, link_latency
        - src_port, dst_port
    """
    # 加载数据
    if isinstance(source, dict):
        data = source
    else:
        with open(source, 'r', encoding='utf-8') as f:
            data = json.load(f)

    # 提取topo部分（兼容多种格式）
    topo = _extract_topo(data)
    nodes = topo.get('node', [])
    links = topo.get('link', [])

    # 构建图
    G = nx.Graph()

    # 添加节点
    for node in nodes:
        node_id = node.get('node_id')
        if not node_id:
            continue

        node_type = node.get('node_type', 0)
        location_str = node.get('node_location', '')
        
        # 解析经纬度
        longitude, latitude = None, None
        if location_str and ',' in location_str:
            try:
                parts = location_str.split(',')
                longitude = float(parts[0].strip())
                latitude = float(parts[1].strip())
            except (ValueError, IndexError):
                pass
        
        # 端口信息
        ports = node.get('node_ports') or []
        port_count = len(ports)
        port_ids = [p.get('port_id', '') for p in ports]
        
        G.add_node(
            node_id,
            node_id=node_id,
            node_type=node_type,
            node_type_name=NODE_TYPE_NAMES.get(node_type, f'类型{node_type}'),
            node_manage_ip_addr=node.get('node_manage_ip_addr', ''),
            node_location=location_str,
            longitude=longitude,
            latitude=latitude,
            node_status=node.get('node_status'),
            port_count=port_count,
            port_ids=port_ids,
        )

    # 添加边
    for link in links:
        src_node = link.get('src', {}).get('src_node')
        dst_node = link.get('dst', {}).get('dst_node')

        if not src_node or not dst_node:
            continue
        if src_node not in G or dst_node not in G:
            continue

        # 处理无效的带宽/延迟值
        bandwidth = link.get('link_bandwidth')
        latency = link.get('link_latency')
        if bandwidth in (-1, "", None):
            bandwidth = None
        if latency in (-1, "", None):
            latency = None

        G.add_edge(
            src_node, dst_node,
            link_id=link.get('link_id', ''),
            link_status=link.get('link_status'),
            link_bandwidth=bandwidth,
            link_latency=latency,
            src_port=link.get('src', {}).get('src_port', ''),
            dst_port=link.get('dst', {}).get('dst_port', ''),
        )

    return G


def update_link_metrics(G: nx.Graph, json_path: Union[str, Path]) -> nx.Graph:
    """
    从link_metric.json文件更新图的边属性
    
    参数:
        G: networkx Graph对象
        json_path: link_metric.json文件路径
    
    返回:
        更新后的Graph对象
    
    更新的属性:
        - link_latency: 链路时延（ms）
        - link_utilization: 链路利用率（0-1）
        - bandwidth_capacity_available: 有效带宽容量（Mbps）
        - link_loss_rate: 链路丢包率（0-1）
        - flow_table_status: 流表状态
    
    使用示例:
        G = parse_topology("topo.json")
        G = update_link_metrics(G, "link_metric.json")
    """
    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取link_metrics数组
    link_metrics = data.get('data', {}).get('link_metrics', [])
    
    if not link_metrics:
        print("警告: link_metrics数据为空")
        return G
    
    # 统计更新情况
    updated_count = 0
    not_found_count = 0
    
    # 遍历每个链路指标
    for metric in link_metrics:
        link_id = metric.get('link_id', '')
        if not link_id:
            continue
        
        # 解析link_id，可能的格式：
        # 1. "节点1_节点2"
        # 2. "节点1:端口1_节点2:端口2"
        parts = link_id.split('_')
        if len(parts) != 2:
            continue
        
        src_part, dst_part = parts
        
        # 提取节点ID（去掉端口号）
        src_node = src_part.split(':')[0] if ':' in src_part else src_part
        dst_node = dst_part.split(':')[0] if ':' in dst_part else dst_part
        
        # 检查边是否存在
        if not G.has_edge(src_node, dst_node):
            not_found_count += 1
            continue
        
        # 获取边的属性字典（Graph 只有一条边）
        edge_data = G[src_node][dst_node]
        
        # 更新边的属性
        _update_edge_metrics(edge_data, metric)
        updated_count += 1
    
    print(f"链路属性更新完成:")
    print(f"  - 总链路数: {len(link_metrics)}")
    print(f"  - 成功更新: {updated_count}")
    print(f"  - 未找到边: {not_found_count}")
    
    return G


def _update_edge_metrics(edge_data: dict, metric: dict) -> None:
    """
    更新边的指标属性
    
    参数:
        edge_data: 边的属性字典
        metric: 从link_metric.json读取的指标数据
    """
    # 更新链路时延
    latency = metric.get('link_latency')
    if latency is not None and latency != -1:
        edge_data['link_latency'] = latency
    
    # 更新链路利用率
    utilization = metric.get('link_utilization')
    if utilization is not None:
        edge_data['link_utilization'] = utilization
    
    # 更新有效带宽容量
    bandwidth_available = metric.get('bandwidth_capacity_available')
    if bandwidth_available is not None:
        edge_data['bandwidth_capacity_available'] = bandwidth_available
    
    # 更新丢包率
    loss_rate = metric.get('link_loss_rate')
    if loss_rate is not None:
        edge_data['link_loss_rate'] = loss_rate
    
    # 更新流表状态
    flow_status = metric.get('flow_table_status')
    if flow_status is not None:
        edge_data['flow_table_status'] = flow_status


def _extract_topo(data: dict) -> dict:
    """从不同格式的JSON中提取topo部分"""
    if 'data' in data and 'topo' in data['data']:
        return data['data']['topo']
    elif 'topo' in data:
        return data['topo']
    return data


# ============================================================================
# 可视化函数
# ============================================================================

def _parse_location(location_str: str) -> Optional[tuple]:
    """
    解析经纬度字符串 "经度,纬度" -> (lon, lat)
    """
    if not location_str or ',' not in location_str:
        return None
    try:
        parts = location_str.split(',')
        lon = float(parts[0].strip())
        lat = float(parts[1].strip())
        return (lon, lat)
    except (ValueError, IndexError):
        return None


def _get_geo_positions(G: nx.Graph, spread_factor: float = 1.5) -> dict:
    """
    根据节点经纬度生成位置字典，并进行归一化和间距优化
    
    参数:
        G: networkx图
        spread_factor: 间距放大因子，用于分散重叠节点
    
    返回:
        pos: {node_id: (x, y)} 位置字典，归一化到合适范围
    """
    import math
    
    raw_pos = {}
    nodes_without_pos = []
    
    # 收集有经纬度的节点位置
    for node in G.nodes():
        location = G.nodes[node].get('node_location', '')
        coords = _parse_location(location)
        if coords:
            raw_pos[node] = coords
        else:
            nodes_without_pos.append(node)
    
    if not raw_pos:
        return nx.spring_layout(G, k=2.5, iterations=60, seed=42)
    
    # 计算边界
    lons = [p[0] for p in raw_pos.values()]
    lats = [p[1] for p in raw_pos.values()]
    
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    
    # 计算范围，避免除零
    lon_range = max(max_lon - min_lon, 1e-6)
    lat_range = max(max_lat - min_lat, 1e-6)
    
    # 归一化到 [0, 1] 范围，然后放大
    pos = {}
    for node, (lon, lat) in raw_pos.items():
        x = (lon - min_lon) / lon_range
        y = (lat - min_lat) / lat_range
        pos[node] = (x, y)
    
    # 检测并分散重叠节点
    pos = _spread_overlapping_nodes(pos, min_dist=0.08 * spread_factor)
    
    # 为没有经纬度的节点生成位置
    if nodes_without_pos:
        # 在图的边缘放置无位置节点
        existing_x = [p[0] for p in pos.values()]
        existing_y = [p[1] for p in pos.values()]
        center_x = sum(existing_x) / len(existing_x)
        center_y = sum(existing_y) / len(existing_y)
        
        for i, node in enumerate(nodes_without_pos):
            angle = 2 * math.pi * i / max(len(nodes_without_pos), 1)
            radius = 0.3 * spread_factor
            pos[node] = (
                center_x + radius * math.cos(angle),
                center_y + radius * math.sin(angle)
            )
    
    return pos


def _spread_overlapping_nodes(pos: dict, min_dist: float = 0.1) -> dict:
    """
    分散重叠或过于接近的节点
    """
    import math
    
    nodes = list(pos.keys())
    new_pos = {n: list(p) for n, p in pos.items()}
    
    # 多轮迭代分散节点
    for _ in range(50):
        moved = False
        for i, n1 in enumerate(nodes):
            for n2 in nodes[i+1:]:
                p1, p2 = new_pos[n1], new_pos[n2]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                dist = math.sqrt(dx*dx + dy*dy)
                
                if dist < min_dist and dist > 0:
                    # 计算推开方向
                    factor = (min_dist - dist) / dist / 2
                    new_pos[n1][0] -= dx * factor
                    new_pos[n1][1] -= dy * factor
                    new_pos[n2][0] += dx * factor
                    new_pos[n2][1] += dy * factor
                    moved = True
                elif dist == 0:
                    # 完全重叠，随机推开
                    angle = hash(str(n1) + str(n2)) % 360 * math.pi / 180
                    new_pos[n2][0] += min_dist * math.cos(angle)
                    new_pos[n2][1] += min_dist * math.sin(angle)
                    moved = True
        
        if not moved:
            break
    
    return {n: tuple(p) for n, p in new_pos.items()}


def visualize(
    G: nx.Graph,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "网络拓扑图",
    figsize: tuple = (16, 12),
    show: bool = False,
    use_geo: bool = True,
    spread_factor: float = 2.0
) -> None:
    """
    可视化拓扑图

    参数:
        G: networkx图
        output_path: 输出文件路径 (None则不保存)
        title: 图标题
        figsize: 图片尺寸
        show: 是否显示图片(plt.show)
        use_geo: 是否使用经纬度定位节点 (默认True)
        spread_factor: 节点间距放大因子 (默认2.0)
    """
    import matplotlib.pyplot as plt
    import matplotlib
    # 使用系统中可用的中文字体
    matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'AR PL UKai CN', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=figsize, dpi=120)
    # 标题显示节点和边数
    full_title = f"{title}\n节点: {G.number_of_nodes()} | 边: {G.number_of_edges()}"
    ax.set_title(full_title, fontsize=16, fontweight='bold', pad=20)
    ax.set_axis_off()
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('#fafafa')

    # 根据选项选择布局方式
    if use_geo:
        pos = _get_geo_positions(G, spread_factor=spread_factor)
    else:
        pos = nx.spring_layout(G, k=2.5, iterations=60, seed=42)

    # 按类型设置颜色和大小
    colors = []
    sizes = []
    for node in G.nodes():
        node_type = G.nodes[node].get('node_type', 0)
        colors.append(NODE_COLORS.get(node_type, DEFAULT_COLOR))
        # 骨干和网关节点更大
        if node_type == 5:
            sizes.append(2000)
        elif node_type == 6:
            sizes.append(1800)
        elif node_type in (3, 4):
            sizes.append(1400)
        else:
            sizes.append(1000)

    # 绘制边（使用不同颜色区分带宽）
    edges = list(G.edges(data=True))
    edge_colors = []
    edge_widths = []
    for u, v, data in edges:
        bw = data.get('link_bandwidth')
        if bw and bw >= 1000:
            edge_colors.append('#2ecc71')  # 高带宽 - 绿色
            edge_widths.append(3.0)
        elif bw and bw > 0:
            edge_colors.append('#3498db')  # 中带宽 - 蓝色
            edge_widths.append(2.0)
        else:
            edge_colors.append('#95a5a6')  # 未知带宽 - 灰色
            edge_widths.append(1.5)
    
    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        alpha=0.7,
        edge_color=edge_colors,
        style='solid',
        ax=ax,
        connectionstyle="arc3,rad=0.1"  # 弧形边，避免重叠
    )

    # 绘制节点
    nx.draw_networkx_nodes(
        G, pos,
        node_color=colors,
        node_size=sizes,
        alpha=0.95,
        edgecolors='#333333',
        linewidths=2,
        ax=ax
    )

    # 绘制标签 (显示IP和类型)
    labels = {}
    for node in G.nodes():
        attrs = G.nodes[node]
        ip = attrs.get('node_manage_ip_addr') or ''
        type_name = attrs.get('node_type_name', '')
        # 显示完整IP或简短类型名
        if ip:
            labels[node] = f"{ip}\n{type_name}"
        else:
            labels[node] = type_name

    nx.draw_networkx_labels(
        G, pos, labels, 
        font_size=9,
        font_weight='bold',
        font_color='#222222',
        ax=ax
    )

    # 添加图例 (节点类型 + 边带宽)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = []
    
    # 节点类型图例
    type_in_graph = set(G.nodes[n].get('node_type', 0) for n in G.nodes())
    for node_type in sorted(type_in_graph):
        color = NODE_COLORS.get(node_type, DEFAULT_COLOR)
        name = NODE_TYPE_NAMES.get(node_type, f'类型{node_type}')
        legend_elements.append(Patch(facecolor=color, edgecolor='#333', label=name))
    
    # 边带宽图例
    legend_elements.append(Line2D([0], [0], color='#2ecc71', linewidth=3, label='高带宽 (≥1000Mbps)'))
    legend_elements.append(Line2D([0], [0], color='#3498db', linewidth=2, label='中带宽 (<1000Mbps)'))
    legend_elements.append(Line2D([0], [0], color='#95a5a6', linewidth=1.5, label='未知带宽'))
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150, facecolor='#fafafa')
        print(f"✓ 已保存拓扑图: {output_path}")

    if show:
        plt.show()

    plt.close()


# ============================================================================
# 辅助函数
# ============================================================================

def get_subgraph(G: nx.Graph, node_types: set) -> nx.Graph:
    """获取指定类型节点的子图"""
    nodes_to_keep = [
        n for n, attrs in G.nodes(data=True)
        if attrs.get('node_type') in node_types
    ]
    return G.subgraph(nodes_to_keep).copy()


def compute_ii_shortest_paths(G: nx.Graph, print_result: bool = True) -> list:
    """
    计算所有II类节点对之间的最短路径（跳数>1的）
    
    参数:
        G: networkx图
        print_result: 是否打印结果 (默认True)
    
    返回:
        list: [(src_node, dst_node, path, hop_count), ...]
              path为节点列表，hop_count为跳数
    """
    # II类节点类型: 3=车载, 4=接入, 5=骨干
    II_TYPES = {3, 4, 5}
    
    # 获取所有II类节点
    ii_nodes = [
        n for n in G.nodes()
        if G.nodes[n].get('node_type') in II_TYPES
    ]
    
    if print_result:
        print("=" * 70)
        print("II类节点最短路径分析")
        print("=" * 70)
        print(f"II类节点数: {len(ii_nodes)}")
        print()
    
    results = []
    computed_pairs = set()  # 避免重复计算
    
    for src in ii_nodes:
        for dst in ii_nodes:
            if src >= dst:  # 避免重复 (A,B) 和 (B,A)
                continue
            
            pair = (src, dst)
            if pair in computed_pairs:
                continue
            computed_pairs.add(pair)
            
            # 检查是否直接相连（一跳）
            if G.has_edge(src, dst):
                continue  # 跳过一跳的
            
            # 计算最短路径
            try:
                path = nx.shortest_path(G, src, dst)
                hop_count = len(path) - 1
                
                if hop_count > 1:  # 只记录跳数>1的
                    results.append((src, dst, path, hop_count))
            except nx.NetworkXNoPath:
                pass  # 无法到达
    
    # 按跳数排序
    results.sort(key=lambda x: x[3])
    
    if print_result:
        if results:
            print(f"{'源节点':<16} {'目标节点':<16} {'跳数':>4}   路径")
            print("-" * 70)
            for src, dst, path, hop_count in results:
                src_ip = G.nodes[src].get('node_manage_ip_addr') or src[:12]
                dst_ip = G.nodes[dst].get('node_manage_ip_addr') or dst[:12]
                
                # 路径显示为IP地址序列
                path_ips = []
                for node in path:
                    ip = G.nodes[node].get('node_manage_ip_addr') or ''
                    if ip:
                        path_ips.append(ip.split('.')[-1])  # 只显示最后一段
                    else:
                        path_ips.append(node[:6])
                path_str = " -> ".join(path_ips)
                
                print(f"{src_ip:<16} {dst_ip:<16} {hop_count:>4}   {path_str}")
            print()
            print(f"共 {len(results)} 条需要多跳的路径")
        else:
            print("所有II类节点对都是直接相连的（一跳）")
        print("=" * 70)
    
    return results


def summary(G: nx.Graph, show_edges: bool = True) -> None:
    """
    打印拓扑简要信息，包括节点统计和边连接关系
    
    参数:
        G: networkx图
        show_edges: 是否显示边连接关系 (默认True)
    """
    from collections import Counter

    print("=" * 60)
    print("拓扑摘要")
    print("=" * 60)
    print(f"节点总数: {G.number_of_nodes()}  |  边总数: {G.number_of_edges()}")
    print()
    
    # 节点类型统计
    print("【节点类型统计】")
    type_counts = Counter(
        G.nodes[n].get('node_type', 0) for n in G.nodes()
    )
    for t, count in sorted(type_counts.items()):
        name = NODE_TYPE_NAMES.get(t, f'类型{t}')
        print(f"  {name}: {count}")
    print()
    
    # 节点列表（含端口和边数统计）
    print("【节点列表】")
    print(f"  {'IP地址':<16} | {'类型':<10} | {'端口数':>4} | {'边数':>4} | {'节点ID'}")
    print("  " + "-" * 70)
    for node in sorted(G.nodes(), key=lambda n: G.nodes[n].get('node_manage_ip_addr') or ''):
        attrs = G.nodes[node]
        ip = attrs.get('node_manage_ip_addr', '') or '(无IP)'
        type_name = attrs.get('node_type_name', '')
        port_count = attrs.get('port_count', 0)
        edge_count = G.degree(node)
        node_short = node[:12] + '...' if len(node) > 15 else node
        print(f"  {ip:<16} | {type_name:<10} | {port_count:>4} | {edge_count:>4} | {node_short}")
    print()
    
    # 边连接关系
    if show_edges:
        print("【边连接关系】")
        print(f"  {'源节点':<16} {'源端口':>4}   {'目标端口':<4} {'目标节点':<16} {'带宽':<12} {'延迟':<10}")
        print("  " + "-" * 80)
        for u, v, data in G.edges(data=True):
            u_ip = G.nodes[u].get('node_manage_ip_addr') or u[:12]
            v_ip = G.nodes[v].get('node_manage_ip_addr') or v[:12]
            
            # 提取端口号
            src_port = data.get('src_port', '')
            dst_port = data.get('dst_port', '')
            src_port_num = src_port.split(':')[-1] if ':' in src_port else '-'
            dst_port_num = dst_port.split(':')[-1] if ':' in dst_port else '-'
            
            bandwidth = data.get('link_bandwidth')
            latency = data.get('link_latency')
            
            # 格式化带宽和延迟
            bw_str = f"{bandwidth}Mbps" if bandwidth else "N/A"
            lat_str = f"{latency}ms" if latency else "N/A"
            
            print(f"  {u_ip:<16} :{src_port_num:<3} <-> :{dst_port_num:<3} {v_ip:<16} {bw_str:<12} {lat_str:<10}")
        print()
    
    print("=" * 60)


# ============================================================================
# 主函数 (演示用法)
# ============================================================================

if __name__ == "__main__":
    # 示例：解析 + 更新链路属性 + 可视化
    json_file = Path(__file__).parent.parent / "jsondata/topo.json"
    link_metric_file = Path(__file__).parent.parent / "jsondata/link_metric.json"

    if json_file.exists():
        # 1. 解析拓扑
        G = parse_topology(json_file)
        print("拓扑解析完成！")
        
        # 2. 更新链路属性
        if link_metric_file.exists():
            print("\n更新链路属性...")
            G = update_link_metrics(G, link_metric_file)
        else:
            print(f"\n警告: 未找到链路属性文件: {link_metric_file}")
        
        # 3. 显示拓扑摘要
        summary(G)
        
        # 4. 计算最短路径
        results = compute_ii_shortest_paths(G, print_result=True)
        
        # 5. 可视化
        visualize(G, "topo.png", title="网络拓扑图")

        # 6. 获取II类子图 (可选)
        G_ii = get_subgraph(G, {3, 4, 5})
        print(f"\nII类子图: {G_ii.number_of_nodes()} 节点, {G_ii.number_of_edges()} 边")
        visualize(G_ii, "topo_ii.png", title="II类网络拓扑")
    else:
        print(f"找不到文件: {json_file}")
