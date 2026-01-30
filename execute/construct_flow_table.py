"""
将端到端路径转换为每台设备上的流表规则
根据 flow_table_rule.md 中的规范生成流表配置
"""

from typing import List, Dict, Any, Optional

def construct_flow_table(
    path_ip_ports: List[List[Dict[str, Any]]],
    # 移除了必须传入的 src_ip 和 dst_ip，改为自动提取
    priority: int = 5000,
    table_id: int = 0,
    origin: int = 1,
    flow_id_prefix: str = "Flow"
) -> List[Dict[str, Any]]:
    """
    将路径信息转换为每台设备上的流表规则
    
    Args:
        path_ip_ports: 路径的IP和端口信息列表
        priority: 流表规则优先级，默认5000
        table_id: 流表ID，默认0
        origin: 来源标识，1=北邮路由调控，默认1
        flow_id_prefix: 流表规则ID前缀，默认"Flow"
    
    Returns:
        每个路径对应的流表规则列表
    """
    all_flow_tables = []
    
    for path_idx, path_info in enumerate(path_ip_ports):
        if not path_info or len(path_info) < 2:
            continue  # 跳过空路径或只有单个节点的路径
        
        # ==========================================================
        # 【新增】自动提取源IP和目的IP
        # ==========================================================
        # 源IP：路径第一个节点的IP
        # 目的IP：路径最后一个节点的IP
        try:
            raw_src_ip = path_info[0].get('ip')
            raw_dst_ip = path_info[-1].get('ip')
            
            if not raw_src_ip or not raw_dst_ip:
                print(f"[警告] 路径 {path_idx} 缺少起点或终点IP，跳过生成。")
                continue

            # 自动补全 /32 掩码 (如果原始IP不带掩码的话)
            src_ip_rule = raw_src_ip if '/' in raw_src_ip else f"{raw_src_ip}/32"
            dst_ip_rule = raw_dst_ip if '/' in raw_dst_ip else f"{raw_dst_ip}/32"
            
            # print(f"[Debug] 自动提取匹配规则: Src={src_ip_rule} -> Dst={dst_ip_rule}")
            
        except IndexError:
            continue

        # ==========================================================
        
        # 为路径上的每个节点（除了最后一个）生成流表规则
        for i in range(len(path_info) - 1):
            current_node = path_info[i]
            next_node = path_info[i + 1]
            
            # 获取当前节点的IP和出端口
            node_ip = current_node.get('ip', '')
            node_idx = current_node.get('node_idx', -1)
            out_port = current_node.get('out_port', '')
            
            # 获取下一跳节点的IP
            next_hop_ip = next_node.get('ip', '')
            
            if not node_ip or not out_port or not next_hop_ip:
                continue  # 跳过缺少必要信息的节点
            
            # 生成流表规则ID
            flow_id = f"{flow_id_prefix}_Path{path_idx}_Node{node_idx}_to_Node{next_node.get('node_idx', 'X')}"
            
            # 构造流表配置
            flowtable_config = {
                "origin": origin,
                "flowtable": [
                    {
                        "id": flow_id,
                        "table_id": table_id,
                        "priority": priority,
                        "match": {
                            "nw_src": src_ip_rule,  # 使用自动提取的源IP
                            "nw_dst": dst_ip_rule,  # 使用自动提取的目的IP
                            "ethernet-match": {
                                "ethernet-type": {"type": 2048}  # IPv4
                            }
                        },
                        "instructions": {
                            "instruction": [
                                {
                                    "order": 1,
                                    "apply-actions": {
                                        "action": [
                                            {
                                                "order": 0,
                                                "output": out_port,  # 物理出接口ID
                                                "nextHop": next_hop_ip  # 下一跳IP地址
                                            }
                                        ]
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
            
            all_flow_tables.append({
                'node_ip': node_ip,
                'node_idx': node_idx,
                'flowtable_config': flowtable_config
            })
    
    return all_flow_tables


def format_flow_table_for_api(flow_tables: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    将流表规则按节点IP分组，便于按节点调用API
    """
    grouped = {}
    
    for flow_table in flow_tables:
        node_ip = flow_table['node_ip']
        if node_ip not in grouped:
            grouped[node_ip] = []
        
        grouped[node_ip].append({
            'flowtable_config': flow_table['flowtable_config'],
            'node_idx': flow_table['node_idx']
        })
    
    return grouped


def merge_flow_tables_for_node(node_flow_tables: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    将同一节点的多个流表规则合并为一个配置（合并flowtable数组）
    """
    if not node_flow_tables:
        return {}
    
    # 使用第一个配置作为基础
    merged_config = node_flow_tables[0]['flowtable_config'].copy()
    
    # 合并所有flowtable规则
    all_flowtables = []
    for item in node_flow_tables:
        config = item['flowtable_config']
        if 'flowtable' in config:
            all_flowtables.extend(config['flowtable'])
    
    merged_config['flowtable'] = all_flowtables
    
    return merged_config


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    import json
    
    # 示例数据（模拟从 rl_reroute.py 传过来的数据）
    # 注意：第一个节点是源（192.168.187.87），最后一个节点是目的（192.168.187.17）
    example_path_ip_ports = [
        [
            {'node_idx': 0, 'ip': '192.168.187.87', 'out_port': '00010435cc19325c6e49a900004400000000:3'},
            {'node_idx': 3, 'ip': '192.168.187.114', 'in_port': '...:1', 'out_port': '...:4'},
            {'node_idx': 4, 'ip': '192.168.187.13', 'in_port': '...:2', 'out_port': '...:3'},
            {'node_idx': 10, 'ip': '192.168.187.17', 'in_port': '...:2'}
        ]
    ]
    test_path = [[{'node_idx': 10, 'ip': '192.168.2.12', 'out_port': '00012400163632633800004400000000:1'}, {'node_idx': 17, 'ip': '192.168.2.24', 'in_port': '00012500166530323400004400000000:6', 'out_port': '00012500163431326600004400000000:3'}, {'node_idx': 16, 'ip': '192.168.2.26', 'in_port': '00012500166530323400004400000000:4', 'out_port': '00012500163431326600004400000000:5'}, {'node_idx': 14, 'ip': '192.168.2.30', 'in_port': '00012400166636363500004400000000:1'}]]
    
    print("=" * 80)
    print("测试自动提取源目的IP功能")
    print("=" * 80)
    
    # 此时调用不再需要传 src_ip 和 dst_ip
    flow_tables = construct_flow_table(
        path_ip_ports=test_path
    )
    
    # 分组并合并
    grouped = format_flow_table_for_api(flow_tables)
    for node_ip, configs in grouped.items():
        print(f"\n节点 {node_ip}:")
        merged = merge_flow_tables_for_node(configs)
        print(json.dumps(merged, indent=2, ensure_ascii=False))