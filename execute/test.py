import networkx as nx
import argparse
import sys
import json  # 引入 json 库

# --- 核心逻辑类 (保持不变) ---
class NetworkController:
    def __init__(self):
        self.topology = nx.DiGraph()
        self._init_mock_topology()

    def _init_mock_topology(self):
        # 模拟拓扑
        edges = [
            ('A', 'B', {'bandwidth': 100, 'weight': 1}),
            ('A', 'C', {'bandwidth': 50,  'weight': 1}),
            ('B', 'D', {'bandwidth': 80,  'weight': 1}),
            ('C', 'D', {'bandwidth': 40,  'weight': 1}),
            ('B', 'E', {'bandwidth': 10,  'weight': 5}),
            ('D', 'E', {'bandwidth': 100, 'weight': 1})
        ]
        self.topology.add_edges_from(edges)

    def compute_path(self, src: str, dst: str, bandwidth_needed: float = 0) -> list:
        if src not in self.topology or dst not in self.topology:
            return None 

        valid_edges = [
            (u, v) for u, v, attrs in self.topology.edges(data=True)
            if attrs.get('bandwidth', 0) >= bandwidth_needed
        ]
        filtered_graph = self.topology.edge_subgraph(valid_edges)

        try:
            return nx.shortest_path(filtered_graph, source=src, target=dst, weight='weight')
        except nx.NetworkXNoPath:
            return None


def main():
    parser = argparse.ArgumentParser(description="path compute tool")
    
    # 参数定义
    parser.add_argument("--src", type=str, required=True, help="源地址")
    parser.add_argument("--dst", type=str, required=True, help="目的地址")
    parser.add_argument("--bw",  type=float, required=False, default=0, help="带宽需求, 默认没有需求也就是可达就行")

    args = parser.parse_args()

    controller = NetworkController()
    path = controller.compute_path(args.src, args.dst, args.bw)
    
    # --- 构建 JSON 输出 ---
    result_data = {
        "src": args.src,
        "dst": args.dst,
        "bandwidth_req": args.bw,
        "success": False, # 默认失败
        "path": None,
        "message": ""
    }

    if path:
        result_data["success"] = True
        result_data["path"] = path
        result_data["message"] = "Path found"
    else:
        result_data["success"] = False
        result_data["message"] = "No path found or node does not exist"

    # 使用 json.dumps 将字典转换为 JSON 字符串并打印
    print(json.dumps(result_data, ensure_ascii=False))

if __name__ == "__main__":
    main()