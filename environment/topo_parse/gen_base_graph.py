import networkx as nx
from pathlib import Path

# 读取 latest_II_class2.graphml 文件, 然后将所有链路和节点置为故障状态
graph_file = Path(__file__).parent.parent / "graph_data/latest_II_class.graphml"
G = nx.read_graphml(graph_file)
for node in G.nodes():
    G.nodes[node]['node_status'] = -1
for edge in G.edges():
    G.edges[edge]['link_status'] = -1
nx.write_graphml(G, graph_file.with_suffix(".base.graphml"))


