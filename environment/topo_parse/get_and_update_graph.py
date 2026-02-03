from kg_sdk import KGClient

api = KGClient(base_url="http://192.168.2.101:5000")  # 知识库ip
NM_topo = api.get_data_attribute("NM_topo")[0]['preset_value']  # 网络拓扑

link_metric = api.get_data_attribute("NM_link_metrics")[0]['preset_value']# 链路状态
e2e_flow_data = api.get_data_attribute("E2E_flow_data") # e2e_flow_data


# 192.168.2.12 -> 192.168.2.30
# 10--->14 有问题
print(NM_topo)
print(link_metric)
# print(e2e_flow_data)