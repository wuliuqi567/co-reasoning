from kg_sdk import KGClient
import time
from datetime import datetime

api = KGClient(base_url="http://192.168.2.101:5000")  # 知识库ip
NM_topo = api.get_data_attribute("NM_topo")[0]['preset_value']  # 网络拓扑

link_metric = api.get_data_attribute("NM_link_metrics")[0]['preset_value']# 链路状态
e2e_flow_data = api.get_data_attribute("E2E_flow_data") # e2e_flow_data


def _get_time_str() -> str:
    """获取当前时间字符串 (时:分:秒:毫秒:微秒)"""
    now = datetime.now()
    # %f provides 6 digits (microseconds). 
    # To get HH:MM:SS:ms:us, we slice the %f part.
    return now.strftime("%H:%M:%S") + f":{now.microsecond // 1000:03d}:{now.microsecond % 1000:03d}"


# 192.168.2.12 -> 192.168.2.30
# 10--->14 有问题
print(NM_topo)
print(link_metric)
# print(e2e_flow_data)
# 执行100次，获取网络拓扑信息，存储在jsondata/data_topo_link_info目录下，文件名称为topo_II_class_time_str.json和link_II_class_time_str.json
count = 0
for i in range(100):
    
    NM_topo = api.get_data_attribute("NM_topo")[0]['preset_value']  # 网络拓扑
    link_metric = api.get_data_attribute("NM_link_metrics")[0]['preset_value']# 链路状态
    count += 1
    time_str = _get_time_str()
    topo_file_name = f"topo_II_class_{time_str}.json"
    link_file_name = f"link_II_class_{time_str}.json"

    with open(f"../jsondata/data_topo_link_info/{topo_file_name}", "w") as f:
        f.write(NM_topo)
    with open(f"../jsondata/data_topo_link_info/{link_file_name}", "w") as f:
        f.write(link_metric)
    print("count: ", count)
    time.sleep(5)
print("done")