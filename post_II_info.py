
import datetime
import time
import json
from typing import Dict, Any
from kg_sdk import KGClient

api_II = KGClient(base_url="http://192.168.2.11:5001")


logic_knowledge = {
    "name": "co_reasoning_II",
    "domain": "网络状态域",
    "input_data": [""],
    "output_data": [""],
    "algorithm_filename": "co_reasoning",
    "title": "II类协同推理策略",
    "is_preset": "1",
    "output_decision": "path", # 将策略结果转为str，替换“path”
    "meaning": "II类本地重路由策略上报给III类知识单元",
    "source": "认知知识",
    "type": "逻辑决策型",
    "scenario": "智能路由",
    "update_time": str(datetime.date.today()),
    "score": "",
}
unique_id = int(time.time())  # 生成唯一ID（例如时间戳）


# II类存储与上报
def post_II_info(local_policy: Dict[str, Any]):
    print("-----II类知识存储与上报测试开始-----")
    # 将local_policy转为json字符串
    local_policy_json = json.dumps(local_policy)
    logic_knowledge['output_decision'] = local_policy_json

    result = api_II.create_logical_decision_model(logic_knowledge)
    print("II类属性知识创建逻辑决策型结果:", result)

    result = api_II.add_relational_calc_relation(
        logic_knowledge.get("name"), "co_reasoning"
    )

    print("II类知识创建关系结果:", result)

# 获取 
def get_II_info(name: str):
    result = api_II.get_logical_decision_model(name)

    local_policy = json.loads(result[0]['output_decision'])
    return local_policy