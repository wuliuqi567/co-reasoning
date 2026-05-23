"""Inner routing knowledge-base helpers.

This module is dedicated to inner_rl_reroute.py. It keeps the original
post_II_info.py unchanged and points the inner flow to the 192.168.1.24 host.
"""

from __future__ import annotations

import datetime
import json
from typing import Any

from kg_sdk import KGClient


INNER_KG_BASE_URL = "http://192.168.1.24:5001"
INNER_II_POLICY_NAME = "co_reasoning_II_1_policy"

api_II = KGClient(base_url=INNER_KG_BASE_URL)


logic_knowledge = {
    "name": INNER_II_POLICY_NAME,
    "domain": "网络状态域",
    "input_data": [""],
    "output_data": [""],
    "algorithm_filename": "inner_co_reasoning",
    "title": "inner II类协同推理策略",
    "is_preset": "1",
    "output_decision": "path",
    "meaning": "inner II类本地重路由策略上报给III类知识单元",
    "source": "认知知识",
    "type": "逻辑决策型",
    "scenario": "智能路由",
    "update_time": str(datetime.date.today()),
    "score": "",
}


def post_II_info(local_policy: dict[str, Any]) -> None:
    print(f"-----inner II类知识存储与上报开始: {INNER_KG_BASE_URL}-----")
    payload = dict(logic_knowledge)
    payload["output_decision"] = json.dumps(local_policy, ensure_ascii=False)
    payload["update_time"] = str(datetime.date.today())

    result = api_II.create_logical_decision_model(payload)
    print("inner II类属性知识创建逻辑决策型结果:", result)

    result = api_II.add_relational_calc_relation(
        payload.get("name"),
        "co_reasoning_II",
    )
    print("inner II类知识创建关系结果:", result)


def get_II_info(name: str = INNER_II_POLICY_NAME) -> dict[str, Any]:
    result = api_II.get_logical_decision_model(name)
    return json.loads(result[0]["output_decision"])
