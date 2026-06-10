"""Inner routing knowledge-base helpers.

This module is dedicated to inner_rl_reroute.py. It keeps the original
post_II_info.py unchanged and points the inner flow to the 192.168.1.24 host.
"""
import datetime
import importlib.util
import json
from pathlib import Path
from typing import Any

from kg_sdk import KGClient

INNER_KG_BASE_URL = "http://192.168.1.24:5001"
INNER_II_POLICY_NAME = "co_reasoning_II_1_policy"
ROOT = Path(__file__).resolve().parent
INNER_GRAPH_DATA_DIR = ROOT / "environment" / "inner_graph_data"

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
        payload.get("name"), "co_reasoning_II",
    )
    print("inner II类知识创建关系结果:", result)


def get_II_info(name: str = INNER_II_POLICY_NAME) -> dict[str, Any]:
    result = api_II.get_logical_decision_model(name)
    return json.loads(result[0]["output_decision"])


def _load_inner_fetcher(filename: str, module_name: str):
    module_path = INNER_GRAPH_DATA_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载在线数据获取脚本: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fetch_online_topology(timeout: float = 10.0, retries: int = 2) -> dict[str, Any]:
    module = _load_inner_fetcher("get-topo-data.py", "get_topo_data")
    data = module.fetch_network_state(module.DEFAULT_URL, timeout=timeout, retries=retries)
    module.validate_api_result(data)
    return data


def fetch_online_link_metrics(timeout: float = 10.0, retries: int = 2) -> dict[str, Any]:
    module = _load_inner_fetcher("get-link-metric-data.py", "get_link_metric_data")
    data = module.fetch_link_metrics(module.DEFAULT_URL, timeout=timeout, retries=retries)
    module.validate_api_result(data)
    return data


def fetch_online_business(timeout: float = 10.0, retries: int = 2) -> dict[str, Any]:
    module = _load_inner_fetcher("get-task-data.py", "get_task_data")
    data = module.fetch_task_data(module.DEFAULT_URL, timeout=timeout, retries=retries)
    module.validate_api_result(data)
    return data


def create_json_data_attribute(
    *,
    name: str,
    title: str,
    meaning: str,
    data: dict[str, Any],
) -> None:
    payload = {
        "meaning": meaning,
        "data_type": "json",
        "unit": "",
        "domain": "网络状态域",
        "is_preset": True,
        "preset_value": json.dumps(data, ensure_ascii=False),
        "value_range": "",
        "dimension": "1",
        "source": "认知知识",
        "name": name,
        "title": title,
        "update_time": str(datetime.date.today()),
        "scenario": "智能路由",
        "type": "数据信息型",
    }

    result = api_II.create_data_attribute(payload)
    print(f"{title} 创建结果:", result)


def post_topo(topo_info: dict[str, Any] | None = None) -> None:
    if topo_info is None:
        topo_info = fetch_online_topology()
    create_json_data_attribute(
        name="NM_topo",
        title="管控上报的全网网络拓扑信息",
        meaning="管控上报的全网网络拓扑信息",
        data=topo_info,
    )


def post_link_metrics(link_metrics: dict[str, Any] | None = None) -> None:
    if link_metrics is None:
        link_metrics = fetch_online_link_metrics()
    create_json_data_attribute(
        name="NM_link_metrics",
        title="管控上报网络链路状态信息",
        meaning="管控上报网络链路状态信息",
        data=link_metrics,
    )


def post_business(business_info: dict[str, Any] | None = None) -> None:
    if business_info is None:
        business_info = fetch_online_business()
    create_json_data_attribute(
        name="E2E_flow_data",
        title="管控上报业务信息",
        meaning="管控上报业务信息",
        data=business_info,
    )


def post_NM(
    topo_info: dict[str, Any] | None = None,
    link_metrics: dict[str, Any] | None = None,
    business_info: dict[str, Any] | None = None,
) -> None:
    if topo_info is None:
        topo_info = fetch_online_topology()
    if link_metrics is None:
        link_metrics = fetch_online_link_metrics()
    if business_info is None:
        business_info = fetch_online_business()

    post_topo(topo_info)
    post_link_metrics(link_metrics)
    post_business(business_info)


if __name__ == "__main__":
    post_NM()
