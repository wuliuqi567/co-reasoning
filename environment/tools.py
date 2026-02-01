"""
通用辅助函数。
"""

from typing import Any, Dict


def _coerce_float(value: Any, default: float = 0.0) -> float:
    """将输入值转换为浮点数。"""
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_int(value: Any, default: int = 0) -> int:
    """将输入值转换为整数。"""
    if value is None or value == "":
        return int(default)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _get_edge_latency(edge_data: Dict[str, Any]) -> float:
    """从边属性获取链路时延 (优先 link_latency, 其次 delay)。"""
    return _coerce_float(edge_data.get("link_latency", edge_data.get("delay", 0.0))) 


def _get_edge_bandwidth(edge_data: Dict[str, Any]) -> float:
    """从边属性获取链路带宽 (优先 link_bandwidth, 其次 bandwidth)。"""
    return _coerce_float(edge_data.get("link_bandwidth", edge_data.get("bandwidth", 0.0)))


def _get_edge_utilization(edge_data: Dict[str, Any]) -> float:
    """从边属性获取链路利用率。"""
    return _coerce_float(edge_data.get("link_utilization", 0.0))


def _get_edge_loss_rate(edge_data: Dict[str, Any]) -> float:
    """从边属性获取链路丢包率。"""
    return _coerce_float(edge_data.get("link_loss_rate", 0.0))


def _is_failed_status(value: Any) -> bool:
    """判断状态值是否为故障标记 (0 表示故障，1 表示在线)。"""
    try:
        return int(float(value)) == 0
    except (TypeError, ValueError):
        return False
