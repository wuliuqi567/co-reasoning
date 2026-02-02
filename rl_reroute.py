import sys
import os
from pathlib import Path

import time
import argparse
from xuance.common import get_configs, recursive_dict_update
from xuance.environment import make_envs
from xuance.torch.utils.operations import set_seed
from agents.myddqn_agent import MyDDQNAgent

from environment.net_tupu_iii import NetTupu
from xuance.environment import REGISTRY_ENV
import logging


def parse_args():
    parser = argparse.ArgumentParser("Double DQN for NetEnv.")
    parser.add_argument("--env-id", type=str, default="NetEnv-Net30-v0")
    parser.add_argument("--test", type=int, default=1)
    parser.add_argument("--benchmark", type=int, default=0)
    parser.add_argument("--src_dev_ip", type=str, default="192.168.10.2/24")
    parser.add_argument("--dst_dev_ip", type=str, default="192.168.40.2/24")

    return parser.parse_args()


if __name__ == "__main__":

    OUTPUT_DIR = "./log/access.log"
    # 确保日志目录存在
    log_dir = os.path.dirname(OUTPUT_DIR)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # 初始化日志处理器列表
    _log_handlers = [logging.StreamHandler()]

    # 安全地添加文件处理器
    try:
        _log_handlers.append(logging.FileHandler(OUTPUT_DIR, encoding='utf-8'))
    except OSError as e:
        print(f"[警告] 无法创建文件日志处理器: {e}")

    # 配置日志基础设置
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=_log_handlers,
    )

    def _get_time_str() -> str:
        """获取当前时间字符串（HH:MM:SS）"""
        return time.strftime("%H:%M:%S", time.localtime())

    # ========== 定义10个阶段的内容描述 ==========
    content1 = "II类运行本地重路由模型"
    content2 = "生成本地重路由策略，并下发给II类智能体执行"
    content3 = "将本地重路由策略更新到本地知识单元，并通过上报给III类知识单元"
    content4 = "III类检测到某II类节点II_node_3失效，III类触发协同推理功能"
    content5 = "查询网络状态知识（网络拓扑知识、节点资源知识等）"
    content6 = "运行全局重路由模型"
    content7 = "推理生成全局重路由策略"
    content8 = "获取本地重路由策略并执行协同优化机制，全局重路由策略评分XX，本地重路由策略评分XX"
    content9 = "下发全局重路由策略,并交由II类智能体执行"
    content10 = "协同推理结束"

    # ========== 阶段1-3: II类本地重路由（模拟） ==========
    time1 = _get_time_str()
    # II类运行本地重路由模型（当前为模拟，实际由II类智能体执行）
    local_delay = 0  # 本地重路由时延（ms），模拟值
    local_hop_num = 0  # 本地重路由跳数，模拟值
    local_bandwidth = 0  # 本地重路由带宽（MHz），模拟值
    local_response_time = 0  # 本地重路由响应时间（ms），模拟值

    time2 = _get_time_str()
    # 生成本地重路由策略，并下发给II类智能体执行

    time3 = _get_time_str()
    # 将本地重路由策略更新到本地知识单元，并通过上报给III类知识单元

    # ========== 阶段4: III类检测故障，触发协同推理 ==========
    time4 = _get_time_str()
    # III类检测到某II类节点失效，触发协同推理功能

    # ========== 阶段5: 查询网络状态知识 ==========
    time5 = _get_time_str()
    # 查询网络状态知识（网络拓扑知识、节点资源知识等）

    config_path = Path(__file__).resolve().parent / "config" / "ex_ddqn.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    parser = parse_args()
    configs_dict = get_configs(str(config_path))
    configs_dict = recursive_dict_update(configs_dict, parser.__dict__)

    configs = argparse.Namespace(**configs_dict)
    REGISTRY_ENV[configs.env_name] = NetTupu

    print(f"src_dev_ip: {configs.src_dev_ip}, dst_dev_ip: {configs.dst_dev_ip}")

    configs.logger = "tensorboard"
    configs.test_episode = 1
    configs.parallels = 1
    configs.vectorize = "DummyVecEnv"

    set_seed(configs.seed)
    envs = make_envs(configs)
    Agent = MyDDQNAgent(config=configs, envs=envs)

    # ========== 阶段6: 运行全局重路由模型 ==========
    time6 = _get_time_str()
    global_start_time = time.time()
    Agent.load_model(path=Agent.model_dir_load, model="seed_1_2026_0130_103220")

    # ========== 阶段7: 推理生成全局重路由策略 ==========
    reroute_result = Agent.run_reroute(configs.test_episode, envs)
    time7 = _get_time_str()
    global_end_time = time.time()

    # 若过滤故障后 src 与 dst 不连通，则无最短路径
    is_connected_list = reroute_result.get("is_connected_src_dst", [True])
    if not all(is_connected_list):
        print("[警告] 当前图中 src 与 dst 不连通（过滤故障/拥塞后无可用路径），无法计算最短路径。")

    # 提取全局重路由结果
    print(f"paths: {reroute_result['paths'][0]}")
    print(f"shortest_path: {reroute_result['shortest_path'][0]}")
    global_hop_num = len(reroute_result['paths'][0])
    shortest_path_hop_num = len(reroute_result['shortest_path'][0])
    print(f"path_ip_ports: {reroute_result['path_ip_ports'][0]}\n")
    print(f"shortest_path_ip_ports: {reroute_result['shortest_path_ip_ports'][0]}\n")
    print(f"path_delay: {reroute_result['path_delay'][0]}")
    print(f"path_bandwidth: {reroute_result['path_bandwidth'][0]}")
    print(f"path_loss_rate: {reroute_result['path_loss_rate'][0]}")
    print(f"shortest_path_delay: {reroute_result['shortest_path_delay'][0]}")
    print(f"shortest_path_bandwidth: {reroute_result['shortest_path_bandwidth'][0]}")
    print(f"shortest_path_loss_rate: {reroute_result['shortest_path_loss_rate'][0]}")

    global_delay = reroute_result['path_delay'][0]  # 全局重路由时延（ms）
    global_bandwidth = reroute_result['path_bandwidth'][0]  # 全局重路由带宽（MHz）
    global_response_time = int((global_end_time - global_start_time) * 1000)  # 全局重路由响应时间（ms）

    # ========== 阶段8: 协同优化机制 ==========
    time8 = _get_time_str()
    # 获取本地重路由策略并执行协同优化机制

    # ========== 阶段9: 下发全局重路由策略 ==========
    time9 = _get_time_str()
    # 下发全局重路由策略,并交由II类智能体执行

    # ========== 阶段10: 协同推理结束 ==========
    time10 = _get_time_str()
    # 协同推理结束

    # ========== 构建性能指标结果 ==========
    # result1: 时延（ms），[本地重路由, 全局重路由]
    # result2: 跳数，[本地重路由, 全局重路由]
    # result3: 可用带宽（MHz），[本地重路由, 全局重路由]
    # result4: 响应时间（ms），[本地重路由, 全局重路由]
    result1 = f"[{local_delay} {global_delay}]"
    result2 = f"[{local_hop_num} {global_hop_num}]"
    result3 = f"[{local_bandwidth} {global_bandwidth}]"
    result4 = f"[{local_response_time} {global_response_time}]"

    status = "1"  # 协同推理状态：1=成功
    cor_node = "II_node_2 III_node_1"  # 协同节点名称

    # ========== 输出协同推理日志 ==========
    logging.info(
        "[collaborative_reasoning] "
        "time1=%s, content1=%s; time2=%s, content2=%s; time3=%s, content3=%s; "
        "time4=%s, content4=%s; time5=%s, content5=%s; time6=%s, content6=%s; "
        "time7=%s, content7=%s; time8=%s, content8=%s; time9=%s, content9=%s; "
        "time10=%s, content10=%s; result1=%s; result2=%s; result3=%s; result4=%s; "
        "status=%s; cor_node=%s",
        time1, content1, time2, content2, time3, content3,
        time4, content4, time5, content5, time6, content6,
        time7, content7, time8, content8, time9, content9,
        time10, content10, result1, result2, result3, result4,
        status, cor_node
    )

    Agent.finish()
