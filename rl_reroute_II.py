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
from datetime import datetime

from .post_table_flow import send_flow_table, policy_compare
from .post_II_info import post_II_info, get_II_info

def parse_args():
    parser = argparse.ArgumentParser("Double DQN for NetEnv.")
    parser.add_argument("--env-id", type=str, default="NetEnv-Net30-v0")
    parser.add_argument("--test", type=int, default=1)
    parser.add_argument("--benchmark", type=int, default=0)
    parser.add_argument("--src_dev_ip", type=str, default="192.168.10.2/24")
    parser.add_argument("--dst_dev_ip", type=str, default="192.168.40.2/24")

    return parser.parse_args()


if __name__ == "__main__":


    def _get_time_str() -> str:
        """获取当前时间字符串 (时:分:秒:毫秒:微秒)"""
        now = datetime.now()
        # %f provides 6 digits (microseconds). 
        # To get HH:MM:SS:ms:us, we slice the %f part.
        return now.strftime("%H:%M:%S") + f":{now.microsecond // 1000:03d}:{now.microsecond % 1000:03d}"

    # ========== 定义10个阶段的内容描述 ==========
    content1 = "II类运行本地重路由模型"
    content2 = "生成本地重路由策略，并下发给II类智能体执行"
    content3 = "将本地重路由策略更新到本地知识单元，并通过上报给III类知识单元"

    time1 = _get_time_str()
    # 开始本地重路由

    config_path = Path(__file__).resolve().parent / "config" / "ex_ddqn_II.yaml"
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
    Agent.load_model(path=Agent.model_dir_load, model="seed_1_2026_0130_103220")
    reroute_result = Agent.run_reroute(configs.test_episode, envs)
    
    time2 = _get_time_str()
    # 生成本地重路由策略，并下发给II类智能体执行
    local_response_time = int(time2 - time1)

    # 若过滤故障后 src 与 dst 不连通，则无最短路径
    is_connected_list = reroute_result.get("is_connected_src_dst", [True])
    if not all(is_connected_list):
        print("[警告] 当前图中 src 与 dst 不连通（过滤故障/拥塞后无可用路径），无法计算最短路径。")

    # 提取全局重路由结果
    print(f"paths: {reroute_result['paths'][0]}")
    print(f"shortest_paths: {reroute_result['shortest_paths'][0]}")
    local_hop_num = len(reroute_result['paths'][0])
    shortest_path_hop_num = len(reroute_result['shortest_paths'][0]) if reroute_result['shortest_paths'][0] else 0
    print(f"path_ip_ports: {reroute_result['path_ip_ports'][0]}\n")
    print(f"shortest_path_ip_ports: {reroute_result['shortest_path_ip_ports'][0]}\n")
    print(f"path_delay: {reroute_result['path_delay'][0]}")
    print(f"path_bandwidth: {reroute_result['path_bandwidth'][0]}")
    print(f"path_loss_rate: {reroute_result['path_loss_rate'][0]}")
    print(f"shortest_path_delay: {reroute_result['shortest_path_delay'][0]}")
    print(f"shortest_path_bandwidth: {reroute_result['shortest_path_bandwidth'][0]}")
    print(f"shortest_path_loss_rate: {reroute_result['shortest_path_loss_rate'][0]}")

    
    local_policy = {"path":reroute_result['path_ip_ports'][0], "delay": reroute_result['path_delay'][0], "bandwidth": reroute_result['path_bandwidth'][0], "response_time": local_response_time}
    time3 = _get_time_str()
    # 将本地重路由策略更新到本地知识单元，并通过上报给III类知识单元

    post_II_info(local_policy)

    Agent.finish()
