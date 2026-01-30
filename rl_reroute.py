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
import os

def parse_args():
    parser = argparse.ArgumentParser("Double DQN for NetEnv.")
    parser.add_argument("--env-id", type=str, default="NetEnv-Net30-v0")
    parser.add_argument("--test", type=int, default=1)
    parser.add_argument("--benchmark", type=int, default=0)
    parser.add_argument("--src", type=int, default=10)
    parser.add_argument("--dst", type=int, default=14)

    return parser.parse_args()

if __name__ == "__main__":

    OUTPUT_DIR = "./log/access.log"  #这里需要先改成自己本地的一个路径，后续再替换成样机这边的路径
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
    # 即使文件处理器创建失败，我们仍然可以使用控制台处理器

    # 配置日志基础设置
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=_log_handlers,
    )
    
    time1 = time.time()
    content1 = "检测到某II类节点II_node_3失效，触发协同推理功能"

    config_path = Path(__file__).resolve().parent / "config" / "ex_ddqn.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    parser = parse_args()
    configs_dict = get_configs(str(config_path))
    configs_dict = recursive_dict_update(configs_dict, parser.__dict__)

    configs = argparse.Namespace(**configs_dict)
    REGISTRY_ENV[configs.env_name] = NetTupu

    time2 = time.time()
    content2 = "查询网络状态知识（网络拓扑知识、节点资源知识等）"
    # NetTupu.load_net_tupo_info(content2) # 这里要实现调用知识库接口解析图然后加载图
    # 获取当前网络中的流量信息, 得到流的src，dst，然后调用全局重路由模型

    print(f"src: {configs.src}, dst: {configs.dst}")

    configs.logger = "tensorboard"
    configs.test_episode = 1
    configs.parallels = 1
    configs.vectorize = "DummyVecEnv"
    configs.execute_reroute = True

    set_seed(configs.seed)
    envs = make_envs(configs)
    Agent = MyDDQNAgent(config=configs, envs=envs)

    time3 = time.time()
    content3 = "运行全局重路由模型"
    # seed_1_2026_0128_172741 
    Agent.load_model(path=Agent.model_dir_load, model="seed_1_2026_0130_103220")

    
    paths, path_ip_ports, shortest_path_ip_ports = Agent.run_reroute(configs.test_episode, envs)
    time4 = time.time()
    content4 = "推理生成全局重路由策略"
    print(f"paths: {paths}")
    print(f"path_ip_ports: {path_ip_ports}")
    print(f"shortest_path_ip_ports: {shortest_path_ip_ports}")

    result = [] # 性能指标，分别是本地重路由和全局重路由的时延（ms）、本地里路由和全局重路由的跳数、本地重路由和全局重路由的可用带宽（MHz）、本
#地重路由和全局重路由的响应时间（ms）

    time5 = time.time()
    content5 = "获取本地重路由策略并执行协同优化机制，全局重路由策略评分XX，本地重路由策略评分XX"


    time6 = time.time()
    content6 = "下发全局重路由策略"

    time7 = time.time()
    content7 = "协同推理结束"

    status = "1"
    cor_node = "II_node_2 III_node_1"

    Agent.finish()
