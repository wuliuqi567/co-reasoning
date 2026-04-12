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

from post_table_flow import send_flow_table, policy_compare
from post_II_info import get_II_info

def parse_args():
    parser = argparse.ArgumentParser("Double DQN for NetEnv.")
    parser.add_argument("--env-id", type=str, default="NetEnv-Net30-v0")
    parser.add_argument("--test", type=int, default=1)
    parser.add_argument("--benchmark", type=int, default=0)
    parser.add_argument("--src_dev_ip", type=str, default="192.168.10.2/24")
    parser.add_argument("--dst_dev_ip", type=str, default="192.168.40.2/24")

    return parser.parse_args()


if __name__ == "__main__":

    # OUTPUT_DIR = "/home/ict/projects/kg_network/semprotocol/log/access.log"  #这里需要先改成自己本地的一个路径，后续再替换成样机这边的路径
    OUTPUT_DIR = "./logs/access.log"  #这里需要先改成自己本地的一个路径，后续再替换成样机这边的路径
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
        """获取当前时间字符串 (年-月-日 时:分:秒:毫秒)"""
        now = datetime.now()
        return now.strftime("%Y-%m-%d %H:%M:%S") + f":{now.microsecond // 1000:03d}"
    
    # def _get_time_str_hms() -> str:
    #     """获取当前时间字符串 (时:分:秒)"""
    #     now = datetime.now()
    #     return now.strftime("%H:%M:%S")

    # ========== 阶段1-3: II类本地重路由（模拟） ==========
    time1 = _get_time_str()
    # II类运行本地重路由模型（当前为模拟，实际由II类智能体执行）
    local_delay = 0  # 本地重路由时延（ms），稍后替换为推理结果
    local_hop_num = 0  # 本地重路由跳数，稍后替换为推理结果
    local_bandwidth = 0  # 本地重路由带宽（MHz），稍后替换为推理结果
    local_response_time = 0  # 本地重路由响应时间（ms），稍后替换为推理结果

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
    Agent.load_model(path=Agent.model_dir_load, model="seed_1_2026_0312_213702")

    # ========== 阶段7: 推理生成全局重路由策略 ==========
    reroute_result = Agent.run_reroute(configs.test_episode, envs)
    time7 = _get_time_str()
    global_end_time = time.time()

    # 若过滤故障后 src 与 dst 不连通，则无最短路径
    is_connected_list = reroute_result.get("is_connected_src_dst", [True])
    if not all(is_connected_list):
        print("[警告] 当前图中 src 与 dst 不连通（过滤故障/拥塞后无可用路径），无法计算最短路径。")
        Agent.finish()
        exit()

    # 提取全局重路由结果
    print(f"paths: {reroute_result['paths'][0]}")

    # 输出路径节点信息
    print(f"path_ip_ports: {reroute_result['path_ip_ports'][0]}\n")

    # 输出路径性能指标
    print(f"path_delay: {reroute_result['path_delay'][0]}")
    print(f"path_bandwidth: {reroute_result['path_bandwidth'][0]}")
    print(f"path_loss_rate: {reroute_result['path_loss_rate'][0]}")

    print("\n-------------------------------最短路径信息（仅供参考）---------------------------------\n")
    print(f"shortest_paths: {reroute_result['shortest_paths'][0]}")
    print(f"shortest_path_ip_ports: {reroute_result['shortest_path_ip_ports'][0]}\n")
    print(f"shortest_path_delay: {reroute_result['shortest_path_delay'][0]}")
    print(f"shortest_path_bandwidth: {reroute_result['shortest_path_bandwidth'][0]}")
    print(f"shortest_path_loss_rate: {reroute_result['shortest_path_loss_rate'][0]}")
    print("\n---------------------------------------------------------------------------------------\n")
    
    # 路径跳数信息
    hop_num = len(reroute_result['paths'][0])
    shortest_path_hop_num = len(reroute_result['shortest_paths'][0]) if reroute_result['shortest_paths'][0] else 0

    # 最短路径版本
    # global_hop_num = shortest_path_hop_num  # 全局重路由跳数（最短路径）
    # global_delay = reroute_result['shortest_path_delay'][0]  # 全局重路由时延（ms）
    # global_bandwidth = reroute_result['shortest_path_bandwidth'][0]  # 全局重路由带宽（MHz）
    # global_response_time = int((global_end_time - global_start_time) * 1000)  # 全局重路由响应时间（ms）
    # global_policy = {"path":reroute_result['shortest_path_ip_ports'][0], "delay": global_delay, "bandwidth": global_bandwidth, "response_time": global_response_time}
    # local_policy = {"path":reroute_result['path_ip_ports'][0], "delay": local_delay, "bandwidth": local_bandwidth, "response_time": local_response_time}
    
    # rl模型版本
    global_hop_num = hop_num  # 全局重路由跳数（最短路径）
    global_delay = reroute_result['path_delay'][0]  # 全局重路由时延（ms）
    global_bandwidth = reroute_result['path_bandwidth'][0]  # 全局重路由带宽（MHz）
    global_response_time = int((global_end_time - global_start_time) * 1000)  # 全局重路由响应时间（ms）
    global_policy = {"path":reroute_result['path_ip_ports'][0], "delay": global_delay, "bandwidth": global_bandwidth, "response_time": global_response_time}

    print("-------------------------------全局重路由策略---------------------------------\n")
    print("global_policy", global_policy)
    print("\n")

    local_policy = get_II_info("co_reasoning_II_1")
    print("-------------------------------II policy---------------------------------\n")
    print("local_policy", local_policy)
    print("\n")

    final_policy = policy_compare(global_policy, local_policy)
    if final_policy is None:
        print("[WARN] policy_compare 返回 None，使用全局策略")
        final_policy = global_policy
    post_table_flow = final_policy.get('path', [])
    # ========== 阶段8: 协同优化机制 ==========
    time8 = _get_time_str()
    
    # 获取本地重路由策略并执行协同优化机制

    # ========== 阶段9: 下发全局重路由策略 ==========
    time9 = _get_time_str()
    # 下发全局重路由策略,并交由II类智能体执行

    # send_flow_table(post_table_flow, timeout=10.0, retries=2, verbose=True)

    # ========== 阶段10: 协同推理结束 ==========
    time10 = _get_time_str()
    # 协同推理结束

    # ========== 构建性能指标结果 ==========
    # result1: 时延（ms），[本地重路由, 全局重路由]
    # result2: 跳数，[本地重路由, 全局重路由]
    # result3: 可用带宽（MHz），[本地重路由, 全局重路由]
    # result4: 响应时间（ms），[本地重路由, 全局重路由]
    result1 = f"[{local_policy.get('delay', 'N/A')} {global_policy.get('delay', 'N/A')}"
    local_path = local_policy.get('path', [])
    global_path = global_policy.get('path', [])
    result2 = f"{len(local_path) if isinstance(local_path, list) else '0'} {len(global_path) if isinstance(global_path, list) else '0'}"
    result3 = f"{local_policy.get('bandwidth', '0')} {global_policy.get('bandwidth', '0')}"
    result4 = f"{local_policy.get('response_time', '0')} {global_policy.get('response_time', '0')}]"
    result = result1 + " " + result2 + " " + result3 + " " + result4
    print(f"性能指标结果: {result}")

    status = "1"  # 协同推理状态：1=成功
    cor_node = "II_node_2 III_node_1"  # 协同节点名称


    # ========== 定义10个阶段的内容描述 ==========
    content1 = "II类运行本地重路由模型"
    content2 = "生成本地重路由策略，并下发给II类智能体执行"
    content3 = "将本地重路由策略更新到本地知识单元，并通过上报给III类知识单元"
    content4 = "III类检测到某II类节点/链路失效，III类触发协同推理功能"
    content5 = "查询网络状态知识"
    content6 = "运行全局重路由模型"
    content7 = "推理生成全局重路由策略"
    content8 = "获取本地重路由策略并执行协同优化机制"
    content9 = "下发全局重路由策略,并交由II类智能体执行"
    content10 = "协同推理结束"

    local_policy_log = local_policy["log_II_info"] if "log_II_info" in local_policy else "N/A"
    if local_policy_log == "N/A":
        print("[警告] local_policy 中缺少 'log_II_info' 字段，使用默认日志格式")
        local_policy_log = (
            f"time1={time1}, content1={content1}; "
            f"time2={time2}, content2={content2}; "
            f"time3={time3}, content3={content3}"
        )

    local_policy_log = str(local_policy_log).strip()
    log_prefix = f"[collaborative_reasoning] {local_policy_log}"

    full_log = (
        f"{log_prefix}; "
        f"time4={time4}, content4={content4}; "
        f"time5={time5}, content5={content5}; "
        f"time6={time6}, content6={content6}; "
        f"time7={time7}, content7={content7}; "
        f"time8={time8}, content8={content8}; "
        f"time9={time9}, content9={content9}; "
        f"time10={time10}, content10={content10}; "
        f"result={result}; status={status}; cor_node={cor_node}"
    )

    # ========== 输出协同推理日志 ==========
    logging.info("%s", full_log)
    # logging.info(
    #     "[collaborative_reasoning] "
    #     "time1=%s, content1=%s; time2=%s, content2=%s; time3=%s, content3=%s; "
    #     "time4=%s, content4=%s; time5=%s, content5=%s; time6=%s, content6=%s; "
    #     "time7=%s, content7=%s; time8=%s, content8=%s; time9=%s, content9=%s; "
    #     "time10=%s, content10=%s; result=%s; "
    #     "status=%s; cor_node=%s",
    #     time1, content1, time2, content2, time3, content3,
    #     time4, content4, time5, content5, time6, content6,
    #     time7, content7, time8, content8, time9, content9,
    #     time10, content10, result,
    #     status, cor_node
    # )

    Agent.finish()
