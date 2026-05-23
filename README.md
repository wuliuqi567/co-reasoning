# Co-Reasoning Routing

本项目用于网络拓扑解析、II/III 类重路由计算，以及协同推理日志组织。`inner_*` 面向内场联调；`rl_reroute*.py` 面向外场 DDQN 模型流程。

## 目录

```text
inner_rl_reroute.py                 # 内场协同编排入口，默认在线知识库模式
inner-rl-reroute_II.py              # 内场 II 类本地路径：按时延最短路径
inner-rl-reroute_III.py             # 内场 III 类全局路径：QoS 约束路由
inner_post_II_info.py               # 内场专用知识库接口，目标 192.168.1.24
inner_post_table_flow.py            # 内场专用流表接口，目标 192.168.1.24
environment/inner_graph_data/       # 内场拓扑、指标、GraphML 与解析工具
rl_reroute.py                       # 外场 III 类 DDQN 流程
rl_reroute_II.py                    # 外场 II 类 DDQN 流程
```

## 数据与状态

内场流程默认使用：

```text
environment/inner_graph_data/json-data/network_topology_state.json
environment/inner_graph_data/json-data/link_metric.json
environment/inner_graph_data/base_ii_topology.graphml
```

状态约定：

```text
node_status = 0  表示离线/故障
link_status = 0  表示离线/故障
```

`link_metric.json` 通过 `link_id` 更新链路指标，包括利用率、可用带宽、丢包率等。如果指标文件中的 `link_id` 与拓扑不匹配，脚本会保留拓扑中的原始带宽，丢包率按默认值计算。

## 推荐运行

### 协同推理入口

默认在线模式，会调用真实知识库接口，并将日志写入在线路径：

```bash
python inner_rl_reroute.py
```

在线默认日志：

```text
/home/ict/projects/kg_network/semprotocol/log/access.log
```

离线调试时使用本地 JSON 模拟知识库：

```bash
python inner_rl_reroute.py --offline
```

离线输出：

```text
logs/access.log
logs/offline_ii_policy.json
```

默认源宿节点：

```text
asu0n0 -> eru1n5
```

指定源宿节点：

```bash
python inner_rl_reroute.py --src asu0n0 --dst eru1n5
```

指定源宿 IP：

```bash
python inner_rl_reroute.py --src-ip 10.104.0.254 --dst-ip 10.103.21.254
```

在线获取最新拓扑和链路指标：

```bash
python inner_rl_reroute.py --fetch-online --fetch-link-metrics
```

如需手动指定日志路径：

```bash
python inner_rl_reroute.py --log-path /tmp/access.log
```

### 单独运行 II 路由

II 脚本只做在线图过滤后的时延最短路径：

```bash
python inner-rl-reroute_II.py
```

常用参数：

```bash
python inner-rl-reroute_II.py --src asu0n0 --dst eru1n5
python inner-rl-reroute_II.py --src-ip 10.104.0.254 --dst-ip 10.103.21.254
python inner-rl-reroute_II.py --fetch-online --fetch-link-metrics
```

### 单独运行 III 路由

III 脚本执行 QoS 路由：先按可用带宽过滤链路，再生成按时延排序的候选路径，返回第一条满足丢包率约束的路径。

```bash
python inner-rl-reroute_III.py
```

QoS 默认值：

```text
最小瓶颈可用带宽：100 Mbps
最大端到端丢包率：0.01
最多检查候选路径数：20
```

常用参数：

```bash
python inner-rl-reroute_III.py --src asu0n0 --dst eru1n5
python inner-rl-reroute_III.py --src-ip 10.104.0.254 --dst-ip 10.103.21.254
python inner-rl-reroute_III.py --fetch-online --fetch-link-metrics
```

## 内场协同流程

`inner_rl_reroute.py` 的执行顺序：

1. 运行 `inner-rl-reroute_II.py`，得到 II 本地路径。
2. 将 II 策略上报知识库；离线模式写入 `logs/offline_ii_policy.json`。
3. 运行 `inner-rl-reroute_III.py`，得到 III QoS 路径。
4. 从知识库读取 II 策略。
5. 使用 `inner_post_table_flow.policy_compare()` 比较 II/III 策略。
6. 参考 `rl_reroute.py` 的格式写协同推理日志。

当前 `inner_rl_reroute.py` 不实际下发流表。流表下发函数在 `inner_post_table_flow.py` 中，默认目标为：

```text
http://192.168.1.24:12590/api/flow/sflowtblCfg
```

知识库接口在 `inner_post_II_info.py` 中，默认目标为：

```text
http://192.168.1.24:5001
```

## 拓扑与指标工具

抓取拓扑：

```bash
python environment/inner_graph_data/get-topo-data.py
```

抓取链路指标：

```bash
python environment/inner_graph_data/get-link-metric-data.py
```

从 JSON 构建或更新 NetworkX 图的核心逻辑在：

```text
environment/inner_graph_data/topology_to_networkx.py
environment/inner_graph_data/qos_routing.py
```

## 外场 DDQN 流程

外场流程使用训练好的 DDQN 模型做推理，入口为：

```bash
python rl_reroute_II.py
python rl_reroute.py
```

建议执行顺序：

1. 先运行 `rl_reroute_II.py`，执行 II 类本地 DDQN 重路由。
2. `rl_reroute_II.py` 会加载 II 类模型，计算路径，并通过 `post_II_info.py` 将本地策略上报到外场知识库。
3. 再运行 `rl_reroute.py`，执行 III 类全局 DDQN 重路由。
4. `rl_reroute.py` 会从知识库读取 II 类策略，使用 `post_table_flow.policy_compare()` 比较 II/III 策略，生成最终策略。
5. 最终协同推理过程会写入 `logs/access.log`。当前外场脚本中的流表下发调用保持注释状态，如需真实下发，需要打开 `send_flow_table(...)`。

常用运行方式：

```bash
python rl_reroute_II.py --src_dev_ip 192.168.10.2/24 --dst_dev_ip 192.168.40.2/24
python rl_reroute.py --src_dev_ip 192.168.10.2/24 --dst_dev_ip 192.168.40.2/24
```

默认源宿业务 IP：

```text
src_dev_ip = 192.168.10.2/24
dst_dev_ip = 192.168.40.2/24
```

外场配置与模型：

```text
II 类配置：config/ex_ddqn_II.yaml
II 类模型：themodels/latest_II_class_ddqn_ii/seed_1_2026_0203_104026

III 类配置：config/ex_ddqn.yaml
III 类模型：themodels/latest_II_class_ddqn_iii/seed_1_2026_0312_213702

环境类：environment/net_tupu_iii.py
拓扑来源：graph_source = latest_II_class_base
在线接口：base_url = http://192.168.2.101:5000
```

外场知识库与流表接口沿用原有外场脚本：

```text
post_II_info.py      # II 策略上报/读取，默认目标 192.168.2.11:5001
post_table_flow.py   # 流表下发与策略比较，默认目标 192.168.2.26:12590
```

输出信息包括：

```text
DDQN 路径、路径端口/IP、时延、带宽、丢包率
最短路径参考结果
II/III 策略比较后的最终策略
协同推理阶段日志
```

注意：外场 `rl_reroute*.py` 依赖 Xuance、训练配置、模型文件和外场在线拓扑/知识库服务；内场 `inner_*` 流程主要用于内场路由与协同联调。两套流程不要混用接口脚本。
