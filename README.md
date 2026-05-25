# Co-Reasoning Routing

本项目包含两套流程：

- `inner_*`：内场自动检测、II/III 重路由、协同日志组织。
- `rl_reroute*.py`：外场 DDQN 路由推理流程。

当前内场推荐主入口是 `auto_inner_reroute.py`。

## 一、默认在线运行流程

本节面向正常运行人员，默认知识库、拓扑接口、链路指标接口都在线。

### 1. 激活运行环境

部署机器上先进入项目目录，并激活 conda 环境：

```bash
cd /path/to/co-reasoning
conda activate co-reasoning
```

后续前台命令默认都在 `co-reasoning` 环境中执行。

### 2. 生成业务流

首次运行或需要重新生成业务时执行：

```bash
python generate_inner_business_flows.py
```

默认生成 20 条源目的业务，并保存到：

```text
environment/inner_graph_data/json-data/inner_business_flows.json
```

如果业务文件已经存在，可以直接进入下一步。

如需指定业务数量：

```bash
python generate_inner_business_flows.py --count 50
```

### 3. 启动自动检测与重路由

前台运行：

```bash
python auto_inner_reroute.py
```

后台运行，关闭终端后继续执行：

```bash
mkdir -p logs
nohup conda run -n co-reasoning python auto_inner_reroute.py > logs/auto_inner_reroute.log 2>&1 &
```

查看日志：

```bash
tail -f logs/auto_inner_reroute.log
```

停止后台进程：

```bash
ps aux | grep auto_inner_reroute.py
kill <PID>
```

### 4. 默认行为

`auto_inner_reroute.py` 默认每 5 秒执行一次检测：

1. 在线获取最新拓扑和链路指标。
2. 检测节点或链路是否故障。
3. 判断故障是否影响 `inner_business_flows.json` 中的业务路径。
4. 对受影响业务调用 `inner_rl_reroute.py` 重新计算路径。
5. 打印受影响业务和重路由后的路径。

路径打印格式：

```text
node_id（ip） -> node_id（ip） -> node_id（ip）
```

调整检测间隔：

```bash
python auto_inner_reroute.py --interval 10
```

## 二、调试文件与工具说明

本节面向调试和开发使用。

### 目录与入口

```text
auto_inner_reroute.py               # 内场自动故障检测与重路由主入口
generate_inner_business_flows.py    # 生成内场业务流路径文件
inner_rl_reroute.py                 # 内场 II/III 协同重路由编排
inner_rl_reroute_II.py              # II 类路由
inner_rl_reroute_III.py             # III 类路由
inner_post_II_info.py               # 内场知识库接口，默认目标 192.168.1.24:5001
inner_post_table_flow.py            # 内场流表接口，默认目标 192.168.1.24:12590
environment/inner_graph_data/       # 内场拓扑、链路指标、GraphML 和解析工具
rl_reroute_II.py                    # 外场 II 类 DDQN 流程
rl_reroute.py                       # 外场 III 类 DDQN 流程
```

### 数据文件

内场默认数据位置：

```text
environment/inner_graph_data/json-data/network_topology_state.json
environment/inner_graph_data/json-data/link_metric.json
environment/inner_graph_data/json-data/inner_business_flows.json
environment/inner_graph_data/base_ii_topology.graphml
```

状态约定：

```text
node_status = 0  表示节点离线/故障
link_status = 0  表示链路离线/故障
```

`link_metric.json` 通过 `link_id` 更新链路利用率、可用带宽、丢包率等指标。如果指标中的 `link_id` 和拓扑不匹配，脚本会保留拓扑里的原始链路属性。

### 节点 ID 约定

内场节点 ID 一般采用：

```text
<节点角色前缀><子图编号>n<子图内节点编号>
```

示例：

```text
bsu0n1  第 0 个子图中的骨干节点 n1
asu0n2  第 0 个子图中的接入节点 n2
eru2n4  第 2 个子图中的车载侧节点 n4
```

常见前缀：

```text
bsu      骨干节点
asu      接入节点
eru/cnu  车载侧节点
```

多个子图通过各自的骨干节点互联，例如 `bsu0n1`、`bsu1n1`、`bsu2n1`。

### 常用运行模式

知识库离线、拓扑和链路接口在线：

```bash
python auto_inner_reroute.py --kg_offline
```

知识库、拓扑、链路指标全部使用本地文件：

```bash
python auto_inner_reroute.py --kg_offline --net_offline
```

只检测一次：

```bash
python auto_inner_reroute.py --kg_offline --net_offline --once
```

打印 `inner_rl_reroute.py` 子进程完整输出：

```bash
python auto_inner_reroute.py --print-child-output
```

### 手动执行协同重路由

默认源宿节点为：

```text
asu0n0 -> eru1n5
```

默认在线运行：

```bash
python inner_rl_reroute.py
```

指定节点 ID：

```bash
python inner_rl_reroute.py --src asu0n0 --dst eru1n5
```

指定节点 IP：

```bash
python inner_rl_reroute.py --src-ip 10.104.0.254 --dst-ip 10.103.21.254
```

知识库离线输出：

```text
logs/access.log
logs/offline_ii_policy.json
```

在线默认协同日志路径：

```text
/home/ict/projects/kg_network/semprotocol/log/access.log
```

### 单独运行 II/III 路由

II 类按链路时延计算最短路径：

```bash
python inner_rl_reroute_II.py
python inner_rl_reroute_II.py --src asu0n0 --dst eru1n5
python inner_rl_reroute_II.py --net_offline
```

III 类执行 QoS 路由，默认约束为最小可用带宽 100 Mbps、最大端到端丢包率 0.01、最多检查 20 条候选路径：

```bash
python inner_rl_reroute_III.py
python inner_rl_reroute_III.py --src asu0n0 --dst eru1n5
python inner_rl_reroute_III.py --net_offline
```

### 生成业务流调试

本地 JSON 生成业务：

```bash
python generate_inner_business_flows.py --net_offline
```

固定随机种子：

```bash
python generate_inner_business_flows.py --seed 1
```

如需添加指定源目的业务，不通过命令行添加，直接修改 `generate_inner_business_flows.py` 顶部配置：

```python
CUSTOM_FLOWS: list[tuple[str, str]] = []
CUSTOM_FLOW_IPS: list[tuple[str, str]] = []
```

### 拓扑和链路指标工具

抓取拓扑：

```bash
python environment/inner_graph_data/get-topo-data.py
```

抓取链路指标：

```bash
python environment/inner_graph_data/get-link-metric-data.py
```

拓扑结构变化后重建 base GraphML：

```bash
python environment/inner_graph_data/rebuild_base_topology.py
```

如果在线接口不可用，但本地 JSON 已经是新结构：

```bash
python environment/inner_graph_data/rebuild_base_topology.py --net_offline
```

拓扑解析和路由核心逻辑：

```text
environment/inner_graph_data/topology_to_networkx.py
environment/inner_graph_data/qos_routing.py
```

### 注入自定义故障

故障目标不从命令行传入，直接修改脚本顶部配置：

```python
FAULT_NODES: list[str] = []
FAULT_LINK_IDS: list[str] = []
FAULT_LINK_ENDPOINTS: list[tuple[str, str]] = []
FAIL_INCIDENT_LINKS = False
CLEAR_EXISTING = False
```

生成故障拓扑：

```bash
python environment/inner_graph_data/inject_topology_faults.py
```

默认输出：

```text
environment/inner_graph_data/json-data/network_topology_state_fault.json
```

配合自动检测脚本验证：

```bash
python auto_inner_reroute.py --kg_offline --net_offline --once \
  --topology-json environment/inner_graph_data/json-data/network_topology_state_fault.json
```

### 内场协同逻辑

`inner_rl_reroute.py` 执行顺序：

1. 调用 `inner_rl_reroute_II.py` 得到 II 路径。
2. 将 II 策略写入知识库；`--kg_offline` 时写入 `logs/offline_ii_policy.json`。
3. 调用 `inner_rl_reroute_III.py` 得到 III 路径。
4. 从知识库读取 II 策略。
5. 使用 `inner_post_table_flow.policy_compare()` 比较 II/III 策略。
6. 按 `rl_reroute.py` 的格式组织协同日志。

当前 `inner_rl_reroute.py` 不实际下发流表；流表接口在 `inner_post_table_flow.py` 中。

### 外场 DDQN 流程

外场入口：

```bash
python rl_reroute_II.py
python rl_reroute.py
```

建议顺序：

1. 先运行 `rl_reroute_II.py`，加载 II 类 DDQN 模型并计算本地策略。
2. `rl_reroute_II.py` 通过 `post_II_info.py` 将 II 策略上报外场知识库。
3. 再运行 `rl_reroute.py`，加载 III 类 DDQN 模型并计算全局策略。
4. `rl_reroute.py` 从知识库读取 II 策略，通过 `post_table_flow.policy_compare()` 比较 II/III 策略。
5. 最终写入 `logs/access.log`，并输出路径、端口/IP、时延、带宽、丢包率和策略比较结果。

常用运行方式：

```bash
python rl_reroute_II.py --src_dev_ip 192.168.10.2/24 --dst_dev_ip 192.168.40.2/24
python rl_reroute.py --src_dev_ip 192.168.10.2/24 --dst_dev_ip 192.168.40.2/24
```

外场默认业务 IP：

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

外场接口脚本：

```text
post_II_info.py      # 默认目标 192.168.2.11:5001
post_table_flow.py   # 默认目标 192.168.2.26:12590
```

注意：内场 `inner_*` 和外场 `rl_reroute*.py` 使用不同接口脚本和不同服务地址，调试时不要混用。
