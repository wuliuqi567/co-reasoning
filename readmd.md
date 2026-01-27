# 给定拓扑的路由系统

待做
- 节点id到ip:port的映射
- 如何输入一个向量，让模型输出下一跳，而不是整条路径
- 基于全局信息的环境
- 如何处理节点故障



receive_traffic_total: float = Field(description="接收流量总量单位MB")
send_traffic_total: float = Field(description=发送流量总量(单位MB)”)
traffic_rate: float = Field(description="流量速率单位Mbps")
packet_loss_rate: float = Field(description="设备丢包率"examples=[0.001, 0.002])


# 全局模型设计

维度对比（num_nodes=18, max_degree=7）
模式	维度	说明
neighbor	23	原始邻居模式
compact	360	单个质量矩阵 (18×18=324) + one-hot (36)
compact+tri	207	上三角压缩 (18×19/2=171) + one-hot (36)
matrix	1008	3 个矩阵 (324×3) + one-hot (36)
hybrid	383	neighbor + compact
compact 模式比 matrix 模式降低约 65% 的维度！



压缩编码方式
编码	公式	含义
quality	bw_norm / (1 + delay_norm)	高带宽低时延 = 高质量
delay	1 - delay_norm	仅时延（反转）
bandwidth	bw_norm	仅带宽
combined	bw_norm*0.5 + (1-delay_norm)*0.5	带宽时延各半权重



一、奖励计算逻辑
奖励由 RewardCalculator.compute_reward() 方法计算，根据动作结果返回 (reward, terminated, reason)。
配置参数（RewardConfig）
参数	默认值	说明
loop_penalty	-0.5	环路惩罚（每次重复访问累乘）
timeout_penalty	-5.0	超时惩罚（步数耗尽）
invalid_action_penalty	-1.0	无效动作惩罚
disconnect_penalty	-5.0	断连惩罚（无法到达目标）
step_penalty	-0.01	每步基础惩罚
progress_scale	0.02	进展奖励系数
success_base	1.0	到达目标基础奖励
success_scale	9.0	到达目标质量奖励系数
计算逻辑流程
1. 无效动作检查   ├─ action 超出邻居范围 → 返回 invalid_action_penalty, 不终止2. 可达性检查   ├─ 当前节点不可达目标 → 返回 disconnect_penalty, 终止   └─ 下一节点不可达目标（非目标本身）→ 返回 disconnect_penalty, 终止3. 到达目标   ├─ 计算 total_delay = path_delay + step_delay   ├─ 计算 quality_ratio = shortest_path_delay / total_delay (裁剪到 [0, 2])   └─ 返回 success_base + success_scale × quality_ratio, 终止4. 环路检查   ├─ 下一节点已在路径中出现过   └─ 返回 loop_penalty × 访问次数, 不终止5. 正常步进   ├─ 计算进展 progress = dist_to_dst[current] - dist_to_dst[next]   └─ 返回 step_penalty + progress_scale × progress, 不终止
奖励公式总结
情况	奖励公式	是否终止
无效动作	-1.0	否
当前/下一节点断连	-5.0	是
到达目标	1.0 + 9.0 × (最短时延/实际时延)	是
环路	-0.5 × 访问次数	否
正常步进	-0.01 + 0.02 × (当前距离 - 下一距离)	否
超时	额外 -5.0	是
让我先查看 `dist_to_dst` 的计算方式以及相关逻辑。


1. **判断整条路径是否最优**：比较 `total_delay` 与 `shortest_path_delay`
2. **判断最后一跳是否最优**：比较 `d_cur`（当前节点到目的地的最短距离）与 `step_delay`

如果 `d_cur < step_delay`，说明从当前节点有更短的路到目的地，但 agent 选择了直接走这条更长的边，即最后一跳绕路了。

[4 tools called]

已完成奖励函数的优化。

## 修改内容

### 1. `RewardConfig` 新增配置项 (151行)

```python
suboptimal_penalty: float = -2.0  # 最后一跳绕路的惩罚
```

### 2. `compute_reward` 优化逻辑 (502-537行)

**核心判断：**
- `d_cur`：当前节点到目的地的**最短路径延迟**
- `step_delay`：实际走的最后一跳延迟
- 如果 `d_cur ≈ step_delay`，说明最后一跳走的是最优路径
- 如果 `d_cur < step_delay`，说明有更短的路径但没选（绕路了）

**三种到达情况：**

| 情况 | 条件 | 奖励 | reason |
|------|------|------|--------|
| **整条路径最优** | `total_delay ≈ shortest_path_delay` | `success_base + success_scale × 1.0` = 10.0 | `arrive_optimal` |
| **最后一跳最优** | 最后一跳最优，但整体绕路 | `success_base + success_scale × quality_ratio` | `arrive` |
| **最后一跳绕路** | `d_cur < step_delay` | `success_base + suboptimal_penalty × detour_ratio` | `arrive_suboptimal` |

**奖励示例（使用默认配置）：**

```
# 走最优路径到达目的地
reward = 1.0 + 9.0 × 1.0 = 10.0  (arrive_optimal)

# 之前绕路了，但最后一跳是最优的，quality_ratio = 0.8
reward = 1.0 + 9.0 × 0.8 = 8.2   (arrive)

# 最后一跳绕路了，detour_ratio = 0.5
reward = 1.0 + (-2.0) × 0.5 = 0.0  (arrive_suboptimal)

# 最后一跳严重绕路，detour_ratio = 1.0
reward = 1.0 + (-2.0) × 1.0 = -1.0  (arrive_suboptimal)
```
二、故障注入机制
故障由 FailureInjector 类管理，采用标记状态方式（不删除节点/边）。
配置参数（FailureConfig）
参数	默认值	说明
enable_failure	True	是否启用故障注入
failure_mode	"edge"	故障类型："edge" 或 "node"
fail_num	2	故障数量
fail_step	-1	注入时机：-1=reset时，>=0=指定步数时
ensure_reachable	True	确保注入后 src→dst 仍可达
max_failure_tries	30	最大重试次数
utilization_threshold	0.85	新增：利用率阈值，超过此值链路不可用
故障注入流程
1. 注入时机   ├─ fail_step = -1 → 在 reset() 时注入   └─ fail_step >= 0 → 在 step() 达到指定步数时注入2. 故障类型   ├─ edge 模式：随机选 fail_num 条边，设置 link_status = -1   └─ node 模式：随机选 fail_num 个节点（排除 src/dst），设置 node_status = -13. 可达性保证   ├─ 如果 ensure_reachable = True   ├─ 注入后检查 src→dst 是否可达（过滤故障+拥塞边）   └─ 不可达则重试，最多 max_failure_tries 次4. 属性保留   └─ 故障节点/边的其他属性（时延、带宽）保持不变
边不可用判定（路由过滤）
边被视为不可用的条件（任一满足）：
link_status = -1（故障标记）
link_utilization > utilization_threshold（拥塞，默认 85%）
def _is_edge_unusable(u, v):    if link_status == -1:      return True  # 故障    if link_utilization > 0.85: return True  # 拥塞    return False
最短路径计算
routing_graph = 过滤(active_graph,     node_ok = (node_status != -1),    edge_ok = (link_status != -1 AND link_utilization <= 0.85))shortest_path = dijkstra(routing_graph, src, dst, weight="link_latency")
三、状态与属性总结
节点状态
node_status = 1：正常
node_status = -1：故障（不参与路由）
边状态
link_status = 1：正常
link_status = -1：故障（不参与路由）
link_utilization > 0.85：拥塞（不参与路由）
边属性字段
字段	说明
link_latency / delay	链路时延（ms），用于最短路径计算
link_bandwidth / bandwidth	链路带宽（Mbps）
link_status	链路状态（1=正常，-1=故障）
link_utilization	链路利用率（0-1），超过阈值视为拥塞
link_type	链路类型（1=有线，2=无线）




拼接顺序
[节点在线状态 n维] + [已访问标记 n维] + [邻居槽位 n×max_degree×5维] + [当前节点one-hot n维] + [目的节点one-hot n维]
每节点特征
维度	含义
1	node_status: 在线=1, 故障=0
1	visited: 已访问=1, 未访问=0
max_degree × 5	邻居槽位
邻居槽位定义（每槽 5 维）
slot_{i,k} = [m_{i,k}, d^{tx}_{i,k}, u_{i,k}, linkOn_{i,k}, p_{i,k}]
索引	字段	说明
0	m	neighbor_mask, 槽位存在=1, 空=0
1	d^{tx}	链路时延（归一化到 [0,1]）
2	u	链路利用率（归一化到 [0,1]）
3	linkOn	链路在线=1, 故障=0
4	p	丢包率（归一化到 [0,1]）
总维度
n × (2 + max_degree × 5) + 2n
例如 n=12, max_degree=8 时：
节点特征: 12 × (2 + 8×5) = 12 × 42 = 504
one-hot: 2 × 12 = 24
总计: 528