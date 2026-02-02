# test_or_train=1 时的运行逻辑与路径计算方式

本文档描述 `NetTupu` 在 **部署模式**（`self.test_or_train == True`）下的完整运行逻辑，以及最短路径与路径指标的计算方式。不涉及代码修改，仅作说明。

---

## 一、test_or_train 的判定

```text
self.test_or_train = getattr(env_config, "test", False) and getattr(env_config, "execute_reroute", False)
```

- **部署模式**：`env_config.test == True` 且 `env_config.execute_reroute == True`。
- 典型入口：`rl_reroute.py`，配置中 `execute_reroute: True`，命令行 `--test=1`，`src`/`dst` 由参数或配置传入。

---

## 二、部署模式下的初始化逻辑（__init__）

### 2.1 图加载与 KG 更新

| 步骤 | 说明 |
|------|------|
| 1 | 使用 `graph_source = "latest_II_class_base"` 加载 **base 图**（`latest_online_graph`）。该图默认所有节点/边为离线（node_status/link_status=0）。 |
| 2 | 调用 `get_latest_metric_from_kg()` 从知识库获取：`NM_topo`、`link_metric`、`e2e_flow_data`。 |
| 3 | 调用 `update_graph_with_latest_metric(latest_online_graph, NM_topo, link_metric, e2e_flow_data)`，用 KG 数据更新图的节点/边状态与指标（node_status、link_status、时延、带宽、利用率等）。更新后图中会存在 **在线(1)** 与 **故障/离线(0)** 的节点和边。 |
| 4 | 调用 `_apply_status_failures(latest_online_graph)`：仅**读取**图中 node_status=0、link_status=0 的节点与边，得到 `status_dead_edges`、`status_dead_nodes`，图本身不删点删边。 |
| 5 | 调用 `_sync_graph_attributes(latest_online_graph)`：根据**全图拓扑**设置 `num_nodes`、`min_degree`、`max_degree`（保证观测/动作维度与训练一致）；根据**在线边**统计 `delay_range`、`bandwidth_range`（用于观测归一化）。 |

### 2.2 工作图与对比图

| 变量 | 含义 |
|------|------|
| **base_graph** | `latest_online_graph.copy()`。即 KG 更新后的图，**含故障标记**（部分 node_status/link_status=0）。作为本局 **工作图**，用于路由决策与最短路径计算。 |
| **base_graph_all_online** | `_graph_all_online(latest_online_graph)`。在同样拓扑上把所有 node_status、link_status 置为 1，得到“假设无故障”的**对比图**，可用于故障前最短路径等对比分析。 |
| **active_graph** | `base_graph.copy()`。每局实际使用的图，reset 时从 base_graph 重新 copy。 |

### 2.3 源/目的与故障注入

- **src / dst**：从 `env_config.src`、`env_config.dst` 读取并转为 int，部署时由外部（如 rl_reroute.py 的 configs）传入。
- **故障注入**：部署阶段**不开启**故障注入（`failure_config.enable_failure` 一般为 False）。故障仅来自 KG 更新后的图内已有状态（status_dead_edges / status_dead_nodes），不再在 reset/step 中调用 `_inject_failure()`。

---

## 三、部署模式下的 reset 逻辑

1. **时间步与路径**：`_current_step=0`，`path`、`path_delay` 清空。
2. **图**：`active_graph = base_graph.copy()`（始终从“KG 更新后的含故障图”恢复）。
3. **故障相关**：`failure_happened=False`，`dead_edges=[]`，`dead_nodes=[]`（本局无注入故障）；`status_dead_edges` / `status_dead_nodes` 不变，仍为图内静态故障。
4. **src/dst**：校验 `src`、`dst` 非 None，并转为 int；否则抛 `ValueError("test_or_train=True 需要传入 src 和 dst")`。
5. **故障注入**：若配置了 `enable_failure` 且 `fail_step < 0`，仍可能按概率注入；部署时通常关闭，因此不会执行。
6. **当前节点**：`current_node = src`，`path = [src]`。
7. **最短路径与距离**：调用 `_recompute_shortest_and_dists()`，得到当前（含故障）图上的最短路径和到 dst 的距离。
8. **返回**：`_build_observation()`、`_build_info(extra={..., "reset": True, "src", "dst", "test_or_train": True})`。

---

## 四、路径计算方式（核心）

### 4.1 路由图：_get_routing_graph()

所有“可路由”的路径与最短路径都基于 **路由图**，而不是原始 active_graph 全图：

```text
routing_graph = nx.subgraph_view(active_graph, filter_node=_node_ok, filter_edge=_edge_ok)
```

- **节点过滤 _node_ok(n)**：`not _is_node_failed(n)`，即 `node_status != 0`（排除故障/离线节点）。
- **边过滤 _edge_ok(u,v)**：`not _is_edge_unusable(u,v)`，即同时满足：
  - `link_status != 0`（非故障/离线边），
  - 链路利用率 ≤ utilization_threshold，
  - 链路丢包率 ≤ loss_rate_threshold。

因此，**路径与最短路径只会走“在线且未过载”的节点与边**。

### 4.2 最短路径与 dist_to_dst：_recompute_shortest_and_dists()

1. 清空：`shortest_path=None`，`shortest_path_delay=inf`，`dist_to_dst={}`。
2. 取路由图：`routing_graph = _get_routing_graph()`。
3. 若 dst 不在路由图中，直接 return，不计算路径。
4. **到 dst 的距离**：`dist_to_dst = nx.single_source_dijkstra_path_length(routing_graph, self.dst, weight="link_latency")`，即以 **dst 为源**、边权为 `link_latency` 的单源最短路长度。
5. **最短路径**：若 src 在路由图中且 src 在 `dist_to_dst` 中，则  
   `shortest_path = nx.shortest_path(routing_graph, self.src, self.dst, weight="link_latency")`，  
   `shortest_path_delay = _calculate_path_delay(shortest_path)`（在 active_graph 上沿路径累加 `link_latency`）。

要点：

- **图**：最短路径在 **routing_graph**（过滤故障与过载）上计算。
- **边权**：仅使用 **link_latency**（时延）。
- **路径时延**：用 `_calculate_path_delay(path)` 在 **active_graph** 上按路径边累加 `link_latency`。

### 4.3 当前路径（path）的指标

- **path**：由 step 中动作逐跳扩展，每一步把 `chosen_node` 加入 `path`，并累加 `path_delay`（单步用 `_get_edge_latency(active_graph[current_node][chosen_node])`）。
- **path 的 delay/bandwidth/loss_rate**：在 `_build_info()` 中通过 `_get_path_metrics(self.path)` 计算：
  - **delay**：`_calculate_path_delay(path)`，在 active_graph 上对路径每条边累加 `link_latency`。
  - **bandwidth**：`_calculate_path_bandwidth(path)`，路径上各边 `link_bandwidth` 的最小值。
  - **loss_rate**：`_calculate_path_loss_rate(path)`，由各边 `link_loss_rate` 按“1 - 各段成功率乘积”得到。

上述三个指标默认都在 **active_graph** 上计算（`_get_path_metrics` 内部用 `self.active_graph`），即包含当前图上的真实链路属性。

### 4.4 邻居与动作空间

- **可走邻居**：`_get_neighbor_list(current_node)` 在 **routing_graph** 上取 `current_node` 的邻居（已过滤故障与不可用边），并排序。
- **动作掩码**：`_get_action_mask()` 根据当前节点的可走邻居数生成长度为 `max_degree` 的 mask，前 `min(len(neighbors), max_degree)` 为 True，其余为 False。
- **step 中的选点**：`chosen_node = neighbors[int(action)]`，即动作索引对应的是**当前可走邻居列表**中的下标，而不是全局节点编号在“所有邻居”中的下标。

---

## 五、部署模式下的 step 逻辑（简要）

1. **故障注入**：若配置了 `enable_failure` 且 `fail_step >= 0` 且当前步数等于 `fail_step`，可能调用 `_inject_failure()` 并重新 `_recompute_shortest_and_dists()`；部署时通常不开启。
2. **邻居与动作**：用 `_get_neighbor_list(current_node)` 得到可走邻居；若无邻居则直接终止（disconnect）。
3. **动作解析**：`chosen_node = neighbors[int(action)]`（合法动作）；非法动作时 `chosen_node=-1`。
4. **奖励与终止**：由 `reward_calculator.compute_reward(...)` 根据是否到达 dst、是否断连、是否无效动作等计算 reward 和 terminated/reason。
5. **状态更新**：若为有效移动且非断连/无效动作，则 `current_node = chosen_node`，`path_delay += step_delay`；无论如何都会执行 `path.append(current_node)`。
6. **info**：`_build_info(extra={action_idx, chosen_node, step_delay, terminated_reason})`，其中包含 path、shortest_path、delay/bandwidth/loss_rate、dead_edges/dead_nodes、is_connected_src_dst 等；若为部署模式且存在故障，还会包含 `dead_nodes_detail`、`dead_edges_detail`。

---

## 六、观测（observation）在部署模式下的含义

- **state 模式**：`graph_for_obs = self.active_graph`（全图，含故障节点/边），由 `obs_builder.build_observation(...)` 构建。节点/边的在线与故障状态通过 node_status、link_status 等体现在观测中（如 node_online、link_on 等），故障边对应槽位可被置为 0 或 mask=0。
- **neighbor 模式**：`graph_for_obs = _get_routing_graph()`，即只包含可路由节点与边的子图。
- **维度**：`num_nodes`、`max_degree` 在 `_sync_graph_attributes` 中按**全图拓扑**计算，不随故障变化，保证与训练时 checkpoint 的观测/动作维度一致。

---

## 七、info 中与路径相关的字段（部署时）

| 字段 | 含义 |
|------|------|
| path | 当前已走过的节点序列（含 src）。 |
| path_ip_port | 将 path 转为带 node_idx、ip、in_port、out_port、in_port_ip、out_port_ip 的列表，便于下发流表等。 |
| path_delay / path_bandwidth / path_loss_rate | 当前 path 在 active_graph 上的时延、瓶颈带宽、丢包率。 |
| shortest_path | 在 routing_graph 上、以 link_latency 为权的最短路径（src→dst）。 |
| shortest_path_ip_port | 同上，转为 ip/port 信息列表。 |
| shortest_path_delay / shortest_path_bandwidth / shortest_path_loss_rate | 最短路径在 active_graph 上的时延、带宽、丢包率。 |
| is_connected_src_dst | routing_graph 上 src 与 dst 是否连通。 |
| dead_edges / dead_nodes | status_dead_edges + dead_edges、status_dead_nodes + dead_nodes（图内静态故障 + 本局注入故障；部署时通常仅前者）。 |
| dead_nodes_detail / dead_edges_detail | 仅部署模式且存在故障时存在，列表元素为 {idx, node_id, ip} 或 {src_idx, dst_idx, link_id}，便于定位故障。 |

---

## 八、小结（test_or_train=1）

1. **图**：加载 base 图 → KG 更新得到含故障的 latest_online_graph → base_graph = 其副本，active_graph 每局从 base_graph copy；不删点删边，仅通过 node_status/link_status 标记故障。
2. **路径计算**：所有“可走”的路径与最短路径都在 **routing_graph** 上计算（过滤 node_status=0、link_status=0 及过载/高丢包边），边权为 **link_latency**；路径的 delay/bandwidth/loss_rate 在 **active_graph** 上计算。
3. **部署特点**：src/dst 由外部传入；不（或很少）启用故障注入；观测/动作维度按全图拓扑固定；info 中提供 path、shortest_path 及其 ip/port 与指标，以及故障明细（dead_*_detail），便于协同推理与流表下发。
