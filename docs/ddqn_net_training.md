# DDQN 训练运行说明

本文说明 `execute/ddqn_net.py` 如何运行，以及训练数据、日志和模型保存在哪里。

## 1. 运行训练

训练前先获取最新网络数据：

```bash
cd /home/co-reasoning/environment/topo_parse
python get_and_update_graph.py
```

该脚本会从知识库地址 `http://192.168.2.101:5000` 获取最新网络拓扑和链路状态：

```text
NM_topo
NM_link_metrics
```

并保存到：

```text
/home/co-reasoning/environment/jsondata/data_topo_link_info/
```

保存文件格式为：

```text
topo_II_class_<timestamp>.json
link_II_class_<timestamp>.json
```

获取完成后，再启动训练。

从项目根目录进入 `execute/` 目录后运行：

```bash
cd /home/co-reasoning/execute
python ddqn_net.py
```

脚本会读取：

```text
/home/co-reasoning/config/ddqn.yaml
```

注意：脚本中配置文件路径写的是 `../config/ddqn.yaml`，所以推荐从 `execute/` 目录运行。

## 2. 查看训练日志

训练日志使用 TensorBoard 保存。当前配置中的日志目录是：

```yaml
logger: "tensorboard"  # Choices: tensorboard, wandb.
```

离线训练时使用 `tensorboard`，日志会保存到本地；如果使用 `wandb`，通常需要联网并登录 Weights & Biases。

```text
/home/co-reasoning/logs/latest_II_class_ddqn_iii/
```

查看日志：

```bash
cd /home/co-reasoning
tensorboard --logdir logs/latest_II_class_ddqn_iii
```

然后打开 TensorBoard 输出的访问地址，通常是：

```text
http://localhost:6006
```

## 3. 缩短训练时间

训练轮数主要在配置文件中调整：

```text
/home/co-reasoning/config/ddqn.yaml
```

当前训练步数配置为：

```yaml
running_steps: 8_500_000
parallels: 50
eval_interval: 50_000
```

实际每个并行环境执行的训练循环约为：

```text
running_steps / parallels
```

如果只是快速验证流程，可以把训练步数调小，例如：

```yaml
running_steps: 100_000
eval_interval: 10_000
```

如果还想进一步减少并行环境开销，也可以降低：

```yaml
parallels: 10
```

建议：

- 快速检查代码是否能跑通：`running_steps` 可设置为 `50_000` 到 `100_000`。
- 正式训练：再恢复到更大的 `running_steps`。
- `eval_interval` 不要大于 `running_steps`，否则训练期间可能不会进行有效评估。

## 4. 模型保存位置

当前配置中的模型目录是：

```text
/home/co-reasoning/models/latest_II_class_ddqn_iii/
```

每次训练会生成一个带 seed 和时间戳的子目录，例如：

```text
/home/co-reasoning/models/latest_II_class_ddqn_iii/seed_1_2026_0610_171003/
```

默认训练会保存最优模型：

```text
best_model.pth
```

训练完成后，最终要用于工程运行的模型需要放到：

```text
/home/co-reasoning/themodels/latest_II_class_ddqn_iii/
```

例如将训练好的模型目录复制过去：

```bash
cp -r /home/co-reasoning/models/latest_II_class_ddqn_iii/seed_1_2026_0610_171003 \
      /home/co-reasoning/themodels/latest_II_class_ddqn_iii/
```

然后修改最终工程脚本：

```text
/home/co-reasoning/rl_reroute.py
```

找到模型加载位置：

```python
Agent.load_model(path=Agent.model_dir_load, model="seed_1_2026_0311_205726")
```

把 `model` 改成新训练模型的目录名，例如：

```python
Agent.load_model(path=Agent.model_dir_load, model="seed_1_2026_0610_171003")
```

`rl_reroute.py` 是最终工程中使用的推理脚本，工程运行时会加载这里指定的模型。

## 5. 训练数据位置

训练环境代码在：

```text
/home/co-reasoning/environment/net_tupu_iii.py
```

当前配置默认使用的图数据是：

```text
/home/co-reasoning/environment/graph_data/latest_II_class.graphml
```

历史动态拓扑数据在：

```text
/home/co-reasoning/environment/jsondata/data_topo_link_info/
```

该目录下的成对文件会在训练中被随机加载，例如：

```text
topo_II_class_<timestamp>.json
link_II_class_<timestamp>.json
```

其他 JSON 数据在：

```text
/home/co-reasoning/environment/jsondata/
```
