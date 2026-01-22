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
