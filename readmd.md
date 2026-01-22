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