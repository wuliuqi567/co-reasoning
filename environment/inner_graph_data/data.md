2.1.3.1 节点（node）字段
参数名	类型	必填	取值范围/格式	示例值
	描述

node_id	string	是	多维标识网络 EID	0001056891d06f04e100004400000000	节点 ID，唯一标识节点，目前使用多维标识网络 EID

node_type	integer	是	1~7	5
	节点类型：1=一类设备终端，2=一类设备簇头，3=二类设备车载，4=二类设备接入，5=二类设备骨干，6=Ⅳ类设备网关，7=III类设备
node_location
	string
	是	经度,纬度（英文逗号分隔）	118.76578522,118.76578522	节点物理位置，经纬度坐标

node_manage_ip_addr	string	否	IPv4 地址	10.100.0.1	节点管理 IP，仅用于 III 类设备下发管控配置
node_ports	array	是	-	-	节点逻辑端口信息列表
2.1.3.2 节点端口（node_ports）字段
参数名	类型	必填	取值范围/格式	示例值	描述
port_id	string
	是	<node_id>:<端口序号>	0001056891d06f04e100004400000000:1	端口 ID，包含节点 ID 和端口序号
status	integer	是	0 或 1	1	端口状态：0=关闭，1=启用
nid	integer	是	≥0	10	多维标识网络的地址
teid	integer	是	≥0	22	端口的隧道端点标识符
ip_address	string	是	IPv4 地址	10.0.1.1	端口的 IP 地址
mac_address	string	是	MAC 地址	00:00:00:00:00:01	端口的 MAC 地址
2.1.3.3 链路（link）字段
参数名	类型	必填	取值范围/格式	示例值	描述
link_id	string	是	<src_port>_<dst_port>
	0001056891d06f04e100004400000000:1_0001056891d06f04e200004400000000:1	链路 ID，由源端口与目标端口拼接而成
link_status	integer	是	-1~2	0	链路状态：-1=断开，0=忙碌，1=闲置，2=停置
link_bandwidth	float	是	≥0（单位 Mbps）	1000.0	链路总带宽
link_latency	float	是	≥0（单位 ms）	51.0	链路延迟
src.src_node	string	是	多维标识网络 EID
	0001056891d06f04e100004400000000	源节点 ID

src.src_port	string	是	<node_id>:<端口序号>	0001056891d06f04e100004400000000:1	源端口 ID
dst.dst_node	string	是	多维标识网络 EID	0001056891d06f04e200004400000000	目标节点 ID
dst.dst_port	string	是	<node_id>:<端口序号>	0001056891d06f04e200004400000000:1	目标端口 ID