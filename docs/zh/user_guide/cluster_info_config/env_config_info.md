# 环境变量配置资源信息

除了通过rank table文件配置资源信息的方式外，开发者还可以通过本节所述环境变量组合的方式配置资源信息。

环境变量配置资源信息的方式仅适用于TensorFlow框架网络的通信域初始化，仅支持如下产品：

Atlas A2 训练系列产品/Atlas A2 推理系列产品

Atlas 训练系列产品

## 配置说明

需要在执行训练的每个AI Server节点上分别配置如下环境变量，进行资源信息的配置，示例如下：

```bash
export CM_CHIEF_IP=192.168.1.1
export CM_CHIEF_PORT=6000
export CM_CHIEF_DEVICE=0
export CM_WORKER_SIZE=8
export CM_WORKER_IP=192.168.0.1
export HCCL_SOCKET_FAMILY=AF_INET
```

- CM_CHIEF_IP：Master节点的Host监听IP，即与其他节点进行通信的IP地址，要求为常规IPv4或IPv6格式。
- CM_CHIEF_PORT：Master节点的监听端口，需要配置为整数，取值范围“0～65520”，请确保端口未被其他进程占用。
- CM_CHIEF_DEVICE：Master节点中统计Server端集群信息的Device逻辑ID。

    该环境变量需要配置为整数，取值范围：\[0，Server内的最大Device数量-1\]。

- CM_WORKER_SIZE：用于配置组网中参与集群训练的Device总数量，需要配置为整数，取值范围“0~32768”。
- CM_WORKER_IP：用于配置当前节点与Master进行通信时所用的网卡IP，要求为常规IPv4或IPv6格式。
- HCCL_SOCKET_FAMILY：**此环境变量可选**，用于控制Device侧通信网卡使用的IP协议版本。AF\_INET代表使用IPv4协议，AF\_INET6代表使用IPv6协议，**缺省时，优先使用IPv4协议**。

说明:

- 如果环境变量“HCCL_SOCKET_FAMILY”指定的IP协议与实际获取到的网卡信息不匹配，则以实际环境上的网卡信息为准。
    例如，环境变量“HCCL_SOCKET_FAMILY”指定为“AF_INET6”，但Device侧只存在IPv4协议的网卡，则实际会使用IPv4协议的网卡。
- 通过以上环境变量的方式配置集群信息时，环境中不能存在环境变量RANK_TABLE_FILE、RANK_ID、RANK_SIZE。
- 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，若业务为单卡多进程场景，建议通过环境变量[HCCL_NPU_SOCKET_PORT_RANGE](../hccl_env/HCCL_NPU_SOCKET_PORT_RANGE.md)配置HCCL在NPU侧使用的通信端口，否则可能会导致端口冲突，但需要注意，多进程会对资源开销、通信性能产生一定的影响，配置示例：

    ```bash
    export HCCL_NPU_SOCKET_PORT_RANGE="auto"
    ```

## 配置示例

假设执行分布式训练的AI Server节点数量为2，Device数量为16为例，每个AI Server节点有8个Device。启动每个Device上的训练进程前，在对应的shell窗口中配置如下环境变量，进行资源信息的配置。

- 节点0，此节点为Master节点，负责集群信息管理、资源分配与调度。

    ```bash
    export CM_CHIEF_IP=192.168.1.1
    export CM_CHIEF_PORT=6000
    export CM_CHIEF_DEVICE=0
    export CM_WORKER_SIZE=16
    export CM_WORKER_IP=192.168.1.1
    ```

- 节点1

    ```bash
    export CM_CHIEF_IP=192.168.1.1
    export CM_CHIEF_PORT=6000
    export CM_CHIEF_DEVICE=0
    export CM_WORKER_SIZE=16
    export CM_WORKER_IP=192.168.2.1
    ```
