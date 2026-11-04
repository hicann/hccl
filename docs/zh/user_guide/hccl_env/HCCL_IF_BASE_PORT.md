# HCCL_IF_BASE_PORT

## 功能描述

当通信域的创建方式为“基于root节点信息创建”时，可以通过该环境变量指定Host网卡起始端口号。配置后，HCCL以“起始端口 + 物理设备 ID”为基础计算各NPU使用的端口，该部分端口同时用于集群信息收集与控制面建链，实际占用端口数量与产品形态及单机NPU数量相关。

该环境变量需要配置为整数，取值范围为\[1024,65520\]，请确保分配的端口未被占用。

## 配置示例

```bash
export HCCL_IF_BASE_PORT=50000
```

## 使用约束

分布式场景下，HCCL会使用Host服务器的部分端口进行集群信息收集及控制面建链，需要操作系统预留该部分端口。

- 若不通过HCCL_IF_BASE_PORT环境变量指定端口，HCCL默认在60000-60031范围内选择端口，需要执行如下命令预留此范围的操作系统端口：

    ```bash
    sysctl -w net.ipv4.ip_local_reserved_ports=60000-60031
    ```

- 若通过HCCL_IF_BASE_PORT环境变量指定端口，例如指定端口为50000，则HCCL使用50000-50031端口，需要执行如下命令预留此范围的操作系统端口：

    ```bash
    sysctl -w net.ipv4.ip_local_reserved_ports=50000-50031
    ```

## 产品支持情况

<!-- npu="950" id3 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id3 -->
<!-- npu="A3" id4 -->
- Atlas A3系列产品：支持
<!-- end id4 -->
<!-- npu="910b" id5 -->
- Atlas A2系列产品：支持
<!-- end id5 -->
<!-- npu="910" id1 -->
- Atlas训练系列产品：支持
<!-- end id1 -->
<!-- npu="310p" id2 -->
- Atlas推理系列产品：支持
<!-- end id2 -->
