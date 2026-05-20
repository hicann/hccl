# HCCL_IF_BASE_PORT

## 功能描述

当通信域的创建方式为“基于root节点信息创建”时，可以通过该环境变量指定Host网卡起始端口号，配置后系统默认占用以该端口起始的32个端口进行集群信息收集。

该环境变量需要配置为整数，取值范围为\[1024,65520\]，请确保分配的端口未被占用。

## 配置示例

```bash
export HCCL_IF_BASE_PORT=50000
```

## 使用约束

分布式场景下，HCCL会使用Host服务器的部分端口进行集群信息收集，需要操作系统预留该部分端口。

- 若不通过HCCL_IF_BASE_PORT环境变量指定端口，默认HCCL使用60000-60031端口，需要执行如下命令预留此范围的操作系统端口：

    ```bash
    sysctl -w net.ipv4.ip_local_reserved_ports=60000-60031
    ```

- 若通过HCCL_IF_BASE_PORT环境变量指定端口，例如指定端口为50000，则HCCL使用50000-50031端口，需要执行如下命令预留此范围的操作系统端口：

    ```bash
    sysctl -w net.ipv4.ip_local_reserved_ports=50000-50031
    ```

## 支持的型号

Ascend 950PR/Ascend 950DT

Atlas A3 训练系列产品/Atlas A3 推理系列产品

Atlas A2 训练系列产品/Atlas A2 推理系列产品（针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，仅支持Atlas 800T A2 训练服务器、Atlas 900 A2 PoD 集群基础单元、Atlas 200T A2 Box16 异构子框。）

<cann-filter npu-type="910">Atlas 训练系列产品</cann-filter>

<cann-filter npu-type="310p">Atlas 推理系列产品（针对Atlas 推理系列产品，仅支持Atlas 300I Duo 推理卡。）</cann-filter>
