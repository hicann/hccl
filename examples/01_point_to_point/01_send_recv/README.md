# 点对点通信 - HcclSend/HcclRecv(基础收发功能)

## 样例介绍

本样例展示如何基于 `HcclSend()` 和 `HcclRecv()` 接口实现点对点通信，包含以下功能点：

- 设备检测，通过 `aclrtGetDeviceCount()` 接口查询可用设备数量。
- 将 rank0 作为 root 节点，通过 `HcclGetRootInfo()` 接口生成 root 节点的 rootinfo 标识信息。

  > rootinfo 标识信息主要包含：Device IP、Device ID 等信息，此信息需广播至集群内所有 rank 用来初始化通信域。

- 在每个线程中，基于 rootinfo 标识信息通过 `HcclCommInitRootInfo()` 接口初始化通信域。
- 调用 `HcclSend()` 和 `HcclRecv()` 接口，0/2/4/6 卡发送数据，1/3/5/7 卡接收数据，并打印接收结果。

## 目录结构

```text
├── main.cc      # 样例源文件
├── Makefile     # 编译/构建配置文件
└── send_recv    # 编译生成的可执行文件
```

## 环境准备

### 环境要求

本样例支持以下昇腾产品：

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term> / <term>Atlas A3 推理系列产品</term>

### 配置环境变量

```bash
# 设置 CANN 环境变量，以 root 用户默认安装路径为例
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 编译执行样例

在本样例代码目录下执行如下命令：

```bash
make
make test
```

## 结果示例

偶数节点的 `sendBuf` 内容初始化为 Device Id，然后将数据发送至下一奇数节点，因此各个奇数节点接收到的是上一节点的 Device Id。

```text
Found 8 NPU device(s) available
rankId: 1, output: [ 0 0 0 0 ]
rankId: 3, output: [ 2 2 2 2 ]
rankId: 5, output: [ 4 4 4 4 ]
rankId: 7, output: [ 6 6 6 6 ]
```
