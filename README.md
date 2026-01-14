# HCCL

## 🔥Latest News

- [2025/11/30] HCCL项目正式开源。

## 🚀 概述

集合通信库（Huawei Collective Communication Library，简称HCCL）是基于昇腾AI处理器的高性能集合通信库，为计算集群提供高性能、高可靠的通信方案，具备以下核心功能：

- 提供单机、多机环境中的高性能集合通信和点对点通信。
- 支持AllReduce、Broadcast、AllGather、ReduceScatter、AlltoAll等集合通信原语。
- 支持Ring、Mesh、Recursive Halving-Doubling（RHD）等通信算法。
- 支持HCCS、RoCE、PCIe等高速通信链路。
- 支持单算子和图模式两种执行模式。

HCCL是CANN的核心组件，对上支持多种AI框架，对下使能多款昇腾AI处理器之间的通信能力，其软件架构如下图所示：

<img src="./docs/figures/architecture.png" alt="hccl-architecture" style="width: 65%;  height:65%;" />

HCCL包含HCCL集合通信库与HCOMM（Huawei Communication）通信基础库：

- HCCL集合通信库：提供多种通信算子的高性能实现，并支持开发者自定义集合通信算子。
- HCOMM通信基础库：提供通信域及通信资源的管理能力。

> [!NOTE]说明
> 为了更好的支持用户定制通信库，HCCL引入了一套全新的通信算子开发架构，并在该架构下提供了Scatter算子的参考实现。其余算子当前仍沿用HCOMM通信基础库中的现有实现。<br>
> 未来我们将基于新架构逐步扩展并完善其余算子的功能，同时原有架构中成熟的算子方案仍会在一段时间内继续维护，以保障商业部署的稳定运行。

## 🔍 目录结构说明

本项目关键目录如下所示：

```
│── src                         # HCCL算子源码目录
|    ├── common
|    └── ops
|        ├── channel
|        ├── inc
|        ├── registry
|        ├── scatter        # Scatter 算子
|        └── topo
├── build.sh                # 源码编译脚本
├── docs                    # 资料文档
└── test                    # 测试代码目录
```

## ⚡️ 快速开始

若您希望快速构建本项目，请访问 [源码构建](./docs/build.md)，了解如何编译、安装本项目，并进行基础测试验证。

## 📖 学习教程

HCCL提供了用户指南、技术文章、培训视频，详细可参见 [HCCL 参考资料](./docs/README.md)。

## 📝 相关信息

- [贡献指南](CONTRIBUTING.md)
- [安全声明](SECURITY.md)
- [许可证](LICENSE)
