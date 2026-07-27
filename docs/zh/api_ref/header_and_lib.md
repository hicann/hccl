# 概述

HCCL（Huawei Collective Communication Library，华为集合通信库）基于昇腾AI处理器，提供单机/多机环境下的高性能集合通信与点对点通信能力，是CANN的核心组件之一。HCCL与通信基础库HCOMM通过dlsym动态加载解耦，HCCL与HCOMM独立编译、独立版本演进，HCOMM的对外头文件与库文件请参考[HCOMM对外头文件与库文件说明](https://gitcode.com/cann/hcomm/blob/master/docs/zh/api_ref/header_and_lib.md)。

本章节介绍HCCL对外接口的头文件与库文件说明。

## 头文件和库文件说明

### 接口分类

HCCL对外接口按功能分为以下类别：

**表1**接口分类

| 接口类别 | 描述 |
| --- | --- |
| 集合通信算子 | AllReduce、Broadcast、AllGather(V)、ReduceScatter(V)、Scatter、Reduce、AlltoAll(V)(C)等11个集合通信接口。 |
| 点对点通信算子 | Send、Recv、BatchSendRecv等3个点对点通信接口。 |
| MC2自定义算子 | HcclKfc\*参数对象与HcclCreateOpResCtx等9个MC2（Kernel Fusion Custom）自定义算子框架接口。 |

### 调用接口依赖的头文件和库文件说明

安装固件、驱动及CANN软件包后，编译、运行应用程序时才能引用到HCCL接口的头文件、库文件。

您需要根据实际使用的HCCL接口来include依赖的文件，各头文件的用途如下表所示。

HCCL接口的头文件在"${INSTALL_DIR}/include/hccl/"目录下，库文件在"${INSTALL_DIR}/lib64/"目录下。${INSTALL_DIR}请替换为CANN软件安装后文件存储路径。以root用户安装为例，安装后文件默认存储路径为：/usr/local/Ascend/cann。

> **须知：**
> 编译HCCL接口程序时，请按照include的头文件依赖对应的库文件，如果引用多余的so文件，可能导致版本功能异常或后续版本升级时存在兼容性问题。

**表2**头文件列表

| 定义接口的头文件 | 用途 | 对应的库文件 |
| --- | --- | --- |
| hccl/hccl.h | 用于定义AllReduce、Broadcast、AllGather(V)、ReduceScatter(V)、Scatter、Reduce、AlltoAll(V)(C)、Send、Recv、BatchSendRecv等14个集合通信与点对点通信算子接口。 | libhccl.so |
| hccl/hccl_mc2.h | 用于定义MC2自定义算子框架接口，包括HcclKfc\*参数对象分配/设置与HcclCreateOpResCtx通信资源上下文创建等9个接口。 | libhccl.so |

HCCL以`cann-hccl_<version>_linux-<arch>.run`安装包形式发布，包含`libhccl.so`、对外头文件与`aicpu_hccl.tar.gz`（HCCL AICPU算子包）；静态构建模式另产出`libhccl_static.a`。

源码编译与安装流程详见[构建指南](../build/build.md)；各算子函数原型、参数说明、数据类型支持与约束详见[通信算子接口](./comm_op_interface/README.md)。
