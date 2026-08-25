# HCCL_RDMA_SL配置错误 (EI0001)

## 问题现象

在打印日志中存在关键字`EI0001`或`Value *** for environment variable *** is invalid`，如下所示：

```text
[PID:3729526]2025-10-23-17:30:40.098.984Config_Error_Invalid_Environment_Variable(EI0001): Value 1000 for environment variable HCCL_RDMA_SL is invalid. Expected value : range[0, 7].
```

<!-- npu="A3,910b,910,310p" id4 -->
针对如下产品，CANN日志的ERROR日志中存在关键字"externalinput.cc"，表示是在读取环境变量配置时报错。

  <!-- npu="A3" id1 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
  <!-- end id1 -->
  <!-- npu="910b" id2 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品
  <!-- end id2 -->
  <!-- npu="910" id5 -->
- Atlas 训练系列产品
  <!-- end id5 -->
  <!-- npu="310p" id3 -->
- Atlas 推理系列产品
  <!-- end id3 -->

报错示例如下所示：

```text
[ERROR]HCCL(3729526,python3.11):2025-10-23-17:30:40.098.973 [externalinput.cc:963] [3729526][Parse][rdmaServerLevel]HCCL_RDMA_SL[1000] is invalid. except: [0, 7]
[ERROR]HCCL(3729526,python3.11):2025-10-23-17:30:40.099.058 [externalinput.cc:169] [3729526][InitGroupStage][EnvConfig]errNo[0x0000000005000001] In init env variable param, parse HCCL_RDMA_SL failed. errno[1]
[ERROR]HCCL(3729526,python3.11):2025-10-23-17:30:40.099.063 [externalinput.cc:47] [3729526][InitExternalInput]call trace: hcclRet -> 1
[ERROR]HCCL(3729526,python3.11):2025-10-23-17:30:40.099.068 [op_base.cc:866] [3729526][HcclGetRootInfo]call trace: hcclRet -> 1
```
<!-- end id4 -->

## 可能的原因及解决方法

环境变量配置参数不符合要求，请基于日志打印的建议调整取值范围，如果仍然有疑问，请参照对应[环境变量参考](../hccl_env/README.md)。
