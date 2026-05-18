# HCCL_BUFFSIZE

## 功能描述

此环境变量用于控制通信域所使用的共享数据缓存区大小。需要配置为整数，取值大于等于1，默认值为200，单位MB。

集合通信中，每个通信域都会占用HCCL_BUFFSIZE大小的缓存区。若集群中存在较多的通信域，整体缓存占用会增加，可能影响模型数据的正常存储。此种场景下，可通过调小该环境变量的值以降低通信域占用的缓存空间；若业务的模型数据量较小，但通信数据量较大，可适当调大该环境变量的值以增大通信域占用的缓存空间，以提升数据通信效率。

大语言模型中的建议配置值如下：

\(MicrobatchSize \* SequenceLength \* hiddenSize \* sizeof \(DataType\) \)/\(1024\*1024\)，向上取整。

此环境变量一般用于以下场景：

- 动态shape网络场景。
- 开发人员调用集合通信库HCCL的C语言接口进行框架对接的场景。

需要注意：

- 该环境变量申请的内存为HCCL独占，不可与其他业务内存复用。
- 每个通信域占用“2\*HCCL_BUFFSIZE”大小的内存，分别用于收发内存。
- 该资源按通信域粒度管理，每个通信域独占一组“2\*HCCL_BUFFSIZE”大小的内存，保证多通信域并发算子互不影响。
- 针对集合通信算子，当数据量超过HCCL_BUFFSIZE的取值时，可能会出现性能下降的情况，建议HCCL_BUFFSIZE的取值大于数据量。

## 配置示例

```bash
export HCCL_BUFFSIZE=200
```

## 使用约束

若您调用HCCL C接口初始化具有特定配置的通信域时，通过“HcclCommConfig”的“hcclBufferSize”参数配置了共享数据的缓存区大小，则以通信域粒度的配置优先。

## 支持的型号

Ascend 950PR/Ascend 950DT

Atlas A3 训练系列产品/Atlas A3 推理系列产品

Atlas A2 训练系列产品/Atlas A2 推理系列产品（针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，仅支持Atlas 800T A2 训练服务器、Atlas 900 A2 PoD 集群基础单元、Atlas 200T A2 Box16 异构子框。）

<cann-filter npu-type="910">Atlas 训练系列产品</cann-filter>

<cann-filter npu-type="310p">Atlas 推理系列产品（针对Atlas 推理系列产品，仅支持Atlas 300I Duo 推理卡。）</cann-filter>
