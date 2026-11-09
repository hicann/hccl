# HcclReduceScatter

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：支持
<!-- end id3 -->
<!-- npu="310p" id4 -->
- Atlas推理系列产品：支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas训练系列产品：支持
<!-- end id5 -->

<!-- npu="910b,310p" id9 -->
> [!NOTE]说明
>
<!-- npu="910b" id17 -->
> - 针对Atlas A2系列产品，仅支持Atlas 800T A2训练服务器、Atlas 900 A2 PoD集群基础单元、Atlas 200T A2 Box16异构子框。
<!-- end id17 -->
<!-- npu="310p" id18 -->
> - 针对Atlas推理系列产品，仅支持Atlas 300I Duo推理卡。
<!-- end id18 -->
<!-- end id9 -->

## 功能说明

集合通信算子ReduceScatter的操作接口，将通信域内所有rank的输入数据均分成rank size份，然后分别取每个rank的rank size之一份数据进行归约操作（如sum、prod、max、min）。最后，将结果按照编号分散到各个rank的输出buffer。

![reducescatter](figures/reducescatter.png)

## 函数原型

```c
HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| sendBuf | 输入 | 源数据buffer地址。 |
| recvBuf | 输出 | 目的数据buffer地址，集合通信结果输出至此buffer中。 |
| recvCount | 输入 | 参与ReduceScatter操作的recvBuf的数据size，sendBuf的数据size则等于recvCount * rank size。 |
| dataType | 输入 | ReduceScatter操作的数据类型，[HcclDataType](https://gitcode.com/cann/hcomm/blob/master/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md)类型。<br>不同的型号支持的数据类型不同，详细请参见[dataType说明](#datatype说明)。|
| op | 输入 | Reduce的操作类型。<br>不同的型号支持的操作类型不同，详细请参见[op说明](#op说明)。|
| comm | 输入 | 集合通信操作所在的通信域。 |
| stream | 输入 | 本rank所使用的stream。 |

### dataType说明

<!-- npu="950" id10 -->
- 针对Ascend 950PR&950DT系列产品，支持数据类型：int8、int16、int32、int64、uint64、float16、float32、float64、bfp16、hifloat8（当前仅在AIV展开模式下支持hifloat8类型）。
<!-- end id10 -->
<!-- npu="A3" id11 -->
- 针对Atlas A3系列产品，支持数据类型：int8、int16、int32、int64、float16、float32、bfp16。
<!-- end id11 -->
<!-- npu="910b" id12 -->
- 针对Atlas A2系列产品，支持数据类型：int8、int16、int32、int64、float16、float32、bfp16。需要注意，针对int64数据类型，性能会有一定的劣化。
<!-- end id12 -->
<!-- npu="910" id13 -->
- 针对Atlas训练系列产品，支持数据类型：int8、int32、int64、float16、float32。
<!-- end id13 -->
<!-- npu="310p" id6 -->
- 针对Atlas推理系列产品，支持数据类型：int8、int16、int32、float16、float32。
<!-- end id6 -->

### op说明

<!-- npu="950" id14 -->
- 针对Ascend 950PR&950DT系列产品，支持的操作类型为sum、prod、max、min，其中prod操作不支持int16、bfp16数据类型。
<!-- end id14 -->
<!-- npu="A3" id15 -->
- 针对Atlas A3系列产品，支持的操作类型为sum、prod、max、min，其中prod操作不支持int16、bfp16数据类型。
<!-- end id15 -->
<!-- npu="910b" id16 -->
- 针对Atlas A2系列产品，支持的操作类型为sum、prod、max、min，其中prod操作不支持int16、bfp16数据类型。
<!-- end id16 -->
<!-- npu="310p" id7 -->
- 针对Atlas推理系列产品，支持的操作类型为sum、prod、max、min，其中prod、max、min操作不支持int16数据类型。
<!-- end id7 -->

## 返回值

[HcclResult](https://gitcode.com/cann/hcomm/blob/master/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclResult.md)

| 返回值 | 说明 |
| --- | --- |
| HCCL_SUCCESS | 接口调用成功。 |
| HCCL_E_PTR | 传入的指针参数为空，如comm、sendBuf、recvBuf、stream等为nullptr。 |
| HCCL_E_PARA | 传入的参数无效，如recvCount超过上限等。 |
| HCCL_E_NOT_SUPPORT | 操作不被支持，如dataType非法或当前型号不支持、prod操作不支持int16/bfp16数据类型等。 |
| HCCL_E_INTERNAL | 内部错误。 |
| HCCL_E_UNAVAIL | 资源不可用，如CCU资源复用异常（复用线程为空、线程数超限）等。 |
| HCCL_E_RUNTIME | 调用底层运行时接口（如内存拷贝、内存清零）失败。 |
| HCCL_E_MEMORY | 内存申请或操作失败。 |

## 约束说明

- 所有rank的recvCount、dataType、op均应相同。
<!-- npu="310p" id8 -->
- 针对Atlas推理系列产品，仅支持单Server场景，单Server中最大支持部署16张推理卡（即32个NPU）。
<!-- end id8 -->
- 算子的输入输出地址（sendBuf与recvBuf）根据不同的数据类型，应满足如下对齐要求：

  - int8、hifloat8按照1Byte地址对齐。
  - int16、float16、bfp16按照2Byte地址对齐。
  - int32、float32按照4Byte地址对齐。
  - int64、uint64、float64按照8Byte地址对齐。
- 多个通信域下的所有通信算子在每个Device上需要保证串行下发，不允许乱序、多线程并发下发，也不支持线程重入。
- 在同一Device上，同一通信域内的所有通信算子的下发线程需要使用相同的Context。
- 当通信域注册了对称内存时，sendBuf指向的输入buffer会被作为对称窗口参与远端rank的直接读写，ReduceScatter过程中部分算法会在sendBuf上就地完成归约（read+reduce），导致sendBuf中的数据被修改（污染）。因此用户需保证：
  - 调用HcclReduceScatter后，sendBuf中的数据不再作为原始输入使用；
  - 如需保留原始输入数据，请在调用前自行备份sendBuf。

## 调用示例

针对Atlas推理系列产品，通信域初始化须在其他涉及Device内存申请的操作之前，参见《[系统约束与限制](../../user_guide/constraints.md#atlas推理系列产品)》。

以下应用层函数展示调用顺序。调用前应完成ACL运行时初始化；`rankId`为通信域内的rank编号，`deviceId`为本机逻辑Device编号。`rootInfo`由rank0调用HcclGetRootInfo生成，经应用的带外通信分发给同一通信域的所有rank后作为参数传入。`rankSize`及各参数应符合本接口的产品与组网约束。

```cpp
#include <cstdint>
#include "hccl/hccl.h"

void ExampleHcclReduceScatter(uint32_t rankSize, uint32_t rankId, uint32_t deviceId, const HcclRootInfo& rootInfo)
{
    // 选择本进程使用的逻辑设备并初始化通信域
    aclrtSetDevice(deviceId);
    HcclComm hcclComm = nullptr;
    HcclCommInitRootInfo(rankSize, &rootInfo, rankId, &hcclComm);

    // 创建任务流
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    uint64_t recvCount = 1; // 每个节点接收的数据数量
    uint64_t sendSize = rankSize * recvCount * sizeof(float);
    uint64_t recvSize = recvCount * sizeof(float);

    // 申请集合通信操作的Device内存
    void *sendBuf = nullptr, *recvBuf = nullptr;
    aclrtMalloc(&sendBuf, sendSize, ACL_MEM_MALLOC_HUGE_ONLY);
    aclrtMalloc(&recvBuf, recvSize, ACL_MEM_MALLOC_HUGE_ONLY);

    // 执行ReduceScatter，将所有rank的sendBuf相加后，再把结果按照rank_id顺序均匀分散到各个rank的recvBuf
    HcclReduceScatter(sendBuf, recvBuf, recvCount, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, hcclComm, stream);
    // 阻塞等待任务流中的集合通信任务执行完成
    aclrtSynchronizeStream(stream);

    // 释放资源
    aclrtFree(sendBuf);         // 释放Device侧内存
    aclrtFree(recvBuf);         // 释放Device侧内存
    aclrtDestroyStream(stream); // 销毁任务流
    HcclCommDestroy(hcclComm);  // 销毁通信域
}
```
