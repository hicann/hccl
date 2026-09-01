# 参数一致性校验(EI0005)

## 问题现象

在打屏日志中存在关键字"The arguments for collective communication are inconsistent between ranks"，如下所示：

```text
EI0005: 2024-04-24-06:32:27.781.599 The arguments for collective communication are inconsistent between ranks:parameter count, local end 16512, remote end 8320
        TraceBack (most recent call last):
        Transport init error. Reason: [Create] [DestLink]Create Dest error! createLink para:rank[5]-localUserrank[4]-localIpAddr[127.10.0.1], dst_rank[6]-remoteUserrank[7]-remote_ip_addr[127.10.0.1]
        Transport init error. Reason: [Create] [DestLink]Create Dest error! createLink para:rank[5]-localUserrank[4]-localIpAddr[127.10.0.1], dst_rank[4]-remoteUserrank[5]-remote_ip_addr[127.10.0.1]
        call hccl op:HcomAllReduce(HcomAllReduce) load task fail[FUNC:Distribute][FILE:hccl_task_info.cc] [LINE:329]
        [[{[node Ge0p3_0]}]]
```

或在CANN日志中存在关键字"CMD information *** check fail"，如下所示：

```text
[ERROR] HCCL(3743927,all_reduce_test):2025-10-25-16:11:16.831.640 [rank_consistentcy_checker.cc:429] [3743951][InitChannelStage][ParameterConflict]CMD information tag check fail. local[AllGather_127.10.0.1%enp_60000_0_1761379874757928], remote[AllReduce_127.10.0.1%enp_60000_0_1761379874757928]
[ERROR] HCCL(3743927,all_reduce_test):2025-10-25-16:11:16.831.666 [rank_consistentcy_checker.cc:439] [3743951][InitChannelStage][ParameterConflict]CMD information cmdType check fail. local[6], remote[2]
[ERROR] HCCL(3743927,all_reduce_test):2025-10-25-16:11:16.831.679 [rank_consistentcy_checker.cc:439] [3743951][InitChannelStage][ParameterConflict]CMD information op check fail. local[255], remote[0]
```

## 可能原因

参数面建链时，在socket建立完成后会进行两端的参数一致性校验，校验的范围包括算子标识符tag、算子类型cmdType、规约类型op、数据量count、HCCL Buffer的大小cclbufferSize、数据类型dataType等，可根据报错里的信息确定不一致的数据。例如下述示例中，两端的算子标识符tag不一致，导致通信算子在建链时一致性校验不通过，local和remote中的数据为两端不一致的数据。

其中参数不一致的两端节点信息可以通过"Transport init error! createLink para:"报错日志确认，比如执行`grep -r "Transport init error! createLink para:"`，查看结果如下：

```text
[ERROR] HCCL(3215542,all_reduce_test):2025-11-20-18:18:03.114.306 [transport_manager.cc:886] [3215599][InitChannelStage][Timeout]Transport init error! createLink para:rank[2]-localUserrank[2]-localIpAddr[127.10.0.1/2], remoteRank[1]-remoteUserrank[1]-remoteIpAddr[127.10.0.1/1], machineType[1], linkMode[1], isUsedRdma[0], tag[AllReduce_127.10.0.1%enp_60000_0_1763633852475745
```

- localUserrank：本端rank编号。
- localIpAddr：本端的节点IP信息。
- remoteUserrank：对端rank编号。
- remoteIpAddr：对端的节点IP信息。
- tag：通信算子标识符。

## 解决方法

1. 如果在未启用SuperKernel时功能正常，但启用了SuperKernel后出现初始化不一致的问题，此时建议将HCCL算子移出SuperKernel的标定范围。具体操作方法可参考《[PyTorch图模式使用指南](https://hiascend.com/document/redirect/pttorchairuseguide)》中的“max-autotune模式功能 > 图内标定SuperKernel范围”章节。
2. 根据报错信息从业务上排查参数校验不一致的两端下发的算子不一致的根因。

    **注意**：日志中部分打印为枚举值，其中cmdType为算子类型，op为规约类型，枚举值对应关系表格如下：

    | cmdType枚举值 | 算子类型 |
    | --- | --- |
    | 1 | BroadCast |
    | 2 | AllReduce |
    | 3 | Reduce |
    | 4 | Send |
    | 5 | Receive |
    | 6 | AllGather |
    | 7 | ReduceScatter |
    | 8 | AlltoAllV |
    | 9 | AlltoAllVC |
    | 10 | AlltoAll |
    | 11 | Gather |
    | 12 | Scatter |
    | 13 | BatchSendRecv |
    | 16 | AllGatherV |
    | 17 | ReduceScatterV |

    op枚举值对应的规约类型如下表所示：

    | op枚举值 | 规约类型 |
    | --- | --- |
    | 0 | SUM |
    | 1 | PROD |
    | 2 | MAX |
    | 3 | MIN |
    | 255 | 非Reduce算子 |
