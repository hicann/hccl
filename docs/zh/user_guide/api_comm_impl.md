# 使用通信库API实现通信功能

## 简介

HCCL提供了C与Python两种语言的开发接口，用于实现分布式能力。

- C语言接口用于实现单算子模式下的框架适配，实现分布式能力。
- Python语言接口用于实现图模式下的框架适配，当前仅用于实现TensorFlow网络在AI处理器执行分布式优化。

**本章节针对如何调用HCCL的C语言接口开发集合通信功能进行介绍。**

开发者调用HCCL C语言接口实现集合通信功能的主要开发流程如下所示。

![集合通信操作流程](figures/hccl_operation_flow.png "集合通信操作流程")

1. 首先进行集群信息配置，创建通信域句柄，并初始化HCCL通信域。
2. 实现通信操作，HCCL通信操作包含两大类：点对点通信与集合通信。
    - 点对点通信，指多NPU环境下两个NPU之间直接传输数据的过程，常用于pipeline并行场景下对激活值的数据收发。HCCL提供了不同粒度的点对点通信，包括单rank到单rank的单收单发接口，以及多个rank之间的批量收发接口。
    - 集合通信，指多个NPU共同参与进行数据传输操作，例如AllReduce，AllGather，Broadcast等，常用于大规模集群中不同NPU之间的梯度同步和参数更新等操作。集合通信操作可让所有计算节点并行、高效、有序执行数据交换，提升数据传输效率。

3. 集合通信操作完成后，需要销毁通信域，释放相关内存、流资源。

## 通信域管理

通信域是集合通信算子执行的上下文，管理对应的通信对象（例如一个NPU就是一个通信对象）和通信所需的资源。通信域中的每个通信对象称为一个rank，每个rank都会分配一个介于0~n-1（n为NPU的数量）的唯一标识。

通信域创建根据用户场景的不同主要有以下几种方式：

- 多机集合通信场景
  - 如果有完整的描述集群信息的rank table文件，可通过HcclCommInitClusterInfo接口创建通信域，或者通过HcclCommInitClusterInfoConfig接口创建具有特定配置的通信域。
  - 如果无完整的rank table文件，可通过HcclGetRootInfo接口与HcclCommInitRootInfo/HcclCommInitRootInfoConfig接口配合使用，基于root节点信息创建通信域。

- 单机集合通信场景，可通过HcclCommInitAll接口在单机内批量创建通信域。
- 基于已有的通信域，可通过HcclCreateSubCommConfig接口切分具有特定配置的子通信域。

> [!NOTE]说明
>
> - 多个通信域下的所有通信算子在每个Device上需要保证串行下发，不允许乱序、多线程并发下发，也不支持线程重入。
> - 在同一Device上，同一通信域内的所有通信算子的下发线程需要使用相同的Context。
> - 同一个通信域内不支持图模式通信和单算子通信混合执行。
> - 同一个通信域内的算子需要由使用者确保串行执行。
> - 同一个NPU上需要串行创建多个通信域。
<cann-filter npu-type="A3">> - 针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，通信域初始化时，如果组网中存在多个超节点，请将属于同一超节点内的AI Server信息配置在一起。假设有两个超节点，标识分别为“0”和“1”，请先配置“0”中的AI Server信息，再配置“1”中的AI Server信息，不支持“0”中的AI Server信息与“1”中的AI Server信息交叉配置。</cann-filter>

### 基于rank table创建通信域

多机集合通信、基于集群信息配置文件（rank table文件）创建通信域的场景，每张卡需要使用一个单独的进程参考如下流程创建通信域：

1. 构造rank table文件（rank table文件的配置可参见[集群信息配置](./cluster_info_config/README.md)）。
2. 每张卡分别调用[HcclCommInitClusterInfo](https://gitcode.com/cann/hcomm/blob/master/docs/api_ref/comm_mgr_c/HcclCommInitClusterInfo.md)接口创建通信域，或者调用[HcclCommInitClusterInfoConfig](https://gitcode.com/cann/hcomm/blob/master/docs/api_ref/comm_mgr_c/HcclCommInitClusterInfoConfig.md)接口创建具有特定配置的通信域。

一个简单的代码示例片段如下：

```c
    int devId = 0;
    // 配置rank table文件路径
    char* rankTableFile = "/home/rank_table.json";
    // 定义通信域句柄
    HcclComm hcclComm;
    // 初始化HCCL通信域
    HcclCommInitClusterInfo(rankTableFile, devId, &hcclComm);
   
    /*  集合通信操作   */

    // 销毁HCCL通信域
    HcclCommDestroy(hcclComm);
```

> [!NOTE]说明
> 针对Ascend 950PR/Ascend 950DT，Atlas A3 训练系列产品/Atlas A3 推理系列产品，Atlas A2 训练系列产品/Atlas A2 推理系列产品，若业务为单卡多进程场景，建议在rank table配置文件中配置“device_port”字段，并且不同的业务进程需要设置不同的端口号，否则业务可能会因为端口冲突运行失败。但需要注意，多进程会对资源开销、通信性能产生一定的影响。

### 基于root节点信息创建通信域

多机集合通信场景，若无完整的集群信息配置文件（rank table文件），HCCL提供了基于root节点信息创建通信域的方式，**主要有如下两种典型使用场景**：

- 每个Device对应一个业务进程的场景，实现流程如下所示：
    1. 针对Ascend 950PR/Ascend 950DT，检查rootinfo文件是否存在，其他产品跳过此步骤。

        基于root节点信息创建通信域前，请检查“/etc/hccl_rootInfo.json”文件是否存在，此文件记录了NPU间通信的EID（Entity ID，通信中发起或接收对象的标识）信息，环境部署完成后自动生成。若无此文件，请在当前源码仓提issue。

    2. 指定HCCL初始化时Host节点使用的通信IP地址或通信网卡（可选）。<a id="set_host_nic"></a>

        - 方式一：在每个Host节点通过环境变量[HCCL_IF_IP](./hccl_env/HCCL_IF_IP.md)配置通信IP地址，该IP地址用于与root节点通信，可以是IPv4或IPv6格式，仅支持配置一个IP地址。配置示例如下：

            ```bash
            export HCCL_IF_IP=10.10.10.1
            ```

        - 方式二：在每个Host节点通过环境变量[HCCL_SOCKET_IFNAME](./hccl_env/HCCL_SOCKET_IFNAME.md)配置通信网卡名，通过[HCCL_SOCKET_FAMILY](./hccl_env/HCCL_SOCKET_FAMILY.md)配置网卡使用的通信协议，HCCL将通过该网卡名获取Host IP，与root节点通信。配置示例如下：

            ```bash
            # 配置HCCL初始化时通信网卡使用的IP协议版本，AF_INET：IPv4；AF_INET6：IPv6
            export HCCL_SOCKET_FAMILY=AF_INET
            
            # 支持以下格式的网卡名配置（4种规格自行选择1种即可，环境变量中可配置多个网卡，多个网卡间使用英文逗号分隔，取最先匹配到的网卡作为通信网卡）
            # 精确匹配网卡
            export HCCL_SOCKET_IFNAME==eth0,enp0   # 使用指定的eth0或enp0网卡
            export HCCL_SOCKET_IFNAME=^=eth0,enp0     # 不使用eth0与enp0网卡
            
            # 模糊匹配网卡
            export HCCL_SOCKET_IFNAME=eth,enp       # 使用所有以eth或enp为前缀的网卡
            export HCCL_SOCKET_IFNAME=^eth,enp      # 不使用任何以eth或enp为前缀的网卡
            ```

        环境变量HCCL_IF_IP的优先级高于HCCL_SOCKET_IFNAME。如果不配置HCCL_IF_IP或HCCL_SOCKET_IFNAME，系统将按照如下优先级自动选择网卡。若当前节点选择的网卡与root节点选择的网卡链路不通，将导致HCCL建链失败。

        ```text
        docker/lo以外网卡(网卡名称的字典序升序) > docker 网卡 > lo网卡
        ```

    3. 在root节点调用[HcclGetRootInfo](https://gitcode.com/cann/hcomm/blob/master/docs/api_ref/comm_mgr_c/HcclGetRootInfo.md)接口，生成root节点rank标识信息“rootInfo”，包括device ip、device id等信息。
    4. 将root节点的rank信息广播至通信域中的所有rank。
    5. 在通信域中所有节点调用[HcclCommInitRootInfo](https://gitcode.com/cann/hcomm/blob/master/docs/api_ref/comm_mgr_c/HcclCommInitRootInfo.md)或者[HcclCommInitRootInfoConfig](https://gitcode.com/cann/hcomm/blob/master/docs/api_ref/comm_mgr_c/HcclCommInitRootInfoConfig.md)接口（创建具有特定配置的通信域），基于接收到的“rootInfo”，以及本rank的rank id等信息，进行通信域初始化。

- 每个AI Server对应一个业务进程，每个线程对应一个Device，通过多线程的方式创建多个通信域的场景，实现流程如下所示：
    1. 针对Ascend 950PR/Ascend 950DT，检查rootinfo文件是否存在，其他产品跳过此步骤。

        基于root节点信息创建通信域前，请检查“/etc/hccl_rootInfo.json”文件是否存在，此文件记录了NPU间通信的EID（Entity ID，通信中发起或接收对象的标识）信息，环境部署完成后自动生成。若无此文件，请在当前代码仓提issue。

    2. 参见[“每个Device对应一个业务进程”场景的步骤2](#set_host_nic)，指定HCCL初始化时Host节点使用的通信IP地址或通信网卡（可选）。
    3. 在主进程中循环执行“指定不同的Device  + 调用HcclGetRootInfo接口”，获取多个“rootInfo”信息。
    4. 每个Device匹配一个线程，分别根据不同的“rootInfo”信息，并发调用[HcclCommInitRootInfo](https://gitcode.com/cann/hcomm/blob/master/docs/api_ref/comm_mgr_c/HcclCommInitRootInfo.md)或者[HcclCommInitRootInfoConfig](https://gitcode.com/cann/hcomm/blob/master/docs/api_ref/comm_mgr_c/HcclCommInitRootInfoConfig.md)接口，进行通信域初始化。

> [!NOTE]说明
> 针对Ascend 950PR/Ascend 950DT，Atlas A3 训练系列产品/Atlas A3 推理系列产品，Atlas A2 训练系列产品/Atlas A2 推理系列产品，若业务为单卡多进程场景，建议通过环境变量“[HCCL_HOST_SOCKET_PORT_RANGE](./hccl_env/HCCL_HOST_SOCKET_PORT_RANGE.md)”与“[HCCL_NPU_SOCKET_PORT_RANGE](./hccl_env/HCCL_NPU_SOCKET_PORT_RANGE.md)”分别配置HCCL在Host侧与NPU侧使用的通信端口，否则可能会导致端口冲突，配置示例如下所示。但需要注意，多进程会对资源开销、通信性能产生一定的影响。
>
> ```bash
> export HCCL_HOST_SOCKET_PORT_RANGE="auto"
> export HCCL_NPU_SOCKET_PORT_RANGE="auto"
> ```

### 单机内批量创建通信域

单机通信场景中，开发者可通过一个进程统一创建多张卡的通信域，其中一张卡对应一个线程，创建流程如下：

1. 构造通信域中的Device列表，例如：\{0, 1, 2, 3, 4, 5, 6, 7\}，其中列表中的Device ID是逻辑ID（可通过**npu-smi info -m**命令查询），HCCL会按照列表中设置的顺序创建通信域。
2. 在进程中调用[HcclCommInitAll](https://gitcode.com/cann/hcomm/blob/master/docs/api_ref/comm_mgr_c/HcclCommInitAll.md)接口创建通信域。

```c
    uint32_t ndev = 8;
    // 构造Device的逻辑ID列表
    int32_t devices[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    // 定义通信域句柄
    HcclComm comms[ndev];
    // 初始化HCCL通信域
    HcclCommInitAll(ndev, devices, comms);

    // 启动线程执行集合通信操作
    std::vector<std::unique_ptr<std::thread> > threads(ndev);
    struct ThreadContext args[ndev];
    for (uint32_t i = 0; i < ndev; i++) {
        args[i].device = i;
        args[i].comm = comms[i];
       /*  集合通信操作   */      
    }

    // 销毁HCCL通信域
    for (uint32_t i = 0; i < ndev; i++) {
        HcclCommDestroy(comms[i]);
    }
```

需要注意，多线程调用集合通信操作API时（例如HcclAllReduce），需要确保不同线程中调用集合通信操作API的前后时间差不超过集合通信的建链超时等待时间（可通过环境变量[HCCL_CONNECT_TIMEOUT](./hccl_env/HCCL_CONNECT_TIMEOUT.md)设置，默认120s），避免建链超时。

### 基于已有通信域切分子通信域

HCCL提供了[HcclCreateSubCommConfig](https://gitcode.com/cann/hcomm/blob/master/docs/api_ref/comm_mgr_c/HcclCreateSubCommConfig.md)接口，实现基于已有通信域切分具有特定配置的子通信域的功能。该子通信域创建方式无需进行socket建链与rank信息交换，可应用于业务故障下的快速通信域创建。

```c
// 初始化全局通信域
HcclComm globalHcclComm;
HcclCommInitClusterInfo(rankTableFile, devId, &globalHcclComm);
// 通信域配置
HcclCommConfig config;
HcclCommConfigInit(&config);
config.hcclBufferSize = 50;
strcpy(config.hcclCommName, "comm_1");
// 初始化子通信域
HcclComm hcclComm;
uint32_t rankIds[4] = {0, 1, 2, 3};  // 子通信域的 Rank 列表
HcclCreateSubCommConfig(&globalHcclComm, 4, rankIds, 1, devId, &config, &hcclComm);
```

> [!NOTE]说明
> 该接口不支持通信域的嵌套切分，即不支持在子通信域中进一步切分子通信域。

### 销毁通信域

集合通信操作完成后，需要调用[HcclCommDestroy](https://gitcode.com/cann/hcomm/blob/master/docs/api_ref/comm_mgr_c/HcclCommDestroy.md)接口销毁指定的通信域，并调用运行时管理接口释放通信所用的内存、Stream、Device资源。

## 集合通信

集合通信是指多个NPU共同参与进行数据传输，从而形成一次集体操作的通信模式，常用于大规模集群中不同NPU之间的梯度同步和参数更新等场景。

HCCL支持AllReduce、Broadcast、AllGather、Scatter、ReduceScatter、Reduce、AlltoAll和AlltoAllV等通信算子，并提供了对应的API供开发者调用，用于快速实现集合通信能力。

### Broadcast

Broadcast操作是将通信域内root节点的数据广播到其他rank。

![Broadcast](figures/broadcast.png)

注意：通信域内只能有一个root节点。

相关接口：[HcclBroadcast](../api_ref/comm_op_interface/HcclBroadcast.md)。

### Scatter

Scatter操作是将通信域内root节点的数据均分并散布至其他rank。

![Scatter](figures/scatter.png)

注意：通信域内只能有一个root节点。

相关接口：[HcclScatter](../api_ref/comm_op_interface/HcclScatter.md)。

### AllGather

AllGather操作是将通信域内所有节点的输入按照rank id重新排序（rank id按照从小到大的顺序排序），然后拼接起来，再将结果发送到所有节点的输出buffer。

![AllGather](figures/allgather.png)

> [!NOTE]说明
> 针对AllGather操作，每个节点都接收按照rank id重新排序后的数据集合，即每个节点的AllGather输出都是一样的。

相关接口：[HcclAllGather](../api_ref/comm_op_interface/HcclAllGather.md)。

### AllGatherV

AllGatherV操作是将通信域内所有节点的输入按照rank id重新排序（rank id按照从小到大的顺序排序），然后拼接起来，再将结果发送到所有节点的输出。与AllGather操作不同的是，AllGatherV操作支持通信域内不同节点的输入配置不同大小的数据量。

![AllGatherV](figures/allgatherv.png)

> [!NOTE]说明
> 针对AllGatherV操作，每个节点都接收按照rank id重新排序后的数据集合，即每个节点的AllGatherV输出都是一样的。

相关接口：[HcclAllGatherV](../api_ref/comm_op_interface/HcclAllGatherV.md)。

### Reduce

Reduce操作是将通信域内所有rank的输入数据进行归约操作后（支持sum、prod、max、min），再把结果发送到root节点的输出buffer。

![Reduce](figures/reduce.png)

注意：通信域内只能有一个root节点。

相关接口：[HcclReduce](../api_ref/comm_op_interface/HcclReduce.md)。

### AllReduce

AllReduce操作是将通信域内所有节点的输入数据进行归约操作后（支持sum、prod、max、min），再把结果发送到所有节点的输出buffer。

![AllReduce](figures/allreduce.png)

注意：每个rank只能有一个输入。

相关接口：[HcclAllReduce](../api_ref/comm_op_interface/HcclAllReduce.md)。

### ReduceScatter

ReduceScatter操作是将通信域内所有rank的输入数据均分成rank size份，然后分别取每个rank的rank size之一份数据进行归约操作（如sum、prod、max、min）。最后，将结果按照编号分散到各个rank的输出buffer。

![ReduceScatter](figures/reduce_scatter.png)

相关接口：[HcclReduceScatter](../api_ref/comm_op_interface/HcclReduceScatter.md)。

### ReduceScatterV

ReduceScatterV操作与ReduceScatter操作类似，不同点是支持为通信域内不同的节点配置不同大小的数据量（同一rank不同编号的数据大小可设置，但不同rank间相同编号的数据大小需保持一致），取每个rank对应编号的数据进行归约操作后（如sum、prod、max、min）后，再把结果按照编号分散到各个rank的输出buffer。

![ReduceScatterV](figures/reduce_scatterv.png)

相关接口：[HcclReduceScatterV](../api_ref/comm_op_interface/HcclReduceScatterV.md)。

### AlltoAll

AlltoAll操作是向通信域内所有rank发送相同数据量的数据，并从所有rank接收相同数据量的数据。

![AlltoAll](figures/alltoall.png)

AlltoAll操作将输入数据在特定的维度切分成特定的块数，并按顺序发送给其他rank，同时从其他rank接收输入数据，按顺序在特定的维度拼接数据。

相关接口：[HcclAlltoAll](../api_ref/comm_op_interface/HcclAlltoAll.md)。

### AlltoAllV

AlltoAllV操作是向通信域内所有rank发送数据（数据量可以定制），并从所有rank接收数据。

![AlltoAllV](figures/alltoallv.png)

相关接口：[HcclAlltoAllV](../api_ref/comm_op_interface/HcclAlltoAllV.md)。

### AlltoAllVC

AlltoAllVC操作是向通信域内所有rank发送数据（数据量可以定制），并从所有rank接收数据。相比于AlltoAllV，AlltoAllVC通过输入参数sendCountMatrix传入所有rank的收发参数。

![AlltoAllVC](figures/alltoallvc.png)

相关接口：[HcclAlltoAllVC](../api_ref/comm_op_interface/HcclAlltoAllVC.md)。

### 接口调用

下面以[HcclAllReduce](../api_ref/comm_op_interface/HcclAllReduce.md)接口为例，介绍其使用示例，HcclAllReduce接口的原型定义如下：

```c
HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream)
```

HcclAllReduce用于将通信域内所有节点的输入进行归约操作，再把结果发送到所有节点的输出，其中op参数用于指定归约的操作类型。HcclAllReduce允许每个节点只有一个输入。

如下代码片段所示，将通信域内所有输入内存中的数据，按照float32的数据格式执行加法操作（示例中每个rank中只有一个数据参与），然后把相加结果发送到所有节点的输出内存。

```c
void* hostBuf = nullptr;
void* sendBuf = nullptr;
void* recvBuf = nullptr;
uint64_t count = 1;
int malloc_kSize = count * sizeof(float);
aclrtStream stream;
aclrtCreateStream(&stream);

//申请集合通信操作的内存 
aclrtMalloc((void**)&sendBuf, malloc_kSize, ACL_MEM_MALLOC_HUGE_ONLY); 
aclrtMalloc((void**)&recvBuf, malloc_kSize, ACL_MEM_MALLOC_HUGE_ONLY);

//初始化输入内存
aclrtMallocHost((void**)&hostBuf, malloc_kSize);
aclrtMemcpy((void*)sendBuf, malloc_kSize, (void*)hostBuf, malloc_kSize, ACL_MEMCPY_HOST_TO_DEVICE);

//执行集合通信操作
HcclAllReduce((void *)sendBuf, (void*)recvBuf, count, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, hcclComm, stream);
```

HcclAllReduce接口调用的完整示例可参见[示例代码](#示例代码)章节不同通信域初始化方式中的“HcclAllReduce操作代码样例”。

## 点对点通信

点对点通信是指多NPU环境下两个NPU之间直接传输数据的通信模式，常用于pipeline并行场景下对激活值的数据收发。

HCCL提供了不同粒度的点对点通信算子，包括单rank到单rank的单发单收算子（Send/Receive），以及多个rank之间的批量收发算子（BatchSendRecv），HCCL提供了对应的接口供开发者调用。

### Send/Receive（单发/单收）

- Send：将一个rank的数据发送到另外一个rank。
- Receive：接收从另一个rank上发送来的数据。

HCCL提供了对应的接口[HcclSend](../api_ref/comm_op_interface/HcclSend.md)、[HcclRecv](../api_ref/comm_op_interface/HcclRecv.md)用于单收单发场景，需严格保序下发并配对使用，收发两端需完成同步后才能进行数据收发，数据收发完成后才能执行后续算子任务。

![SendRecv](figures/send_recv.png)

一个简单的代码示例片段如下：

```c
if(rankId == 0){
    uint32_t destRank = 1;
    uint32_t srcRank = 1;
    HcclSend(sendBuf, count, dataType, destRank, hcclComm, stream);
    HcclRecv(recvBuf, count, dataType, srcRank, hcclComm, stream);
}
if(rankId == 1){
    uint32_t srcRank = 0;
    uint32_t destRank = 0;
    HcclRecv(recvBuf, count, dataType, srcRank, hcclComm, stream);
    HcclSend(sendBuf, count, dataType, destRank, hcclComm, stream);
}
```

### BatchSendRecv（批量收发）

HCCL提供了[HcclBatchSendRecv](../api_ref/comm_op_interface/HcclBatchSendRecv.md)接口用于通信域内多个rank之间的数据收发，该接口有两个特征：

- 接口内部会对批量数据收发顺序进行重排，所以不严格要求单次接口调用中批量收发的任务顺序，但需要确保一次接口调用中的数据发送与数据接收操作个数完全匹配。
- 收发过程独立调度执行，收发不相互阻塞，从而实现双工链路并发。

**该接口使用时需要注意：**单次接口调用下，两个rank之间单向数据流仅支持传递一块内存数据，避免收发过程中混淆多块内存数据的收发地址。

一个简单的代码示例片段如下：

```c
HcclSendRecvItem sendRecvInfo[itemNum];
HcclSendRecvType currType;
for (size_t i = 0; i < op_type.size(); ++i) {
    if (op_type[i] == "isend") {
        currType = HcclSendRecvType::HCCL_SEND;
    } else if (op_type[i] == "irecv") {
        currType = HcclSendRecvType::HCCL_RECV;
    } 
    sendRecvInfo[i] = HcclSendRecvItem{currType,
                                       tensor_ptr_list[i],
                                       count_list[i],
                                       type_list[i],
                                       remote_rank_list[i]
                                       };
}
HcclBatchSendRecv(sendRecvInfo, itemNum, hcclComm, stream);
```

## 示例代码

HCCL提供了不同场景下使用通信库API实现集合通信功能的代码示例，开发者可根据实际需求选择参考。

### 通信域管理代码示例

- [每个进程管理一个NPU设备（基于root节点信息初始化通信域）](https://gitcode.com/cann/hcomm/tree/master/examples/01_communicators/01_one_device_per_process/)
- [每个进程管理一个NPU设备（基于rank table初始化通信域）](https://gitcode.com/cann/hcomm/tree/master/examples/01_communicators/02_one_device_per_process_rank_table/)
- [每个线程管理一个NPU设备](https://gitcode.com/cann/hcomm/tree/master/examples/01_communicators/03_one_device_per_pthread/)

### 点对点通信代码示例

- [HcclSend/HcclRecv（基础收发功能）](https://gitcode.com/cann/hccl/tree/master/examples/01_point_to_point/01_send_recv/)
- [HcclBatchSendRecv（实现Ring环状通信）](https://gitcode.com/cann/hccl/tree/master/examples/01_point_to_point/02_batch_send_recv_ring)

### 集合通信代码示例

- [AllReduce](../../../examples/02_collectives/01_allreduce)
- [Broadcast](../../../examples/02_collectives/02_broadcast)
- [AllGather](../../../examples/02_collectives/03_allgather)
- [ReduceScatter](../../../examples/02_collectives/04_reduce_scatter)
- [Reduce](../../../examples/02_collectives/05_reduce)
- [AlltoAll](../../../examples/02_collectives/06_alltoall)
- [AlltoAllV](../../../examples/02_collectives/07_alltoallv)
- [AlltoAllVC](../../../examples/02_collectives/08_alltoallvc)
- [Scatter](../../../examples/02_collectives/09_scatter)
