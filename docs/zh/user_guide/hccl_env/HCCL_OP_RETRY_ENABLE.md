# HCCL_OP_RETRY_ENABLE

## 功能描述

此环境变量用于配置是否开启HCCL算子的重执行特性。HCCL算子重执行以**通信域**为粒度，当通信算子执行报SDMA或者RDMA CQE类型的错误时，HCCL会尝试重新执行此通信算子。

在集群环境中，可能会存在硬件闪断的情况，此时通信算子会执行报错，通过此环境变量开启HCCL的重执行特性，可以更好地避免由于硬件闪断造成的通信中断，提升通信稳定性。HCCL算子重执行实际就是提供一个软件层面的尽力而为的故障恢复手段，执行示意图如下所示：

**图 1**  重执行流程示意图  
![重执行流程示意图](figures/reexec_flow_diagram.png)

主要步骤有如下三步：

1. 故障发现：AI CPU检测到故障信号，通知Host开始准备进入重执行流程。
2. 集群管理：Host通过Host Socket进行信息交互，并判断当前故障算子是否满足一系列重执行条件，详细可参见[重执行使用须知](#重执行使用须知)。
3. 重新下发：通知AI CPU Kernel重新下发SQE、WQE，进行HCCL算子重执行。

## 配置说明

通过此环境变量，开发者可以在Server间、超节点间两个物理层级的通信域中配置是否开启重执行特性，每个层级支持配置两种状态：开启或关闭。

**配置方法如下：**

**export HCCL_OP_RETRY_ENABLE="L1:0,L2:0"**

- L1代表通信域的物理范围为Server间通信域，取值为0表示通信域内Server间通信task不开启重执行，取值为1表示通信域内Server间通信task开启重执行，默认值为0。
- L2代表通信域的物理范围为超节点间通信域，取值为0表示通信域内超节点间通信task不开启重执行，取值为1表示通信域内超节点间通信task开启重执行，默认值为0。

    L2配置为“1”时，超节点间通信若某一Device网卡故障，重执行时会使用备用Device网卡进行通信，称之为“借轨通信”。备用网卡为属于同一NPU中的另一个Die网卡，借轨通信正常执行的条件以及借轨通信的影响详细可参见[借轨通信使用须知](#借轨通信使用须知)。

  - 如果通信域的创建方式为“基于rank table创建通信域”，需要开发者在rank table文件中通过“backup_device_ip”参数配置备用网卡。
  - 如果通信域的创建方式为“基于root节点信息创建通信域”，会自动将同一NPU下的两个Die互为备用网卡，无需开发者手工配置。

另外，开发者可以通过环境变量[HCCL_OP_RETRY_PARAMS](HCCL_OP_RETRY_PARAMS.md)配置第一次重执行的等待时间、最大重执行的次数以及两次重执行的间隔时间。

**配置建议：**

- 开启重执行特性后会有一定的性能损失，针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，Server间与超节点间，会经过光互联域，稳定性较低，建议开启HCCL重执行。
- 此环境变量在各个超节点上的配置需要保持一致，否则超节点间建链会超时。

## 重执行使用须知

开启HCCL重执行特性时，需要满足以下约束条件，约束条件不满足时重执行会失败。

1. 通信算子的展开模式为AI_CPU。重执行特性仅在AI_CPU调度模式下使能，需要通过环境变量[HCCL_OP_EXPANSION_MODE](HCCL_OP_EXPANSION_MODE.md)设置，非AI_CPU调度模式下会采用无重执行流程。

    ```bash
    export HCCL_OP_EXPANSION_MODE="AI_CPU"
    ```

2. 基于rank table创建通信域的场景下，rank table中“host_ip”字段必须配置，否则重执行不生效，走无重执行流程。
3. 通信算子的输入内存在执行过程中不能存在被污染的风险。

    一个集合通信算子是一系列任务的组合，HCCL重执行以通信算子为粒度，从算子的输入内存开始将一个通信算子的系列任务重新执行一遍。若通信算子的输入内存在执行过程中存在被污染的风险，可能会造成重执行失败、系统报错退出。

    输入内存存在被污染风险的场景主要有以下几种：

    - 开启零拷贝功能的场景：零拷贝功能开启后，ReduceScatter和AllReduce算子会修改用户的输入内存，因此这两类算子无法支持重执行。
    - 包含In-Place操作的场景：此场景下，算子的输入和输出共享同一块内存，例如PyTorch的[ReduceScatter](https://pytorch.org/docs/stable/distributed.html#torch.distributed.reduce_scatter)/[AllGather](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_gather)算子，所以包含In-Place操作的场景也不支持重执行。
    - 图模式场景：在图模式下，通信可以直接在算子的输入输出上进行，例如PyTorch的[AllReduce](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce)算子，其入参tensor即作为算子的输入，又作为算子的输出，在算子的通信过程中，部分结果写入后，tensor内容就会发生变化，如果在污染的input上重新执行一遍，会得到错误的计算结果。所以此场景也不支持重执行。

4. 故障发生时通信域内所有rank都停在同一个通信算子。若不同的rank停在不同的通信算子上，则不能重执行。

    故障发生的时刻是不可预测的，发生故障时，整个通信域各rank处于什么状态与重执行的成功率相关。以下图的通信域为例，包含三个rank，[表1](#table1)分别列举故障发生在不同时刻的重执行情况。

    **图 2**  通信域故障示意图1  
    ![通信域故障示意图1](figures/comm_domain_fault_1.png)

    **表 1**  通信域故障重执行情况<a id="table1"></a>

    | 故障发生时刻 | 是否可重执行 | 重执行的算子 |
    | --- | --- | --- |
    | A | 是 | HCCL OP1。<br>由于计算算子感知不到链路故障，直到执行到通信算子HCCL OP1时感知到链路故障，此时三个rank都停止在HCCL OP1，满足重执行条件并启动重执行。 |
    | B | 是 | HCCL OP1。<br>rank0与rank2继续执行直到通信算子HCCL OP1，rank1也停止在HCCL OP1，满足重执行条件并启动重执行。 |
    | C | 是 | HCCL OP1。 |
    | D | 否 | rank0和rank1的HCCL OP1已执行完成，D时刻故障发生时，会继续执行到HCCL OP2，而rank2仍然停在HCCL OP1，不满足重执行条件。 |
    | E | 是 | HCCL OP3。<br>三个rank都继续执行，最终停止在HCCL OP3，满足重执行条件并启动重执行。 |

    下面说明下集合通信为什么不能完全保证故障发生时停止在同一个通信算子，以集合通信常见的算法RHD（Recursive Halving-Doubling）为例。

    **图 3**  通信域故障示意图2  
    ![通信域故障示意图2](figures/comm_domain_fault_2.png)

    四个AI Server，每个AI Server一个rank组成四个rank的通信域，假设故障刚好发生在HD算法的第一步数据交换后，则会出现以下情况：

    rank2和rank3能正常运行结束，而rank0和rank1无法运行结束。rank2和rank3后续的计算算子或通信算子，都有可能使用任意内存，并且再次执行时在rank2、rank3上无法找到对应的上下文信息。所以如果故障发生时刻如上图，则不能进行重执行。

5. 判断Host侧socket网络通信是否正常。重执行时会使用Host侧socket通信进行通信域中各卡状态的协商，如果socket网络故障，则无法进行重执行。
6. 确保故障的链路已恢复，例如路由收敛成功，光模块闪断恢复成功或者借用备用网卡通信成功等。如果故障的链路无法恢复，再次执行通信任务仍然会失败，当重执行次数超过设置的最大重传次数后（可参见环境变量[HCCL_OP_RETRY_PARAMS](HCCL_OP_RETRY_PARAMS.md)），重执行失败。

> [!NOTE]说明
>
> - 若Host侧调试日志中出现关键字为“\[OpRetry\]...timeout”的ERROR错误信息，说明HCCL重执行过程中Host侧socket通信异常，此时可以搜集通信域内所有节点的日志，进一步定位故障发生位置。
> - 若Host侧调试日志中出现关键字为“can not retry”的ERROR错误信息，说明当前场景不满足HCCL重执行条件。
> Host侧应用程序产生的调试日志默认存储路径为：$HOME/ascend/log/debug/plog/。

## 借轨通信使用须知

1. 为了确保借轨通信功能正常执行，需要满足如下条件：
    - 备用网卡的通信链路正常。
    - 互为主备的Device均需在业务可见范围内。

        例如，NPU1中包含Device0与Device1两个Die，互为主备，假设通过环境变量ASCEND_RT_VISIBLE_DEVICES指定了业务可见Device为Device0，Device1不对业务可见，则借轨功能无法执行。

2. 如果通信过程中发生了借轨（假设某个NPU的Die0网卡故障，启用了备用的Die1网卡），原Die0的网卡流量也会通过Die1网卡收发，导致Die1的流量增大，总体性能会由于物理带宽减半、端口冲突导致下降。
3. 开启借轨通信场景下，若NPU0的Die0网卡故障，会切换到其备用网卡Die1，由于两个NPU之间的通信要求本端与对端同时切换为备用网卡，因此NPU1也会从Die0切换为Die1，即“图示二”所示。但如果Die0与Die1之间本身就存在通信任务，此时借轨功能无法执行。

    **图 4**  借轨通信切换示例  
    ![借轨通信切换示例](figures/borrow_comm_switch_example.png)

4. 开启借轨通信功能时，建议一个NPU的两个Die分配给同一个训练或推理任务。

    如果同一个NPU的两个Die分给两个不同的训练或推理任务，一个任务发生故障，会借用另一个任务的网卡，两个任务均会发生一定程度的性能下降。

5. 同一NPU仅支持发生一次借轨，且不支持回切。

    如[图5](#figure5)所示，“图示一”中NPU0 -\> NPU1间通信链路故障，启用了备用链路，发生借轨，通信正常进行；若再发生“图示二”所示故障，则不再支持借轨，会报错退出。

    **图 5**  同一NPU仅支持一次借轨示例<a id="figure5"></a>  
    ![同一NPU仅支持一次借轨示例](figures/npu_single_borrow_example.png)

## 异常处理

若开启重执行特性后，出现"\[OpRetryConnection\]\[RecvAckTag\] Recv unmatched ack"的错误，可能是由于HCCL通信时使用的默认端口被占用，导致HCCL连接了错误的Server，解决方法如下：

1. 使用“sysctl -w net.ipv4.ip_local_reserved_ports”命令预留HCCL使用的默认端口60000-60015，避免端口被操作系统随机分配。

    ```bash
    sysctl -w net.ipv4.ip_local_reserved_ports=60000-60015
    ```

2. 如果前一种方法仍出现该错误，那么建议使用[HCCL_IF_BASE_PORT](HCCL_IF_BASE_PORT.md)环境变量修改HCCL使用的默认端口，同时使用“sysctl -w net.ipv4.ip_local_reserved_ports”命令预留指定的端口。

    ```bash
    # 例如指定HCCL使用以17777端口开始的连续16个端口
    export HCCL_IF_BASE_PORT=17777
    # 预留17777-17792共16个端口
    sysctl -w net.ipv4.ip_local_reserved_ports=17777-17792
    ```

## 其他约束

若您调用HCCL C接口初始化具有特定配置的通信域时，通过“HcclCommConfig”的“hcclRetryEnable”参数配置了是否开启HCCL算子的重执行特性，则以通信域粒度的配置优先。

## 重执行对整网性能说明

请参见[通信算子重执行对整网性能说明](comm_retry_perf_impact.md)。

## 支持的型号

Atlas A3 训练系列产品/Atlas A3 推理系列产品
