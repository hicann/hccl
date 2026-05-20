# Profiling数据中通信算子行为分析

## 通信算子下发

通信算子的下发在Profiling数据中的CANN层，如图所示，一个“AscendCL@hcom_allReduce_”则对应一次allreduce算子下发：

![通信算子下发](figures/comm_op_dispatch.png)

集合通信算子在host侧编排下发，并在device侧异步执行。一般来说，通信算子的下发时间和异步执行时间互相掩盖，可以充分利用device的资源。当通信算子的下发成为瓶颈时，device侧需等待通信算子下发，此时会出现空泡，针对利用率下降的情况，需要优化集合通信算子的下发性能，常见的优化手段包括：

- 由于通信算子的下发是在host侧执行，因此下发的耗时受host侧CPU的调度影响，可通过CPU绑核的方式防止CPU切核带来的性能损耗，提升通信算子的下发性能。
- 通过环境变量切换为AIV模式，执行**export HCCL\_OP\_EXPANSION\_MODE="AIV"**，但需注意AIV模式的支持场景有限，若业务中存在多个通信域并发执行的场景，会出现互相抢核导致死锁等不可预期的行为。

## 通信算子执行

通信算子的执行对应Profiling数据中的Communication\(HCCL\)层，如下图所示：

![通信算子执行](figures/comm_op_execution.png)

- Group：表示一个通信域。
- Plane0-X：表示不同通信流，每个Plane对应一个通信流，HCCL的通信算子编排会通过多流并发来充分利用HCCS物理链路资源。
- hcom_allReduce_xx：表示通信算子的执行流程，在详细信息中可以看到通信算子的耗时、数据量及数据类型等信息。

由于通信算子由多个notify同步任务及memcpy内存拷贝任务编排而成，若需要在Profiling中显示具体的通信任务编排信息，需至少采集level 1级别的Profiling数据**。**

## 同步任务

- Notify Record：同步任务，置notify寄存器为1。
- Notify Wait：同步任务，等待notify寄存器为1，然后将其清0。
- RDMASend：机间RoCE同步任务，置对端notify寄存器为1。

同时对于同步任务可以从任务的详细信息中获取到任务的耗时、notify_id、本端（src rank）及对端（dst rank）等。

![同步任务](figures/syn_task.png)

## 数据通信任务

- **Memcpy：**内存拷贝任务，机内或者片内的内存拷贝。
- **Reduce\_Inline：**内存拷贝任务，数据拷贝的同时完成随路归约计算。
- **RDMASend：**机间Roce通信任务，对应着机间的内存拷贝任务。

同时对于数据通信任务可以从任务的详细信息中获取到任务的耗时、本端（src rank）及对端（dst rank）、数据量（size）、带宽（bandwidth）等。

![数据通信任务](figures/data_comm_task.png)

> [!NOTE]说明
>
> - 在Profiling数据中RDMASend任务会对应同步或者数据通信任务，可通过其数据量分析区分是同步任务还是数据通信任务，同步任务的数据量为固定的4字节，而数据通信任务的数据量以实际通信量为准。
> - 若RDMASend任务为数据通信任务时，其任务执行的耗时并不等于实际的通信耗时，只是将通信任务的WQE下发到QP队列中的耗时，实际的通信耗时可参考数据量和带宽值计算得到或参考其后续紧跟着的下一个notify wait任务的耗时。
