# Communication Operator Behavior Analysis in Profile Data

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-10T09:33:45.076Z pushedAt=2026-08-10T11:49:57.479Z -->

## Communication Operator Dispatch

Communication operators are dispatched at the CANN layer in profile data. As shown in the figure, one `AscendCL@hcom_allReduce_` corresponds to one AllReduce operator dispatch:

![Communication operator dispatch](figures/comm_op_dispatch.png)

Collective communication operators are orchestrated and dispatched on the host and executed asynchronously on the device. Generally, the dispatch time and asynchronous execution time of communication operators hide each other, allowing full utilization of device resources. When communication operator dispatch becomes a bottleneck, the device must wait for the communication operator to be dispatched, resulting in bubbles. To address the decline in utilization, you need to optimize the dispatch performance of collective communication operators. Common optimization methods include:

- Bind CPU cores to prevent performance loss caused by CPU core switching, since communication operator dispatch latency is affected by CPU scheduling on the host.

- Switch to AIV mode by setting the environment variable **export HCCL_OP_EXPANSION_MODE="AIV"**. Note that AIV mode has limited supported use cases. If multiple communicators execute concurrently, unexpected behaviors such as deadlocks caused by core contention may occur.

## Communication Operator Execution

The execution of communication operators corresponds to the Communication (HCCL) layer in the profile data, as shown in the following figure:

![Communication operator execution](figures/comm_op_execution.png)

- `Group`: the communicator.

- `Plane 0-X`: different communication streams. Each plane corresponds to a communication stream. HCCL communication operator orchestration leverages multi-stream concurrency to fully utilize HCCS physical link resources.

- `hcom_allReduce_xx`: the execution flow of a communication operator. In the detailed information, you can see the latency, data volume, and data type of the communication operator.

Since a communication operator is orchestrated from multiple notify and memcpy tasks, you need to collect at least level 1 profile data to display specific communication task orchestration information in profiling**.**

## Synchronization Tasks

- Notify Record: a task that sets the notify register to 1.

- Notify Wait: a task that waits for the notify register to become 1 and then clears it to 0.

- RDMASend: an inter-node RoCE synchronization task that sets the peer notify register to 1.

For synchronous tasks, you can also obtain the task duration, notify_id, local end (src rank), and peer end (dst rank) from the task details.

![Synchronization tasks](figures/syn_task.png)

## Data Communication Task

- **Memcpy**: a memory copy task for intra-node or intra-chip memory copy.

- **Reduce_Inline**: a memory copy task, which completes on-the-fly reduction computation while copying data.

- **RDMASend**: an inter-node RoCE communication task, which corresponds to the inter-node memory copy task.

For data communication tasks, you can also obtain the task duration, local end (src rank) and peer end (dst rank), data volume (size), bandwidth, and other details from the task details.

![Data communication tasks](figures/data_comm_task.png)

> [!NOTE] Note
>
> - In the profile data, an RDMASend task corresponds to either a synchronization task or a data communication task. You can distinguish between them by analyzing the data volume: the data volume of a synchronization task is fixed at 4 bytes, while the data volume of a data communication task is based on the actual communication volume.
> - If the RDMASend task is a data communication task, its task execution duration is not equal to the actual communication duration. It only represents the time taken to deliver the communication task's WQE to the QP. The actual communication duration can be calculated by referencing the data volume and bandwidth values, or by referencing the duration of the subsequent notify wait task that immediately follows.
