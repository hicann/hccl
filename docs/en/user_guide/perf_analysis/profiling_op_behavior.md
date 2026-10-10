# Analyzing Communication Operator Behavior from Profile Data

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-12T11:04:40.681Z pushedAt=2026-09-18T08:03:58.288Z -->

## Communication Operator Dispatch

Communication operators are dispatched at the CANN layer in the profile data. As shown in the figure, one `AscendCL@OpType::ALLGATHER` corresponds to one AllGather operator dispatch:

![Communication operator dispatch](figures/comm_op_dispatch.png)

Collective communication operators are orchestrated and dispatched on the host and executed asynchronously on the device. Generally, the dispatch time and asynchronous execution time of communication operators overlap with each other, allowing full utilization of device resources. When communication operator dispatch becomes a bottleneck, the device must wait for the communication operator to be dispatched, resulting in bubbles. To address the decline in utilization, you need to optimize the dispatch performance of collective communication operators. Common optimization methods include:

- Bind CPU cores to prevent performance loss caused by CPU core switching, since communication operator dispatch latency is affected by CPU scheduling on the host.
- Switch to AIV mode by setting the environment variable **export HCCL_OP_EXPANSION_MODE="AIV"**. Note that AIV mode has limited support use cases. If multiple communicators execute concurrently, unexpected behaviors such as deadlocks caused by core contention may occur.

## Communication Operator Execution

The execution of communication operators corresponds to the Communication (HCCL) layer in the profile data, as shown in the following figure:

![Communication operator execution](figures/comm_op_execution.png)

- `Group`: the communicator.
- `Plane 0-X`: different communication streams. Each plane corresponds to a communication stream. HCCL communication operator orchestration leverages multi-stream concurrency to fully utilize HCCS physical link resources.
- `hcom_allReduce_xx`: the execution flow of a communication operator. In the detailed information, you can see the latency, data volume, and data type of the communication operator.

Since a communication operator is orchestrated from multiple notify and memcpy tasks, you need to collect at least level 1 profile data to display specific communication task orchestration information in profiling.**

## Synchronization Tasks

- Notify Record: a task that sets the notify register to 1.
- Notify Wait: a task that waits for the notify register to become 1 and then clears it to 0.
- RDMASend: an inter-node RoCE synchronization task that sets the peer notify register to 1.

For synchronization tasks, you can also obtain the task duration, notify_id, local end (src rank), and peer end (dst rank) from the task details.

![Synchronization tasks](figures/syn_task.png)

## Data Communication Task

- **Memcpy**: a memory copy task for intra-node or intra-chip memory copy.
- **Reduce_Inline**: a memory copy task, which completes on-the-fly reduction computation while copying data.
- **RDMASend**: an inter-node RoCE communication task, which corresponds to the inter-node memory copy task.

For data communication tasks, you can also obtain the task duration, local end (src rank) and peer end (dst rank), data volume (size), bandwidth, and other details from the task details.

![Data communication tasks](figures/data_comm_task.png)

> [!NOTE]
>
> - In the profile data, an RDMASend task corresponds to either a synchronization task or a data communication task. You can distinguish between them by analyzing the data volume. The data volume of a synchronization task is fixed at 4 bytes, while the data volume of a data communication task is subject to the actual communication volume.
> - If an RDMASend task is a data communication task, its execution duration is not the actual communication duration, but the duration of dispatching the WQE of the communication task to the QP queue. The actual communication duration can be calculated based on the data volume and bandwidth, or obtained by referring to the duration of the next notify wait task that immediately follows it.
