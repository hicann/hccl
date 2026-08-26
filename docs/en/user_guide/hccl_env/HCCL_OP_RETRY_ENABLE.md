# HCCL_OP_RETRY_ENABLE

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-05T02:57:00.327Z pushedAt=2026-08-11T03:30:15.903Z -->

## Function

Configures whether to enable retry for HCCL operators. HCCL operator retry is performed at the **communicator** granularity. When a communication operator reports an SDMA or RDMA CQE error during execution, HCCL retries the execution.

In a cluster, hardware intermittent disconnections may occur, causing communication operators to report errors during execution. Enabling HCCL retry through this environment variable can prevent communication interruptions caused by hardware intermittent disconnections. HCCL operator retry essentially provides a best-effort fault recovery method at the software level.

**Figure 1**  Retry flow diagram  
![Retry flow diagram](figures/reexec_flow_diagram.png)

Major steps:

1. Fault discovery: The AI CPU detects a fault signal and notifies the host to prepare for the retry.

2. Cluster management: The host exchanges information through the Host Socket and determines whether the current faulty operator meets a series of retry conditions. For details, see [Retry Instructions](#retry-instructions).

3. Re-dispatch: Notify the AI CPU Kernel to re-dispatch SQE and WQE for HCCL operator retry.

## Configuration

Through this environment variable, you can configure whether to enable retry for communicators at two physical levels: inter-server and inter-SuperPoD. Each level supports two states: enabled or disabled.

**How to configure:**

**export HCCL_OP_RETRY_ENABLE="L1:0,L2:0"**

- `L1` indicates that the physical scope of the communicator is inter-server. Value `0` disables retry for inter-server communication tasks within the communicator, while value `1` enables retry. The default value is `0`.

- `L2` indicates that the physical scope of the communicator is inter-SuperPoD. Value `0` disables retry for inter-SuperPoD communication tasks within the communicator, while value `1` enables retry. The default value is `0`.

    When `L2` is set to `1`, if a device NIC fails during inter-SuperPoD communication, a standby device NIC is used for communication during retry. This is called rail-failover communication. The standby NIC is one on another die within the same NPU. For details about the conditions for normal execution of rail-failover communication and its impact, see [Rail-failover Communication Instructions](#rail-failover-communication-instructions).

  - If the communicator is created based on a rank table, you need to configure the standby NIC using `backup_device_ip` in the rank table.

  - If the communicator is created based on root node information, the two dies on the same NPU are automatically configured as standby NICs for each other, requiring no manual configuration.

In addition, you can configure the wait time before the first retry, the maximum number of retries, and the interval between two retries through [HCCL_OP_RETRY_PARAMS](HCCL_OP_RETRY_PARAMS.md).

**Configuration recommendations:**

- Enabling retry incurs a certain performance loss. For Atlas A3 training products/Atlas A3 inference products, inter-server and inter-SuperPoD communication passes through optical interconnection domains, which have lower stability. You are advised to enable HCCL retry.

- This environment variable must be set consistently across all SuperPoDs. Otherwise, inter-SuperPoD link establishment will time out.

- When retry is enabled, the number of communicators should not exceed 5. Otherwise, communication operators may occupy all AI CPU cores, preventing computation operator execution on the AI CPU and causing service exceptions.

## Retry Instructions

When enabling HCCL retry, the following constraints must be met; otherwise, retry fails.

1. The communication operator expansion mode is `AI_CPU` (set through [HCCL_OP_EXPANSION_MODE](HCCL_OP_EXPANSION_MODE.md)). Otherwise, no retry is performed.

    ```bash
    export HCCL_OP_EXPANSION_MODE="AI_CPU"
    ```

2. When a communicator is created based on a rank table, the `host_ip` field in the rank table must be configured. Otherwise, retry does not take effect and the no-retry process is used.

3. The input memory of the communication operator must not be at risk of corruption during execution.

    A collective communication operator is a combination of a series of tasks. HCCL retry uses the communication operator as the granularity, starting from the operator's input memory and retrying the tasks of a communication operator. If the input memory of the communication operator is at risk of being corrupted during execution, retry may fail and the system may report an error and exit.

    Scenarios where input memory is at risk of being corrupted:

    - Zero-copy enabled: The ReduceScatter and AllReduce operators modify the user's input memory, so these two types of operators are not supported for retry.

    - In-place operations: The operator's input and output share the same memory block, for example, the [ReduceScatter](https://pytorch.org/docs/stable/distributed.html#torch.distributed.reduce_scatter)/[AllGather](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_gather) operators in PyTorch.

    - Graph mode: Communication can be performed directly on the operator's input and output. For example, for the [AllReduce](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce) operator in PyTorch, its input tensor serves as both the operator's input and output. During the operator's communication process, after partial results are written, the tensor content changes. If retry is performed on the corrupted input, incorrect computation results will be obtained.

4. When a fault occurs, all ranks in the communicator must stop at the same communication operator. If different ranks stop at different communication operators, retry cannot be performed.

    The moment of fault occurrence is unpredictable. When a fault occurs, the state of each rank in the entire communicator is related to the success rate of retry. Taking the communicator shown in the following figure as an example, which contains three ranks, [Table 1](#table1) lists the retry status when faults occur at different moments.

    **Figure 2** Communicator fault diagram 1  
    ![Communicator fault diagram 1](figures/comm_domain_fault_1.png)

    **Table 1** Retry status upon communicator fault<a id="table1"></a>

    | Fault Occurrence Moment | Retry | Operator for Retry |
    | --- | --- | --- |
    | A | Yes | HCCL OP1.<br>Since the computation operator cannot detect the link fault, the fault is detected only when the communication operator HCCL OP1 is executed. At this point, all three ranks stop at HCCL OP1, meeting the retry condition. The retry starts. |
    | B | Yes | HCCL OP1.<br>Rank0 and rank2 continue execution until reaching the communication operator HCCL OP1, and rank1 also stops at HCCL OP1, meeting the retry condition. The retry starts. |
    | C | Yes | HCCL OP1. |
    | D | No | HCCL OP1 of rank0 and rank1 has already completed. When the fault occurs at moment D, they continue execution until reaching HCCL OP2, while rank2 remains stopped at HCCL OP1, failing to meet the retry condition. |
    | E | Yes | HCCL OP3.<br>All three ranks continue execution and eventually stop at HCCL OP3, meeting the retry condition. The retry starts. |

    The following explains why collective communication cannot fully guarantee that all ranks stop at the same communication operator when a fault occurs, using the common collective communication algorithm Recursive Halving-Doubling (RHD) as an example.

    **Figure 3** Communicator fault diagram 2  
    ![Communicator fault diagram 2](figures/comm_domain_fault_2.png)

    Four AI servers, each with one rank, form a four-rank communicator. If the fault happens to occur after the first step (data exchange) of the RHD algorithm, the following situation arises:

    rank2 and rank3 can complete normally, while rank0 and rank1 cannot. Subsequent computation operators or communication operators on rank2 and rank3 may use arbitrary memory, and the corresponding context information cannot be found on rank2 and rank3 during retry. Therefore, if the fault occurrence moment is as shown in the figure above, retry cannot be performed.

5. Check whether the host-side socket network communication is normal. During retry, host-side socket communication is used for status negotiation among devices in the communicator. If the socket network is faulty, retry cannot be performed.

6. Ensure that the faulty link has recovered, for example, route convergence is successful, the optical module is rectified from an intermittent disconnection, or the communication is restored by using the standby NIC. If the faulty link cannot be recovered, the communication task will still fail when executed again. When the retry count exceeds the maximum retransmission count (see [HCCL_OP_RETRY_PARAMS](HCCL_OP_RETRY_PARAMS.md)), the retry fails.

> [!NOTE] Note
>
> - If an ERROR message with the keyword `[OpRetry]...timeout` appears in the host-side debug log, it indicates a host-side socket communication anomaly during HCCL retry. In this case, collect the logs of all nodes in the communicator to further locate the fault.
> - If an ERROR message with the keyword `can not retry` appears in the host-side debug log, it indicates that the current situation does not meet the HCCL retry conditions.
> The default storage path for debug logs generated by the host-side applications is `$HOME/ascend/log/debug/plog/`.

## Rail-failover Communication Instructions

1. To ensure proper rail-failover communication, the following conditions must be met:

    - The communication link of the standby NIC is normal.

    - Both devices in active/standby mode must be visible to services.

        For example, NPU1 contains two dies, Device0 and Device1, in active/backup mode. If `ASCEND_RT_VISIBLE_DEVICES` specifies only Device0 as service-visible, the rail-failover communication cannot be executed.

2. If rail failover occurs during communication (for example, the Die0 NIC of a certain NPU fails and the standby Die1 NIC is enabled), the traffic originally on the Die0 NIC will be sent and received through the Die1 NIC, increasing the traffic on Die1. The overall performance will degrade due to halved physical bandwidth and port conflicts.

3. In rail-failover communication, if the Die0 NIC of NPU0 fails, it switches to its standby NIC Die1. Because communication between two NPUs requires both the local end and the peer end to switch to the standby NIC simultaneously, NPU1 also switches from Die0 to Die1, as shown in Figure 2. However, if a communication task already exists between Die0 and Die1, the rail-failover communication cannot be executed.

    **Figure 4**  Rail-failover communication switch example  
    ![Rail-failover communication switch example](figures/borrow_comm_switch_example.png)

4. When rail-failover communication is enabled, you are advised to assign both Dies of an NPU to the same training or inference task.

    If the two Dies of the same NPU are assigned to two different training or inference tasks, a fault in one task will borrow the NIC of the other task, causing a certain degree of performance degradation in both tasks.

5. The same NPU allows only once rail failover, and switchback is not allowed.

    As shown in [Figure 5](#figure5), in Diagram 1, the communication link between NPU0 and NPU1 is faulty, and the standby link is enabled. Rail failover occurs, and communication proceeds normally. If the fault shown in Diagram 2 occurs again, rail failover is no longer supported, and the system exits with an error.

    **Figure 5** Only once rail failover allowed on the same NPU<a id="figure5"></a>  
    ![Only once rail failover allowed on the same NPU](figures/npu_single_borrow_example.png)

## Troubleshooting

If error `[OpRetryConnection][RecvAckTag] Recv unmatched ack` occurs after retry is enabled, the default port used for HCCL communication may be occupied, causing HCCL to connect to an incorrect server. The solution is as follows:

1. Run `sysctl -w net.ipv4.ip_local_reserved_ports` to reserve the default ports 60000-60015 used by HCCL, preventing the ports from being randomly assigned by the operating system.

    ```bash
    sysctl -w net.ipv4.ip_local_reserved_ports=60000-60015
    ```

2. If the error persists, use [HCCL_IF_BASE_PORT](HCCL_IF_BASE_PORT.md) to modify the default port used by HCCL, and run `sysctl -w net.ipv4.ip_local_reserved_ports` to reserve the specified ports.

    ```bash
    # For example, specify that HCCL uses 16 consecutive ports starting from port 17777.
    export HCCL_IF_BASE_PORT=17777
    # Reserve 16 ports from 17777 to 17792.
    sysctl -w net.ipv4.ip_local_reserved_ports=17777-17792
    ```

## Other Constraints

- If you call the HCCL C API to initialize a communicator with specific configurations and configure whether to enable retry for HCCL operators through `hcclRetryEnable` of `HcclCommConfig`, the communicator-level configuration takes precedence.

- Enabling retry will disable calling the `HcclCreateSubCommConfig` API to split a sub-communicator in multiple processes or threads on a single device.

## Impact of Retry on Network Performance

See [Impact of Communication Operator Retry on Overall Network Performance](comm_retry_perf_impact.md).

## Applicable Products

Atlas A3 training products/Atlas A3 inference products
