# System Constraints and Limitations

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-07-31T04:00:09.978Z pushedAt=2026-08-14T02:35:19.633Z -->

## General Constraints

- Collective Communication does not support use in Ascend virtualization instance scenarios.

  > An Ascend virtualization instance refers to a vNPU (virtual NPU instance) created by partitioning the NPU configured on a physical machine or VM through resource virtualization and mounting it to the target environment. The virtualization management approach enables unified allocation and reclamation of resources with different specifications, meeting the operational requirements of multiple users repeatedly applying for and releasing resources.

- When the ACL Graph feature is used, if a captured Graph instance contains communication operators, all Ranks in the communicator must execute the same Graph instance generated from the same capture session during the execution phase. Pairing Graph instances generated from different capture sessions to execute communication operations is not supported. Even if the internal operator sequences of these Graph instances are identical, communication operator execution exceptions will occur.

- In single-machine multi-container scenarios, it is recommended to configure the same IP address for all containers to avoid performance degradation compared with bare-metal environments in performance comparison tests.

## Ascend 950PR/Ascend 950DT

- In scenarios where Ascend computing modules are adapted to third-party x86 servers, the operating system kernel version must be later than 5.18. If this requirement is not met, the haveged entropy supplement tool must be manually installed and its service started.

- Single-card multi-process scenarios:

  - Since the CCU does not support multi-process usage, in single-card multi-process scenarios, the creation of the first communicator on different NPUs must belong to the same group of service processes, ensuring that the CCU is occupied by the services of the same process group.

  - If the expansion mode of the communication operator is AI_CPU (default value), the single-card process concurrency count is recommended not to exceed 6.

  - If the expansion mode of the communication operator is AIV, concurrent execution of single-card multi-process is not recommended. It is recommended that multiple processes be executed serially.

  Follow the recommended configuration above. Otherwise, there is a risk of task deadlock. The expansion mode of the communication operator can be set through the environment variable "HCCL_OP_EXPANSION_MODE".

- In graph mode (Ascend IR) or graph capture (aclgraph) scenarios, when the communication algorithm uses the default AI CPU mode, the number of concurrent graphs on a single card cannot exceed 6. Otherwise, communication may be blocked due to AI CPU cores being fully occupied.

## Atlas A3 Training Products/Atlas A3 Inference Products

- If your driver/firmware version is 25.0.RC1 or later, single-card multi-process service scenarios are supported, meaning that multiple service processes can share a single NPU simultaneously. Note that multi-process execution has a certain impact on resource overhead and communication performance. If too many processes run on the same NPU, service execution may fail due to insufficient resources. If your driver/firmware version does not meet this requirement, only single-process execution is supported.

  If the expansion mode of the communication operator is AI_CPU (default value), the recommended concurrency count of single-card processes does not exceed 6. If the expansion mode of the communication operator is AIV, concurrent execution of multiple processes on a single card is not recommended, and serial execution among multiple processes is advised. Follow the configuration recommendations above; otherwise, there is a risk of task deadlock. The expansion mode of the communication operator can be set through the environment variable "HCCL_OP_EXPANSION_MODE".

- It is recommended that the number of servers in each SuperPoD be consistent, and the number of AI processors in each server be consistent. Inconsistency may cause performance degradation.

- In graph mode (Ascend IR) or graph capture (aclgraph) scenarios, when the communication algorithm uses the default AI CPU mode, the number of concurrent graphs on a single card must not exceed 6. Otherwise, communication may be blocked due to AI CPU cores being fully occupied.

## Atlas A2 Training Products/Atlas A2 Inference Products

- If your driver firmware is version 25.0.RC1 or later, single-card multi-process service scenarios are supported, meaning that multiple service processes can share one NPU simultaneously. Note that multi-process execution may have a certain impact on resource overhead and communication performance. If too many processes run on the same NPU, service execution may fail due to insufficient resources. If your driver firmware does not meet the version requirement, single-process execution is used.

    If the expansion mode of the communication operator is HOST (default value), the concurrency count of single-card processes should not exceed 8. If the expansion mode of the communication operator is AIV, concurrent execution of single-card multi-process is not recommended, and serial execution among multiple processes is advised. Follow the configuration recommendations above; otherwise, there is a risk of task deadlock. The operator expansion mode for the Atlas A2 training products/Atlas A2 inference products can be set through the environment variable "HCCL_OP_EXPANSION_MODE".

- In a single-server scenario, there is no limit on the number of AI processors participating in collective communication. In a server cluster scenario, the number of AI processors participating in collective communication must be (1 to 8) * n, where n is the number of servers participating in training. It is recommended that the number of AI processors participating in collective communication in each server be consistent; otherwise, performance degradation may occur.

<!-- npu="910" id1 -->

## Atlas Training Products

- The single-card multi-process service scenario is not supported, meaning that multiple service processes cannot share a single NPU simultaneously.

- In a single-server scenario, the number of AI processors actually participating in collective communication must be only 1, 2, 4, or 8, and cards 0-3 and cards 4-7 each form a separate network. When using 2 or 4 cards for training, cross-networking device cluster creation is not supported. In a server cluster scenario, the number of AI processors participating in collective communication must be only 1\*n, 2\*n, 4\*n, or 8\*n (where n is the number of servers participating in training), and cluster performance is optimal when n is a power of two. It is recommended that users prioritize this approach for cluster networking.<!-- end id1 -->

<!-- npu="310p" id2 -->

## Atlas 300I Duo Inference Card

- Single-card multi-process service scenarios are not supported, meaning that multiple service processes cannot share a single NPU simultaneously.

- Only single-server scenarios are supported. For details about the maximum number of NPUs supported by each collective communication operation, see the [specific API](../api_ref/comm_op_interface/README.md).

- Communicator initialization must be performed before any other operations involving Device memory allocation. Otherwise, initialization may fail due to insufficient P2P memory.

<!-- end id2 -->
