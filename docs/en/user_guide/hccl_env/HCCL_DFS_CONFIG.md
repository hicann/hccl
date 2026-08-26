# HCCL_DFS_CONFIG

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-05T02:54:58.652Z pushedAt=2026-08-11T03:21:55.275Z -->

## Function

HCCL provides switch settings for detecting multiple faults, including connection faults, cluster heartbeat, and process hang. Enabling these switches allows you to quickly locate faults with clearly displayed fault information, facilitating timely troubleshooting.

This environment variable supports the following configuration items:

- **connection_fault_detection_time**: connection fault detection time switch.

    When a connection times out, HCCL initiates root node locating for connection failures and propagates the root node information. The entire process takes the value of `connection_fault_detection_time` plus 10 seconds (for root node information propagation).

    The supported values of `connection_fault_detection_time` are `0` and [20, 7200]. The unit is second, and the default value is `20`.

    The value `0` means the connection fault detection is disabled. In this case, no additional wait time is incurred upon a connection failure, and the connection process exits immediately.

- **cluster_heartbeat**: cluster heartbeat monitoring switch. When a communication operation execution times out, this function propagates fault information and records the root cause node information in the run log.

    This parameter supports two values: `on` (enables heartbeat monitoring) and `off` (disables heartbeat monitoring). Defaults to `on`.

    **Note**: After cluster heartbeat monitoring is disabled, timeout exceptions during communication operation execution cannot be detected, the cluster fault information is not propagated, and root node fault information is not recorded in the run log.

- **stuck_detection**: process hang detection switch.

    This parameter supports two values: `on` (enables process hang detection) and `off` (disables process hang detection). Defaults to `on`.

    For tasks that are highly sensitive to communication performance, you can use this parameter to disable the process hang detection. However, note that after disabling it, the system no longer proactively detects and reports hang faults.

- **inconsistent_check**: operator dispatch inconsistency detection switch.

    This parameter supports three values: `on` (enables detection), `first` (checks only the first operator), and `off` (disables detection). Defaults to `off`.

This parameter enables operator dispatch inconsistency detection, but it causes a certain degree of performance degradation. Note that after disabling it, operator dispatch inconsistency issues are no longer actively detected and recorded.

**Note**: This function does not support the HcclBatchSendRecv operator and the graph mode. When enabled, it generates a data cache that occupies host-side memory.

- **task_exception**: task execution exception detection switch. When a communication operator fails during asynchronous execution on a device, the callback function notifies HCCL of the abnormal task information (stream and taskId). HCCL then retrieves the task information at dispatch time and prints the detailed information of the failed task and its operator, which helps locate exception causes.

This parameter supports two values: `on` (enables detection) and `off` (disables detection). Defaults to `on`.

After disabling this switch, HCCL no longer records and retrieves detailed information about abnormal tasks, and maintenance and debugging information related to task execution exceptions becomes unavailable.

    **Note**: This function is supported only on Ascend 950PR/Ascend 950DT.

- **task_monitor_interval**: operator task execution duration monitoring switch when the operator expansion mode is AI CPU.

    The value ranges from 0 to 7200000, in ms, and defaults to `0`.

  - The value `0` disables the monitoring.

  - When set to a value greater than "0": Enables the monitoring. When the execution duration of a single task exceeds the configured value, the task information is printed and stored in `$HOME/ascend/log/run/device-*/`, with the log keyword `StreamTaskMonitor`. Timing restarts after printing, meaning that if the execution duration of a single task is a multiple of the configured value, multiple rounds of information are printed.

    **Note**:

    1. This function currently works only on Atlas A3 training/inference products, and takes effect only when the communication operator expansion mode is AI_CPU.

    2. This function is a diagnostic tool for exceptions. Enabling it affects service execution performance, so it is not recommended when no exceptions occur.

    3. A value less than 100 ms cannot guarantee functional completeness, and it may significantly impact service execution performance and functionality, potentially causing service execution failures.

    4. A small value may lead to a risk of log flooding in the `$HOME/ascend/log/run/device-*/` directory.

    5. This function is a maintenance and debugging aid for task execution exceptions. You are advised to set a value slightly smaller than half of the value of HCCL_EXEC_TIMEOUT.

> [!NOTE] Note
> The detection capabilities provided by this environment variable are only intended to assist in pinpointing faults within a cluster. In certain complex cases, it may not find the actual cause. You need to confirm the actual fault-causing root node based on the generation time of detection events and the specific error reports of the detected nodes.

## Configuration Example

```bash
export HCCL_DFS_CONFIG="connection_fault_detection_time:30,cluster_heartbeat:on,stuck_detection:on,inconsistent_check:off,task_monitor_interval:0"
```

## Constraints

**In the current version, Ascend 950PR and Ascend 950DT only support the following three fields: task_exception, cluster_heartbeat, and inconsistent_check.**

## Applicable Products

Ascend 950PR/Ascend 950DT

Atlas A3 training products/Atlas A3 inference products

Atlas A2 training products/Atlas A2 inference products
