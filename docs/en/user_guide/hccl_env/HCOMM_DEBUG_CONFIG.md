# HCOMM_DEBUG_CONFIG

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-12T10:32:28.442Z pushedAt=2026-09-18T08:03:58.252Z -->

## Description

After this environment variable is enabled, run logs (that is, logs in the `$HOME/ascend/log/run` directory) will contain the detailed running information of specific HCOMM modules. By now, the following configuration items are supported: TASK or task (task orchestration module), DATA_OP or data_op (data plane API module), RESOURCE or resource (resource management module, including resource allocation and release operations), and CHANNEL or channel (channel module).

You can set this environment variable in the following two ways:

- Forward configuration: Set one or multiple modules, separated by commas (,). TASK (or task), DATA_OP (or data_op), RESOURCE (or resource), and CHANNEL (or channel) are case-insensitive.

    ```bash
    # Running information of the task module in the run log.
    export HCOMM_DEBUG_CONFIG="TASK" 
    # Running information of the task, data_op, resource, and channel modules in the run log.
    export HCOMM_DEBUG_CONFIG="task,data_op,resource,channel" 
    ```

- Backward configuration: Add `^` before the first module name, indicating that the detailed running information of all other modules except the configured modules is recorded in the run log.

    ```bash
    # Running information of all other modules except the data_op module in the run log (that is, the running information of the task, resource, and channel modules is recorded).
    export HCOMM_DEBUG_CONFIG="^data_op"
    # Running information of all other modules except the task and data_op modules in the run log (that is, the running information of the resource and channel modules is recorded).
    export HCOMM_DEBUG_CONFIG="^task,data_op"
    ```

**Note**

- No extra spaces are allowed when configuring the environment variable; otherwise, the configuration is invalid. For example, `export HCOMM_DEBUG_CONFIG="task, data_op "` contains extra spaces before and after `data_op`, so this environment variable configuration is invalid.
- The TASK module takes effect if either the HCOMM_DEBUG_CONFIG or HCCL_DEBUG_CONFIG environment variable is enabled. For details, see [HCCL_DEBUG_CONFIG](./HCCL_DEBUG_CONFIG.md).

**Recommendation**: When TASK module logs print communication operator call information, set `HCCL_ENTRY_LOG_ENABLE=1` to print the call behavior log of communication operators in real time, so that the TASK logs of different operators can be distinguished. For details, see [HCCL_ENTRY_LOG_ENABLE](./HCCL_ENTRY_LOG_ENABLE.md).

## Configuration Example

```bash
export HCOMM_DEBUG_CONFIG="TASK,DATA_OP,RESOURCE,CHANNEL" 
```

## Constraints

None.

## Applicable Products

<!-- npu="950" id3 -->
- 950PR/950DT: Supported
<!-- end id3 -->
<!-- npu="A3" id1 -->
- Atlas A3 training products/Atlas A3 inference products: Not supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2 training products/Atlas A2 inference products: Not supported
<!-- end id2 -->
<!-- npu="910" id4 -->
- Atlas training products: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas inference products: Not supported
<!-- end id5 -->
