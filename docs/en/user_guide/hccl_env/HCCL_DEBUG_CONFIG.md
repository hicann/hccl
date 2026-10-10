# HCCL_DEBUG_CONFIG

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-12T09:10:21.859Z pushedAt=2026-09-18T08:03:58.175Z -->

## Description

Enabling this environment variable will include detailed running information of specific HCCL submodules in the run log (the log in the `$HOME/ascend/log/run` directory). Currently, the following configuration items are supported: `ALG` or `alg` (algorithm orchestration module), `TASK` or `task` (task orchestration module), and `RESOURCE` or `resource` (resource management module, including resource application and release operations).

You can set this environment variable in the following two ways:

- Forward configuration: Configure one or more modules, separated by commas. TASK (or task), ALG (or alg), and RESOURCE (or resource) are case-insensitive.

    ```bash
    # Record the run information of the task module in run logs.
    export HCCL_DEBUG_CONFIG="TASK" 
    # Record the run information of the alg, task, and resource modules in run logs.
    export HCCL_DEBUG_CONFIG="alg,task,resource" 
    ```

- Backward configuration: Add `^` before the first module name, indicating that the detailed running information of all other modules except the configured submodules is recorded in the run log.

    ```bash
    # Record the run information of all modules except the task module in run logs (that is, the run information of the alg and resource modules is recorded).
    export HCCL_DEBUG_CONFIG="^task"
    # Record the run information of all modules except the task and alg modules in run logs (that is, the run information of the resource module is recorded).
    export HCCL_DEBUG_CONFIG="^task,alg"
    ```

**Note**

- When configuring this environment variable, no extra spaces are allowed; otherwise, the config is invalid. For example, in `export HCCL_DEBUG_CONFIG="alg, task "`, there are extra spaces before and after `task`, making this environment variable configuration invalid.
- The TASK module takes effect if it is enabled by either the HCCL_DEBUG_CONFIG or HCOMM_DEBUG_CONFIG environment variable. For details, see [HCOMM_DEBUG_CONFIG](./HCOMM_DEBUG_CONFIG.md).

**Recommendation:** When the TASK module logs the communication operator call information, you can set `HCCL_ENTRY_LOG_ENABLE=1` to print the call behavior logs of communication operators in real time, so that you can distinguish the TASK logs of different operator intervals. For details, see [HCCL_ENTRY_LOG_ENABLE](./HCCL_ENTRY_LOG_ENABLE.md).

## Configuration Example

```bash
export HCCL_DEBUG_CONFIG="ALG,TASK,RESOURCE" 
```

## Constraints

None.

## Applicable Products

<!-- npu="950" id3 -->
- 950PR/950DT: Supported
<!-- end id3 -->
<!-- npu="A3" id1 -->
- Atlas A3 training products/Atlas A3 inference products: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2 training products/Atlas A2 inference products: Supported
<!-- end id2 -->
<!-- npu="910" id4 -->
- Atlas training products: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas inference products: Not supported
<!-- end id5 -->
