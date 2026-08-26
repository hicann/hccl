# HCCL_DEBUG_CONFIG

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-05T02:54:43.485Z pushedAt=2026-08-06T06:43:53.403Z -->

## Function

Enabling this environment variable will include detailed running information of specific HCCL submodules in the run log (the log in the `$HOME/ascend/log/run` directory). Currently, the following configuration items are supported: `ALG` or `alg` (algorithm orchestration module), `TASK` or `task` (task orchestration module), and `RESOURCE` or `resource` (resource management module, including resource application and release operations).

You can set this environment variable in the following two ways:

- Forward configuration: Configure one or more modules, separated by commas. TASK (or task), ALG (or alg), and RESOURCE (or resource) are case-insensitive.

    ```bash
    # Record the running information of the task module in the run log.
    export HCCL_DEBUG_CONFIG="TASK" 
    # Record the running information of the alg, task, and resource modules in the run log.
    export HCCL_DEBUG_CONFIG="alg,task,resource" 
    ```

- Backward configuration: Add `^` before the first module name, indicating that the detailed running information of all other modules except the configured submodules is recorded in the run log.

    ```bash
    # Record the running information of all modules except the task module in the run log (meaning the running information of the alg and resource modules is recorded).
    export HCCL_DEBUG_CONFIG="^task"
    # Record the running information of all modules except the task and alg modules in the run log (meaning the running information of the resource module is recorded).
    export HCCL_DEBUG_CONFIG="^task,alg"
    ```

**Note:** When configuring this environment variable, no extra spaces are allowed; otherwise, the config is invalid. For example, in `export HCCL_DEBUG_CONFIG="alg, task "`, there are extra spaces before and after `task`, making this environment variable configuration invalid.

## Configuration Example

```bash
export HCCL_DEBUG_CONFIG="ALG,TASK,RESOURCE" 
```

## Constraints

None.

## Applicable Products

Atlas A3 training products/Atlas A3 inference products

Atlas A2 training products/Atlas A2 inference products
