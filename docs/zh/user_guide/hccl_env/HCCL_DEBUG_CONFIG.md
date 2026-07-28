# HCCL_DEBUG_CONFIG

## 功能描述

启用此环境变量后，运行日志（即“$HOME/ascend/log/run”目录下的日志）将包含HCCL特定子模块的详细运行信息。目前支持ALG或alg（算法编排模块）、TASK或task（任务编排模块）、RESOURCE或resource（资源管理模块，包括资源的申请和释放操作）几个配置项。

该环境变量支持如下两种形式的配置：

- 正向配置：支持配置1个或多个模块，各模块间使用英文逗号分隔，其中TASK（或task）、ALG（或alg）、RESOURCE（或resource）不区分大小写。

    ```bash
    # 运行日志中记录task模块的运行信息。
    export HCCL_DEBUG_CONFIG="TASK" 
    # 运行日志中记录alg、task、resource模块的运行信息。
    export HCCL_DEBUG_CONFIG="alg,task,resource" 
    ```

- 反向配置：在第一个模块名前面加上“^”，表示除了配置的子模块外，运行日志中会记录其他模块的详细运行信息。

    ```bash
    # 运行日志中记录除了task模块之外的其他所有模块的运行信息（代表记录alg、resource模块的运行信息）。
    export HCCL_DEBUG_CONFIG="^task"
    # 运行日志中记录除了task与alg模块之外的其他所有模块的运行信息（代表记录resource模块的运行信息）。
    export HCCL_DEBUG_CONFIG="^task,alg"
    ```

**注意**：

- 环境变量配置时，不允许存在多余空格，否则配置无效，例如：export HCCL_DEBUG_CONFIG="alg, task "，task前后存在多余空格，此环境变量配置无效。
- TASK模块在环境变量HCCL_DEBUG_CONFIG与HCOMM_DEBUG_CONFIG中任一开启即生效。详见[HCOMM_DEBUG_CONFIG](./HCOMM_DEBUG_CONFIG.md)。

**建议**：TASK模块日志在打印通信算子调用信息时，为区分不同算子区间的TASK日志，可以开启HCCL_ENTRY_LOG_ENABLE=1实时打印通信算子的调用行为日志。详见[HCCL_ENTRY_LOG_ENABLE](./HCCL_ENTRY_LOG_ENABLE.md)。

## 配置示例

```bash
export HCCL_DEBUG_CONFIG="ALG,TASK,RESOURCE" 
```

## 使用约束

无

## 产品支持情况

<!-- npu="950" id3 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id3 -->
<!-- npu="A3" id1 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持
<!-- end id2 -->
<!-- npu="910" id4 -->
- Atlas 训练系列产品：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas 推理系列产品：不支持
<!-- end id5 -->
