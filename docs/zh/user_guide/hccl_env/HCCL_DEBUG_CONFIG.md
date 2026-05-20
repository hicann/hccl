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

**注意**：环境变量配置时，不允许存在多余空格，否则配置无效，例如：export HCCL_DEBUG_CONFIG="alg, task "，task前后存在多余空格，此环境变量配置无效。

## 配置示例

```bash
export HCCL_DEBUG_CONFIG="ALG,TASK,RESOURCE" 
```

## 使用约束

无

## 支持的型号

Atlas A3 训练系列产品/Atlas A3 推理系列产品

Atlas A2 训练系列产品/Atlas A2 推理系列产品（针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，仅支持Atlas 800T A2 训练服务器、Atlas 900 A2 PoD 集群基础单元、Atlas 200T A2 Box16 异构子框。）
