# 定位思路

在业务的日志中出现"EI0001"故障码意味着HCCL的环境变量配置异常，一般情况下打印日志的ERROR MESSAGE和CANN日志中会显示配置异常的环境变量名称及错误原因，以及合理的配置范围，如果有疑问请参照[环境变量参考](../hccl_env/README.md)。

针对Ascend 950PR&950DT系列产品，可在plog中检索`[InitGroupStage][EnvConfig]`，再根据环境变量名、配置值和错误原因定位问题。例如，设置`HCCL_ENTRY_LOG_ENABLE=-1`时，日志正文包含以下信息，表示该环境变量仅支持0或1：

```text
[InitGroupStage][EnvConfig] ... Env config "HCCL_ENTRY_LOG_ENABLE" value "-1" is invalid. ... Should be 0 or 1
```

不同版本的日志正文可能略有差异。若未找到上述完整示例，可结合`[InitGroupStage][EnvConfig]`和环境变量名检索，并根据报错中的实际值与允许范围修正配置。
