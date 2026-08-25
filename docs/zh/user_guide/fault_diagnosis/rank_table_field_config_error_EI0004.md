# rank table字段配置错误（EI0004）

## 问题现象

针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，在CANN日志中存在关键字“RanktableCheck”，如下所示：

```text
[ERROR] HCCL(1265,):2025-10-21 07:56:47.198.454 [topoinfo_ranktableConcise.cc:727][15326][InitGroupStage][RanktableCheck]errNo[0x0000000005010001] super_device_id[] is invalid
```

## 可能原因

rank table的"version"字段为"1.2"，但rank table里"super_device_id"字段填写为空，导致rank table校验失败。

## 解决方法

在rank table文件中补充"super_device_id"字段，配置说明可参考[rank table配置资源信息（Atlas A3 训练系列产品/Atlas A3 推理系列产品）](../cluster_info_config/rank_table_config_a3.md)。
