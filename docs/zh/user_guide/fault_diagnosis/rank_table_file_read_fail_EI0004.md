# rank table文件读取失败（EI0004）

## 问题现象

在CANN日志中存在关键字"is not a valid real path"，如下所示：

```text
[ERROR] HCCL(1104629,test_one_side):2025-10-28-17:45:13.679.684 [param_check.cc:66] [1104629][InitGroupStage][RanktableConfig]errNo[0x0000000005010001] path /ranktable.json is not a valid real path
```

## 可能原因

基于rank table文件初始化通信域时，传入的rank table文件路径不存在或者权限不足。

## 解决方法

修改正确的rank table文件路径或者配置正确的可读权限。
