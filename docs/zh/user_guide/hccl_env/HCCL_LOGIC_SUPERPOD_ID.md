# HCCL_LOGIC_SUPERPOD_ID

## 功能描述

针对Atlas A3 训练系列产品/Atlas A3 推理系列产品的超节点模式组网，若不使用rank table文件配置集群资源信息，可通过此环境变量指定当前节点运行进程所属的超节点ID，实现将一个物理超节点划分为多个逻辑超节点的功能。

该环境变量取值为string类型，长度需要小于128个字符，默认值为空字符串。

若不配置此环境变量，会获取环境中“Super Pod ID”的值作为超节点ID，“Super Pod ID”的取值可通过`npu-smi info -t spod-info -i <id> -c <chip_id>`命令查看。

## 配置示例

```bash
export HCCL_LOGIC_SUPERPOD_ID=super_pod_id_1
```

## 使用约束

- 此环境变量仅适用于超节点模式组网下未使用rank table文件配置集群信息的场景，若使用了rank table文件，则优先使用rank table文件中的配置。
- 此环境变量的作用为将一个物理超节点划分为多个逻辑超节点，不支持将归属于不同物理超节点的rank配置到一个逻辑超节点内。

## 支持的型号

Atlas A3 训练系列产品/Atlas A3 推理系列产品
