# HCCL_ALG_MULTIPLE_DIMENSION_SPLIT_RATIO

## 功能描述

该环境变量用于配置AllReduce、AllGather、ReduceScatter、Broadcast、Reduce算子在特定两维度并行通信算法下的数据切分比例。

两维度并行指Server内通信与Server间通信并行执行，Server内通信使用Mesh算法，Server间通信使用NHR算法的场景。在此场景下，HCCL会将每轮待通信的数据切分为两片，并分别走两条可并行的通信路径：一片数据先执行Mesh通信、再执行NHR通信，另一片数据先执行NHR通信、再执行Mesh通信。该环境变量用于调整这两片数据的大小，使耗时较长的通信路径分配较少数据，耗时较短的通信路径分配较多数据，从而改善两条路径的负载均衡。

该环境变量需要配置为数字，取值范围：\[0,1\]，默认值：0.5。

需要注意：

- 该环境变量仅影响HCCL已经选择两维度并行通信算法时AllReduce、AllGather、ReduceScatter、Broadcast、Reduce算子的两片数据分配比例，不用于选择通信算法。
- AllReduce和ReduceScatter算子中，两片数据0、1的目标切分比例为“环境变量取值”和“1-环境变量取值”；AllGather算子中，算法内部使用的两片数据切分顺序与AllReduce、ReduceScatter、Broadcast、Reduce相反，即两片数0、1的目标切分比例为“1-环境变量取值”和“环境变量取值”。
- 实际切分时会根据数据类型大小、对齐要求和尾块数据量进行取整或对齐处理，因此实际切分比例可能与配置值存在少量偏差。
- 若未配置该环境变量，或配置的数字超出\[0,1\]范围，系统使用默认值0.5。若配置为非数字格式，系统会在初始化环境变量时返回错误。
- 一般情况下，用户保持默认值即可。仅建议在确认当前业务处于跨框、两维度通信场景，且需要针对特定组网、数据量进行性能调优时修改该环境变量。

以`R`表示HCCL_ALG_MULTIPLE_DIMENSION_SPLIT_RATIO的取值。

![HCCL_ALG_MULTIPLE_DIMENSION_SPLIT_RATIO含义示意](./figures/HCCL_ALG_MULTIPLE_DIMENSION_SPLIT_RATIO.png)

ReduceScatter和AllGather配置相同的`R`时，`R`对应的数据片不同，原因如下：

- ReduceScatter和AllGather的通信语义相反：ReduceScatter将完整输入数据规约并分散到各rank，AllGather则将各rank上的分片数据收集还原为完整输出数据。
- 两维度并行通信算法会使用“数据片0”和“数据片1”对应两条通信路径，但两个算子的分片与通信路径的对应关系相反。ReduceScatter中，`R`对应数据片0，即“Mesh -> NHR”通信路径；AllGather中，`R`对应数据片1，即“NHR -> Mesh”通信路径。
- 因此，同样设置`R=0.6`时，ReduceScatter表示“Mesh -> NHR”通信路径分配约60%的数据；AllGather表示“NHR -> Mesh”通信路径分配约60%的数据。

ReduceScatter算子中，数据片0的大小为`R`，数据片1的大小为`1-R`：

```text
每轮待通信数据
|---------------- 数据片0：R ----------------|------ 数据片1：1-R ------|

并行阶段1：
数据片0：Mesh  =============================>
数据片1：NHR   =============================>

并行阶段2：
数据片0：NHR   =============================>
数据片1：Mesh  =============================>

数据片0通信路径：Mesh -> NHR，大小为R
数据片1通信路径：NHR  -> Mesh，大小为1-R
```

若ReduceScatter算子中“Mesh -> NHR”通信路径较慢，可调小`R`；若“NHR -> Mesh”通信路径较慢，可调大`R`。

AllGather算子中，数据片0的大小为`1-R`，数据片1的大小为`R`：

```text
每轮待通信数据
|------ 数据片0：1-R ------|---------------- 数据片1：R ----------------|

并行阶段1：
数据片0：Mesh  =============================>
数据片1：NHR   =============================>

并行阶段2：
数据片0：NHR   =============================>
数据片1：Mesh  =============================>

数据片0通信路径：Mesh -> NHR，大小为1-R
数据片1通信路径：NHR  -> Mesh，大小为R
```

若AllGather算子中“Mesh -> NHR”通信路径较慢，可调大`R`；若“NHR -> Mesh”通信路径较慢，可调小`R`。

## 配置示例

```bash
export HCCL_ALG_MULTIPLE_DIMENSION_SPLIT_RATIO=0.5
```

## 使用约束

- 该环境变量仅在AllReduce、AllGather、ReduceScatter、Broadcast、Reduce算子选择两维度并行通信算法时生效。若当前拓扑、数据量、数据类型、reduce类型或算子展开模式选择了其他通信算法，则该环境变量不生效。
- 建议业务结合实际组网和通信数据量进行性能验证后再调整该环境变量。过小或过大的配置可能导致两片数据负载不均，影响通信性能。

## 支持的型号

Ascend 950PR/Ascend 950DT
