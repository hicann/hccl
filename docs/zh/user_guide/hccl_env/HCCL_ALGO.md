# HCCL_ALGO

## 功能描述

此环境变量用于配置集合通信Server间通信算法以及超节点间通信算法，支持全局配置算法类型与按算子配置算法类型两种配置方式。

> [!NOTE]说明
>
> - HCCL提供自适应算法选择功能，默认会根据产品形态、数据量和Server个数选择合适的算法，一般情况下用户无需手工指定。若通过此环境变量指定了Server间或超节点间通信算法，则自适应算法选择功能不再生效。
> - 在某些通信算子中，当使用特定类型的AI处理器且数据量较小时，通信算法会由HCCL自适应选择，不受此环境变量的控制。
> - 本节所列出的算法为HCCL支持配置的全量通信算法，不同产品下支持的Server间通信算法与超节点间通信算法可参见[Server间通信算法支持度列表](inter_server_algo_support.md)与[超节点间通信算法支持度列表](inter_superpod_algo_support.md)。

<!-- npu="950" id2 -->
## 配置说明（Ascend 950PR&950DT系列产品）

### 全局配置

全局配置就是不指定算子OpType的配置，对所有算子都生效。

```bash
export HCCL_ALGO="<ExecutorType>{level0=<algoType>,level1=<algoType>};not(<ExecutorType>{level3=<algoType>})"
```

- `<ExecutorType>{level0=<algoType>,level1=<algoType>}`：指定算法执行器，并定义此执行器下不同层级使用的算法。<a id="executor_type_define"></a>
  - `<ExecutorType>`为算法执行器的类型，支持如下配置：
    - Sole（sole）：单算法执行器。
    - Sequence（sequence）：顺序执行器，串行执行一组算法，当前仅支持配置2级或3级算法。
    - Parallel（paralle）：两维度并行执行器，同一时刻两维度并行执行不同算法。
    - PipeLine（pipeline）：流水线并行执行器。
    - Concur（concur）：并发执行器，同一时刻相同level并行执行不同算法。
    - StrictOrdered（strictordered）：严格保序执行器。

    若不配置`<ExecutorType>`，默认使用Sole单算法执行器。

  - `level*=<algoType>`：指定层级使用的算法，多个层级的配置之间使用逗号“,”分隔。<a id="algo_define"></a>

    - `level*`：指定的算法层级，例如：level0、level1。
    - `<algoType>`：通信算法的类型，支持如下配置：

      - Mesh（mesh）：全互联mesh算法。
      - NHR（nhr）：华为自研非均衡层次环算法。
      - Mesh2Die（mesh2die）：单NPU分布端口跨Die mesh算法。
      - MeshOneShot（meshoneshot）：OneShot mesh算法。
      - MeshTwoShot（meshtwoshot）：TwoShot mesh算法。
      - MeshConcur（meshsoncur）：CLOS Z轴绕路。
      - MeshMultiLink（meshmultilink）：多链路Mesh通信算法，类似Concur。
      - MeshChunk（meshchunk）：Mesh通信算法，支持chunk。
      - MeshChunkTwoShot（meshchunktwoshot）：TwoShot mesh算法，支持chunk。
      - MeshSingleChannel（meshsinglechannel）：单通道Mesh算法。
      - MeshMultiJetty（meshmultijetty）：多Jetty Mesh通信算法。
      - MeshConcurrent（meshconcurrent）：多路径并发Mesh通信算法。
      - NHRMultiLink（nhrmultilink）：多链路NHR通信算法。
      - NHRAicpuReduce（nhraicpureduce）：INT64/FP64等数据类型对应的NHR算法。
      - NHRMultiJetty（nhrmultijetty）：多Jetty NHR通信算法。

    - 若不指定`level`，`<algoType>`默认按照算法层级从低到高匹配。例如{Mesh, NHR}，代表level0使用Mesh算法，level1使用NHR算法。

- `not(a1)`：取非, 表示不支持某种算法, 其他的都可选。取非支持作用在`<algoType>`上, 也支持作用在`executor{a1,a2}`上。

配置示例：

```bash
# 所有层级的算子都优先使用NHR算法，使用默认的算法执行器Sole
export HCCL_ALGO="nhr"
# 所有算子都不使用sole(nhr)算法
export HCCL_ALGO="not(nhr)"
# 算法顺序执行，level0执行Mesh算法，level1执行NHR算法
export HCCL_ALGO="sequence{mesh,nhr}"
# 算法顺序执行，level0可执行NHR算法以外的其他算法，level1执行NHR算法
export HCCL_ALGO="sequence{level0=not(nhr),level1=nhr}"
# 不支持parallel{mesh,nhr}算法（即level0采用Mesh，level1采用NHR，且level0与level1并行执行），其他都可以选择
export HCCL_ALGO="not(parallel{mesh,nhr})"
```

### 按算子类型配置

指定算子类型OpType，配置的算法仅对指定算子生效。

```bash
export HCCL_ALGO="<OpType>:<ExecutorType>{level0=<algoType>,level1=<algoType>,...};<OpType>:<ExecutorType>{...}"
```

- `<OpType>`：通信算子的类型，支持如下配置：
  - AllGather（allgather）
  - AllGatherV（allgatherv）
  - AllReduce（allreduce）
  - AllToAll（alltoall）
  - AllToAllV（alltoallv）
  - AllToAllVC（alltoallvc）
  - Broadcast（broadcast）
  - Reduce（reduce）
  - ReduceScatter（reducescatter）
  - ReduceScatterV（reducescatterv）
  - Scatter（scatter）
- `<ExecutorType>`：算法执行器的类型，支持的配置同[全局配置中的ExecutorType](#executor_type_define)。
- `level*=<algoType>`：指定层级使用的算法，多个层级的配置之间使用逗号“,”分隔。配置方式同[全局配置中的algoType](#algo_define)。
- 多个算子的配置之间使用分号“;”分隔。

配置示例：

```bash
# AllReduce算子采用Sequence执行器顺序执行算法，level0执行Mesh算法, level1执行NHR算法。
export HCCL_ALGO="allreduce:sequence{mesh,nhr}"
# AllGather算子禁止使用level0执行Mesh算法、level1执行NHR算法的Parallel并行执行器。
export HCCL_ALGO="allgather:not(parallel{mesh,nhr})"
```
<!-- end id2 -->

<!-- npu="A3,910b,910" id3 -->
## 配置说明（Atlas A3系列产品/Atlas A2系列产品/Atlas训练系列产品）

### 全局配置

```bash
export HCCL_ALGO="level0:NA;level1:<algo>;level2:<algo>"
```

- level0代表Server内通信算法，当前仅支持配置为“NA”。
- level1代表Server间通信算法，支持如下取值：
  - ring：基于环结构的通信算法，通信步数多（线性复杂度），时延相对较高，但通信关系简单，受网络拥塞影响较小。适合通信域内Server个数较少、通信数据量较小、网络存在明显拥塞、且pipeline算法不适用的场景。
  - H-D_R：递归二分和倍增算法（Recursive Halving-Doubling：RHD），通信步数少（对数复杂度），时延相对较低，但在非2的整数次幂节点规模下会引入额外的通信量。适合通信域内Server个数是2的整数次幂且pipeline算法不适用的场景，或Server个数不是2的整数次幂但通信数据量较小的场景。
  - NHR：非均衡的层次环算法（Nonuniform Hierarchical Ring），通信步数少（对数复杂度），时延相对较低。适合通信域内Server个数较多且pipeline算法不适用的场景。
  - NHR_V1：对应历史版本的NHR算法，通信步数少（根复杂度），时延相对较低，适合通信域内Server数为非2的整数次幂且pipeline算法不适用的场景。NHR_V1算法理论性能低于新版NHR算法，该配置项未来会逐步停用，建议开发者使用NHR算法。
  - NB：非均匀的数据块通信算法（Nonuniform Bruck），通信步数少（对数复杂度），时延相对较低。适合通信域内Server个数较多且pipeline算法不适用的场景。
  - AHC：层次化集合通信算法（Asymmetric Hierarchical Concatenate），适用于通信域内NPU分布存在多个层次、多个层次间NPU对称或者非对称分布（即卡数非对称）的场景，当通信域内层次间存在带宽收敛时相对收益会更好。
    注意：当level1（Server间通信算法）配置为“AHC”时，level2（超节点间通信算法）将自动采用“AHC”算法，无需另行配置，即使level2设置了其他算法，这些设置也不会生效。
  - pipeline：流水线并行算法，可并发使用Server内与Server间的链路，适合通信数据量较大且通信域内每机包含多卡的场景。
  - pairwise：逐对通信算法，仅用于AlltoAll、AlltoAllV与AlltoAllVC算子，通信步数较多（线性复杂度），时延相对较高，且需要额外申请内存，内存大小与数据量成正比，但可以避免网络中出现一打多现象，适合通信数据量较大、需要规避网络一打多的场景。

  不设置level1时：
    <!-- npu="A3" id4 -->
  - 针对Atlas A3系列产品，内部会根据产品形态、节点数以及数据量自动选择算法。
    <!-- end id4 -->
    <!-- npu="910b" id5 -->
  - 针对Atlas A2系列产品，内部会根据产品形态、节点数以及数据量自动选择算法。
    <!-- end id5 -->
    <!-- npu="910" id6 -->
  - 针对Atlas训练系列产品，当通信域内Server的个数为非2的整数次幂时，默认使用ring算法；其他场景默认使用H-D_R算法。
    <!-- end id6 -->
- level2代表超节点间通信算法，支持如下取值：
  - ring：基于环结构的通信算法，通信步数多（线性复杂度），时延相对较高，但通信关系简单，受网络拥塞影响较小。适合通信域内超节点个数较少且不是2的整数次幂的场景。
  - H-D_R：递归二分和倍增算法（Recursive Halving-Doubling：RHD），通信步数少（对数复杂度），时延相对较低，但在非2的整数次幂节点规模下会引入额外的通信量。适合通信域内超节点个数是2的整数次幂的场景，或超节点个数不是2的整数次幂但通信数据量较小的场景。
  - NHR：非均衡的层次环算法（Nonuniform Hierarchical Ring），通信步数少（对数复杂度），时延相对较低。适合通信域内超节点个数较多的场景。
  - NB：非均匀的数据块通信算法（Nonuniform Bruck），通信步数少（对数复杂度），时延相对较低。适合通信域内超节点个数较多的场景。
  - pipeline：流水线并行算法，可并发使用超节点内与超节点间的链路，适合通信数据量较大且通信域内每个超节点包含多卡的场景。

  超节点间每种通信算法支持的通信算子、数据类型、网络运行模式等说明可参见[超节点间通信算法支持度列表](inter_superpod_algo_support.md)。

  不设置level2时，当通信域内超节点个数小于8且不是2的整数次幂时，采用ring算法；其他场景采用H-D_R算法。

  level2配置当前仅适用于Atlas A3系列产品，仅支持通信算子展开模式为AI_CPU的场景。

  通信算子展开模式可通过环境变量[HCCL_OP_EXPANSION_MODE](HCCL_OP_EXPANSION_MODE.md)配置。

配置示例：

```bash
export HCCL_ALGO="level0:NA;level1:NHR"
```

### 按算子类型配置

```bash
export HCCL_ALGO="<op0>=level0:NA;level1:<algo0>;level2:<algo1>/<op1>=level0:NA;level1:<algo3>;level2:<algo4>"
```

- <op\>为通信算子的类型，支持如下配置：
  - allgather：对应通信算子AllGather和AllGatherV。
  - reducescatter：对应通信算子ReduceScatter和ReduceScatterV。
  - allreduce：对应通信算子AllReduce。
  - broadcast：对应通信算子Broadcast。
  - reduce：对应通信算子Reduce。
  - scatter：对应通信算子Scatter。
  - alltoall：对应通信算子AlltoAll、AlltoAllV和AlltoAllVC。
- <algo\>为指定的通信算子采用的通信算法，支持的配置同全局配置方法中的level1取值与level2取，请确保指定的通信算法为通信算子支持的算法类型，每种算法支持的通信算子可参见[Server间通信法支持度列表](inter_server_algo_support.md)与[超节点间通信算法支持度列表](inter_superpod_algo_support.md)，未指定通信算法的通信算子会根据产品形态、节点数以及数据自动选择通信算法。
- 多个算子之间的配置使用“/”分隔。

配置示例：

```bash
# AllReduce算子使用Ring算法，AllGather算子使用RHD算法，其他算子根据产品形态、节点数以及数据量自动选择通信算法。
export HCCL_ALGO="allreduce=level0:NA;level1:ring/allgather=level0:NA;level1:H-D_R"
```
<!-- end id3 -->

## 使用约束

- 若您调用HCCL C接口初始化具有特定配置的通信域时，通过“HcclCommConfig”的“hcclAlgo”参数指定了通信算法，则以通信域粒度的配置优先。
- 如果输入的HCCL_ALGO配置字符串格式错误，则配置不会生效，系统打印警告信息，但不会阻碍后续执行流程。
<!-- npu="910b" id7 -->
- 针对Atlas A2系列产品，在严格确定性计算的保序场景下，不建议配置HCCL_ALGO环境变量。
<!-- end id7 -->

## 产品支持情况

<!-- npu="950" id10 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id10 -->
<!-- npu="A3" id9 -->
- Atlas A3系列产品：支持
<!-- end id9 -->
<!-- npu="910b" id8 -->
- Atlas A2系列产品：支持
<!-- end id8 -->
<!-- npu="910" id1 -->
- Atlas训练系列产品：支持
<!-- end id1 -->
<!-- npu="310p" id11 -->
- Atlas推理系列产品：不支持
<!-- end id11 -->
