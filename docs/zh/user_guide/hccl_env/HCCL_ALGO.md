# HCCL_ALGO

## 功能描述

此环境变量用于配置集合通信Server间通信算法以及超节点间通信算法，支持全局配置算法类型与按算子配置算法类型两种配置方式。

> [!NOTE]说明
>
>- HCCL提供自适应算法选择功能，默认会根据产品形态、数据量和Server个数选择合适的算法，一般情况下用户无需手工指定。若通过此环境变量指定了Server间或超节点间通信算法，则自适应算法选择功能不再生效。
>- 在某些通信算子中，当使用特定类型的AI处理器且数据量较小时，通信算法会由HCCL自适应选择，不受此环境变量的控制。
>- 本节所列出的算法为HCCL支持配置的全量通信算法，不同产品下支持的Server间通信算法与超节点间通信算法可参见[Server间通信算法支持度列表](inter_server_algo_support.md)与[超节点间通信算法支持度列表](inter_superpod_algo_support.md)。

- **全局配置算法类型，配置方式如下：**

  ```bash
  export HCCL_ALGO="level0:NA;level1:<algo>;level2:<algo>"
  ```

  - level0代表Server内通信算法，当前仅支持配置为NA。
  - level1代表Server间通信算法，支持如下取值：

    - ring：基于环结构的通信算法，通信步数多（线性复杂度），时延相对较高，但通信关系简单，受网络拥塞影响较小。适合通信域内Server个数较少、通信数据量较小、网络存在明显拥塞、且pipeline算法不适用的场景。
    - H-D_R：递归二分和倍增算法（Recursive Halving-Doubling：RHD），通信步数少（对数复杂度），时延相对较低，但在非2的整数次幂节点规模下会引入额外的通信量。适合通信域内Server个数是2的整数次幂且pipeline算法不适用的场景，或Server个数不是2的整数次幂但通信数据量较小的场景。
    - NHR：非均衡的层次环算法（Nonuniform Hierarchical Ring），通信步数少（对数复杂度），时延相对较低。适合通信域内Server个数较多且pipeline算法不适用的场景。

      **当前版本Ascend 950PR/Ascend 950DT仅支持配置NHR算法。**

    - NHR_V1：对应历史版本的NHR算法，通信步数少（根复杂度），时延相对较低，适合通信域内Server数为非2的整数次幂且pipeline算法不适用的场景。NHR_V1算法理论性能低于新版NHR算法，该配置项未来会逐步停用，建议开发者使用NHR算法。
    - NB：非均匀的数据块通信算法（Nonuniform Bruck），通信步数少（对数复杂度），时延相对较低。适合通信域内Server个数较多且pipeline算法不适用的场景。
    - AHC：层次化集合通信算法（Asymmetric Hierarchical Concatenate），适用于通信域内NPU分布存在多个层次、多个层次间NPU对称或者非对称分布（即卡数非对称）的场景，当通信域内层次间存在带宽收敛时相对收益会更好。

      注意：当level1（Server间通信算法）配置为“AHC”时，level2（超节点间通信算法）将自动采用“AHC”算法，无需另行配置，即使level2设置了其他算法，这些设置也不会生效。

    - pipeline：流水线并行算法，可并发使用Server内与Server间的链路，适合通信数据量较大且通信域内每机包含多卡的场景。
    - pairwise：逐对通信算法，仅用于AlltoAll、AlltoAllV与AlltoAllVC算子，通信步数较多（线性复杂度），时延相对较高，且需要额外申请内存，内存大小与数据量成正比，但可以避免网络中出现一打多现象，适合通信数据量较大、需要规避网络一打多的场景。

    不设置level1时：

    - 针对Ascend 950PR/Ascend 950DT，默认使用NHR算法。
    - 针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，内部会根据产品形态、节点数以及数据量自动选择算法。
    - 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，内部会根据产品形态、节点数以及数据量自动选择算法。
    - 针对Atlas 训练系列产品，当通信域内Server的个数为非2的整数次幂时，默认使用ring算法；其他场景默认使用H-D_R算法。

  - level2代表超节点间通信算法，支持如下取值：

    - ring：基于环结构的通信算法，通信步数多（线性复杂度），时延相对较高，但通信关系简单，受网络拥塞影响较小。适合通信域内超节点个数较少且不是2的整数次幂的场景。
    - H-D_R：递归二分和倍增算法（Recursive Halving-Doubling：RHD），通信步数少（对数复杂度），时延相对较低，但在非2的整数次幂节点规模下会引入额外的通信量。适合通信域内超节点个数是2的整数次幂的场景，或超节点个数不是2的整数次幂但通信数据量较小的场景。
    - NHR：非均衡的层次环算法（Nonuniform Hierarchical Ring），通信步数少（对数复杂度），时延相对较低。适合通信域内超节点个数较多的场景。
    - NB：非均匀的数据块通信算法（Nonuniform Bruck），通信步数少（对数复杂度），时延相对较低。适合通信域内超节点个数较多的场景。
    - pipeline：流水线并行算法，可并发使用超节点内与超节点间的链路，适合通信数据量较大且通信域内每个超节点包含多卡的场景。

        超节点间每种通信算法支持的通信算子、数据类型、网络运行模式等说明可参见[超节点间通信算法支持度列表](inter_superpod_algo_support.md)。

        不设置level2时，当通信域内超节点个数小于8且不是2的整数次幂时，采用ring算法；其他场景采用H-D_R算法。

    level2配置当前仅适用于如下场景：

    - 仅支持Atlas A3 训练系列产品/Atlas A3 推理系列产品。
    - 仅支持通信算子展开模式为AI_CPU的场景，通信算子展开模式可通过环境变量[HCCL_OP_EXPANSION_MODE](HCCL_OP_EXPANSION_MODE.md)配置。

- **按算子类型配置通信算法，配置方式如下：**

    ```bash
    export HCCL_ALGO="<op0>=level0:NA;level1:<algo0>;level2:<algo1>/<op1>=level0:NA;level1:<algo3>;level2:<algo4>"
    ```

    其中：

  - <op\>为通信算子的类型，支持如下配置：
    - allgather：对应通信算子AllGather和AllGatherV。
    - reducescatter：对应通信算子ReduceScatter和ReduceScatterV。
    - allreduce：对应通信算子AllReduce。
    - broadcast：对应通信算子Broadcast。
    - reduce：对应通信算子Reduce。
    - scatter：对应通信算子Scatter。
    - alltoall：对应通信算子AlltoAll、AlltoAllV和AlltoAllVC。

  - <algo\>为指定的通信算子采用的通信算法，支持的配置同全局配置方法中的level1取值与level2取值，请确保指定的通信算法为通信算子支持的算法类型，每种算法支持的通信算子可参见[Server间通信算法支持度列表](inter_server_algo_support.md)与[超节点间通信算法支持度列表](inter_superpod_algo_support.md)，未指定通信算法的通信算子会根据产品形态、节点数以及数据量自动选择通信算法。
  - 多个算子之间的配置使用“/”分隔。

## 配置示例

- 全局配置算法类型

    ```bash
    export HCCL_ALGO="level0:NA;level1:NHR"
    ```

- 按算子配置算法类型

    ```bash
    # AllReduce算子使用Ring算法，AllGather算子使用RHD算法，其他算子根据产品形态、节点数以及数据量自动选择通信算法。
    export HCCL_ALGO="allreduce=level0:NA;level1:ring/allgather=level0:NA;level1:H-D_R"
    ```

## 使用约束

- 当前版本Server内通信算法仅支持配置为“NA”。
- 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，在严格确定性计算的保序场景下，不建议配置HCCL_ALGO环境变量。
- 若您调用HCCL C接口初始化具有特定配置的通信域时，通过“HcclCommConfig”的“hcclAlgo”参数指定了通信算法，则以通信域粒度的配置优先。

## 支持的型号

Ascend 950PR/Ascend 950DT

Atlas A3 训练系列产品/Atlas A3 推理系列产品

Atlas A2 训练系列产品/Atlas A2 推理系列产品

<cann-filter npu-type="910">
Atlas 训练系列产品</cann-filter>
