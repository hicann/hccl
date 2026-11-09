/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_COLL_ALG_SELECTOR_COST_MODEL
#define HCCLV2_COLL_ALG_SELECTOR_COST_MODEL

#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "alg_param.h"
#include "log.h"
#include "topo_host.h"
#include "template_utils.h"

namespace ops_hccl {

// 前向声明: AlgoType 定义在 alg_parse.h, 而 alg_parse.h 包含本头文件, 不能反向 include
enum class AlgoType : uint8_t;

typedef struct {
    const char* algName;
    const char* executorName;
    const char** templateName;
    int templateNum;
    HcclCMDType opType;
} AlgElement;

typedef struct {
    AlgElement* algElements;
    int count;
    int capacity;
} AllAlgos;

AllAlgos* GetAllAlgos();

HcclResult AddAlgToAllAlgos(
    HcclCMDType opType, const char* algName, const char* executorName, const char** templateName, int templateNum);

// 检查算法是否匹配当前拓扑，返回 true=匹配，false=不匹配
// needSoftPolicyCheck=false 时跳过软策略 topoCustomCheck；topoPrecondCheck 硬前提始终执行。
bool IsAlgoMatchTopo(
    const std::string& algName, const TopoInfoWithNetLayerDetails* topoInfo, bool needSoftPolicyCheck = true);

// topo 匹配检查结果（不打日志，供调用方决定日志级别）
struct TopoMatchResult {
    bool matched = true;
    std::string reason; // matched=false 时的过滤原因
};
TopoMatchResult CheckAlgoMatchTopoWithReason(
    const std::string& algName, const TopoInfoWithNetLayerDetails* topoInfo, bool needSoftPolicyCheck = true);

typedef struct {
    float A; // 用来描述跨卡传输的时间随DataSize变化的趋势，会受到UB带宽利用率的影响
    float B; // 用来描述本地传输的时间随DataSize变化的趋势，不受到UB带宽利用率的影响
    float C; // 用来描述一些基本时延的常数项
    float D;
    // AICPU 规约总量折算为 dataSize 的份数(字节口径, 与 dtype 无关): 规约总字节 = reduceCopies×dataSize;
    // 64bit/PROD 时由框架 per-call 按 count×1.2ns 折算进 B。存量 {A,B,C,D} 聚合初始化自动补 0
    float reduceCopies = 0.0f;
} CostModelParam;

// host 内联标量规约实测斜率(1.2ns/count, fp64 1GB 0.162s / 128M count); 64bit/PROD 的 LocalReduce 走该路径
constexpr float REDUCE_SEC_PER_COUNT = 1.2e-9f; // 取值为秒/count, 与 A/B/C/D 的秒口径一致

// 64bit/PROD 时按 count×1.2ns 把 reduceCopies 折算为 B 的每字节斜率(不除 util, host 项与 B 口径一致):
// Breduce×dataSize = 1.2ns × reduceCopies×dataSize/dts。INT64/UINT64/FP64 是仅有的 8 字节类型,
// dataTypeSize==8 与 wrapper 的 64bit 路由条件等价(另加 PROD)
inline float CalcHostReduceSlope(float reduceCopies, u32 dataTypeSize, HcclReduceOp reduceOp)
{
    bool hostPath = dataTypeSize == 8 || reduceOp == HcclReduceOp::HCCL_REDUCE_PROD;
    return (hostPath && dataTypeSize > 0 && reduceCopies > 0.0f) ?
               REDUCE_SEC_PER_COUNT * reduceCopies / static_cast<float>(dataTypeSize) :
               0.0f;
}

enum class CostAggMode : int {
    SUM = 0, // 多组 cost 求和
    MAX = 1, // 多组 cost 取最大值
};

struct AlgNetMeta {
    std::vector<CommTopo> netTypes;                // 每个 template 一个，顺序与 costmodel 中 A/B/C 一致
    std::vector<float> dataRatios;                 // 每个 segment 的 dataRatio，用于查 UB 利用率
    std::vector<u32> rankSizes;                    // 每个 segment 使用的 rankSize
    CostAggMode intraGroupMode = CostAggMode::SUM; // 组内聚合方式
    CostAggMode interGroupMode = CostAggMode::SUM; // 组间聚合方式，默认为SUM
    std::vector<u32> groupSizes;                   // 每组 template 数量，为空时按每组1个兜底
    // 逐段算法类型(如 Parallel 的4段 [MESH,NHR,NHR,MESH]); 为空时 cost_table 回退用 AlgAttrs 按算法名
    // 解析的 algoTypes(该值同时是 topo_match 的层级数依据, 只能保留层级个条目, 不能扩展成段数)
    std::vector<AlgoType> algoTypes;
};

typedef struct {
    const char* algName;
    // 所有权：param 指向的内存由 costModel_ 持有（FilterByTopoAndCalibrate 深拷贝）。
    // FreeCostModel 会逐个释放 param 指向的内存。
    const CostModelParam* param;
    int count;
    // HCCL_ALGO 优先级: 1=正向指定(优先), -1=否定指定(排除), 0=未配置。
    // 只提供优先级不强制过滤: costtable 过滤后在幸存者中按最高优先级层收敛, 全为否定层时放开回退;
    // 正向指定(preferred)的算法跳过软策略 customCheck, 排他时不被删除
    int hcclAlgoPriority = 0;
} CostAlgoParams;

typedef struct {
    CostAlgoParams* costAlgoParams;
    int count;
} CostModel;

enum class EngineType : int {
    AICPU = 0,
    CCU = 1,
    CCU_CIR_MODE = 2,
    AIV = 3,
    CPU = 4,
};

class CostModelManager {
public:
    struct RankSizePerLevel {
        u32 level0 = 0;
        u32 level1 = 0;
        u32 level2 = 0;
    };

    CostModelManager();
    ~CostModelManager() = default;
    static CostModelManager* Global();

    // ===== 通信域阶段选路管线(对应运行时 GenerateCostTable) =====
    // 软策略让位: preferred(hcclAlgoPriority>0) 或 tuner 加载(HcclTunerIsLoaded 现查)时跳过 customCheck、排他豁免
    // 生成 costModel: step1 分配填充 → step2 引擎过滤 → step3 HCCL_ALGO 打标 → step4 topo 过滤+标定 → step5
    // topoPriority 排他 candidateEngines/candidatePrefixes: selector 引擎策略(回退序)给定的管线输入
    static HcclResult GenerateCostModel(
        HcclComm comm, CostModel& cm, const OpParam& param, TopoInfoWithNetLayerDetails* topoInfo,
        const std::vector<OpExecuteConfig>& candidateEngines, const std::vector<std::string>& candidatePrefixes);

    static void FreeCostModel(CostModel& costModel);
    void InitBandwidth();
    void InitDpuSftCost();
    RankSizePerLevel CalcRankSizeByTopo(const TopoInfoWithNetLayerDetails* topoInfo) const;

    // n: 每次发送数据量占总数据量的比例
    // netType: 组网类型（COMM_TOPO_1DMESH 或 COMM_TOPO_CLOS）
    // portNum: clos组网下使用的端口数量，mesh组网时为0
    // A: 出参，接收计算得到的A值
    // 计算Mesh算法的A参数
    void CalcMeshParam(
        float n, CommTopo netType, int portNum, u32 rankSize, float& A, bool isPod = false,
        EngineType engine = EngineType::AICPU);
    // 计算NHR算法的A参数; halvePodPort=false 时不做 isPod CLOS 端口减半(单server pod内CLOS无跨pod端口预留)
    void CalcNHRParams(
        float n, CommTopo netType, int portNum, u32 rankSize, float& A, bool isPod = false, bool halvePodPort = true,
        EngineType engine = EngineType::AICPU);
    // n: 输入数据占总数据量DataSize的比例
    // B: 出参，接收计算得到的B值
    // 计算本地拷贝的B参数
    void CalcLocalCopyParams(float n, EngineType scene, float& B);
    // 计算本地reduce的B参数
    void CalcLocalReduceParams(float n, EngineType scene, float& B);
    // 计算Latency参数, taskNum需要写算法的人预估
    void CalcLatencyParams(int taskNum, EngineType engine, float& C);
    // 计算dpu Latency参数
    void CalcDpuLatencyParams(int stepNum, int syncNum, int channelNum, int sndRcvnum, float& C);
    // 计算Launch参数, taskNum需要写算法的人预估
    void CalcLaunchParams(int taskNum, EngineType engine, float& D);
    // 计算一般通信流程的跨卡传输task数, 返回5*(rankSize-1)
    static int CalcTransTaskNum(u32 rankSize);
    // 计算主从流同步的task数, 返回2*(rankSize-1)
    static int CalcSyncTaskNum(u32 rankSize);
    // 公共工厂: 按全量算法构建工作模型(内部复用 InitModel), 供失败回溯重建 HCCL_ALGO 优先级等场景复用
    static HcclResult BuildAllAlgoModel(CostModel& cm);

private:
    // step1: 分配并填充工作模型(条目 count=1, 与 allAlgos 对齐)
    static HcclResult InitModel(CostModel& cm);
    // step2: 引擎过滤(非候选引擎/无名/无 attrs 条目 count 置 0)
    static void FilterByEngine(CostModel& cm, const std::vector<OpExecuteConfig>& candidateEngines);
    // step3: HCCL_ALGO 打标(preferred=+1/否定=-1), 只打标不过滤
    static HcclResult
    ApplyHcclAlgoPriority(HcclComm comm, CostModel& cm, const std::vector<std::string>& candidatePrefixes);
    // step4: topo 过滤 + cost 标定, 原地压缩(count 变为幸存条目数)
    static HcclResult
    FilterByTopoAndCalibrate(HcclComm comm, CostModel& cm, const OpParam& param, TopoInfoWithNetLayerDetails* topoInfo);
    // step5: topoPriority 排他
    static void ApplyTopoPriority(CostModel& cm, const TopoInfoWithNetLayerDetails* topoInfo);

    // 带宽的单位都是GB/s
    float localCopyBw_{};            // 本地拷贝带宽
    float localReduceBw_{};          // 本地reduce带宽
    float crossChipBw_{};            // 跨片带宽
    float crossChipReduceBw_{};      // 跨片reduce带宽
    float roceBw_{};                 // host ROCE链路带宽(跨机ROCE线速)
    float ccuLocalCopyBw_{};         // ccu场景本地拷贝带宽
    float ccuLocalReduceBw_{};       // ccu场景本地reduce带宽
    float ccuCircleLocalCopyBw_{};   // ccu环形场景本地拷贝带宽
    float ccuCircleLocalReduceBw_{}; // ccu环形场景本地reduce带宽

    // dpu软件栈开销
    float preSync_{};
    float channelFence_{};
    float threadFence_{};
    float sndRcvCall_{};
    float hDLatency_{};
};

// CalcCostCoeff 的参数结构体，新增参数只需在此结构体加成员，无需改动所有调用点签名
struct CalcCostCoeffParam {
    u32 rankSize = 0;
    float dataRatio = 0.0f; // 用来说明传入数据量和总数据量之间的关系
    CommTopo netType = CommTopo::COMM_TOPO_1DMESH;
    BufferType inputBuffer;
    BufferType outputBuffer;
    BufferType scratchBuffer;
    std::vector<u32> portNum;
    bool isPod = false;
    const char* algName = nullptr;
    HcclComm comm = nullptr;
    const TopoInfoWithNetLayerDetails* topoInfo = nullptr;
    u32 repeatednum = 1;
    // 跨物理层模板(ZAxisDetour)用: 逐层对应 topomatch 匹配结果; 为空走默认值(旧行为)
    std::vector<PhysicalLevelIndex> phyLevelIdxs = {};   // 各层物理层idx
    std::vector<CommTopo> phyLevelNetTypes = {};         // 各层互联形态
    std::vector<std::vector<u32>> phyLevelPortNums = {}; // 各层各channel端口数(不求和)
};

} // namespace ops_hccl

#endif
