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
bool IsAlgoMatchTopo(const std::string& algName, const TopoInfoWithNetLayerDetails* topoInfo);

// topo 匹配检查结果（不打日志，供调用方决定日志级别）
struct TopoMatchResult {
    bool matched = true;
    std::string reason; // matched=false 时的过滤原因
};
TopoMatchResult CheckAlgoMatchTopoWithReason(const std::string& algName, const TopoInfoWithNetLayerDetails* topoInfo);

typedef struct {
    float A; // 用来描述跨卡传输的时间随DataSize变化的趋势，会受到UB带宽利用率的影响
    float B; // 用来描述本地传输的时间随DataSize变化的趋势，不受到UB带宽利用率的影响
    float C; // 用来描述一些基本时延的常数项
    float D;
} CostModelParam;

typedef struct {
    const char* algName;
    // 所有权：param 指向的内存由 costModel_ 持有（InitCostModel 深拷贝）。
    // FreeCostModel 会逐个释放 param 指向的内存。
    const CostModelParam* param;
    int count;
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

    HcclResult
    InitCostModel(HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, CostModel& costModel, const OpParam& param);
    static void FreeCostModel(CostModel& costModel);
    void InitBandwidth();
    RankSizePerLevel CalcRankSizeByTopo(const TopoInfoWithNetLayerDetails* topoInfo) const;

    // n: 每次发送数据量占总数据量的比例
    // netType: 组网类型（COMM_TOPO_1DMESH 或 COMM_TOPO_CLOS）
    // portNum: clos组网下使用的端口数量，mesh组网时为0
    // A: 出参，接收计算得到的A值
    // 计算Mesh算法的A参数
    void CalcMeshParam(float n, CommTopo netType, int portNum, u32 rankSize, float& A, bool isPod = false);
    // 计算NHR算法的A参数
    void CalcNHRParams(float n, CommTopo netType, int portNum, u32 rankSize, float& A, bool isPod = false);
    // n: 输入数据占总数据量DataSize的比例
    // B: 出参，接收计算得到的B值
    // 计算本地拷贝的B参数
    void CalcLocalCopyParams(float n, EngineType scene, float& B);
    // 计算本地reduce的B参数
    void CalcLocalReduceParams(float n, EngineType scene, float& B);
    // 计算Latency参数, taskNum需要写算法的人预估
    void CalcLatencyParams(int taskNum, EngineType engine, float& C);
    // 计算Launch参数, taskNum需要写算法的人预估
    void CalcLaunchParams(int taskNum, EngineType engine, float& D);
    // 计算一般通信流程的跨卡传输task数, 返回5*(rankSize-1)
    static int CalcTransTaskNum(u32 rankSize);
    // 计算主从流同步的task数, 返回2*(rankSize-1)
    static int CalcSyncTaskNum(u32 rankSize);

private:
    // 带宽的单位都是GB/s
    float localCopyBw_{};            // 本地拷贝带宽
    float localReduceBw_{};          // 本地reduce带宽
    float crossChipBw_{};            // 跨片带宽
    float crossChipReduceBw_{};      // 跨片reduce带宽
    float ccuLocalCopyBw_{};         // ccu场景本地拷贝带宽
    float ccuLocalReduceBw_{};       // ccu场景本地reduce带宽
    float ccuCircleLocalCopyBw_{};   // ccu环形场景本地拷贝带宽
    float ccuCircleLocalReduceBw_{}; // ccu环形场景本地reduce带宽
};

enum class CostAggMode : int {
    SUM = 0, // 多组 cost 求和
    MAX = 1, // 多组 cost 取最大值
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
};

struct AlgNetMeta {
    std::vector<CommTopo> netTypes;                // 每个 template 一个，顺序与 costmodel 中 A/B/C 一致
    std::vector<float> dataRatios;                 // 每个 segment 的 dataRatio，用于查 UB 利用率
    std::vector<u32> rankSizes;                    // 每个 segment 使用的 rankSize
    CostAggMode intraGroupMode = CostAggMode::SUM; // 组内聚合方式
    CostAggMode interGroupMode = CostAggMode::SUM; // 组间聚合方式，默认为SUM
    std::vector<u32> groupSizes;                   // 每组 template 数量，为空时按每组1个兜底
};

class AlgNetMetaRegistry {
public:
    static AlgNetMetaRegistry* Global();
    void Register(const std::string& algName, AlgNetMeta meta);
    bool Query(const std::string& algName, AlgNetMeta& meta) const;

private:
    std::map<std::string, AlgNetMeta> metas_;
    mutable std::mutex mu_;
};

} // namespace ops_hccl

#endif
