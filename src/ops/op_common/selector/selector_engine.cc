/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "selector_engine.h"

#include <memory>
#include <set>

#include "alg_env_config.h"
#include "alg_parse.h"
#include "algo_name_mapper.h"
#include "coll_alg_v2_exec_registry.h"
#include "config_log.h"
#include "cost_model.h"
#include "cost_table.h"
#include "alg_attrs.h"
#include "alg_attrs_registry.h"
#include "hccl_algo_dims.h"
#include "hccl_common.h"
#include "tuner_setup.h"
#include "auto_selector_base.h"
#include "op_common.h"

namespace ops_hccl {

static constexpr const char* COST_MODEL_TAG = "costmodel";
static constexpr const char* TUNER_INIT_TAG = "tuner_init";
static constexpr u32 ALGO_NAME_COL_WIDTH = 48;

SelectorEngine* SelectorEngine::Global()
{
    static SelectorEngine* globalSelectorEngine = new SelectorEngine;
    return globalSelectorEngine;
}

bool SelectorEngine::IsOpSupported(HcclCMDType opType)
{
    // 本迭代新选择器仅支持以下算子, 其他算子走老流程
    static const std::set<HcclCMDType> supportedOps = {
        HcclCMDType::HCCL_CMD_ALLREDUCE,  HcclCMDType::HCCL_CMD_REDUCE_SCATTER,  HcclCMDType::HCCL_CMD_ALLGATHER,
        HcclCMDType::HCCL_CMD_REDUCE,     HcclCMDType::HCCL_CMD_SCATTER,         HcclCMDType::HCCL_CMD_BROADCAST,
        HcclCMDType::HCCL_CMD_BARRIER,    HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, HcclCMDType::HCCL_CMD_ALLTOALLV,
        HcclCMDType::HCCL_CMD_ALLTOALLVC, HcclCMDType::HCCL_CMD_ALLGATHER_V,     HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V,
        HcclCMDType::HCCL_CMD_SEND,       HcclCMDType::HCCL_CMD_RECEIVE,         HcclCMDType::HCCL_CMD_ALLTOALL,
    };
    return supportedOps.count(opType) > 0;
}

// 候选引擎列表转为前缀字符串列表
static std::vector<std::string> CandidateEnginesToPrefixes(const std::vector<OpExecuteConfig>& engines)
{
    std::set<OpExecuteConfig> engineSet(engines.begin(), engines.end());
    std::vector<std::string> prefixes;
    int count = 0;
    const EnginePrefixEntry* entries = GetEnginePrefixEntries(count);
    for (int i = 0; i < count; ++i) {
        if (engineSet.count(entries[i].engine) != 0) {
            prefixes.emplace_back(entries[i].pascal);
        }
    }
    return prefixes;
}

// 候选引擎优先级(回退序)
static std::vector<OpExecuteConfig> GetEnginePriority(OpExecuteConfig opExecuteConfig)
{
    switch (opExecuteConfig) {
        case OpExecuteConfig::CCU_MS:
            return {
                OpExecuteConfig::CCU_MS, OpExecuteConfig::CCU_SCHED, OpExecuteConfig::AICPU_TS,
                OpExecuteConfig::HOSTCPU};
        case OpExecuteConfig::CCU_SCHED:
            return {OpExecuteConfig::CCU_SCHED, OpExecuteConfig::AICPU_TS, OpExecuteConfig::HOSTCPU};
        case OpExecuteConfig::CCU_MS_ONLY:
            return {OpExecuteConfig::CCU_MS};
        case OpExecuteConfig::CCU_SCHED_ONLY:
            return {OpExecuteConfig::CCU_SCHED};
        case OpExecuteConfig::AIV:
            return {OpExecuteConfig::AIV, OpExecuteConfig::AICPU_TS, OpExecuteConfig::HOSTCPU};
        case OpExecuteConfig::AIV_ONLY:
            return {OpExecuteConfig::AIV};
        case OpExecuteConfig::AICPU_TS:
            return {OpExecuteConfig::AICPU_TS, OpExecuteConfig::HOSTCPU};
        case OpExecuteConfig::HOSTCPU:
            return {OpExecuteConfig::HOSTCPU};
        default:
            return {OpExecuteConfig::AICPU_TS, OpExecuteConfig::HOSTCPU};
    }
}

// only 模式(AIV_ONLY/CCU_MS_ONLY/CCU_SCHED_ONLY)对应诊断的基引擎
static OpExecuteConfig GetOnlyModeDiagnoseEngine(OpExecuteConfig opExecuteConfig)
{
    switch (opExecuteConfig) {
        case OpExecuteConfig::CCU_MS_ONLY:
            return OpExecuteConfig::CCU_MS;
        case OpExecuteConfig::CCU_SCHED_ONLY:
            return OpExecuteConfig::CCU_SCHED;
        case OpExecuteConfig::AIV_ONLY:
            return OpExecuteConfig::AIV;
        default:
            return opExecuteConfig;
    }
}

// costModel 按引擎分 tag 存储, Run(查找) 与 InitCostModel(创建) 共用
static std::string MakeCostModelTag(OpExecuteConfig engine)
{
    return std::string(COST_MODEL_TAG) + "_" + ENGINE_STR_MAP.at(engine);
}

// 从全量算法构建并按 HCCL_ALGO 配置标注优先级的临时 costModel(调用方负责释放);
// ctx 里的 costModel 是管线压缩后的幸存集, 被 topo/executor 早期剔除的 preferred 算法查不到,
// 回溯需与 MarkHcclAlgoPriority 同源重建, 避免对 preferred 算法套用其未经历的软策略检查;
// 复刻管线 FilterByEngine 语义: 非候选引擎条目 count=0, 与管线打标时的模型形态一致
static CostModel* BuildPriorityMarkedModel(
    HcclComm comm, const std::vector<OpExecuteConfig>& candidateEngines,
    const std::vector<std::string>& candidatePrefixes)
{
    auto* cm = new (std::nothrow) CostModel{nullptr, 0};
    if (cm == nullptr) {
        return nullptr;
    }
    // BuildAllAlgoModel 内部复用 InitModel; 随后复刻管线 FilterByEngine 语义(非候选引擎条目 count=0)
    if (CostModelManager::BuildAllAlgoModel(*cm) != HCCL_SUCCESS) {
        delete cm;
        return nullptr;
    }
    std::set<OpExecuteConfig> candidateSet(candidateEngines.begin(), candidateEngines.end());
    for (int i = 0; i < cm->count; ++i) {
        const char* algName = cm->costAlgoParams[i].algName;
        const AlgAttrs* attrs = (algName != nullptr) ? AlgAttrsRegistry::Instance().Get(algName) : nullptr;
        if (attrs == nullptr || candidateSet.count(attrs->engine) == 0) {
            cm->costAlgoParams[i].count = 0;
        }
    }
    if (MarkHcclAlgoPriority(comm, *cm, candidatePrefixes) != HCCL_SUCCESS) {
        CostModelManager::FreeCostModel(*cm);
        delete cm;
        return nullptr;
    }
    return cm;
}

static int GetHcclAlgoPriority(const CostModel* cm, const std::string& algName)
{
    if (cm == nullptr) {
        return 0;
    }
    for (int i = 0; i < cm->count; ++i) {
        if (cm->costAlgoParams[i].algName != nullptr && algName == cm->costAlgoParams[i].algName) {
            return cm->costAlgoParams[i].hcclAlgoPriority;
        }
    }
    return 0;
}

// 单算法过滤回溯的公共上下文(逐算法恒定项收敛, 避免长形参串)
struct AlgoFilterCtx {
    HcclComm comm;
    const OpParam* param;
    TopoInfoWithNetLayerDetails* topoInfo;
    const std::set<OpExecuteConfig>* candidateSet;
};

// 单算法过滤回溯(引擎→topo→executor→op), 与选路管线 FilterByTopoAndCalibrate/OpFilterAndCalcCost 阶段一一对应
static void
LogAlgoFilterReason(const AlgoFilterCtx& ctx, const AlgElement& alg, const AlgAttrs& attrs, bool needSoftCheck)
{
    std::string algName = alg.algName;

    // 引擎层: 非候选引擎(回退序见 GetEnginePriority)不参与本场景选路
    if (ctx.candidateSet->count(attrs.engine) == 0) {
        HCCL_ERROR(
            "[SelectorEngine] algName=%s filtered by engine: engine=%s not in candidate engines.", algName.c_str(),
            ENGINE_STR_MAP.at(attrs.engine));
        return;
    }

    // topo 层检查(管线 FilterByTopoAndCalibrate)
    auto topoResult = CheckAlgoMatchTopoWithReason(algName, ctx.topoInfo, needSoftCheck);
    if (!topoResult.matched) {
        HCCL_ERROR("[SelectorEngine] algName=%s filtered by topo: %s.", algName.c_str(), topoResult.reason.c_str());
        return;
    }

    // executor 层检查(管线 topo 后立即现查): 未注册或 CalcCostCoeff 返回空(executor 内部 topo 匹配失败,
    // 如匹配层链路降级)即被过滤
    std::unique_ptr<InsCollAlgBase> exec = CollAlgExecRegistryV2::Instance().GetAlgExec(alg.opType, alg.algName);
    if (exec == nullptr) {
        HCCL_ERROR("[SelectorEngine] algName=%s filtered by executor: not registered.", algName.c_str());
        return;
    }
    if (exec->CalcCostCoeff(ctx.comm, ctx.topoInfo, alg.algName, *ctx.param).empty()) {
        HCCL_ERROR(
            "[SelectorEngine] algName=%s filtered by executor: CalcCostCoeff returned empty (executor topo match "
            "failed, e.g. matched layer link downgraded).",
            algName.c_str());
        return;
    }

    // op 层检查(管线 costtable Phase1 OpFilterAndCalcCost)
    auto opResult = CheckAlgoMatchOpWithReason(attrs, *ctx.param, ctx.topoInfo, needSoftCheck);
    if (!opResult.matched) {
        HCCL_ERROR("[SelectorEngine] algName=%s filtered by op: %s.", algName.c_str(), opResult.reason.c_str());
        return;
    }

    HCCL_ERROR(
        "[SelectorEngine] algName=%s passed engine/topo/executor/op filters; if still unselected, check "
        "opPriority/topoPriority/hcclAlgoPriority exclusion logs.",
        algName.c_str());
}

void SelectorEngine::LogAlgoNotMatchDetails(HcclComm comm, const OpParam& param, TopoInfoWithNetLayerDetails* topoInfo)
{
    // only 模式(AIV_ONLY/CCU_MS_ONLY/CCU_SCHED_ONLY)保留专属报错前缀与明细标题契约(直接报错不回退);
    // AIV_ONLY 前缀取自 auto_selector_base.h 常量单源, 其余场景打通用报错
    const OpExecuteConfig diagnoseEngine = GetOnlyModeDiagnoseEngine(param.opExecuteConfig);
    const bool isOnlyMode = (diagnoseEngine != param.opExecuteConfig);
    std::string prefix;
    std::string detailsTitle;
    if (isOnlyMode) {
        prefix = (param.opExecuteConfig == OpExecuteConfig::AIV_ONLY) ?
                     std::string(AIV_ONLY_NOT_MATCH_PREFIX) :
                     std::string("Failed to select ") + ENGINE_STR_MAP.at(diagnoseEngine)
                         + " algorithm while configured as " + ENGINE_STR_MAP.at(param.opExecuteConfig) + ". ";
        detailsTitle
            = std::string("[SelectorEngine] ") + ENGINE_STR_MAP.at(diagnoseEngine) + " algorithm filter details:";
    } else {
        prefix = std::string("Failed to select algorithm with opExecuteConfig=")
                 + ENGINE_STR_MAP.at(param.opExecuteConfig) + ". ";
        detailsTitle = "[SelectorEngine] candidate algorithm filter details:";
    }
    HCCL_ERROR(
        "%sCurrent topology: topoLevelNums=%u, level0Topo=%u, level0PcieMix=%d, level2UbRtp=%d, "
        "Level1Nhr=%d, userRankSize=%u. opType=%d, dataType=%d, dataSize=%llu, reduceOp=%d.",
        prefix.c_str(), topoInfo->topoLevelNums, static_cast<uint32_t>(topoInfo->level0Topo),
        static_cast<int>(topoInfo->level0PcieMix), static_cast<int>(topoInfo->level2UbRtp),
        static_cast<int>(topoInfo->Level1Nhr), topoInfo->userRankSize, static_cast<int>(param.opType),
        static_cast<int>(param.DataDes.dataType), param.inputSize, static_cast<int>(param.reduceType));

    // 回溯检查所有候选算法被过滤的原因(引擎→topo→executor→op, 与管线阶段一一对应);
    // topoPriority/opPriority/hcclAlgoPriority 排他/收敛阶段恒保留至少一个候选(keep 集为空则不动作),
    // 不会导致空表, 其逐算法剔除由管线 INFO 日志另行打点; 逐算法明细与头部同为 ERROR 级, 保证失败诊断默认可见
    HCCL_ERROR("%s", detailsTitle.c_str());
    const AllAlgos* allAlgos = GetAllAlgos();
    if (allAlgos == nullptr) {
        return;
    }
    // 软策略开关与管线同口径: tuner 接管或 HCCL_ALGO 正向指定(preferred)的算法跳过 customCheck
    const bool tunerLoaded = HcclTunerIsLoaded();
    const std::vector<OpExecuteConfig> candidateEngines = GetEnginePriority(param.opExecuteConfig);
    const std::set<OpExecuteConfig> candidateSet(candidateEngines.begin(), candidateEngines.end());
    const AlgoFilterCtx ctx{comm, &param, topoInfo, &candidateSet};
    CostModel* markedCm
        = BuildPriorityMarkedModel(comm, candidateEngines, CandidateEnginesToPrefixes(candidateEngines));
    if (markedCm == nullptr) {
        HCCL_WARNING("[SelectorEngine] build priority-marked model failed, soft-policy conclusions may be inaccurate.");
    }

    for (int i = 0; i < allAlgos->count; ++i) {
        const AlgElement& alg = allAlgos->algElements[i];
        if (alg.opType != param.opType || alg.algName == nullptr) {
            continue;
        }
        const AlgAttrs* attrs = AlgAttrsRegistry::Instance().Get(alg.algName);
        if (attrs == nullptr) {
            // 与管线 FilterByEngine 同口径: 无 attrs 属注册异常, 亦为一种过滤原因
            HCCL_ERROR("[SelectorEngine] algName=%s filtered by attrs: not registered.", alg.algName);
            continue;
        }
        const bool needSoftCheck = !(tunerLoaded || GetHcclAlgoPriority(markedCm, alg.algName) > 0);
        LogAlgoFilterReason(ctx, alg, *attrs, needSoftCheck);
    }

    if (markedCm != nullptr) {
        CostModelManager::FreeCostModel(*markedCm);
        delete markedCm;
    }
}

// 按引擎 tag 创建/复用通信域级 costModel 并落 ctx
HcclResult
SelectorEngine::InitCostModel(HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, OpParam& param, CostModel*& cm)
{
    HCCL_INFO("[SelectorEngine] Initializing costModel for comm, engine=%d.", static_cast<int>(param.opExecuteConfig));

    // 全局一次性：构建算法名→3D 维度映射缓存
    static std::once_flag algoMapperFlag;
    std::call_once(algoMapperFlag, []() {
        AlgoNameMapper::Global()->Init(*GetAllAlgos());
        HCCL_INFO("[SelectorEngine] algo name mapper initialized.");
    });

    // 通信域阶段选路管线(语义见 cost_model.h); 引擎策略归 selector: 候选引擎(含回退序)及其前缀域作管线输入
    CostModel srcCm{nullptr, 0};

    // RAII 兜底: 管线任何 phase 失败早退均释放临时模型(内容成功拷入 ctx 后的析构释放亦无害)
    struct SrcCmGuard {
        CostModel& cmRef;
        ~SrcCmGuard() { CostModelManager::FreeCostModel(cmRef); }
    } srcCmGuard{srcCm};

    const std::vector<OpExecuteConfig> candidateEngines = GetEnginePriority(param.opExecuteConfig);
    const std::vector<std::string> candidatePrefixes = CandidateEnginesToPrefixes(candidateEngines);
    CHK_RET(CostModelManager::GenerateCostModel(comm, srcCm, param, topoInfo, candidateEngines, candidatePrefixes));

    // 自包含序列化布局: [CostModel 头][CostAlgoParams 条目][pad][CostModelParam 连续区]
    // param 只存内容, 条目指针修正为 buffer 内部地址 → comm 销毁随整块 ctx 释放, 零泄漏
    uint64_t entriesBytes = static_cast<uint64_t>(srcCm.count) * sizeof(CostAlgoParams);
    uint64_t paramOffset = sizeof(CostModel) + entriesBytes;
    paramOffset = (paramOffset + alignof(CostModelParam) - 1) / alignof(CostModelParam) * alignof(CostModelParam);
    uint64_t paramsBytes = 0;
    for (int i = 0; i < srcCm.count; ++i) {
        paramsBytes += static_cast<uint64_t>(srcCm.costAlgoParams[i].count) * sizeof(CostModelParam);
    }
    uint64_t flatSize = paramOffset + paramsBytes;

    void* ctxPtr = nullptr;
    std::string costModelTag = MakeCostModelTag(param.opExecuteConfig);
    CHK_RET(HcclEngineCtxCreate(comm, costModelTag.c_str(), CommEngine::COMM_ENGINE_CPU, flatSize, &ctxPtr));

    CostModel* storedCm = static_cast<CostModel*>(ctxPtr);
    storedCm->count = srcCm.count;
    storedCm->costAlgoParams = reinterpret_cast<CostAlgoParams*>(storedCm + 1);
    if (srcCm.count > 0 && srcCm.costAlgoParams != nullptr) {
        CHK_SAFETY_FUNC_RET(memcpy_s(storedCm->costAlgoParams, entriesBytes, srcCm.costAlgoParams, entriesBytes));
    }

    // param 内容拷入 ctx 连续区, 条目指针改为 buffer 内部地址(等价序列化的 offset 引用)
    char* paramBase = reinterpret_cast<char*>(storedCm) + paramOffset;
    uint64_t offset = 0;
    for (int i = 0; i < srcCm.count; ++i) {
        int n = srcCm.costAlgoParams[i].count;
        const CostModelParam* srcParam = srcCm.costAlgoParams[i].param;
        if (n <= 0 || srcParam == nullptr) {
            continue;
        }
        CostModelParam* dstParam = reinterpret_cast<CostModelParam*>(paramBase + offset);
        CHK_SAFETY_FUNC_RET(memcpy_s(
            dstParam, static_cast<uint64_t>(n) * sizeof(CostModelParam), srcParam,
            static_cast<uint64_t>(n) * sizeof(CostModelParam)));
        storedCm->costAlgoParams[i].param = dstParam;
        offset += static_cast<uint64_t>(n) * sizeof(CostModelParam);
    }

    cm = storedCm;

    HCCL_INFO(
        "[SelectorEngine] costModel initialized and stored in comm ctx, count=%d flatSize=%llu.", storedCm->count,
        flatSize);
    return HCCL_SUCCESS;
}

HcclResult SelectorEngine::TunerEnrichCostTable(HcclComm comm, CostTable& ct, const OpParam& param, bool& tunerModified)
{
    tunerModified = false;
    if (ct.count <= 0 || !HcclTunerIsLoaded()) {
        return HCCL_SUCCESS;
    }
    // tuner: Enrich 填 3D 名 + 调用插件改 cost
    AlgoNameMapper::Global()->Enrich(ct.costs, ct.count);
    // AllToAll(V/VC) 的 dataType 存在 all2AllVDataDes.sendType 中，而非 DataDes.dataType
    HcclDataType tunerDataType = param.DataDes.dataType;
    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV
        || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        tunerDataType = param.all2AllVDataDes.sendType;
    }
    CHK_RET(HcclTunerCallGetCollInfo(
        comm, param.opType, param.inputSize, tunerDataType, ct.costs, ct.count, &tunerModified));
    if (tunerModified) {
        HCCL_INFO("[SelectorEngine] tuner modified cost table.");
    } else {
        HCCL_INFO("[SelectorEngine] tuner did not modify cost table, using CostModel selection.");
    }
    return HCCL_SUCCESS;
}

HcclResult
SelectorEngine::Run(HcclComm comm, OpParam& param, TopoInfoWithNetLayerDetails* topoInfo, std::string& algName)
{
    HCCL_INFO(
        "[SelectorEngine] Run start, opType=%d, opExecuteConfig=%d.", static_cast<int>(param.opType),
        static_cast<int>(param.opExecuteConfig));

    if (AutoSelectorBase::IsRollBackAiv(param, topoInfo)) {
        if (param.opExecuteConfig == OpExecuteConfig::CCU_MS_ONLY
            || param.opExecuteConfig == OpExecuteConfig::CCU_SCHED_ONLY) {
            HCCL_ERROR(
                "[SelectorEngine] need to roll back AIV algo, opType=%d, ccu only mode does not support fallback.",
                static_cast<int>(param.opType));
            return HCCL_E_NOT_SUPPORT;
        }
        if (param.opExecuteConfig != OpExecuteConfig::AIV_ONLY) {
            HCCL_INFO("[SelectorEngine] Need to roll back AIV algo, opType=%d.", static_cast<int>(param.opType));
            param.opExecuteConfig = OpExecuteConfig::AIV_ONLY;
        }
    }

    // step 0: tuner 初始化（每通信域仅一次，独立于 costModel 引擎副本）
    void* tunerCtxPtr = nullptr;
    uint64_t tunerCtxSize = 0;
    if (HcclEngineCtxGet(comm, TUNER_INIT_TAG, CommEngine::COMM_ENGINE_CPU, &tunerCtxPtr, &tunerCtxSize)
        != HCCL_SUCCESS) {
        CHK_RET(HcclTunerInit(comm, topoInfo));
        CHK_RET(HcclEngineCtxCreate(comm, TUNER_INIT_TAG, CommEngine::COMM_ENGINE_CPU, 1, &tunerCtxPtr));
    }

    // step 1: 从通信域 ctx 获取或初始化 costModel（按引擎区分 tag，回退时引擎变更会命中不同副本）
    CostModel* cm = nullptr;
    void* ctxPtr = nullptr;
    uint64_t ctxSize = 0;
    std::string costModelTag = MakeCostModelTag(param.opExecuteConfig);
    if (HcclEngineCtxGet(comm, costModelTag.c_str(), CommEngine::COMM_ENGINE_CPU, &ctxPtr, &ctxSize) == HCCL_SUCCESS) {
        cm = static_cast<CostModel*>(ctxPtr);
        HCCL_INFO("[SelectorEngine] costModel found in comm ctx, count=%d.", cm->count);
    } else {
        CHK_RET(InitCostModel(comm, topoInfo, param, cm));
    }

    // step 2: 生成 costTable(Phase1 过滤+算cost → Phase2 排他 → Phase3 收敛);
    // 软策略(opCustomCheck/opPriority)是否让位由各过滤点按 tuner 加载状态现场判定，硬约束不让位。
    CostTable ct{nullptr, 0};
    // RAII 兜底: tuner 失败等早退路径同样释放 costTable(经 FreeCostTable, 申请/释放同模块)
    struct CostTableGuard {
        CostTable& ctRef;
        ~CostTableGuard() { CostTableManager::FreeCostTable(ctRef); }
    } ctGuard{ct};

    CHK_RET(CostTableManager::Global()->GenerateCostTable(*cm, ct, topoInfo, param));

    // step 3: tuner 改 cost(未加载或空表跳过)
    bool tunerModified = false;
    CHK_RET(TunerEnrichCostTable(comm, ct, param, tunerModified));

    // step 4: min(ct)
    HcclResult ret = SelectMinCost(ct, param, tunerModified, algName);

    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[SelectorEngine] Run failed, no algorithm selected.");
        LogAlgoNotMatchDetails(comm, param, topoInfo);
        return ret;
    }

    LogSelectedAlgo(param, topoInfo, algName);

    return HCCL_SUCCESS;
}

// 将u32列表拼接为"a,b,c"形式, 用于config日志
static std::string JoinUintList(const std::vector<u32>& list)
{
    std::string result;
    for (size_t i = 0; i < list.size(); ++i) {
        if (i > 0) {
            result += ",";
        }
        result += std::to_string(list[i]);
    }
    return result;
}

// 将每层各网络实例的rank数拼接为"8,8;64"形式(层间分号分隔), 用于config日志
static std::string JoinInstSizeListOfLayer(const std::vector<std::vector<u32>>& instSizeListOfLayer)
{
    std::string result;
    for (size_t i = 0; i < instSizeListOfLayer.size(); ++i) {
        if (i > 0) {
            result += ";";
        }
        result += JoinUintList(instSizeListOfLayer[i]);
    }
    return result;
}

void SelectorEngine::LogSelectedAlgo(
    const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo, const std::string& algName)
{
    if (IsNegotiationCommName(param.commName)) {
        HCCL_INFO(
            "[SelectorEngine] The opExecuteConfig is %s, comm is[%s], the negotiation op algo type is %s",
            ENGINE_STR_MAP.at(param.opExecuteConfig), param.commName, algName.c_str());
    } else {
        HCCL_INFO(
            "[SelectorEngine] The opExecuteConfig is %s, comm is[%s], the selected algo type is %s",
            ENGINE_STR_MAP.at(param.opExecuteConfig), param.commName, algName.c_str());
    }

    // rank graph层级/网络实例划分信息; 全部取自序列化字段, topoInfo经ctx缓存(DeSerialize)恢复时同样有效
    std::string netLayersStr = JoinUintList(topoInfo->netLayerDetails.netLayers);
    std::string netInstNumOfLayerStr = JoinUintList(topoInfo->netLayerDetails.netInstNumOfLayer);
    std::string localNetInsSizeOfLayerStr = JoinUintList(topoInfo->netLayerDetails.localNetInsSizeOfLayer);
    // 每层各topoInstance的拓扑类型(按层索引, 与netLayerDetails同维); ranksInTopo为全rank列表、
    // rankNumForTopoType为其派生汇总, 信息冗余且过长, 不打印
    std::vector<std::vector<u32>> topoInstTypeOfLayer;
    topoInstTypeOfLayer.reserve(topoInfo->topoInstDetailsOfLayer.size());
    for (const auto& details : topoInfo->topoInstDetailsOfLayer) {
        std::vector<u32> typeList;
        typeList.reserve(details.typeOfTopo.size());
        for (auto type : details.typeOfTopo) {
            typeList.push_back(static_cast<u32>(type));
        }
        topoInstTypeOfLayer.push_back(std::move(typeList));
    }
    std::string topoInstTypeOfLayerStr = JoinInstSizeListOfLayer(topoInstTypeOfLayer);

    HCCL_CONFIG_INFO(
        HCCL_ALG,
        "op[%s] algName[%s] engine[%s] executor[%s] templates[%s] "
        "opMode[%d] deterministic[%u] "
        "dataSize[%llu] dataType[%s] reduceType[%s] root[%u] "
        "userRank[%u] rankSize[%u] "
        "topoLevelNums[%u] level0Topo[%d] level0MeshType[%d] level0PcieMix[%d] "
        "deviceType[%d] commName[%s] "
        "netLayers[%s] netInstNumOfLayer[%s] localNetInsSizeOfLayer[%s] topoInstTypeOfLayer[%s]",
        HcclCMDTypeToString(param.opType).c_str(), algName.c_str(), ENGINE_STR_MAP.at(param.opExecuteConfig),
        QueryExecutorName(algName).c_str(), QueryTemplateInfo(algName).c_str(), static_cast<int>(param.opMode),
        GetExternalInputHcclDeterministic(), param.inputSize, GetDataTypeEnumStr(GetOpDataType(param)).c_str(),
        GetReduceOpEnumStr(param.reduceType).c_str(), param.root, topoInfo->userRank, topoInfo->userRankSize,
        topoInfo->topoLevelNums, static_cast<int>(topoInfo->level0Topo), static_cast<int>(topoInfo->level0MeshType),
        static_cast<int>(topoInfo->level0PcieMix), static_cast<int>(param.deviceType), param.commName,
        netLayersStr.c_str(), netInstNumOfLayerStr.c_str(), localNetInsSizeOfLayerStr.c_str(),
        topoInstTypeOfLayerStr.c_str());
}

// 按名查找注册的算法条目
static const AlgElement* FindAlgElement(const std::string& algName)
{
    AllAlgos* allAlgos = GetAllAlgos();
    if (allAlgos == nullptr) {
        return nullptr;
    }
    for (int i = 0; i < allAlgos->count; ++i) {
        if (allAlgos->algElements[i].algName != nullptr && algName == allAlgos->algElements[i].algName) {
            return &allAlgos->algElements[i];
        }
    }
    return nullptr;
}

std::string SelectorEngine::QueryTemplateInfo(const std::string& algName)
{
    std::string templateInfo;
    const AlgElement* alg = FindAlgElement(algName);
    if (alg == nullptr) {
        return templateInfo;
    }
    for (int t = 0; t < alg->templateNum; ++t) {
        if (t > 0) {
            templateInfo += ",";
        }
        templateInfo += alg->templateName[t];
    }
    return templateInfo;
}

std::string SelectorEngine::QueryExecutorName(const std::string& algName)
{
    const AlgElement* alg = FindAlgElement(algName);
    return (alg != nullptr && alg->executorName != nullptr) ? alg->executorName : "";
}

HcclResult SelectorEngine::SelectMinCost(const CostTable& ct, OpParam& param, bool tunerModified, std::string& algName)
{
    if (ct.count <= 0) {
        // 拓扑与算子条件由随后的回溯头部统一打印, 此处不重复
        HCCL_ERROR("[SelectorEngine] No support alg for current topoType/opInfo/externalConfig, please check.");
        return HCCL_E_NOT_SUPPORT;
    }

    // 遍历找最小 cost 并打印 costTable 明细, 格式: | idx | algName | cost |
    HCCL_INFO(
        "[SelectorEngine] SelectMinCost: costTable count=%d, opType=%d, dataSize=%llu.", ct.count,
        static_cast<int>(param.opType), param.inputSize);
    HCCL_INFO("[SelectorEngine] "
              "+-----+--------------------------------------------------+--------------+");
    HCCL_INFO("[SelectorEngine] | idx | algName                                          | cost         |");
    HCCL_INFO("[SelectorEngine] "
              "+-----+--------------------------------------------------+--------------+");
    int minIdx = -1;
    float minCost = 0.0f;
    std::vector<std::string> tiedAlgos;
    for (int i = 0; i < ct.count; ++i) {
        const char* name = ct.costs[i].algName;
        float cost = ct.costs[i].cost;
        std::string nameStr = name != nullptr ? name : "-";
        char costBuf[32];
        const char* fmt = (cost >= 0.0f && cost < 1.0f) ? "%.6f" : "%.2f";
        int costRet = sprintf_s(costBuf, sizeof(costBuf), fmt, cost);
        if (costRet < 0) {
            HCCL_ERROR("[SelectorEngine] SelectMinCost: sprintf_s failed.");
            return HCCL_E_INTERNAL;
        }
        HCCL_INFO(
            "[SelectorEngine] | %3d | %-48s | %12s |", i, nameStr.substr(0, ALGO_NAME_COL_WIDTH).c_str(), costBuf);

        if (name == nullptr || cost < 0.0f) {
            continue;
        }
        if (minIdx == -1 || cost < minCost) {
            minIdx = i;
            minCost = cost;
            tiedAlgos.clear();
            tiedAlgos.emplace_back(name);
        } else if (cost == minCost) {
            tiedAlgos.emplace_back(name);
        }
    }
    HCCL_INFO("[SelectorEngine] "
              "+-----+--------------------------------------------------+----------+--------------+----------+");

    if (minIdx < 0) {
        HCCL_ERROR(
            "[SelectorEngine] SelectMinCost: no valid algorithm found, expansionMode=%d, costTable count=%d, "
            "tunerModified=%d.",
            static_cast<int>(param.commOpExpansionMode), ct.count, static_cast<int>(tunerModified));
        // 补充打印被跳过的无效条目(null 名或负 cost, 多为 tuner 标记), 避免与回溯 "passed all filters" 矛盾
        for (int i = 0; i < ct.count; ++i) {
            if (ct.costs[i].algName == nullptr || ct.costs[i].cost < 0.0f) {
                HCCL_ERROR(
                    "[SelectorEngine] SelectMinCost: skipped invalid entry [%d] algName=%s cost=%f.", i,
                    ct.costs[i].algName != nullptr ? ct.costs[i].algName : "(null)", ct.costs[i].cost);
            }
        }
        return HCCL_E_NOT_SUPPORT;
    }

    if (tiedAlgos.size() > 1) {
        std::string algoNames;
        for (size_t i = 0; i < tiedAlgos.size(); ++i) {
            if (i > 0) {
                algoNames += ", ";
            }
            algoNames += tiedAlgos[i];
        }
        HCCL_WARNING(
            "[SelectorEngine] multiple algos with same cost=%f: [%s], selecting %s.", minCost, algoNames.c_str(),
            tiedAlgos[0].c_str());
    }

    algName = ct.costs[minIdx].algName;
    // 引擎取 attrs->engine(Phase 1 保证幸存者必有 attrs)
    const AlgAttrs* selectedAttrs = AlgAttrsRegistry::Instance().Get(algName);
    if (selectedAttrs != nullptr) {
        param.opExecuteConfig = selectedAttrs->engine;
    }
    HCCL_INFO(
        "[SelectorEngine] SelectMinCost: selected algName=%s, engine=%s, cost=%f.", algName.c_str(),
        ENGINE_STR_MAP.at(param.opExecuteConfig), minCost);
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
