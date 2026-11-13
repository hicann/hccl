/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "selector_deps_stub.h"

#include <algorithm>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include <memory>

#include "alg_env_config.h"
#include "alg_parse.h"
#include "coll_alg_v2_exec_registry.h"
#include "algo_name_mapper.h"
#include "config_log.h"
#include "cost_model.h"
#include "cost_table.h"
#include "hccl_res_dl.h"
#include "op_common.h"
#include "tuner_setup.h"

namespace hccl_ut {
namespace {
    std::vector<ops_hccl::OpExecuteConfig>& CandidateEnginesRef()
    {
        static std::vector<ops_hccl::OpExecuteConfig> engines;
        return engines;
    }

    std::vector<std::string>& CandidatePrefixesRef()
    {
        static std::vector<std::string> prefixes;
        return prefixes;
    }

    std::vector<std::string>& CostModelAlgosRef()
    {
        static std::vector<std::string> algos;
        return algos;
    }

    HcclResult& TunerInitResultRef()
    {
        static HcclResult ret = HCCL_SUCCESS;
        return ret;
    }
} // namespace

const std::vector<ops_hccl::OpExecuteConfig>& LastCandidateEngines() { return CandidateEnginesRef(); }

void RecordCandidateEngines(const std::vector<ops_hccl::OpExecuteConfig>& engines) { CandidateEnginesRef() = engines; }

const std::vector<std::string>& LastCandidatePrefixes() { return CandidatePrefixesRef(); }

void RecordCandidatePrefixes(const std::vector<std::string>& prefixes) { CandidatePrefixesRef() = prefixes; }

void SetTunerInitResult(HcclResult ret) { TunerInitResultRef() = ret; }

void SetCostModelAlgos(const std::vector<std::string>& algos) { CostModelAlgosRef() = algos; }

void ResetSelectorStubState()
{
    CandidateEnginesRef().clear();
    CandidatePrefixesRef().clear();
    CostModelAlgosRef().clear();
    TunerInitResultRef() = HCCL_SUCCESS;
}
} // namespace hccl_ut

namespace ops_hccl {
u64 GetDebugConfig() { return 0; }

AllAlgos* GetAllAlgos()
{
    static AllAlgos g_allAlgos = {nullptr, 0, 0};
    return &g_allAlgos;
}

// 与 alg_parse.cc 真实表保持一致(含条目顺序): 桩表缺失某引擎前缀会使对应正向用例选不到算法
const EnginePrefixEntry* GetEnginePrefixEntries(int& count)
{
    static const std::vector<EnginePrefixEntry> entries = {
        {"CcuSched", OpExecuteConfig::CCU_SCHED}, {"CcuMS", OpExecuteConfig::CCU_MS},
        {"Aicpu", OpExecuteConfig::AICPU_TS},     {"Aiv", OpExecuteConfig::AIV},
        {"Dpu", OpExecuteConfig::HOSTCPU},
    };
    count = static_cast<int>(entries.size());
    return entries.data();
}

std::string HcclCMDTypeToString(HcclCMDType opType)
{
    (void)opType;
    return "ut_op";
}

std::string OpExecuteConfigToString(OpExecuteConfig engine)
{
    (void)engine;
    return "ut_engine";
}

HcclResult CheckHostDPUOnly(const HcclComm comm, const TopoInfoWithNetLayerDetails* topoInfo, bool& hostDPUOnly)
{
    (void)comm;
    (void)topoInfo;
    hostDPUOnly = false;
    return HCCL_SUCCESS;
}

// auto_selector_base.cc NDA 判定调用，定义在 op_common.cc（本 UT 不链接），桩为不支持 NDA
HcclResult CheckSupportNda(const HcclComm comm, const TopoInfoWithNetLayerDetails* topoInfo, bool& isSupportNda)
{
    (void)comm;
    (void)topoInfo;
    isSupportNda = false;
    return HCCL_SUCCESS;
}

bool IsNegotiationCommName(const char* commName)
{
    (void)commName;
    return false;
}

TopoMatchResult CheckAlgoMatchTopoWithReason(
    const std::string& algName, const TopoInfoWithNetLayerDetails* topoInfo, bool needSoftPolicyCheck)
{
    (void)algName;
    (void)topoInfo;
    (void)needSoftPolicyCheck;
    return TopoMatchResult{};
}

OpMatchResult CheckAlgoMatchOpWithReason(
    const AlgAttrs& attrs, const OpParam& opParam, const TopoInfoWithNetLayerDetails* topoInfo,
    bool needSoftPolicyCheck)
{
    (void)attrs;
    (void)opParam;
    (void)topoInfo;
    (void)needSoftPolicyCheck;
    return OpMatchResult{};
}

const u8& GetExternalInputHcclDeterministic()
{
    static u8 val = 0;
    return val;
}

const std::map<HcclCMDType, std::vector<HcclAlgoType>> GetExternalInputHcclAlgoConfigAllType() { return {}; }

// 空注入→空 costModel(count=0), Run 走 SelectMinCost 的"选不到算法"路径;
// 注入算法集→按候选前缀过滤(模拟真实管线 step2 引擎过滤), 生成 count>0 模型走正向选中路径
HcclResult CostModelManager::GenerateCostModel(
    HcclComm comm, CostModel& cm, const OpParam& param, TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<OpExecuteConfig>& candidateEngines, const std::vector<std::string>& candidatePrefixes)
{
    (void)comm;
    (void)param;
    (void)topoInfo;
    hccl_ut::RecordCandidateEngines(candidateEngines);
    hccl_ut::RecordCandidatePrefixes(candidatePrefixes);

    // 静态存储: 仅需在 InitCostModel 完成拷贝前有效; reserve 防止 push_back 扩容使已记录的 param 指针失效
    static std::vector<CostAlgoParams> entries;
    static std::vector<CostModelParam> modelParams;
    entries.clear();
    modelParams.clear();
    modelParams.reserve(hccl_ut::CostModelAlgosRef().size());
    for (const std::string& alg : hccl_ut::CostModelAlgosRef()) {
        bool isCandidate
            = std::any_of(candidatePrefixes.begin(), candidatePrefixes.end(), [&alg](const std::string& prefix) {
                  return alg.rfind(prefix, 0) == 0;
              });
        if (!isCandidate) {
            continue;
        }
        modelParams.push_back(CostModelParam{1.0f, 1.0f, 1.0f, 1.0f});
        CostAlgoParams entry = {};
        entry.algName = alg.c_str();
        entry.param = &modelParams.back();
        entry.count = 1;
        entries.push_back(entry);
    }
    cm.costAlgoParams = entries.empty() ? nullptr : entries.data();
    cm.count = static_cast<int>(entries.size());
    return HCCL_SUCCESS;
}

void CostModelManager::FreeCostModel(CostModel& costModel) { (void)costModel; }

// 失败回溯用: 按全量算法建工作模型。UT 不校验 HCCL_ALGO 优先级, 返回空模型即可
// (调用方 GetHcclAlgoPriority 查不到条目时按 0/未配置处理, 与真实实现的退化路径一致)
HcclResult CostModelManager::BuildAllAlgoModel(CostModel& cm)
{
    cm.costAlgoParams = nullptr;
    cm.count = 0;
    return HCCL_SUCCESS;
}

// HCCL_ALGO 优先级打标桩: UT 不覆盖配置解析, 不打标(全部按未配置处理)
HcclResult MarkHcclAlgoPriority(HcclComm comm, CostModel& cm, const std::vector<std::string>& candidateEngineNames)
{
    (void)comm;
    (void)cm;
    (void)candidateEngineNames;
    return HCCL_SUCCESS;
}

// executor 注册表桩: 回溯的 executor 层检查在 UT 中恒视为"未注册"(日志为空操作, 不影响断言)
CollAlgExecRegistryV2& CollAlgExecRegistryV2::Instance()
{
    static CollAlgExecRegistryV2 instance;
    return instance;
}

std::unique_ptr<InsCollAlgBase> CollAlgExecRegistryV2::GetAlgExec(const HcclCMDType type, const std::string& tag)
{
    (void)type;
    (void)tag;
    return nullptr;
}

CostTableManager* CostTableManager::Global()
{
    static CostTableManager instance;
    return &instance;
}

CostTableManager::~CostTableManager() {}

// 由 costModel 填充 costs(每条 cost=1.0f), 空模型保持空表; costs 由 FreeCostTable 释放
HcclResult CostTableManager::GenerateCostTable(
    CostModel& cm, CostTable& ct, const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam)
{
    (void)topoInfo;
    (void)opParam;
    if (cm.count <= 0 || cm.costAlgoParams == nullptr) {
        ct.costs = nullptr;
        ct.count = 0;
        return HCCL_SUCCESS;
    }
    ct.costs = static_cast<AlgoCost*>(malloc(sizeof(AlgoCost) * static_cast<size_t>(cm.count)));
    if (ct.costs == nullptr) {
        return HCCL_E_MEMORY;
    }
    for (int i = 0; i < cm.count; ++i) {
        (void)HcclTunerAlgoCostEntryInit(&ct.costs[i]);
        ct.costs[i].algName = cm.costAlgoParams[i].algName;
        ct.costs[i].engineName = nullptr;
        ct.costs[i].executorName = nullptr;
        ct.costs[i].templateName = nullptr;
        ct.costs[i].cost = 1.0f;
    }
    ct.count = cm.count;
    return HCCL_SUCCESS;
}

void CostTableManager::FreeCostTable(CostTable& ct)
{
    if (ct.costs != nullptr) {
        free(ct.costs);
        ct.costs = nullptr;
    }
    ct.count = 0;
}

AlgoNameMapper* AlgoNameMapper::Global()
{
    static AlgoNameMapper instance;
    return &instance;
}

void AlgoNameMapper::Init(const AllAlgos& allAlgos) { (void)allAlgos; }

bool AlgoNameMapper::Lookup(const char* algName, AlgoDims& dims) const
{
    (void)algName;
    dims.engineUser = "";
    dims.executorUser = "";
    dims.templateUser = "";
    return false;
}

AlgAttrsRegistry& AlgAttrsRegistry::Instance()
{
    static AlgAttrsRegistry instance;
    return instance;
}

const AlgAttrs* AlgAttrsRegistry::Get(const std::string& name) const
{
    (void)name;
    return nullptr;
}
} // namespace ops_hccl

// ---- 全局作用域桩: HCOMM ctx 与 tuner 接口 ----
HcclResult HcclTunerInit(HcclComm comm, const ops_hccl::TopoInfoWithNetLayerDetails* topoInfo)
{
    (void)comm;
    (void)topoInfo;
    return hccl_ut::TunerInitResultRef();
}

bool HcclTunerIsLoaded() { return false; }

HcclResult HcclTunerCallGetCollInfo(
    HcclComm comm, const HcclTunerCollOpInfo* collOpInfo, HcclTunerAlgoCostEntry* algoCostEntries, uint32_t algoCount,
    bool* modified)
{
    (void)comm;
    (void)collOpInfo;
    (void)algoCostEntries;
    (void)algoCount;
    if (modified != nullptr) {
        *modified = false;
    }
    return HCCL_SUCCESS;
}

extern "C" {
// 恒未命中: 逼 Run 走 TunerInit 与 InitCostModel 的创建分支
HcclResult HcclEngineCtxGet(HcclComm comm, const char* ctxTag, CommEngine engine, void** ctx, uint64_t* size)
{
    (void)comm;
    (void)ctxTag;
    (void)engine;
    if (ctx != nullptr) {
        *ctx = nullptr;
    }
    if (size != nullptr) {
        *size = 0;
    }
    return HCCL_E_INTERNAL;
}

HcclResult HcclEngineCtxCreate(HcclComm comm, const char* ctxTag, CommEngine engine, uint64_t size, void** ctx)
{
    (void)comm;
    (void)ctxTag;
    (void)engine;
    if (ctx == nullptr) {
        return HCCL_E_PARA;
    }
    *ctx = malloc(size > 0 ? static_cast<size_t>(size) : 1);
    return (*ctx != nullptr) ? HCCL_SUCCESS : HCCL_E_PARA;
}

HcclResult HcclGetDeviceType(HcclDevType& devType)
{
    devType = HcclDevType::DEV_TYPE_910B;
    return HCCL_SUCCESS;
}
} // extern "C"
