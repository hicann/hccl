/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "executor_v2_base.h"
#include "alg_attrs_registry.h"
#include "cost_model.h"

/* 序列执行器 UT 桩：注册宏静态初始化引出的 host-only 符号与 CANN 包符号 HcclEngineCtxCreate 兜底。 */

namespace ops_hccl {
// AlgAttrsRegistry 桩：支持注入，未注入恒空
namespace {
    std::map<std::string, AlgAttrs>& ProbeAttrsTable()
    {
        static std::map<std::string, AlgAttrs> table;
        return table;
    }
    void*& EngineCtx()
    {
        static void* ctx = nullptr;
        return ctx;
    }
} // namespace

namespace ut_stub {
    void InjectProbeAlgAttrs(const std::string& name, const AlgAttrs& attrs) { ProbeAttrsTable()[name] = attrs; }

    // 记录最近一次 engineCtx 指针
    void RecordEngineCtx(void* ctx) { EngineCtx() = ctx; }
    void ResetEngineCtx()
    {
        std::free(EngineCtx());
        EngineCtx() = nullptr;
    }
    void* LastEngineCtx() { return EngineCtx(); }
} // namespace ut_stub

AlgAttrsRegistry& AlgAttrsRegistry::Instance()
{
    static AlgAttrsRegistry instance;
    return instance;
}

void AlgAttrsRegistry::Register(const AlgAttrs& attrs) { (void)attrs; }

const AlgAttrs* AlgAttrsRegistry::Get(const std::string& name) const
{
    auto& table = ProbeAttrsTable();
    auto it = table.find(name);
    return (it != table.end()) ? &it->second : nullptr;
}

const std::vector<AlgAttrs>& AlgAttrsRegistry::GetAll() const
{
    static const std::vector<AlgAttrs> empty;
    return empty;
}

void AlgAttrsRegistry::ParseAlgName(const std::string& algoName, AlgAttrs& attrs)
{
    (void)algoName;
    (void)attrs;
}

std::string AlgAttrsRegistry::GetAlgAttrsSummary(const AlgAttrs& attrs)
{
    (void)attrs;
    return "";
}

std::string AlgAttrsRegistry::GetDataTypesSummary(const AlgAttrs& attrs)
{
    (void)attrs;
    return "";
}

std::vector<AlgoType> AlgAttrsRegistry::BuildSegAlgoTypes(const char* algName, const std::vector<uint32_t>& segLevelIdx)
{
    (void)algName;
    (void)segLevelIdx;
    return {};
}

// CostModelManager 桩（cost_model.cc 未链接入 UT）：按现行公式保真复刻
CostModelManager::CostModelManager()
{
    localCopyBw_ = 1400.0f * 1000 * 1000 * 1000;
    crossChipBw_ = 56.0f * 1000 * 1000 * 1000;
    roceBw_ = 25.0f * 1000 * 1000 * 1000;
    ccuLocalCopyBw_ = 200.0f * 1000 * 1000 * 1000;
    ccuCircleLocalCopyBw_ = 47.6f * 1024 * 1024 * 1024;
}

CostModelManager* CostModelManager::Global()
{
    static CostModelManager* instance = new CostModelManager;
    return instance;
}

void CostModelManager::CalcMeshParam(
    float n, CommTopo netType, int portNum, u32 rankSize, float& A, bool isPod, EngineType engine)
{
    A = 0.0f;
    double portNumEff = static_cast<double>(portNum);
    if (isPod && netType == CommTopo::COMM_TOPO_CLOS) {
        portNumEff = portNumEff / 2.0;
    }
    float linkBw = crossChipBw_;
    if (engine == EngineType::CPU) {
        linkBw = roceBw_;
    }
    if (netType == CommTopo::COMM_TOPO_1DMESH) {
        A = n / linkBw;
    } else if (netType == CommTopo::COMM_TOPO_CLOS) {
        A = static_cast<float>((n * (rankSize - 1)) / (portNumEff * linkBw));
    }
}

void CostModelManager::CalcLocalCopyParams(float n, EngineType scene, float& B)
{
    float bw = localCopyBw_;
    if (scene == EngineType::CCU) {
        bw = ccuLocalCopyBw_;
    } else if (scene == EngineType::CCU_CIR_MODE) {
        bw = ccuCircleLocalCopyBw_;
    }
    B = n / bw;
}

void CostModelManager::CalcLatencyParams(int taskNum, EngineType engine, float& C)
{
    C = 0.0f;
    if (engine == EngineType::AICPU || engine == EngineType::CCU) {
        C = 0.000002f * taskNum;
    } else if (engine == EngineType::AIV) {
        C = 0.000001f * taskNum;
    }
}

void CostModelManager::CalcLaunchParams(int taskNum, EngineType engine, float& D)
{
    D = 0.0f;
    if (engine == EngineType::AICPU) {
        D = 0.0000005f * taskNum;
    }
}

int CostModelManager::CalcTransTaskNum(u32 rankSize) { return static_cast<int>(5 * (rankSize - 1)); }

int CostModelManager::CalcSyncTaskNum(u32 rankSize) { return static_cast<int>(2 * (rankSize - 1)); }

// cost_model.h 声明（注册宏调用）
HcclResult AddAlgToAllAlgos(
    HcclCMDType opType, const char* algName, const char* executorName, const char** templateName, int templateNum)
{
    (void)opType;
    (void)algName;
    (void)executorName;
    (void)templateName;
    (void)templateNum;
    return HcclResult::HCCL_SUCCESS;
}

// DFX 日志序列化桩
std::string HcclCMDTypeToString(HcclCMDType opType)
{
    (void)opType;
    return "";
}

std::string OpExecuteConfigToString(OpExecuteConfig engine)
{
    (void)engine;
    return "";
}
} // namespace ops_hccl

// CANN hccl_res.h 声明；桩分配真实内存并记录
extern "C" HcclResult
HcclEngineCtxCreate(HcclComm comm, const char* ctxTag, CommEngine engine, uint64_t size, void** ctx)
{
    (void)comm;
    (void)ctxTag;
    (void)engine;
    if (ctx == nullptr) {
        return HCCL_E_PTR;
    }
    *ctx = std::malloc(size == 0 ? 1 : size);
    ops_hccl::ut_stub::RecordEngineCtx(*ctx);
    return (*ctx != nullptr) ? HCCL_SUCCESS : HCCL_E_INTERNAL;
}
