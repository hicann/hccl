/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "selector_test_common.h"
#include "registration_stub.h"
#include <cstring>
#include <map>
#include "selector_registry.h"
#include "reduce_auto_selector.h"
#include "reduce_scatter_auto_selector.h"
#include "algo_name_mapper.h"
#include "op_common.h"
#include "tuner_setup.h"

namespace {
bool g_primitiveAvailable = true;
u8 g_environmentLevel = 0;
u32 g_deterministicLevel = 0;
bool g_tunerLoaded = false;
std::string g_preferredAlgorithm;
std::map<std::pair<HcclComm, std::string>, std::vector<char>> g_contexts;
std::map<HcclComm, u32> g_costModelCreates;
int32_t
WriteReduceStub(ThreadHandle, ChannelHandle, void*, const void*, uint64_t, HcommDataType, HcommReduceOp, uint32_t)
{
    ADD_FAILURE() << "Selector must not execute data transfer";
    return HCCL_E_INTERNAL;
}
} // namespace

namespace inline_reduce_ut {
void SetPrimitiveAvailable(bool available)
{
    g_primitiveAvailable = available;
    HcommPrimitivesDlInit(nullptr);
}
void SetEnvironmentDeterministicLevel(DeterministicEnableLevel level) { g_environmentLevel = static_cast<u8>(level); }
void SetDeterministicLevel(DeterministicEnableLevel level)
{
    g_deterministicLevel = static_cast<u32>(level);
    DlHcommFunction::GetInstance().dlHcclConfigGetInfo = [](HcclComm, HcclConfigType, uint32_t, void* info) {
        *static_cast<u32*>(info) = g_deterministicLevel;
        return HCCL_SUCCESS;
    };
}
u32 CostModelCreationCount(HcclComm comm) { return g_costModelCreates[comm]; }
void SetPreferredAlgorithm(const std::string& name) { g_preferredAlgorithm = name; }
void SetTunerLoaded(bool loaded) { g_tunerLoaded = loaded; }
void ResetSelectorState()
{
    g_contexts.clear();
    g_costModelCreates.clear();
    g_preferredAlgorithm.clear();
    g_tunerLoaded = false;
    SetDpuInlineReduceLogCaptureEnabled(false);
    SetPrimitiveAvailable(true);
    SetEnvironmentDeterministicLevel(DeterministicEnableLevel::DETERMINISTIC_DISABLE);
    SetDeterministicLevel(DeterministicEnableLevel::DETERMINISTIC_DISABLE);
    ClearDpuInlineReduceCapturedLogs();
}
OpParam MakeParam(HcclCMDType op)
{
    OpParam param{};
    param.opType = op;
    param.opExecuteConfig = OpExecuteConfig::HOSTCPU;
    param.DataDes.dataType = HCCL_DATA_TYPE_FP32;
    param.DataDes.count = 1024;
    param.reduceType = HCCL_REDUCE_SUM;
    return param;
}
TopoInfoWithNetLayerDetails MakeTopo(Level0Shape shape)
{
    TopoInfoWithNetLayerDetails topo{};
    topo.deviceType = HcclDevType::DEV_TYPE_950;
    SetDomainCapability(topo, true);
    topo.hostDpuOnly = true;
    topo.userRankSize = 6;
    topo.deviceNumPerModule = 3;
    topo.serverNum = 2;
    topo.topoLevelNums = 2;
    topo.level0Topo = shape;
    topo.netLayerDetails.localNetInsSizeOfLayer = {3, 6};
    topo.netLayerDetails.instSizeListOfLayer = {{3, 3}, {6}};
    return topo;
}
std::string AlgorithmName(HcclCMDType op, bool useInline)
{
    const std::string prefix = op == HcclCMDType::HCCL_CMD_ALLREDUCE ? "DpuAllReduce" :
                               op == HcclCMDType::HCCL_CMD_REDUCE    ? "DpuReduce" :
                                                                       "DpuReduceScatter";
    return prefix + (useInline ? "SequenceMeshNHR" : "SequenceMeshMesh");
}
SelectorStatus SelectLegacy(const OpParam& param, const TopoInfoWithNetLayerDetails& topo, std::string& name)
{
    auto selectors = SelectorRegistry::Global()->GetSelectorsByOpType(param.opType);
    if (selectors.size() != 1)
        return SelectorStatus::NOT_MATCH;
    auto workParam = param;
    auto workTopo = topo;
    return selectors.begin()->second->Select(workParam, &workTopo, name);
}
} // namespace inline_reduce_ut

extern "C" {
void* HcclDlsym(void*, const char* name)
{
    return g_primitiveAvailable && std::strcmp(name, "HcommWriteReduceWithNotifyOnThread") == 0 ?
               reinterpret_cast<void*>(&WriteReduceStub) :
               nullptr;
}
int GetHcommVersion() { return CANN_VERSION(9, 2, 0, 2); }
HcclResult HcclGetHcclBuffer(HcclComm, void**, uint64_t*)
{
    ADD_FAILURE() << "HostDPU selector UT unexpectedly entered AIV resource preparation";
    return HCCL_E_NOT_SUPPORT;
}
HcclResult HcclEngineCtxGet(HcclComm comm, const char* tag, CommEngine, void** ctx, uint64_t* size)
{
    const auto found = g_contexts.find({comm, tag});
    if (found == g_contexts.end()) {
        *ctx = nullptr;
        *size = 0;
        return HCCL_E_NOT_FOUND;
    }
    *ctx = found->second.data();
    *size = found->second.size();
    return HCCL_SUCCESS;
}
HcclResult HcclEngineCtxCreate(HcclComm comm, const char* tag, CommEngine, uint64_t size, void** ctx)
{
    if (std::strncmp(tag, "costmodel_", 10) == 0)
        ++g_costModelCreates[comm];
    auto& buffer = g_contexts[{comm, tag}];
    buffer.resize(size == 0 ? 1 : size);
    *ctx = buffer.data();
    return HCCL_SUCCESS;
}
HcclResult HcclGetDeviceType(HcclDevType& type)
{
    type = HcclDevType::DEV_TYPE_950;
    return HCCL_SUCCESS;
}
}

namespace ops_hccl {
u64 GetDebugConfig() { return 0; }
std::string GetEnv(std::string name)
{
    const char* value = std::getenv(name.c_str());
    return value ? value : "EmptyString";
}
AlgoNameMapper* AlgoNameMapper::Global()
{
    static AlgoNameMapper instance;
    return &instance;
}
const u8& GetExternalInputHcclDeterministic() { return g_environmentLevel; }
const std::map<HcclCMDType, std::vector<HcclAlgoType>> GetExternalInputHcclAlgoConfigAllType() { return {}; }
DlHcommFunction::~DlHcommFunction() = default;
DlHcommFunction& DlHcommFunction::GetInstance()
{
    static DlHcommFunction instance;
    return instance;
}
bool IsNegotiationCommName(const char*) { return false; }
HcclResult CheckHostDPUOnly(HcclComm, const TopoInfoWithNetLayerDetails* topo, bool& only)
{
    only = topo->hostDpuOnly;
    return HCCL_SUCCESS;
}
// 本UT不链接op_common.cc，固定NDA探测结果，确保选路继续验证DPU路径。
HcclResult CheckSupportNda(HcclComm, const TopoInfoWithNetLayerDetails*, bool& isSupportNda)
{
    isSupportNda = false;
    return HCCL_SUCCESS;
}
void AlgoNameMapper::Init(const AllAlgos&) {}
bool AlgoNameMapper::Lookup(const char*, AlgoDims& dims) const
{
    dims.engineUser = "";
    dims.executorUser = "";
    dims.templateUser = "";
    return false;
}
HcclResult HcclGetHcclAlgo(HcclComm, std::string& config)
{
    if (g_preferredAlgorithm.empty())
        config.clear();
    else if (g_preferredAlgorithm.find("AllReduce") != std::string::npos)
        config = "allreduce:sequence{level0=mesh,level1=nhr}";
    else if (g_preferredAlgorithm.find("ReduceScatter") != std::string::npos)
        config = "reducescatter:sequence{level0=mesh,level1=nhr}";
    else
        config = "reduce:sequence{level0=mesh,level1=nhr}";
    return HCCL_SUCCESS;
}
InsCollAlgBase::InsCollAlgBase() = default;
InsCollAlgBase::~InsCollAlgBase() = default;
std::string InsCollAlgBase::Describe() const { return "UT constant-cost executor"; }
HcclResult InsCollAlgBase::FastLaunch(const OpParam&, const CcuFastLaunchCtx*) { return HCCL_E_NOT_SUPPORT; }
HcclResult InsCollAlgBase::RestoreChannelMap(
    const AlgResourceCtxSerializable&, std::vector<std::map<u32, std::vector<ChannelInfo>>>&) const
{
    return HCCL_E_NOT_SUPPORT;
}
HcclResult InsCollAlgBase::OrchestrateWithThread(const OpParam&, const AlgResourceCtxSerializable&, ThreadHandle)
{
    return HCCL_E_NOT_SUPPORT;
}
} // namespace ops_hccl

HcclResult HcclTunerInit(HcclComm, const ops_hccl::TopoInfoWithNetLayerDetails*) { return HCCL_SUCCESS; }
bool HcclTunerIsLoaded() { return g_tunerLoaded; }
HcclResult
HcclTunerCallGetCollInfo(HcclComm, const HcclTunerCollOpInfo*, HcclTunerAlgoCostEntry*, uint32_t, bool* modified)
{
    *modified = false;
    return HCCL_SUCCESS;
}
