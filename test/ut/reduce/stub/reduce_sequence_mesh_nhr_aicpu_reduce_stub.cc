/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Reduce 分层算法 UT 打桩：为被测执行器 TU 的完整模板闭包提供链接边界，
// 签名均取自对应头文件。数据面原语/日志/HCOMM 批量模式等运行期边界为空实现；
// CostModel 参数取非零哨兵值供守卫用例核对公式接线；
// GetNHRStepNum 保留生产公式（模板步数计算依赖其被链接）。
#include <cerrno>
#include <cstring>
#include <string>
#include <vector>

#include "alg_data_trans_wrapper.h"
#include "alg_env_config.h"
#include "auto_selector_base.h"
#include "dpu_inline_reduce_selector.h"
#include "channel.h"
#include "cost_model.h"
#include "hccl_res.h"
#include "log.h"
#include "op_common.h"
#include "securec.h"
#include "template_utils.h"

// ==== 数据面调用记录：LocalCopy/LocalReduce 逐次记录缓冲基址，
// 用例按 userBuf/cclMem 基址区分读写归属，断言段序与 in-place 时序 ====
namespace {
struct StubDataPlaneCall {
    const char* op;
    const void* srcBuff;
    u64 srcOff;
    const void* dstBuff;
    u64 dstOff;
};
std::vector<StubDataPlaneCall> g_stubDataPlaneCalls;
} // namespace

void StubResetDataPlaneCalls() { g_stubDataPlaneCalls.clear(); }

size_t StubFirstCallIdxWithSrcBuff(const void* buff)
{
    for (size_t i = 0; i < g_stubDataPlaneCalls.size(); i++) {
        if (g_stubDataPlaneCalls[i].srcBuff == buff) {
            return i;
        }
    }
    return SIZE_MAX;
}

size_t StubLastCallIdxWithSrcBuff(const void* buff)
{
    for (size_t i = g_stubDataPlaneCalls.size(); i > 0; i--) {
        if (g_stubDataPlaneCalls[i - 1].srcBuff == buff) {
            return i - 1;
        }
    }
    return SIZE_MAX;
}

size_t StubFirstCallIdxWithDstBuff(const void* buff)
{
    for (size_t i = 0; i < g_stubDataPlaneCalls.size(); i++) {
        if (g_stubDataPlaneCalls[i].dstBuff == buff) {
            return i;
        }
    }
    return SIZE_MAX;
}

size_t StubCountCallsWithDstBuff(const void* buff)
{
    size_t n = 0;
    for (const auto& c : g_stubDataPlaneCalls) {
        if (c.dstBuff == buff) {
            n++;
        }
    }
    return n;
}

// in-place 逐 slice 时序：多轮循环下读写交错，按偏移配对证明写不覆盖未读输入
bool StubInplaceReadsBeforeWritesPerSlice(const void* buff)
{
    for (size_t w = 0; w < g_stubDataPlaneCalls.size(); w++) {
        const auto& write = g_stubDataPlaneCalls[w];
        if (write.dstBuff != buff) {
            continue;
        }
        bool readBefore = false;
        for (size_t r = 0; r < g_stubDataPlaneCalls.size(); r++) {
            const auto& read = g_stubDataPlaneCalls[r];
            if (read.srcBuff == buff && read.srcOff == write.dstOff && r < w) {
                readBefore = true;
                break;
            }
        }
        if (!readBefore) {
            return false;
        }
    }
    return true;
}

// 日志桩：HcclCheckLogLevel 返回 false 使日志体不执行（与其它 UT 目录一致，全局非命名空间函数）。
bool HcclCheckLogLevel(int logType, int moduleId)
{
    (void)logType;
    (void)moduleId;
    return false;
}

bool IsErrorToWarn() { return false; }

// HCOMM 资源接口桩：满足链接，返回空地址/零长度
HcclResult HcclGetHcclBuffer(HcclComm comm, void** buffer, uint64_t* size)
{
    (void)comm;
    if (buffer != nullptr) {
        *buffer = nullptr;
    }
    if (size != nullptr) {
        *size = 0;
    }
    return HCCL_SUCCESS;
}

namespace ops_hccl {

// 算法注册元数据落表桩：UT 只校验执行器注册表
HcclResult AddAlgToAllAlgos(
    HcclCMDType opType, const char* algName, const char* executorName, const char** templateName, int templateNum)
{
    (void)opType;
    (void)algName;
    (void)executorName;
    (void)templateName;
    (void)templateNum;
    return HCCL_SUCCESS;
}

// 拓扑判定桩：供 topoCustomCheck 用例经 g_utStubLayerAllConnected 控制全互连开关（默认全互连）
bool g_utStubLayerAllConnected = true;

bool AutoSelectorBase::IsLayerAllConnetedWithTopo(
    const TopoInfoWithNetLayerDetails* topoInfo, const u32 netLayer, const CommTopo topoType)
{
    (void)topoInfo;
    (void)netLayer;
    (void)topoType;
    return g_utStubLayerAllConnected;
}

// 大数据阈值桩：与生产公式一致，行为不受用例控制
bool AutoSelectorBase::IsLargeData(const u64 dataSize) const { return dataSize >= LARGE_COUNT_1024KB; }

// 基类选路虚函数默认体桩：仅用于 vtable/typeinfo 发射，UT 不触达
SelectorStatus AutoSelectorBase::SelectCcuMsAlgo(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& selectAlgName) const
{
    (void)topoInfo;
    (void)opParam;
    (void)configAlgMap;
    (void)selectAlgName;
    return SelectorStatus::NOT_MATCH;
}

SelectorStatus AutoSelectorBase::SelectCcuScheduleAlgo(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& selectAlgName) const
{
    (void)topoInfo;
    (void)opParam;
    (void)configAlgMap;
    (void)selectAlgName;
    return SelectorStatus::NOT_MATCH;
}

SelectorStatus AutoSelectorBase::SelectAicpuAlgo(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& selectAlgName) const
{
    (void)topoInfo;
    (void)opParam;
    (void)configAlgMap;
    (void)selectAlgName;
    return SelectorStatus::NOT_MATCH;
}

SelectorStatus AutoSelectorBase::SelectAivAlgo(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& selectAlgName) const
{
    (void)topoInfo;
    (void)opParam;
    (void)configAlgMap;
    (void)selectAlgName;
    return SelectorStatus::NOT_MATCH;
}

SelectorStatus AutoSelectorBase::SelectDPUAlgo(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& selectAlgName) const
{
    (void)topoInfo;
    (void)opParam;
    (void)configAlgMap;
    (void)selectAlgName;
    return SelectorStatus::NOT_MATCH;
}

// 与生产基类同形（auto_selector_base.cc：NDA 选路默认 NOT_MATCH，reduce selector 未覆写）
SelectorStatus AutoSelectorBase::SelectNdaAlgo(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
    const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& selectAlgName) const
{
    (void)topoInfo;
    (void)opParam;
    (void)configAlgMap;
    (void)selectAlgName;
    return SelectorStatus::NOT_MATCH;
}

// ---------- 数据面原语桩：模板闭包整体编译所需的跨卡/local 搬运入口，UT 不触发 ----------
HcclResult SendWrite(const DataInfo& sendInfo, const ThreadHandle& thread)
{
    (void)sendInfo;
    (void)thread;
    return HCCL_SUCCESS;
}

HcclResult RecvWrite(const DataInfo& recvInfo, const ThreadHandle& thread)
{
    (void)recvInfo;
    (void)thread;
    return HCCL_SUCCESS;
}

HcclResult SendRecvWrite(const SendRecvInfo& sendRecvInfo, const ThreadHandle& thread)
{
    (void)sendRecvInfo;
    (void)thread;
    return HCCL_SUCCESS;
}

HcclResult SendRecvRead(const SendRecvInfo& sendRecvInfo, const ThreadHandle& thread)
{
    (void)sendRecvInfo;
    (void)thread;
    return HCCL_SUCCESS;
}

HcclResult SendRecvBatchWrite(const SendRecvInfo& sendRecvInfo, const ThreadHandle& thread)
{
    (void)sendRecvInfo;
    (void)thread;
    return HCCL_SUCCESS;
}

HcclResult SendRecvBatchRead(const SendRecvInfo& sendRecvInfo, const ThreadHandle& thread)
{
    (void)sendRecvInfo;
    (void)thread;
    return HCCL_SUCCESS;
}

HcclResult SendRecvReadReduce(const SendRecvReduceInfo& sendRecvInfo, const ThreadHandle& thread)
{
    (void)sendRecvInfo;
    (void)thread;
    return HCCL_SUCCESS;
}

HcclResult SendRecvBatchWriteReduce(const SendRecvReduceInfo& sendRecvInfo, const ThreadHandle& thread)
{
    (void)sendRecvInfo;
    (void)thread;
    return HCCL_SUCCESS;
}

HcclResult SendRecvBatchReadReduce(const SendRecvReduceInfo& sendRecvInfo, const ThreadHandle& thread)
{
    (void)sendRecvInfo;
    (void)thread;
    return HCCL_SUCCESS;
}

HcclResult LocalCopy(const ThreadHandle& thread, const DataSlice& srcSlice, const DataSlice& dstSlice)
{
    (void)thread;
    g_stubDataPlaneCalls.push_back({"LocalCopy", srcSlice.addr_, srcSlice.offset_, dstSlice.addr_, dstSlice.offset_});
    return HCCL_SUCCESS;
}

HcclResult LocalReduce(
    const ThreadHandle& thread, const DataSlice& srcSlice, const DataSlice& dstSlice, const HcclDataType dataType,
    const HcclReduceOp reduceOp)
{
    (void)thread;
    (void)dataType;
    (void)reduceOp;
    g_stubDataPlaneCalls.push_back({"LocalReduce", srcSlice.addr_, srcSlice.offset_, dstSlice.addr_, dstSlice.offset_});
    return HCCL_SUCCESS;
}

HcclResult PreSyncInterThreads(
    const ThreadHandle& mainThread, const std::vector<ThreadHandle>& subThreads,
    const std::vector<u32>& notifyIdxMainToSub)
{
    (void)mainThread;
    (void)subThreads;
    (void)notifyIdxMainToSub;
    return HCCL_SUCCESS;
}

HcclResult PostSyncInterThreads(
    const ThreadHandle& mainThread, const std::vector<ThreadHandle>& subThreads,
    const std::vector<u32>& notifyIdxSubToMain)
{
    (void)mainThread;
    (void)subThreads;
    (void)notifyIdxSubToMain;
    return HCCL_SUCCESS;
}

// ---------- 模板公共工具桩 ----------
HcclResult CalcDataSplitByPortGroupCommon(
    const u64 totalDataCount, const u64 dataTypeSize, const std::vector<ChannelInfo>& channels,
    std::vector<u64>& elemCountOut, std::vector<u64>& sizeOut, std::vector<u64>& elemOffset, const u32 channelsPerRank)
{
    // 与生产同形的分片公式：空实现会使模板索引空向量，保留加权分片（末通道吃余量）
    elemCountOut.clear();
    sizeOut.clear();
    elemOffset.clear();

    std::vector<u32> portGroups;
    u32 totalPorts = 0;
    u32 taskCount
        = (static_cast<int>(channels.size()) > channelsPerRank) ? channelsPerRank : static_cast<int>(channels.size());
    for (u32 i = 0; i < taskCount; i++) {
        portGroups.push_back(channels[i].portGroupSize);
        totalPorts += channels[i].portGroupSize;
    }

    u32 channelsize = portGroups.size();
    u64 accumCount = 0;
    u64 offset = 0;
    for (u32 channelIdx = 0; channelIdx < channelsize; channelIdx++) {
        u64 elemCount = 0;
        u64 elemSize = 0;
        if (channelIdx == channelsize - 1) {
            elemCount = totalDataCount - accumCount;
        } else {
            CHK_PRT_RET(
                totalPorts == 0, HCCL_ERROR("[CalcDataSplitByPortGroup] totalPorts [%u] is 0.", totalPorts),
                HcclResult::HCCL_E_INTERNAL);
            elemCount = static_cast<u64>((totalDataCount * portGroups[channelIdx]) / totalPorts);
        }
        elemOffset.push_back(offset);
        elemCountOut.push_back(elemCount);
        elemSize = elemCount * dataTypeSize;
        sizeOut.push_back(elemSize);
        offset += elemSize;
        accumCount += elemCount;
    }

    return HcclResult::HCCL_SUCCESS;
}

// Z 轴绕行模板的按带宽切分桩：UT 不触发 KernelRun，仅需链接通过。
HcclResult CalcDataSplitByBandwidthZAxisDetour(
    const u64 totalDataCount, const u64 dataTypeSize, const std::vector<ChannelInfo>& channels,
    std::vector<u64>& elemCountOut, std::vector<u64>& sizeOut, std::vector<u64>& elemOffset,
    const u32 level0ChannelNumPerRank, const u32 level1ChannelNumPerRank, const u32 serverRankSize)
{
    (void)totalDataCount;
    (void)dataTypeSize;
    (void)channels;
    (void)elemCountOut;
    (void)sizeOut;
    (void)elemOffset;
    (void)level0ChannelNumPerRank;
    (void)level1ChannelNumPerRank;
    (void)serverRankSize;
    return HCCL_SUCCESS;
}

HcclResult GetAlgRank(const u32 virtRank, const std::vector<u32>& rankIds, u32& algRank)
{
    (void)virtRank;
    (void)rankIds;
    algRank = 0;
    return HCCL_SUCCESS;
}

u32 GetNHRStepNum(u32 rankSize)
{
    // 与生产逐字一致，含 rankSize==0 无符号下溢返回 32 的退化行为
    u32 nSteps = 0;
    for (u32 tmp = rankSize - 1; tmp != 0; tmp >>= 1, nSteps++) {
    }
    return nSteps;
}

HcclResult FillChannelSymWinPeerAddrs(
    void* inputSymWindow, u64 inputOffset, void* outputSymWindow, u64 outputOffset, ChannelInfo& channel)
{
    (void)inputSymWindow;
    (void)inputOffset;
    (void)outputSymWindow;
    (void)outputOffset;
    (void)channel;
    return HCCL_SUCCESS;
}

// ---------- 通道申请桩 ----------
HcclResult CalcChannelRequestMesh1D(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc>& channels)
{
    (void)comm;
    (void)param;
    (void)topoInfo;
    (void)subcommInfo;
    (void)channels;
    return HCCL_SUCCESS;
}

HcclResult CalcChannelRequestMesh1DLevel0(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc>& channels)
{
    (void)comm;
    (void)param;
    (void)topoInfo;
    (void)subcommInfo;
    (void)channels;
    return HCCL_SUCCESS;
}

HcclResult CalcChannelRequestMesh1DLevel1(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc>& channels)
{
    (void)comm;
    (void)param;
    (void)topoInfo;
    (void)subcommInfo;
    (void)channels;
    return HCCL_SUCCESS;
}

// DPU 选路符号桩：夹具未设置 deviceType（默认 DEV_TYPE_COUNT，非 950/960），生产决策同样恒 false
bool CanUseDpuInlineReduce(const OpParam& opParam, const TopoInfoWithNetLayerDetails* topoInfo)
{
    (void)opParam;
    (void)topoInfo;
    return false;
}

// master 新增（CalcRes 路径链接需要），与同族通道请求桩同形
HcclResult CalcChannelRequestMeshClosMultiJetty(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc>& channels, bool isIsolation,
    bool exceptMesh)
{
    (void)comm;
    (void)param;
    (void)topoInfo;
    (void)subcommInfo;
    (void)channels;
    (void)isIsolation;
    (void)exceptMesh;
    return HCCL_SUCCESS;
}

HcclResult CalcChannelRequestMesh1DWithPriorityTopo(
    HcclComm comm, const OpParam& param, const TopoInfo* topoInfo, const std::vector<std::vector<u32>>& subcommInfo,
    std::vector<HcclChannelDesc>& channels, CommTopo priorityTopo)
{
    (void)comm;
    (void)param;
    (void)topoInfo;
    (void)subcommInfo;
    (void)channels;
    (void)priorityTopo;
    return HCCL_SUCCESS;
}

HcclResult CalcChannelRequestNhr(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc>& channels)
{
    (void)comm;
    (void)param;
    (void)topoInfo;
    (void)subcommInfo;
    (void)channels;
    return HCCL_SUCCESS;
}

HcclResult CalcChannelRequestNhrMultiJettyUbx(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc>& channels, u64 isoThreshold)
{
    (void)comm;
    (void)param;
    (void)topoInfo;
    (void)subcommInfo;
    (void)channels;
    (void)isoThreshold;
    return HCCL_SUCCESS;
}

// ---------- 环境与 DFX 桩 ----------
std::string GetEnv(std::string IdName)
{
    (void)IdName;
    return std::string();
}

HcclResult HcclGetHcclAlgo(HcclComm comm, std::string& hcclAlgo)
{
    (void)comm;
    (void)hcclAlgo;
    return HCCL_SUCCESS;
}

// ---------- CostModel 桩：模板 CalcCostCoeff 仅在选路阶段调用，UT 不涉及代价计算 ----------
CostModelManager::CostModelManager() {}

CostModelManager* CostModelManager::Global()
{
    static CostModelManager manager;
    return &manager;
}

void CostModelManager::CalcMeshParam(
    float n, CommTopo netType, int portNum, u32 rankSize, float& A, bool isPod, EngineType engine)
{
    // 哨兵值：供守卫用例核对段级公式接线（Mesh 类段 A=1.0）
    (void)n;
    (void)netType;
    (void)portNum;
    (void)rankSize;
    (void)isPod;
    (void)engine;
    A = 1.0f;
}

void CostModelManager::CalcLocalCopyParams(float n, EngineType scene, float& B)
{
    // 哨兵值：本地拷贝 B=3.0
    (void)n;
    (void)scene;
    B = 3.0f;
}

void CostModelManager::CalcNHRParams(
    float n, CommTopo netType, int portNum, u32 rankSize, float& A, bool isPod, bool halvePodPort, EngineType engine)
{
    // 哨兵值：NHR 环搬运 A=2.0
    (void)n;
    (void)netType;
    (void)portNum;
    (void)rankSize;
    (void)isPod;
    (void)halvePodPort;
    (void)engine;
    A = 2.0f;
}

void CostModelManager::CalcLocalReduceParams(float n, EngineType scene, float& B)
{
    // 哨兵值：AICPU 本地规约 B=4.0
    (void)n;
    (void)scene;
    B = 4.0f;
}

void CostModelManager::CalcLatencyParams(int taskNum, EngineType engine, float& C)
{
    // 哨兵值：时延 C=5.0
    (void)taskNum;
    (void)engine;
    C = 5.0f;
}

void CostModelManager::CalcLaunchParams(int taskNum, EngineType engine, float& D)
{
    // 按引擎分支与生产同形（cost_model.cc）：AICPU 路径 D=5e-7*taskNum，AIV/CCU 为 0
    if (engine == EngineType::AICPU) {
        D = 0.0000005f * static_cast<float>(taskNum);
    } else {
        D = 0.0f;
    }
}

int CostModelManager::CalcTransTaskNum(u32 rankSize)
{
    // 哨兵值：非零，保证 taskNum>0、D=5e-7*taskNum>0 可断言（系数与 CalcLaunchParams 一致）
    (void)rankSize;
    return 1;
}

int CostModelManager::CalcSyncTaskNum(u32 rankSize)
{
    // 哨兵值：非零，保证 taskNum>0、D=5e-7*taskNum>0 可断言（系数与 CalcLaunchParams 一致）
    (void)rankSize;
    return 2;
}

} // namespace ops_hccl

// HCOMM 线程 join 桩：部分模板的 KernelRun 会调用它等待本流任务排空。
// 真实实现由 hcomm_primitives_dl.cc 经 dlsym 提供（weak 定义），UT 内用强符号桩替代，
// 仅表示"等待成功"，不做任何同步。
HcclResult HcommThreadJoin(ThreadHandle thread, uint32_t timeout)
{
    (void)thread;
    (void)timeout;
    return HCCL_SUCCESS;
}

extern "C" {

// HCOMM 批量模式桩：真实实现经 dlsym 加载，UT 不触发。
int32_t HcommBatchModeStart(const char* batchTag)
{
    (void)batchTag;
    return 0;
}

int32_t HcommBatchModeEnd(const char* batchTag)
{
    (void)batchTag;
    return 0;
}

// 平台接口桩：executor_v2_base 引用，UT 不触发。
HcclResult HcclEngineCtxCreate(HcclComm comm, const char* ctxTag, CommEngine engine, uint64_t size, void** ctx)
{
    (void)comm;
    (void)ctxTag;
    (void)engine;
    (void)size;
    (void)ctx;
    return HCCL_SUCCESS;
}

// securec 桩：executor_v2_base 的 CCU FastLaunch 路径引用（UT 不触发）。
// 拷贝语义对齐 test/ut/aicpu_task_cache/stub/securec_stub.cc：真实拷贝 + 基本边界检查，
// 避免被测代码误触发时被无操作成功掩盖。
errno_t strcpy_s(char* strDest, size_t destMax, const char* strSrc)
{
    if (strDest == nullptr || strSrc == nullptr || destMax == 0) {
        return EINVAL;
    }
    const size_t length = strlen(strSrc);
    if (length >= destMax) {
        strDest[0] = '\0';
        return ERANGE;
    }
    (void)memcpy(strDest, strSrc, length + 1);
    return EOK;
}

} // extern "C"
