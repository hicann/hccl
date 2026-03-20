/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log.h"
#include "hcom_dl.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

// 定义全局函数指针
HcclResult (*hcomGetRankSizePtr)(const char*, u32*) = nullptr;
HcclResult (*hcomGetLocalRankSizePtr)(const char*, u32*) = nullptr;
HcclResult (*hcomGetRankIdPtr)(const char*, u32*) = nullptr;
HcclResult (*hcomGetLocalRankIdPtr)(const char*, u32*) = nullptr;
HcclResult (*hcomGetWorldRankFromGroupRankPtr)(const char*, u32, u32*) = nullptr;
HcclResult (*hcomGetGroupRankFromWorldRankPtr)(u32, const char*, u32*) = nullptr;
HcclResult (*hcomCreateGroupPtr)(const char*, u32, u32*) = nullptr;
HcclResult (*hcomDestroyGroupPtr)(const char*) = nullptr;
HcclResult (*hcomSetGradFusionByIndexPtr)(const char*, u32, const u32*) = nullptr;
HcclResult (*hcomSetGradFusionBySizePtr)(const char*, u32, const float*) = nullptr;
HcclResult (*hcomInitByRankTablePtr)(const char*, uint32_t) = nullptr;
HcclResult (*hcomDestroyPtr)() = nullptr;
HcclResult (*hcomGetCommHandleByGroupPtr)(const char*, HcclComm*) = nullptr;
HcclResult (*getGroupNameByOpBaseHcomPtr)(s64, char**) = nullptr;
HcclResult (*hcomCreateComResourceByCommPtr)(HcclComm, u32, bool, void**, bool) = nullptr;
void (*hcomTopoInfoRegCallbackPtr)(HcclResult (*)(const char *, uint32_t), void (*)(const char *)) = nullptr;
HcclResult (*hcomGetandClearOverFlowTasksPtr)(const char*, hccl::HcclDumpInfo**, s32*) = nullptr;
HcclWorkflowMode (*hcomGetWorkflowModePtr)() = nullptr;
HcclResult (*hcomSetWorkflowModePtr)(HcclWorkflowMode) = nullptr;
HcclResult (*hcomCalcOpOnlinePtr)(HcomOpParam*, HcomResResponse*) = nullptr;
HcclResult (*hcomCalcOpResOfflinePtr)(HcomOpParam*, HcomResResponse*) = nullptr;
HcclResult (*hcomGetMemTypePtr)(const char*, const char*, bool, u32*, bool*, bool, bool) = nullptr;
HcclResult (*hcomGetBandWidthPerNPUPtr)(u32, float*) = nullptr;
HcclResult (*hcomGetServerNumAndDeviceNumPerServerPtr)(u32*, u32*, u32*) = nullptr;
bool (*hcomGetSecAddrCopyFlagPtr)(const char*) = nullptr;
HcclResult (*hcomInitByStringPtr)(const char*, const char*, WorkMode, HcomInitConfig*) = nullptr;
HcclResult (*hcomInitByMasterInfoPtr)(const char*, const char*, const char*, const char*, const char*, HcomInitConfig*) = nullptr;
HcclResult (*hcomCreateCommCCLbufferPtr)(const char*) = nullptr;
HcclResult (*hcomGetInCCLbufferPtr)(const char*, void**, u64*) = nullptr;
HcclResult (*hcomGetOutCCLbufferPtr)(const char*, void**, u64*) = nullptr;
void (*hcomSetLaunchKernelModePtr)(bool) = nullptr;
HcclResult (*hcomGetAicpuOpStreamNotifyPtr)(const char*, HcclRtStream*, u8, void**) = nullptr;
HcclResult (*hcomMc2AiCpuStreamAllocAndGetPtr)(const char*, u32, rtStream_t*) = nullptr;
void (*hcomSetDumpDebugModePtr)(bool) = nullptr;
HcclResult (*hcomGetAlgorithmPtr)(u32, char**) = nullptr;
HcclResult (*hcomGetAlgExecParamPtr)(const char*, const char*, u64, void*, void*, HcclCMDType, bool, HcclDataType, HcclReduceOp, void**, u64*, u32) = nullptr;
void (*hcomSetAutoTuneModePtr)(bool) = nullptr;
DevType (*hcomGetDeviceTypePtr)() = nullptr;
HcclResult (*hcomSetProfilingModePtr)(HcomProfilingMode, const char*) = nullptr;
HcclResult (*hcomGetSplitStrategyPtr)(const char*, const struct model_feature*, u32**, u32*, bool*, GradSplitForceMode, OriginalGraphShapeType) = nullptr;
bool (*hcomFindGroupPtr)(const char*) = nullptr;
HcclResult (*hcomSelectAlgPtr)(s64, const char*, u64, void*, HcclDataType, HcclReduceOp, HcclCMDType, int32_t, bool&, char*) = nullptr;
HcclResult (*hcomCalcAivCoreNumPtr)(const char*, HcclCMDType, u64, void*, HcclDataType, int32_t, char*, u32*) = nullptr;
HcclResult (*hcomSetWorkspaceResourcePtr)(const char*, const char*, rtStream_t*, s32, void*, u64) = nullptr;
HcclResult (*hcomSetGlobalWorkSpacePtr)(const char*, void**, u32) = nullptr;
HcclResult (*hcomSetAivCoreLimitPtr)(const char*, u32) = nullptr;
HcclResult (*hcomReleaseSubCommsPtr)() = nullptr;
HcclResult (*hcomUnloadTaskPtr)(const char*, const char*) = nullptr;
HcclResult (*hcomClearAivSyncBufPtr)(const char*, bool) = nullptr;
HcclResult (*hcomSetAttachedStreamPtr)(const char*, u32, const rtStream_t*, s32) = nullptr;
HcclResult (*hcomSupportDeterministicOptimPtr)(const char*, bool*) = nullptr;
HcclResult (*hcomTbeMemCleanPtr)(int64_t[], int64_t[], uint32_t, rtStream_t, int32_t) = nullptr;
HcclResult (*hcomGetInitStatusPtr)(bool*) = nullptr;
HcclResult (*hcomAllGatherPtr)(const char*, void*, void*, u64, HcclDataType, const char*, rtStream_t) = nullptr;
HcclResult (*hcomAllGatherVPtr)(const char*, const void*, u64, const void*, const void*, const void*, HcclDataType, const char*, rtStream_t) = nullptr;
HcclResult (*hcomAllReducePtr)(const char*, void*, void*, u64, HcclDataType, HcclReduceOp, const char*, rtStream_t) = nullptr;
HcclResult (*hcomReducePtr)(const char*, void*, void*, u64, HcclDataType, HcclReduceOp, u32, const char*, rtStream_t) = nullptr;
HcclResult (*hcomBroadcastPtr)(const char*, void*, u64, HcclDataType, u32, const char*, rtStream_t) = nullptr;
HcclResult (*hcomReduceScatterPtr)(const char*, void*, void*, u64, HcclDataType, HcclReduceOp, const char*, rtStream_t) = nullptr;
HcclResult (*hcomReduceScatterVPtr)(const char*, void*, const void*, const void*, void*, u64, HcclDataType, HcclReduceOp, const char*, rtStream_t) = nullptr;
HcclResult (*hcomSendPtr)(const char*, void*, u64, HcclDataType, u32, u32, const char*, rtStream_t) = nullptr;
HcclResult (*hcomReceivePtr)(const char*, void*, u64, HcclDataType, u32, u32, const char*, rtStream_t) = nullptr;
HcclResult (*hcomAlltoAllVPtr)(const void*, const void*, const void*, HcclDataType, const void*, const void*, const void*, HcclDataType, const char*, rtStream_t, const char*) = nullptr;
HcclResult (*hcomAlltoAllVCPtr)(const void*, const void*, HcclDataType, const void*, HcclDataType, const char*, rtStream_t, const char*) = nullptr;
HcclResult (*hcomAllToAllPtr)(const void*, u64, HcclDataType, const void*, u64, HcclDataType, const char*, rtStream_t, const char*) = nullptr;
HcclResult (*hcomGetHcclCommPtr)(int64_t, std::string&) = nullptr;
HcclResult (*hcomGenerateCclOpTagPtr)(const char*, s64, const char*, char*) = nullptr;
HcclResult (*hcomGetCommCCLBufferSizePtr)(const char*, uint64_t&) = nullptr;
HcclResult (*hcomGetL0TopoTypeExPtr)(const char*, CommTopo*, uint32_t) = nullptr;
HcclResult (*hcomGetRankSizeExPtr)(const char*, uint32_t*, uint32_t) = nullptr;
static HcclResult (*hcomInitByFilePtr)(const char*, const char*) = nullptr;
static HcclResult (*hcomGetWorkspaceSubStreamNumPtr)(const char*, u64&, u64, HcclDataType, u32, HcclReduceOp, u64, HcclCMDType) = nullptr;
static HcclResult (*hcomGetWorkspaceMemSizePtr)(const std::string&, u64, HcclDataType, const char*, u64&) = nullptr;
static HcclResult (*hcomSetAlgorithmPtr)(const char*) = nullptr;
static HcclResult (*hcomGetAlltoAllStagedWorkSpaceMemSizePtr)(const char*, u64*, u64*, HcclDataType, u64*, u64*, HcclDataType, u64&) = nullptr;

// 支持标志（静态，默认 false）
#define DEFINE_SUPPORT_FLAG(name) static bool g_##name##Supported = false

DEFINE_SUPPORT_FLAG(HcomGetRankSize);
DEFINE_SUPPORT_FLAG(HcomGetLocalRankSize);
DEFINE_SUPPORT_FLAG(HcomGetRankId);
DEFINE_SUPPORT_FLAG(HcomGetLocalRankId);
DEFINE_SUPPORT_FLAG(HcomGetWorldRankFromGroupRank);
DEFINE_SUPPORT_FLAG(HcomGetGroupRankFromWorldRank);
DEFINE_SUPPORT_FLAG(HcomCreateGroup);
DEFINE_SUPPORT_FLAG(HcomDestroyGroup);
DEFINE_SUPPORT_FLAG(HcomSetGradFusionByIndex);
DEFINE_SUPPORT_FLAG(HcomSetGradFusionBySize);
DEFINE_SUPPORT_FLAG(HcomInitByRankTable);
DEFINE_SUPPORT_FLAG(HcomDestroy);
DEFINE_SUPPORT_FLAG(HcomGetCommHandleByGroup);
DEFINE_SUPPORT_FLAG(GetGroupNameByOpBaseHcom);
DEFINE_SUPPORT_FLAG(HcomCreateComResourceByComm);
DEFINE_SUPPORT_FLAG(HcomTopoInfoRegCallback);
DEFINE_SUPPORT_FLAG(HcomGetandClearOverFlowTasks);
DEFINE_SUPPORT_FLAG(HcomGetWorkflowMode);
DEFINE_SUPPORT_FLAG(HcomSetWorkflowMode);
DEFINE_SUPPORT_FLAG(HcomCalcOpOnline);
DEFINE_SUPPORT_FLAG(HcomCalcOpResOffline);
DEFINE_SUPPORT_FLAG(HcomGetMemType);
DEFINE_SUPPORT_FLAG(HcomGetBandWidthPerNPU);
DEFINE_SUPPORT_FLAG(HcomGetServerNumAndDeviceNumPerServer);
DEFINE_SUPPORT_FLAG(HcomGetSecAddrCopyFlag);
DEFINE_SUPPORT_FLAG(HcomInitByString);
DEFINE_SUPPORT_FLAG(HcomInitByMasterInfo);
DEFINE_SUPPORT_FLAG(HcomCreateCommCCLbuffer);
DEFINE_SUPPORT_FLAG(HcomGetInCCLbuffer);
DEFINE_SUPPORT_FLAG(HcomGetOutCCLbuffer);
DEFINE_SUPPORT_FLAG(HcomSetLaunchKernelMode);
DEFINE_SUPPORT_FLAG(HcomGetAicpuOpStreamNotify);
DEFINE_SUPPORT_FLAG(HcomMc2AiCpuStreamAllocAndGet);
DEFINE_SUPPORT_FLAG(HcomSetDumpDebugMode);
DEFINE_SUPPORT_FLAG(HcomGetAlgorithm);
DEFINE_SUPPORT_FLAG(HcomGetAlgExecParam);
DEFINE_SUPPORT_FLAG(HcomSetAutoTuneMode);
DEFINE_SUPPORT_FLAG(HcomGetDeviceType);
DEFINE_SUPPORT_FLAG(HcomSetProfilingMode);
DEFINE_SUPPORT_FLAG(HcomGetSplitStrategy);
DEFINE_SUPPORT_FLAG(HcomFindGroup);
DEFINE_SUPPORT_FLAG(HcomSelectAlg);
DEFINE_SUPPORT_FLAG(HcomCalcAivCoreNum);
DEFINE_SUPPORT_FLAG(HcomSetWorkspaceResource);
DEFINE_SUPPORT_FLAG(HcomSetGlobalWorkSpace);
DEFINE_SUPPORT_FLAG(HcomSetAivCoreLimit);
DEFINE_SUPPORT_FLAG(HcomReleaseSubComms);
DEFINE_SUPPORT_FLAG(HcomUnloadTask);
DEFINE_SUPPORT_FLAG(HcomClearAivSyncBuf);
DEFINE_SUPPORT_FLAG(HcomSetAttachedStream);
DEFINE_SUPPORT_FLAG(HcomSupportDeterministicOptim);
DEFINE_SUPPORT_FLAG(HcomTbeMemClean);
DEFINE_SUPPORT_FLAG(HcomGetInitStatus);
DEFINE_SUPPORT_FLAG(HcomAllGather);
DEFINE_SUPPORT_FLAG(HcomAllGatherV);
DEFINE_SUPPORT_FLAG(HcomAllReduce);
DEFINE_SUPPORT_FLAG(HcomReduce);
DEFINE_SUPPORT_FLAG(HcomBroadcast);
DEFINE_SUPPORT_FLAG(HcomReduceScatter);
DEFINE_SUPPORT_FLAG(HcomReduceScatterV);
DEFINE_SUPPORT_FLAG(HcomSend);
DEFINE_SUPPORT_FLAG(HcomReceive);
DEFINE_SUPPORT_FLAG(HcomAlltoAllV);
DEFINE_SUPPORT_FLAG(HcomAlltoAllVC);
DEFINE_SUPPORT_FLAG(HcomAllToAll);
DEFINE_SUPPORT_FLAG(HcomGetHcclComm);
DEFINE_SUPPORT_FLAG(HcomGenerateCclOpTag);
DEFINE_SUPPORT_FLAG(HcomGetCommCCLBufferSize);
DEFINE_SUPPORT_FLAG(HcomGetL0TopoTypeEx);
DEFINE_SUPPORT_FLAG(HcomGetRankSizeEx);
DEFINE_SUPPORT_FLAG(HcomInitByFile);
DEFINE_SUPPORT_FLAG(HcomGetWorkspaceSubStreamNum);
DEFINE_SUPPORT_FLAG(HcomGetWorkspaceMemSize);
DEFINE_SUPPORT_FLAG(HcomSetAlgorithm);
DEFINE_SUPPORT_FLAG(HcomGetAlltoAllStagedWorkSpaceMemSize);


// ---------- 桩函数定义 ----------
static HcclResult StubHcomGetRankSize(const char* group, u32* rankSize) {
    (void)group; (void)rankSize; HCCL_ERROR("[HcclWrapper] HcomGetRankSize not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomGetLocalRankSize(const char* group, u32* localRankSize) {
    (void)group; (void)localRankSize; HCCL_ERROR("[HcclWrapper] HcomGetLocalRankSize not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomGetRankId(const char* group, u32* rankId) {
    (void)group; (void)rankId; HCCL_ERROR("[HcclWrapper] HcomGetRankId not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomGetLocalRankId(const char* group, u32* localRankId) {
    (void)group; (void)localRankId; HCCL_ERROR("[HcclWrapper] HcomGetLocalRankId not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomGetWorldRankFromGroupRank(const char* group, u32 groupRank, u32* worldRank) {
    (void)group; (void)groupRank; (void)worldRank; HCCL_ERROR("[HcclWrapper] HcomGetWorldRankFromGroupRank not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomGetGroupRankFromWorldRank(u32 worldRank, const char* group, u32* groupRank) {
    (void)worldRank; (void)group; (void)groupRank; HCCL_ERROR("[HcclWrapper] HcomGetGroupRankFromWorldRank not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomCreateGroup(const char* group, u32 rankNum, u32* rankIds) {
    (void)group; (void)rankNum; (void)rankIds; HCCL_ERROR("[HcclWrapper] HcomCreateGroup not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomDestroyGroup(const char* group) {
    (void)group; HCCL_ERROR("[HcclWrapper] HcomDestroyGroup not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomSetGradFusionByIndex(const char* group, u32 segmentNum, const u32* inputIdxList) {
    (void)group; (void)segmentNum; (void)inputIdxList; HCCL_ERROR("[HcclWrapper] HcomSetGradFusionByIndex not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomSetGradFusionBySize(const char* group, u32 segmentNum, const float* sizeList) {
    (void)group; (void)segmentNum; (void)sizeList; HCCL_ERROR("[HcclWrapper] HcomSetGradFusionBySize not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomInitByRankTable(const char* rankTable, uint32_t rankId) {
    (void)rankTable; (void)rankId; HCCL_ERROR("[HcclWrapper] HcomInitByRankTable not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomDestroy() {
    HCCL_ERROR("[HcclWrapper] HcomDestroy not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomGetCommHandleByGroup(const char* group, HcclComm* commHandle) {
    (void)group; (void)commHandle; HCCL_ERROR("[HcclWrapper] HcomGetCommHandleByGroup not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubGetGroupNameByOpBaseHcom(s64 opBaseHcom, char** groupname) {
    (void)opBaseHcom; (void)groupname; HCCL_ERROR("[HcclWrapper] GetGroupNameByOpBaseHcom not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomCreateComResourceByComm(HcclComm comm, u32 streamMode, bool isOpbaseMode, void** commContext, bool isMC2) {
    (void)comm; (void)streamMode; (void)isOpbaseMode; (void)commContext; (void)isMC2; HCCL_ERROR("[HcclWrapper] HcomCreateComResourceByComm not supported"); return HCCL_E_NOT_SUPPORTED;
}
static void StubHcomTopoInfoRegCallback(HcclResult (*p1)(const char *, uint32_t), void (*p2)(const char *)) {
    (void)p1; (void)p2; HCCL_ERROR("[HcclWrapper] HcomTopoInfoRegCallback not supported");
}
static HcclResult StubHcomGetandClearOverFlowTasks(const char* group, hccl::HcclDumpInfo** hcclDumpInfoPtr, s32* len) {
    (void)group; (void)hcclDumpInfoPtr; (void)len; HCCL_ERROR("[HcclWrapper] HcomGetandClearOverFlowTasks not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclWorkflowMode StubHcomGetWorkflowMode() {
    HCCL_ERROR("[HcclWrapper] HcomGetWorkflowMode not supported"); return HcclWorkflowMode::HCCL_WORKFLOW_MODE_RESERVED;
}
static HcclResult StubHcomSetWorkflowMode(HcclWorkflowMode mode) {
    (void)mode; HCCL_ERROR("[HcclWrapper] HcomSetWorkflowMode not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomCalcOpOnline(HcomOpParam* hcomOpParam, HcomResResponse* hcomResResponse) {
    (void)hcomOpParam; (void)hcomResResponse; HCCL_ERROR("[HcclWrapper] HcomCalcOpOnline not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomCalcOpResOffline(HcomOpParam* hcomOpParam, HcomResResponse* hcomResResponse) {
    (void)hcomOpParam; (void)hcomResResponse; HCCL_ERROR("[HcclWrapper] HcomCalcOpResOffline not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomGetMemType(const char* group, const char* socVersion, bool isMalloc, u32* memType, bool* isTsMem, bool withoutImplCompile, bool level2Address) {
    (void)group; (void)socVersion; (void)isMalloc; (void)memType; (void)isTsMem; (void)withoutImplCompile; (void)level2Address; HCCL_ERROR("[HcclWrapper] HcomGetMemType not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomGetBandWidthPerNPU(u32 level, float* bandWidth) {
    (void)level; (void)bandWidth; HCCL_ERROR("[HcclWrapper] HcomGetBandWidthPerNPU not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomGetServerNumAndDeviceNumPerServer(u32* serverNum, u32* deviceNumPerServer, u32* deviceNumPerAggregation) {
    (void)serverNum; (void)deviceNumPerServer; (void)deviceNumPerAggregation; HCCL_ERROR("[HcclWrapper] HcomGetServerNumAndDeviceNumPerServer not supported"); return HCCL_E_NOT_SUPPORTED;
}
static bool StubHcomGetSecAddrCopyFlag(const char* socVersion) {
    (void)socVersion; HCCL_ERROR("[HcclWrapper] HcomGetSecAddrCopyFlag not supported"); return false;
}
static HcclResult StubHcomInitByString(const char* rankTableM, const char* identify, WorkMode commWorkMode, HcomInitConfig* initConfig) {
    (void)rankTableM; (void)identify; (void)commWorkMode; (void)initConfig; HCCL_ERROR("[HcclWrapper] HcomInitByString not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomInitByMasterInfo(const char* masterIp, const char* masterPort, const char* masterDeviceId, const char* rankSize, const char* rankIp, HcomInitConfig* initConfig) {
    (void)masterIp; (void)masterPort; (void)masterDeviceId; (void)rankSize; (void)rankIp; (void)initConfig; HCCL_ERROR("[HcclWrapper] HcomInitByMasterInfo not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomCreateCommCCLbuffer(const char* group) {
    (void)group; HCCL_ERROR("[HcclWrapper] HcomCreateCommCCLbuffer not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomGetInCCLbuffer(const char* group, void** buffer, u64* size) {
    (void)group; (void)buffer; (void)size; HCCL_ERROR("[HcclWrapper] HcomGetInCCLbuffer not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomGetOutCCLbuffer(const char* group, void** buffer, u64* size) {
    (void)group; (void)buffer; (void)size; HCCL_ERROR("[HcclWrapper] HcomGetOutCCLbuffer not supported"); return HCCL_E_NOT_SUPPORTED;
}
static void StubHcomSetLaunchKernelMode(bool state) {
    (void)state; HCCL_ERROR("[HcclWrapper] HcomSetLaunchKernelMode not supported");
}
static HcclResult StubHcomGetAicpuOpStreamNotify(const char* group, HcclRtStream* opStream, u8 aicpuNotifyNum, void** aicpuNotify) {
    (void)group; (void)opStream; (void)aicpuNotifyNum; (void)aicpuNotify; HCCL_ERROR("[HcclWrapper] HcomGetAicpuOpStreamNotify not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomMc2AiCpuStreamAllocAndGet(const char* group, u32 streamMode, rtStream_t* aiCpuStream) {
    (void)group; (void)streamMode; (void)aiCpuStream; HCCL_ERROR("[HcclWrapper] HcomMc2AiCpuStreamAllocAndGet not supported"); return HCCL_E_NOT_SUPPORTED;
}
static void StubHcomSetDumpDebugMode(bool dumpDebug) {
    (void)dumpDebug; HCCL_ERROR("[HcclWrapper] HcomSetDumpDebugMode not supported");
}
static HcclResult StubHcomGetAlgorithm(u32 level, char** algo) {
    (void)level; (void)algo; HCCL_ERROR("[HcclWrapper] HcomGetAlgorithm not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomGetAlgExecParam(const char* tag, const char* group, u64 count, void* inputPtr, void* outputPtr, HcclCMDType opType, bool clearEnable, HcclDataType dataType, HcclReduceOp op, void** commContext, u64* len, u32 aivCoreLimit) {
    (void)tag; (void)group; (void)count; (void)inputPtr; (void)outputPtr; (void)opType; (void)clearEnable; (void)dataType; (void)op; (void)commContext; (void)len; (void)aivCoreLimit; HCCL_ERROR("[HcclWrapper] HcomGetAlgExecParam not supported"); return HCCL_E_NOT_SUPPORTED;
}
static void StubHcomSetAutoTuneMode(bool autoTuneMode) {
    (void)autoTuneMode; HCCL_ERROR("[HcclWrapper] HcomSetAutoTuneMode not supported");
}
static DevType StubHcomGetDeviceType() {
    HCCL_ERROR("[HcclWrapper] HcomGetDeviceType not supported"); return DevType::DEV_TYPE_910;
}
static HcclResult StubHcomSetProfilingMode(HcomProfilingMode profilingMode, const char* profilingOption) {
    (void)profilingMode; (void)profilingOption; HCCL_ERROR("[HcclWrapper] HcomSetProfilingMode not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomGetSplitStrategy(const char* group, const struct model_feature* feature, u32** segmentIdxPtr, u32* len, bool* configured, GradSplitForceMode force, OriginalGraphShapeType shapeType) {
    (void)group; (void)feature; (void)segmentIdxPtr; (void)len; (void)configured; (void)force; (void)shapeType; HCCL_ERROR("[HcclWrapper] HcomGetSplitStrategy not supported"); return HCCL_E_NOT_SUPPORTED;
}
static bool StubHcomFindGroup(const char* group) {
    (void)group; HCCL_ERROR("[HcclWrapper] HcomFindGroup not supported"); return false;
}
static HcclResult StubHcomSelectAlg(s64 comm, const char* group, u64 count, void* counts, HcclDataType dataType, HcclReduceOp op, HcclCMDType opType, int32_t aivCoreLimit, bool& ifAiv, char* algName) {
    (void)comm; (void)group; (void)count; (void)counts; (void)dataType; (void)op; (void)opType; (void)aivCoreLimit; (void)ifAiv; (void)algName; HCCL_ERROR("[HcclWrapper] HcomSelectAlg not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomCalcAivCoreNum(const char* group, HcclCMDType opType, u64 count, void* counts, HcclDataType dataType, int32_t aivCoreLimit, char* algName, u32* numBlocks) {
    (void)group; (void)opType; (void)count; (void)counts; (void)dataType; (void)aivCoreLimit; (void)algName; (void)numBlocks; HCCL_ERROR("[HcclWrapper] HcomCalcAivCoreNum not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomSetWorkspaceResource(const char* tag, const char* group, rtStream_t* stream, s32 len, void* memPtr, u64 maxSize) {
    (void)tag; (void)group; (void)stream; (void)len; (void)memPtr; (void)maxSize; HCCL_ERROR("[HcclWrapper] HcomSetWorkspaceResource not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomSetGlobalWorkSpace(const char* group, void** globalWorkSpaceAddr, u32 len) {
    (void)group; (void)globalWorkSpaceAddr; (void)len; HCCL_ERROR("[HcclWrapper] HcomSetGlobalWorkSpace not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomSetAivCoreLimit(const char* group, u32 aivCoreLimit) {
    (void)group; (void)aivCoreLimit; HCCL_ERROR("[HcclWrapper] HcomSetAivCoreLimit not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomReleaseSubComms() {
    HCCL_ERROR("[HcclWrapper] HcomReleaseSubComms not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomUnloadTask(const char* group, const char* tag) {
    (void)group; (void)tag; HCCL_ERROR("[HcclWrapper] HcomUnloadTask not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomClearAivSyncBuf(const char* group, bool aivClearEnable) {
    (void)group; (void)aivClearEnable; HCCL_ERROR("[HcclWrapper] HcomClearAivSyncBuf not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomSetAttachedStream(const char* group, u32 graphId, const rtStream_t* stream, s32 len) {
    (void)group; (void)graphId; (void)stream; (void)len; HCCL_ERROR("[HcclWrapper] HcomSetAttachedStream not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomSupportDeterministicOptim(const char* group, bool* isDeterministicOptim) {
    (void)group; (void)isDeterministicOptim; HCCL_ERROR("[HcclWrapper] HcomSupportDeterministicOptim not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomTbeMemClean(int64_t addrList[], int64_t sizeList[], uint32_t count, rtStream_t stream, int32_t deviceLogicId) {
    (void)addrList; (void)sizeList; (void)count; (void)stream; (void)deviceLogicId; HCCL_ERROR("[HcclWrapper] HcomTbeMemClean not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomGetInitStatus(bool* initiated) {
    (void)initiated; HCCL_ERROR("[HcclWrapper] HcomGetInitStatus not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomAllGather(const char* tag, void* inputPtr, void* outputPtr, u64 inputCount, HcclDataType dataType, const char* group, rtStream_t stream) {
    (void)tag; (void)inputPtr; (void)outputPtr; (void)inputCount; (void)dataType; (void)group; (void)stream; HCCL_ERROR("[HcclWrapper] HcomAllGather not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomAllGatherV(const char* tag, const void* sendBuf, u64 sendCount, const void* recvBuf, const void* recvCounts, const void* rdispls, HcclDataType dataType, const char* group, rtStream_t stream) {
    (void)tag; (void)sendBuf; (void)sendCount; (void)recvBuf; (void)recvCounts; (void)rdispls; (void)dataType; (void)group; (void)stream; HCCL_ERROR("[HcclWrapper] HcomAllGatherV not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomAllReduce(const char* tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op, const char* group, rtStream_t stream) {
    (void)tag; (void)inputPtr; (void)outputPtr; (void)count; (void)dataType; (void)op; (void)group; (void)stream; HCCL_ERROR("[HcclWrapper] HcomAllReduce not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomReduce(const char* tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op, u32 root, const char* group, rtStream_t stream) {
    (void)tag; (void)inputPtr; (void)outputPtr; (void)count; (void)dataType; (void)op; (void)root; (void)group; (void)stream; HCCL_ERROR("[HcclWrapper] HcomReduce not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomBroadcast(const char* tag, void* ptr, u64 count, HcclDataType dataType, u32 root, const char* group, rtStream_t stream) {
    (void)tag; (void)ptr; (void)count; (void)dataType; (void)root; (void)group; (void)stream; HCCL_ERROR("[HcclWrapper] HcomBroadcast not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomReduceScatter(const char* tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op, const char* group, rtStream_t stream) {
    (void)tag; (void)inputPtr; (void)outputPtr; (void)count; (void)dataType; (void)op; (void)group; (void)stream; HCCL_ERROR("[HcclWrapper] HcomReduceScatter not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomReduceScatterV(const char* tag, void* sendBuf, const void* sendCounts, const void* sdispls, void* recvBuf, u64 recvCount, HcclDataType dataType, HcclReduceOp op, const char* group, rtStream_t stream) {
    (void)tag; (void)sendBuf; (void)sendCounts; (void)sdispls; (void)recvBuf; (void)recvCount; (void)dataType; (void)op; (void)group; (void)stream; HCCL_ERROR("[HcclWrapper] HcomReduceScatterV not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomSend(const char* tag, void* inputPtr, u64 count, HcclDataType dataType, u32 destRank, u32 srTag, const char* group, rtStream_t stream) {
    (void)tag; (void)inputPtr; (void)count; (void)dataType; (void)destRank; (void)srTag; (void)group; (void)stream; HCCL_ERROR("[HcclWrapper] HcomSend not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomReceive(const char* tag, void* outputPtr, u64 count, HcclDataType dataType, u32 srcRank, u32 srTag, const char* group, rtStream_t stream) {
    (void)tag; (void)outputPtr; (void)count; (void)dataType; (void)srcRank; (void)srTag; (void)group; (void)stream; HCCL_ERROR("[HcclWrapper] HcomReceive not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomAlltoAllV(const void* sendBuf, const void* sendCounts, const void* sdispls, HcclDataType sendType, const void* recvBuf, const void* recvCounts, const void* rdispls, HcclDataType recvType, const char* group, rtStream_t stream, const char* tag) {
    (void)sendBuf; (void)sendCounts; (void)sdispls; (void)sendType; (void)recvBuf; (void)recvCounts; (void)rdispls; (void)recvType; (void)group; (void)stream; (void)tag; HCCL_ERROR("[HcclWrapper] HcomAlltoAllV not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomAlltoAllVC(const void* sendBuf, const void* sendCountMatrix, HcclDataType sendType, const void* recvBuf, HcclDataType recvType, const char* group, rtStream_t stream, const char* tag) {
    (void)sendBuf; (void)sendCountMatrix; (void)sendType; (void)recvBuf; (void)recvType; (void)group; (void)stream; (void)tag; HCCL_ERROR("[HcclWrapper] HcomAlltoAllVC not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomAllToAll(const void* sendBuf, u64 sendCount, HcclDataType sendType, const void* recvBuf, u64 recvCount, HcclDataType recvType, const char* group, rtStream_t stream, const char* tag) {
    (void)sendBuf; (void)sendCount; (void)sendType; (void)recvBuf; (void)recvCount; (void)recvType; (void)group; (void)stream; (void)tag; HCCL_ERROR("[HcclWrapper] HcomAllToAll not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomGetHcclComm(int64_t comm, std::string& group) {
    (void)comm; (void)group; HCCL_ERROR("[HcclWrapper] HcomGetHcclComm not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomGenerateCclOpTag(const char* opType, s64 hcomComm, const char* group, char* sTag) {
    (void)opType; (void)hcomComm; (void)group; (void)sTag; HCCL_ERROR("[HcclWrapper] HcomGenerateCclOpTag not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomGetCommCCLBufferSize(const char* group, uint64_t& size) {
    (void)group; (void)size; HCCL_ERROR("[HcclWrapper] HcomGetCommCCLBufferSize not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomGetL0TopoTypeEx(const char* group, CommTopo* topoType, uint32_t flag) {
    (void)group; (void)topoType; (void)flag; HCCL_ERROR("[HcclWrapper] HcomGetL0TopoTypeEx not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomGetRankSizeEx(const char* group, uint32_t* rankSize, uint32_t flag) {
    (void)group; (void)rankSize; (void)flag; HCCL_ERROR("[HcclWrapper] HcomGetRankSizeEx not supported"); return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomInitByFile(const char* rankTablePath, const char* identify) {
    (void)rankTablePath; (void)identify;
    HCCL_ERROR("[HcclWrapper] HcomInitByFile not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomGetWorkspaceSubStreamNum(const char* group, u64& streamNum, u64 dataSize,
    HcclDataType dataType, u32 aivCoreLimit, HcclReduceOp reduceOp, u64 count, HcclCMDType optype) {
    (void)group; (void)streamNum; (void)dataSize; (void)dataType; (void)aivCoreLimit; (void)reduceOp; (void)count; (void)optype;
    HCCL_ERROR("[HcclWrapper] HcomGetWorkspaceSubStreamNum not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomGetWorkspaceMemSize(const std::string& opType, u64 count,
    HcclDataType dataType, const char* group, u64& memSize) {
    (void)opType; (void)count; (void)dataType; (void)group; (void)memSize;
    HCCL_ERROR("[HcclWrapper] HcomGetWorkspaceMemSize not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomSetAlgorithm(const char* algo) {
    (void)algo;
    HCCL_ERROR("[HcclWrapper] HcomSetAlgorithm not supported");
    return HCCL_E_NOT_SUPPORTED;
}
static HcclResult StubHcomGetAlltoAllStagedWorkSpaceMemSize(const char *group, u64 *sendCounts, u64 *sdispls,
    HcclDataType sendType, u64 *recvCounts, u64 *rdispls, HcclDataType recvType, u64 &memSize) {
    (void)group; (void)sendCounts; (void)sdispls; (void)sendType; (void)recvCounts; (void)rdispls; (void)recvType; (void)memSize;
    HCCL_ERROR("[HcclWrapper] HcomGetAlltoAllStagedWorkSpaceMemSize not supported");
    return HCCL_E_NOT_SUPPORTED;
}

// ---------- 初始化函数 ----------
void HcomDlInit(void* libHcommHandle) {
    #define SET_PTR(ptr, handle, name, stub, support_flag) \
        do { \
            ptr = (decltype(ptr))dlsym(handle, name); \
            if (ptr == nullptr) { \
                ptr = stub; \
                support_flag = false; \
                HCCL_DEBUG("[HcclWrapper] %s not supported", name); \
            } else { \
                support_flag = true; \
            } \
        } while(0)

    SET_PTR(hcomGetRankSizePtr, libHcommHandle, "HcomGetRankSize", StubHcomGetRankSize, g_HcomGetRankSizeSupported);
    SET_PTR(hcomGetLocalRankSizePtr, libHcommHandle, "HcomGetLocalRankSize", StubHcomGetLocalRankSize, g_HcomGetLocalRankSizeSupported);
    SET_PTR(hcomGetRankIdPtr, libHcommHandle, "HcomGetRankId", StubHcomGetRankId, g_HcomGetRankIdSupported);
    SET_PTR(hcomGetLocalRankIdPtr, libHcommHandle, "HcomGetLocalRankId", StubHcomGetLocalRankId, g_HcomGetLocalRankIdSupported);
    SET_PTR(hcomGetWorldRankFromGroupRankPtr, libHcommHandle, "HcomGetWorldRankFromGroupRank", StubHcomGetWorldRankFromGroupRank, g_HcomGetWorldRankFromGroupRankSupported);
    SET_PTR(hcomGetGroupRankFromWorldRankPtr, libHcommHandle, "HcomGetGroupRankFromWorldRank", StubHcomGetGroupRankFromWorldRank, g_HcomGetGroupRankFromWorldRankSupported);
    SET_PTR(hcomCreateGroupPtr, libHcommHandle, "HcomCreateGroup", StubHcomCreateGroup, g_HcomCreateGroupSupported);
    SET_PTR(hcomDestroyGroupPtr, libHcommHandle, "HcomDestroyGroup", StubHcomDestroyGroup, g_HcomDestroyGroupSupported);
    SET_PTR(hcomSetGradFusionByIndexPtr, libHcommHandle, "HcomSetGradFusionByIndex", StubHcomSetGradFusionByIndex, g_HcomSetGradFusionByIndexSupported);
    SET_PTR(hcomSetGradFusionBySizePtr, libHcommHandle, "HcomSetGradFusionBySize", StubHcomSetGradFusionBySize, g_HcomSetGradFusionBySizeSupported);
    SET_PTR(hcomInitByRankTablePtr, libHcommHandle, "HcomInitByRankTable", StubHcomInitByRankTable, g_HcomInitByRankTableSupported);
    SET_PTR(hcomDestroyPtr, libHcommHandle, "HcomDestroy", StubHcomDestroy, g_HcomDestroySupported);
    SET_PTR(hcomGetCommHandleByGroupPtr, libHcommHandle, "HcomGetCommHandleByGroup", StubHcomGetCommHandleByGroup, g_HcomGetCommHandleByGroupSupported);
    SET_PTR(getGroupNameByOpBaseHcomPtr, libHcommHandle, "GetGroupNameByOpBaseHcom", StubGetGroupNameByOpBaseHcom, g_GetGroupNameByOpBaseHcomSupported);
    SET_PTR(hcomCreateComResourceByCommPtr, libHcommHandle, "HcomCreateComResourceByComm", StubHcomCreateComResourceByComm, g_HcomCreateComResourceByCommSupported);
    SET_PTR(hcomTopoInfoRegCallbackPtr, libHcommHandle, "HcomTopoInfoRegCallback", StubHcomTopoInfoRegCallback, g_HcomTopoInfoRegCallbackSupported);
    SET_PTR(hcomGetandClearOverFlowTasksPtr, libHcommHandle, "HcomGetandClearOverFlowTasks", StubHcomGetandClearOverFlowTasks, g_HcomGetandClearOverFlowTasksSupported);
    SET_PTR(hcomGetWorkflowModePtr, libHcommHandle, "HcomGetWorkflowMode", StubHcomGetWorkflowMode, g_HcomGetWorkflowModeSupported);
    SET_PTR(hcomSetWorkflowModePtr, libHcommHandle, "HcomSetWorkflowMode", StubHcomSetWorkflowMode, g_HcomSetWorkflowModeSupported);
    SET_PTR(hcomCalcOpOnlinePtr, libHcommHandle, "HcomCalcOpOnline", StubHcomCalcOpOnline, g_HcomCalcOpOnlineSupported);
    SET_PTR(hcomCalcOpResOfflinePtr, libHcommHandle, "HcomCalcOpResOffline", StubHcomCalcOpResOffline, g_HcomCalcOpResOfflineSupported);
    SET_PTR(hcomGetMemTypePtr, libHcommHandle, "HcomGetMemType", StubHcomGetMemType, g_HcomGetMemTypeSupported);
    SET_PTR(hcomGetBandWidthPerNPUPtr, libHcommHandle, "HcomGetBandWidthPerNPU", StubHcomGetBandWidthPerNPU, g_HcomGetBandWidthPerNPUSupported);
    SET_PTR(hcomGetServerNumAndDeviceNumPerServerPtr, libHcommHandle, "HcomGetServerNumAndDeviceNumPerServer", StubHcomGetServerNumAndDeviceNumPerServer, g_HcomGetServerNumAndDeviceNumPerServerSupported);
    SET_PTR(hcomGetSecAddrCopyFlagPtr, libHcommHandle, "HcomGetSecAddrCopyFlag", StubHcomGetSecAddrCopyFlag, g_HcomGetSecAddrCopyFlagSupported);
    SET_PTR(hcomInitByStringPtr, libHcommHandle, "HcomInitByString", StubHcomInitByString, g_HcomInitByStringSupported);
    SET_PTR(hcomInitByMasterInfoPtr, libHcommHandle, "HcomInitByMasterInfo", StubHcomInitByMasterInfo, g_HcomInitByMasterInfoSupported);
    SET_PTR(hcomCreateCommCCLbufferPtr, libHcommHandle, "HcomCreateCommCCLbuffer", StubHcomCreateCommCCLbuffer, g_HcomCreateCommCCLbufferSupported);
    SET_PTR(hcomGetInCCLbufferPtr, libHcommHandle, "HcomGetInCCLbuffer", StubHcomGetInCCLbuffer, g_HcomGetInCCLbufferSupported);
    SET_PTR(hcomGetOutCCLbufferPtr, libHcommHandle, "HcomGetOutCCLbuffer", StubHcomGetOutCCLbuffer, g_HcomGetOutCCLbufferSupported);
    SET_PTR(hcomSetLaunchKernelModePtr, libHcommHandle, "HcomSetLaunchKernelMode", StubHcomSetLaunchKernelMode, g_HcomSetLaunchKernelModeSupported);
    SET_PTR(hcomGetAicpuOpStreamNotifyPtr, libHcommHandle, "HcomGetAicpuOpStreamNotify", StubHcomGetAicpuOpStreamNotify, g_HcomGetAicpuOpStreamNotifySupported);
    SET_PTR(hcomMc2AiCpuStreamAllocAndGetPtr, libHcommHandle, "HcomMc2AiCpuStreamAllocAndGet", StubHcomMc2AiCpuStreamAllocAndGet, g_HcomMc2AiCpuStreamAllocAndGetSupported);
    SET_PTR(hcomSetDumpDebugModePtr, libHcommHandle, "HcomSetDumpDebugMode", StubHcomSetDumpDebugMode, g_HcomSetDumpDebugModeSupported);
    SET_PTR(hcomGetAlgorithmPtr, libHcommHandle, "HcomGetAlgorithm", StubHcomGetAlgorithm, g_HcomGetAlgorithmSupported);
    SET_PTR(hcomGetAlgExecParamPtr, libHcommHandle, "HcomGetAlgExecParam", StubHcomGetAlgExecParam, g_HcomGetAlgExecParamSupported);
    SET_PTR(hcomSetAutoTuneModePtr, libHcommHandle, "HcomSetAutoTuneMode", StubHcomSetAutoTuneMode, g_HcomSetAutoTuneModeSupported);
    SET_PTR(hcomGetDeviceTypePtr, libHcommHandle, "HcomGetDeviceType", StubHcomGetDeviceType, g_HcomGetDeviceTypeSupported);
    SET_PTR(hcomSetProfilingModePtr, libHcommHandle, "HcomSetProfilingMode", StubHcomSetProfilingMode, g_HcomSetProfilingModeSupported);
    SET_PTR(hcomGetSplitStrategyPtr, libHcommHandle, "HcomGetSplitStrategy", StubHcomGetSplitStrategy, g_HcomGetSplitStrategySupported);
    SET_PTR(hcomFindGroupPtr, libHcommHandle, "HcomFindGroup", StubHcomFindGroup, g_HcomFindGroupSupported);
    SET_PTR(hcomSelectAlgPtr, libHcommHandle, "HcomSelectAlg", StubHcomSelectAlg, g_HcomSelectAlgSupported);
    SET_PTR(hcomCalcAivCoreNumPtr, libHcommHandle, "HcomCalcAivCoreNum", StubHcomCalcAivCoreNum, g_HcomCalcAivCoreNumSupported);
    SET_PTR(hcomSetWorkspaceResourcePtr, libHcommHandle, "HcomSetWorkspaceResource", StubHcomSetWorkspaceResource, g_HcomSetWorkspaceResourceSupported);
    SET_PTR(hcomSetGlobalWorkSpacePtr, libHcommHandle, "HcomSetGlobalWorkSpace", StubHcomSetGlobalWorkSpace, g_HcomSetGlobalWorkSpaceSupported);
    SET_PTR(hcomSetAivCoreLimitPtr, libHcommHandle, "HcomSetAivCoreLimit", StubHcomSetAivCoreLimit, g_HcomSetAivCoreLimitSupported);
    SET_PTR(hcomReleaseSubCommsPtr, libHcommHandle, "HcomReleaseSubComms", StubHcomReleaseSubComms, g_HcomReleaseSubCommsSupported);
    SET_PTR(hcomUnloadTaskPtr, libHcommHandle, "HcomUnloadTask", StubHcomUnloadTask, g_HcomUnloadTaskSupported);
    SET_PTR(hcomClearAivSyncBufPtr, libHcommHandle, "HcomClearAivSyncBuf", StubHcomClearAivSyncBuf, g_HcomClearAivSyncBufSupported);
    SET_PTR(hcomSetAttachedStreamPtr, libHcommHandle, "HcomSetAttachedStream", StubHcomSetAttachedStream, g_HcomSetAttachedStreamSupported);
    SET_PTR(hcomSupportDeterministicOptimPtr, libHcommHandle, "HcomSupportDeterministicOptim", StubHcomSupportDeterministicOptim, g_HcomSupportDeterministicOptimSupported);
    SET_PTR(hcomTbeMemCleanPtr, libHcommHandle, "HcomTbeMemClean", StubHcomTbeMemClean, g_HcomTbeMemCleanSupported);
    SET_PTR(hcomGetInitStatusPtr, libHcommHandle, "HcomGetInitStatus", StubHcomGetInitStatus, g_HcomGetInitStatusSupported);
    SET_PTR(hcomAllGatherPtr, libHcommHandle, "HcomAllGather", StubHcomAllGather, g_HcomAllGatherSupported);
    SET_PTR(hcomAllGatherVPtr, libHcommHandle, "HcomAllGatherV", StubHcomAllGatherV, g_HcomAllGatherVSupported);
    SET_PTR(hcomAllReducePtr, libHcommHandle, "HcomAllReduce", StubHcomAllReduce, g_HcomAllReduceSupported);
    SET_PTR(hcomReducePtr, libHcommHandle, "HcomReduce", StubHcomReduce, g_HcomReduceSupported);
    SET_PTR(hcomBroadcastPtr, libHcommHandle, "HcomBroadcast", StubHcomBroadcast, g_HcomBroadcastSupported);
    SET_PTR(hcomReduceScatterPtr, libHcommHandle, "HcomReduceScatter", StubHcomReduceScatter, g_HcomReduceScatterSupported);
    SET_PTR(hcomReduceScatterVPtr, libHcommHandle, "HcomReduceScatterV", StubHcomReduceScatterV, g_HcomReduceScatterVSupported);
    SET_PTR(hcomSendPtr, libHcommHandle, "HcomSend", StubHcomSend, g_HcomSendSupported);
    SET_PTR(hcomReceivePtr, libHcommHandle, "HcomReceive", StubHcomReceive, g_HcomReceiveSupported);
    SET_PTR(hcomAlltoAllVPtr, libHcommHandle, "HcomAlltoAllV", StubHcomAlltoAllV, g_HcomAlltoAllVSupported);
    SET_PTR(hcomAlltoAllVCPtr, libHcommHandle, "HcomAlltoAllVC", StubHcomAlltoAllVC, g_HcomAlltoAllVCSupported);
    SET_PTR(hcomAllToAllPtr, libHcommHandle, "HcomAllToAll", StubHcomAllToAll, g_HcomAllToAllSupported);
    SET_PTR(hcomGetHcclCommPtr, libHcommHandle, "HcomGetHcclComm", StubHcomGetHcclComm, g_HcomGetHcclCommSupported);
    SET_PTR(hcomGenerateCclOpTagPtr, libHcommHandle, "HcomGenerateCclOpTag", StubHcomGenerateCclOpTag, g_HcomGenerateCclOpTagSupported);
    SET_PTR(hcomGetCommCCLBufferSizePtr, libHcommHandle, "HcomGetCommCCLBufferSize", StubHcomGetCommCCLBufferSize, g_HcomGetCommCCLBufferSizeSupported);
    SET_PTR(hcomGetL0TopoTypeExPtr, libHcommHandle, "HcomGetL0TopoTypeEx", StubHcomGetL0TopoTypeEx, g_HcomGetL0TopoTypeExSupported);
    SET_PTR(hcomGetRankSizeExPtr, libHcommHandle, "HcomGetRankSizeEx", StubHcomGetRankSizeEx, g_HcomGetRankSizeExSupported);
    SET_PTR(hcomInitByFilePtr, libHcommHandle, "HcomInitByFile", StubHcomInitByFile, g_HcomInitByFileSupported);
    SET_PTR(hcomGetWorkspaceSubStreamNumPtr, libHcommHandle, "HcomGetWorkspaceSubStreamNum", StubHcomGetWorkspaceSubStreamNum, g_HcomGetWorkspaceSubStreamNumSupported);
    SET_PTR(hcomGetWorkspaceMemSizePtr, libHcommHandle, "HcomGetWorkspaceMemSize", StubHcomGetWorkspaceMemSize, g_HcomGetWorkspaceMemSizeSupported);
    SET_PTR(hcomSetAlgorithmPtr, libHcommHandle, "HcomSetAlgorithm", StubHcomSetAlgorithm, g_HcomSetAlgorithmSupported);
    SET_PTR(hcomGetAlltoAllStagedWorkSpaceMemSizePtr, libHcommHandle, "HcomGetAlltoAllStagedWorkSpaceMemSize", StubHcomGetAlltoAllStagedWorkSpaceMemSize, g_HcomGetAlltoAllStagedWorkSpaceMemSizeSupported);

    #undef SET_PTR
}

void HcomDlFini(void) {
    // 重置为桩函数，支持标志置 false（可选）
    #define RESET_PTR(ptr, stub, support_flag) do { ptr = stub; support_flag = false; } while(0)

    RESET_PTR(hcomGetRankSizePtr, StubHcomGetRankSize, g_HcomGetRankSizeSupported);
    RESET_PTR(hcomGetLocalRankSizePtr, StubHcomGetLocalRankSize, g_HcomGetLocalRankSizeSupported);
    RESET_PTR(hcomGetRankIdPtr, StubHcomGetRankId, g_HcomGetRankIdSupported);
    RESET_PTR(hcomGetLocalRankIdPtr, StubHcomGetLocalRankId, g_HcomGetLocalRankIdSupported);
    RESET_PTR(hcomGetWorldRankFromGroupRankPtr, StubHcomGetWorldRankFromGroupRank, g_HcomGetWorldRankFromGroupRankSupported);
    RESET_PTR(hcomGetGroupRankFromWorldRankPtr, StubHcomGetGroupRankFromWorldRank, g_HcomGetGroupRankFromWorldRankSupported);
    RESET_PTR(hcomCreateGroupPtr, StubHcomCreateGroup, g_HcomCreateGroupSupported);
    RESET_PTR(hcomDestroyGroupPtr, StubHcomDestroyGroup, g_HcomDestroyGroupSupported);
    RESET_PTR(hcomSetGradFusionByIndexPtr, StubHcomSetGradFusionByIndex, g_HcomSetGradFusionByIndexSupported);
    RESET_PTR(hcomSetGradFusionBySizePtr, StubHcomSetGradFusionBySize, g_HcomSetGradFusionBySizeSupported);
    RESET_PTR(hcomInitByRankTablePtr, StubHcomInitByRankTable, g_HcomInitByRankTableSupported);
    RESET_PTR(hcomDestroyPtr, StubHcomDestroy, g_HcomDestroySupported);
    RESET_PTR(hcomGetCommHandleByGroupPtr, StubHcomGetCommHandleByGroup, g_HcomGetCommHandleByGroupSupported);
    RESET_PTR(getGroupNameByOpBaseHcomPtr, StubGetGroupNameByOpBaseHcom, g_GetGroupNameByOpBaseHcomSupported);
    RESET_PTR(hcomCreateComResourceByCommPtr, StubHcomCreateComResourceByComm, g_HcomCreateComResourceByCommSupported);
    RESET_PTR(hcomTopoInfoRegCallbackPtr, StubHcomTopoInfoRegCallback, g_HcomTopoInfoRegCallbackSupported);
    RESET_PTR(hcomGetandClearOverFlowTasksPtr, StubHcomGetandClearOverFlowTasks, g_HcomGetandClearOverFlowTasksSupported);
    RESET_PTR(hcomGetWorkflowModePtr, StubHcomGetWorkflowMode, g_HcomGetWorkflowModeSupported);
    RESET_PTR(hcomSetWorkflowModePtr, StubHcomSetWorkflowMode, g_HcomSetWorkflowModeSupported);
    RESET_PTR(hcomCalcOpOnlinePtr, StubHcomCalcOpOnline, g_HcomCalcOpOnlineSupported);
    RESET_PTR(hcomCalcOpResOfflinePtr, StubHcomCalcOpResOffline, g_HcomCalcOpResOfflineSupported);
    RESET_PTR(hcomGetMemTypePtr, StubHcomGetMemType, g_HcomGetMemTypeSupported);
    RESET_PTR(hcomGetBandWidthPerNPUPtr, StubHcomGetBandWidthPerNPU, g_HcomGetBandWidthPerNPUSupported);
    RESET_PTR(hcomGetServerNumAndDeviceNumPerServerPtr, StubHcomGetServerNumAndDeviceNumPerServer, g_HcomGetServerNumAndDeviceNumPerServerSupported);
    RESET_PTR(hcomGetSecAddrCopyFlagPtr, StubHcomGetSecAddrCopyFlag, g_HcomGetSecAddrCopyFlagSupported);
    RESET_PTR(hcomInitByStringPtr, StubHcomInitByString, g_HcomInitByStringSupported);
    RESET_PTR(hcomInitByMasterInfoPtr, StubHcomInitByMasterInfo, g_HcomInitByMasterInfoSupported);
    RESET_PTR(hcomCreateCommCCLbufferPtr, StubHcomCreateCommCCLbuffer, g_HcomCreateCommCCLbufferSupported);
    RESET_PTR(hcomGetInCCLbufferPtr, StubHcomGetInCCLbuffer, g_HcomGetInCCLbufferSupported);
    RESET_PTR(hcomGetOutCCLbufferPtr, StubHcomGetOutCCLbuffer, g_HcomGetOutCCLbufferSupported);
    RESET_PTR(hcomSetLaunchKernelModePtr, StubHcomSetLaunchKernelMode, g_HcomSetLaunchKernelModeSupported);
    RESET_PTR(hcomGetAicpuOpStreamNotifyPtr, StubHcomGetAicpuOpStreamNotify, g_HcomGetAicpuOpStreamNotifySupported);
    RESET_PTR(hcomMc2AiCpuStreamAllocAndGetPtr, StubHcomMc2AiCpuStreamAllocAndGet, g_HcomMc2AiCpuStreamAllocAndGetSupported);
    RESET_PTR(hcomSetDumpDebugModePtr, StubHcomSetDumpDebugMode, g_HcomSetDumpDebugModeSupported);
    RESET_PTR(hcomGetAlgorithmPtr, StubHcomGetAlgorithm, g_HcomGetAlgorithmSupported);
    RESET_PTR(hcomGetAlgExecParamPtr, StubHcomGetAlgExecParam, g_HcomGetAlgExecParamSupported);
    RESET_PTR(hcomSetAutoTuneModePtr, StubHcomSetAutoTuneMode, g_HcomSetAutoTuneModeSupported);
    RESET_PTR(hcomGetDeviceTypePtr, StubHcomGetDeviceType, g_HcomGetDeviceTypeSupported);
    RESET_PTR(hcomSetProfilingModePtr, StubHcomSetProfilingMode, g_HcomSetProfilingModeSupported);
    RESET_PTR(hcomGetSplitStrategyPtr, StubHcomGetSplitStrategy, g_HcomGetSplitStrategySupported);
    RESET_PTR(hcomFindGroupPtr, StubHcomFindGroup, g_HcomFindGroupSupported);
    RESET_PTR(hcomSelectAlgPtr, StubHcomSelectAlg, g_HcomSelectAlgSupported);
    RESET_PTR(hcomCalcAivCoreNumPtr, StubHcomCalcAivCoreNum, g_HcomCalcAivCoreNumSupported);
    RESET_PTR(hcomSetWorkspaceResourcePtr, StubHcomSetWorkspaceResource, g_HcomSetWorkspaceResourceSupported);
    RESET_PTR(hcomSetGlobalWorkSpacePtr, StubHcomSetGlobalWorkSpace, g_HcomSetGlobalWorkSpaceSupported);
    RESET_PTR(hcomSetAivCoreLimitPtr, StubHcomSetAivCoreLimit, g_HcomSetAivCoreLimitSupported);
    RESET_PTR(hcomReleaseSubCommsPtr, StubHcomReleaseSubComms, g_HcomReleaseSubCommsSupported);
    RESET_PTR(hcomUnloadTaskPtr, StubHcomUnloadTask, g_HcomUnloadTaskSupported);
    RESET_PTR(hcomClearAivSyncBufPtr, StubHcomClearAivSyncBuf, g_HcomClearAivSyncBufSupported);
    RESET_PTR(hcomSetAttachedStreamPtr, StubHcomSetAttachedStream, g_HcomSetAttachedStreamSupported);
    RESET_PTR(hcomSupportDeterministicOptimPtr, StubHcomSupportDeterministicOptim, g_HcomSupportDeterministicOptimSupported);
    RESET_PTR(hcomTbeMemCleanPtr, StubHcomTbeMemClean, g_HcomTbeMemCleanSupported);
    RESET_PTR(hcomGetInitStatusPtr, StubHcomGetInitStatus, g_HcomGetInitStatusSupported);
    RESET_PTR(hcomAllGatherPtr, StubHcomAllGather, g_HcomAllGatherSupported);
    RESET_PTR(hcomAllGatherVPtr, StubHcomAllGatherV, g_HcomAllGatherVSupported);
    RESET_PTR(hcomAllReducePtr, StubHcomAllReduce, g_HcomAllReduceSupported);
    RESET_PTR(hcomReducePtr, StubHcomReduce, g_HcomReduceSupported);
    RESET_PTR(hcomBroadcastPtr, StubHcomBroadcast, g_HcomBroadcastSupported);
    RESET_PTR(hcomReduceScatterPtr, StubHcomReduceScatter, g_HcomReduceScatterSupported);
    RESET_PTR(hcomReduceScatterVPtr, StubHcomReduceScatterV, g_HcomReduceScatterVSupported);
    RESET_PTR(hcomSendPtr, StubHcomSend, g_HcomSendSupported);
    RESET_PTR(hcomReceivePtr, StubHcomReceive, g_HcomReceiveSupported);
    RESET_PTR(hcomAlltoAllVPtr, StubHcomAlltoAllV, g_HcomAlltoAllVSupported);
    RESET_PTR(hcomAlltoAllVCPtr, StubHcomAlltoAllVC, g_HcomAlltoAllVCSupported);
    RESET_PTR(hcomAllToAllPtr, StubHcomAllToAll, g_HcomAllToAllSupported);
    RESET_PTR(hcomGetHcclCommPtr, StubHcomGetHcclComm, g_HcomGetHcclCommSupported);
    RESET_PTR(hcomGenerateCclOpTagPtr, StubHcomGenerateCclOpTag, g_HcomGenerateCclOpTagSupported);
    RESET_PTR(hcomGetCommCCLBufferSizePtr, StubHcomGetCommCCLBufferSize, g_HcomGetCommCCLBufferSizeSupported);
    RESET_PTR(hcomGetL0TopoTypeExPtr, StubHcomGetL0TopoTypeEx, g_HcomGetL0TopoTypeExSupported);
    RESET_PTR(hcomGetRankSizeExPtr, StubHcomGetRankSizeEx, g_HcomGetRankSizeExSupported);
    RESET_PTR(hcomInitByFilePtr, StubHcomInitByFile, g_HcomInitByFileSupported);
    RESET_PTR(hcomGetWorkspaceSubStreamNumPtr, StubHcomGetWorkspaceSubStreamNum, g_HcomGetWorkspaceSubStreamNumSupported);
    RESET_PTR(hcomGetWorkspaceMemSizePtr, StubHcomGetWorkspaceMemSize, g_HcomGetWorkspaceMemSizeSupported);
    RESET_PTR(hcomSetAlgorithmPtr, StubHcomSetAlgorithm, g_HcomSetAlgorithmSupported);
    RESET_PTR(hcomGetAlltoAllStagedWorkSpaceMemSizePtr, StubHcomGetAlltoAllStagedWorkSpaceMemSize, g_HcomGetAlltoAllStagedWorkSpaceMemSizeSupported);

    #undef RESET_PTR
}

#ifdef __cplusplus
extern "C" {
#endif
// ---------- 对外API实现（通过函数指针转发）----------
HcclResult HcomGetRankSize(const char* group, u32* rankSize) {
    return hcomGetRankSizePtr(group, rankSize);
}
HcclResult HcomGetLocalRankSize(const char* group, u32* localRankSize) {
    return hcomGetLocalRankSizePtr(group, localRankSize);
}
HcclResult HcomGetRankId(const char* group, u32* rankId) {
    return hcomGetRankIdPtr(group, rankId);
}
HcclResult HcomGetLocalRankId(const char* group, u32* localRankId) {
    return hcomGetLocalRankIdPtr(group, localRankId);
}
HcclResult HcomGetWorldRankFromGroupRank(const char* group, u32 groupRank, u32* worldRank) {
    return hcomGetWorldRankFromGroupRankPtr(group, groupRank, worldRank);
}
HcclResult HcomGetGroupRankFromWorldRank(u32 worldRank, const char* group, u32* groupRank) {
    return hcomGetGroupRankFromWorldRankPtr(worldRank, group, groupRank);
}
HcclResult HcomCreateGroup(const char* group, u32 rankNum, u32* rankIds) {
    return hcomCreateGroupPtr(group, rankNum, rankIds);
}
HcclResult HcomDestroyGroup(const char* group) {
    return hcomDestroyGroupPtr(group);
}
HcclResult HcomSetGradFusionByIndex(const char* group, u32 segmentNum, const u32* inputIdxList) {
    return hcomSetGradFusionByIndexPtr(group, segmentNum, inputIdxList);
}
HcclResult HcomSetGradFusionBySize(const char* group, u32 segmentNum, const float* sizeList) {
    return hcomSetGradFusionBySizePtr(group, segmentNum, sizeList);
}
HcclResult HcomInitByRankTable(const char* rankTable, uint32_t rankId) {
    return hcomInitByRankTablePtr(rankTable, rankId);
}
HcclResult HcomDestroy() {
    return hcomDestroyPtr();
}
HcclResult HcomGetCommHandleByGroup(const char* group, HcclComm* commHandle) {
    return hcomGetCommHandleByGroupPtr(group, commHandle);
}
HcclResult GetGroupNameByOpBaseHcom(s64 opBaseHcom, char** groupname) {
    return getGroupNameByOpBaseHcomPtr(opBaseHcom, groupname);
}
HcclResult HcomCreateComResourceByComm(HcclComm comm, u32 streamMode, bool isOpbaseMode, void** commContext, bool isMC2) {
    return hcomCreateComResourceByCommPtr(comm, streamMode, isOpbaseMode, commContext, isMC2);
}
void HcomTopoInfoRegCallback(HcclResult (*p1)(const char *, uint32_t), void (*p2)(const char *)) {
    hcomTopoInfoRegCallbackPtr(p1, p2);
}
HcclResult HcomGetandClearOverFlowTasks(const char* group, hccl::HcclDumpInfo** hcclDumpInfoPtr, s32* len) {
    return hcomGetandClearOverFlowTasksPtr(group, hcclDumpInfoPtr, len);
}
HcclWorkflowMode HcomGetWorkflowMode() {
    return hcomGetWorkflowModePtr();
}
HcclResult HcomSetWorkflowMode(HcclWorkflowMode mode) {
    return hcomSetWorkflowModePtr(mode);
}
HcclResult HcomCalcOpOnline(HcomOpParam* hcomOpParam, HcomResResponse* hcomResResponse) {
    return hcomCalcOpOnlinePtr(hcomOpParam, hcomResResponse);
}
HcclResult HcomCalcOpResOffline(HcomOpParam* hcomOpParam, HcomResResponse* hcomResResponse) {
    return hcomCalcOpResOfflinePtr(hcomOpParam, hcomResResponse);
}
HcclResult HcomGetMemType(const char* group, const char* socVersion, bool isMalloc, u32* memType, bool* isTsMem, bool withoutImplCompile, bool level2Address) {
    return hcomGetMemTypePtr(group, socVersion, isMalloc, memType, isTsMem, withoutImplCompile, level2Address);
}
HcclResult HcomGetBandWidthPerNPU(u32 level, float* bandWidth) {
    return hcomGetBandWidthPerNPUPtr(level, bandWidth);
}
HcclResult HcomGetServerNumAndDeviceNumPerServer(u32* serverNum, u32* deviceNumPerServer, u32* deviceNumPerAggregation) {
    return hcomGetServerNumAndDeviceNumPerServerPtr(serverNum, deviceNumPerServer, deviceNumPerAggregation);
}
bool HcomGetSecAddrCopyFlag(const char* socVersion) {
    return hcomGetSecAddrCopyFlagPtr(socVersion);
}
HcclResult HcomInitByString(const char* rankTableM, const char* identify, WorkMode commWorkMode, HcomInitConfig* initConfig) {
    return hcomInitByStringPtr(rankTableM, identify, commWorkMode, initConfig);
}
HcclResult HcomInitByMasterInfo(const char* masterIp, const char* masterPort, const char* masterDeviceId, const char* rankSize, const char* rankIp, HcomInitConfig* initConfig) {
    return hcomInitByMasterInfoPtr(masterIp, masterPort, masterDeviceId, rankSize, rankIp, initConfig);
}
HcclResult HcomCreateCommCCLbuffer(const char* group) {
    return hcomCreateCommCCLbufferPtr(group);
}
HcclResult HcomGetInCCLbuffer(const char* group, void** buffer, u64* size) {
    return hcomGetInCCLbufferPtr(group, buffer, size);
}
HcclResult HcomGetOutCCLbuffer(const char* group, void** buffer, u64* size) {
    return hcomGetOutCCLbufferPtr(group, buffer, size);
}
void HcomSetLaunchKernelMode(bool state) {
    hcomSetLaunchKernelModePtr(state);
}
HcclResult HcomGetAicpuOpStreamNotify(const char* group, HcclRtStream* opStream, u8 aicpuNotifyNum, void** aicpuNotify) {
    return hcomGetAicpuOpStreamNotifyPtr(group, opStream, aicpuNotifyNum, aicpuNotify);
}
HcclResult HcomMc2AiCpuStreamAllocAndGet(const char* group, u32 streamMode, rtStream_t* aiCpuStream) {
    return hcomMc2AiCpuStreamAllocAndGetPtr(group, streamMode, aiCpuStream);
}
void HcomSetDumpDebugMode(bool dumpDebug) {
    hcomSetDumpDebugModePtr(dumpDebug);
}
HcclResult HcomGetAlgorithm(u32 level, char** algo) {
    return hcomGetAlgorithmPtr(level, algo);
}
HcclResult HcomGetAlgExecParam(const char* tag, const char* group, u64 count, void* inputPtr, void* outputPtr, HcclCMDType opType, bool clearEnable, HcclDataType dataType, HcclReduceOp op, void** commContext, u64* len, u32 aivCoreLimit) {
    return hcomGetAlgExecParamPtr(tag, group, count, inputPtr, outputPtr, opType, clearEnable, dataType, op, commContext, len, aivCoreLimit);
}
void HcomSetAutoTuneMode(bool autoTuneMode) {
    hcomSetAutoTuneModePtr(autoTuneMode);
}
DevType HcomGetDeviceType() {
    return hcomGetDeviceTypePtr();
}
HcclResult HcomSetProfilingMode(HcomProfilingMode profilingMode, const char* profilingOption) {
    return hcomSetProfilingModePtr(profilingMode, profilingOption);
}
HcclResult HcomGetSplitStrategy(const char* group, const struct model_feature* feature, u32** segmentIdxPtr, u32* len, bool* configured, GradSplitForceMode force, OriginalGraphShapeType shapeType) {
    return hcomGetSplitStrategyPtr(group, feature, segmentIdxPtr, len, configured, force, shapeType);
}
bool HcomFindGroup(const char* group) {
    return hcomFindGroupPtr(group);
}
HcclResult HcomSelectAlg(s64 comm, const char* group, u64 count, void* counts, HcclDataType dataType, HcclReduceOp op, HcclCMDType opType, int32_t aivCoreLimit, bool& ifAiv, char* algName) {
    return hcomSelectAlgPtr(comm, group, count, counts, dataType, op, opType, aivCoreLimit, ifAiv, algName);
}
HcclResult HcomCalcAivCoreNum(const char* group, HcclCMDType opType, u64 count, void* counts, HcclDataType dataType, int32_t aivCoreLimit, char* algName, u32* numBlocks) {
    return hcomCalcAivCoreNumPtr(group, opType, count, counts, dataType, aivCoreLimit, algName, numBlocks);
}
HcclResult HcomSetWorkspaceResource(const char* tag, const char* group, rtStream_t* stream, s32 len, void* memPtr, u64 maxSize) {
    return hcomSetWorkspaceResourcePtr(tag, group, stream, len, memPtr, maxSize);
}
HcclResult HcomSetGlobalWorkSpace(const char* group, void** globalWorkSpaceAddr, u32 len) {
    return hcomSetGlobalWorkSpacePtr(group, globalWorkSpaceAddr, len);
}
HcclResult HcomSetAivCoreLimit(const char* group, u32 aivCoreLimit) {
    return hcomSetAivCoreLimitPtr(group, aivCoreLimit);
}
HcclResult HcomReleaseSubComms() {
    return hcomReleaseSubCommsPtr();
}
HcclResult HcomUnloadTask(const char* group, const char* tag) {
    return hcomUnloadTaskPtr(group, tag);
}
HcclResult HcomClearAivSyncBuf(const char* group, bool aivClearEnable) {
    return hcomClearAivSyncBufPtr(group, aivClearEnable);
}
HcclResult HcomSetAttachedStream(const char* group, u32 graphId, const rtStream_t* stream, s32 len) {
    return hcomSetAttachedStreamPtr(group, graphId, stream, len);
}
HcclResult HcomSupportDeterministicOptim(const char* group, bool* isDeterministicOptim) {
    return hcomSupportDeterministicOptimPtr(group, isDeterministicOptim);
}
HcclResult HcomTbeMemClean(int64_t addrList[], int64_t sizeList[], uint32_t count, rtStream_t stream, int32_t deviceLogicId) {
    return hcomTbeMemCleanPtr(addrList, sizeList, count, stream, deviceLogicId);
}
HcclResult HcomGetInitStatus(bool* initiated) {
    return hcomGetInitStatusPtr(initiated);
}
HcclResult HcomAllGather(const char* tag, void* inputPtr, void* outputPtr, u64 inputCount, HcclDataType dataType, const char* group, rtStream_t stream) {
    return hcomAllGatherPtr(tag, inputPtr, outputPtr, inputCount, dataType, group, stream);
}
HcclResult HcomAllGatherV(const char* tag, const void* sendBuf, u64 sendCount, const void* recvBuf, const void* recvCounts, const void* rdispls, HcclDataType dataType, const char* group, rtStream_t stream) {
    return hcomAllGatherVPtr(tag, sendBuf, sendCount, recvBuf, recvCounts, rdispls, dataType, group, stream);
}
HcclResult HcomAllReduce(const char* tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op, const char* group, rtStream_t stream) {
    return hcomAllReducePtr(tag, inputPtr, outputPtr, count, dataType, op, group, stream);
}
HcclResult HcomReduce(const char* tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op, u32 root, const char* group, rtStream_t stream) {
    return hcomReducePtr(tag, inputPtr, outputPtr, count, dataType, op, root, group, stream);
}
HcclResult HcomBroadcast(const char* tag, void* ptr, u64 count, HcclDataType dataType, u32 root, const char* group, rtStream_t stream) {
    return hcomBroadcastPtr(tag, ptr, count, dataType, root, group, stream);
}
HcclResult HcomReduceScatter(const char* tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op, const char* group, rtStream_t stream) {
    return hcomReduceScatterPtr(tag, inputPtr, outputPtr, count, dataType, op, group, stream);
}
HcclResult HcomReduceScatterV(const char* tag, void* sendBuf, const void* sendCounts, const void* sdispls, void* recvBuf, u64 recvCount, HcclDataType dataType, HcclReduceOp op, const char* group, rtStream_t stream) {
    return hcomReduceScatterVPtr(tag, sendBuf, sendCounts, sdispls, recvBuf, recvCount, dataType, op, group, stream);
}
HcclResult HcomSend(const char* tag, void* inputPtr, u64 count, HcclDataType dataType, u32 destRank, u32 srTag, const char* group, rtStream_t stream) {
    return hcomSendPtr(tag, inputPtr, count, dataType, destRank, srTag, group, stream);
}
HcclResult HcomReceive(const char* tag, void* outputPtr, u64 count, HcclDataType dataType, u32 srcRank, u32 srTag, const char* group, rtStream_t stream) {
    return hcomReceivePtr(tag, outputPtr, count, dataType, srcRank, srTag, group, stream);
}
HcclResult HcomAlltoAllV(const void* sendBuf, const void* sendCounts, const void* sdispls, HcclDataType sendType, const void* recvBuf, const void* recvCounts, const void* rdispls, HcclDataType recvType, const char* group, rtStream_t stream, const char* tag) {
    return hcomAlltoAllVPtr(sendBuf, sendCounts, sdispls, sendType, recvBuf, recvCounts, rdispls, recvType, group, stream, tag);
}
HcclResult HcomAlltoAllVC(const void* sendBuf, const void* sendCountMatrix, HcclDataType sendType, const void* recvBuf, HcclDataType recvType, const char* group, rtStream_t stream, const char* tag) {
    return hcomAlltoAllVCPtr(sendBuf, sendCountMatrix, sendType, recvBuf, recvType, group, stream, tag);
}
HcclResult HcomAllToAll(const void* sendBuf, u64 sendCount, HcclDataType sendType, const void* recvBuf, u64 recvCount, HcclDataType recvType, const char* group, rtStream_t stream, const char* tag) {
    return hcomAllToAllPtr(sendBuf, sendCount, sendType, recvBuf, recvCount, recvType, group, stream, tag);
}
HcclResult HcomGetHcclComm(int64_t comm, std::string& group) {
    return hcomGetHcclCommPtr(comm, group);
}
HcclResult HcomGenerateCclOpTag(const char* opType, s64 hcomComm, const char* group, char* sTag) {
    return hcomGenerateCclOpTagPtr(opType, hcomComm, group, sTag);
}
HcclResult HcomGetCommCCLBufferSize(const char* group, uint64_t& size) {
    return hcomGetCommCCLBufferSizePtr(group, size);
}
HcclResult HcomGetL0TopoTypeEx(const char* group, CommTopo* topoType, uint32_t flag) {
    return hcomGetL0TopoTypeExPtr(group, topoType, flag);
}
HcclResult HcomGetRankSizeEx(const char* group, uint32_t* rankSize, uint32_t flag) {
    return hcomGetRankSizeExPtr(group, rankSize, flag);
}
HcclResult HcomInitByFile(const char* rankTablePath, const char* identify) {
    return hcomInitByFilePtr(rankTablePath, identify);
}
HcclResult HcomGetWorkspaceSubStreamNum(const char* group, u64& streamNum, u64 dataSize,
    HcclDataType dataType, u32 aivCoreLimit, HcclReduceOp reduceOp, u64 count, HcclCMDType optype) {
    return hcomGetWorkspaceSubStreamNumPtr(group, streamNum, dataSize, dataType, aivCoreLimit, reduceOp, count, optype);
}
HcclResult HcomGetWorkspaceMemSize(const std::string& opType, u64 count,
    HcclDataType dataType, const char* group, u64& memSize) {
    return hcomGetWorkspaceMemSizePtr(opType, count, dataType, group, memSize);
}
HcclResult HcomSetAlgorithm(const char* algo) {
    return hcomSetAlgorithmPtr(algo);
}
HcclResult HcomGetAlltoAllStagedWorkSpaceMemSize(const char *group, u64 *sendCounts, u64 *sdispls,
    HcclDataType sendType, u64 *recvCounts, u64 *rdispls, HcclDataType recvType, u64 &memSize) {
    return hcomGetAlltoAllStagedWorkSpaceMemSizePtr(group, sendCounts, sdispls, sendType, recvCounts, rdispls, recvType, memSize);
}
#ifdef __cplusplus
}
#endif
// ---------- 查询函数实现 ----------
#define DEFINE_QUERY(name) extern "C" bool HcommIsSupport##name(void) { return g_##name##Supported; }

DEFINE_QUERY(HcomGetRankSize)
DEFINE_QUERY(HcomGetLocalRankSize)
DEFINE_QUERY(HcomGetRankId)
DEFINE_QUERY(HcomGetLocalRankId)
DEFINE_QUERY(HcomGetWorldRankFromGroupRank)
DEFINE_QUERY(HcomGetGroupRankFromWorldRank)
DEFINE_QUERY(HcomCreateGroup)
DEFINE_QUERY(HcomDestroyGroup)
DEFINE_QUERY(HcomSetGradFusionByIndex)
DEFINE_QUERY(HcomSetGradFusionBySize)
DEFINE_QUERY(HcomInitByRankTable)
DEFINE_QUERY(HcomDestroy)
DEFINE_QUERY(HcomGetCommHandleByGroup)
DEFINE_QUERY(GetGroupNameByOpBaseHcom)
DEFINE_QUERY(HcomCreateComResourceByComm)
DEFINE_QUERY(HcomTopoInfoRegCallback)
DEFINE_QUERY(HcomGetandClearOverFlowTasks)
DEFINE_QUERY(HcomGetWorkflowMode)
DEFINE_QUERY(HcomSetWorkflowMode)
DEFINE_QUERY(HcomCalcOpOnline)
DEFINE_QUERY(HcomCalcOpResOffline)
DEFINE_QUERY(HcomGetMemType)
DEFINE_QUERY(HcomGetBandWidthPerNPU)
DEFINE_QUERY(HcomGetServerNumAndDeviceNumPerServer)
DEFINE_QUERY(HcomGetSecAddrCopyFlag)
DEFINE_QUERY(HcomInitByString)
DEFINE_QUERY(HcomInitByMasterInfo)
DEFINE_QUERY(HcomCreateCommCCLbuffer)
DEFINE_QUERY(HcomGetInCCLbuffer)
DEFINE_QUERY(HcomGetOutCCLbuffer)
DEFINE_QUERY(HcomSetLaunchKernelMode)
DEFINE_QUERY(HcomGetAicpuOpStreamNotify)
DEFINE_QUERY(HcomMc2AiCpuStreamAllocAndGet)
DEFINE_QUERY(HcomSetDumpDebugMode)
DEFINE_QUERY(HcomGetAlgorithm)
DEFINE_QUERY(HcomGetAlgExecParam)
DEFINE_QUERY(HcomSetAutoTuneMode)
DEFINE_QUERY(HcomGetDeviceType)
DEFINE_QUERY(HcomSetProfilingMode)
DEFINE_QUERY(HcomGetSplitStrategy)
DEFINE_QUERY(HcomFindGroup)
DEFINE_QUERY(HcomSelectAlg)
DEFINE_QUERY(HcomCalcAivCoreNum)
DEFINE_QUERY(HcomSetWorkspaceResource)
DEFINE_QUERY(HcomSetGlobalWorkSpace)
DEFINE_QUERY(HcomSetAivCoreLimit)
DEFINE_QUERY(HcomReleaseSubComms)
DEFINE_QUERY(HcomUnloadTask)
DEFINE_QUERY(HcomClearAivSyncBuf)
DEFINE_QUERY(HcomSetAttachedStream)
DEFINE_QUERY(HcomSupportDeterministicOptim)
DEFINE_QUERY(HcomTbeMemClean)
DEFINE_QUERY(HcomGetInitStatus)
DEFINE_QUERY(HcomAllGather)
DEFINE_QUERY(HcomAllGatherV)
DEFINE_QUERY(HcomAllReduce)
DEFINE_QUERY(HcomReduce)
DEFINE_QUERY(HcomBroadcast)
DEFINE_QUERY(HcomReduceScatter)
DEFINE_QUERY(HcomReduceScatterV)
DEFINE_QUERY(HcomSend)
DEFINE_QUERY(HcomReceive)
DEFINE_QUERY(HcomAlltoAllV)
DEFINE_QUERY(HcomAlltoAllVC)
DEFINE_QUERY(HcomAllToAll)
DEFINE_QUERY(HcomGetHcclComm)
DEFINE_QUERY(HcomGenerateCclOpTag)
DEFINE_QUERY(HcomGetCommCCLBufferSize)
DEFINE_QUERY(HcomGetL0TopoTypeEx)
DEFINE_QUERY(HcomGetRankSizeEx)
DEFINE_QUERY(HcomInitByFile)
DEFINE_QUERY(HcomGetWorkspaceSubStreamNum)
DEFINE_QUERY(HcomGetWorkspaceMemSize)
DEFINE_QUERY(HcomSetAlgorithm)
DEFINE_QUERY(HcomGetAlltoAllStagedWorkSpaceMemSize)