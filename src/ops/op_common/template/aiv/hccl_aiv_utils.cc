/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <mutex>
#include <condition_variable>
#include <vector>
#include <iostream>
#include <fstream>
#include <limits>
#include <unordered_map>
#include <deque>
#include <sstream>
#include <algorithm>
#include "rt_external.h"
#include "adapter_acl.h"
#include "hccl_aiv_utils.h"
#include "aiv_kernel_def.h"
#include "universal_concurrent_map.h"
#include "alg_env_config.h"
#ifdef HCCL_STATIC_MODE
#include "acl_rt.h"
#endif

using namespace std;
using namespace ops_hccl;

extern "C" rtError_t rtRegTaskFailCallbackByModule(const char_t *moduleName,
    rtTaskFailCallback callback) __attribute__((weak));

#ifdef HCCL_STATIC_MODE
// 静态库模式下，AIV kernel `.o` 已通过 `ld -r -b binary` 内嵌进 libhccl_static.a。
// 这里声明 ld 自动生成的 _binary_<name>_start/end 符号，
// 并在 RegisterKernel 时用 aclrtBinaryLoadFromData 从内存加载，
// 无需依赖 ${ASCEND_HOME_PATH}/lib64 下的 .o 文件。
extern "C" {
#define DECL_AIV_EMBED(stem)                                               \
    extern const char _binary_##stem##_bin_start[];                         \
    extern const char _binary_##stem##_bin_end[]
DECL_AIV_EMBED(hccl_aiv_all_gather_op_910_95);
DECL_AIV_EMBED(hccl_aiv_all_reduce_op_910_95);
DECL_AIV_EMBED(hccl_aiv_all_to_all_op_910_95);
DECL_AIV_EMBED(hccl_aiv_all_to_all_v_op_910_95);
DECL_AIV_EMBED(hccl_aiv_broadcast_op_910_95);
DECL_AIV_EMBED(hccl_aiv_reduce_op_910_95);
DECL_AIV_EMBED(hccl_aiv_reduce_scatter_op_910_95);
DECL_AIV_EMBED(hccl_aiv_scatter_op_910_95);
DECL_AIV_EMBED(hccl_aiv_send_op_910_95);
DECL_AIV_EMBED(hccl_aiv_recv_op_910_95);
#undef DECL_AIV_EMBED
}

namespace {
struct AivEmbedSymbol {
    const char *binaryName;
    const char *start;
    const char *end;
};

#define EMBED_ENTRY(stem)                                                  \
    { #stem ".o", _binary_##stem##_bin_start, _binary_##stem##_bin_end }
static const AivEmbedSymbol kAivEmbedTable[] = {
    EMBED_ENTRY(hccl_aiv_all_gather_op_910_95),
    EMBED_ENTRY(hccl_aiv_all_reduce_op_910_95),
    EMBED_ENTRY(hccl_aiv_all_to_all_op_910_95),
    EMBED_ENTRY(hccl_aiv_all_to_all_v_op_910_95),
    EMBED_ENTRY(hccl_aiv_broadcast_op_910_95),
    EMBED_ENTRY(hccl_aiv_reduce_op_910_95),
    EMBED_ENTRY(hccl_aiv_reduce_scatter_op_910_95),
    EMBED_ENTRY(hccl_aiv_scatter_op_910_95),
    EMBED_ENTRY(hccl_aiv_send_op_910_95),
    EMBED_ENTRY(hccl_aiv_recv_op_910_95),
};
#undef EMBED_ENTRY

static HcclResult LoadAivKernelFromEmbed(const std::string &aivBinaryName,
    aclrtBinHandle &binHandle)
{
    for (const auto &entry : kAivEmbedTable) {
        if (aivBinaryName != entry.binaryName) {
            continue;
        }
        size_t length = static_cast<size_t>(entry.end - entry.start);
        aclrtBinaryLoadOptions loadOptions = {0};
        aclrtBinaryLoadOption option;
        loadOptions.numOpt = 1;
        loadOptions.options = &option;
        option.type = ACL_RT_BINARY_LOAD_OPT_LAZY_LOAD;
        option.value.cpuKernelMode = 1;
        aclError aclRet = aclrtBinaryLoadFromData(
            entry.start, length, &loadOptions, &binHandle);
        if (aclRet != ACL_SUCCESS) {
            HCCL_ERROR("[LoadAivKernelFromEmbed]errNo[0x%016llx] load aiv binary[%s] "
                "from embedded data failed, length[%zu], ret[%d].",
                HCCL_ERROR_CODE(HCCL_E_RUNTIME), aivBinaryName.c_str(), length, aclRet);
            return HCCL_E_RUNTIME;
        }
        HCCL_INFO("[LoadAivKernelFromEmbed]aiv binary[%s] loaded from embedded data, "
            "length[%zu].", aivBinaryName.c_str(), length);
        return HCCL_SUCCESS;
    }
    HCCL_ERROR("[LoadAivKernelFromEmbed]aiv binary[%s] not found in embed table.",
        aivBinaryName.c_str());
    return HCCL_E_NOT_FOUND;
}
}  // namespace
#endif  // HCCL_STATIC_MODE

namespace ops_hccl {
constexpr u32 SIG_MOVE_LEFT_BITS = 20;

constexpr u32 AIV_BUFFER_PING_PONG_FACTOR = 2;

constexpr u32 MAX_BIN_FILE_SIZE = 100 * 1024 * 1024; // 最大读取100m的bin file到string中

constexpr s32 RESET_TAIL_SYNC_TAG = 2;
constexpr u32 AIV_TASK_CONTEXT_SIZE = 50;
constexpr u32 AIV_TASK_QUEUE_LIMIT = 2048;
constexpr u32 AIV_FLAG_UB_ALIGN_SIZE = 32;
constexpr u32 AIV_FLAG_PRINT_SIZE = 4096;

static mutex g_mut;
static condition_variable g_launchCv;
static u32 g_activeLaunchCount = 0;
static bool g_unregistering = false;

struct AivKernelEntry {
    aclrtBinHandle binHandle = nullptr;
    aclrtFuncHandle funcHandle = nullptr;
    std::string kernelName;

    AivKernelEntry() = default;
    AivKernelEntry(aclrtBinHandle binHandle, aclrtFuncHandle funcHandle, const std::string &kernelName)
        : binHandle(binHandle), funcHandle(funcHandle), kernelName(kernelName)
    {
    }
};

struct AivKernelLookupResult {
    s32 deviceId = 0;
    AivKernelEntry entry;
};

struct AivDeviceRegistry {
    bool initialized = false;
    std::unordered_map<std::string, aclrtBinHandle> binHandles;
    std::unordered_map<s8*, AivKernelEntry> kernels;
};

static std::unordered_map<s32, AivDeviceRegistry> g_aivRegistryByDevice;
struct TaskParamAiv {
    u64 taskId = 0;
    u64 streamId = 0;
    HcclCMDType cmdType = HcclCMDType::HCCL_CMD_INVALID;
    u32 tag = 0;
    u64 size = 0;
    u32 blockDim = 0;
    u32 rankSize = 0;
    s32 aivRdmaStep = 0;
    void *flagMem = nullptr;
    u32 rank = 0;
    bool isOpbase = false;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_RESERVED;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_RESERVED;
    uint64_t beginTime = 0;
};

static mutex g_aivTaskMutex;
static std::unordered_map<u64, std::deque<TaskParamAiv>> g_aivTaskByStream;

thread_local HcclComm g_aivCurrentComm = nullptr;
thread_local std::string g_aivCurrentCommName;

static void RegisterAivExceptionCallback()
{
    if (rtRegTaskFailCallbackByModule == nullptr) {
        HCCL_INFO("[AIV][RegisterAivExceptionCallback] runtime callback registration is not supported.");
        return;
    }
    rtError_t rtRet = rtRegTaskFailCallbackByModule("HCCL_OPS", ProcessAivExceptionCallBack);
    if (rtRet != RT_ERROR_NONE) {
        HCCL_WARNING("[AIV][RegisterAivExceptionCallback] register callback failed, ret[%d].", rtRet);
    } else {
        HCCL_INFO("[AIV][RegisterAivExceptionCallback] register callback success.");
    }
}

static HcclResult SaveAivDfxTaskInfo(const AivOpArgs &opArgs)
{
    u32 taskId = 0;
    u32 streamId = 0;
    rtError_t rtRet = rtGetTaskIdAndStreamID(&taskId, &streamId);
    CHK_PRT_RET(rtRet != RT_ERROR_NONE,
        HCCL_ERROR("[AIV][SaveAivDfxTaskInfo] rtGetTaskIdAndStreamID failed, ret[%d].", rtRet), HCCL_E_RUNTIME);

    TaskParamAiv taskInfo;
    taskInfo.taskId = taskId;
    taskInfo.streamId = streamId;
    taskInfo.cmdType = opArgs.cmdType;
    taskInfo.tag = opArgs.sliceId;
    taskInfo.size = opArgs.count;
    taskInfo.blockDim = opArgs.numBlocks;
    taskInfo.rankSize = opArgs.rankSize;
    taskInfo.aivRdmaStep = 0;
    taskInfo.flagMem = static_cast<void *>(opArgs.buffersIn == nullptr ? nullptr :
        static_cast<u8 *>(opArgs.buffersIn) + AIV_FLAG_ADDR_OFFSET);
    taskInfo.rank = opArgs.rank;
    taskInfo.isOpbase = opArgs.isOpBase;
    taskInfo.reduceOp = opArgs.op;
    taskInfo.dataType = opArgs.dataType;
    taskInfo.beginTime = opArgs.beginTime;

    HCCL_INFO("Begin to SaveAivDfxTaskInfo taskType[%d]", static_cast<int32_t>(opArgs.cmdType));
    std::lock_guard<mutex> lock(g_aivTaskMutex);
    auto &taskQueue = g_aivTaskByStream[streamId];
    taskQueue.push_back(taskInfo);
    if (taskQueue.size() > AIV_TASK_QUEUE_LIMIT) {
        taskQueue.pop_front();
    }
    return HCCL_SUCCESS;
}

static bool FindAivTask(u32 streamId, u32 taskId, TaskParamAiv &taskInfo, std::deque<TaskParamAiv> &taskQueue)
{
    std::lock_guard<mutex> lock(g_aivTaskMutex);
    auto streamIt = g_aivTaskByStream.find(streamId);
    if (streamIt == g_aivTaskByStream.end()) {
        return false;
    }
    auto taskIt = std::find_if(streamIt->second.begin(), streamIt->second.end(),
        [taskId](const TaskParamAiv &task) { return task.taskId == taskId; });
    if (taskIt == streamIt->second.end()) {
        return false;
    }
    taskInfo = *taskIt;
    taskQueue.assign(streamIt->second.begin(), std::next(taskIt));
    return true;
}

static std::string SerializeAivFlag(const TaskParamAiv &taskInfo)
{
    if (taskInfo.flagMem == nullptr) {
        return "flagMem is nullptr";
    }
    std::vector<s32> flagMem(AIV_FLAG_PRINT_SIZE / sizeof(s32));
    HcclResult ret = haclrtMemcpy(flagMem.data(), AIV_FLAG_PRINT_SIZE, taskInfo.flagMem, AIV_FLAG_PRINT_SIZE,
        ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != HCCL_SUCCESS) {
        return "copy flagMem failed";
    }

    std::stringstream ss;
    const u32 alignStep = AIV_FLAG_UB_ALIGN_SIZE / sizeof(s32);
    for (u32 i = 0; i < flagMem.size(); i += alignStep) {
        ss << flagMem[i] << " ";
    }
    return ss.str();
}

void ProcessAivExceptionCallBack(aclrtExceptionInfo *exceptionInfo)
{
    if (exceptionInfo == nullptr) {
        HCCL_ERROR("[TaskExceptionHandler][AIV] exceptionInfo is nullptr.");
        return;
    }

    const u32 taskId = aclrtGetTaskIdFromExceptionInfo(exceptionInfo);
    const u32 streamId = aclrtGetStreamIdFromExceptionInfo(exceptionInfo);
    const u32 deviceId = aclrtGetDeviceIdFromExceptionInfo(exceptionInfo);

    TaskParamAiv taskInfo;
    std::deque<TaskParamAiv> taskQueue;
    if (!FindAivTask(streamId, taskId, taskInfo, taskQueue)) {
        HCCL_RUN_INFO("[TaskExceptionHandler][AIV] task not found, streamId[%u], taskId[%u].",
            streamId, taskId);
        return;
    }

    HCCL_ERROR("[TaskExceptionHandler][AIV]Task run failed, para information is deviceId[%u], streamId[%u], "
        "TaskId[%u], cmdType[%u], tag[%u], rank[%u], rankSize[%u], dataCount[%llu], blockDim[%u], "
        "dataType:[%u], beginTime:[%llu], flagMem[%p]",
        deviceId, streamId, taskId, taskInfo.cmdType, taskInfo.tag,
        taskInfo.rank, taskInfo.rankSize, taskInfo.size, taskInfo.blockDim, taskInfo.dataType, taskInfo.beginTime,
        taskInfo.flagMem);

    HCCL_ERROR("[TaskExceptionHandler][AIV]Task run failed, para information is deviceId[%u], streamId[%u], "
        "TaskId[%u]. flag: %s", deviceId, streamId, taskId,
        SerializeAivFlag(taskInfo).c_str());

    HCCL_ERROR("[TaskExceptionHandler][AIV]Task run failed, para information is deviceId[%u], streamId[%u], "
        "TaskId[%u]. task info before failed task is:", deviceId, streamId, taskId);

    u32 printed = 0;
    for (auto it = taskQueue.rbegin(); it != taskQueue.rend() && printed < AIV_TASK_CONTEXT_SIZE; ++it) {
        if (it->taskId == taskInfo.taskId) {
            continue;
        }
        HCCL_ERROR("[TaskExceptionHandler][AIV] previous TaskId[%llu], streamId[%llu], cmdType[%u], "
            "tag[%u], rank[%u], rankSize[%u], dataCount[%llu], blockDim[%u], dataType:[%u], beginTime:[%llu], "
            "flagMem[%p]", it->taskId, it->streamId, it->cmdType, it->tag, it->rank, it->rankSize, it->size,
            it->blockDim, it->dataType, it->beginTime, it->flagMem);
        ++printed;
    }
}

class AivLaunchGuard {
public:
    HcclResult Acquire()
    {
        unique_lock<mutex> lock(g_mut);
        if (g_unregistering) {
            return HCCL_E_RUNTIME;
        }
        ++g_activeLaunchCount;
        active_ = true;
        return HCCL_SUCCESS;
    }

    ~AivLaunchGuard()
    {
        if (!active_) {
            return;
        }
        unique_lock<mutex> lock(g_mut);
        if (g_activeLaunchCount > 0) {
            --g_activeLaunchCount;
        }
        lock.unlock();
        g_launchCv.notify_all();
    }

private:
    bool active_ = false;
};

thread_local std::shared_ptr<InsQueue> g_recordingQueue = nullptr;
thread_local bool g_recordOnlyMode = false;
thread_local u64 g_baseInputAddr = 0;
thread_local u64 g_baseOutputAddr = 0;
static HcclResult GetMinAndMaxNpuSchedTimeOut(u64 &minNpuSchedTimeout, u64 &maxNpuSchedTimeout)
{
    uint64_t interval = 0;
    aclError aclRet = aclrtGetOpTimeOutInterval(&interval);
    CHK_PRT_RET(aclRet != ACL_SUCCESS, HCCL_WARNING("[GetMinAndMaxNpuSchedTimeOut] get timeout interval failed, ret[%d].",
        aclRet), HCCL_E_RUNTIME);

    constexpr u64 MAX_INTERVAL = 254;
    minNpuSchedTimeout = interval;
    maxNpuSchedTimeout = MAX_INTERVAL * interval;
    return HCCL_SUCCESS;
}

static u32 GetAivTimeout()
{
    constexpr u32 AIV_TIMEOUT_DEFAULT = 1091;
    constexpr u32 AIV_TIMEOUT_DEFAULT_US = AIV_TIMEOUT_DEFAULT * TIME_S_TO_US;

    u32 timeoutUs = AIV_TIMEOUT_DEFAULT_US;
    double execTimeOut = AIV_TIMEOUT_DEFAULT;
    if (GetExternalInputExecTimeout(execTimeOut)) {
        timeoutUs = execTimeOut * TIME_S_TO_US;
        if (timeoutUs > static_cast<double>(std::numeric_limits<u32>::max())) {
            HCCL_WARNING("[GetAivTimeout]Get input timeout[%.2f] is out of valid range.", timeoutUs);
            timeoutUs = AIV_TIMEOUT_DEFAULT_US;
        } else if (static_cast<u32>(timeoutUs) == 0) {
            timeoutUs = AIV_TIMEOUT_DEFAULT_US;
        }
    }

    u64 minNpuSchedTimeout = 0;
    u64 maxNpuSchedTimeout = 0;
    if (GetMinAndMaxNpuSchedTimeOut(minNpuSchedTimeout, maxNpuSchedTimeout) != HCCL_SUCCESS) {
        HCCL_WARNING("[GetAivTimeout] get npu sched timeout range failed, use default[%u]us.", AIV_TIMEOUT_DEFAULT_US);
        return AIV_TIMEOUT_DEFAULT_US;
    }
    u32 finalTimeout = (timeoutUs < minNpuSchedTimeout) ? minNpuSchedTimeout
            : (timeoutUs > maxNpuSchedTimeout) ? maxNpuSchedTimeout
            : timeoutUs;
    HCCL_INFO("[GetAivTimeout]timeout[%u]us, execTimeOut[%.2f]s, minNpuSchedTimeout[%u]us, maxNpuSchedTimeout[%u]us.",
        finalTimeout, execTimeOut, minNpuSchedTimeout, maxNpuSchedTimeout);

    return finalTimeout;
}

using AivKernelArgs = struct AivKernelArgsDef {
    const void* buffersIn; // 注册的CCLIN地址，所有卡可访问
    u64 input;
    u64 output;
    u32 rank;
    u32 sendRecvRemoteRank;
    u32 rankSize;
    u64 xRankSize;
    u64 yRankSize;
    u64 zRankSize;
    u64 len;
    u32 dataType;
    u32 reduceOp;
    u32 root;
    u32 tag; // 第几次调用，定时重置成1
    u64 inputSliceStride;
    u64 outputSliceStride;
    u64 repeatNum;
    u64 inputRepeatStride;
    u64 outputRepeatStride;
    u32 numBlocks;
    bool isOpBase;
    const void* headCountMem;
    const void* tailCountMem;
    const void* addOneMem;
    u32 counterMemSize;
    bool isEnableCounter;

    AivKernelArgsDef(const void* buffIn, u64 input, u64 output, u32 rank, u32 sendRecvRemoteRank,
        u32 rankSize, u64 xRankSize, u64 yRankSize, u64 zRankSize,
        u64 len, u32 dataType, u32 reduceOp, u32 root, u32 tag,
        u64 inputSliceStride, u64 outputSliceStride, u64 repeatNum, u64 inputRepeatStride, u64 outputRepeatStride, u32 numBlocks,
        bool isOpBase = true,
        const void* headCountMem = nullptr, const void* tailCountMem = nullptr, const void* addOneMem = nullptr,
        u32 counterMemSize = 0, bool isEnableCounter = false)
        : buffersIn(buffIn),input(input), output(output), rank(rank), sendRecvRemoteRank(sendRecvRemoteRank), rankSize(rankSize), xRankSize(xRankSize), yRankSize(yRankSize), zRankSize(zRankSize),
        len(len) ,dataType(dataType),
        reduceOp(reduceOp), root(root), tag(tag),
        inputSliceStride(inputSliceStride), outputSliceStride(outputSliceStride), repeatNum(repeatNum), inputRepeatStride(inputRepeatStride), outputRepeatStride(outputRepeatStride),
        numBlocks(numBlocks), isOpBase(isOpBase),
        headCountMem(headCountMem), tailCountMem(tailCountMem), addOneMem(addOneMem),
        counterMemSize(counterMemSize), isEnableCounter(isEnableCounter)
    {
    }
};

using AivExtraKernelArgs = struct AivExtraKernelArgsDef {
    const void* buffersIn; // 注册的CCLIN地址，所有卡可访问
    u64 input;
    u64 output;
    u32 rank;
    u32 sendRecvRemoteRank;
    u32 rankSize;
    u64 xRankSize;
    u64 yRankSize;
    u64 zRankSize;
    u64 len;
    u32 dataType;
    u32 reduceOp;
    u32 root;
    u32 tag; // 第几次调用，定时重置成1
    u64 inputSliceStride;
    u64 outputSliceStride;
    u64 repeatNum;
    u64 inputRepeatStride;
    u64 outputRepeatStride;
    u32 numBlocks;
    bool isOpBase;
    const void* headCountMem;
    const void* tailCountMem;
    const void* addOneMem;
    u32 counterMemSize;
    bool isEnableCounter;
    ExtraArgs extraArgs;

    AivExtraKernelArgsDef(const void* buffIn, u64 input, u64 output, u32 rank, u32 sendRecvRemoteRank,
        u32 rankSize, u64 xRankSize, u64 yRankSize, u64 zRankSize,
        u64 len, u32 dataType, u32 reduceOp, u32 root, u32 tag,
        u64 inputSliceStride, u64 outputSliceStride, u64 repeatNum, u64 inputRepeatStride, u64 outputRepeatStride, u32 numBlocks,
        bool isOpBase = true,
        const void* headCountMem = nullptr, const void* tailCountMem = nullptr, const void* addOneMem = nullptr,
        u32 counterMemSize = 0, const ExtraArgs* extraArgsPtr = nullptr)
        : buffersIn(buffIn),input(input), output(output), rank(rank), sendRecvRemoteRank(sendRecvRemoteRank), rankSize(rankSize), xRankSize(xRankSize), yRankSize(yRankSize), zRankSize(zRankSize),
        len(len) ,dataType(dataType),
        reduceOp(reduceOp), root(root), tag(tag),
        inputSliceStride(inputSliceStride), outputSliceStride(outputSliceStride), repeatNum(repeatNum), inputRepeatStride(inputRepeatStride), outputRepeatStride(outputRepeatStride),
        numBlocks(numBlocks), isOpBase(isOpBase),
        headCountMem(headCountMem), tailCountMem(tailCountMem), addOneMem(addOneMem),
        counterMemSize(counterMemSize)
    {
        if (extraArgsPtr != nullptr) {
            extraArgs = *extraArgsPtr;
        }
    }
};

HcclResult GetAivOpBinaryPath(const std::string &aivBinaryName, std::string &binaryPath)
{
    // 获取二进制文件路径
    std::string libPath;
    char *getPath = getenv("ASCEND_HOME_PATH");
    if (getPath != nullptr) {
        libPath = getPath;
    } else {
        libPath = "/usr/local/Ascend/cann";
        HCCL_WARNING("[GetAivOpBinaryPath]ENV:ASCEND_HOME_PATH is not set");
    }
    binaryPath = libPath + "/lib64";

    // 拼接应该加载的文件
    binaryPath += "/" + aivBinaryName;
    HCCL_INFO("[GetAivOpBinaryPath]op binary file path[%s]", binaryPath.c_str());
    return HCCL_SUCCESS;
}

s8* GetFuncKey(HcclCMDType cmdType, HcclDataType dataType, KernelArgsType argsType = KernelArgsType::ARGS_TYPE_SERVER)
{
    return reinterpret_cast<s8*>(
        (((static_cast<u64>(cmdType) << SIG_MOVE_LEFT_BITS) + static_cast<u64>(dataType)) << SIG_MOVE_LEFT_BITS) +
        static_cast<u64>(argsType));
}

static HcclResult GetCurrentDeviceId(s32 &deviceId)
{
    ACLCHECK(aclrtGetDevice(&deviceId));
    return HCCL_SUCCESS;
}

static HcclResult RegisterBinaryKernel(AivDeviceRegistry &registry, const char* funcName,
    const aclrtBinHandle binHandle, const s8* funcKey)
{
    if (funcKey == nullptr) {
        return HCCL_E_PARA;
    }

    aclrtFuncHandle funcHandle;
    aclError aclRet = aclrtBinaryGetFunction(binHandle, funcName, &funcHandle);
    CHK_PRT_RET(aclRet != ACL_SUCCESS, HCCL_ERROR("[RegisterBinaryKernel]errNo[0x%016llx] get function from binary error.", aclRet),
        HCCL_E_NOT_FOUND);

    registry.kernels[const_cast<s8*>(funcKey)] = AivKernelEntry(binHandle, funcHandle, std::string(funcName));

    return HCCL_SUCCESS;
}

static HcclResult GetKernelEntry(AivKernelLookupResult &lookupResult, const s8* funcKey)
{
    if (funcKey == nullptr) {
        return HCCL_E_PARA;
    }

    s32 deviceId = 0;
    CHK_RET(GetCurrentDeviceId(deviceId));
    lock_guard<mutex> guard(g_mut);
    auto registryIt = g_aivRegistryByDevice.find(deviceId);
    if (registryIt == g_aivRegistryByDevice.end() || !registryIt->second.initialized) {
        return HCCL_E_PARA;
    }

    auto kernelIt = registryIt->second.kernels.find(const_cast<s8*>(funcKey));
    if (kernelIt == registryIt->second.kernels.end()) {
        return HCCL_E_PARA;
    }
    lookupResult.deviceId = deviceId;
    lookupResult.entry = kernelIt->second;
    return HCCL_SUCCESS;
}

static HcclResult UpdateKernelFunc(s32 deviceId, const s8* funcKey, const aclrtFuncHandle funcHandle)
{
    if (funcKey == nullptr) {
        return HCCL_E_PARA;
    }

    lock_guard<mutex> guard(g_mut);
    auto registryIt = g_aivRegistryByDevice.find(deviceId);
    if (registryIt == g_aivRegistryByDevice.end()) {
        return HCCL_E_PARA;
    }

    auto kernelIt = registryIt->second.kernels.find(const_cast<s8*>(funcKey));
    if (kernelIt == registryIt->second.kernels.end()) {
        return HCCL_E_PARA;
    }
    kernelIt->second.funcHandle = funcHandle;
    return HCCL_SUCCESS;
}

static HcclResult ClearDeviceRegistry(AivDeviceRegistry &registry)
{
    HcclResult result = HCCL_SUCCESS;
    for (auto binIt = registry.binHandles.begin(); binIt != registry.binHandles.end();) {
        aclError aclRet = aclrtBinaryUnLoad(binIt->second);
        if (aclRet != ACL_SUCCESS) {
            HCCL_ERROR("[ClearDeviceRegistry] unload aiv binary[%s] failed, ret[%d].", binIt->first.c_str(), aclRet);
            result = HCCL_E_RUNTIME;
            ++binIt;
            continue;
        }
        binIt = registry.binHandles.erase(binIt);
    }
    if (result == HCCL_SUCCESS) {
        registry.kernels.clear();
        registry.initialized = false;
    }
    return result;
}

// Kernel注册入口，每个device只需要初始化一次
HcclResult RegisterKernel()
{
    s32 deviceId = 0;
    CHK_RET(GetCurrentDeviceId(deviceId));

    lock_guard<mutex> guard(g_mut);
    if (g_unregistering) {
        HCCL_ERROR("[AIV][RegisterKernel] aiv kernel is unregistering.");
        return HCCL_E_RUNTIME;
    }
    AivDeviceRegistry &registry = g_aivRegistryByDevice[deviceId];
    if (registry.initialized) {
        return HCCL_SUCCESS;
    }

    for (const auto& item : g_aivKernelInfoMap) {
        const HcclCMDType cmdType = item.first;
        const std::string& aivBinaryName = item.second.first;
        const std::vector<AivKernelInfo>& aivKernelInfoList = item.second.second;

        HcclResult ret;
        aclrtBinHandle binHandle = nullptr;
        auto binHandleIt = registry.binHandles.find(aivBinaryName);
        if (binHandleIt != registry.binHandles.end()) {
            binHandle = binHandleIt->second;
        } else {
#ifdef HCCL_STATIC_MODE
            ret = LoadAivKernelFromEmbed(aivBinaryName, binHandle);
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR("[AIV][RegisterKernel] load aiv kernel from embedded data failed");
                ClearDeviceRegistry(registry);
                return HCCL_E_RUNTIME;
            }
#else
            string binFilePath;
            ret = GetAivOpBinaryPath(aivBinaryName, binFilePath);
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR("[AIV][RegisterKernel] get aiv op binary path failed");
                ClearDeviceRegistry(registry);
                return HCCL_E_RUNTIME;
            }
            ret = LoadBinaryFromFile(binFilePath.c_str(), ACL_RT_BINARY_LOAD_OPT_LAZY_LOAD, 1, binHandle);
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR("[AIV][RegisterKernel] read aiv kernel bin file failed");
                ClearDeviceRegistry(registry);
                return HCCL_E_RUNTIME;
            }
#endif
            registry.binHandles[aivBinaryName] = binHandle;
        }

        for (auto &aivKernelInfo: aivKernelInfoList) {
            ret = RegisterBinaryKernel(registry, aivKernelInfo.kernelName, binHandle,
                GetFuncKey(cmdType, aivKernelInfo.dataType, aivKernelInfo.argsType));
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR("[AIV][RegisterKernel] register binary kernel for kernelName[%s] cmdType[%d] "
                    "dataType[%s] argsType[%d] failed", aivKernelInfo.kernelName, cmdType,
                    GetDataTypeEnumStr(aivKernelInfo.dataType).c_str(), aivKernelInfo.argsType);
                ClearDeviceRegistry(registry);
                return HCCL_E_RUNTIME;
            }
        }
    }

    registry.initialized = true;

    RegisterAivExceptionCallback();

    return HCCL_SUCCESS;
}

HcclResult UnRegisterAivKernel()
{
    unique_lock<mutex> lock(g_mut);
    g_unregistering = true;
    g_launchCv.wait(lock, []() { return g_activeLaunchCount == 0; });
    HcclResult result = HCCL_SUCCESS;
    s32 currentDeviceId = 0;
    bool needRestoreDevice = (aclrtGetDevice(&currentDeviceId) == ACL_SUCCESS);
    for (auto registryIt = g_aivRegistryByDevice.begin(); registryIt != g_aivRegistryByDevice.end();) {
        aclError aclRet = aclrtSetDevice(registryIt->first);
        if (aclRet != ACL_SUCCESS) {
            HCCL_ERROR("[UnRegisterAivKernel] set device[%d] failed, ret[%d].", registryIt->first, aclRet);
            result = HCCL_E_RUNTIME;
            ++registryIt;
            continue;
        }

        HcclResult clearRet = ClearDeviceRegistry(registryIt->second);
        if (clearRet != HCCL_SUCCESS) {
            result = clearRet;
        }
        if (clearRet == HCCL_SUCCESS) {
            registryIt = g_aivRegistryByDevice.erase(registryIt);
        } else {
            ++registryIt;
        }
    }
    if (needRestoreDevice) {
        aclError aclRet = aclrtSetDevice(currentDeviceId);
        if (aclRet != ACL_SUCCESS) {
            HCCL_ERROR("[UnRegisterAivKernel] restore device[%d] failed, ret[%d].", currentDeviceId, aclRet);
            result = HCCL_E_RUNTIME;
        }
    }
    g_unregistering = false;
    lock.unlock();
    g_launchCv.notify_all();

    return result;
}

// cache工具接口
void HashAppend(u64 &hash, const void *data, size_t size)
{
    const u8 *bytes = static_cast<const u8 *>(data);
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= FNV_PRIME;
    }
}

void HashAppendString(u64 &hash, const std::string &value)
{
    HashAppend(hash, value.data(), value.size());
    const char separator = '\0';
    HashAppend(hash, &separator, sizeof(separator));
}

template <typename T>
void HashAppendValue(u64 &hash, const T &value)
{
    HashAppend(hash, &value, sizeof(T));
}

u64 CalcAivCacheKeyHash(const AivOpCacheArgs &cacheKey)
{
    u64 hash = FNV_OFFSET_BASIS;
    HashAppendValue(hash, cacheKey.count);
    HashAppendValue(hash, cacheKey.dataType);
    HashAppendValue(hash, cacheKey.opType);
    HashAppendValue(hash, cacheKey.reduceOp);
    HashAppendValue(hash, cacheKey.root);
    HCCL_INFO("[%s] hashKey[%llu]", __func__, hash);
    return hash;
}

HcclResult BuildAivCacheCtxTag(u64 keyHash, std::string &ctxTag)
{
    char tag[AIV_CACHE_CTX_TAG_MAX_LENGTH] = {};
    int ret = snprintf_s(tag, sizeof(tag), sizeof(tag) - 1, "AivCache_%016llx", keyHash);
    CHK_PRT_RET(ret <= 0, HCCL_ERROR("[%s] failed to fill aiv cache ctx tag", __func__), HCCL_E_INTERNAL);
    ctxTag = tag;
    return HCCL_SUCCESS;
}

HcclResult GetOrCreateAivCacheIndexCtx(HcclComm comm, AivCacheIndexCtx **indexCtx)
{
    CHK_PTR_NULL(indexCtx);
    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    HcclResult ret = HcclEngineCtxGet(comm, AIV_CACHE_INDEX_CTX_TAG, CommEngine::COMM_ENGINE_CPU_TS, &ctx, &ctxSize);
    if (ret == HCCL_E_NOT_FOUND || ret == HCCL_E_PARA) {
        HCCL_INFO("[%s] aiv cache indexCtx not created", __func__);
        CHK_RET(HcclEngineCtxCreate(comm, AIV_CACHE_INDEX_CTX_TAG, CommEngine::COMM_ENGINE_CPU_TS,
            sizeof(AivCacheIndexCtx), &ctx));
        *indexCtx = static_cast<AivCacheIndexCtx *>(ctx);
        // 初始化
        (*indexCtx)->head = 0;
        (*indexCtx)->tail = 0;
        (*indexCtx)->size = 0;
        return HCCL_SUCCESS;
    } else if (ret == HCCL_SUCCESS) {
        HCCL_DEBUG("[%s] aiv cache indexCtx already created", __func__);
        *indexCtx = static_cast<AivCacheIndexCtx *>(ctx);
        return HCCL_SUCCESS;
    } else {
        return ret;
    }
}

HcclResult EvictAivCacheIfNeeded(HcclComm comm, AivCacheIndexCtx *indexCtx)
{
    if (indexCtx->size < AIV_CACHE_INDEX_MAX_ENTRY) {
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[%s] begin to clean aiv cache", __func__);

    u32 clearCount = static_cast<u32>(AIV_CACHE_INDEX_MAX_ENTRY * AIV_CACHE_INDEX_CLEAR_PERCENT / 100);
    for (u32 i = 0; i < clearCount; ++i) {
        u32 evictIndex = indexCtx->head;
        const char *ctxTag = indexCtx->ctxTags[evictIndex];
        HcclResult ret = HcclEngineCtxDestroy(comm, ctxTag, CommEngine::COMM_ENGINE_CPU_TS);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[%s] failed to destroy aiv cache ctx, tag[%s]", __func__, ctxTag), ret);
        indexCtx->size--;
        indexCtx->head = (evictIndex + 1) % AIV_CACHE_INDEX_MAX_ENTRY;
        HCCL_DEBUG("[%s] cur head[%u] tail[%u] size[%u]", __func__, indexCtx->head, indexCtx->tail, indexCtx->size);
    }
    return HCCL_SUCCESS;
}

HcclResult LookupAivCacheCtx(HcclComm comm, const std::string &ctxTag, u64 keyHash, bool &cacheHit,
                             std::string &algName, AivInstruction *&instructions, u32 &insCount)
{
    cacheHit = false;
    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    HcclResult ret = HcclEngineCtxGet(comm, ctxTag.c_str(), CommEngine::COMM_ENGINE_CPU_TS, &ctx, &ctxSize);
    if (ret == HCCL_E_NOT_FOUND || ret == HCCL_E_PARA) {
        return HCCL_SUCCESS;
    }
    CHK_RET(ret);

    AivCacheCtxHeader *header = static_cast<AivCacheCtxHeader *>(ctx);
    CHK_PRT_RET(header->keyHash != keyHash,
        HCCL_ERROR("[%s] invalid aiv cache ctx header, tag[%s], keyHash[%llu]",
            __func__, ctxTag.c_str(), header->keyHash), HCCL_E_INTERNAL);
    uint64_t expectedSize = sizeof(AivCacheCtxHeader) + header->algNameLen +
        header->insCount * sizeof(AivInstruction);
    CHK_PRT_RET(ctxSize != expectedSize,
        HCCL_ERROR("[%s] invalid aiv cache ctx size[%llu], expected[%llu], tag[%s]",
            __func__, ctxSize, expectedSize, ctxTag.c_str()), HCCL_E_INTERNAL);

    const char *algNamePtr = reinterpret_cast<const char *>(
        static_cast<u8 *>(ctx) + sizeof(AivCacheCtxHeader));
    algName.assign(algNamePtr, header->algNameLen);
    instructions = reinterpret_cast<AivInstruction *>(
        static_cast<u8 *>(ctx) + sizeof(AivCacheCtxHeader) + header->algNameLen);
    insCount = header->insCount;

    HCCL_INFO("[%s] aiv cache hits, ctxTag[%s], keyHash[%llu], insCount[%u], algName[%s]",
        __func__, ctxTag.c_str(), header->keyHash, insCount, algName.c_str());
    cacheHit = true;
    return HCCL_SUCCESS;
}

HcclResult ReplayAivInstructions(const AivInstruction *instructions, u32 insCount, OpParam &param)
{
    for (uint32_t i = 0; i < insCount; ++i) {
        AivOpArgs newArgs = instructions[i].opArgs;
        newArgs.stream = param.stream;
        newArgs.input = reinterpret_cast<u64>(param.inputPtr) + instructions[i].inputOffset;
        newArgs.output = reinterpret_cast<u64>(param.outputPtr) + instructions[i].outputOffset;
        CHK_RET(ExecuteKernelLaunch(newArgs));
    }
    return HCCL_SUCCESS;
}

HcclResult StoreAivCacheCtx(HcclComm comm, const std::string &ctxTag, u64 keyHash, const std::string &algName,
                            AivCacheIndexCtx *indexCtx)
{
    const InsQueue &queue = *g_recordingQueue;
    u32 algNameLen = static_cast<u32>(algName.size());
    uint64_t ctxSize = sizeof(AivCacheCtxHeader) + algNameLen + queue.size() * sizeof(AivInstruction);
    void *ctx = nullptr;
    CHK_RET(HcclEngineCtxCreate(comm, ctxTag.c_str(), CommEngine::COMM_ENGINE_CPU_TS, ctxSize, &ctx));
    AivCacheCtxHeader header { keyHash, static_cast<u32>(queue.size()), algNameLen };
    CHK_SAFETY_FUNC_RET(memcpy_s(ctx, ctxSize, &header, sizeof(header)));
    u8 *ptr = static_cast<u8 *>(ctx) + sizeof(AivCacheCtxHeader);
    if (algNameLen > 0) {
        CHK_SAFETY_FUNC_RET(memcpy_s(ptr, algNameLen, algName.c_str(), algNameLen));
        ptr += algNameLen;
    }
    if (!queue.empty()) {
        CHK_SAFETY_FUNC_RET(memcpy_s(ptr, ctxSize - sizeof(AivCacheCtxHeader) - algNameLen,
            queue.data(), queue.size() * sizeof(AivInstruction)));
    }
    // 增加index中的记录
    u32 pushIndex = indexCtx->tail;
    char *ctxTagPtr = indexCtx->ctxTags[pushIndex];
    s32 ret = strncpy_s(ctxTagPtr, AIV_CACHE_CTX_TAG_MAX_LENGTH, ctxTag.c_str(), AIV_CACHE_CTX_TAG_MAX_LENGTH - 1);
    CHK_PRT_RET(ret != EOK, HCCL_ERROR("[%s] failed to copy aiv cache ctx tag, ret[%d]", __func__, ret),
        HCCL_E_MEMORY);
    indexCtx->tail = (pushIndex + 1) % AIV_CACHE_INDEX_MAX_ENTRY;
    indexCtx->size++;
    HCCL_INFO("[%s] ctxTag[%s] keyHash[%llu] cur head[%u] tail[%u] size[%u]",
        __func__, ctxTag.c_str(), keyHash, indexCtx->head, indexCtx->tail, indexCtx->size);
    return HCCL_SUCCESS;
}

// KernelLaunch内部接口
HcclResult ExecuteKernelLaunchInner(const AivOpArgs &opArgs, void* args, u32 argsSize)
{
    AivLaunchGuard launchGuard;
    HcclResult guardRet = launchGuard.Acquire();
    CHK_PRT_RET(guardRet != HCCL_SUCCESS, HCCL_ERROR("[ExecuteKernelLaunchInner] aiv kernel is unregistering."),
        HCCL_E_RUNTIME);

    HCCL_INFO("[ExecuteKernelLaunchInner] sendbuff [%p] recvbuff [%p] rank [%u] sendRecvRemoteRank [%u] rankSize [%u] count [%llu] "
        "dataType [%d] reduceOp [%d] root [%u] tag [%u] isOpBase [%d] "
        "extraArgsPtr [%p] argsSize [%u]", opArgs.input, opArgs.output, opArgs.rank, opArgs.sendRecvRemoteRank, opArgs.rankSize, opArgs.count,
        opArgs.dataType, opArgs.op, opArgs.root, opArgs.sliceId, opArgs.isOpBase, args, argsSize);

    aclrtLaunchKernelCfg cfg;
    aclrtLaunchKernelAttr attr[AIV_ATTRNUM_THREE];
    attr[0].id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
    attr[0].value.schemMode = 1;
    attr[1].id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT_US;
    attr[1].value.timeoutUs.timeoutLow = GetAivTimeout();
    attr[1].value.timeoutUs.timeoutHigh = 0;
    attr[2].id = ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE;
    attr[2].value.engineType = ACL_RT_ENGINE_TYPE_AIV;
    cfg.numAttrs = AIV_ATTRNUM_THREE;
    cfg.attrs = attr;

    HCCL_INFO("[ExecuteKernelLaunchInner] KernelAttr attr[0]: id=%u, schemMode=%u; attr[1]: id=%u, timeoutLow=%u, "
        "timeoutHigh=%u; attr[2]: id=%u, engineType=%u; cfg: numAttrs=%u",
        attr[0].id, attr[0].value.schemMode, attr[1].id, attr[1].value.timeoutUs.timeoutLow,
        attr[1].value.timeoutUs.timeoutHigh, attr[2].id, attr[2].value.engineType, cfg.numAttrs);

    s8* funcKey = GetFuncKey(opArgs.cmdType, opArgs.dataType, opArgs.argsType);
    AivKernelLookupResult kernelLookupResult;
    HcclResult ret = GetKernelEntry(kernelLookupResult, funcKey);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ExecuteKernelLaunchInner] funcKey[%p] errNo[0x%016llx] GetKernelEntry failed, "
        "return[%d]", funcKey, HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret), HCCL_E_RUNTIME);

    aclrtFuncHandle funcHandle = kernelLookupResult.entry.funcHandle;
    aclError aclRet = aclrtLaunchKernelWithHostArgs(funcHandle, opArgs.numBlocks, opArgs.stream,
        &cfg, args, argsSize, nullptr, 0);
    if (aclRet == ACL_ERROR_RT_INVALID_HANDLE) {
        HCCL_WARNING("[ExecuteKernelLaunchInner] handle invalid, retry to get function");
        aclRet = aclrtBinaryGetFunction(kernelLookupResult.entry.binHandle, kernelLookupResult.entry.kernelName.c_str(), &funcHandle);
        CHK_PRT_RET(aclRet != ACL_SUCCESS, HCCL_ERROR("[ExecuteKernelLaunchInner] retry get function failed, error[%d]", aclRet), HCCL_E_RUNTIME);
        ret = UpdateKernelFunc(kernelLookupResult.deviceId, funcKey, funcHandle);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ExecuteKernelLaunchInner] update function handle failed, ret[%d]", ret), HCCL_E_RUNTIME);
        aclRet = aclrtLaunchKernelWithHostArgs(funcHandle, opArgs.numBlocks, opArgs.stream, &cfg, args, argsSize, nullptr, 0);
    }
    CHK_PRT_RET(aclRet != ACL_SUCCESS, HCCL_ERROR("[ExecuteKernelLaunchInner]errNo[0x%016llx] aclrtLaunchKernelWithHostArgs error[%d].",
        HCCL_ERROR_CODE(HCCL_E_RUNTIME), aclRet), HCCL_E_RUNTIME);
    if (GetExternalInputTaskExceptionEnable()) {
        HcclResult dfxRet = SaveAivDfxTaskInfo(opArgs);
        if (dfxRet != HCCL_SUCCESS) {
            HCCL_WARNING("[ExecuteKernelLaunchInner] SaveAivDfxTaskInfo failed, ret[%d].", dfxRet);
        }
    }
    return HCCL_SUCCESS;
}

// Kernel单次调用Launch外部接口
HcclResult ExecuteKernelLaunch(const AivOpArgs &opArgs)
{
    // Recording Logic
    if (g_recordingQueue) {
        AivInstruction ins;
        ins.opArgs = opArgs;
        // Calculate offsets
        // Note: opArgs.input is absolute address.
        if (opArgs.input >= g_baseInputAddr) {
             ins.inputOffset = opArgs.input - g_baseInputAddr;
        } else {
             ins.inputOffset = 0;
        }
        
        if (opArgs.output >= g_baseOutputAddr) {
             ins.outputOffset = opArgs.output - g_baseOutputAddr;
        } else {
             ins.outputOffset = 0;
        }
        g_recordingQueue->push_back(ins);

        if (g_recordOnlyMode) {
            return HCCL_SUCCESS;
        }
    }

    if (opArgs.cmdType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
        AivExtraKernelArgs aivExtraKernelArgs {
            opArgs.buffersIn, opArgs.input, opArgs.output,
            opArgs.rank, opArgs.sendRecvRemoteRank, opArgs.rankSize, opArgs.xRankSize, opArgs.yRankSize, opArgs.zRankSize, opArgs.count, opArgs.dataType, opArgs.op, opArgs.root, opArgs.sliceId,
            opArgs.inputSliceStride, opArgs.outputSliceStride, opArgs.repeatNum, opArgs.inputRepeatStride, opArgs.outputRepeatStride,
            opArgs.numBlocks, opArgs.isOpBase,
            reinterpret_cast<void*>(opArgs.counter.headCountMem),
            reinterpret_cast<void*>(opArgs.counter.tailCountMem), reinterpret_cast<void*>(opArgs.counter.addOneMem),
            opArgs.counter.memSize, &opArgs.extraArgs
        };
        CHK_RET(ExecuteKernelLaunchInner(opArgs, &aivExtraKernelArgs, sizeof(aivExtraKernelArgs)));
    } else {
        AivKernelArgs aivKernelArgs {
            opArgs.buffersIn, opArgs.input, opArgs.output,
            opArgs.rank, opArgs.sendRecvRemoteRank, opArgs.rankSize, opArgs.xRankSize, opArgs.yRankSize, opArgs.zRankSize, opArgs.count, opArgs.dataType, opArgs.op, opArgs.root, opArgs.sliceId,
            opArgs.inputSliceStride, opArgs.outputSliceStride, opArgs.repeatNum, opArgs.inputRepeatStride, opArgs.outputRepeatStride,
            opArgs.numBlocks, opArgs.isOpBase,
            reinterpret_cast<void*>(opArgs.counter.headCountMem),
            reinterpret_cast<void*>(opArgs.counter.tailCountMem), reinterpret_cast<void*>(opArgs.counter.addOneMem),
            opArgs.counter.memSize
        };
        CHK_RET(ExecuteKernelLaunchInner(opArgs, &aivKernelArgs, sizeof(aivKernelArgs)));
    }
    return HCCL_SUCCESS;
}

}   // ~~ namespace hccl
