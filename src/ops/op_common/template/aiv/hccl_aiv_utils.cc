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
    double execTimeOut = 0;
    if (!GetExternalInputExecTimeout(execTimeOut)) {
        return CUSTOM_TIMEOUT * TIME_S_TO_US;
    }
    
    u32 timeout = CUSTOM_TIMEOUT * TIME_S_TO_US;
    double timeoutUs = execTimeOut * TIME_S_TO_US;
    if (timeoutUs > static_cast<double>(std::numeric_limits<s32>::max())) {
        HCCL_WARNING("[GetAivTimeout]Get input timeout[%.2f] is out of valid range.", timeoutUs);
        return CUSTOM_TIMEOUT * TIME_S_TO_US;
    }

    u32 timeoutUsInt = static_cast<u32>(timeoutUs);
    if (timeoutUsInt == 0) {
        timeoutUsInt = CUSTOM_TIMEOUT * TIME_S_TO_US;
    }

    u64 minNpuSchedTimeout = 0;
    u64 maxNpuSchedTimeout = 0;
    if (GetMinAndMaxNpuSchedTimeOut(minNpuSchedTimeout, maxNpuSchedTimeout) != HCCL_SUCCESS) {
        HCCL_WARNING("[GetAivTimeout] get npu sched timeout range failed, use default[%u]us.", CUSTOM_TIMEOUT * TIME_S_TO_US);
        return CUSTOM_TIMEOUT * TIME_S_TO_US;
    }

    timeout = (timeoutUsInt < minNpuSchedTimeout) ? minNpuSchedTimeout
                : (timeoutUsInt > maxNpuSchedTimeout) ? maxNpuSchedTimeout
                : timeoutUsInt;
    HCCL_INFO("[GetAivTimeout]timeout[%u]us, execTimeOut[%.2f]s, minNpuSchedTimeout[%u]us, maxNpuSchedTimeout[%u]us.",
        timeout, execTimeOut, minNpuSchedTimeout, maxNpuSchedTimeout);

    return timeout;
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
    bool isOpBase;
    const void* headCountMem;
    const void* tailCountMem;
    const void* addOneMem;
    u32 counterMemSize;
    bool isEnableCounter;

    AivKernelArgsDef(const void* buffIn, u64 input, u64 output, u32 rank, u32 sendRecvRemoteRank,
        u32 rankSize, u64 xRankSize, u64 yRankSize, u64 zRankSize,
        u64 len, u32 dataType, u32 reduceOp, u32 root, u32 tag,
        u64 inputSliceStride, u64 outputSliceStride, u64 repeatNum, u64 inputRepeatStride, u64 outputRepeatStride,
        bool isOpBase = true,
        const void* headCountMem = nullptr, const void* tailCountMem = nullptr, const void* addOneMem = nullptr,
        u32 counterMemSize = 0, bool isEnableCounter = false)
        : buffersIn(buffIn),input(input), output(output), rank(rank), sendRecvRemoteRank(sendRecvRemoteRank), rankSize(rankSize), xRankSize(xRankSize), yRankSize(yRankSize), zRankSize(zRankSize),
        len(len) ,dataType(dataType),
        reduceOp(reduceOp), root(root), tag(tag),
        inputSliceStride(inputSliceStride), outputSliceStride(outputSliceStride), repeatNum(repeatNum), inputRepeatStride(inputRepeatStride), outputRepeatStride(outputRepeatStride),
        isOpBase(isOpBase),
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
        u64 inputSliceStride, u64 outputSliceStride, u64 repeatNum, u64 inputRepeatStride, u64 outputRepeatStride,
        bool isOpBase = true,
        const void* headCountMem = nullptr, const void* tailCountMem = nullptr, const void* addOneMem = nullptr,
        u32 counterMemSize = 0, const ExtraArgs* extraArgsPtr = nullptr)
        : buffersIn(buffIn),input(input), output(output), rank(rank), sendRecvRemoteRank(sendRecvRemoteRank), rankSize(rankSize), xRankSize(xRankSize), yRankSize(yRankSize), zRankSize(zRankSize),
        len(len) ,dataType(dataType),
        reduceOp(reduceOp), root(root), tag(tag),
        inputSliceStride(inputSliceStride), outputSliceStride(outputSliceStride), repeatNum(repeatNum), inputRepeatStride(inputRepeatStride), outputRepeatStride(outputRepeatStride),
        isOpBase(isOpBase),
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

// KernelLaunch内部接口
HcclResult ExecuteKernelLaunchInner(const AivOpArgs &opArgs, void* args, u32 argsSize)
{
    AivLaunchGuard launchGuard;
    HcclResult guardRet = launchGuard.Acquire();
    CHK_PRT_RET(guardRet != HCCL_SUCCESS, HCCL_ERROR("[ExecuteKernelLaunchInner] aiv kernel is unregistering."),
        HCCL_E_RUNTIME);

    HCCL_INFO("[ExecuteKernelLaunchInner] sendbuff [%p] recvbuff [%p] rank [%u] sendRecvRemoteRank [%u] rankSize [%u] count [%llu] "
        "dataType [%d] reduceOp [%d] root [%u] tag [%u] isOpBase [%d] "
        "extraArgsPtr [%p] argsSize [%u]", opArgs.input,
        opArgs.output, opArgs.rank, opArgs.sendRecvRemoteRank, opArgs.rankSize, opArgs.count,
        opArgs.dataType, opArgs.op, opArgs.root,
        opArgs.sliceId, opArgs.isOpBase, args, argsSize);

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
        aclRet = aclrtBinaryGetFunction(kernelLookupResult.entry.binHandle,
            kernelLookupResult.entry.kernelName.c_str(), &funcHandle);
        CHK_PRT_RET(aclRet != ACL_SUCCESS, HCCL_ERROR("[ExecuteKernelLaunchInner] retry get function failed, error[%d]", aclRet), HCCL_E_RUNTIME);
        ret = UpdateKernelFunc(kernelLookupResult.deviceId, funcKey, funcHandle);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ExecuteKernelLaunchInner] update function handle failed, ret[%d]", ret), HCCL_E_RUNTIME);
        aclRet = aclrtLaunchKernelWithHostArgs(funcHandle, opArgs.numBlocks, opArgs.stream,
            &cfg, args, argsSize, nullptr, 0);
    }
    CHK_PRT_RET(aclRet != ACL_SUCCESS, HCCL_ERROR("[ExecuteKernelLaunchInner]errNo[0x%016llx] aclrtLaunchKernelWithHostArgs error[%d].",
        HCCL_ERROR_CODE(HCCL_E_RUNTIME), aclRet), HCCL_E_RUNTIME);
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
            opArgs.isOpBase,
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
            opArgs.isOpBase,
            reinterpret_cast<void*>(opArgs.counter.headCountMem),
            reinterpret_cast<void*>(opArgs.counter.tailCountMem), reinterpret_cast<void*>(opArgs.counter.addOneMem),
            opArgs.counter.memSize
        };
        CHK_RET(ExecuteKernelLaunchInner(opArgs, &aivKernelArgs, sizeof(aivKernelArgs)));
    }
    return HCCL_SUCCESS;
}

}   // ~~ namespace hccl