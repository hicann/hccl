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
#include <vector>
#include <iostream>
#include <fstream>
#include "mmpa_api.h"
#include "adapter_acl.h"
#include "hccl_aiv_utils.h"
#include "universal_concurrent_map.h"

using namespace std;
using namespace ops_hccl;

namespace ops_hccl {
constexpr u32 SIG_MOVE_LEFT_BITS = 20;

constexpr u32 AIV_BUFFER_PING_PONG_FACTOR = 2;

constexpr u32 MAX_BIN_FILE_SIZE = 100 * 1024 * 1024; // 最大读取100m的bin file到string中

constexpr s32 RESET_TAIL_SYNC_TAG = 2;

static bool g_init = false;
static mutex g_mut;
static aclrtBinHandle g_binHandle;
static std::unordered_map<s8*, aclrtFuncHandle> g_aivFuncMap;

using AivExtraKernelArgs = struct AivExtraKernelArgsDef {
    const void* buffersIn; // 注册的CCLIN地址，所有卡可访问
    u64 input;
    u64 output;
    u32 rank;
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

    AivExtraKernelArgsDef(const void* buffIn, u64 input, u64 output, u32 rank,
        u32 rankSize, u64 xRankSize, u64 yRankSize, u64 zRankSize,
        u64 len, u32 dataType, u32 reduceOp, u32 root, u32 tag,
        u64 inputSliceStride, u64 outputSliceStride, u64 repeatNum, u64 inputRepeatStride, u64 outputRepeatStride,
        bool isOpBase = true,
        const void* headCountMem = nullptr, const void* tailCountMem = nullptr, const void* addOneMem = nullptr,
        u32 counterMemSize = 0, const ExtraArgs* extraArgsPtr = nullptr)
        : buffersIn(buffIn),input(input), output(output), rank(rank), rankSize(rankSize), xRankSize(xRankSize), yRankSize(yRankSize), zRankSize(zRankSize),
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
    char *getPath = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ASCEND_HOME_PATH, getPath);
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
        (((static_cast<s64>(cmdType) << SIG_MOVE_LEFT_BITS) + static_cast<s64>(dataType)) << SIG_MOVE_LEFT_BITS) +
        static_cast<s64>(argsType));
}

HcclResult RegisterBinaryKernel(const char* funcName, const aclrtBinHandle binHandle, s8* funcKey)
{
    if (funcKey == nullptr) {
        return HCCL_E_PARA;
    }

    aclrtFuncHandle funcHandle;
    aclError aclRet = aclrtBinaryGetFunction(binHandle, funcName, &funcHandle);
    CHK_PRT_RET(aclRet != ACL_SUCCESS, HCCL_ERROR("[RegisterBinaryKernel]errNo[0x%016llx] get function from binary error.", aclRet),
        HCCL_E_NOT_FOUND);

    g_aivFuncMap[funcKey] = funcHandle;

    return HCCL_SUCCESS;
}

HcclResult GetKernelFunc(aclrtFuncHandle& funcHandle, s8* funcKey)
{
    if (funcKey == nullptr || g_aivFuncMap.find(funcKey) == g_aivFuncMap.end()) {
        return HCCL_E_PARA;
    }
    funcHandle = g_aivFuncMap[funcKey];
    return HCCL_SUCCESS;
}

// Kernel注册入口，全局只需要初始化一次
HcclResult RegisterKernel(HcclCMDType cmdType, const std::string &aivBinaryName, const std::vector<AivKernelInfo> &aivKernelInfoList)
{
    lock_guard<mutex> guard(g_mut);
    if (g_init) {
        return HCCL_SUCCESS;
    }

    HcclResult ret;
    string binFilePath;
    ret = GetAivOpBinaryPath(aivBinaryName, binFilePath);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AIV][RegisterKernel] get aiv op binary path failed"), HCCL_E_RUNTIME);

    ret = LoadBinaryFromFile(binFilePath.c_str(), ACL_RT_BINARY_LOAD_OPT_LAZY_LOAD, 1, g_binHandle);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AIV][RegisterKernel] read aiv kernel bin file failed"),
        HCCL_E_RUNTIME);

    for (auto &aivKernelInfo: aivKernelInfoList) {
        ret = RegisterBinaryKernel(aivKernelInfo.kernelName, g_binHandle,
            GetFuncKey(cmdType, aivKernelInfo.dataType, aivKernelInfo.argsType));
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[AIV][RegisterKernel] register binary kernel for kernelName[%s] "
            "cmdType[%d] dataType[%s] argsType[%d] failed", aivKernelInfo.kernelName, cmdType,
            GetDataTypeEnumStr(aivKernelInfo.dataType).c_str(), aivKernelInfo.argsType), HCCL_E_RUNTIME);
    }

    g_init = true;

    return HCCL_SUCCESS;
}

HcclResult UnRegisterAivKernel()
{
    lock_guard<mutex> guard(g_mut);
    if (g_init) {
        ACLCHECK(aclrtBinaryUnLoad(g_binHandle));
        g_aivFuncMap.clear();

        g_init = false;
    }

    return HCCL_SUCCESS;
}

// commTag需要拼接单算子或者图模式
HcclResult GetAivCountTag(const std::string &commTag, u32 rank, s32 &aivCountTag)
{
    static UniversalConcurrentMap<std::string, AivCountTagArray> aivCountTagMap;
    auto builder = []() -> AivCountTagArray {
        AivCountTagArray array;
        array.fill(0);
        return array;
    };

    std::pair<UniversalConcurrentMap<std::string, AivCountTagArray>::Iterator, bool> tagIt;
    EXECEPTION_CATCH(tagIt = aivCountTagMap.EmplaceIfNotExist(commTag, builder), return HCCL_E_INTERNAL);
    if (tagIt.second) {
        HCCL_INFO("[GetAivCountTag]aivCountTagMap new insert, commTag[%s]", commTag.c_str());
    }
    AivCountTagArray &aivCountTags = aivCountTagMap[commTag];
    aivCountTags[rank]++;
    if (aivCountTags[rank] > TAG_RESET_COUNT) {
        aivCountTags[rank] = TAG_INIT_VALUE;
    }
    aivCountTag = aivCountTags[rank];
    HCCL_INFO("[GetAivCountTag]aivCountTag[%u]", aivCountTag);
    return HCCL_SUCCESS;
}

// 满足isAivClearEnable条件时，在KernelLaunch前调用，清零标记区
HcclResult ClearAivSyncBuf(const OpParam &param, AlgResourceCtxSerializable& resCtx)
{
    ACLCHECK(aclrtMemcpy(static_cast<u8*>(resCtx.aivCommInfoPtr) + AIV_FLAG_ADDR_OFFSET, AIV_FLAG_AREA_SIZE,
        static_cast<u8*>(resCtx.aivCommInfoPtr) + AIV_FLAG_CLEAR_OFFSET, AIV_FLAG_AREA_SIZE, ACL_MEMCPY_DEVICE_TO_DEVICE));
    HCCL_INFO("[ClearAivSyncBuf] clearaiv done.");
    return HCCL_SUCCESS;
}

// KernelLaunch内部接口
HcclResult ExecuteKernelLaunchInner(const AivOpArgs &opArgs, void* args, u32 argsSize)
{
    HCCL_INFO("[ExecuteKernelLaunchInner] sendbuff [%p] recvbuff [%p] rank [%u] rankSize [%u] count [%llu] "
        "dataType [%d] reduceOp [%d] root [%u] tag [%u] isOpBase [%d] "
        "extraArgsPtr [%p] argsSize [%u]", opArgs.input,
        opArgs.output, opArgs.rank, opArgs.rankSize, opArgs.count,
        opArgs.dataType, opArgs.op, opArgs.root,
        opArgs.aivCountTag, opArgs.isOpBase, args, argsSize);

    aclrtLaunchKernelCfg cfg;
    aclrtLaunchKernelAttr attr[AIV_ATTRNUM_THREE];
    attr[0].id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
    attr[0].value.schemMode = 1;
    attr[1].id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT_US;
    attr[1].value.timeoutUs.timeoutLow = CUSTOM_TIMEOUT * TIME_S_TO_US;
    attr[1].value.timeoutUs.timeoutHigh = 0;
    attr[2].id = ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE;
    attr[2].value.engineType = ACL_RT_ENGINE_TYPE_AIV;
    cfg.numAttrs = AIV_ATTRNUM_THREE;
    cfg.attrs = attr;

    aclrtFuncHandle funcHandle;
    HcclResult ret = GetKernelFunc(funcHandle, GetFuncKey(opArgs.cmdType, opArgs.dataType, opArgs.argsType));
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ExecuteKernelLaunchInner] errNo[0x%016llx] GetKernelFunc failed, "
        "return[%d]", HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret), HCCL_E_RUNTIME);

    aclError aclRet = aclrtLaunchKernelWithHostArgs(funcHandle, opArgs.numBlocks, opArgs.stream,
        &cfg, args, argsSize, nullptr, 0);
    CHK_PRT_RET(aclRet != ACL_SUCCESS, HCCL_ERROR("[ExecuteKernelLaunchInner]errNo[0x%016llx] aclrtLaunchKernelWithHostArgs error[%d].",
        HCCL_ERROR_CODE(HCCL_E_RUNTIME), aclRet), HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
}

// Kernel单次调用Launch外部接口
HcclResult ExecuteKernelLaunch(const AivOpArgs &opArgs)
{
    AivExtraKernelArgs aivExtraKernelArgs {
        opArgs.buffersIn, opArgs.input, opArgs.output,
        opArgs.rank, opArgs.rankSize, opArgs.xRankSize, opArgs.yRankSize, opArgs.zRankSize, opArgs.count, opArgs.dataType, opArgs.op, opArgs.root, opArgs.aivCountTag,
        opArgs.inputSliceStride, opArgs.outputSliceStride, opArgs.repeatNum, opArgs.inputRepeatStride, opArgs.outputRepeatStride,
        opArgs.isOpBase,
        reinterpret_cast<void*>(opArgs.counter.headCountMem),
        reinterpret_cast<void*>(opArgs.counter.tailCountMem), reinterpret_cast<void*>(opArgs.counter.addOneMem),
        opArgs.counter.memSize, &opArgs.extraArgs
    };
    CHK_RET(ExecuteKernelLaunchInner(opArgs, &aivExtraKernelArgs, sizeof(aivExtraKernelArgs)));
    return HCCL_SUCCESS;
}

}   // ~~ namespace hccl