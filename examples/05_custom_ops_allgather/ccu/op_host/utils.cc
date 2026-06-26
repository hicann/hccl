/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <vector>
#include "log.h"
#include "common.h"
#include "ccu_kernel.h"
#include "ccu_launch.h"
#include "hccl_ccu_res.h"

namespace ops_hccl_ag {
constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;

HcclResult GetDeviceType(DeviceType *deviceType) {
    const char *socNamePtr = aclrtGetSocName();
    if (socNamePtr == nullptr) {
        HCCL_ERROR("[GetDeviceType] Failed to get soc name");
        return HCCL_E_RUNTIME;
    }

    std::string socName(socNamePtr);
    if (socName.find("Ascend910B") != std::string::npos) {
        *deviceType = DEVICE_TYPE_A2;
        return HCCL_SUCCESS;
    }
    if (socName.find("Ascend910_93") != std::string::npos) {
        *deviceType = DEVICE_TYPE_A3;
        return HCCL_SUCCESS;
    }
    if (socName.find("Ascend950") != std::string::npos) {
        *deviceType = DEVICE_TYPE_A5;
        return HCCL_SUCCESS;
    }
    HCCL_ERROR("[GetDeviceType] Unsupported soc name: %s", socName.c_str());
    return HCCL_E_NOT_SUPPORT;
}

HcclResult GetThreadForCcu(HcclComm comm, const OpParam &param, AlgResourceCtxSerializable &resCtxHost) {
    // 只考虑threadNum = 1场景
    ThreadHandle thread;
    CHK_RET(HcclThreadAcquireWithStream(comm, param.engine, param.stream,
        resCtxHost.notifyNumOnMainThread, &thread)); // host模式下，将主流封装为thread，并创建主流上的notify
    resCtxHost.threads.push_back(thread);
    return HCCL_SUCCESS;
}

HcclResult GetChannelForCcu(HcclComm comm, const OpParam &param, std::vector<ChannelHandle> &kernelChannels) {
    uint32_t channelNum = param.rankSize - 1;
    kernelChannels.resize(channelNum);

    uint32_t channelIndex = 0;
    for(uint32_t remoteRank = 0; remoteRank < param.rankSize; remoteRank++) {
        if (remoteRank == param.myRank) {
            continue;
        }

        uint32_t netLayer = 0, listSize = 0;
        CommLink *linkList = nullptr;
        CHK_RET(HcclRankGraphGetLinks(comm, netLayer, param.myRank, remoteRank, &linkList,
                                      &listSize)); // 获取srcRank和dstRank间link信息

        HcclChannelDesc desc;
        CHK_RET(HcclChannelDescInit(&desc, 1));
        CommProtocol protocol = CommProtocol::COMM_PROTOCOL_UBC_CTP;
        bool protocolExists = false;
        for (uint32_t idx = 0; idx < listSize; idx++) {
            CommLink link = linkList[idx];
            if (link.linkAttr.linkProtocol == protocol) {
                desc.remoteRank = remoteRank;
                desc.notifyNum = CHANNEL_NOTIFY_NUM;
                desc.channelProtocol = link.linkAttr.linkProtocol;
                desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
                desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
                desc.localEndpoint.loc = link.srcEndpointDesc.loc;
                desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
                desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
                desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
                protocolExists = true;
                break;
            }
        }
        if (!protocolExists) {
            HCCL_ERROR("[GetChannelForCcu] Protocol %d not found between rank %u and rank %u",
                protocol, param.myRank, remoteRank);
            return HCCL_E_NOT_FOUND;
        }
        CHK_RET(HcclChannelAcquire(comm, param.engine, &desc, 1, &kernelChannels[channelIndex])); // 获取channelhandle
        channelIndex++;
    }

    return HCCL_SUCCESS;
}

HcclResult GetCcuKernel(HcclComm comm, const OpParam &param, AlgResourceCtxSerializable &resCtxHost, 
                        const std::vector<ChannelHandle> &kernelChannels, CcuKernelInfo &kernelInfo) {
    
    // 设置kernel函数名和函数指针
    strcpy_s(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuAllGatherMesh1DMem2MemKernel");
    kernelInfo.kernelFunc = reinterpret_cast<void *>(CcuAllGatherMesh1DMem2MemKernel);

    auto kernelArg = std::make_shared<CcuKernelArgAllGatherMesh1DMem2Mem>();
    kernelArg->rankSize = param.rankSize;
    kernelArg->rankId = param.myRank;
    kernelInfo.setKernelArg(kernelArg);

    auto* kernelArgBase = static_cast<CcuKernelArgBase*>(kernelInfo.kernelArg);
    if (!kernelArgBase) {
        HCCL_ERROR("[GetCcuKernel] kernelArg ptr is err.");
        return HCCL_E_INTERNAL;
    }

    for (uint32_t i = 0; i < kernelChannels.size(); ++i) {
        kernelArgBase->channels[i] = kernelChannels[i];  // 将channelhandle保存在kernelArgBase中
    }
    kernelArgBase->channelCount = static_cast<uint32_t>(kernelChannels.size());

    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1,
        HCCL_ERROR("[GetCcuKernel] HcclCommQueryCcuIns fail! insNum is [%u]", insNum),
        HCCL_E_INTERNAL);

    resCtxHost.ccuKernels.resize(1); // 只注册1个kernel

    CcuResult regStartRet = HcommCcuKernelRegisterStart(insHandle);
    if (regStartRet != CCU_SUCCESS) {
        HCCL_ERROR("ccu kernel register start failed: ccuRet -> %d", regStartRet);
        return ConvertCcuToHccl(regStartRet);
    }

    CcuKernelHandle kernelHandle;
    const void *kernelArgs[] = {kernelInfo.kernelArg};

    constexpr uint32_t dieId = 0; // 预留接口，暂无含义
    constexpr uint32_t kernelArgNum = 1;
    CcuResult regRet = HcommCcuKernelRegister(insHandle, dieId, kernelInfo.kernelFuncName,
                                                reinterpret_cast<void*>(kernelInfo.kernelFunc),
                                                kernelArgs, kernelArgNum, &kernelHandle); // 注册kernel

    if (regRet != CCU_SUCCESS) {
        HCCL_ERROR("ccu kernel register failed: ccuRet -> %d", regRet);
        return ConvertCcuToHccl(regRet);
    }
    resCtxHost.ccuKernels[0] = kernelHandle;

    CcuResult regEndRet = HcommCcuKernelRegisterEnd(insHandle);
    if (regEndRet != CCU_SUCCESS) {
        HCCL_ERROR("ccu kernel register start failed: ccuRet -> %d", regEndRet);
        return ConvertCcuToHccl(regEndRet);
    }
    resCtxHost.ccuKernelNum = {1};
    
    return HCCL_SUCCESS;
}

HcclResult AllocAlgResource(HcclComm comm, const OpParam &param, AlgResourceCtxSerializable &resCtxHost) {
    HCCL_INFO("Start to execute AllocAlgResourceCCU.");
    void *cclBufferAddr;
    uint64_t cclBufferSize;
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize)); // 从通信域获取CCL buffer
    resCtxHost.cclMem = CommBuffer{cclBufferAddr, cclBufferSize};
    uint32_t threadNum = 1; // 需要一条流
    resCtxHost.notifyNumOnMainThread = 0;

    CHK_RET(GetThreadForCcu(comm, param, resCtxHost)); // 申请流资源

    std::vector<ChannelHandle> kernelChannels;
    CHK_RET(GetChannelForCcu(comm, param, kernelChannels)); // 申请channel资源
    
    CcuKernelInfo kernelInfo;
    CHK_RET(GetCcuKernel(comm, param, resCtxHost, kernelChannels, kernelInfo)); // 注册kernel

    HCCL_INFO("End to execute AllocAlgResourceCCU success.");
    return HCCL_SUCCESS;
}

constexpr uint64_t SetBits(uint16_t start, uint16_t end)
{
    return ((uint64_t(1) << (end - start + 1)) - uint64_t(1)) << start;
}

constexpr uint64_t SetBits(uint16_t end)
{
    return ((uint64_t(1) << (end + 1)) - uint64_t(1));
}

uint64_t GetMaxLoopIterNum()
{
    constexpr uint16_t loopNumBitNum = 12;
    return SetBits(loopNumBitNum);
}

uint64_t GetLoopParam(uint64_t loopCtxId, uint64_t gsaOffset, uint64_t loopIterNum)
{
    constexpr uint16_t ctxIdBitNum     = 8;
    constexpr uint16_t ctxIdShiftBit   = 45;
    constexpr uint16_t gsaBitNum       = 32;
    constexpr uint16_t gsaShiftBit     = 13;
    constexpr uint16_t loopNumBitNum   = 13;
    constexpr uint16_t loopNumShiftBit = 0;
    return ((loopCtxId & SetBits(ctxIdBitNum)) << ctxIdShiftBit) | ((gsaOffset & SetBits(gsaBitNum)) << gsaShiftBit)
           | ((loopIterNum & SetBits(loopNumBitNum)) << loopNumShiftBit);
}

uint64_t GetParallelParam(uint64_t repeatNum, uint64_t repeatLoopIndex, uint64_t totalLoopNum)
{
    constexpr uint16_t repeatBitNum       = 7;
    constexpr uint16_t repeatNumShiftBit  = 55;
    constexpr uint16_t repeatLoopBitNum   = 7;
    constexpr uint16_t repeatLoopShiftBit = 48;
    constexpr uint16_t totalLoopBitNum    = 7;
    constexpr uint16_t totalLoopShiftBit  = 41;
    return ((repeatNum & SetBits(repeatBitNum)) << repeatNumShiftBit)
           | ((repeatLoopIndex & SetBits(repeatLoopBitNum)) << repeatLoopShiftBit)
           | ((totalLoopNum & SetBits(totalLoopBitNum)) << totalLoopShiftBit);
}

uint64_t GetOffsetParam(uint64_t gsaOffset, uint64_t msOffset, uint64_t ckeOffset)
{
    constexpr uint16_t gsaBitNum   = 32;
    constexpr uint16_t gsaShiftBit = 21;
    constexpr uint16_t msBitNum    = 11;
    constexpr uint16_t msShiftBit  = 10;
    constexpr uint16_t ckeBitNum   = 10;
    constexpr uint16_t ckeShiftBit = 0;
    return ((gsaOffset & SetBits(gsaBitNum)) << gsaShiftBit) | ((msOffset & SetBits(msBitNum)) << msShiftBit)
           | ((ckeOffset & SetBits(ckeBitNum)) << ckeShiftBit);
}

std::vector<uint64_t> CalGoSize(uint64_t size, const LoopGroupConfig &config)
{
    uint64_t loopSize = config.loopCount * config.memSlice;
    uint64_t maxSize  = loopSize * (GetMaxLoopIterNum() + 1);

    uint64_t m = size / loopSize;
    uint64_t n = (size - m * loopSize) / config.memSlice;
    uint64_t p = size - m * loopSize - n * config.memSlice;

    if (size == maxSize) {
        m = GetMaxLoopIterNum();
        n = config.loopCount - 1;
        p = config.memSlice;
    }

    uint64_t offset      = config.memSlice * config.loopCount * m;
    uint64_t loopIterNum = m;

    uint64_t loopExtendNum = 0;
    uint64_t tailSize      = 0;
    uint64_t LoopNumTwo    = 2;

    if (n == 0 && p == 0) {
        loopExtendNum = 0;
        tailSize      = 0;
    } else if (n != 0 && p == 0) {
        loopExtendNum = GetParallelParam(n - 1, 0, 1);
        tailSize      = config.memSlice;
    } else if (n == 0 && p != 0) {
        loopExtendNum = GetParallelParam(0, 0, 1);
        tailSize      = p;
    } else {
        loopExtendNum = GetParallelParam(n - 1, 1, LoopNumTwo);
        tailSize      = p;
    }
    
    return {offset, loopIterNum, loopExtendNum, tailSize};
}

} // namespace ops_hccl_ag