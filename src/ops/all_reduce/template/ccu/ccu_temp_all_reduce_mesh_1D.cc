
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "channel.h"
#include "ccu_temp_all_reduce_mesh_1D.h"
#include "kernel/ccu_kernel_all_reduce_mesh1d.h"
#include <set>
#include "ccu_launch_dl.h"

namespace ops_hccl {

CcuTempAllReduceMesh1D::CcuTempAllReduceMesh1D(const OpParam& param, 
                                                const u32 rankId,
                                                const std::vector<std::vector<u32>> &subCommRanks)
: CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    // 获取本卡在子通信域(如果有)中的rankid
    auto it = std::find(subCommRanks[0].begin(), subCommRanks[0].end(), rankId);
    if (it != subCommRanks[0].end()) {
        mySubCommRank_ = std::distance(subCommRanks[0].begin(), it);
    }
    templateRankSize_ = subCommRanks[0].size();
    reduceOp_ = param.reduceType;
    dataType_ = param.DataDes.dataType;
    outputDataType_ =  param.DataDes.outputType;
    opType_ = param.opType;
}

CcuTempAllReduceMesh1D::~CcuTempAllReduceMesh1D()
{
}

uint64_t CcuTempAllReduceMesh1D::RoundUp(uint64_t dividend, uint64_t divisor) const
{
    return dividend / divisor + ((dividend % divisor != 0) ? 1 : 0);
}

HcclResult CcuTempAllReduceMesh1D::CalcSliceInfo(const u64 dataSize, RankSliceInfo &sliceInfoVec)
{
    CHK_PRT_RET(templateRankSize_ == 0, HCCL_ERROR("[CcuTempAllReduceMesh1D] templateRankSize_ is 0"),
        HcclResult::HCCL_E_INTERNAL);
    std::vector<SliceInfo> tmp(subCommRanks_.size());
    sliceInfoVec.resize(templateRankSize_, tmp);

    u64 unitAllignSize = DataTypeSizeGet(dataType_);
    u64 chunkSize      = RoundUp(dataSize, (templateRankSize_ * unitAllignSize)) * unitAllignSize;
    HCCL_INFO("chunkSize[%llu], dataSize[%llu], templateRankSize_[%u], unitAllignSize[%llu]", chunkSize, dataSize,
              templateRankSize_, unitAllignSize);
    u64 accumOff = 0;
    for (u32 rankIdx = 0; rankIdx < templateRankSize_; rankIdx++) {
        u64       currChunkSize  = ((dataSize - accumOff) > chunkSize) ? chunkSize : (dataSize - accumOff);
        SliceInfo slice          = {accumOff, currChunkSize};
        sliceInfoVec[rankIdx][0] = slice;
        accumOff += currChunkSize;
    }

    CHK_PRT_RET(
        (sliceInfoVec[templateRankSize_ - 1][0].offset + sliceInfoVec[templateRankSize_ - 1][0].size != dataSize),
        HCCL_ERROR(
            "[CcuTempAllReduceMesh1D] chunkSize:[%llu], Rank:[%d], SliceInfo calculation error!",
            chunkSize, myRank_),
        HcclResult::HCCL_E_INTERNAL);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceMesh1D::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo, AlgResourceRequest& resourceRequest)
{
    // 不需要从流
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;
    // 多少个kernel
    resourceRequest.ccuKernelNum.push_back(1);
    HCCL_DEBUG("[CcuTempAllReduceMesh1D::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    // 创建每个kernel的ctxArg，放入kernelInfo, 然后将kernelinfo放入resourceRequest.ccuKernelInfos
    CcuKernelInfo kernelInfo;
    strcpy_s(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuKernelAllReduceMesh1D");
 	kernelInfo.kernelFunc = reinterpret_cast<void *>(CcuAllReduceMesh1DKernel);
    
    std::vector<HcclChannelDesc> channelDescs;
    if(topoInfo->level0Topo != Level0Shape::MESH_1D_CLOS) {
        CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));
    } else {
        std::vector<HcclChannelDesc> myChannelDescs;
        CHK_RET(CalcChannelRequestMesh1DWithPriorityTopo(comm, param, topoInfo, subCommRanks_, myChannelDescs, CommTopo::COMM_TOPO_1DMESH));
        for(auto channel : myChannelDescs) {
            if(channel.channelProtocol == COMM_PROTOCOL_UBC_CTP) {
                channelDescs.push_back(channel);
            }
        }
        HCCL_DEBUG("[CcuTempAllReduceMesh1D::CalcRes] Get Mesh Channel Success!");
    } 
    auto kernelArg = std::make_shared<CcuKernelArgAllReduceMesh1D>();
    kernelArg->rankSize = templateRankSize_;
    kernelArg->rankId = mySubCommRank_;
    kernelArg->opParam = param;
    kernelArg->subCommRanks = subCommRanks_;
    kernelInfo.setKernelArg(kernelArg);
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG("[CcuTempAllReduceMesh1D::CalcRes] channelDescs.size()=%llu, templateRankSize_=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), templateRankSize_, resourceRequest.ccuKernelInfos.size());
    return HcclResult::HCCL_SUCCESS;
}

/* CCU数据类型校验规则
 * Reduce算子：
 *      高精度模式，当dataType==outputDataType时，可选类型为FP32、FP16、BF16、UINT8、INT16、INT32；
 *      低精度模式，当dataType!=outputDataType时，dataType可选范围HIF8、E4M3、E5M2、INT8；outputDataType可选范围FP32、FP16、BF16；
 * 非Reduce算子：任意数据类型，dataType==outputDataType即可。
 */
HcclResult CcuTempAllReduceMesh1D::CheckCcuDataType() const 
{
    if (dataType_ == outputDataType_) {
        // allreduce算子高精度模式
        HCCL_INFO("HIGH PRECISION");
        std::set<HcclDataType> highPrecisionSupportedInputDataType
            = {HcclDataType::HCCL_DATA_TYPE_FP32,  HcclDataType::HCCL_DATA_TYPE_FP16,  HcclDataType::HCCL_DATA_TYPE_BFP16,
               HcclDataType::HCCL_DATA_TYPE_UINT8, HcclDataType::HCCL_DATA_TYPE_INT16, HcclDataType::HCCL_DATA_TYPE_INT32};
        if (highPrecisionSupportedInputDataType.count(dataType_) == 0) {
            HCCL_ERROR("Unsupported HcclDataType [%d] For OpType [%d].",
                dataType_, opType_);
            return HcclResult::HCCL_E_PARA;
        }
    } else if (outputDataType_ != HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        // allreduce算子低精度模式
        HCCL_INFO("LOW PRECISION");
        HCCL_ERROR("Unsupported LOW PRECISION, Output HcclDataType [%d] For OpType [%d].",
            outputDataType_, opType_);
        return HcclResult::HCCL_E_PARA;
    }
    HCCL_INFO("CheckCcuDataType Success!");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceMesh1D::KernelRun(const OpParam& param, const TemplateDataParams& templateDataParams,
                                             TemplateResource& templateResource)
{
    if (outputDataType_ == HcclDataType::HCCL_DATA_TYPE_RESERVED) {
        outputDataType_ = dataType_;
    }
    CHK_RET(CheckCcuDataType());
    buffInfo_ = templateDataParams.buffInfo;
    RankSliceInfo sliceInfoVec;
    CHK_RET(CalcSliceInfo(templateDataParams.sliceSize, sliceInfoVec));

    uint64_t inputAddr          = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr         = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token;
    CHK_RET(GetToken(buffInfo_, token));

    uint64_t sliceSize = sliceInfoVec[myRank_][0].size;  // 获取本rank需要处理的数据量
    uint64_t offSet = sliceInfoVec[myRank_][0].offset;   // 自己需要 reduce 的数据基于 inputAddr 的偏移
    LoopGroupConfig config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount    = CCU_MS_DEFAULT_LOOP_COUNT;
    config.memSlice     = CCU_MS_SIZE;
    auto goSize         = CalGoSize(sliceSize, config);

    std::vector<uint64_t> taskArgs = {inputAddr, outputAddr,token, offSet, goSize[0], goSize[1], goSize[2], goSize[3]};
    uint64_t argSize = 8;

    HCCL_INFO("[CcuTempAllReduceMesh1D] inputAddr[%llu], outputAddr[%llu], sliceSize[%llu], offSet[%llu]",
              inputAddr, outputAddr, sliceSize, offSet);

    CcuResult launchRet =  HcommCcuKernelLaunch(templateResource.threads[0], templateResource.ccuKernels[0], taskArgs.data(), argSize);
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[CcuTempAllReduceMesh1D::KernelRun] kernel launch failed, ccuRet -> %d", launchRet);
        return ConvertCcuToHccl(launchRet);
    }
    CcuKernelSubmitInfo submitInfo;
    submitInfo.kernelHandle = templateResource.ccuKernels[0];
    CHK_RET(FillCachedArgs(submitInfo, inputAddr, outputAddr, token, offSet, 
            goSize[0], goSize[1], goSize[2], goSize[3], 
            buffInfo_.inBuffBaseOff, buffInfo_.outBuffBaseOff));
    templateResource.submitInfos.push_back(submitInfo);
    
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllReduceMesh1D::FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    if (tempFastLaunchCtx.ccuKernelSubmitInfos.size() == 0) {
        HCCL_INFO("[CcuTempAllReduceMesh1D::FastLaunch] ccu kernel num is 0, just success.");
        return HCCL_SUCCESS;
    }
    HCCL_DEBUG("[CcuTempAllReduceMesh1D::FastLaunch] start");
    uint64_t *args = const_cast<uint64_t*>(tempFastLaunchCtx.ccuKernelSubmitInfos[0].cachedArgs);
    constexpr u32 inputIdx = 0;
    constexpr u32 outputIdx = 1;
    constexpr u32 inputOffsetIdx = 8;
    constexpr u32 outputOffsetIdx = 9;
    uint64_t argSize = 8;

    args[inputIdx] = PointerToAddr(tempFastLaunchCtx.buffInfo.inputPtr) + args[inputOffsetIdx];
    args[outputIdx] = PointerToAddr(tempFastLaunchCtx.buffInfo.outputPtr) + args[outputOffsetIdx];

    void *taskArgs = reinterpret_cast<void*>(args);
    CcuResult launchRet = HcommCcuKernelLaunch(tempFastLaunchCtx.threads[0],
                                               tempFastLaunchCtx.ccuKernelSubmitInfos[0].kernelHandle,
                                               taskArgs, argSize);
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[CcuTempAllReduceMesh1D::FastLaunch] kernel launch failed, ccuRet -> %d", launchRet);
        return ConvertCcuToHccl(launchRet);
    }

    HCCL_DEBUG("[CcuTempAllReduceMesh1D::FastLaunch] end");
    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempAllReduceMesh1D::GetThreadNum() const
{
    return 1;
}
} // namespace Hccl
