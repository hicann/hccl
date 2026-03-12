
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
#include "hccl_ccu_res.h"
#include "ccu_assist_pub.h"
#include "ccu_temp_all_reduce_mesh_1D.h"
#include "kernel/ccu_kernel_all_reduce_mesh1d.h"
#include <set>

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
    
    kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
                             return std::make_unique<CcuKernelAllReduceMesh1D>(arg);
                         };
    std::vector<HcclChannelDesc> channelDescs;
    CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs);
    std::vector<uint64_t> dimSize;
    dimSize.emplace_back(subCommRanks_[0].size());
    kernelInfo.kernelArg = std::make_shared<CcuKernelArgAllReduceMesh1D>(dimSize,
                                                                        mySubCommRank_,
                                                                        param,
                                                                        subCommRanks_);
    kernelInfo.channels = channelDescs;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG("[CcuTempAllReduceMesh1D::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());
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
                                             const TemplateResource& templateResource)
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
    uint64_t token              = hcomm::CcuRep::GetTokenInfo(reinterpret_cast<uint64_t>(buffInfo_.inputPtr),
                                                       static_cast<uint64_t>(buffInfo_.inputSize));  

    uint64_t sliceSize = sliceInfoVec[myRank_][0].size;  // 获取本rank需要处理的数据量
    uint64_t offSet = sliceInfoVec[myRank_][0].offset;   // 自己需要 reduce 的数据基于 inputAddr 的偏移

    HCCL_INFO("[CcuTempAllReduceMesh1D] inputAddr[%llu], outputAddr[%llu], sliceSize[%llu], offSet[%llu]",
              inputAddr, outputAddr, sliceSize, offSet);
    std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgAllReduceMesh1D>(
                                        inputAddr, outputAddr, sliceSize, offSet, token);

    void* taskArgPtr = static_cast<void*>(taskArg.get());
    CHK_RET(HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[0], templateResource.ccuKernels[0], taskArgPtr));
    return HcclResult::HCCL_SUCCESS;
}
} // namespace Hccl
