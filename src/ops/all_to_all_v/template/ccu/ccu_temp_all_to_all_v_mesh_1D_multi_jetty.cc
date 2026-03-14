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
#include "ccu_kernel_all_to_all_v_mesh1d_multi_jetty.h"
#include "ccu_temp_all_to_all_v_mesh_1D_multi_jetty.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {
constexpr uint32_t CONST_1 = 1;
constexpr uint32_t CONST_4 = 4;

CcuTempAllToAllVMesh1DMultiJetty::CcuTempAllToAllVMesh1DMultiJetty(const OpParam& param, const u32 rankId,
                                       const std::vector<std::vector<u32>> &subCommRanks)
: CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    myRank_ = rankId;
    templateRankSize_ = subCommRanks[0].size();
}

CcuTempAllToAllVMesh1DMultiJetty::~CcuTempAllToAllVMesh1DMultiJetty()
{
}

HcclResult CcuTempAllToAllVMesh1DMultiJetty::CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                                                      AlgResourceRequest& resourceRequest)
{
    // 不需要从流
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.slaveThreadNum = 0;
    // 多少个kernel
    resourceRequest.ccuKernelNum.push_back(1);
    HCCL_DEBUG("[CcuTempAllToAllVMesh1DMultiJetty::CalcRes] notifyNumOnMainThread[%u] slaveThreadNum[%u]",
               resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    // 创建每个kernel的ctxArg，放入kernelInfo, 然后将kernelinfo放入resourceRequest.ccuKernelInfos
    CcuKernelInfo kernelInfo;
    
    kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
                             return std::make_unique<CcuKernelAllToAllVMesh1DMultiJetty>(arg);
                         };
    std::vector<HcclChannelDesc> channelDescs;
    // 计算建链诉求以COMM_TOPO_1DMESH为优先级，优先建mesh链，没有mesh链建clos链
    CHK_RET(CalcChannelRequestMesh1DWithPriorityTopo(comm, param, topoInfo, subCommRanks_, channelDescs, CommTopo::COMM_TOPO_1DMESH));
    kernelInfo.channels = channelDescs;

    std::vector<uint32_t> jettyNums;
    CHK_RET(SetJettyNums(jettyNums, true));
    kernelInfo.kernelArg = std::make_shared<CcuKernelArgAllToAllVMesh1DMultiJetty>(subCommRanks_[0].size(),
                                                                                    myRank_,
                                                                                    param,
                                                                                    subCommRanks_,
                                                                                    jettyNums);
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);

    HCCL_DEBUG("[CcuTempAllToAllVMesh1DMultiJetty::CalcRes] channelDescs.size()=%llu, dimsize=%llu, "
               "ccuKernelInfos.size()=%llu",
               channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempAllToAllVMesh1DMultiJetty::KernelRun(const OpParam& param, const TemplateDataParams& templateDataParams,
                                                        const TemplateResource& templateResource)
{
    buffInfo_ = templateDataParams.buffInfo;

    uint64_t inputAddr          = PointerToAddr(buffInfo_.inputPtr) + buffInfo_.inBuffBaseOff;
    uint64_t outputAddr         = PointerToAddr(buffInfo_.outputPtr) + buffInfo_.outBuffBaseOff;
    uint64_t token              = hcomm::CcuRep::GetTokenInfo(PointerToAddr(buffInfo_.outputPtr), buffInfo_.outputSize);
    uint64_t srcOffset = 0; // alltoallv假设src起始地址为发送rank的对应块起始地址
    uint64_t dstOffset = 0; // alltoallv假设dst起始地址为接收rank的对应块起始地址

    std::unique_ptr<hcomm::CcuTaskArg> taskArg = std::make_unique<CcuTaskArgAllToAllVMesh1DMultiJetty>(
        inputAddr, outputAddr, token, srcOffset, dstOffset, localSendRecvInfo_);

    void* taskArgPtr = static_cast<void*>(taskArg.get());

    HcclCcuKernelLaunch(param.hcclComm, templateResource.threads[0], templateResource.ccuKernels[0], taskArgPtr);

    HCCL_DEBUG("[CcuTempAllToAllVMesh1DMultiJetty::KernelRun] end");

    return HcclResult::HCCL_SUCCESS;
}

// executor在orchestra中调用
void CcuTempAllToAllVMesh1DMultiJetty::SetA2ASendRecvInfo(const A2ASendRecvInfo &sendRecvInfo)
{
    localSendRecvInfo_ = sendRecvInfo;
}

HcclResult CcuTempAllToAllVMesh1DMultiJetty::SetJettyNums(std::vector<uint32_t>& jettyNums, const bool multijetty) const
{
    jettyNums.resize(templateRankSize_, 0);
    for (int i = 0; i < templateRankSize_; i++) {
        if (i == myRank_) {
            jettyNums[i] = CONST_1;
        } else if (multijetty) {
            jettyNums[i] = CONST_4;
        } else {
            jettyNums[i] = CONST_1;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}
} // namespace ops_hccl