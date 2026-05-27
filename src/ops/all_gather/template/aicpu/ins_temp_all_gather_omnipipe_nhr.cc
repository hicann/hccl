/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_temp_all_gather_omnipipe_nhr.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {
InsTempAllGatherOmniPipeNHR::InsTempAllGatherOmniPipeNHR(const OpParam& param,
                                                         const u32 rankId,  // 传通信域的rankId，userRank
                                                         const std::vector<std::vector<u32>>& subCommRanks)
    : InsTempAllGatherNHR(param, rankId, subCommRanks)
{
}

InsTempAllGatherOmniPipeNHR::~InsTempAllGatherOmniPipeNHR()
{
}

HcclResult InsTempAllGatherOmniPipeNHR::KernelRun(const OpParam& param, const TemplateDataParams& tempAlgParams,
                                                TemplateResource& templateResource)
{
    HCCL_INFO("[InsTempAllGatherNHR] Run start");
    if (templateRankSize_ == 1) {
        HCCL_INFO("[InsTempAllGatherNHR] Rank [%d], template ranksize is 1.", myRank_);
        return HcclResult::HCCL_SUCCESS;
    }
    threadNum_ = 1;
    tempAlgParams_ = tempAlgParams;
    CHK_PRT_RET(threadNum_ != templateResource.threads.size(),
                HCCL_ERROR("[InsTempAllGatherNHR] Rank [%d], requiredQueNum [%u] not equals templateQueNum [%zu].",
                           myRank_, threadNum_, templateResource.threads.size()),
                HcclResult::HCCL_E_INTERNAL);

    CHK_RET(RunAllGatherNHR(templateResource.threads, templateResource.channels));

    HCCL_INFO("[InsTempAllGatherNHR] Run End");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult InsTempAllGatherOmniPipeNHR::RunAllGatherNHR(const std::vector<ThreadHandle>& threads,
                                                        const std::map<u32, std::vector<ChannelInfo>>& channels)
{
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    const u32 nSteps = GetNHRStepNum(templateRankSize_);  // NHR 通信步数， celi(log2(rankSize))
    bool isPcieProtocal = IsPcieProtocol(channels);  // 判断是否存在pcie链路

    for (u32 step = 0; step < nSteps; ++step) {
        AicpuNHRStepInfo stepInfo;
        CHK_RET(GetStepInfo(step, nSteps, stepInfo));  // 计算当前step要通信的卡，数据

        const ChannelInfo& channelRecv = channels.at(GetRankFromMap(stepInfo.fromRank))[0];
        const ChannelInfo& channelSend = channels.at(GetRankFromMap(stepInfo.toRank))[0];
        // 构造SendRecv， 都是Scratch到Scratch的传输，没有DMA消减
        std::vector<DataSlice> txSrcSlices;
        std::vector<DataSlice> txDstSlices;
        std::vector<DataSlice> rxSrcSlices;
        std::vector<DataSlice> rxDstSlices;

        void* sendCclBuffAddr = channelSend.remoteCclMem.addr;
        void* recvCclBuffAddr = channelRecv.remoteCclMem.addr;

        HCCL_DEBUG(
            "[InsTempAllGatherNHR] rank[%d] rankSize[%u] recvFrom[%u] sendTo[%u] step[%u] nSteps[%u] nSlices[%u]",
            myRank_, templateRankSize_, stepInfo.fromRank, stepInfo.toRank, step, nSteps, stepInfo.nSlices);

        for (u32 i = 0; i < stepInfo.nSlices; ++i) {
            const u32 txIdx = stepInfo.txSliceIdxs[i];
            const u32 rxIdx = stepInfo.rxSliceIdxs[i];
            for (u32 rpt = 0; rpt < tempAlgParams_.stepSliceInfo.inputOmniPipeSliceStride[myAlgRank].size(); ++rpt) {
                uint64_t txScratchBase = tempAlgParams_.buffInfo.inBuffBaseOff +
                                         tempAlgParams_.stepSliceInfo.inputOmniPipeSliceStride[txIdx][rpt];
                uint64_t rxScratchBase = tempAlgParams_.buffInfo.outBuffBaseOff +
                                         tempAlgParams_.stepSliceInfo.outputOmniPipeSliceStride[rxIdx][rpt];

                const u64 txScratchOff = txScratchBase + tempAlgParams_.stepSliceInfo.stepInputSliceStride[txIdx];
                const u64 rxScratchOff = rxScratchBase + tempAlgParams_.stepSliceInfo.stepInputSliceStride[rxIdx];

                txSrcSlices.emplace_back(tempAlgParams_.buffInfo.hcclBuff.addr, txScratchOff,
                                         tempAlgParams_.stepSliceInfo.stepSliceSize[txIdx][rpt],
                                         tempAlgParams_.stepSliceInfo.stepCount[txIdx][rpt]);
                txDstSlices.emplace_back(sendCclBuffAddr, txScratchOff,
                                         tempAlgParams_.stepSliceInfo.stepSliceSize[txIdx][rpt],
                                         tempAlgParams_.stepSliceInfo.stepCount[txIdx][rpt]);
                rxSrcSlices.emplace_back(recvCclBuffAddr, rxScratchOff,
                                         tempAlgParams_.stepSliceInfo.stepSliceSize[rxIdx][rpt],
                                         tempAlgParams_.stepSliceInfo.stepCount[rxIdx][rpt]);
                rxDstSlices.emplace_back(tempAlgParams_.buffInfo.hcclBuff.addr, rxScratchOff,
                                         tempAlgParams_.stepSliceInfo.stepSliceSize[rxIdx][rpt],
                                         tempAlgParams_.stepSliceInfo.stepCount[rxIdx][rpt]);
            }
            // write模式使用tx,rx地址不生效，仅使用对端link做Post/Wait
            // read 模式使用rx, tx地址不生效，仅使用对端link做Post/Wait
        }
        TxRxSlicesList sendRecvSlicesList({txSrcSlices, txDstSlices}, {rxSrcSlices, rxDstSlices});
        TxRxChannels sendRecvChannels(channelSend, channelRecv);
        SendRecvInfo sendRecvInfo(sendRecvChannels, sendRecvSlicesList);

        if (isPcieProtocal) {
            CHK_PRT_RET(SendRecvRead(sendRecvInfo, threads[0]),
                    HCCL_ERROR("[InsTempAllGatherNHR] sendrecv failed (step=%u)", step),
                    HcclResult::HCCL_E_INTERNAL);
        }else {
            CHK_PRT_RET(SendRecvWrite(sendRecvInfo, threads[0]),
                    HCCL_ERROR("[InsTempAllGatherNHR] sendrecv failed (step=%u)", step),
                    HcclResult::HCCL_E_INTERNAL);
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

}  // namespace ops_hccl