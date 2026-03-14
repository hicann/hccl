/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_TEMP_ALL_TO_ALL_V_MESH_1D_H
#define HCCL_CCU_TEMP_ALL_TO_ALL_V_MESH_1D_H

#include "utils.h"
#include "ccu_alg_template_base.h"

namespace ops_hccl {

class CcuTempAlltoAllVMesh1D : public CcuAlgTemplateBase {
public:
    explicit  CcuTempAlltoAllVMesh1D(const OpParam& param, 
                                        const u32 rankId, // 传通信域的rankId，userRank
                                        const std::vector<std::vector<u32>> &subCommRanks);

    ~CcuTempAlltoAllVMesh1D() override;

    std::string Describe() const override
    {
        return StringFormat("Template of All to All ccu mesh 1D with tempRankSize [%u].",
                            tempRankSize_);
    }

    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                       AlgResourceRequest& resourceRequest) override;

    HcclResult KernelRun(const OpParam& param,
                         const TemplateDataParams& templateDataParams,
                         const TemplateResource& templateResource) override;

    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;

    void InitInsAlgTemplate(
        std::vector<u64> &sendCounts, std::vector<u64> &recvCounts,
        std::vector<u64> &sdispls, std::vector<u64> &rdispls);

    void SetA2ASendRecvInfo(const A2ASendRecvInfo &sendRecvInfo);

private:
    A2ASendRecvInfo localSendRecvInfo_;
    u32             concurrentSendRecvNum_ = 8;
    u64 buffBlockSize_ = 0;
    BuffInfo buffInfo_{0};
    uint64_t sendStrideSize_ = 0;  // Bytes
    uint64_t recvStrideSize_ = 0;  // Bytes
    uint32_t mySubCommRank_ = 0;
    uint32_t tempRankSize_ = 0;
    std::vector<u64> sendCounts_;
    std::vector<u64> recvCounts_;
    std::vector<u64> sdispls_;
    std::vector<u64> rdispls_;
};

}// namespace ops_hccl

#endif// HCCL_CCU_TEMP_ALL_TO_ALL_MESH_1D_H