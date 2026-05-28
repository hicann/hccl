/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_ALL_GATHER_HD_H
#define INS_TEMP_ALL_GATHER_HD_H

#include "ins_temp_all_gather_nhr.h"

namespace ops_hccl {

class InsTempAllGatherHD : public InsTempAllGatherNHR {
public:
    InsTempAllGatherHD() = default;
    explicit InsTempAllGatherHD(const OpParam &param, const u32 rankId,
                                const std::vector<std::vector<u32>> &subCommRanks);
    ~InsTempAllGatherHD() override;

    std::string Describe() const override
    {
        std::string info = "Template of all gather HD with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

protected:
    HcclResult GetStepInfo(u32 step, u32 nSteps, AicpuNHRStepInfo &stepInfo) override;
};

}  // namespace ops_hccl

#endif  // INS_TEMP_ALL_GATHER_HD_H
