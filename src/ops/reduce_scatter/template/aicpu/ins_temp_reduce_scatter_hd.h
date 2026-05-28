/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_REDUCE_SCATTER_HD_H
#define INS_TEMP_REDUCE_SCATTER_HD_H

#include "ins_temp_reduce_scatter_nhr.h"

namespace ops_hccl {

class InsTempReduceScatterHD : public InsTempReduceScatterNHR {
public:
    InsTempReduceScatterHD() = default;
    explicit InsTempReduceScatterHD(const OpParam &param, const u32 rankId,
                                    const std::vector<std::vector<u32>> &subCommRanks);
    ~InsTempReduceScatterHD() override;

    std::string Describe() const override
    {
        std::string info = "Template of reduce scatter HD with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

protected:
    HcclResult GetStepInfoList(std::vector<AicpuNHRStepInfo> &stepInfoList) override;
};

}  // namespace ops_hccl

#endif  // INS_TEMP_REDUCE_SCATTER_HD_H
