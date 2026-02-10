/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCE_SCATTER_MESH_HOST_DPU_TEMP_H
#define REDUCE_SCATTER_MESH_HOST_DPU_TEMP_H

#include "alg_template_base.h"

namespace ops_hccl {

class ReduceScatterHostDpuTemplate : public AlgTemplateBase {
public:
    explicit ReduceScatterHostDpuTemplate();
    ~ReduceScatterHostDpuTemplate() override;

    HcclResult RunAsync(const OpParam& param);
    HcclResult RunAsync(const DPUAlgResourceCtx *dpuResCtx);
    HcclResult RunDataSend(u32 dstRank, u32 srcRank,
                        const std::vector<Slice>& slices,
                        std::vector<ChannelInfo>& channels,
                        const DPUAlgResourceCtx *dpuResCtx);
};

}
#endif