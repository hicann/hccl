/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCE_SCATTER_MESH_LOCAL_REDUCE_H
#define REDUCE_SCATTER_MESH_LOCAL_REDUCE_H

#include "alg_template_base.h"

namespace ops_hccl {
class ReduceScatterLocalReduceTemplate : public AlgTemplateBase {
public:
    explicit ReduceScatterLocalReduceTemplate();
    ~ReduceScatterLocalReduceTemplate() override;
    HcclResult RunAsync(const u32 rank, const u32 rankSize, std::vector<ChannelInfo> &channels);
};

}
#endif
