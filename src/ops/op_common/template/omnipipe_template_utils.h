/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OMNIPIPE_TEMPLATE_UTILS_H
#define OMNIPIPE_TEMPLATE_UTILS_H

#include <cstdint>
#include <map>
#include <vector>

#include "common_alg_template_base.h"
#include "template_utils.h"
#include "hccl_common.h"

namespace ops_hccl {

// 填充各通信步骤共用的分片参数。普通路径在 ccl scratch 上处理；对称路径直接访问 user input（完整
// 分片布局），输入基址还需叠加当前 loop 在完整输入布局中的偏移。
HcclResult FillOmniPipeTemplateAlgParams(
    TemplateDataParams& tempAlgParams, const StepSliceInfo& stepSliceInfo, bool supportSymmetricMemory = false,
    u64 processedDataCount = 0, u64 dataTypeSize = 0);

HcclResult PrepareOmniPipeDataSplitForMultiChannel(
    CommonAlgTemplateBase* algTemplate, const TemplateDataParams& tempAlgParams, HcclDataType dataType,
    TemplateResource& templateResource, std::vector<std::vector<std::vector<u64>>>& dataSplitVec,
    std::vector<std::vector<std::vector<u64>>>& dataOffsetVec);

HcclResult ClassifyOmniPipeChannelsByLevel(
    u32 localRank, const std::vector<std::vector<ChannelInfo>>& channels,
    const std::vector<const std::vector<std::vector<u32>>*>& subCommsByLevel,
    const std::vector<uint64_t>& rankSizesByLevel,
    std::vector<std::map<u32, std::vector<ChannelInfo>>>& channelsByLevel);

} // namespace ops_hccl

#endif
