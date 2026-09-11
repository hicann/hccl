/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_BASE_TEMPLATE_H
#define AICPU_BASE_TEMPLATE_H

#include <vector>

#include "base_template.h"
#include "alg_param.h"
#include "data_ops.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

// AICPU 引擎基类
class AicpuBaseTemplate : public BaseTemplate {
public:
    AicpuBaseTemplate(u32 myRank, std::vector<u32> ranks, TemplateDesc templateDesc)
        : BaseTemplate(myRank, std::move(ranks), templateDesc)
    {}
    ~AicpuBaseTemplate() = default;

    HcclResult KernelRun(
        const DataParams& tempAlgParams, TemplateResource& templateResource,
        std::vector<u32>& ranksForOutputData) override;

protected:
    // 子类实现：具体通信编排（RunMeshAllGather / RunNhrAllGather）
    virtual HcclResult RunAlgorithm(std::vector<DataSlicesList>& txRxSlicesLists, std::vector<u32>& ranksForOutputData)
        = 0;

    virtual HcclResult PreCopy(const std::vector<ThreadHandle>& threads);
    virtual HcclResult PostCopy(const std::vector<ThreadHandle>& threads);
    virtual HcclResult SendAll(
        const std::vector<DataSlicesList>& txRxSlicesLists, TemplateResource& templateResource,
        const std::vector<ThreadHandle>& threads);

    // 单 rank 合并直拷：inputBufferPtr → outputBufferPtr，跳过 cclBuffer 中转
    virtual HcclResult CopyInputToOutput(const std::vector<ThreadHandle>& threads);

    // flag: true=NHR 在搬运边界 sync，false=Mesh 在通信段边界 sync
    bool syncAtCopyBoundary_{false};

    // 非虚 helper：从线程同步（空线程时安全返回）
    HcclResult PreSyncSubThreads(
        const ThreadHandle& mainThread, const std::vector<ThreadHandle>& subThreads,
        const std::vector<u32>& notifyIdxMainToSub);
    HcclResult PostSyncSubThreads(
        const ThreadHandle& mainThread, const std::vector<ThreadHandle>& subThreads,
        const std::vector<u32>& notifyIdxSubToMain);

    bool IsPcieProtocol(const std::map<u32, std::vector<ChannelInfo>>& channels) const
    {
        for (auto it = channels.begin(); it != channels.end(); ++it) {
            if (!it->second.empty() && it->second[0].protocol == CommProtocol::COMM_PROTOCOL_PCIE) {
                return true;
            }
        }
        return false;
    }

    HcclResult PrepareDataSplit(const std::map<u32, std::vector<ChannelInfo>>& channels);

    // KernelRun 子函数：初始化算法参数与 rank 规模
    void InitKernelRunParams(const DataParams& tempAlgParams);

    // KernelRun 子函数：准备从线程列表与 notify 索引
    void PrepareSubThreads(
        const std::vector<ThreadHandle>& threads, std::vector<ThreadHandle>& subThreads,
        std::vector<u32>& notifyIdxMainToSub, std::vector<u32>& notifyIdxSubToMain) const;

    // KernelRun 子函数：处理单 rank 场景
    HcclResult RunSingleRank(
        TemplateResource& templateResource, const std::vector<ThreadHandle>& subThreads,
        const std::vector<u32>& notifyIdxMainToSub, const std::vector<u32>& notifyIdxSubToMain,
        std::vector<u32>& ranksForOutputData);

    // KernelRun 子函数：正常多 rank 通信流程
    HcclResult RunMultiRank(
        TemplateResource& templateResource, const std::vector<ThreadHandle>& subThreads,
        const std::vector<u32>& notifyIdxMainToSub, const std::vector<u32>& notifyIdxSubToMain,
        std::vector<u32>& ranksForOutputData);

    DataParams tempAlgParams_{};
    u32 templateRankSize_{0};
    std::vector<u32> ranksForOutputData_{};

    ChannelSplitInfo channelSplit_{};
};

} // namespace ops_hccl

#endif // AICPU_BASE_TEMPLATE_H
