/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BASE_TEMPLATE_H
#define BASE_TEMPLATE_H

#include "algo_desc.h"
#include "alg_param.h"
#include "data_types.h"

namespace ops_hccl {

// 通信模板基类
class BaseTemplate {
public:
    explicit BaseTemplate(const u32 myRank, const std::vector<u32>& ranks, TemplateDesc templateDesc)
        : myRank_(myRank),
          ranks_(ranks),
          templateDesc_(templateDesc)
    {}
    virtual ~BaseTemplate() = default;

    void SetNetLayer(int netLayer) { netLayer_ = netLayer; }
    void SetChannelsPerRank(u32 val) { channelsPerRank_ = val; }
    void SetDataSize(u64 size) { dataSize_ = size; }

    // 计算 channel/notify/thread 资源请求
    virtual HcclResult CalcRes(HcclComm comm, HcclAlgEngineType engineType, AlgResourceRequest& res)
    {
        const u32 rankSize = static_cast<u32>(ranks_.size());
        if (rankSize <= 1) {
            res.channels.emplace_back();
            return HCCL_SUCCESS;
        }

        std::vector<std::vector<u32>> subcommInfo = {ranks_};
        std::vector<HcclChannelDesc> levelChannels;
        OpParam chanParam;
        chanParam.engine = engineType;
        chanParam.userRank = myRank_;
        TopoInfoWithNetLayerDetails chanTopoInfo;
        chanTopoInfo.userRank = myRank_;
        CHK_RET(DoCalcChannelRequest(comm, chanParam, &chanTopoInfo, subcommInfo, levelChannels));
        for (const auto& desc : levelChannels) {
            channels_.push_back(desc);
        }
        res.channels.push_back(levelChannels);

        channelsPerRank_ = CalcChannelsPerRankInternal(levelChannels);

        CHK_RET(GetRes(res));
        return HCCL_SUCCESS;
    }

    // 计算线程数和 notify 数
    virtual HcclResult GetRes(AlgResourceRequest& res) const
    {
        u32 threadNum = DoCalcThreadNum();
        u32 notifyPerThread = DoCalcNotifyPerThread();
        res.slaveThreadNum = threadNum - 1;
        res.notifyNumPerThread.assign(res.slaveThreadNum, notifyPerThread);
        res.notifyNumOnMainThread = threadNum - 1;
        return HCCL_SUCCESS;
    }

    // 算法编排入口，子类实现具体通信逻辑
    virtual HcclResult
    KernelRun(const DataParams& tempAlgParams, TemplateResource& templateResource, std::vector<u32>& ranksForOutputData)
    {
        ranksForOutputData.clear();
        return HCCL_SUCCESS;
    }

protected:
    // 算法专属：计算 channel 请求
    virtual HcclResult DoCalcChannelRequest(
        HcclComm comm, const OpParam& param, TopoInfoWithNetLayerDetails* topoInfo,
        const std::vector<std::vector<u32>>& subcommInfo, std::vector<HcclChannelDesc>& levelChannels)
        = 0;

    // 算法专属：计算线程数
    virtual u32 DoCalcThreadNum() const = 0;

    // 算法专属：计算每线程 notify 数
    virtual u32 DoCalcNotifyPerThread() const = 0;

    static u32 CalcChannelsPerRankInternal(const std::vector<HcclChannelDesc>& channels)
    {
        u32 channelsPerRank = 1;
        u32 currentRank = INVALID_VALUE_RANKID;
        u32 currentCount = 0;
        for (const auto& channel : channels) {
            if (channel.remoteRank == currentRank) {
                currentCount++;
            } else {
                if (currentCount > channelsPerRank) {
                    channelsPerRank = currentCount;
                }
                currentRank = channel.remoteRank;
                currentCount = 1;
            }
        }
        if (currentCount > channelsPerRank) {
            channelsPerRank = currentCount;
        }
        return channelsPerRank;
    }

    std::vector<HcclChannelDesc> channels_;
    u32 myRank_ = INVALID_VALUE_RANKID;
    std::vector<u32> ranks_;
    TemplateDesc templateDesc_;
    int netLayer_ = -1;
    u32 channelsPerRank_ = 1;
    u64 dataSize_ = 0;
};

} // namespace ops_hccl

#endif // BASE_TEMPLATE_H
