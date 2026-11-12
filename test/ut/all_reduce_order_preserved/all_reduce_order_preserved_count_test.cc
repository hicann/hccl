/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <numeric>
#include <vector>
#include "gtest/gtest.h"
#include "ins_v2_all_reduce_order_preserved_executor.h"
#include "alg_attrs_registry.h"

// 白盒包含真实执行器实现；UT无需注册全局算法。
#undef REGISTER_EXEC_V2_MULTI
#define REGISTER_EXEC_V2_MULTI(...)
#undef REGISTER_ALG_ATTRS
#define REGISTER_ALG_ATTRS(...)
#include "ins_v2_all_reduce_order_preserved_executor.cc"

namespace ops_hccl {

// 本用例不涉及端口分流或对称内存；意外进入这些路径时立即失败。
HcclResult CalcDataSplitByPortGroupCommon(
    const u64, const u64, const std::vector<ChannelInfo>&, std::vector<u64>&, std::vector<u64>&, std::vector<u64>&,
    const u32)
{
    ADD_FAILURE() << "Unexpected port-group split";
    return HCCL_E_INTERNAL;
}

HcclResult FillChannelSymWinPeerAddrs(void*, u64, void*, u64, ChannelInfo&)
{
    ADD_FAILURE() << "Unexpected symmetric memory lookup";
    return HCCL_E_INTERNAL;
}

namespace {

    constexpr u64 HALF_CCL_BUFF_SIZE = 1024 * 1024;
    constexpr u64 DATA_TYPE_SIZE = 4;
    constexpr u32 RANK_SIZE = 8;
    std::vector<TemplateDataParams> rsCalls;
    std::vector<TemplateDataParams> agCalls;
    std::vector<MemBlockInfo> rsBlocks;

    // 仅替换通信模板下发，真实执行器负责分轮、切片大小和偏移计算。
    template <bool IS_RS>
    class RecordingTemplate : public InsAlgTemplateBase {
    public:
        using InsAlgTemplateBase::InsAlgTemplateBase;
        std::string Describe() const override { return "RecordingTemplate"; }
        void GetNotifyIdxMainToSub(std::vector<u32>&) override {}
        void GetNotifyIdxSubToMain(std::vector<u32>&) override {}
        HcclResult GetRes(AlgResourceRequest& request) const override
        {
            request.slaveThreadNum = 0;
            return HCCL_SUCCESS;
        }
        HcclResult SetchannelsPerRank(const std::map<u32, std::vector<ChannelInfo>>&) override { return HCCL_SUCCESS; }
        void SetMemBlockInfo(const MemBlockInfo& info) { rsBlocks.push_back(info); }
        HcclResult KernelRun(const OpParam&, const TemplateDataParams& params, TemplateResource&) override
        {
            auto& calls = IS_RS ? rsCalls : agCalls;
            calls.push_back(params);
            // 固定输入最多三轮，防止回归导致零进展循环时UT挂死。
            return calls.size() <= 3 ? HCCL_SUCCESS : HCCL_E_INTERNAL;
        }
    };

    struct UnusedTopoMatch {
        HcclResult MatchTopo(TopoInfoWithNetLayerDetails*, AlgHierarchyInfoForAllLevel&, const AlgAttrs&)
        {
            ADD_FAILURE() << "Unexpected topology matching";
            return HCCL_E_INTERNAL;
        }
    };

    class LoopExecutor :
        public InsV2AllReduceOrderPreservedExecutor<
            UnusedTopoMatch, RecordingTemplate<true>, RecordingTemplate<false>, RecordingTemplate<false>> {
    public:
        explicit LoopExecutor(u64 count)
        {
            myRank_ = 0;
            rankSize_ = RANK_SIZE;
            dataTypeSize_ = DATA_TYPE_SIZE;
            dataCount_ = count;
            remoteRankToChannelInfo_.resize(1);
        }
        using InsV2AllReduceOrderPreservedExecutor::OrchestrateLoop;
    };

    TEST(AllReduceOrderPreservedCountTest, ClampedTailAfterFullLoop)
    {
        rsCalls.clear();
        agCalls.clear();
        rsBlocks.clear();
        OpParam param{};
        param.DataDes.count = 524287; // 2M-1，M为1 MiB前半区可容纳的FP32元素数。
        param.DataDes.dataType = HCCL_DATA_TYPE_FP32;
        param.reduceType = HCCL_REDUCE_SUM;
        param.inputSize = param.outputSize = param.DataDes.count * DATA_TYPE_SIZE;
        AlgResourceCtxSerializable resCtx{};
        resCtx.cclMem.size = 2 * HALF_CCL_BUFF_SIZE;
        resCtx.threads = {0};
        resCtx.algHierarchyInfo.infos = {{{0, 1, 2, 3, 4, 5, 6, 7}}};
        LoopExecutor executor(param.DataDes.count);
        ASSERT_EQ(executor.OrchestrateLoop(param, resCtx), HCCL_SUCCESS);

        // 完整一轮后触发尾块钳制，剩余6个元素必须另起一轮。
        const std::vector<u64> expectedCounts{262144, 262137, 6};
        ASSERT_EQ(rsCalls.size(), expectedCounts.size());
        ASSERT_EQ(agCalls.size(), expectedCounts.size());
        ASSERT_EQ(rsBlocks.size(), expectedCounts.size());
        u64 processedBytes = 0;
        for (size_t i = 0; i < expectedCounts.size(); ++i) {
            const u64 expectedBytes = expectedCounts[i] * DATA_TYPE_SIZE;
            EXPECT_EQ(
                std::accumulate(rsCalls[i].allRankSliceSize.begin(), rsCalls[i].allRankSliceSize.end(), u64{0}),
                expectedBytes);
            EXPECT_EQ(agCalls[i].allRankSliceSize, rsCalls[i].allRankSliceSize);
            EXPECT_EQ(rsCalls[i].buffInfo.inBuffBaseOff, processedBytes);
            EXPECT_EQ(agCalls[i].buffInfo.outBuffBaseOff, processedBytes);
            EXPECT_LE(RANK_SIZE * rsCalls[i].tailSize, HALF_CCL_BUFF_SIZE);
            EXPECT_LE(expectedBytes, HALF_CCL_BUFF_SIZE);
            ASSERT_EQ(rsBlocks[i].outputOffsets.size(), RANK_SIZE);
            EXPECT_LE(rsBlocks[i].outputOffsets.back() + rsCalls[i].tailSize, resCtx.cclMem.size);
            processedBytes += expectedBytes;
        }
        EXPECT_EQ(processedBytes, param.inputSize);
    }

} // namespace
} // namespace ops_hccl
