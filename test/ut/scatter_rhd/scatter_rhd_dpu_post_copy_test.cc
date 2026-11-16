/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details.
 * You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// PostLocalCopy 输出契约 UT（DPU inter-node 模板）：走公开 KernelRun（eager 舞步已桩化）驱动，
// 在 HcommLocalCopyOnThread 桩处录制 wrapper 合成后的 (src, dst, len)，反推 DataSlice 偏移断言。
// 契约面：散射单份退化（回归基线）/ 非零 base+stride（Broadcast 中间层形状）/ 同址免搬 /
// 同偏移不同缓冲（防裸 offset 相等误跳）/ 多 repeat / outputPtr 空。
// 布局常量：rankSize=4（2^N → σ 恒等 ownUnit==myAlgRank，RearrangeToSigma 零开销跳过），
// myRank=11 → myAlgRank=ownUnit=1；均匀 16B unit → ownOffset=16、ownSize=16、totalUnitBytes=64。
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <string>
#include <vector>
#include "ins_temp_scatter_rhd_dpu_inter_node.h"
#include "template_utils.h"

using namespace ops_hccl;

// ut_stub.cc 定义的录制口（定义见该文件；两处 UtLocalCopyCall 定义逐字一致，ODR 安全）
struct UtLocalCopyCall {
    const void* src;
    void* dst;
    uint64_t len;
};
extern std::vector<UtLocalCopyCall> g_utLocalCopyCalls;

namespace {
class ScatterRhdDpuPostCopyTest : public ::testing::Test {
protected:
    void SetUp() override { g_utLocalCopyCalls.clear(); }

    static constexpr u32 kMyUserRank = 11; // subCommRanks_[0]={10,11,12,13} → myAlgRank=1

    OpParam MakeParam()
    {
        OpParam param{};
        const std::string tag = "rhd_post_copy_ut";
        std::copy(tag.begin(), tag.end(), param.algTag); // OpParam{} 零初始化，尾部天然 NUL
        param.DataDes.dataType = HCCL_DATA_TYPE_INT32;   // dataTypeSize_=4 → ownCount=4
        return param;
    }

    TemplateDataParams
    MakeParams(void* cclAddr, void* outPtr, u64 repeatNum, u64 stride, u64 hcclBaseOff, u64 outBaseOff)
    {
        TemplateDataParams params{};
        params.count = 4;
        params.sliceSize = 16;
        params.repeatNum = repeatNum;
        params.outputSliceStride = stride;
        params.root = 10; // 首成员 → rootAlgRank=0（RhdScatterParamValid 通过）
        params.dataType = HCCL_DATA_TYPE_INT32;
        params.allRankSliceSize = {16, 16, 16, 16};
        params.buffInfo.inputPtr = cclAddr;
        params.buffInfo.outputPtr = outPtr;
        params.buffInfo.hcclBuff.addr = cclAddr;
        params.buffInfo.hcclBuff.size = 1024;
        params.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
        params.buffInfo.outBuffType = (outPtr == cclAddr) ? BufferType::HCCL_BUFFER : BufferType::OUTPUT;
        params.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
        params.buffInfo.hcclBuffBaseOff = hcclBaseOff;
        params.buffInfo.outBuffBaseOff = outBaseOff;
        return params;
    }

    HcclResult RunPostCopy(TemplateDataParams& params)
    {
        OpParam param = MakeParam();
        InsTempScatterRhdDpuInterNode tmpl(param, kMyUserRank, {{10, 11, 12, 13}});
        TemplateResource resource{};
        resource.threads = {1}; // ThreadHandle（uint64_t）哑句柄，桩不消费
        resource.npu2DpuShmemPtr = npu2Dpu_.data();
        resource.dpu2NpuShmemPtr = dpu2Npu_.data();
        return tmpl.KernelRun(param, params, resource);
    }

    std::array<uint8_t, 256> ccl_{};
    std::array<uint8_t, 256> out_{};
    std::array<uint8_t, 64> npu2Dpu_{};
    std::array<uint8_t, 64> dpu2Npu_{};
};
} // namespace

// 散射接线退化（repeatNum=1/stride=0/用户输出）：与旧单槽位行为逐字节一致的回归基线
TEST_F(ScatterRhdDpuPostCopyTest, ScatterWiringDegeneratesToSingleSlotCopy)
{
    TemplateDataParams params = MakeParams(ccl_.data(), out_.data(), 1, 0, 0, 0);
    ASSERT_EQ(RunPostCopy(params), HcclResult::HCCL_SUCCESS);
    ASSERT_EQ(g_utLocalCopyCalls.size(), 1u);
    EXPECT_EQ(g_utLocalCopyCalls[0].src, static_cast<const void*>(ccl_.data() + 16)); // 0+ownOffset(16)
    EXPECT_EQ(g_utLocalCopyCalls[0].dst, static_cast<void*>(out_.data()));            // 0+myAlgRank*0
    EXPECT_EQ(g_utLocalCopyCalls[0].len, 16u);
}

// Broadcast 中间层形状（非零 base+stride）：src 按块内 σ 槽位、dst 按自然槽位+rank 维 stride
TEST_F(ScatterRhdDpuPostCopyTest, BroadcastShapeConsumesBaseAndStride)
{
    TemplateDataParams params = MakeParams(ccl_.data(), out_.data(), 1, 16, 0x40, 0x80);
    ASSERT_EQ(RunPostCopy(params), HcclResult::HCCL_SUCCESS);
    ASSERT_EQ(g_utLocalCopyCalls.size(), 1u);
    EXPECT_EQ(g_utLocalCopyCalls[0].src, static_cast<const void*>(ccl_.data() + 0x50)); // 0x40+16
    EXPECT_EQ(g_utLocalCopyCalls[0].dst, static_cast<void*>(out_.data() + 0x90));       // 0x80+1*16
    EXPECT_EQ(g_utLocalCopyCalls[0].len, 16u);
}

// 同址免搬：输出即本 ccl 且偏移重合 → 不产生 LocalCopy（同段地址重叠写会被 Checker 判冲突）
TEST_F(ScatterRhdDpuPostCopyTest, SameBufferSameOffsetIsSkipped)
{
    TemplateDataParams params = MakeParams(ccl_.data(), ccl_.data(), 1, 16, 0x40, 0x40);
    // scratch = 0x40+ownOffset(16) = 0x50；out = 0x40+myAlgRank(1)*16 = 0x50 → 重合
    ASSERT_EQ(RunPostCopy(params), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(g_utLocalCopyCalls.size(), 0u);
}

// 同偏移不同缓冲（防裸 offset 相等误跳）：偏移巧合相等但 ccl→用户输出是必发真拷贝
TEST_F(ScatterRhdDpuPostCopyTest, SameOffsetDistinctBuffersStillCopies)
{
    TemplateDataParams params = MakeParams(ccl_.data(), out_.data(), 1, 0, 0x10, 0x20);
    // scratch = 0x10+16 = 0x20；out = 0x20+0 = 0x20 → 偏移相等、缓冲不同 → 必须拷
    ASSERT_EQ(RunPostCopy(params), HcclResult::HCCL_SUCCESS);
    ASSERT_EQ(g_utLocalCopyCalls.size(), 1u);
    EXPECT_EQ(g_utLocalCopyCalls[0].src, static_cast<const void*>(ccl_.data() + 0x20));
    EXPECT_EQ(g_utLocalCopyCalls[0].dst, static_cast<void*>(out_.data() + 0x20));
    EXPECT_EQ(g_utLocalCopyCalls[0].len, 16u);
}

// 多 repeat：src 按 repeat×totalUnitBytes 块步长、dst 按 repeat×sliceSize 份步长
TEST_F(ScatterRhdDpuPostCopyTest, MultiRepeatAdvancesBlockAndShareOffsets)
{
    TemplateDataParams params = MakeParams(ccl_.data(), out_.data(), 2, 16, 0, 0x10);
    ASSERT_EQ(RunPostCopy(params), HcclResult::HCCL_SUCCESS);
    ASSERT_EQ(g_utLocalCopyCalls.size(), 2u);
    // r=0：src = 0+0+ownOffset(16)；dst = 0x10+1*16+0 = 0x20
    EXPECT_EQ(g_utLocalCopyCalls[0].src, static_cast<const void*>(ccl_.data() + 0x10));
    EXPECT_EQ(g_utLocalCopyCalls[0].dst, static_cast<void*>(out_.data() + 0x20));
    EXPECT_EQ(g_utLocalCopyCalls[0].len, 16u);
    // r=1：src = 64+0+16 = 0x50；dst = 0x10+0x10+0x10 = 0x30
    EXPECT_EQ(g_utLocalCopyCalls[1].src, static_cast<const void*>(ccl_.data() + 0x50));
    EXPECT_EQ(g_utLocalCopyCalls[1].dst, static_cast<void*>(out_.data() + 0x30));
    EXPECT_EQ(g_utLocalCopyCalls[1].len, 16u);
}

// outputPtr 为空（中间层免落位形态）：不产生任何 LocalCopy（既有守卫保持）
TEST_F(ScatterRhdDpuPostCopyTest, NullOutputPtrCopiesNothing)
{
    TemplateDataParams params = MakeParams(ccl_.data(), nullptr, 1, 16, 0x40, 0x40);
    ASSERT_EQ(RunPostCopy(params), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(g_utLocalCopyCalls.size(), 0u);
}
