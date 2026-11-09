/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// NormalizeStrideLayout 双约定归一化 + LocalDataCopy/PostLocalCopy 的寻址偏移契约。

#include "nhr_test_access.h"

namespace nhr_ut {
namespace {

    constexpr u64 kInBuffBaseOff = 8;
    constexpr u64 kOutBuffBaseOff = 4;
    constexpr u64 kHcclBuffBaseOff = 40;

    TemplateResource MakeResource()
    {
        TemplateResource resource;
        resource.threads = {1};
        resource.npu2DpuShmemPtr = reinterpret_cast<void*>(0x4000);
        resource.dpu2NpuShmemPtr = reinterpret_cast<void*>(0x5000);
        return resource;
    }

    TemplateDataParams MakeSplitParams()
    {
        TemplateDataParams params = MakeStrideParams();
        params.allRankSliceSize = {10, 0, 20};
        params.allRankProcessedDataCount = {1, 0, 2};
        params.allRankDispls = {5, 0, 7};
        return params;
    }

    TEST(NhrLayoutTest, NormalizeKeepsSplitTableWhenProvided)
    {
        InsTempReduceScatterNhrDpu tpl;
        SetupTemplate(tpl, 0, MakeRankIds(3));

        TemplateDataParams raw = MakeSplitParams();
        TemplateDataParams work = raw;
        work.sliceSize = 999; // 哨兵：若被改写则判定失败
        work.outputSliceStride = 888;

        EXPECT_EQ(tpl.NormalizeStrideLayout(raw, work, raw.allRankSliceSize.empty()), HCCL_SUCCESS);
        EXPECT_EQ(work.sliceSize, 999u);
        EXPECT_EQ(work.outputSliceStride, 888u);
        EXPECT_EQ(work.allRankSliceSize, raw.allRankSliceSize);
        EXPECT_EQ(work.allRankDispls, raw.allRankDispls);
    }

    TEST(NhrLayoutTest, NormalizeRejectsInconsistentSplitTableLength)
    {
        InsTempReduceScatterNhrDpu tpl;
        SetupTemplate(tpl, 0, MakeRankIds(3));

        // 三张表长度不一致：后续 .at() 会越界，必须在校验阶段返回参数错误
        TemplateDataParams raw = MakeSplitParams();
        raw.allRankDispls.pop_back();
        TemplateDataParams work = raw;

        EXPECT_EQ(tpl.NormalizeStrideLayout(raw, work, raw.allRankSliceSize.empty()), HCCL_E_PARA);
        EXPECT_EQ(work.allRankSliceSize, raw.allRankSliceSize);
    }

    TEST(NhrLayoutTest, NormalizeGeneratesUniformSlicesForStrideConvention)
    {
        InsTempReduceScatterNhrDpu tpl;
        SetupTemplate(tpl, 0, MakeRankIds(3));

        TemplateDataParams raw = MakeStrideParams();
        TemplateDataParams work = raw;
        ASSERT_EQ(tpl.NormalizeStrideLayout(raw, work, raw.allRankSliceSize.empty()), HCCL_SUCCESS);

        ASSERT_EQ(work.allRankSliceSize.size(), 3u);
        ASSERT_EQ(work.allRankProcessedDataCount.size(), 3u);
        ASSERT_EQ(work.allRankDispls.size(), 3u);
        for (u32 rankIdx = 0; rankIdx < 3; ++rankIdx) {
            EXPECT_EQ(work.allRankSliceSize[rankIdx], raw.sliceSize);
            EXPECT_EQ(work.allRankProcessedDataCount[rankIdx], raw.count);
            EXPECT_EQ(work.allRankDispls[rankIdx], static_cast<u64>(rankIdx) * raw.outputSliceStride);
        }
    }

    TEST(NhrLayoutTest, LocalDataCopyStrideConventionAddressing)
    {
        constexpr u32 kRankSize = 3;
        InsTempReduceScatterNhrDpu tpl;
        SetupTemplate(tpl, 0, MakeRankIds(kRankSize));
        ResetStubState();

        TemplateDataParams raw = MakeStrideParams();
        TemplateDataParams work = raw;
        ASSERT_EQ(tpl.NormalizeStrideLayout(raw, work, raw.allRankSliceSize.empty()), HCCL_SUCCESS);
        ASSERT_EQ(tpl.LocalDataCopy(raw, work, true, MakeResource()), HCCL_SUCCESS);

        // 外层分片、内层 repeat
        ASSERT_EQ(CopyCalls().size(), static_cast<size_t>(kRankSize * raw.repeatNum));
        for (u32 rankIdx = 0; rankIdx < kRankSize; ++rankIdx) {
            for (u32 rpt = 0; rpt < raw.repeatNum; ++rpt) {
                const CopyCall& call = CopyCalls()[rankIdx * raw.repeatNum + rpt];
                const u64 expectedSrc
                    = kInBuffBaseOff + rpt * raw.inputRepeatStride + static_cast<u64>(rankIdx) * raw.inputSliceStride;
                const u64 expectedDst = kHcclBuffBaseOff + rpt * kRankSize * raw.sliceSize
                                        + static_cast<u64>(rankIdx) * raw.outputSliceStride;
                EXPECT_EQ(call.srcAddr, reinterpret_cast<void*>(0x1000));
                EXPECT_EQ(call.srcOffset, expectedSrc) << "rankIdx=" << rankIdx << " rpt=" << rpt;
                EXPECT_EQ(call.srcSize, raw.sliceSize);
                EXPECT_EQ(call.srcCount, raw.count);
                EXPECT_EQ(call.dstAddr, reinterpret_cast<void*>(0x2000));
                EXPECT_EQ(call.dstOffset, expectedDst) << "rankIdx=" << rankIdx << " rpt=" << rpt;
                EXPECT_EQ(call.dstSize, raw.sliceSize);
                EXPECT_EQ(call.dstCount, raw.count);
            }
        }
    }

    TEST(NhrLayoutTest, LocalDataCopySplitConventionAddressing)
    {
        constexpr u32 kRankSize = 3;
        InsTempReduceScatterNhrDpu tpl;
        SetupTemplate(tpl, 0, MakeRankIds(kRankSize));
        ResetStubState();

        TemplateDataParams raw = MakeSplitParams();
        TemplateDataParams work = raw;
        ASSERT_EQ(tpl.NormalizeStrideLayout(raw, work, raw.allRankSliceSize.empty()), HCCL_SUCCESS);
        ASSERT_EQ(tpl.LocalDataCopy(raw, work, false, MakeResource()), HCCL_SUCCESS);

        // 中间槽位 size == 0 被跳过
        ASSERT_EQ(CopyCalls().size(), 4u);
        const u64 sizes[4] = {10, 10, 20, 20};
        const u64 expectedSrc[4] = {
            kInBuffBaseOff + 5,
            kInBuffBaseOff + raw.inputRepeatStride + 5,
            kInBuffBaseOff + 7,
            kInBuffBaseOff + raw.inputRepeatStride + 7,
        };
        const u64 expectedDst[4] = {
            kHcclBuffBaseOff + 5,
            kHcclBuffBaseOff + kRankSize * 10 + 5,
            kHcclBuffBaseOff + 7,
            kHcclBuffBaseOff + kRankSize * 20 + 7,
        };
        for (size_t i = 0; i < CopyCalls().size(); ++i) {
            EXPECT_EQ(CopyCalls()[i].srcOffset, expectedSrc[i]) << "call=" << i;
            EXPECT_EQ(CopyCalls()[i].dstOffset, expectedDst[i]) << "call=" << i;
            EXPECT_EQ(CopyCalls()[i].srcSize, sizes[i]) << "call=" << i;
        }
        EXPECT_EQ(CopyCalls()[0].srcCount, 1u);
        EXPECT_EQ(CopyCalls()[2].srcCount, 2u);
    }

    TEST(NhrLayoutTest, LocalDataCopySkipsEmptyStrideSlices)
    {
        InsTempReduceScatterNhrDpu tpl;
        SetupTemplate(tpl, 0, MakeRankIds(2));
        ResetStubState();

        TemplateDataParams raw = MakeStrideParams();
        raw.sliceSize = 0;
        TemplateDataParams work = raw;
        ASSERT_EQ(tpl.NormalizeStrideLayout(raw, work, raw.allRankSliceSize.empty()), HCCL_SUCCESS);
        ASSERT_EQ(tpl.LocalDataCopy(raw, work, true, MakeResource()), HCCL_SUCCESS);
        EXPECT_TRUE(CopyCalls().empty());
    }

    TEST(NhrLayoutTest, PostLocalCopyStrideConventionAddressing)
    {
        constexpr u32 kRankSize = 3;
        InsTempReduceScatterNhrDpu tpl;
        SetupTemplate(tpl, 1, MakeRankIds(kRankSize));
        ResetStubState();

        TemplateDataParams raw = MakeStrideParams();
        TemplateDataParams work = raw;
        ASSERT_EQ(tpl.NormalizeStrideLayout(raw, work, raw.allRankSliceSize.empty()), HCCL_SUCCESS);
        ASSERT_EQ(tpl.PostLocalCopy(raw, work, MakeResource()), HCCL_SUCCESS);

        ASSERT_EQ(CopyCalls().size(), raw.repeatNum);
        for (u32 rpt = 0; rpt < raw.repeatNum; ++rpt) {
            const u64 expectedSrc = kHcclBuffBaseOff + rpt * kRankSize * raw.sliceSize + raw.outputSliceStride;
            const u64 expectedDst = kOutBuffBaseOff + rpt * raw.outputRepeatStride;
            EXPECT_EQ(CopyCalls()[rpt].srcAddr, reinterpret_cast<void*>(0x2000));
            EXPECT_EQ(CopyCalls()[rpt].srcOffset, expectedSrc) << "rpt=" << rpt;
            EXPECT_EQ(CopyCalls()[rpt].dstAddr, reinterpret_cast<void*>(0x3000));
            EXPECT_EQ(CopyCalls()[rpt].dstOffset, expectedDst) << "rpt=" << rpt;
            EXPECT_EQ(CopyCalls()[rpt].srcSize, raw.sliceSize);
            EXPECT_EQ(CopyCalls()[rpt].srcCount, raw.count);
        }
    }

    TEST(NhrLayoutTest, PostLocalCopySplitConventionAddressing)
    {
        constexpr u32 kRankSize = 3;
        InsTempReduceScatterNhrDpu tpl;
        SetupTemplate(tpl, 2, MakeRankIds(kRankSize));
        ResetStubState();

        TemplateDataParams raw = MakeSplitParams();
        TemplateDataParams work = raw;
        ASSERT_EQ(tpl.PostLocalCopy(raw, work, MakeResource()), HCCL_SUCCESS);

        // 本 rank 逻辑下标 2：sliceSize=20，displs=7
        ASSERT_EQ(CopyCalls().size(), raw.repeatNum);
        for (u32 rpt = 0; rpt < raw.repeatNum; ++rpt) {
            EXPECT_EQ(CopyCalls()[rpt].srcOffset, kHcclBuffBaseOff + rpt * kRankSize * 20 + 7);
            EXPECT_EQ(CopyCalls()[rpt].dstOffset, kOutBuffBaseOff + rpt * raw.outputRepeatStride);
            EXPECT_EQ(CopyCalls()[rpt].srcSize, 20u);
            EXPECT_EQ(CopyCalls()[rpt].srcCount, 2u);
        }
    }

    TEST(NhrLayoutTest, PostLocalCopyReturnsEarlyWithoutOutputBuffer)
    {
        InsTempReduceScatterNhrDpu tpl;
        SetupTemplate(tpl, 0, MakeRankIds(2));
        ResetStubState();

        TemplateDataParams raw = MakeStrideParams();
        raw.buffInfo.outputPtr = nullptr;
        TemplateDataParams work = raw;
        ASSERT_EQ(tpl.NormalizeStrideLayout(raw, work, raw.allRankSliceSize.empty()), HCCL_SUCCESS);
        ASSERT_EQ(tpl.PostLocalCopy(raw, work, MakeResource()), HCCL_SUCCESS);
        EXPECT_TRUE(CopyCalls().empty());
    }

    TEST(NhrLayoutTest, PostLocalCopySkipsEmptyLocalSlice)
    {
        InsTempReduceScatterNhrDpu tpl;
        SetupTemplate(tpl, 1, MakeRankIds(3));
        ResetStubState();

        TemplateDataParams raw = MakeSplitParams();
        TemplateDataParams work = raw;
        ASSERT_EQ(tpl.PostLocalCopy(raw, work, MakeResource()), HCCL_SUCCESS);
        EXPECT_TRUE(CopyCalls().empty());
    }

    TEST(NhrLayoutTest, RepeatStrideTermsAreNotInterchangeable)
    {
        // scratch 侧 rpt 步长 = N * sliceSize（N 个槽位），
        // output 侧 = outputRepeatStride（紧凑 1/N），二者相差 N 倍
        constexpr u32 kRankSize = 3;
        InsTempReduceScatterNhrDpu tpl;
        SetupTemplate(tpl, 0, MakeRankIds(kRankSize));
        ResetStubState();

        TemplateDataParams raw = MakeStrideParams();
        TemplateDataParams work = raw;
        ASSERT_EQ(tpl.NormalizeStrideLayout(raw, work, raw.allRankSliceSize.empty()), HCCL_SUCCESS);
        ASSERT_EQ(tpl.PostLocalCopy(raw, work, MakeResource()), HCCL_SUCCESS);
        ASSERT_EQ(CopyCalls().size(), 2u);

        const u64 scratchStep = kRankSize * raw.sliceSize;
        const u64 outputStep = raw.outputRepeatStride;
        EXPECT_EQ(CopyCalls()[1].srcOffset - CopyCalls()[0].srcOffset, scratchStep);
        EXPECT_EQ(CopyCalls()[1].dstOffset - CopyCalls()[0].dstOffset, outputStep);
        EXPECT_NE(scratchStep, outputStep);
    }

} // namespace
} // namespace nhr_ut
