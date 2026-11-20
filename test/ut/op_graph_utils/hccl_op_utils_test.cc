/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// hccl_op_utils.cc 公共纯函数 UT: 真实编译被测源文件, hcomm/rt/log 依赖由 stub/ 提供
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "hccl_op_utils.h"
#include "hccl/hcom.h" // HCCL_KERNEL_OP_TYPE_* 常量
#include "hccl_custom_op.h"

using namespace hccl;

// ---------------------------------------------------------------------------
// GeDataTypeToHccl: 全类型表 (含 1b2debaf 补齐的 BOOL/FP8 族)
// ---------------------------------------------------------------------------
TEST(GeDataTypeToHcclTest, BasicTypes)
{
    EXPECT_EQ(HCCL_DATA_TYPE_FP32, GeDataTypeToHccl(ge::DT_FLOAT));
    EXPECT_EQ(HCCL_DATA_TYPE_FP16, GeDataTypeToHccl(ge::DT_FLOAT16));
    EXPECT_EQ(HCCL_DATA_TYPE_INT8, GeDataTypeToHccl(ge::DT_INT8));
    EXPECT_EQ(HCCL_DATA_TYPE_INT16, GeDataTypeToHccl(ge::DT_INT16));
    EXPECT_EQ(HCCL_DATA_TYPE_INT32, GeDataTypeToHccl(ge::DT_INT32));
    EXPECT_EQ(HCCL_DATA_TYPE_INT64, GeDataTypeToHccl(ge::DT_INT64));
    EXPECT_EQ(HCCL_DATA_TYPE_UINT8, GeDataTypeToHccl(ge::DT_UINT8));
    EXPECT_EQ(HCCL_DATA_TYPE_UINT16, GeDataTypeToHccl(ge::DT_UINT16));
    EXPECT_EQ(HCCL_DATA_TYPE_UINT32, GeDataTypeToHccl(ge::DT_UINT32));
    EXPECT_EQ(HCCL_DATA_TYPE_UINT64, GeDataTypeToHccl(ge::DT_UINT64));
    EXPECT_EQ(HCCL_DATA_TYPE_FP64, GeDataTypeToHccl(ge::DT_DOUBLE));
    EXPECT_EQ(HCCL_DATA_TYPE_BFP16, GeDataTypeToHccl(ge::DT_BF16));
}

TEST(GeDataTypeToHcclTest, BoolMapsToInt8)
{
    // 旧路径 kConstOpHcclDataType 口径: 搬移类算子将 int8 当 bool
    EXPECT_EQ(HCCL_DATA_TYPE_INT8, GeDataTypeToHccl(ge::DT_BOOL));
}

TEST(GeDataTypeToHcclTest, Fp8Family)
{
    EXPECT_EQ(HCCL_DATA_TYPE_HIF8, GeDataTypeToHccl(ge::DT_HIFLOAT8));
    EXPECT_EQ(HCCL_DATA_TYPE_FP8E5M2, GeDataTypeToHccl(ge::DT_FLOAT8_E5M2));
    EXPECT_EQ(HCCL_DATA_TYPE_FP8E4M3, GeDataTypeToHccl(ge::DT_FLOAT8_E4M3FN));
    EXPECT_EQ(HCCL_DATA_TYPE_FP8E8M0, GeDataTypeToHccl(ge::DT_FLOAT8_E8M0));
}

TEST(GeDataTypeToHcclTest, UnmappedGoesReserved)
{
    EXPECT_EQ(HCCL_DATA_TYPE_RESERVED, GeDataTypeToHccl(ge::DT_STRING));
    EXPECT_EQ(HCCL_DATA_TYPE_RESERVED, GeDataTypeToHccl(ge::DT_UNDEFINED));
}

// ---------------------------------------------------------------------------
// GetHcclDataTypeSize: 尺寸表与映射表同源 (cross-check) + INT128
// ---------------------------------------------------------------------------
TEST(GetHcclDataTypeSizeTest, AllSizes)
{
    EXPECT_EQ(1U, GetHcclDataTypeSize(HCCL_DATA_TYPE_INT8));
    EXPECT_EQ(1U, GetHcclDataTypeSize(HCCL_DATA_TYPE_UINT8));
    EXPECT_EQ(2U, GetHcclDataTypeSize(HCCL_DATA_TYPE_INT16));
    EXPECT_EQ(2U, GetHcclDataTypeSize(HCCL_DATA_TYPE_UINT16));
    EXPECT_EQ(2U, GetHcclDataTypeSize(HCCL_DATA_TYPE_FP16));
    EXPECT_EQ(2U, GetHcclDataTypeSize(HCCL_DATA_TYPE_BFP16));
    EXPECT_EQ(4U, GetHcclDataTypeSize(HCCL_DATA_TYPE_INT32));
    EXPECT_EQ(4U, GetHcclDataTypeSize(HCCL_DATA_TYPE_UINT32));
    EXPECT_EQ(4U, GetHcclDataTypeSize(HCCL_DATA_TYPE_FP32));
    EXPECT_EQ(8U, GetHcclDataTypeSize(HCCL_DATA_TYPE_INT64));
    EXPECT_EQ(8U, GetHcclDataTypeSize(HCCL_DATA_TYPE_UINT64));
    EXPECT_EQ(8U, GetHcclDataTypeSize(HCCL_DATA_TYPE_FP64));
    EXPECT_EQ(16U, GetHcclDataTypeSize(HCCL_DATA_TYPE_INT128));
}

TEST(GetHcclDataTypeSizeTest, Fp8FamilyIsOneByte)
{
    EXPECT_EQ(1U, GetHcclDataTypeSize(HCCL_DATA_TYPE_HIF8));
    EXPECT_EQ(1U, GetHcclDataTypeSize(HCCL_DATA_TYPE_FP8E4M3));
    EXPECT_EQ(1U, GetHcclDataTypeSize(HCCL_DATA_TYPE_FP8E5M2));
    EXPECT_EQ(1U, GetHcclDataTypeSize(HCCL_DATA_TYPE_FP8E8M0));
}

TEST(GetHcclDataTypeSizeTest, EveryMappedTypeHasPositiveSize)
{
    // 不变量: 映射表产出的每个合法类型尺寸必为正 (映射表与尺寸表一致性的守卫测试)
    const ge::DataType mapped[]
        = {ge::DT_FLOAT, ge::DT_FLOAT16,  ge::DT_INT8,        ge::DT_INT16,         ge::DT_INT32,      ge::DT_INT64,
           ge::DT_UINT8, ge::DT_UINT16,   ge::DT_UINT32,      ge::DT_UINT64,        ge::DT_DOUBLE,     ge::DT_BF16,
           ge::DT_BOOL,  ge::DT_HIFLOAT8, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0};
    for (ge::DataType dt : mapped) {
        HcclDataType hcclType = GeDataTypeToHccl(dt);
        ASSERT_NE(HCCL_DATA_TYPE_RESERVED, hcclType);
        EXPECT_GT(GetHcclDataTypeSize(hcclType), 0U) << "type=" << static_cast<int>(hcclType);
    }
}

// ---------------------------------------------------------------------------
// StringToReduceOp
// ---------------------------------------------------------------------------
TEST(StringToReduceOpTest, AllSupportedOps)
{
    EXPECT_EQ(HCCL_REDUCE_SUM, StringToReduceOp("sum"));
    EXPECT_EQ(HCCL_REDUCE_PROD, StringToReduceOp("prod"));
    EXPECT_EQ(HCCL_REDUCE_MAX, StringToReduceOp("max"));
    EXPECT_EQ(HCCL_REDUCE_MIN, StringToReduceOp("min"));
}

TEST(StringToReduceOpTest, NullAndUnknownGoReserved)
{
    EXPECT_EQ(HCCL_REDUCE_RESERVED, StringToReduceOp(nullptr));
    EXPECT_EQ(HCCL_REDUCE_RESERVED, StringToReduceOp("avg"));
    EXPECT_EQ(HCCL_REDUCE_RESERVED, StringToReduceOp(""));
}

// ---------------------------------------------------------------------------
// CalcLoopMaxTime: 公式表 + 边界 + 受检窄化 (de995fd8)
// ---------------------------------------------------------------------------
TEST(CalcLoopMaxTimeTest, Group1AllReduce)
{
    uint32_t loop = 0;
    ASSERT_EQ(ge::GRAPH_SUCCESS, CalcLoopMaxTime(HCCL_KERNEL_OP_TYPE_ALLREDUCE, 100, 8, 400, 1, loop));
    EXPECT_EQ(2U, loop); // ceil(100*8/400)
}

TEST(CalcLoopMaxTimeTest, Group1ExactDivision)
{
    uint32_t loop = 0;
    ASSERT_EQ(ge::GRAPH_SUCCESS, CalcLoopMaxTime(HCCL_KERNEL_OP_TYPE_ALLREDUCE, 50, 8, 400, 1, loop));
    EXPECT_EQ(1U, loop);
}

TEST(CalcLoopMaxTimeTest, Group1SharedByBroadcastReduceSendReceive)
{
    const std::string ops[]
        = {HCCL_KERNEL_OP_TYPE_BROADCAST, HCCL_KERNEL_OP_TYPE_REDUCE, HCCL_KERNEL_OP_TYPE_SEND,
           HCCL_KERNEL_OP_TYPE_RECEIVE};
    for (const std::string& op : ops) {
        uint32_t loop = 0;
        ASSERT_EQ(ge::GRAPH_SUCCESS, CalcLoopMaxTime(op, 100, 8, 400, 1, loop));
        EXPECT_EQ(2U, loop) << "op=" << op;
    }
}

TEST(CalcLoopMaxTimeTest, Group2ReduceScatterAllGatherWithRankSize)
{
    // maxCountPerLoop = 400/(4*8) = 12, loop = ceil(100/12) = 9。
    // 注: InfoStore 的 ceil(count*u*R/C)=ceil(3200/400)=8 在非整除时偏小 (tag 不足), 本实现按
    // maxCountPerLoop 分母推导防不足 (见 hccl_op_utils.h 公式表注释), 故 9 为期望值
    uint32_t loop = 0;
    ASSERT_EQ(ge::GRAPH_SUCCESS, CalcLoopMaxTime(HCCL_KERNEL_OP_TYPE_REDUCESCATTER, 100, 8, 400, 4, loop));
    EXPECT_EQ(9U, loop);
    ASSERT_EQ(ge::GRAPH_SUCCESS, CalcLoopMaxTime(HCCL_KERNEL_OP_TYPE_ALLGATHER, 100, 8, 400, 4, loop));
    EXPECT_EQ(9U, loop);
}

TEST(CalcLoopMaxTimeTest, NonLoopOpsReturnZero)
{
    const std::string ops[]
        = {HCCL_KERNEL_OP_TYPE_ALLGATHERV, HCCL_KERNEL_OP_TYPE_REDUCESCATTERV, HCCL_KERNEL_OP_TYPE_ALLTOALL,
           HCCL_KERNEL_OP_TYPE_ALLTOALLV};
    for (const std::string& op : ops) {
        uint32_t loop = 123; // 非零哨兵: 验证会被覆写为 0
        ASSERT_EQ(ge::GRAPH_SUCCESS, CalcLoopMaxTime(op, 100, 8, 400, 4, loop));
        EXPECT_EQ(0U, loop) << "op=" << op;
    }
}

TEST(CalcLoopMaxTimeTest, ZeroUnitSizeFails)
{
    uint32_t loop = 0;
    EXPECT_EQ(ge::GRAPH_FAILED, CalcLoopMaxTime(HCCL_KERNEL_OP_TYPE_ALLREDUCE, 100, 0, 400, 1, loop));
}

TEST(CalcLoopMaxTimeTest, Group2ZeroRankSizeFails)
{
    uint32_t loop = 0;
    EXPECT_EQ(ge::GRAPH_FAILED, CalcLoopMaxTime(HCCL_KERNEL_OP_TYPE_REDUCESCATTER, 100, 8, 400, 0, loop));
}

TEST(CalcLoopMaxTimeTest, BufferCannotHoldOneElementFails)
{
    uint32_t loop = 0;
    EXPECT_EQ(ge::GRAPH_FAILED, CalcLoopMaxTime(HCCL_KERNEL_OP_TYPE_ALLREDUCE, 1, 8, 4, 1, loop));
}

TEST(CalcLoopMaxTimeTest, LoopCountOverUint32Fails)
{
    // 受检窄化 (de995fd8): 迭代数超 u32 值域显式报错而非静默截断
    uint32_t loop = 0;
    const uint64_t hugeCount = 1ULL << 33; // loop = hugeCount > UINT32_MAX
    EXPECT_EQ(ge::GRAPH_FAILED, CalcLoopMaxTime(HCCL_KERNEL_OP_TYPE_ALLREDUCE, hugeCount, 1, 1, 1, loop));
}

TEST(CalcLoopMaxTimeTest, LoopCountAtUint32BoundarySucceeds)
{
    uint32_t loop = 0;
    ASSERT_EQ(ge::GRAPH_SUCCESS, CalcLoopMaxTime(HCCL_KERNEL_OP_TYPE_ALLREDUCE, UINT32_MAX, 1, 1, 1, loop));
    EXPECT_EQ(UINT32_MAX, loop);
}

// ---------------------------------------------------------------------------
// CalcAlignedCount: RoundUp 公式 + unitSize==0 防御
// ---------------------------------------------------------------------------
TEST(CalcAlignedCountTest, RoundUpFormula)
{
    EXPECT_EQ(32U, CalcAlignedCount(100, 4, 32)); // RoundUp(100,32)=128 → /4
    EXPECT_EQ(4U, CalcAlignedCount(1, 8, 32));    // RoundUp(1,32)=32 → /8
    EXPECT_EQ(32U, CalcAlignedCount(128, 4, 32)); // 已对齐
    EXPECT_EQ(0U, CalcAlignedCount(0, 4, 32));
}

TEST(CalcAlignedCountTest, ZeroUnitSizeDefendsToZero) { EXPECT_EQ(0U, CalcAlignedCount(100, 0, 32)); }

// ---------------------------------------------------------------------------
// SummarizeVCounts: min 钳制 (bf4fff2e) + 截断提示
// ---------------------------------------------------------------------------
TEST(SummarizeVCountsTest, EqualLengthPairs)
{
    std::vector<uint64_t> counts = {1, 2, 3};
    std::vector<uint64_t> displs = {0, 5, 10};
    const std::string out = SummarizeVCounts(counts, displs);
    EXPECT_NE(std::string::npos, out.find("0:1/0"));
    EXPECT_NE(std::string::npos, out.find("1:2/5"));
    EXPECT_NE(std::string::npos, out.find("2:3/10"));
}

TEST(SummarizeVCountsTest, ShortDisplsClampedToMin)
{
    // displs 为可选输入 (缺失时短于 counts): 按两侧较短者为界, 不越界读
    std::vector<uint64_t> counts = {1, 2, 3, 4};
    std::vector<uint64_t> displs = {0, 5};
    const std::string out = SummarizeVCounts(counts, displs);
    EXPECT_NE(std::string::npos, out.find("0:1/0"));
    EXPECT_NE(std::string::npos, out.find("1:2/5"));
    EXPECT_EQ(std::string::npos, out.find("2:")); // 第三对不输出
}

TEST(SummarizeVCountsTest, EmptyVectorsGiveEmptyString)
{
    std::vector<uint64_t> counts;
    std::vector<uint64_t> displs;
    EXPECT_TRUE(SummarizeVCounts(counts, displs).empty());
}

TEST(SummarizeVCountsTest, OverEightRanksGetTotalHint)
{
    std::vector<uint64_t> counts(10, 7);
    std::vector<uint64_t> displs(10, 1);
    const std::string out = SummarizeVCounts(counts, displs);
    EXPECT_NE(std::string::npos, out.find("total 10 ranks"));
    EXPECT_EQ(std::string::npos, out.find("8:7")); // 第 9 对不逐条输出
}

// ---------------------------------------------------------------------------
// GetCCLBufferAvailableSize: 环境变量推导 + 范围检查
// ---------------------------------------------------------------------------
class GetCCLBufferAvailableSizeTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        // 隔离环境变量副作用
#ifdef _WIN32
        // N/A
#else
        unsetenv("HCCL_BUFFSIZE");
#endif
    }
    void TearDown() override { unsetenv("HCCL_BUFFSIZE"); }
};

TEST_F(GetCCLBufferAvailableSizeTest, DefaultIs200MbMinus1Mb)
{
    uint64_t size = 0;
    ASSERT_EQ(ge::GRAPH_SUCCESS, GetCCLBufferAvailableSize(size));
    EXPECT_EQ(199ULL * 1024 * 1024, size);
}

TEST_F(GetCCLBufferAvailableSizeTest, EnvValueUsedWhenValid)
{
    setenv("HCCL_BUFFSIZE", "100", 1);
    uint64_t size = 0;
    ASSERT_EQ(ge::GRAPH_SUCCESS, GetCCLBufferAvailableSize(size));
    EXPECT_EQ(99ULL * 1024 * 1024, size);

    setenv("HCCL_BUFFSIZE", "2048", 1);
    ASSERT_EQ(ge::GRAPH_SUCCESS, GetCCLBufferAvailableSize(size));
    EXPECT_EQ(2047ULL * 1024 * 1024, size);
}

TEST_F(GetCCLBufferAvailableSizeTest, InvalidEnvFails)
{
    const char* invalid[] = {"abc", "-5", "0", "1", "12x", "4294967296"};
    for (const char* v : invalid) {
        setenv("HCCL_BUFFSIZE", v, 1);
        uint64_t size = 0;
        EXPECT_EQ(ge::GRAPH_FAILED, GetCCLBufferAvailableSize(size)) << "HCCL_BUFFSIZE=" << v;
    }
}

// ---------------------------------------------------------------------------
// BuildTagVec: 桩 tag (UT_BASE_TAG) 的展开规则
// ---------------------------------------------------------------------------
TEST(BuildTagVecTest, ZeroLoopYieldsSingleBaseTag)
{
    std::vector<std::string> tags;
    ASSERT_EQ(ge::GRAPH_SUCCESS, BuildTagVec(HCCL_KERNEL_OP_TYPE_ALLREDUCE, nullptr, 0, tags));
    ASSERT_EQ(1U, tags.size());
    EXPECT_EQ("UT_BASE_TAG", tags[0]);
}

TEST(BuildTagVecTest, LoopSuffixPerIteration)
{
    std::vector<std::string> tags;
    ASSERT_EQ(ge::GRAPH_SUCCESS, BuildTagVec(HCCL_KERNEL_OP_TYPE_ALLREDUCE, "grp", 3, tags));
    ASSERT_EQ(3U, tags.size());
    EXPECT_EQ("UT_BASE_TAG_looptime_0", tags[0]);
    EXPECT_EQ("UT_BASE_TAG_looptime_1", tags[1]);
    EXPECT_EQ("UT_BASE_TAG_looptime_2", tags[2]);
}

TEST(BuildTagVecTest, OneLoopYieldsTwoTags)
{
    std::vector<std::string> tags;
    ASSERT_EQ(ge::GRAPH_SUCCESS, BuildTagVec(HCCL_KERNEL_OP_TYPE_SEND, "grp", 1, tags));
    ASSERT_EQ(1U, tags.size());
    EXPECT_EQ("UT_BASE_TAG_looptime_0", tags[0]);
}

// ---------------------------------------------------------------------------
// BuildSendRecvBaseTag: P2P baseTag 格式
// ---------------------------------------------------------------------------
TEST(BuildSendRecvBaseTagTest, Format)
{
    EXPECT_EQ("SendRecv_5_1_2", BuildSendRecvBaseTag(5, 1, 2));
    EXPECT_EQ("SendRecv_0_0_7", BuildSendRecvBaseTag(0, 0, 7));
}

// ---------------------------------------------------------------------------
// CrackCleaner::Clean: 纯校验路径 (bf4fff2e/b78313c9 — ctx 不触达)
// ---------------------------------------------------------------------------
TEST(CrackCleanerTest, LengthMismatchFails)
{
    HcclOpState st;
    CrackCleaner cleaner;
    std::vector<void*> addrs = {reinterpret_cast<void*>(0x1000)};
    std::vector<uint64_t> sizes; // 长度不一致 → GRAPH_FAILED
    EXPECT_EQ(ge::GRAPH_FAILED, cleaner.Clean(st, addrs, sizes));
}

TEST(CrackCleanerTest, BothEmptySucceeds)
{
    HcclOpState st;
    CrackCleaner cleaner;
    std::vector<void*> addrs;
    std::vector<uint64_t> sizes;
    EXPECT_EQ(ge::GRAPH_SUCCESS, cleaner.Clean(st, addrs, sizes));
}

// ---------------------------------------------------------------------------
// HcclOpState 缺省值 (含 aivCoreLimit 缺省 56, MAX_NUM_BLOCKS_DEFAULT)
// ---------------------------------------------------------------------------
TEST(HcclOpStateTest, Defaults)
{
    HcclOpState st;
    EXPECT_EQ(MAX_NUM_BLOCKS_DEFAULT, st.aivCoreLimit);
    EXPECT_EQ(56U, st.aivCoreLimit);
    EXPECT_FALSE(st.isDynamicShape);
    EXPECT_FALSE(st.needRefresh);
    EXPECT_FALSE(st.needCleanSeparately);
    EXPECT_TRUE(st.crackOffsets.empty());
    EXPECT_TRUE(st.sendCountMatrix.empty());
    EXPECT_EQ(0U, st.totalElemCount);
}
