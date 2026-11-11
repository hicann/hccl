/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "alg_param.h"

namespace ops_hccl {
namespace {

    using AlgCtx = AlgResourceCtxSerializable;

    constexpr uint32_t USER_RANK = 5;
    constexpr uint32_t USER_RANK_SIZE = 8;

    AlgCtx MakeContext()
    {
        AlgCtx ctx;
        ctx.notifyNumOnMainThread = 3;
        ctx.slaveThreadNum = 2;
        ctx.waitTimeout = 100;
        ctx.fullTimeout = 200;
        ctx.notifyNumPerThread = {1, 2};
        ctx.topoInfo.userRank = USER_RANK;
        ctx.topoInfo.userRankSize = USER_RANK_SIZE;
        ctx.topoInfo.topoLevelNums = 1;
        // 内容级校验依赖的拓扑不变式：netLayerNum 与实际层数一致，且 topoLevelNums 不超过它。
        ctx.topoInfo.netLayerDetails.netLayerNum = 1;
        ctx.topoInfo.netLayerDetails.netLayers = {0};
        return ctx;
    }

    void WriteTopoInfoSeqSize(std::vector<char>& data, size_t headerLen, uint32_t value)
    {
        ASSERT_GE(headerLen, sizeof(value));
        std::memcpy(data.data() + headerLen - sizeof(value), &value, sizeof(value));
    }

    TEST(AlgParamDeserializeTest, ValidContextRoundTrip)
    {
        AlgCtx ctx = MakeContext();
        std::vector<char> data = ctx.Serialize();
        ASSERT_FALSE(data.empty());
        ASSERT_GT(ctx.topoInfoSeqSize, 0u);

        AlgCtx out;
        out.DeSerialize(data);
        EXPECT_TRUE(out.deserializeValid);
        EXPECT_EQ(out.notifyNumOnMainThread, ctx.notifyNumOnMainThread);
        EXPECT_EQ(out.slaveThreadNum, ctx.slaveThreadNum);
        EXPECT_EQ(out.waitTimeout, ctx.waitTimeout);
        EXPECT_EQ(out.fullTimeout, ctx.fullTimeout);
        EXPECT_EQ(out.notifyNumPerThread, ctx.notifyNumPerThread);
        EXPECT_EQ(out.topoInfo.userRank, USER_RANK);
        EXPECT_EQ(out.topoInfo.userRankSize, USER_RANK_SIZE);
        EXPECT_EQ(out.topoInfo.topoLevelNums, 1u);
    }

    // 长度包络完全合法、仅内容被破坏（内存位翻转/部分覆盖）的用例：旧实现会静默半解析并置
    // deserializeValid=true。零化定长字段 userRankSize 不改变任何长度字段，只能靠内容级校验发现。
    TEST(AlgParamDeserializeTest, ZeroedUserRankSizeContentRejected)
    {
        AlgCtx ctx = MakeContext();
        std::vector<char> data = ctx.Serialize();
        const size_t tailStart = data.size() - ctx.topoInfoSeqSize;
        // topoInfo 定长字段按 VisitFields 声明顺序写入，userRank(u32) 之后紧邻 userRankSize(u32)。
        const size_t userRankSizeOffset = sizeof(uint32_t);
        ASSERT_LE(tailStart + userRankSizeOffset + sizeof(uint32_t), data.size());

        const uint32_t zero = 0;
        std::memcpy(data.data() + tailStart + userRankSizeOffset, &zero, sizeof(zero));

        AlgCtx out;
        out.DeSerialize(data);
        EXPECT_FALSE(out.deserializeValid);
        // 失败时 topoInfo 被重置为默认值，调用方不会拿到半解析结果。
        EXPECT_EQ(out.topoInfo.userRank, 0u);
        EXPECT_EQ(out.topoInfo.userRankSize, 0u);
    }

    // physicalLevelNum 位于 topoInfo 段末尾（physicalLevels 为空时即最后 4 字节）。破坏为超过
    // 上界的值后总长度不变，旧实现仅告警丢弃该段并让外层置 deserializeValid=true。
    TEST(AlgParamDeserializeTest, ImplausiblePhysicalLevelNumRejected)
    {
        AlgCtx ctx = MakeContext();
        ASSERT_TRUE(ctx.topoInfo.physicalLevels.empty());
        std::vector<char> data = ctx.Serialize();
        ASSERT_GE(ctx.topoInfoSeqSize, sizeof(uint32_t));

        const uint32_t badPhysicalLevelNum = PHYSICAL_LEVEL_NUM_LIMIT + 1;
        std::memcpy(
            data.data() + data.size() - sizeof(badPhysicalLevelNum), &badPhysicalLevelNum, sizeof(badPhysicalLevelNum));

        AlgCtx out;
        out.DeSerialize(data);
        EXPECT_FALSE(out.deserializeValid);
    }

    TEST(AlgParamDeserializeTest, TopoInfoDeSerializeRequiresExactConsumption)
    {
        AlgCtx ctx = MakeContext();
        std::vector<char> seq = ctx.topoInfo.Serialize();

        TopoInfoWithNetLayerDetails parsed;
        EXPECT_TRUE(parsed.DeSerialize(seq));
        EXPECT_EQ(parsed.userRank, USER_RANK);
        EXPECT_EQ(parsed.userRankSize, USER_RANK_SIZE);

        seq.push_back('\0');
        TopoInfoWithNetLayerDetails withTrailing;
        EXPECT_FALSE(withTrailing.DeSerialize(seq));
    }

    TEST(AlgParamDeserializeTest, ZeroTopoInfoSeqSizeRejected)
    {
        AlgCtx ctx = MakeContext();
        std::vector<char> data = ctx.Serialize();
        const size_t headerLen = data.size() - ctx.topoInfoSeqSize;
        WriteTopoInfoSeqSize(data, headerLen, 0);

        AlgCtx out;
        out.DeSerialize(data);
        EXPECT_FALSE(out.deserializeValid);
        // 失败时 topoInfo 被重置为默认值，调用方不会拿到半解析结果。
        EXPECT_EQ(out.topoInfo.userRank, 0u);
        EXPECT_EQ(out.topoInfo.userRankSize, 0u);
        EXPECT_EQ(out.topoInfo.topoLevelNums, 0u);
    }

    TEST(AlgParamDeserializeTest, OversizedTopoInfoSeqSizeRejected)
    {
        AlgCtx ctx = MakeContext();
        std::vector<char> data = ctx.Serialize();
        const size_t headerLen = data.size() - ctx.topoInfoSeqSize;
        WriteTopoInfoSeqSize(data, headerLen, static_cast<uint32_t>(data.size()));

        AlgCtx out;
        out.DeSerialize(data);
        EXPECT_FALSE(out.deserializeValid);
    }

    TEST(AlgParamDeserializeTest, TruncatedTopoInfoRejected)
    {
        AlgCtx ctx = MakeContext();
        std::vector<char> data = ctx.Serialize();
        data.pop_back();

        AlgCtx out;
        out.DeSerialize(data);
        EXPECT_FALSE(out.deserializeValid);
    }

    TEST(AlgParamDeserializeTest, TrailingBytesRejected)
    {
        AlgCtx ctx = MakeContext();
        std::vector<char> data = ctx.Serialize();
        data.push_back('\0');

        AlgCtx out;
        out.DeSerialize(data);
        EXPECT_FALSE(out.deserializeValid);
    }

    TEST(AlgParamDeserializeTest, TruncatedHeaderRejected)
    {
        AlgCtx ctx = MakeContext();
        std::vector<char> data = ctx.Serialize();
        ASSERT_GT(data.size(), 4u);
        std::vector<char> truncated(data.begin(), data.begin() + 4);

        AlgCtx out;
        out.DeSerialize(truncated);
        EXPECT_FALSE(out.deserializeValid);
    }

    TEST(BinaryStreamTest, StringTruncatedLength)
    {
        std::vector<char> buffer(3, 0);
        BinaryStream stream(buffer);
        std::string value = "abc";

        stream >> value;
        EXPECT_TRUE(value.empty());
    }

    TEST(BinaryStreamTest, StringTruncatedPayload)
    {
        std::vector<char> buffer;
        const size_t size = 4;
        buffer.resize(sizeof(size));
        std::memcpy(buffer.data(), &size, sizeof(size));
        buffer.push_back('a');
        buffer.push_back('b');

        BinaryStream stream(buffer);
        std::string value;
        stream >> value;
        EXPECT_TRUE(value.empty());
    }

    TEST(BinaryStreamTest, VectorTruncatedLength)
    {
        std::vector<char> buffer(3, 0);
        BinaryStream stream(buffer);
        std::vector<uint32_t> value = {1, 2};

        stream >> value;
        EXPECT_TRUE(value.empty());
    }

    TEST(BinaryStreamTest, VectorTruncatedPayload)
    {
        std::vector<char> buffer;
        const size_t size = 2;
        buffer.resize(sizeof(size));
        std::memcpy(buffer.data(), &size, sizeof(size));
        const uint32_t firstElement = 7;
        buffer.resize(sizeof(size) + sizeof(firstElement));
        std::memcpy(buffer.data() + sizeof(size), &firstElement, sizeof(firstElement));

        BinaryStream stream(buffer);
        std::vector<uint32_t> value;
        stream >> value;
        ASSERT_EQ(value.size(), 2u);
        EXPECT_EQ(value[0], firstElement);
        EXPECT_EQ(value[1], 0u);
        EXPECT_EQ(stream.GetReadPos(), BinaryStream::INVALID_POS);
    }

    TEST(BinaryStreamTest, StringCorruptedHugeLengthRejected)
    {
        std::vector<char> buffer(sizeof(size_t), 0);
        const size_t hugeSize = std::numeric_limits<size_t>::max();
        std::memcpy(buffer.data(), &hugeSize, sizeof(hugeSize));

        BinaryStream stream(buffer);
        std::string value = "keep";
        stream >> value;
        EXPECT_TRUE(value.empty());                                // 长度不可信时清空，不按脏长度分配
        EXPECT_EQ(stream.GetReadPos(), BinaryStream::INVALID_POS); // 置 failbit，调用方据此判错
    }

    TEST(BinaryStreamTest, VectorCorruptedHugeLengthRejected)
    {
        std::vector<char> buffer(sizeof(size_t), 0);
        const size_t hugeSize = std::numeric_limits<size_t>::max();
        std::memcpy(buffer.data(), &hugeSize, sizeof(hugeSize));

        BinaryStream stream(buffer);
        std::vector<uint32_t> value = {1, 2};
        stream >> value;
        EXPECT_TRUE(value.empty());
        EXPECT_EQ(stream.GetReadPos(), BinaryStream::INVALID_POS);
    }

    // 损坏的长度字段位于 DeSerialize 读取的第一个变长字段（algHierarchyInfo.infos）：
    // 未加长度上界时 resize(极大值) 会抛 length_error/bad_alloc，加了校验后应返回错误标记而非异常。
    TEST(AlgParamDeserializeTest, CorruptedVectorLengthRejectedWithoutThrow)
    {
        AlgCtx ctx = MakeContext();
        std::vector<char> data = ctx.Serialize();
        ASSERT_GT(data.size(), sizeof(AlgType) + sizeof(size_t));
        const size_t hugeSize = std::numeric_limits<size_t>::max();
        std::memcpy(data.data() + sizeof(AlgType), &hugeSize, sizeof(hugeSize));

        AlgCtx out;
        out.DeSerialize(data);
        EXPECT_FALSE(out.deserializeValid);
        EXPECT_EQ(out.topoInfo.userRank, 0u);
    }

    TEST(BinaryStreamTest, MapTruncatedLength)
    {
        std::vector<char> buffer(3, 0);
        BinaryStream stream(buffer);
        std::map<uint32_t, uint32_t> value = {{1, 2}};

        stream >> value;
        EXPECT_EQ(value.size(), 1u);
    }

    TEST(BinaryStreamTest, MapTruncatedPayload)
    {
        std::vector<char> buffer;
        const size_t size = 1;
        buffer.resize(sizeof(size));
        std::memcpy(buffer.data(), &size, sizeof(size));
        const uint32_t key = 9;
        buffer.resize(sizeof(size) + sizeof(key));
        std::memcpy(buffer.data() + sizeof(size), &key, sizeof(key));

        BinaryStream stream(buffer);
        std::map<uint32_t, uint32_t> value;
        stream >> value;
        EXPECT_TRUE(value.empty());
    }

    TEST(BinaryStreamTest, ReadPosTracksConsumedBytes)
    {
        AlgCtx ctx = MakeContext();
        std::vector<char> data = ctx.Serialize();
        ASSERT_GT(data.size(), sizeof(uint32_t));

        BinaryStream stream(data);
        EXPECT_EQ(stream.GetReadPos(), 0u);
        uint32_t firstField = 0;
        stream >> firstField;
        EXPECT_EQ(stream.GetReadPos(), sizeof(firstField));
    }

    TEST(BinaryStreamTest, ReadPosInvalidAfterFailedRead)
    {
        std::vector<char> buffer(2, 0);
        BinaryStream stream(buffer);
        uint32_t value = 0;

        stream >> value;
        EXPECT_EQ(stream.GetReadPos(), BinaryStream::INVALID_POS);
    }

} // namespace
} // namespace ops_hccl
