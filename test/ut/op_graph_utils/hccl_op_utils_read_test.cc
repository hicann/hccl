/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// ReadU64 系 (gert::Tensor 依赖路径) UT: 真实构造 gert::Tensor (toolkit 内联构造器),
// host 直读与 device D2H 两条分支的契约全路径
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "exe_graph/runtime/runtime_tensor.h"
#include "hccl_op_utils.h"
#include "hccl_custom_op.h"

using namespace hccl;
using gert::StorageFormat;
using gert::StorageShape;
using gert::Tensor;
using gert::TensorPlacement;

namespace {
// 构造 host 放置的张量: shape/dtype/缓冲区三要素真实 (StorageShape 吃 initializer_list)
Tensor MakeHostTensor(std::initializer_list<int64_t> shape, ge::DataType dtype, const void* buf)
{
    return Tensor(
        StorageShape(shape, shape), StorageFormat(), gert::kOnHost, dtype,
        reinterpret_cast<gert::TensorAddress>(const_cast<void*>(buf)));
}

Tensor MakeDeviceTensor(std::initializer_list<int64_t> shape, ge::DataType dtype, const void* buf)
{
    // UT 环境 device 内存不可用: 以 host 缓冲地址挂 device 放置, D2H 由 op_deps_stub 的
    // 真拷贝实现搬运 — 被测代码路径 (placement 分派 + 拷贝 + 字节转换) 与真机一致
    return Tensor(
        StorageShape(shape, shape), StorageFormat(), gert::kOnDeviceHbm, dtype,
        reinterpret_cast<gert::TensorAddress>(const_cast<void*>(buf)));
}
} // namespace

// ---------------------------------------------------------------------------
// ReadHostU64Vector: host 直读契约 (InferShape 编译期场景)
// ---------------------------------------------------------------------------
TEST(ReadHostU64VectorTest, Int64HostTensor)
{
    const int64_t data[] = {10, 20, 30};
    Tensor t = MakeHostTensor({3}, ge::DT_INT64, data);
    std::vector<uint64_t> values;
    ASSERT_EQ(ge::GRAPH_SUCCESS, ReadHostU64Vector(&t, values));
    ASSERT_EQ(3U, values.size());
    EXPECT_EQ(10U, values[0]);
    EXPECT_EQ(20U, values[1]);
    EXPECT_EQ(30U, values[2]);
}

TEST(ReadHostU64VectorTest, Int32HostTensor)
{
    const int32_t data[] = {1, 2, 3, 4};
    Tensor t = MakeHostTensor({4}, ge::DT_INT32, data);
    std::vector<uint64_t> values;
    ASSERT_EQ(ge::GRAPH_SUCCESS, ReadHostU64Vector(&t, values));
    ASSERT_EQ(4U, values.size());
    EXPECT_EQ(3U, values[2]);
}

TEST(ReadHostU64VectorTest, NegativeValueIntercepted)
{
    // counts/displacements 语义非负: 负值 (符号扩展成巨大 u64 参与乘除) 显式拦截
    const int64_t data[] = {10, -1};
    Tensor t = MakeHostTensor({2}, ge::DT_INT64, data);
    std::vector<uint64_t> values;
    EXPECT_EQ(ge::GRAPH_FAILED, ReadHostU64Vector(&t, values));

    const int32_t data32[] = {-5};
    Tensor t32 = MakeHostTensor({1}, ge::DT_INT32, data32);
    EXPECT_EQ(ge::GRAPH_FAILED, ReadHostU64Vector(&t32, values));
}

TEST(ReadHostU64VectorTest, UnsupportedDtypeFails)
{
    const float data[] = {1.0f, 2.0f};
    Tensor t = MakeHostTensor({2}, ge::DT_FLOAT, data);
    std::vector<uint64_t> values;
    EXPECT_EQ(ge::GRAPH_FAILED, ReadHostU64Vector(&t, values));
}

TEST(ReadHostU64VectorTest, NullTensorFails)
{
    std::vector<uint64_t> values;
    EXPECT_EQ(ge::GRAPH_FAILED, ReadHostU64Vector(nullptr, values));
}

TEST(ReadHostU64VectorTest, NullAddrFails)
{
    Tensor t = MakeHostTensor({3}, ge::DT_INT64, nullptr);
    std::vector<uint64_t> values;
    EXPECT_EQ(ge::GRAPH_FAILED, ReadHostU64Vector(&t, values));
}

TEST(ReadHostU64VectorTest, EmptyShapeWithNullAddrFails)
{
    // 契约 (hccl_op_utils.h): 输入侧 counts/displs 张量 0 元素/空地址属喂参异常, 显式报错
    Tensor t = MakeHostTensor({0}, ge::DT_INT64, nullptr);
    std::vector<uint64_t> values;
    EXPECT_EQ(ge::GRAPH_FAILED, ReadHostU64Vector(&t, values));
}

TEST(ReadHostU64VectorTest, ShapeDrivesLengthNotBuffer)
{
    // 契约: 读取长度以 shape 声明为准 — 缓冲区大于 shape 时只读 shape 声明的元素
    const int64_t bigBuf[64] = {1, 2, 3, 4, 5, 6, 7, 8};
    Tensor t = MakeHostTensor({2}, ge::DT_INT64, bigBuf);
    std::vector<uint64_t> values;
    ASSERT_EQ(ge::GRAPH_SUCCESS, ReadHostU64Vector(&t, values));
    ASSERT_EQ(2U, values.size());
    EXPECT_EQ(2U, values[1]);
}

// ---------------------------------------------------------------------------
// ReadU64VectorWithD2H: 运行期双分支 (host 直读 / device 同步 D2H)
// ---------------------------------------------------------------------------
TEST(ReadU64VectorWithD2HTest, HostPlacementReadsDirectly)
{
    const int64_t data[] = {11, 22};
    Tensor t = MakeHostTensor({2}, ge::DT_INT64, data);
    std::vector<uint64_t> values;
    ASSERT_EQ(ge::GRAPH_SUCCESS, ReadU64VectorWithD2H(&t, values, nullptr)); // 无流: host 分支无需同步
    ASSERT_EQ(2U, values.size());
    EXPECT_EQ(22U, values[1]);
}

TEST(ReadU64VectorWithD2HTest, DevicePlacementGoesD2H)
{
    const int64_t data[] = {100, 200, 300};
    Tensor t = MakeDeviceTensor({3}, ge::DT_INT64, data);
    std::vector<uint64_t> values;
    // stream=null → SyncStreamBeforeD2H no-op (stub 的 aclrtSynchronizeStream 恒成功)
    ASSERT_EQ(ge::GRAPH_SUCCESS, ReadU64VectorWithD2H(&t, values, nullptr));
    ASSERT_EQ(3U, values.size());
    EXPECT_EQ(100U, values[0]);
    EXPECT_EQ(300U, values[2]);
}

TEST(ReadU64VectorWithD2HTest, DeviceInt32NegativeIntercepted)
{
    const int32_t data[] = {5, -7};
    Tensor t = MakeDeviceTensor({2}, ge::DT_INT32, data);
    std::vector<uint64_t> values;
    EXPECT_EQ(ge::GRAPH_FAILED, ReadU64VectorWithD2H(&t, values, nullptr));
}

TEST(ReadU64VectorWithD2HTest, DeviceEmptyTensorWithNullAddrFails)
{
    // 同输入侧契约: 空张量配空地址显式报错, 不发起 0 字节 D2H
    Tensor t = MakeDeviceTensor({0}, ge::DT_INT64, nullptr);
    std::vector<uint64_t> values;
    EXPECT_EQ(ge::GRAPH_FAILED, ReadU64VectorWithD2H(&t, values, nullptr));
}

TEST(ReadU64VectorWithD2HTest, NullTensorFails)
{
    std::vector<uint64_t> values;
    EXPECT_EQ(ge::GRAPH_FAILED, ReadU64VectorWithD2H(nullptr, values, nullptr));
}

// ---------------------------------------------------------------------------
// SyncStreamBeforeD2H
// ---------------------------------------------------------------------------
TEST(SyncStreamBeforeD2HTest, NullStreamIsNoOp) { EXPECT_EQ(ge::GRAPH_SUCCESS, SyncStreamBeforeD2H(nullptr)); }

// ---------------------------------------------------------------------------
// ReadOptionalVDispls: 可选 displs 的三态语义
// ---------------------------------------------------------------------------
TEST(ReadOptionalVDisplsTest, AbsentMarksFallback)
{
    // 未接入 displs 输入 → 三态标记 hasDispls=false (前缀和回退由 ResolveVDispls 消费该标记)
    std::vector<uint64_t> rawDispls = {9};
    bool hasDispls = true; // 哨兵: 应回落为 false
    ASSERT_EQ(ge::GRAPH_SUCCESS, ReadOptionalVDispls(nullptr, "ut", "displs", 4, rawDispls, hasDispls, nullptr));
    EXPECT_FALSE(hasDispls);
}

TEST(ResolveVDisplsTest, AbsentBuildsPrefixSumFromCounts)
{
    // hasDispls=false → 按 counts 前缀和: {2,3,4} → {0,2,5}
    const std::vector<uint64_t> counts = {2, 3, 4};
    std::vector<uint64_t> rawDispls;
    std::vector<uint64_t> displs;
    ResolveVDispls(false, rawDispls, counts, displs);
    ASSERT_EQ(3U, displs.size());
    EXPECT_EQ(0U, displs[0]);
    EXPECT_EQ(2U, displs[1]);
    EXPECT_EQ(5U, displs[2]);
}

TEST(ResolveVDisplsTest, PresentTakesRawDirectly)
{
    const std::vector<uint64_t> counts = {2, 3, 4};
    const std::vector<uint64_t> raw = {0, 10, 20};
    std::vector<uint64_t> displs;
    ResolveVDispls(true, raw, counts, displs);
    ASSERT_EQ(3U, displs.size());
    EXPECT_EQ(20U, displs[2]);
}

TEST(ReadOptionalVDisplsTest, SizeMismatchFallsBack)
{
    // 张量存在但长度 != rankSize: HCCL_WARNING + 前缀和回退
    const uint64_t raw[] = {0, 5};
    Tensor displsTensor = MakeHostTensor({2}, ge::DT_INT64, raw);
    const uint64_t counts[] = {1, 1, 1};
    Tensor countsTensor = MakeHostTensor({3}, ge::DT_INT64, counts);
    std::vector<uint64_t> rawDispls;
    bool hasDispls = true;
    ASSERT_EQ(ge::GRAPH_SUCCESS, ReadOptionalVDispls(&displsTensor, "ut", "displs", 3, rawDispls, hasDispls, nullptr));
    EXPECT_FALSE(hasDispls);
    // 三态语义: 标记回落后 raw 读取值保留原样 (调用方经 ResolveVDispls 按前缀和重建, 丢弃 raw)
    ASSERT_EQ(2U, rawDispls.size());
    EXPECT_EQ(5U, rawDispls[1]);
    (void)countsTensor;
}

TEST(ReadOptionalVDisplsTest, ExactSizeTakesDirect)
{
    const uint64_t raw[] = {0, 10, 20};
    Tensor displsTensor = MakeHostTensor({3}, ge::DT_INT64, raw);
    std::vector<uint64_t> rawDispls;
    bool hasDispls = false;
    ASSERT_EQ(ge::GRAPH_SUCCESS, ReadOptionalVDispls(&displsTensor, "ut", "displs", 3, rawDispls, hasDispls, nullptr));
    EXPECT_TRUE(hasDispls);
    ASSERT_EQ(3U, rawDispls.size());
    EXPECT_EQ(20U, rawDispls[2]);
}

// ---------------------------------------------------------------------------
// SyncStreamBeforeD2H 失败分支 (检视复核意见: stub 恒 SUCCESS, 失败路径无用例)
// ---------------------------------------------------------------------------
extern aclError g_stubSyncStreamRet;

class SyncStreamFailTest : public ::testing::Test {
protected:
    void TearDown() override { g_stubSyncStreamRet = ACL_SUCCESS; }
};

TEST_F(SyncStreamFailTest, SyncFailureReturnsGraphFailed)
{
    aclrtStream fakeStream = reinterpret_cast<aclrtStream>(0x1000);
    g_stubSyncStreamRet = ACL_ERROR_RT_FAILURE; // 注入同步失败
    EXPECT_EQ(ge::GRAPH_FAILED, SyncStreamBeforeD2H(fakeStream));
}

TEST_F(SyncStreamFailTest, SyncSuccessReturnsGraphSuccess)
{
    aclrtStream fakeStream = reinterpret_cast<aclrtStream>(0x1000);
    g_stubSyncStreamRet = ACL_SUCCESS;
    EXPECT_EQ(ge::GRAPH_SUCCESS, SyncStreamBeforeD2H(fakeStream));
}
