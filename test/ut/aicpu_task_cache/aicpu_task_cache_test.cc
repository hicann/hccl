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
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>
#include "aicpu_task_cache_policy.h"
#include "aicpu_task_cache_key.h"
#include "aicpu_task_cache_utils.h"
#include "hccl_sym_win_dl.h"

namespace {
// 仅替代窗口查询边界; 被测地址收集和回退逻辑仍使用生产实现。
struct CacheTestSymmetricWindow {
    std::vector<uint8_t*> peerBases;
    uint32_t failedRank = UINT32_MAX;
    uint32_t queryCount = 0;
};

thread_local bool g_symRemoteAddrSupported = true;
thread_local bool g_symPeerPointerSupported = false;
} // namespace

extern "C" bool HcommIsSupportHcclSymWinGetRemoteAddr(void) { return g_symRemoteAddrSupported; }

extern "C" bool HcommIsSupportHcclSymWinGetPeerPointer(void) { return g_symPeerPointerSupported; }

extern "C" HcclResult GetSymWinRemoteMem(HcclCommSymWindow handle, size_t offset, uint32_t rank, void** ptr)
{
    auto* window = static_cast<CacheTestSymmetricWindow*>(handle);
    if (window == nullptr || ptr == nullptr || rank >= window->peerBases.size()) {
        return HCCL_E_PARA;
    }
    ++window->queryCount;
    if (rank == window->failedRank) {
        return HCCL_E_NOT_SUPPORT;
    }
    *ptr = window->peerBases[rank] + offset;
    return HCCL_SUCCESS;
}

namespace ops_hccl {
namespace {

    TEST(AicpuTaskCacheKeyTest, VTypesShareKeyAcrossDataTypesAndInputSizes)
    {
        for (auto opType : {HcclCMDType::HCCL_CMD_ALLTOALLV, HcclCMDType::HCCL_CMD_ALLTOALLVC}) {
            OpParam param{};
            param.opType = opType;
            param.all2AllVDataDes.sendType = HCCL_DATA_TYPE_UINT8;
            std::string byteTag;
            ASSERT_EQ(AicpuTaskCacheKey::GetAicpuTaskCacheTag(param, 3, 0, 0, byteTag), HCCL_SUCCESS);
            for (auto dataType : {HCCL_DATA_TYPE_UINT32, HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_FP64}) {
                param.all2AllVDataDes.sendType = dataType;
                std::string tag;
                ASSERT_EQ(AicpuTaskCacheKey::GetAicpuTaskCacheTag(param, 24, 0, 0, tag), HCCL_SUCCESS);
                EXPECT_EQ(tag, byteTag);
            }
        }
    }

    TEST(AicpuTaskCacheKeyTest, NonVTypeStillDistinguishesDataTypes)
    {
        OpParam param{};
        param.opType = HcclCMDType::HCCL_CMD_ALLTOALL;
        param.all2AllDataDes.sendType = HCCL_DATA_TYPE_UINT8;
        std::string byteTag;
        ASSERT_EQ(AicpuTaskCacheKey::GetAicpuTaskCacheTag(param, 24, 0, 0, byteTag), HCCL_SUCCESS);
        param.all2AllDataDes.sendType = HCCL_DATA_TYPE_UINT32;
        std::string tag;
        ASSERT_EQ(AicpuTaskCacheKey::GetAicpuTaskCacheTag(param, 24, 0, 0, tag), HCCL_SUCCESS);
        EXPECT_NE(tag, byteTag);
    }

    TEST(AicpuTaskCacheKeyTest, InplaceOverlapFeatureDistinguishesTags)
    {
        OpParam param{};
        param.opType = HcclCMDType::HCCL_CMD_ALLGATHER;
        param.DataDes.dataType = HCCL_DATA_TYPE_UINT32;
        const uint64_t inputSize = 1024;

        std::string outPlaceTag;
        ASSERT_EQ(AicpuTaskCacheKey::GetAicpuTaskCacheTag(param, inputSize, 0, 0, outPlaceTag), HCCL_SUCCESS);
        std::string identicalTag;
        ASSERT_EQ(AicpuTaskCacheKey::GetAicpuTaskCacheTag(param, inputSize, 0, inputSize, identicalTag), HCCL_SUCCESS);
        std::string partialTag;
        ASSERT_EQ(AicpuTaskCacheKey::GetAicpuTaskCacheTag(param, inputSize, 512, 512, partialTag), HCCL_SUCCESS);

        // 重叠形态不同的调用不得共享缓存(tag互异), 相同特征tag确定
        EXPECT_NE(identicalTag, outPlaceTag);
        EXPECT_NE(partialTag, outPlaceTag);
        EXPECT_NE(partialTag, identicalTag);
        std::string repeatTag;
        ASSERT_EQ(AicpuTaskCacheKey::GetAicpuTaskCacheTag(param, inputSize, 0, inputSize, repeatTag), HCCL_SUCCESS);
        EXPECT_EQ(repeatTag, identicalTag);
    }

    TEST(AicpuTaskCacheKeyTest, CaptureStateDistinguishesTagsIndependentlyOfOpMode)
    {
        OpParam param{};
        param.opType = HcclCMDType::HCCL_CMD_ALLGATHER;
        param.DataDes.dataType = HCCL_DATA_TYPE_UINT32;
        for (auto mode : {OpMode::OPBASE, OpMode::OFFLOAD}) {
            for (bool symmetric : {false, true}) {
                param.opMode = mode;
                param.supportSymmetricMemory = symmetric;
                param.isCapture = false;
                std::string normalTag;
                ASSERT_EQ(AicpuTaskCacheKey::GetAicpuTaskCacheTag(param, 1024, 0, 0, normalTag), HCCL_SUCCESS);
                param.isCapture = true;
                std::string captureTag;
                ASSERT_EQ(AicpuTaskCacheKey::GetAicpuTaskCacheTag(param, 1024, 0, 0, captureTag), HCCL_SUCCESS);
                EXPECT_NE(captureTag, normalTag);
                param.isCapture = false;
                std::string repeatTag;
                ASSERT_EQ(AicpuTaskCacheKey::GetAicpuTaskCacheTag(param, 1024, 0, 0, repeatTag), HCCL_SUCCESS);
                EXPECT_EQ(repeatTag, normalTag);
            }
        }
    }

    TEST(AicpuTaskCacheKeyTest, SignedOverlapOffsetsHaveDistinctAndExactEncoding)
    {
        OpParam param{};
        param.opType = HcclCMDType::HCCL_CMD_ALLGATHER;
        param.DataDes.dataType = HCCL_DATA_TYPE_UINT32;
        std::vector<std::string> tags;
        const std::vector<int64_t> offsets = {0, 512, -512, std::numeric_limits<int64_t>::min()};
        for (int64_t offset : offsets) {
            std::string tag;
            ASSERT_EQ(AicpuTaskCacheKey::GetAicpuTaskCacheTag(param, 1024, offset, 512, tag), HCCL_SUCCESS);
            for (const auto& previous : tags) {
                EXPECT_NE(tag, previous);
            }
            tags.push_back(tag);
        }
        EXPECT_NE(tags[2].find("--512-512-"), std::string::npos);
        EXPECT_NE(tags[3].find("--9223372036854775808-512-"), std::string::npos);
    }

    TEST(AicpuTaskCacheKeyTest, GraphAndOverlapFeaturesPreserveKeyIsolation)
    {
        OpParam param{};
        param.opType = HcclCMDType::HCCL_CMD_ALLGATHER;
        param.DataDes.dataType = HCCL_DATA_TYPE_UINT32;
        std::vector<std::string> tags;
        const std::vector<int64_t> offsets = {0, 512, -512, std::numeric_limits<int64_t>::min()};
        for (auto mode : {OpMode::OPBASE, OpMode::OFFLOAD}) {
            for (bool symmetric : {false, true}) {
                for (bool capture : {false, true}) {
                    for (int64_t diff : offsets) {
                        param.opMode = mode;
                        param.supportSymmetricMemory = symmetric;
                        param.isCapture = capture;
                        std::string normalTag;
                        ASSERT_EQ(
                            AicpuTaskCacheKey::GetAicpuTaskCacheTag(param, 1024, diff, 512, normalTag), HCCL_SUCCESS);
                        for (const auto& previous : tags) {
                            EXPECT_NE(normalTag, previous);
                        }
                        tags.push_back(normalTag);

                        std::string repeatTag;
                        ASSERT_EQ(
                            AicpuTaskCacheKey::GetAicpuTaskCacheTag(param, 1024, diff, 512, repeatTag), HCCL_SUCCESS);
                        EXPECT_EQ(repeatTag, normalTag);
                    }
                }
            }
        }
    }

#if CANN_VERSION_NUM >= CANN_VERSION(9, 1, 0)
    class AlltoallvCachePolicyTest : public testing::TestWithParam<HcclCMDType> {
    protected:
        void SetUp() override
        {
            param_.aicpuCacheEnable = true;
            param_.deviceType = HcclDevType::DEV_TYPE_950;
            param_.opMode = OpMode::OPBASE;
            param_.opType = GetParam();
            const char* algName
                = GetParam() == HcclCMDType::HCCL_CMD_ALLTOALLV ? "AicpuAllToAllVSoleMesh" : "AicpuAllToAllVCSoleMesh";
            const auto nameLen = std::strlen(algName);
            ASSERT_LT(nameLen, sizeof(param_.algName));
            std::copy_n(algName, nameLen + 1, param_.algName);
            param_.inputPtr = input_;
            param_.outputPtr = output_;
            param_.inputSize = sizeof(input_);
            param_.outputSize = sizeof(output_);
            param_.all2AllVDataDes
                = {HCCL_DATA_TYPE_UINT32, HCCL_DATA_TYPE_UINT32, counts_, counts_, displs_, displs_, nullptr};
            resCtx_.topoInfo.userRankSize = RANK_SIZE;
            resCtx_.topoInfo.topoLevelNums = 1;
            resCtx_.topoInfo.netLayerDetails.netLayerNum = 1;
            resCtx_.topoInfo.netLayerDetails.netLayers = {0};
            resCtx_.topoInfo.netLayerDetails.netInstNumOfLayer = {1};
            resCtx_.algHierarchyInfo.infos = {{{0, 1, 2}}};
            resCtx_.cclMem.size = CCL_BUFFER_SIZE;
            ChannelInfo channel{};
            channel.isValid = true;
            channel.remoteRank = 1;
            channel.protocol = CommProtocol::COMM_PROTOCOL_UB_CTP;
            resCtx_.channels = {{channel}};
        }

        static constexpr uint32_t RANK_SIZE = 3;
        static constexpr uint64_t CCL_BUFFER_SIZE = 1024;
        uint32_t input_[CCL_BUFFER_SIZE / sizeof(uint32_t)] = {};
        uint32_t output_[CCL_BUFFER_SIZE / sizeof(uint32_t)] = {};
        uint64_t counts_[RANK_SIZE] = {};
        uint64_t displs_[RANK_SIZE] = {};
        OpParam param_{};
        AlgResourceCtxSerializable resCtx_{};
    };

    TEST_P(AlltoallvCachePolicyTest, HonorsHostCacheDisable)
    {
        counts_[1] = 1;
        bool isCacheEnable = false;
        ASSERT_EQ(AicpuTaskCachePolicy::IsAicpuTaskCacheEnable(param_, resCtx_, isCacheEnable), HCCL_SUCCESS);
        ASSERT_TRUE(isCacheEnable);

        // Host关闭缓存后，Device侧不得重新使能。
        param_.aicpuCacheEnable = false;
        isCacheEnable = true;
        ASSERT_EQ(AicpuTaskCachePolicy::IsAicpuTaskCacheEnable(param_, resCtx_, isCacheEnable), HCCL_SUCCESS);
        EXPECT_FALSE(isCacheEnable);
    }

    TEST_P(AlltoallvCachePolicyTest, ChecksSplitBoundInBytesAcrossDataTypes)
    {
        counts_[1] = 129; // 129 bytes fit; 129 uint64 elements require splitting.
        for (auto dataType : {HCCL_DATA_TYPE_UINT8, HCCL_DATA_TYPE_UINT64}) {
            param_.all2AllVDataDes.sendType = dataType;
            param_.all2AllVDataDes.recvType = dataType;
            bool isCacheEnable = true;
            ASSERT_EQ(AicpuTaskCachePolicy::IsAicpuTaskCacheEnable(param_, resCtx_, isCacheEnable), HCCL_SUCCESS);
            EXPECT_EQ(isCacheEnable, dataType == HCCL_DATA_TYPE_UINT8);
        }
    }

    TEST_P(AlltoallvCachePolicyTest, EnablesCacheForSelfCopyOnlyRank)
    {
        for (uint32_t rank = 0; rank < RANK_SIZE; ++rank) {
            SCOPED_TRACE(rank);
            resCtx_.topoInfo.userRank = rank;
            counts_[rank] = 1;
            bool isCacheEnable = true;
            ASSERT_EQ(AicpuTaskCachePolicy::IsAicpuTaskCacheEnable(param_, resCtx_, isCacheEnable), HCCL_SUCCESS);
            EXPECT_TRUE(isCacheEnable);
            counts_[rank] = 0;
        }
    }

    TEST_P(AlltoallvCachePolicyTest, RejectsSelfCopyCountRequiringSplit)
    {
        resCtx_.topoInfo.userRank = 0;
        counts_[0] = CCL_BUFFER_SIZE / (RANK_SIZE - 1) / sizeof(input_[0]) + 1;
        bool isCacheEnable = true;
        ASSERT_EQ(AicpuTaskCachePolicy::IsAicpuTaskCacheEnable(param_, resCtx_, isCacheEnable), HCCL_SUCCESS);
        EXPECT_FALSE(isCacheEnable);
    }

    TEST_P(AlltoallvCachePolicyTest, DisablesCacheForInvalidRankWithoutError)
    {
        for (uint32_t rankSize : {uint32_t{0}, RANK_SIZE}) {
            SCOPED_TRACE(rankSize);
            resCtx_.topoInfo.userRankSize = rankSize;
            resCtx_.topoInfo.userRank = rankSize;
            bool isCacheEnable = true;
            ASSERT_EQ(AicpuTaskCachePolicy::IsAicpuTaskCacheEnable(param_, resCtx_, isCacheEnable), HCCL_SUCCESS);
            EXPECT_FALSE(isCacheEnable);
        }
    }

    TEST_P(AlltoallvCachePolicyTest, DisablesCacheForMissingCountsWithoutError)
    {
        for (bool missingSendCounts : {true, false}) {
            SCOPED_TRACE(missingSendCounts);
            param_.all2AllVDataDes.sendCounts = missingSendCounts ? nullptr : counts_;
            param_.all2AllVDataDes.recvCounts = missingSendCounts ? counts_ : nullptr;
            bool isCacheEnable = true;
            ASSERT_EQ(AicpuTaskCachePolicy::IsAicpuTaskCacheEnable(param_, resCtx_, isCacheEnable), HCCL_SUCCESS);
            EXPECT_FALSE(isCacheEnable);
        }
    }

    TEST_P(AlltoallvCachePolicyTest, DisablesCacheForUnsupportedHierarchyWithoutError)
    {
        using HierarchyInfos = decltype(resCtx_.algHierarchyInfo.infos);
        const std::vector<HierarchyInfos> cases = {{}, {{}}, {{{}}}, {{{0}}}, {{{0, 1}, {}}}};
        for (size_t index = 0; index < cases.size(); ++index) {
            SCOPED_TRACE(index);
            resCtx_.algHierarchyInfo.infos = cases[index];
            bool isCacheEnable = true;
            ASSERT_EQ(AicpuTaskCachePolicy::IsAicpuTaskCacheEnable(param_, resCtx_, isCacheEnable), HCCL_SUCCESS);
            EXPECT_FALSE(isCacheEnable);
        }
    }

    TEST_P(AlltoallvCachePolicyTest, DisablesCacheForInvalidDataTypeWithoutError)
    {
        for (auto dataType : {HCCL_DATA_TYPE_RESERVED, static_cast<HcclDataType>(-1)}) {
            SCOPED_TRACE(static_cast<int>(dataType));
            param_.all2AllVDataDes.sendType = dataType;
            bool isCacheEnable = true;
            ASSERT_EQ(AicpuTaskCachePolicy::IsAicpuTaskCacheEnable(param_, resCtx_, isCacheEnable), HCCL_SUCCESS);
            EXPECT_FALSE(isCacheEnable);
        }
    }

    TEST_P(AlltoallvCachePolicyTest, EnablesCacheWithOneChannelPerPeer)
    {
        counts_[0] = 1;
        ChannelInfo channel = resCtx_.channels[0][0];
        channel.remoteRank = 2;
        resCtx_.channels[0].push_back(channel);
        bool isCacheEnable = true;
        ASSERT_EQ(AicpuTaskCachePolicy::IsAicpuTaskCacheEnable(param_, resCtx_, isCacheEnable), HCCL_SUCCESS);
        EXPECT_TRUE(isCacheEnable);
    }

    TEST_P(AlltoallvCachePolicyTest, RejectsMultipleChannelsForPeerEvenWithZeroCount)
    {
        resCtx_.channels[0].push_back(resCtx_.channels[0][0]);
        counts_[0] = 1;
        for (uint64_t peerCount : {uint64_t{0}, uint64_t{1}}) {
            SCOPED_TRACE(peerCount);
            counts_[1] = peerCount;
            bool isCacheEnable = true;
            ASSERT_EQ(AicpuTaskCachePolicy::IsAicpuTaskCacheEnable(param_, resCtx_, isCacheEnable), HCCL_SUCCESS);
            EXPECT_FALSE(isCacheEnable);
        }
    }

    TEST_P(AlltoallvCachePolicyTest, IgnoresInvalidDuplicateChannel)
    {
        counts_[0] = 1;
        ChannelInfo channel = resCtx_.channels[0][0];
        channel.isValid = false;
        resCtx_.channels[0].push_back(channel);
        bool isCacheEnable = true;
        ASSERT_EQ(AicpuTaskCachePolicy::IsAicpuTaskCacheEnable(param_, resCtx_, isCacheEnable), HCCL_SUCCESS);
        EXPECT_TRUE(isCacheEnable);
    }

    TEST_P(AlltoallvCachePolicyTest, LeavesNonVCacheAdmissionUnchanged)
    {
        param_.opType = HcclCMDType::HCCL_CMD_ALLGATHER;
        param_.DataDes = {1, HCCL_DATA_TYPE_UINT32, HCCL_DATA_TYPE_UINT32, 0};
        param_.inputSize = sizeof(input_[0]);
        param_.outputSize = RANK_SIZE * sizeof(output_[0]);
        resCtx_.channels[0].push_back(resCtx_.channels[0][0]);
        bool isCacheEnable = false;
        ASSERT_EQ(AicpuTaskCachePolicy::IsAicpuTaskCacheEnable(param_, resCtx_, isCacheEnable), HCCL_SUCCESS);
        EXPECT_TRUE(isCacheEnable);
    }

    TEST_P(AlltoallvCachePolicyTest, AllowsUbChannelsRegardlessOfCoverLayer)
    {
        struct TopologyCase {
            std::vector<uint32_t> layers;
            std::vector<uint32_t> instances;
            uint32_t levelCount;
        };
        const std::vector<TopologyCase> cases = {
            {{0, 1, 2}, {1, 1, 1}, 1},
            {{0, 1, 2}, {3, 1, 1}, 2},
            {{0, 1, 2}, {3, 2, 1}, 3},
            {{0, 2}, {2, 0, 1}, 2},
            {{0, 3}, {2, 0, 0, 1}, 2},
            {{}, {}, 0},
            {{0}, {1}, 0},
            {{0}, {1}, 2},
            {{0, 1}, {3}, 2},
            {{0, 1}, {3, 0}, 2},
            {{0, 1}, {3, 2}, 2},
        };
        counts_[0] = 1;
        for (auto protocol : {CommProtocol::COMM_PROTOCOL_UB_CTP, CommProtocol::COMM_PROTOCOL_UBOE}) {
            SCOPED_TRACE(static_cast<uint32_t>(protocol));
            resCtx_.channels[0][0].protocol = protocol;
            for (size_t index = 0; index < cases.size(); ++index) {
                SCOPED_TRACE(index);
                const auto& testCase = cases[index];
                auto& topo = resCtx_.topoInfo;
                topo.topoLevelNums = testCase.levelCount;
                topo.netLayerDetails.netLayerNum = static_cast<uint32_t>(testCase.layers.size());
                topo.netLayerDetails.netLayers = testCase.layers;
                topo.netLayerDetails.netInstNumOfLayer = testCase.instances;
                bool isCacheEnable = true;
                ASSERT_EQ(AicpuTaskCachePolicy::IsAicpuTaskCacheEnable(param_, resCtx_, isCacheEnable), HCCL_SUCCESS);
                EXPECT_TRUE(isCacheEnable);
            }
        }
    }

    TEST_P(AlltoallvCachePolicyTest, RejectsUnsupportedProtocol)
    {
        counts_[0] = 1;
        resCtx_.channels[0][0].protocol = CommProtocol::COMM_PROTOCOL_ROCE;
        bool isCacheEnable = true;
        ASSERT_EQ(AicpuTaskCachePolicy::IsAicpuTaskCacheEnable(param_, resCtx_, isCacheEnable), HCCL_SUCCESS);
        EXPECT_FALSE(isCacheEnable);
    }

    TEST_P(AlltoallvCachePolicyTest, AllowsUbChannelsAtScaleOutLayerForNonVOperator)
    {
        param_.opType = HcclCMDType::HCCL_CMD_ALLGATHER;
        param_.DataDes = {1, HCCL_DATA_TYPE_UINT32, HCCL_DATA_TYPE_UINT32, 0};
        resCtx_.topoInfo.topoLevelNums = 2;
        resCtx_.topoInfo.netLayerDetails.netLayerNum = 2;
        resCtx_.topoInfo.netLayerDetails.netLayers = {0, 2};
        resCtx_.topoInfo.netLayerDetails.netInstNumOfLayer = {2, 0, 1};
        for (auto protocol : {CommProtocol::COMM_PROTOCOL_UB_CTP, CommProtocol::COMM_PROTOCOL_UBOE}) {
            SCOPED_TRACE(static_cast<uint32_t>(protocol));
            resCtx_.channels[0][0].protocol = protocol;
            bool isCacheEnable = false;
            ASSERT_EQ(AicpuTaskCachePolicy::IsAicpuTaskCacheEnable(param_, resCtx_, isCacheEnable), HCCL_SUCCESS);
            EXPECT_TRUE(isCacheEnable);
        }
    }

    TEST_P(AlltoallvCachePolicyTest, AllowsCaptureWithOtherwiseSupportedInput)
    {
        counts_[1] = 1;
        param_.isCapture = true;
        bool enabled = false;
        ASSERT_EQ(AicpuTaskCachePolicy::IsAicpuTaskCacheEnable(param_, resCtx_, enabled), HCCL_SUCCESS);
        EXPECT_TRUE(enabled);
    }

    TEST_P(AlltoallvCachePolicyTest, RejectsInplaceOverlapForVariableOps)
    {
        counts_[1] = 1;
        // 构造完全重叠: V 类算子(变长)inplace 显式拦截, 不进 cache
        param_.outputPtr = param_.inputPtr;
        param_.outputSize = param_.inputSize;
        bool isCacheEnable = true;
        ASSERT_EQ(AicpuTaskCachePolicy::IsAicpuTaskCacheEnable(param_, resCtx_, isCacheEnable), HCCL_SUCCESS);
        EXPECT_FALSE(isCacheEnable);
    }

    // 非V类算子的inplace放行(本需求的行为翻转)
    class AicpuTaskCacheInplacePolicyTest : public testing::Test {
    protected:
        void SetUp() override
        {
            param_.aicpuCacheEnable = true;
            param_.deviceType = HcclDevType::DEV_TYPE_950;
            param_.opMode = OpMode::OPBASE;
            param_.opType = HcclCMDType::HCCL_CMD_ALLGATHER;
            param_.DataDes = {1, HCCL_DATA_TYPE_UINT32, HCCL_DATA_TYPE_UINT32, 0};
            param_.inputPtr = buf_;
            param_.outputPtr = buf_; // 完全相同
            param_.inputSize = sizeof(buf_);
            param_.outputSize = sizeof(buf_);
            resCtx_.topoInfo.userRankSize = 3;
            resCtx_.cclMem.size = 1024;
            ChannelInfo channel{};
            channel.isValid = true;
            channel.remoteRank = 1;
            channel.protocol = CommProtocol::COMM_PROTOCOL_UB_CTP;
            resCtx_.channels = {{channel}};
        }

        uint32_t buf_[256] = {};
        OpParam param_{};
        AlgResourceCtxSerializable resCtx_{};
    };

    TEST_F(AicpuTaskCacheInplacePolicyTest, AdmitsInplaceOverlapForNonVariableOp)
    {
        // 本需求: 完全相同的 inplace 调用(AllGather)放行进 cache
        bool isCacheEnable = false;
        ASSERT_EQ(AicpuTaskCachePolicy::IsAicpuTaskCacheEnable(param_, resCtx_, isCacheEnable), HCCL_SUCCESS);
        EXPECT_TRUE(isCacheEnable);
    }

    TEST_F(AicpuTaskCacheInplacePolicyTest, AdmitsGraphSymmetricAndCaptureOutplaceCalls)
    {
        uint32_t output[256] = {};
        param_.outputPtr = output;
        for (auto mode : {OpMode::OPBASE, OpMode::OFFLOAD}) {
            for (bool symmetric : {false, true}) {
                for (bool capture : {false, true}) {
                    param_.opMode = mode;
                    param_.supportSymmetricMemory = symmetric;
                    param_.isCapture = capture;
                    bool enabled = false;
                    ASSERT_EQ(AicpuTaskCachePolicy::IsAicpuTaskCacheEnable(param_, resCtx_, enabled), HCCL_SUCCESS);
                    EXPECT_TRUE(enabled);
                }
            }
        }
    }

    TEST_F(AicpuTaskCacheInplacePolicyTest, RejectsGraphInplaceCalls)
    {
        param_.opMode = OpMode::OFFLOAD;
        bool enabled = true;
        ASSERT_EQ(AicpuTaskCachePolicy::IsAicpuTaskCacheEnable(param_, resCtx_, enabled), HCCL_SUCCESS);
        EXPECT_FALSE(enabled);
    }

    TEST_P(AlltoallvCachePolicyTest, RejectsGraphAndSymmetricMemoryForVariableOps)
    {
        counts_[1] = 1;
        param_.opMode = OpMode::OFFLOAD;
        bool enabled = true;
        ASSERT_EQ(AicpuTaskCachePolicy::IsAicpuTaskCacheEnable(param_, resCtx_, enabled), HCCL_SUCCESS);
        EXPECT_FALSE(enabled);
        param_.opMode = OpMode::OPBASE;
        param_.supportSymmetricMemory = true;
        enabled = true;
        ASSERT_EQ(AicpuTaskCachePolicy::IsAicpuTaskCacheEnable(param_, resCtx_, enabled), HCCL_SUCCESS);
        EXPECT_FALSE(enabled);
    }

    class AicpuTaskCacheRemoteMemTest : public testing::Test {
    protected:
        void SetUp() override
        {
            g_symRemoteAddrSupported = true;
            g_symPeerPointerSupported = false;
            param_.opType = HcclCMDType::HCCL_CMD_ALLGATHER;
            param_.opMode = OpMode::OFFLOAD;
            param_.inputPtr = inputs_[1];
            param_.outputPtr = outputs_[1];
            param_.inputSize = 32;
            param_.outputSize = 64;
            resCtx_.topoInfo.userRankSize = 3;
            resCtx_.topoInfo.userRank = 1;
            // 故意打乱rank顺序并分散在两层channel中。
            resCtx_.channels = {{MakeChannel(2)}, {MakeChannel(0)}};
            inputWindow_.peerBases = {inputs_[0], inputs_[1], inputs_[2]};
            outputWindow_.peerBases = {outputs_[0], outputs_[1], outputs_[2]};
        }

        void TearDown() override
        {
            g_symRemoteAddrSupported = true;
            g_symPeerPointerSupported = false;
        }

        ChannelInfo MakeChannel(uint32_t rank)
        {
            ChannelInfo channel{};
            channel.isValid = true;
            channel.remoteRank = rank;
            channel.remoteInputGraphMode.addr = inputs_[rank];
            channel.remoteInputGraphMode.size = 128;
            channel.remoteOutputGraphMode.addr = outputs_[rank];
            channel.remoteOutputGraphMode.size = 192;
            return channel;
        }

        void UseSymmetricMemory()
        {
            param_.opMode = OpMode::OPBASE;
            param_.supportSymmetricMemory = true;
            param_.inputSymWindow = &inputWindow_;
            param_.outputSymWindow = &outputWindow_;
            param_.inputOffset = 8;
            param_.outputOffset = 16;
            param_.inputPtr = inputs_[1] + param_.inputOffset;
            param_.outputPtr = outputs_[1] + param_.outputOffset;
            resCtx_.channels.clear(); // 对称内存不能依赖channel中缓存的图地址。
        }

        void Collect(bool expectedComplete)
        {
            // 先设为相反值, 确认每次采集都会更新complete。
            bool complete = !expectedComplete;
            ASSERT_EQ(
                AicpuTaskCacheUtils::GetRemoteMemInfoForCache(param_, resCtx_, addrs_, sizes_, complete), HCCL_SUCCESS);
            EXPECT_EQ(complete, expectedComplete);
        }

        uint8_t inputs_[3][256] = {};
        uint8_t outputs_[3][256] = {};
        CacheTestSymmetricWindow inputWindow_;
        CacheTestSymmetricWindow outputWindow_;
        OpParam param_{};
        AlgResourceCtxSerializable resCtx_{};
        std::vector<void*> addrs_;
        std::vector<uint64_t> sizes_;
    };

    TEST_F(AicpuTaskCacheRemoteMemTest, CollectsGraphMemoryInRankOrderAcrossLevels)
    {
        resCtx_.channels[0].push_back(MakeChannel(1)); // 本rank使用param中的地址和大小。
        ChannelInfo invalid{};
        invalid.remoteRank = UINT32_MAX;
        resCtx_.channels[0].push_back(invalid);
        resCtx_.channels[1].push_back(MakeChannel(2)); // 一致的重复channel允许复用。
        Collect(true);
        EXPECT_EQ(
            addrs_,
            (std::vector<void*>{
                inputs_[1], outputs_[1], inputs_[0], outputs_[0], inputs_[1], outputs_[1], inputs_[2], outputs_[2]}));
        EXPECT_EQ(sizes_, (std::vector<uint64_t>{32, 64, 128, 192, 32, 64, 128, 192}));
    }

    TEST_F(AicpuTaskCacheRemoteMemTest, PreparationErrorsDisableCacheWithoutMaskingSizeValidation)
    {
        struct InputCase {
            HcclDataType type;
            uint64_t count;
            bool missingCounts;
        };
        const std::vector<InputCase> cases = {
            {HCCL_DATA_TYPE_UINT64, 1, true},
            {HCCL_DATA_TYPE_RESERVED, 1, false},
            {static_cast<HcclDataType>(-1), 1, false},
            {HCCL_DATA_TYPE_UINT64, UINT64_MAX, false},     // count * typeSize overflows.
            {HCCL_DATA_TYPE_UINT64, UINT64_MAX / 8, false}, // bytes * rankSize overflows.
        };
        param_.aicpuCacheEnable = true;
        param_.deviceType = HcclDevType::DEV_TYPE_950;
        for (auto& level : resCtx_.channels) {
            for (auto& channel : level) {
                channel.protocol = CommProtocol::COMM_PROTOCOL_UB_CTP;
            }
        }
        for (bool symmetric : {false, true}) {
            if (symmetric) {
                UseSymmetricMemory();
            }
            for (bool capture : {false, true}) {
                param_.isCapture = capture;
                for (size_t i = 0; i < cases.size(); ++i) {
                    SCOPED_TRACE(symmetric);
                    SCOPED_TRACE(capture);
                    SCOPED_TRACE(i);
                    uint64_t count = cases[i].count;
                    param_.opType = HcclCMDType::HCCL_CMD_ALLTOALL;
                    param_.all2AllDataDes.sendType = cases[i].type;
                    param_.all2AllVDataDes.sendCounts = cases[i].missingCounts ? nullptr : &count;
                    uint64_t inputSize = 0;
                    uint64_t outputSize = 0;
                    // The size helper still reports invalid input; only optional cache preparation falls back.
                    EXPECT_EQ(
                        AicpuTaskCacheUtils::GetInputOutputInfoForCache(param_, 3, inputSize, outputSize), HCCL_E_PARA);
                    bool enabled = true;
                    ASSERT_EQ(AicpuTaskCachePolicy::IsAicpuTaskCacheEnable(param_, resCtx_, enabled), HCCL_SUCCESS);
                    EXPECT_FALSE(enabled);
                    addrs_ = {param_.inputPtr};
                    sizes_ = {param_.inputSize};
                    Collect(false);
                    EXPECT_TRUE(addrs_.empty());
                    EXPECT_TRUE(sizes_.empty());

                    // Failure must not poison a later valid invocation.
                    count = 1;
                    param_.all2AllDataDes.sendType = HCCL_DATA_TYPE_UINT64;
                    param_.all2AllVDataDes.sendCounts = &count;
                    ASSERT_EQ(AicpuTaskCachePolicy::IsAicpuTaskCacheEnable(param_, resCtx_, enabled), HCCL_SUCCESS);
                    EXPECT_TRUE(enabled);
                    Collect(true);
                }
            }
        }
    }

    TEST_F(AicpuTaskCacheRemoteMemTest, RejectsInvalidRankLayoutAndClearsPreviousArrays)
    {
        Collect(true);
        resCtx_.topoInfo.userRankSize = 0;
        Collect(false);
        EXPECT_TRUE(addrs_.empty());
        EXPECT_TRUE(sizes_.empty());
        resCtx_.topoInfo.userRankSize = 3;
        resCtx_.topoInfo.userRank = 3;
        Collect(false);
        resCtx_.topoInfo.userRank = 1;
        resCtx_.topoInfo.userRankSize = UINT32_MAX;
        Collect(false); // 扩展数组长度超出协议上限, 不应尝试分配。
    }

    TEST_F(AicpuTaskCacheRemoteMemTest, RejectsUnsupportedModesAndVariableOperators)
    {
        param_.opMode = OpMode::OPBASE;
        Collect(false);
        param_.opMode = OpMode::OFFLOAD;
        for (auto op : {HcclCMDType::HCCL_CMD_ALLTOALLV, HcclCMDType::HCCL_CMD_ALLTOALLVC}) {
            param_.opType = op;
            Collect(false);
        }
    }

    TEST_F(AicpuTaskCacheRemoteMemTest, RejectsMissingAndOutOfRangeRemoteRanks)
    {
        resCtx_.channels[1].clear();
        Collect(false);
        resCtx_.channels[1].push_back(MakeChannel(0));
        resCtx_.channels[0][0].remoteRank = 3;
        Collect(false);
    }

    TEST_F(AicpuTaskCacheRemoteMemTest, RejectsConflictingPeerAddressesAndSizes)
    {
        for (bool inputSide : {false, true}) {
            for (bool changeSize : {false, true}) {
                ChannelInfo duplicate = MakeChannel(2);
                HcclMem& mem = inputSide ? duplicate.remoteInputGraphMode : duplicate.remoteOutputGraphMode;
                if (changeSize) {
                    ++mem.size;
                } else {
                    mem.addr = inputs_[0];
                }
                resCtx_.channels[1].push_back(duplicate);
                Collect(false);
                resCtx_.channels[1].pop_back();
            }
        }
    }

    TEST_F(AicpuTaskCacheRemoteMemTest, RejectsInvalidRemoteMemoryRanges)
    {
        for (bool inputSide : {false, true}) {
            HcclMem& mem = inputSide ? resCtx_.channels[0][0].remoteInputGraphMode :
                                       resCtx_.channels[0][0].remoteOutputGraphMode;
            const HcclMem original = mem;
            mem.addr = nullptr;
            Collect(false);
            mem = original;
            mem.size = 0;
            Collect(false);
            mem = original;
            mem.addr = reinterpret_cast<void*>(UINT64_MAX - mem.size + 1);
            Collect(false);
            mem = original;
        }
    }

    TEST_F(AicpuTaskCacheRemoteMemTest, RejectsInvalidLocalMemoryRanges)
    {
        param_.inputPtr = nullptr;
        Collect(false);
        param_.inputPtr = inputs_[1];
        param_.outputSize = 0;
        Collect(false);
        param_.outputSize = 64;
        param_.outputPtr = reinterpret_cast<void*>(UINT64_MAX - 63);
        Collect(false);
    }

    TEST_F(AicpuTaskCacheRemoteMemTest, QueriesSymmetricWindowsAgainForEachCall)
    {
        UseSymmetricMemory();
        Collect(true);
        EXPECT_EQ(
            addrs_, (std::vector<void*>{
                        inputs_[1] + 8, outputs_[1] + 16, inputs_[0] + 8, outputs_[0] + 16, inputs_[1] + 8,
                        outputs_[1] + 16, inputs_[2] + 8, outputs_[2] + 16}));
        EXPECT_EQ(sizes_, (std::vector<uint64_t>{32, 64, 32, 64, 32, 64, 32, 64}));
        EXPECT_EQ(inputWindow_.queryCount, 2U);
        EXPECT_EQ(outputWindow_.queryCount, 2U);

        // 模拟命中时更换窗口与偏移, 必须重新查询而不是沿用上次采集的地址。
        CacheTestSymmetricWindow nextInput{{outputs_[0], outputs_[1], outputs_[2]}};
        param_.inputSymWindow = &nextInput;
        param_.inputOffset = 24;
        param_.inputPtr = outputs_[1] + 24;
        Collect(true);
        ASSERT_EQ(addrs_.size(), 8U);
        EXPECT_EQ(addrs_[0], outputs_[1] + 24);
        EXPECT_EQ(addrs_[2], outputs_[0] + 24);
        EXPECT_EQ(addrs_[6], outputs_[2] + 24);
        EXPECT_EQ(nextInput.queryCount, 2U);
    }

    TEST_F(AicpuTaskCacheRemoteMemTest, RejectsMissingSymmetricWindowsAndQueryCapabilities)
    {
        UseSymmetricMemory();
        param_.inputSymWindow = nullptr;
        Collect(false);
        param_.inputSymWindow = &inputWindow_;
        param_.outputSymWindow = nullptr;
        Collect(false);
        param_.outputSymWindow = &outputWindow_;
        g_symRemoteAddrSupported = false;
        Collect(false);
        EXPECT_EQ(inputWindow_.queryCount, 0U);
        g_symPeerPointerSupported = true;
        Collect(true); // 仅有旧peer-pointer查询能力时也允许采集。
    }

    TEST_F(AicpuTaskCacheRemoteMemTest, FallsBackWhenEitherSymmetricWindowQueryFails)
    {
        UseSymmetricMemory();
        for (bool inputSide : {false, true}) {
            auto& window = inputSide ? inputWindow_ : outputWindow_;
            window.failedRank = 2;
            Collect(false);
            window.failedRank = UINT32_MAX;
            Collect(true);
        }
    }

    INSTANTIATE_TEST_SUITE_P(
        AlltoallvTypes, AlltoallvCachePolicyTest,
        testing::Values(HcclCMDType::HCCL_CMD_ALLTOALLV, HcclCMDType::HCCL_CMD_ALLTOALLVC));
#endif

    TEST(AicpuTaskCacheUtilsTest, CalcInplaceDiffComputesPositionAndWidth)
    {
        struct OverlapCase {
            int64_t inputOffset; // output 起点相对 input 起点
            uint64_t inputSize;
            uint64_t outputSize;
            int64_t expectDiff;
            uint64_t expectOverlap;
        };
        const std::vector<OverlapCase> cases = {
            {2048, 1024, 1024, 0, 0},      // 不相交: output 在后, 输出(0,0)
            {-2048, 1024, 1024, 0, 0},     // 不相交: output 在前, 输出(0,0)
            {0, 1024, 1024, 0, 1024},      // 完全相同
            {512, 1024, 1024, 512, 512},   // 部分重叠: output 在后
            {-512, 1024, 1024, -512, 512}, // 部分重叠: output 在前
            {256, 1024, 512, 256, 512},    // 嵌套: output 含于 input(宽度封顶)
            {-256, 512, 1024, -256, 512},  // 嵌套: input 含于 output
        };
        for (size_t index = 0; index < cases.size(); ++index) {
            SCOPED_TRACE(index);
            const auto& testCase = cases[index];
            uint8_t input[1024] = {};
            uint8_t output[1024] = {};
            const uint64_t inputStart = reinterpret_cast<uint64_t>(input);
            uint64_t outputStart = reinterpret_cast<uint64_t>(output);
            // 用偏移构造期望的重叠几何, 避免依赖栈地址的实际间距
            if (testCase.inputOffset >= 0) {
                outputStart = inputStart + static_cast<uint64_t>(testCase.inputOffset);
            } else {
                outputStart = inputStart - static_cast<uint64_t>(-testCase.inputOffset);
            }
            OpParam param{};
            param.inputPtr = reinterpret_cast<void*>(inputStart);
            param.outputPtr = reinterpret_cast<void*>(outputStart);

            int64_t diff = 1;
            uint64_t overlapBytes = 1;
            AicpuTaskCacheUtils::CalcInplaceDiff(param, testCase.inputSize, testCase.outputSize, diff, overlapBytes);
            EXPECT_EQ(diff, testCase.expectDiff);
            EXPECT_EQ(overlapBytes, testCase.expectOverlap);
        }
    }

    TEST(AicpuTaskCacheUtilsTest, AppendVOpInfoConvertsCountsAndDisplacementsToBytes)
    {
        uint8_t input[32] = {};
        uint8_t output[32] = {};
        uint64_t sendCounts[] = {2, 0};
        uint64_t recvCounts[] = {1, 3};
        uint64_t sendDispls[] = {1, 100}; // Empty peer may retain a displacement beyond the buffer.
        uint64_t recvDispls[] = {2, 3};
        OpParam param{};
        param.all2AllVDataDes
            = {HCCL_DATA_TYPE_UINT32, HCCL_DATA_TYPE_UINT16, sendCounts, recvCounts, sendDispls, recvDispls, nullptr};
        std::vector<void*> addrs = {input, output};
        std::vector<uint64_t> sizes = {sizeof(input), sizeof(output)};

        ASSERT_EQ(AicpuTaskCacheUtils::AppendVOpInfoForCache(param, 2, addrs, sizes), HCCL_SUCCESS);
        ASSERT_EQ(addrs.size(), 6U);
        EXPECT_EQ(addrs[0], input);
        EXPECT_EQ(addrs[1], output);
        EXPECT_EQ(addrs[2], input + 4);
        EXPECT_EQ(addrs[3], reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(input) + 400));
        EXPECT_EQ(addrs[4], output + 4);
        EXPECT_EQ(addrs[5], output + 6);
        EXPECT_EQ(sizes, (std::vector<uint64_t>{32, 32, 8, 0, 2, 6}));
    }

    TEST(AicpuTaskCacheUtilsTest, AppendVOpInfoRejectsNonEmptyRangeOutsideUserBuffer)
    {
        uint8_t input[8] = {};
        uint8_t output[8] = {};
        uint64_t counts[] = {2};
        uint64_t sendDispls[] = {1};
        uint64_t recvDispls[] = {0};
        OpParam param{};
        param.all2AllVDataDes
            = {HCCL_DATA_TYPE_UINT32, HCCL_DATA_TYPE_UINT32, counts, counts, sendDispls, recvDispls, nullptr};
        std::vector<void*> addrs = {input, output};
        std::vector<uint64_t> sizes = {sizeof(input), sizeof(output)};

        EXPECT_EQ(AicpuTaskCacheUtils::AppendVOpInfoForCache(param, 1, addrs, sizes), HCCL_E_PARA);
    }

    TEST(AicpuTaskCacheUtilsTest, AppendVOpInfoRejectsByteConversionOverflow)
    {
        uint8_t input[8] = {};
        uint8_t output[8] = {};
        uint64_t sendCounts[] = {std::numeric_limits<uint64_t>::max()};
        uint64_t recvCounts[] = {1};
        uint64_t displs[] = {0};
        OpParam param{};
        param.all2AllVDataDes
            = {HCCL_DATA_TYPE_UINT32, HCCL_DATA_TYPE_UINT32, sendCounts, recvCounts, displs, displs, nullptr};
        std::vector<void*> addrs = {input, output};
        std::vector<uint64_t> sizes = {sizeof(input), sizeof(output)};

        EXPECT_EQ(AicpuTaskCacheUtils::AppendVOpInfoForCache(param, 1, addrs, sizes), HCCL_E_PARA);
    }

} // namespace
} // namespace ops_hccl
