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

namespace ops_hccl {
namespace {

    TEST(AicpuTaskCacheKeyTest, VTypesShareKeyAcrossDataTypesAndInputSizes)
    {
        for (auto opType : {HcclCMDType::HCCL_CMD_ALLTOALLV, HcclCMDType::HCCL_CMD_ALLTOALLVC}) {
            OpParam param{};
            param.opType = opType;
            param.all2AllVDataDes.sendType = HCCL_DATA_TYPE_UINT8;
            std::string byteTag;
            ASSERT_EQ(AicpuTaskCacheKey::GetAicpuTaskCacheTag(param, 3, byteTag), HCCL_SUCCESS);
            for (auto dataType : {HCCL_DATA_TYPE_UINT32, HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_FP64}) {
                param.all2AllVDataDes.sendType = dataType;
                std::string tag;
                ASSERT_EQ(AicpuTaskCacheKey::GetAicpuTaskCacheTag(param, 24, tag), HCCL_SUCCESS);
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
        ASSERT_EQ(AicpuTaskCacheKey::GetAicpuTaskCacheTag(param, 24, byteTag), HCCL_SUCCESS);
        param.all2AllDataDes.sendType = HCCL_DATA_TYPE_UINT32;
        std::string tag;
        ASSERT_EQ(AicpuTaskCacheKey::GetAicpuTaskCacheTag(param, 24, tag), HCCL_SUCCESS);
        EXPECT_NE(tag, byteTag);
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
        bool isCacheEnable = true;
        ASSERT_EQ(AicpuTaskCachePolicy::IsAicpuTaskCacheEnable(param_, resCtx_, isCacheEnable), HCCL_SUCCESS);
        ASSERT_FALSE(isCacheEnable);

        // Host关闭缓存后，Device侧不得重新使能。
        param_.aicpuCacheEnable = false;
        isCacheEnable = true;
        ASSERT_EQ(AicpuTaskCachePolicy::IsAicpuTaskCacheEnable(param_, resCtx_, isCacheEnable), HCCL_SUCCESS);
        EXPECT_FALSE(isCacheEnable);
    }

    TEST_P(AlltoallvCachePolicyTest, DisablesCacheAcrossDataTypes)
    {
        counts_[1] = 129; // 覆盖不同字节长度，V类缓存均保持禁用。
        for (auto dataType : {HCCL_DATA_TYPE_UINT8, HCCL_DATA_TYPE_UINT64}) {
            param_.all2AllVDataDes.sendType = dataType;
            param_.all2AllVDataDes.recvType = dataType;
            bool isCacheEnable = true;
            ASSERT_EQ(AicpuTaskCachePolicy::IsAicpuTaskCacheEnable(param_, resCtx_, isCacheEnable), HCCL_SUCCESS);
            EXPECT_FALSE(isCacheEnable);
        }
    }

    TEST_P(AlltoallvCachePolicyTest, DisablesCacheForSelfCopyOnlyRank)
    {
        for (uint32_t rank = 0; rank < RANK_SIZE; ++rank) {
            SCOPED_TRACE(rank);
            resCtx_.topoInfo.userRank = rank;
            counts_[rank] = 1;
            bool isCacheEnable = true;
            ASSERT_EQ(AicpuTaskCachePolicy::IsAicpuTaskCacheEnable(param_, resCtx_, isCacheEnable), HCCL_SUCCESS);
            EXPECT_FALSE(isCacheEnable);
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

    TEST_P(AlltoallvCachePolicyTest, DisablesCacheWithOneChannelPerPeer)
    {
        counts_[0] = 1;
        ChannelInfo channel = resCtx_.channels[0][0];
        channel.remoteRank = 2;
        resCtx_.channels[0].push_back(channel);
        bool isCacheEnable = true;
        ASSERT_EQ(AicpuTaskCachePolicy::IsAicpuTaskCacheEnable(param_, resCtx_, isCacheEnable), HCCL_SUCCESS);
        EXPECT_FALSE(isCacheEnable);
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

    TEST_P(AlltoallvCachePolicyTest, DisablesCacheWithInvalidDuplicateChannel)
    {
        counts_[0] = 1;
        ChannelInfo channel = resCtx_.channels[0][0];
        channel.isValid = false;
        resCtx_.channels[0].push_back(channel);
        bool isCacheEnable = true;
        ASSERT_EQ(AicpuTaskCachePolicy::IsAicpuTaskCacheEnable(param_, resCtx_, isCacheEnable), HCCL_SUCCESS);
        EXPECT_FALSE(isCacheEnable);
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

    TEST_P(AlltoallvCachePolicyTest, DisablesCacheRegardlessOfCoverLayer)
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
                EXPECT_FALSE(isCacheEnable);
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

    INSTANTIATE_TEST_SUITE_P(
        AlltoallvTypes, AlltoallvCachePolicyTest,
        testing::Values(HcclCMDType::HCCL_CMD_ALLTOALLV, HcclCMDType::HCCL_CMD_ALLTOALLVC));
#endif

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
