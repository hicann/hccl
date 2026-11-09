/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "wrapper_test_common.h"

namespace wrapper_ut {

namespace {

    TemplateResource MakeKernelResource(void* npu2DpuShmem, void* dpu2NpuShmem)
    {
        TemplateResource resource{};
        resource.threads = {0x11};
        resource.npu2DpuShmemPtr = npu2DpuShmem;
        resource.dpu2NpuShmemPtr = dpu2NpuShmem;
        return resource;
    }

} // namespace

// host → DPU 的任务书交接：KernelRun 把 reduceType/dataType 写进 DPURunInfo 并序列化发给 DPU，
// 这里捕获 HcommSendRequest 的载荷反序列化，确认 DPU 侧拿到的就是 host 侧打算下发的语义。
TEST_F(DpuWriteReduceTest, KernelRunSerializesHandoffPayloadWithReduceTypeAndDataType)
{
    OpParam param{};
    const std::string algTag = "inline_reduce_seam";
    std::copy(algTag.begin(), algTag.end(), param.algTag);
    param.DataDes.dataType = HCCL_DATA_TYPE_FP32;
    param.reduceType = HCCL_REDUCE_MAX;
    param.opConfig.execTimeout = 29;

    InsTempReduceScatterNhrDpu algorithm(param, 10, {{10, 20, 30, 40}});

    std::array<u8, 256> localCcl{};
    std::array<u8, 64> npu2Dpu{};
    std::array<u8, 64> dpu2Npu{};
    TemplateDataParams params = MakeNhrParams(localCcl.data());
    // 工作参数里的 dataType/reduceOp 故意设置成与 OpParam 不同，验证 KernelRun 会用 OpParam 覆盖。
    params.dataType = HCCL_DATA_TYPE_INT8;
    params.reduceOp = HCCL_REDUCE_SUM;

    TemplateResource resource = MakeKernelResource(npu2Dpu.data(), dpu2Npu.data());
    resource.channels[20] = {MakeChannel(0x2001)};

    ASSERT_EQ(algorithm.KernelRun(param, params, resource), HCCL_SUCCESS);

    // 预拷贝按非零分片（3 片）× repeat（2 遍）执行；batch 包裹与线程同步各按 host 流程调用。
    EXPECT_EQ(g_localCopyCount, 6u);
    EXPECT_EQ(g_batchModeStartCount, 2u);
    EXPECT_EQ(g_batchModeEndCount, 1u);
    EXPECT_EQ(g_threadSyncCount, 1u);
    EXPECT_EQ(g_sendRequestShmemHandle, reinterpret_cast<uint64_t>(npu2Dpu.data()));
    EXPECT_EQ(g_sendRequestTag, algTag);

    ASSERT_FALSE(g_sendRequestPayload.empty());
    DPURunInfo restored{};
    restored.DeSerialize(g_sendRequestPayload);
    EXPECT_EQ(restored.templateName, "InsTempReduceScatterNhrDpu");
    EXPECT_EQ(restored.tempAlgParams.reduceOp, HCCL_REDUCE_MAX);
    EXPECT_EQ(restored.tempAlgParams.dataType, HCCL_DATA_TYPE_FP32);
    EXPECT_EQ(restored.myRank, 10u);
    EXPECT_EQ(restored.subCommRanks, (std::vector<std::vector<u32>>{{10, 20, 30, 40}}));
    EXPECT_EQ(restored.execTimeout, 29u);

    TemplateDataParams expected = params;
    expected.dataType = param.DataDes.dataType;
    expected.reduceOp = param.reduceType;
    ExpectTemplateDataParamsEqual(restored.tempAlgParams, expected);

    ASSERT_EQ(restored.channels.size(), 1u);
    ASSERT_EQ(restored.channels.at(20).size(), 1u);
    EXPECT_EQ(restored.channels.at(20).front().handle, 0x2001u);
}

// CalcRes：不申请从线程/notify，只按 level1 通道请求下发一组 channel。
TEST_F(DpuWriteReduceTest, CalcResRequestsLevel1ChannelsAndZeroNotifyThreads)
{
    g_channelRequestDescs.resize(3);
    g_channelRequestDescs[0].remoteRank = 20;
    g_channelRequestDescs[1].remoteRank = 30;
    g_channelRequestDescs[2].remoteRank = 40;

    InsTempReduceScatterNhrDpu algorithm(OpParam{}, 10, {{10, 20, 30, 40}});

    AlgResourceRequest request{};
    request.slaveThreadNum = 9;
    request.notifyNumOnMainThread = 7;
    request.notifyNumPerThread = {5, 6};

    OpParam param{};
    TopoInfoWithNetLayerDetails topoInfo{};
    ASSERT_EQ(algorithm.CalcRes(nullptr, param, &topoInfo, request), HCCL_SUCCESS);

    EXPECT_EQ(request.slaveThreadNum, 0u);
    EXPECT_EQ(request.notifyNumOnMainThread, 0u);
    EXPECT_TRUE(request.notifyNumPerThread.empty());
    ASSERT_EQ(request.channels.size(), 1u);
    ASSERT_EQ(request.channels.front().size(), 3u);
    EXPECT_EQ(request.channels.front()[0].remoteRank, 20u);
    EXPECT_EQ(request.channels.front()[1].remoteRank, 30u);
    EXPECT_EQ(request.channels.front()[2].remoteRank, 40u);

    EXPECT_EQ(g_channelRequestCallCount, 1u);
    EXPECT_EQ(g_channelRequestSubCommRanks, (std::vector<std::vector<u32>>{{10, 20, 30, 40}}));
}

// CalcCostCoeff：只有 isPod + CLOS + 至少两个端口时才把前两个端口求和，否则取第一个端口。
TEST_F(DpuWriteReduceTest, CalcCostCoeffAggregatesPortsOnlyForPodClosWithTwoOrMorePorts)
{
    struct Case {
        bool isPod;
        CommTopo netType;
        std::vector<u32> portNum;
        u32 expectedPortNum;
    };
    const std::array<Case, 4> cases = {{
        {true, CommTopo::COMM_TOPO_CLOS, {2, 3, 7}, 5},
        {false, CommTopo::COMM_TOPO_CLOS, {2, 3}, 2},
        {true, CommTopo::COMM_TOPO_1DMESH, {2, 3}, 2},
        {true, CommTopo::COMM_TOPO_CLOS, {4}, 4},
    }};

    for (const Case& c : cases) {
        SCOPED_TRACE(testing::Message() << "isPod=" << c.isPod << " net=" << static_cast<int>(c.netType));
        ResetHostPathCapture();
        CalcCostCoeffParam param{};
        param.rankSize = 8;
        param.dataRatio = 0.5f;
        param.netType = c.netType;
        param.portNum = c.portNum;
        param.isPod = c.isPod;

        const std::vector<CostModelParam> costs = InsTempReduceScatterNhrDpu::CalcCostCoeff(param);
        ASSERT_EQ(costs.size(), 1u);
        EXPECT_EQ(g_meshPortNum, static_cast<int>(c.expectedPortNum));
        EXPECT_EQ(g_meshNetType, c.netType);
        EXPECT_FLOAT_EQ(g_meshN, param.dataRatio);
        EXPECT_EQ(g_meshRankSize, param.rankSize);
        EXPECT_EQ(g_meshEngine, EngineType::CPU);

        EXPECT_FLOAT_EQ(costs[0].A, g_meshA);
        EXPECT_FLOAT_EQ(costs[0].B, g_localCopyB);
        EXPECT_FLOAT_EQ(costs[0].C, g_dpuLatencyC);
        EXPECT_FLOAT_EQ(costs[0].D, 0.0f);
        EXPECT_FLOAT_EQ(costs[0].reduceCopies, 0.0f);
    }
}

// CalcCostCoeff：B 用放大后的 dataRatio，C 用 NHR 步数与固定的 DPU 软件栈参数。
TEST_F(DpuWriteReduceTest, CalcCostCoeffFeedsLocalCopyAndDpuLatencyParams)
{
    CalcCostCoeffParam param{};
    param.rankSize = 4;
    param.dataRatio = 0.5f;
    param.netType = CommTopo::COMM_TOPO_CLOS;
    param.portNum = {1, 1};
    param.isPod = true;

    ASSERT_EQ(InsTempReduceScatterNhrDpu::CalcCostCoeff(param).size(), 1u);

    EXPECT_FLOAT_EQ(g_meshN, 0.5f);
    EXPECT_EQ(g_meshPortNum, 2);
    EXPECT_FLOAT_EQ(g_localCopyN, 0.5f * (1.0f + 1.0f / 4.0f));
    EXPECT_EQ(g_localCopyEngine, EngineType::AICPU);
    EXPECT_EQ(g_dpuStepNum, static_cast<int>(GetNHRStepNum(4)));
    EXPECT_EQ(g_dpuSyncNum, 1);
    EXPECT_EQ(g_dpuChannelNum, 2);
    EXPECT_EQ(g_dpuSndRcvNum, 3);
}

// CalcRes：通道请求失败时直接透传错误码，且不产生 channels、notify 保持清零。
TEST_F(DpuWriteReduceTest, CalcResPropagatesChannelRequestFailure)
{
    g_channelRequestCode = HCCL_E_INTERNAL;

    InsTempReduceScatterNhrDpu algorithm(OpParam{}, 10, {{10, 20, 30, 40}});

    AlgResourceRequest request{};
    request.slaveThreadNum = 9;
    request.notifyNumOnMainThread = 7;
    request.notifyNumPerThread = {5, 6};

    OpParam param{};
    TopoInfoWithNetLayerDetails topoInfo{};
    EXPECT_EQ(algorithm.CalcRes(nullptr, param, &topoInfo, request), HCCL_E_INTERNAL);

    EXPECT_TRUE(request.channels.empty());
    EXPECT_EQ(request.slaveThreadNum, 0u);
    EXPECT_EQ(request.notifyNumOnMainThread, 0u);
    EXPECT_TRUE(request.notifyNumPerThread.empty());
    EXPECT_EQ(g_channelRequestCallCount, 1u);
}

// KernelRun 预拷贝失败：LocalDataCopy 的错误直接透传，且不会进入 batch 收尾/线程同步/下发。
TEST_F(DpuWriteReduceTest, KernelRunPropagatesLocalDataCopyFailure)
{
    OpParam param{};
    param.DataDes.dataType = HCCL_DATA_TYPE_FP16;
    param.reduceType = HCCL_REDUCE_SUM;
    InsTempReduceScatterNhrDpu algorithm(param, 10, {{10, 20, 30, 40}});

    std::array<u8, 256> localCcl{};
    std::array<u8, 64> npu2Dpu{};
    std::array<u8, 64> dpu2Npu{};
    TemplateDataParams params = MakeNhrParams(localCcl.data());
    TemplateResource resource = MakeKernelResource(npu2Dpu.data(), dpu2Npu.data());

    g_localCopyFailOccurrence = 2;
    g_localCopyCode = HCCL_E_INTERNAL;

    EXPECT_EQ(algorithm.KernelRun(param, params, resource), HCCL_E_INTERNAL);
    EXPECT_EQ(g_localCopyCount, 2u);
    EXPECT_EQ(g_batchModeStartCount, 1u);
    EXPECT_EQ(g_batchModeEndCount, 0u);
    EXPECT_EQ(g_threadSyncCount, 0u);
    EXPECT_TRUE(g_sendRequestPayload.empty());
}

// KernelRun 收尾回写失败：任务书已下发，PostLocalCopy 的错误码作为最终返回值。
TEST_F(DpuWriteReduceTest, KernelRunPropagatesPostLocalCopyFailure)
{
    OpParam param{};
    param.DataDes.dataType = HCCL_DATA_TYPE_FP16;
    param.reduceType = HCCL_REDUCE_SUM;
    InsTempReduceScatterNhrDpu algorithm(param, 10, {{10, 20, 30, 40}});

    std::array<u8, 256> localCcl{};
    std::array<u8, 64> output{};
    std::array<u8, 64> npu2Dpu{};
    std::array<u8, 64> dpu2Npu{};
    TemplateDataParams params = MakeNhrParams(localCcl.data());
    params.buffInfo.outputPtr = output.data();
    TemplateResource resource = MakeKernelResource(npu2Dpu.data(), dpu2Npu.data());

    // 预拷贝 3 片 × 2 遍 = 6 次，收尾回写从第 7 次开始。
    g_localCopyFailOccurrence = 7;
    g_localCopyCode = HCCL_E_INTERNAL;

    EXPECT_EQ(algorithm.KernelRun(param, params, resource), HCCL_E_INTERNAL);
    EXPECT_EQ(g_localCopyCount, 7u);
    EXPECT_EQ(g_batchModeStartCount, 2u);
    EXPECT_EQ(g_batchModeEndCount, 1u);
    EXPECT_EQ(g_threadSyncCount, 1u);
    EXPECT_FALSE(g_sendRequestPayload.empty());
}

// KernelRun 第二次 BatchModeStart 失败：CHK_RET 透传，DPU 任务书不会下发。
TEST_F(DpuWriteReduceTest, KernelRunPropagatesSecondBatchModeStartFailure)
{
    OpParam param{};
    param.DataDes.dataType = HCCL_DATA_TYPE_FP16;
    param.reduceType = HCCL_REDUCE_SUM;
    InsTempReduceScatterNhrDpu algorithm(param, 10, {{10, 20, 30, 40}});

    std::array<u8, 256> localCcl{};
    std::array<u8, 64> npu2Dpu{};
    std::array<u8, 64> dpu2Npu{};
    TemplateDataParams params = MakeNhrParams(localCcl.data());
    TemplateResource resource = MakeKernelResource(npu2Dpu.data(), dpu2Npu.data());

    g_batchModeStartFailOccurrence = 2;

    EXPECT_EQ(algorithm.KernelRun(param, params, resource), HCCL_E_INTERNAL);
    EXPECT_EQ(g_localCopyCount, 6u);
    EXPECT_EQ(g_batchModeStartCount, 2u);
    EXPECT_EQ(g_batchModeEndCount, 1u);
    EXPECT_EQ(g_threadSyncCount, 0u);
    EXPECT_TRUE(g_sendRequestPayload.empty());
}

} // namespace wrapper_ut
