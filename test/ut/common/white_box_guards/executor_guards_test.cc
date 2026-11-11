/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Send/Recv 执行器防御性校验 UT：
// 1) dataType 不在产品合法集合（RESERVED、保留 gap 13、负值、越界值）时在入口返回 HCCL_E_PARA，
//    不进入查表/除法；
// 2) maxLoopTransCount_ / maxRoundTransferSize 为 0 时返回 HCCL_E_INTERNAL，
//    避免除零或 while (resDataSize > 0) 死循环。

#include <gtest/gtest.h>

#include "ins_recv_executor.h"
#include "ins_send_executor.h"
#include "ins_temp_recv_dpu.h"
#include "ins_temp_send_dpu.h"
#include "ins_v2_recv_sole_executor.h"
#include "ins_v2_send_sole_executor.h"

namespace ops_hccl {
namespace {

    // InsSendExecutor/InsRecvExecutor 的 OrchestrateOpbase 为 protected，测试子类仅做可见性提升。
    class TestableInsSendExecutor : public InsSendExecutor {
    public:
        using InsSendExecutor::OrchestrateOpbase;

        void SetDataTypeSize(u64 size) { dataTypeSize_ = size; }
    };

    class TestableInsRecvExecutor : public InsRecvExecutor {
    public:
        using InsRecvExecutor::OrchestrateOpbase;

        void SetDataTypeSize(u64 size) { dataTypeSize_ = size; }
    };

    // V2 sole executor 走 Orchestrate 前置的线程/通道校验，构造最小可用 resCtx。
    AlgResourceCtxSerializable MakeSoleResCtx(u32 myRank, u32 remoteRank)
    {
        AlgResourceCtxSerializable resCtx;
        resCtx.topoInfo.userRank = myRank;
        resCtx.threads.push_back(0);
        resCtx.algHierarchyInfo.infos.resize(1);
        resCtx.channels.resize(1);
        ChannelInfo channel;
        channel.remoteRank = remoteRank;
        resCtx.channels[0].push_back(channel);
        return resCtx;
    }

    OpParam MakeP2pParam(u32 remoteRank, HcclDataType dataType, u64 count = 1)
    {
        OpParam param;
        param.sendRecvRemoteRank = remoteRank;
        param.DataDes.dataType = dataType;
        param.DataDes.count = count;
        return param;
    }

    // 非法 dataType 覆盖：RESERVED、保留 gap(13)、负值（不得作为 size 表下标）、
    // 超出 size 表有效区间的值。旧判定（上界 + 表项非 0）会放行 13 且对负值越界查表。
    const HcclDataType INVALID_DATA_TYPES[] = {
        HCCL_DATA_TYPE_RESERVED,
        static_cast<HcclDataType>(13),
        static_cast<HcclDataType>(-1),
        static_cast<HcclDataType>(18),
    };

    TEST(ExecutorGuardTest, InsSendExecutorRejectsInvalidDataType)
    {
        TestableInsSendExecutor executor;
        for (HcclDataType dataType : INVALID_DATA_TYPES) {
            SCOPED_TRACE(static_cast<int>(dataType));
            OpParam param = MakeP2pParam(1, dataType);
            EXPECT_EQ(executor.Orchestrate(param, AlgResourceCtxSerializable{}), HCCL_E_PARA);
        }
    }

    TEST(ExecutorGuardTest, InsRecvExecutorRejectsInvalidDataType)
    {
        TestableInsRecvExecutor executor;
        for (HcclDataType dataType : INVALID_DATA_TYPES) {
            SCOPED_TRACE(static_cast<int>(dataType));
            OpParam param = MakeP2pParam(1, dataType);
            EXPECT_EQ(executor.Orchestrate(param, AlgResourceCtxSerializable{}), HCCL_E_PARA);
        }
    }

    TEST(ExecutorGuardTest, V2SendSoleExecutorRejectsInvalidDataType)
    {
        InsV2SendSoleExecutor<InsTempSendDpu> executor;
        for (HcclDataType dataType : INVALID_DATA_TYPES) {
            SCOPED_TRACE(static_cast<int>(dataType));
            OpParam param = MakeP2pParam(1, dataType);
            EXPECT_EQ(executor.Orchestrate(param, AlgResourceCtxSerializable{}), HCCL_E_PARA);
        }
    }

    TEST(ExecutorGuardTest, V2RecvSoleExecutorRejectsInvalidDataType)
    {
        InsV2RecvSoleExecutor<InsTempRecvDpu> executor;
        for (HcclDataType dataType : INVALID_DATA_TYPES) {
            SCOPED_TRACE(static_cast<int>(dataType));
            OpParam param = MakeP2pParam(1, dataType);
            EXPECT_EQ(executor.Orchestrate(param, AlgResourceCtxSerializable{}), HCCL_E_PARA);
        }
    }

    TEST(ExecutorGuardTest, InsSendExecutorRejectsZeroLoopTransCount)
    {
        TestableInsSendExecutor executor;
        executor.SetDataTypeSize(1);
        OpParam param;
        AlgResourceCtxSerializable resCtx;
        ChannelInfo channel;
        channel.remoteCclMem.size = 0; // maxLoopTransSize_ = min(UB_MAX_DATA_SIZE, 0) = 0

        EXPECT_EQ(executor.OrchestrateOpbase(param, resCtx, 0, channel), HCCL_E_INTERNAL);
    }

    TEST(ExecutorGuardTest, InsRecvExecutorRejectsZeroLoopTransCount)
    {
        TestableInsRecvExecutor executor;
        executor.SetDataTypeSize(1);
        OpParam param;
        AlgResourceCtxSerializable resCtx;
        ChannelInfo channel;
        channel.remoteCclMem.size = 0;

        EXPECT_EQ(executor.OrchestrateOpbase(param, resCtx, 0, channel), HCCL_E_INTERNAL);
    }

    TEST(ExecutorGuardTest, V2SendSoleExecutorRejectsZeroRoundTransferSize)
    {
        InsV2SendSoleExecutor<InsTempSendDpu> executor;
        OpParam param = MakeP2pParam(1, HCCL_DATA_TYPE_INT8);
        AlgResourceCtxSerializable resCtx = MakeSoleResCtx(0, 1);
        resCtx.cclMem.size = 0; // maxRoundTransferSize = (0 / 1) * 1 = 0

        EXPECT_EQ(executor.Orchestrate(param, resCtx), HCCL_E_INTERNAL);
    }

    TEST(ExecutorGuardTest, V2RecvSoleExecutorRejectsZeroRoundTransferSize)
    {
        InsV2RecvSoleExecutor<InsTempRecvDpu> executor;
        OpParam param = MakeP2pParam(1, HCCL_DATA_TYPE_INT8);
        AlgResourceCtxSerializable resCtx = MakeSoleResCtx(0, 1);
        resCtx.cclMem.size = 0;

        EXPECT_EQ(executor.Orchestrate(param, resCtx), HCCL_E_INTERNAL);
    }

} // namespace
} // namespace ops_hccl
