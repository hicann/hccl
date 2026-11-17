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
#include <mockcpp/mockcpp.hpp>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "hcomm_stub.h"
#include "log.h"
#include "op_common.h"
#include "securec.h"

using namespace ops_hccl;
using namespace ut_stub;

class CcuResourceExpansionTest : public testing::Test {
protected:
    void SetUp() override
    {
        errorLogCount_ = 0;
        g_resources = MockResources{};
        auto original = KernelNumbers();
        original[HCOMM_CCU_RES_TYPE_VARIABLE] = ORIGINAL_CAPACITY;
        original[HCOMM_CCU_RES_TYPE_ADDRESS] = ORIGINAL_CAPACITY;
        g_resources.instances[ORIGINAL_INS] = original;
        param_.engine = COMM_ENGINE_CCU;
        param_.opType = HCCL_CMD_ALLTOALLV;
        param_.opMode = OpMode::OPBASE;
        param_.commOpExpansionMode = HCCL_OP_EXPANSION_MODE_CCU_SCHED;
        ASSERT_EQ(strcpy_s(param_.algTag, sizeof(param_.algTag), "alltoallv_tag_a"), EOK);
        ASSERT_EQ(strcpy_s(param_.algName, sizeof(param_.algName), "CcuSchedAllToAllVSoleMeshMultiLink"), EOK);
        CcuKernelInfo kernel{};
        ASSERT_EQ(strcpy_s(kernel.kernelFuncName, sizeof(kernel.kernelFuncName), "ExpansionTestKernel"), EOK);
        kernel.kernelFunc = reinterpret_cast<void*>(&KernelForResourceQuery);
        kernel.setKernelArg(std::make_shared<CcuKernelArgBase>());
        request_.ccuKernelInfos.push_back(kernel);
        request_.ccuKernelNum = {1};
        context_ = std::make_unique<AlgResourceCtxSerializable>();
        MOCKER(HcclGetThread).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER(HcclGetChannelForCcu).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER(HcclCheckLogLevel).stubs().will(invoke(CheckLogLevel));
    }

    void TearDown() override { GlobalMockObject::verify(); }

    // 关闭实际日志输出，同时检查代码是否进入ERROR分支。
    static bool CheckLogLevel(int logType, int)
    {
        if (logType == HCCL_LOG_ERROR) {
            ++errorLogCount_;
        }
        return false;
    }

    inline static uint32_t errorLogCount_ = 0;

    HcclComm Comm() { return static_cast<HcclComm>(&commIdentity_); }

    HcclResult Apply() { return HcclAllocAlgResourceCcu(Comm(), param_, request_, context_, ResPackGraphMode{}); }

    void CacheExtra(bool sufficient = true)
    {
        g_resources.instances[EXTRA_INS] = sufficient ? KernelNumbers() : g_resources.instances.at(ORIGINAL_INS);
        g_resources.oldKernelOwner = EXTRA_INS;
        ExpansionCtx record;
        record.insHandle = EXTRA_INS;
        auto& bytes = g_resources.contexts[MakeKey(Comm(), EXTRA_TAG, COMM_ENGINE_CCU)];
        bytes.resize(sizeof(record));
        ASSERT_EQ(memcpy_s(bytes.data(), bytes.size(), &record, sizeof(record)), EOK);
    }

    void ExpectExtraPublished()
    {
        auto it = g_resources.contexts.find(MakeKey(Comm(), EXTRA_TAG, COMM_ENGINE_CCU));
        ASSERT_NE(it, g_resources.contexts.end());
        ASSERT_GE(it->second.size(), sizeof(ExpansionCtx));
        ExpansionCtx record;
        ASSERT_EQ(memcpy_s(&record, sizeof(record), it->second.data(), sizeof(record)), EOK);
        EXPECT_EQ(record.insHandle, EXTRA_INS);
        EXPECT_EQ(g_resources.assignCalls, 0u);
        EXPECT_NE(g_resources.cleanupCallback, nullptr);
        EXPECT_EQ(g_resources.callbackArgs, nullptr);
    }

    void ExpectExtraRegistered()
    {
        ASSERT_EQ(g_resources.registerTargets, std::vector<CcuInsHandle>{EXTRA_INS});
        ASSERT_EQ(context_->ccuKernels.size(), 1u);
        EXPECT_NE(context_->ccuKernels.front(), 0u);
    }

    int commIdentity_ = 0;
    OpParam param_{};
    AlgResourceRequest request_{};
    std::unique_ptr<AlgResourceCtxSerializable> context_;
};

TEST_F(CcuResourceExpansionTest, ExtraInstanceReceivesFullRequirement)
{
    ASSERT_EQ(Apply(), HCCL_SUCCESS);
    ASSERT_EQ(g_resources.createRequests.size(), 1u);
    const ResourceNumbers expected = {
        {HCOMM_CCU_RES_TYPE_LOOP, 16},
        {HCOMM_CCU_RES_TYPE_CCU_BUF, 128},
        {HCOMM_CCU_RES_TYPE_VARIABLE, KERNEL_NEED},
        {HCOMM_CCU_RES_TYPE_ADDRESS, KERNEL_NEED},
        {HCOMM_CCU_RES_TYPE_EVENT, 48},
        {HCOMM_CCU_RES_TYPE_CCU_THREAD, 2},
    };
    EXPECT_EQ(g_resources.createRequests.front(), expected);
    ExpectExtraPublished();
    ExpectExtraRegistered();
    EXPECT_TRUE(g_resources.descriptors.empty());
    ASSERT_EQ(g_resources.releaseCounts.size(), g_resources.nextDesc - 1);
    for (const auto& released : g_resources.releaseCounts) {
        EXPECT_EQ(released.second, 1u) << "descriptor=" << released.first;
    }
}

TEST_F(CcuResourceExpansionTest, MsModeCreatesExtraInstance)
{
    param_.commOpExpansionMode = HCCL_OP_EXPANSION_MODE_CCU_MS;
    ASSERT_EQ(Apply(), HCCL_SUCCESS);
    ASSERT_EQ(g_resources.createRequests.size(), 1u);
    const ResourceNumbers expected = {
        {HCOMM_CCU_RES_TYPE_LOOP, 128},
        {HCOMM_CCU_RES_TYPE_CCU_BUF, 1024},
        {HCOMM_CCU_RES_TYPE_VARIABLE, KERNEL_NEED},
        {HCOMM_CCU_RES_TYPE_ADDRESS, KERNEL_NEED},
        {HCOMM_CCU_RES_TYPE_EVENT, 160},
        {HCOMM_CCU_RES_TYPE_CCU_THREAD, 2},
    };
    EXPECT_EQ(g_resources.createRequests.front(), expected);
    ExpectExtraPublished();
    ExpectExtraRegistered();
    EXPECT_TRUE(g_resources.descriptors.empty());
}

TEST_F(CcuResourceExpansionTest, NewAlgorithmTagReusesExtraInstance)
{
    CacheExtra();
    ASSERT_EQ(strcpy_s(param_.algTag, sizeof(param_.algTag), "alltoallv_tag_b"), EOK);
    ASSERT_EQ(Apply(), HCCL_SUCCESS);
    ExpectExtraRegistered();
    EXPECT_TRUE(g_resources.createRequests.empty());
    ASSERT_FALSE(g_resources.contextQueries.empty());
    EXPECT_TRUE(
        std::all_of(g_resources.contextQueries.begin(), g_resources.contextQueries.end(), [](const std::string& tag) {
            return tag == EXTRA_TAG;
        }));
}

TEST_F(CcuResourceExpansionTest, InsufficientExtraDoesNotCreateThirdInstance)
{
    CacheExtra(false);
    g_resources.instances[ORIGINAL_INS] = KernelNumbers();
    EXPECT_EQ(Apply(), HCCL_E_UNAVAIL);
    EXPECT_EQ(g_resources.queriedInstances, std::vector<CcuInsHandle>{EXTRA_INS});
    EXPECT_TRUE(g_resources.createRequests.empty());
    EXPECT_TRUE(g_resources.registerTargets.empty());
}

TEST_F(CcuResourceExpansionTest, InsufficientDeviceDoesNotCreateInstance)
{
    g_resources.remain = ORIGINAL_CAPACITY;
    EXPECT_EQ(Apply(), HCCL_E_UNAVAIL);
    EXPECT_NE(
        std::find(g_resources.remainQueryDies.begin(), g_resources.remainQueryDies.end(), 0u),
        g_resources.remainQueryDies.end());
    EXPECT_TRUE(g_resources.createRequests.empty());
    EXPECT_TRUE(g_resources.contexts.empty());
}

TEST_F(CcuResourceExpansionTest, OriginalInstanceInsufficientDoesNotLogError)
{
    MOCKER(HcclCommQueryAssignedCcuIns).stubs().will(returnValue(HCCL_E_UNAVAIL));
    g_resources.instances.clear();
    g_resources.remain = ORIGINAL_CAPACITY;

    EXPECT_EQ(Apply(), HCCL_E_UNAVAIL);
    EXPECT_EQ(errorLogCount_, 0u);
    EXPECT_TRUE(g_resources.createRequests.empty());
    EXPECT_EQ(g_resources.assignCalls, 0u);
    EXPECT_TRUE(g_resources.registerTargets.empty());
    EXPECT_TRUE(g_resources.descriptors.empty());
}

TEST_F(CcuResourceExpansionTest, OriginalInstanceCreateErrorStillLogsError)
{
    MOCKER(HcclCommQueryAssignedCcuIns).stubs().will(returnValue(HCCL_E_UNAVAIL));
    MOCKER(HcommCcuInsCreate).stubs().will(returnValue(CCU_E_INTERNAL));
    g_resources.instances.clear();

    EXPECT_EQ(Apply(), HCCL_E_INTERNAL);
    EXPECT_GT(errorLogCount_, 0u);
    EXPECT_EQ(g_resources.assignCalls, 0u);
    EXPECT_TRUE(g_resources.registerTargets.empty());
    EXPECT_TRUE(g_resources.descriptors.empty());
}

TEST_F(CcuResourceExpansionTest, RegisterErrorPreservesExistingExtraAndOldKernel)
{
    CacheExtra();
    g_resources.registerResult = CCU_E_INTERNAL;
    EXPECT_EQ(Apply(), HCCL_E_INTERNAL);
    EXPECT_EQ(g_resources.registerTargets, std::vector<CcuInsHandle>{EXTRA_INS});
    EXPECT_TRUE(g_resources.destroyedInstances.empty());
    EXPECT_EQ(g_resources.contexts.count(MakeKey(Comm(), EXTRA_TAG, COMM_ENGINE_CCU)), 1u);
    EXPECT_EQ(g_resources.oldKernels, std::vector<CcuKernelHandle>{OLD_KERNEL});
    EXPECT_EQ(g_resources.contexts.count(MakeKey(Comm(), param_.algTag, COMM_ENGINE_CCU)), 0u);
}

TEST_F(CcuResourceExpansionTest, OtherRankFailureReturnsUnavailableAfterLocalExpansion)
{
    g_resources.negotiationResult = HCCL_E_UNAVAIL;
    TopoInfoWithNetLayerDetails topo{};
    AlgHierarchyInfoForAllLevel hierarchy{};
    void* serialized = nullptr;
    uint64_t size = 0;
    EXPECT_EQ(
        GetAlgResWithEngine(
            Comm(), param_, request_, context_, &topo, hierarchy, &serialized, size, false, ResPackGraphMode{}),
        HCCL_E_UNAVAIL);
    EXPECT_EQ(g_resources.createRequests.size(), 1u);
    EXPECT_EQ(g_resources.registerTargets, std::vector<CcuInsHandle>{EXTRA_INS});
    EXPECT_EQ(g_resources.negotiationCalls, 1u);
    EXPECT_TRUE(g_resources.localResourcesAvailable);
    ExpectExtraPublished();
    EXPECT_TRUE(g_resources.destroyedInstances.empty());
}

TEST_F(CcuResourceExpansionTest, GraphCaptureCanCreateExtraInstance)
{
    param_.opMode = OpMode::OFFLOAD;
    MOCKER(IsStreamInCaptureMode).stubs().will(returnValue(true));
    ASSERT_EQ(Apply(), HCCL_SUCCESS);
    ExpectExtraPublished();
    ExpectExtraRegistered();
}

TEST_F(CcuResourceExpansionTest, Mock960CanCreateExtraInstance)
{
    g_resources.device = HcclDevType::DEV_TYPE_960;
    ASSERT_EQ(Apply(), HCCL_SUCCESS);
    EXPECT_GT(g_resources.deviceQueries, 0u);
    ExpectExtraPublished();
    ExpectExtraRegistered();
}

TEST_F(CcuResourceExpansionTest, UnsupportedDeviceDoesNotCreateExtraInstance)
{
    g_resources.device = HcclDevType::DEV_TYPE_NOSOC;
    EXPECT_EQ(Apply(), HCCL_E_UNAVAIL);
    EXPECT_EQ(g_resources.queriedInstances, std::vector<CcuInsHandle>{ORIGINAL_INS});
    EXPECT_EQ(g_resources.contextQueries, std::vector<std::string>{EXTRA_TAG});
    EXPECT_TRUE(g_resources.createRequests.empty());
    EXPECT_TRUE(g_resources.contexts.empty());
}

TEST_F(CcuResourceExpansionTest, DeviceQueryFailureKeepsOriginalInstance)
{
    MOCKER(HcclGetDeviceType).stubs().will(returnValue(HCCL_E_RUNTIME));
    g_resources.instances[ORIGINAL_INS] = KernelNumbers();

    EXPECT_EQ(Apply(), HCCL_SUCCESS);
    EXPECT_EQ(g_resources.registerTargets, std::vector<CcuInsHandle>{ORIGINAL_INS});
    EXPECT_EQ(g_resources.contextQueries, std::vector<std::string>{EXTRA_TAG});
    EXPECT_TRUE(g_resources.createRequests.empty());
    EXPECT_TRUE(g_resources.contexts.empty());
}

TEST_F(CcuResourceExpansionTest, DeviceQueryFailureDoesNotCreateExtraInstance)
{
    MOCKER(HcclGetDeviceType).stubs().will(returnValue(HCCL_E_RUNTIME));

    EXPECT_EQ(Apply(), HCCL_E_UNAVAIL);
    EXPECT_EQ(g_resources.queriedInstances, std::vector<CcuInsHandle>{ORIGINAL_INS});
    EXPECT_EQ(g_resources.contextQueries, std::vector<std::string>{EXTRA_TAG});
    EXPECT_TRUE(g_resources.createRequests.empty());
    EXPECT_TRUE(g_resources.contexts.empty());
}

TEST_F(CcuResourceExpansionTest, MissingCallbackSymbolKeepsOriginalInstance)
{
    g_resources.callbackSupported = false;
    g_resources.instances[ORIGINAL_INS] = KernelNumbers();

    ASSERT_EQ(Apply(), HCCL_SUCCESS);

    EXPECT_EQ(g_resources.registerTargets, std::vector<CcuInsHandle>{ORIGINAL_INS});
    EXPECT_TRUE(g_resources.createRequests.empty());
    EXPECT_TRUE(g_resources.contexts.empty());
}

TEST_F(CcuResourceExpansionTest, MissingCallbackSymbolDoesNotCreateExtra)
{
    g_resources.callbackSupported = false;

    EXPECT_EQ(Apply(), HCCL_E_UNAVAIL);

    EXPECT_TRUE(g_resources.createRequests.empty());
    EXPECT_TRUE(g_resources.registerTargets.empty());
    EXPECT_TRUE(g_resources.contexts.empty());
}

TEST_F(CcuResourceExpansionTest, CallbackRegistrationErrorDoesNotAllocateInstance)
{
    g_resources.callbackRegistrationResult = HCCL_E_MEMORY;
    EXPECT_EQ(Apply(), HCCL_E_UNAVAIL);
    EXPECT_TRUE(g_resources.createRequests.empty());
    EXPECT_TRUE(g_resources.contexts.empty());
    EXPECT_TRUE(g_resources.registerTargets.empty());
    EXPECT_TRUE(g_resources.destroyedInstances.empty());
    EXPECT_EQ(g_resources.instances.count(ORIGINAL_INS), 1u);
}

TEST_F(CcuResourceExpansionTest, ContextCreationErrorRollsBackNewInstance)
{
    g_resources.contextCreateResult = HCCL_E_MEMORY;
    EXPECT_EQ(Apply(), HCCL_E_MEMORY);
    EXPECT_EQ(g_resources.destroyedInstances, std::vector<CcuInsHandle>{EXTRA_INS});
    EXPECT_TRUE(g_resources.contexts.empty());
    EXPECT_TRUE(g_resources.registerTargets.empty());
}

TEST_F(CcuResourceExpansionTest, AllGatherReusesExtraInstance)
{
    MOCKER(HcclGetDeviceType).expects(never());
    CacheExtra();
    param_.opType = HCCL_CMD_ALLGATHER;
    ASSERT_EQ(Apply(), HCCL_SUCCESS);
    EXPECT_EQ(g_resources.queriedInstances, std::vector<CcuInsHandle>{EXTRA_INS});
    ExpectExtraRegistered();
    EXPECT_TRUE(g_resources.createRequests.empty());
    EXPECT_EQ(g_resources.assignCalls, 0u);
    EXPECT_TRUE(g_resources.destroyedInstances.empty());
}

TEST_F(CcuResourceExpansionTest, AllGatherDoesNotCreateExtraInstance)
{
    MOCKER(HcclGetDeviceType).expects(never());
    param_.opType = HCCL_CMD_ALLGATHER;
    EXPECT_EQ(Apply(), HCCL_E_UNAVAIL);
    EXPECT_EQ(g_resources.queriedInstances, std::vector<CcuInsHandle>{ORIGINAL_INS});
    EXPECT_TRUE(g_resources.createRequests.empty());
    EXPECT_TRUE(g_resources.registerTargets.empty());
    EXPECT_TRUE(g_resources.contexts.empty());
}

TEST_F(CcuResourceExpansionTest, DeviceQueryFailureReusesExtraInstance)
{
    MOCKER(HcclGetDeviceType).stubs().will(returnValue(HCCL_E_RUNTIME));
    CacheExtra();
    ASSERT_EQ(Apply(), HCCL_SUCCESS);
    EXPECT_EQ(g_resources.queriedInstances, std::vector<CcuInsHandle>{EXTRA_INS});
    ExpectExtraRegistered();
    EXPECT_TRUE(g_resources.createRequests.empty());
}

TEST_F(CcuResourceExpansionTest, DestroyPreReleasesExtraOnceAndLeavesContextForHcomm)
{
    ASSERT_EQ(Apply(), HCCL_SUCCESS);
    ASSERT_NE(g_resources.cleanupCallback, nullptr);
    EXPECT_EQ(g_resources.cleanupCallback(Comm(), HCCL_COMM_STATE_PHASE_DESTROY_PRE, nullptr), HCCL_SUCCESS);
    EXPECT_EQ(g_resources.cleanupCallback(Comm(), HCCL_COMM_STATE_PHASE_DESTROY_PRE, nullptr), HCCL_SUCCESS);
    EXPECT_EQ(g_resources.destroyedInstances, std::vector<CcuInsHandle>{EXTRA_INS});
    auto it = g_resources.contexts.find(MakeKey(Comm(), EXTRA_TAG, COMM_ENGINE_CCU));
    ASSERT_NE(it, g_resources.contexts.end());
    ExpansionCtx record{};
    ASSERT_EQ(memcpy_s(&record, sizeof(record), it->second.data(), sizeof(record)), EOK);
    EXPECT_EQ(record.insHandle, 0u);
    EXPECT_EQ(g_resources.instances.count(ORIGINAL_INS), 1u);
}

TEST_F(CcuResourceExpansionTest, ClearedExtraInstanceReturnsUnavailable)
{
    ASSERT_EQ(Apply(), HCCL_SUCCESS);
    ASSERT_NE(g_resources.cleanupCallback, nullptr);
    ASSERT_EQ(g_resources.cleanupCallback(Comm(), HCCL_COMM_STATE_PHASE_DESTROY_PRE, nullptr), HCCL_SUCCESS);
    ASSERT_EQ(strcpy_s(param_.algTag, sizeof(param_.algTag), "alltoallv_after_cleanup"), EOK);

    EXPECT_EQ(Apply(), HCCL_E_UNAVAIL);
    EXPECT_EQ(g_resources.createRequests.size(), 1u);
    EXPECT_EQ(g_resources.registerTargets, std::vector<CcuInsHandle>{EXTRA_INS});
    EXPECT_EQ(g_resources.queriedInstances, std::vector<CcuInsHandle>{ORIGINAL_INS});
    EXPECT_EQ(g_resources.contexts.count(MakeKey(Comm(), EXTRA_TAG, COMM_ENGINE_CCU)), 1u);
    EXPECT_EQ(g_resources.instances.count(EXTRA_INS), 0u);
    EXPECT_TRUE(g_resources.descriptors.empty());
}

TEST_F(CcuResourceExpansionTest, CommNameStubRejectsNullPointers)
{
    char name[COMM_INDENTIFIER_MAX_LENGTH] = {};
    EXPECT_EQ(HcclGetCommName(Comm(), nullptr), HCCL_E_PTR);
    EXPECT_EQ(HcclGetCommName(nullptr, name), HCCL_E_PTR);
}

TEST_F(CcuResourceExpansionTest, OtherCallbackPhasesKeepExtraInstance)
{
    ASSERT_EQ(Apply(), HCCL_SUCCESS);
    ASSERT_NE(g_resources.cleanupCallback, nullptr);
    for (auto phase :
         {HCCL_COMM_STATE_PHASE_DESTROY_POST, HCCL_COMM_STATE_PHASE_RESUME_PRE, HCCL_COMM_STATE_PHASE_RESUME_POST}) {
        EXPECT_EQ(g_resources.cleanupCallback(Comm(), phase, nullptr), HCCL_SUCCESS);
    }
    EXPECT_TRUE(g_resources.destroyedInstances.empty());
    ExpectExtraPublished();
}

TEST_F(CcuResourceExpansionTest, OtherCommunicationDomainHasNoExtraAndIsSkipped)
{
    ASSERT_EQ(Apply(), HCCL_SUCCESS);
    ASSERT_NE(g_resources.cleanupCallback, nullptr);
    int other = 0;
    EXPECT_EQ(g_resources.cleanupCallback(&other, HCCL_COMM_STATE_PHASE_DESTROY_PRE, nullptr), HCCL_SUCCESS);
    EXPECT_TRUE(g_resources.destroyedInstances.empty());
    ExpectExtraPublished();
}

TEST_F(CcuResourceExpansionTest, DestroyFailureDoesNotBlockCommDestroy)
{
    ASSERT_EQ(Apply(), HCCL_SUCCESS);
    ASSERT_NE(g_resources.cleanupCallback, nullptr);
    g_resources.destroyResult = CCU_E_INTERNAL;
    EXPECT_EQ(g_resources.cleanupCallback(Comm(), HCCL_COMM_STATE_PHASE_DESTROY_PRE, nullptr), HCCL_SUCCESS);
    ExpectExtraPublished();
    EXPECT_EQ(g_resources.destroyedInstances, (std::vector<CcuInsHandle>{EXTRA_INS}));
}

// 复现当前清理限制：PRE释放失败后，context被底座回收，POST不会补偿释放额外实例。
TEST_F(CcuResourceExpansionTest, DestroyFailureDoesNotRetryAfterContextRelease)
{
    ASSERT_EQ(Apply(), HCCL_SUCCESS);
    ASSERT_NE(g_resources.cleanupCallback, nullptr);
    g_resources.destroyResult = CCU_E_INTERNAL;
    ASSERT_EQ(g_resources.cleanupCallback(Comm(), HCCL_COMM_STATE_PHASE_DESTROY_PRE, nullptr), HCCL_SUCCESS);
    ExpectExtraPublished();

    // 模拟底座继续回收context；随后释放接口恢复正常，检查真实HCCL回调是否再次尝试释放。
    ASSERT_EQ(HcclEngineCtxDestroy(Comm(), EXTRA_TAG, COMM_ENGINE_CCU), HCCL_SUCCESS);
    g_resources.destroyResult = CCU_SUCCESS;
    EXPECT_EQ(g_resources.cleanupCallback(Comm(), HCCL_COMM_STATE_PHASE_DESTROY_POST, nullptr), HCCL_SUCCESS);
    EXPECT_TRUE(g_resources.contexts.empty());
    EXPECT_EQ(g_resources.destroyedInstances, (std::vector<CcuInsHandle>{EXTRA_INS}));
    EXPECT_EQ(g_resources.instances.count(EXTRA_INS), 1u);
    EXPECT_EQ(g_resources.instances.count(ORIGINAL_INS), 1u);

    // 以下仅为测试清理，不代表产品自动补偿。
    EXPECT_EQ(HcommCcuInsDestroy(EXTRA_INS), CCU_SUCCESS);
}

TEST_F(CcuResourceExpansionTest, InvalidCleanupContextDoesNotBlockCommDestroy)
{
    ASSERT_EQ(Apply(), HCCL_SUCCESS);
    ASSERT_NE(g_resources.cleanupCallback, nullptr);
    g_resources.contexts.at(MakeKey(Comm(), EXTRA_TAG, COMM_ENGINE_CCU)).resize(sizeof(ExpansionCtx) + 1);
    EXPECT_EQ(g_resources.cleanupCallback(Comm(), HCCL_COMM_STATE_PHASE_DESTROY_PRE, nullptr), HCCL_SUCCESS);
    EXPECT_TRUE(g_resources.destroyedInstances.empty());
}
