/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NHR_TEST_ACCESS_H
#define NHR_TEST_ACCESS_H

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

// 被测几何/寻址接口为模板私有方法，UT 目标通过 -fno-access-control 放开访问权限
#include "ins_temp_reduce_scatter_nhr_dpu.h"

namespace nhr_ut {
using namespace ops_hccl;

// LocalCopy 桩记录的一次拷贝调用
struct CopyCall {
    void* srcAddr = nullptr;
    u64 srcOffset = 0;
    u64 srcSize = 0;
    u64 srcCount = 0;
    void* dstAddr = nullptr;
    u64 dstOffset = 0;
    u64 dstSize = 0;
    u64 dstCount = 0;
};

std::vector<CopyCall>& CopyCalls();

// batch 模式窗口计数：用于固化 KernelRun 的 Start/End 次数
struct BatchModeStats {
    u32 startCount = 0;
    u32 endCount = 0;
};

BatchModeStats& BatchModeCounts();

// CostModelManager 桩捕获的入参与返回系数：用于覆盖 CalcCostCoeff 的参数传递与引擎选择
struct CostCapture {
    float meshN = 0.0f;
    CommTopo meshNetType = CommTopo::COMM_TOPO_1DMESH;
    int meshPortNum = -1;
    u32 meshRankSize = 0;
    bool meshIsPod = false;
    EngineType meshEngine = EngineType::AICPU;
    float meshA = 1.5f;
    float localCopyN = 0.0f;
    EngineType localCopyEngine = EngineType::AICPU;
    float localCopyB = 2.5f;
    int dpuStepNum = 0;
    int dpuSyncNum = 0;
    int dpuChannelNum = 0;
    int dpuSndRcvNum = 0;
    float dpuLatencyC = 3.5f;
};

CostCapture& CostCalls();

// 原语返回值注入，用于覆盖 KernelRun 的错误路径
void SetBatchModeStartResult(s32 result);
void SetBatchModeEndResult(s32 result);
void SetThreadSyncResult(s32 result);
void SetSendRequestResult(s32 result);
void SetWaitResponseResult(s32 result);
void SetSendMsgId(u32 msgId);
void SetWaitResponseMsgId(u32 msgId);
void ResetStubState();

inline void SetupTemplate(InsTempReduceScatterNhrDpu& tpl, u32 myRank, const std::vector<u32>& rankIds)
{
    tpl.myRank_ = myRank;
    tpl.subCommRanks_ = {rankIds};
    tpl.templateRankSize_ = static_cast<u32>(rankIds.size());
}

inline std::vector<u32> MakeRankIds(u32 rankSize)
{
    std::vector<u32> rankIds(rankSize);
    for (u32 i = 0; i < rankSize; ++i) {
        rankIds[i] = i;
    }
    return rankIds;
}

inline TemplateDataParams MakeStrideParams()
{
    TemplateDataParams params;
    params.sliceSize = 16;
    params.count = 4;
    params.inputSliceStride = 32;
    params.outputSliceStride = 16;
    params.inputRepeatStride = 1000;
    params.outputRepeatStride = 70;
    params.repeatNum = 2;
    params.buffInfo.inputPtr = reinterpret_cast<void*>(0x1000);
    params.buffInfo.outputPtr = reinterpret_cast<void*>(0x3000);
    params.buffInfo.hcclBuff.type = HcclMemType::HCCL_MEM_TYPE_DEVICE;
    params.buffInfo.hcclBuff.addr = reinterpret_cast<void*>(0x2000);
    params.buffInfo.hcclBuff.size = 4096;
    params.buffInfo.inBuffBaseOff = 8;
    params.buffInfo.outBuffBaseOff = 4;
    params.buffInfo.hcclBuffBaseOff = 40;
    return params;
}

} // namespace nhr_ut

#endif // NHR_TEST_ACCESS_H
