/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_SRC_OPS_INC_COLL_GATHER_OMNIPIPEDATASLICECALC
#define OPS_HCCL_SRC_OPS_INC_COLL_GATHER_OMNIPIPEDATASLICECALC
#include <cmath>
#include <stdint.h>
#include <vector>
#include <string>
#include <sstream>
#include "template_utils.h"
#include "alg_template_base.h"
#include "omnipipe_data_slice_calc.h"

namespace ops_hccl {
constexpr double BW_OMNI_UBX_CCU_SCHED_R_RS_CLOS = 210;
constexpr double BW_OMNI_UBX_CCU_SCHED_G_MESH = 47;
constexpr double BW_OMNI_UBX_CCU_SCHED_G_CLOS = 162;
constexpr double BW_OMNI_UBX_CCU_MS_SCHED_G_MESH = 47;
constexpr double BW_OMNI_UBX_CCU_MS_SCHED_G_CLOS = 180;
// Gather slice 计算的共享上下文，封装步数/偏移/数据大小等中间状态，减少子函数参数列表
struct GatherSliceContext {
    u64 xRankSize{0};
    u64 yRankSize{0};
    u64 zRankSize{0};
    u64 rankSize{0};
    u64 xAxis{0};
    u64 yAxis{0};
    u64 zAxis{0};
    double xB{0};
    double yB{0};
    double zB{0};
    double xyB{0};
    bool yGeX{false};
    int maxStepNum{MAX_STEP_NUM};
    CommEngine engine{CommEngine::COMM_ENGINE_AICPU_TS};
    u64 dataTypeSize{0};
    u64 maxDataPieceId{0};
    std::vector<u64> dataSize;
    std::vector<u64> dataSizePerLoop;
    std::vector<OmniPipeSplitSliceInfo> perLoop;
    std::vector<OmniPipeSplitSliceInfo> total;
    std::vector<std::vector<u64>> zGDS, xyGDS, zGOff, xyGOff;
    std::vector<std::vector<std::vector<u64>>> xGDS, yGDS, xGOff, yGOff;
    u64 outerStepNum{0};
    u64 innerStepNum{0};
    int zCornerStep{1};
    int xyCornerStep{1};
    int xInCornerStep{1};
    int yInCornerStep{1};
    u64 xCclBufOff{0};
    u64 yCclBufOff{0};
    u64 zCclBufOff{0};
};

// 初始化一个 StepSliceInfo：in/out 偏移为 0，hcclBuff 偏移为 cclBufOff
StepSliceInfo MakeGatherStep(u64 cclBufOff);
void PushGatherRankEntry(StepSliceInfo &s, u64 dataTypeSize, u64 inStride, u64 outStride,
    std::vector<u64> sz, std::vector<u64> inOff, std::vector<u64> outOff);
void InitGatherDataArrays(GatherSliceContext &ctx);
void CalcGatherStepDataAndOffset(GatherSliceContext &ctx);
void CalcGatherXY2DOffset(GatherSliceContext &ctx);
std::vector<StepSliceInfo> BuildGatherZSteps(GatherSliceContext &ctx);
void BuildGatherXInnerSteps(GatherSliceContext &ctx, std::vector<StepSliceInfo> &out);
void BuildGatherXOuterSteps(GatherSliceContext &ctx, std::vector<StepSliceInfo> &out);
void BuildGatherYInnerSteps(GatherSliceContext &ctx, std::vector<StepSliceInfo> &out);
void BuildGatherYOuterSteps(GatherSliceContext &ctx, std::vector<StepSliceInfo> &out);
OmniPipeSliceInfo CalcGatherOmniPipeSliceInfo(OmniPipeSliceParam& omniPipeSliceParam);
void CollectGatherInnerCornerPieces(const GatherSliceContext &ctx, u64 osn, u64 isn, u64 oneDid,
    bool isX, std::vector<u64> &sz, std::vector<u64> &inOff, std::vector<u64> &outOff);
void CollectGatherOuterSameAxisPieces(const GatherSliceContext &ctx, u64 osn, u64 isn, u64 oneDid,
    bool isX, std::vector<u64> &sz, std::vector<u64> &inOff, std::vector<u64> &outOff);
void CollectGatherOuterCornerPieces(const GatherSliceContext &ctx, u64 osn, u64 isn, u64 oneDid,
    bool isX, std::vector<u64> &sz, std::vector<u64> &inOff, std::vector<u64> &outOff);
}  // namespace ops_hccl
#endif
