/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_SRC_OPS_INC_COLL_SCATTER_OMNIPIPEDATASLICECALC
#define OPS_HCCL_SRC_OPS_INC_COLL_SCATTER_OMNIPIPEDATASLICECALC
#include <cmath>
#include <stdint.h>
#include <vector>
#include <string>
#include <sstream>
#include "template_utils.h"
#include "alg_template_base.h"
#include "omnipipe_data_slice_calc.h"

namespace ops_hccl {
constexpr double BW_OMNI_UBX_CCU_SCHED_SC_MESH = 47;
constexpr double BW_OMNI_UBX_CCU_SCHED_SC_CLOS = 175;
constexpr double BW_OMNI_UBX_AICPU_SC_CLOS = 256;
constexpr u64 MAX_RANK_SIZE = 2048;
constexpr u64 MAX_STEP_NUM_SC = 3;

// Scatter专用数据切分和偏移计算的相关函数
struct ScatterTopoInfo {
    u64 xRankSize;
    u64 yRankSize;
    u64 zRankSize;
    u64 rankSize;
    u64 xAxis;
    u64 yAxis;
    u64 zAxis;
    u64 rankid;
    u64 rootx;
    u64 rooty;
    u64 rootz;
    double xB;
    double yB;
    double zB;
    double xyB;
    u64 maxDataPieceId;
    u64 dataTypeSize;
    u64 processedDataEachRank;
};

struct ScatterStepState {
    u64 outerStepNum;
    u64 innerStepNum;
    u64 xyCornerStep;
    u64 xInCornerStep;
    u64 yInCornerStep;
    u64 zCornerStep;
    bool isZSlowAxis;
    bool isXSlowAxis;
};

// 4-vector 样板聚合：统一 sz/cnt/in/out 四个 vector 的声明，消除重复声明
struct ScatterPieceVecs {
    std::vector<u64> sz;
    std::vector<u64> cnt;
    std::vector<u64> in;
    std::vector<u64> out;
};
void CalScatter2DOffset(
    u64* xSOffset, u64* ySOffset, u64 stepNum, u64 xRankSize, u64 yRankSize, u64* xSDataSize, u64* ySDataSize);
void CalcScatterStepAndScale(
    double bandwidthRatio, double omniPipeRatio, u64 xRankSize, u64 maxStep, u64& step, double& scale);
void CalcScatterFirstStepSize(
    u64* xStepP2pDataSize, u64* yStepP2pDataSize, double bandwidthRatio, u64 xRankSize, u64 yRankSize,
    u64 dataSizeEachRank, double scale, u64 step);
void CalcScatterMidStepsSize(
    u64* xStepP2pDataSize, u64* yStepP2pDataSize, double bandwidthRatio, u64 xRankSize, u64 dataSizeEachRank, u64 step,
    u64& sumXDataSize, u64& sumYDataSize);
u64 CalScatterDataSize2D(
    u64* xStepP2pDataSize, u64* yStepP2pDataSize, double xB, double yB, u64 xRankSize, u64 yRankSize,
    u64 dataSizeEachRank, u64 maxStep);
void CheckRootOrSameAxisAsRoot(
    u64 xRankSize, u64 yRankSize, u64 zRankSize, uint32_t root, uint32_t rankId, bool& ifRoot, bool& ifSameAxisAsRoot);
void PushStepFields(
    StepSliceInfo& s, const std::vector<u64>& sz, const std::vector<u64>& cnt, const std::vector<u64>& in,
    const std::vector<u64>& out, u64 inStride, u64 outStride);
void PushStepZeros(StepSliceInfo& s, u64 n, u64 inStride, u64 outStride);
void PushRootOrZeros(
    StepSliceInfo& s, const std::vector<u64>& sz, const std::vector<u64>& cnt, const std::vector<u64>& in,
    const std::vector<u64>& out, u64 peerIdx, u64 peerRoot, u64 outStride);
void CalcAndPushPiece(
    u64 pieceId, u64 xyBaseOffset, u64 sDataSize, const std::vector<OmniPipeSplitSliceInfo>& perLoop,
    const std::vector<OmniPipeSplitSliceInfo>& total, u64 dataTypeSize, std::vector<u64>& sz, std::vector<u64>& cnt,
    std::vector<u64>& in, std::vector<u64>& out);
void ZeroInitScatterDataArrays(
    u64 rankSize, u64 zSDataSize[][MAX_STEP_NUM_SC], u64 xySDataSize[][MAX_STEP_NUM_SC],
    u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 zSOffset[][MAX_STEP_NUM_SC], u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 xySOffset[][MAX_STEP_NUM_SC]);
void CalcScatterOneRankDataSize(
    const ScatterTopoInfo& topo, ScatterStepState& state, u64 rs, u64 finStepMark, double slowBw, double fastBw,
    u64 slowRankSize, u64 fastRankSize, u64 zSDataSize[][MAX_STEP_NUM_SC], u64 xySDataSize[][MAX_STEP_NUM_SC],
    u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 zSOffset[][MAX_STEP_NUM_SC], u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 xySOffset[][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& omniPipeSplitSliceInfoListPerLoop);
void CalcScatterOuterCornerStep(const ScatterTopoInfo& topo, ScatterStepState& state, u64 finStepMark);
void CalcScatterAllRankDataSize(
    const ScatterTopoInfo& topo, ScatterStepState& state, uint32_t root, u64 finStepMark,
    u64 zSDataSize[][MAX_STEP_NUM_SC], u64 xySDataSize[][MAX_STEP_NUM_SC],
    u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 zSOffset[][MAX_STEP_NUM_SC], u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 xySOffset[][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& omniPipeSplitSliceInfoListPerLoop);
void PushScatterXAllSteps(
    std::vector<StepSliceInfo>& dataSliceLevelx, const ScatterTopoInfo& topo, const ScatterStepState& state,
    u64 zSDataSize[][MAX_STEP_NUM_SC], u64 xySDataSize[][MAX_STEP_NUM_SC],
    u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 zSOffset[][MAX_STEP_NUM_SC], u64 xySOffset[][MAX_STEP_NUM_SC], u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& omniPipeSplitSliceInfoListPerLoop,
    const std::vector<OmniPipeSplitSliceInfo>& omniPipeSplitSliceInfoListTotal, uint32_t root, u64 xCclBufferBaseOff,
    u64 yCclBufferBaseOff, const std::vector<u64>& dataSizePerLoop);
void PushScatterYAllSteps(
    std::vector<StepSliceInfo>& dataSliceLevely, const ScatterTopoInfo& topo, const ScatterStepState& state,
    u64 zSDataSize[][MAX_STEP_NUM_SC], u64 xySDataSize[][MAX_STEP_NUM_SC],
    u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 zSOffset[][MAX_STEP_NUM_SC], u64 xySOffset[][MAX_STEP_NUM_SC], u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& omniPipeSplitSliceInfoListPerLoop,
    const std::vector<OmniPipeSplitSliceInfo>& omniPipeSplitSliceInfoListTotal, uint32_t root, u64 xCclBufferBaseOff,
    u64 yCclBufferBaseOff, const std::vector<u64>& dataSizePerLoop);
OmniPipeSliceInfo CalcScatterOmniPipeSliceInfo(OmniPipeSliceParam& omniPipeSliceParam, uint32_t root);
void PushScatterZDiagSteps(
    std::vector<StepSliceInfo>& dataSliceLevelz, u64 zSDataSize[][MAX_STEP_NUM_SC], u64 zSOffset[][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    const ScatterTopoInfo& topo, u64 zCclBufferBaseOff, u64 zCornerStep, uint32_t root);
void PushScatterZSameAxisSteps(
    std::vector<StepSliceInfo>& dataSliceLevelz, u64 zSDataSize[][MAX_STEP_NUM_SC], u64 xySDataSize[][MAX_STEP_NUM_SC],
    u64 zSOffset[][MAX_STEP_NUM_SC], u64 xySOffset[][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    const ScatterTopoInfo& topo, u64 zCclBufferBaseOff, u64 zCornerStep, u64 outerStepNum, uint32_t root);
void PushScatterZDiagStepsZgXY(
    std::vector<StepSliceInfo>& dataSliceLevelz, u64 zSDataSize[][MAX_STEP_NUM_SC], u64 zSOffset[][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    const ScatterTopoInfo& topo, u64 zCclBufferBaseOff, u64 zCornerStep, uint32_t root);
void PushScatterZSameAxisStepsZgXY(
    std::vector<StepSliceInfo>& dataSliceLevelz, u64 zSDataSize[][MAX_STEP_NUM_SC], u64 xySDataSize[][MAX_STEP_NUM_SC],
    u64 zSOffset[][MAX_STEP_NUM_SC], u64 xySOffset[][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    const ScatterTopoInfo& topo, u64 zCclBufferBaseOff, u64 zCornerStep, u64 outerStepNum, uint32_t root);
void PushScatterXInnerCornerOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevelx, u64 osn, u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 xySOffset[][MAX_STEP_NUM_SC], u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    const ScatterTopoInfo& topo, const std::vector<u64>& dataSizePerLoop, u64 xCclBufferBaseOff, u64 xInCornerStep,
    uint32_t root);
void PushScatterXInnerSameAxisOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevelx, u64 osn, u64 xySDataSize[][MAX_STEP_NUM_SC],
    u64 zSDataSize[][MAX_STEP_NUM_SC], u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 xySOffset[][MAX_STEP_NUM_SC],
    u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    const ScatterTopoInfo& topo, const std::vector<u64>& dataSizePerLoop, u64 xCclBufferBaseOff, u64 xInCornerStep,
    u64 innerStepNum, uint32_t root);
void PushScatterXOverSameAxisOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevelx, u64 osn, u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 xySOffset[][MAX_STEP_NUM_SC],
    u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    const ScatterTopoInfo& topo, const std::vector<u64>& dataSizePerLoop, u64 xCclBufferBaseOff, u64 xInCornerStep,
    u64 innerStepNum, uint32_t root);
void PushScatterXOuterLECornerOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevelx, u64 osn, u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 xySOffset[][MAX_STEP_NUM_SC], u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    const ScatterTopoInfo& topo, uint32_t root, u64 xCclBufferBaseOff, u64 xInCornerStep);
void PushScatterXOuterLEFwdOneRank(
    StepSliceInfo& stepSliceInfotmp, u64 xySDataSize[][MAX_STEP_NUM_SC],
    u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 xySOffset[][MAX_STEP_NUM_SC], u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], const std::vector<OmniPipeSplitSliceInfo>& perLoop, uint32_t root,
    u64 osn, u64 isn, u64 innerStepNum, const ScatterTopoInfo& topo, u64 oneDid,
    const std::vector<u64>& dataSizePerLoop);
void PushScatterXOuterLESameAxisOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevelx, u64 osn, u64 xySDataSize[][MAX_STEP_NUM_SC],
    u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 xySOffset[][MAX_STEP_NUM_SC], u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], const std::vector<OmniPipeSplitSliceInfo>& perLoop,
    const std::vector<OmniPipeSplitSliceInfo>& total, const ScatterTopoInfo& topo, uint32_t root, u64 xCclBufferBaseOff,
    u64 xInCornerStep, u64 innerStepNum, const std::vector<u64>& dataSizePerLoop);
void PushScatterXOuterGTCornerOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevelx, u64 osn, u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 xySOffset[][MAX_STEP_NUM_SC], u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    const ScatterTopoInfo& topo, uint32_t root, u64 yCclBufferBaseOff, u64 xInCornerStep);
void PushScatterXOuterGTFwdOneRank(
    StepSliceInfo& stepSliceInfotmp, u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 xySOffset[][MAX_STEP_NUM_SC], u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, uint32_t root, u64 osn, const ScatterTopoInfo& topo, u64 one,
    u64 dataTypeSize);
void PushScatterXOuterGTSameAxisOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevelx, u64 osn, u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 xySOffset[][MAX_STEP_NUM_SC],
    u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    const ScatterTopoInfo& topo, uint32_t root, u64 yCclBufferBaseOff, u64 xInCornerStep, u64 innerStepNum);
void PushScatterYInnerCornerOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevely, u64 osn, u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 xySOffset[][MAX_STEP_NUM_SC], u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    const ScatterTopoInfo& topo, u64 yCclBufferBaseOff, u64 yInCornerStep, uint32_t root);
void PushScatterYInnerSameAxisOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevely, u64 osn, u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 xySOffset[][MAX_STEP_NUM_SC],
    u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    const ScatterTopoInfo& topo, const std::vector<u64>& dataSizePerLoop, u64 yCclBufferBaseOff, u64 yInCornerStep,
    u64 innerStepNum, uint32_t root);
void PushScatterYOverSameAxisOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevely, u64 osn, u64 xySDataSize[][MAX_STEP_NUM_SC],
    u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 xySOffset[][MAX_STEP_NUM_SC], u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], const std::vector<OmniPipeSplitSliceInfo>& perLoop,
    const std::vector<OmniPipeSplitSliceInfo>& total, const ScatterTopoInfo& topo,
    const std::vector<u64>& dataSizePerLoop, u64 yCclBufferBaseOff, u64 yInCornerStep, u64 innerStepNum, uint32_t root);
void PushScatterYOuterLECornerOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevely, u64 osn, u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 xySOffset[][MAX_STEP_NUM_SC], u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    const ScatterTopoInfo& topo, uint32_t root, u64 yCclBufferBaseOff, u64 yInCornerStep);
void PushScatterYOuterLEFwdOneRank(
    StepSliceInfo& stepSliceInfotmp, u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 xySOffset[][MAX_STEP_NUM_SC], u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, uint32_t root, u64 osn, const ScatterTopoInfo& topo,
    u64 oneDid);
void PushScatterYOuterLESameAxisOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevely, u64 osn, u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 xySOffset[][MAX_STEP_NUM_SC],
    u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    const ScatterTopoInfo& topo, uint32_t root, u64 yCclBufferBaseOff, u64 yInCornerStep, u64 innerStepNum);
void PushScatterYOuterGTCornerOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevely, u64 osn, u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 xySOffset[][MAX_STEP_NUM_SC], u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    const ScatterTopoInfo& topo, uint32_t root, u64 xCclBufferBaseOff, u64 yInCornerStep);
void PushScatterYOuterGTFwdOneRank(
    StepSliceInfo& stepSliceInfotmp, u64 xySDataSize[][MAX_STEP_NUM_SC],
    u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 xySOffset[][MAX_STEP_NUM_SC], u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], const std::vector<OmniPipeSplitSliceInfo>& perLoop, uint32_t root,
    u64 osn, u64 isn, u64 innerStepNum, const ScatterTopoInfo& topo, const std::vector<u64>& dataSizePerLoop);
void PushScatterYOuterGTSameAxisOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevely, u64 osn, u64 xySDataSize[][MAX_STEP_NUM_SC],
    u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 xySOffset[][MAX_STEP_NUM_SC], u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], const std::vector<OmniPipeSplitSliceInfo>& perLoop,
    const std::vector<OmniPipeSplitSliceInfo>& total, const ScatterTopoInfo& topo, uint32_t root, u64 xCclBufferBaseOff,
    u64 yInCornerStep, u64 innerStepNum, const std::vector<u64>& dataSizePerLoop);
void PushScatterXInnerCornerOneDiag(
    ScatterPieceVecs& pieces, u64 osn, u64 isn, u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 xySOffset[][MAX_STEP_NUM_SC], u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    const ScatterTopoInfo& topo, const std::vector<u64>& dataSizePerLoop, u64 oneDid, uint32_t root);
void CalcScatterXOuterLEOffset(
    u64& inputPieceIdOffset, u64& outputPieceIdOffset, u64& sliceSizeOnePiece,
    u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 xySOffset[][MAX_STEP_NUM_SC], u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, uint32_t root, u64 osn, u64 isn, u64 innerStepNum, u64 pieceId,
    const ScatterTopoInfo& topo);
void PushScatterYInnerCornerOneDiag(
    ScatterPieceVecs& pieces, u64 osn, u64 isn, u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 xySOffset[][MAX_STEP_NUM_SC], u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    const ScatterTopoInfo& topo, u64 oneDid, uint32_t root);
void CalcScatterYOuterGTOffset(
    u64& inputPieceIdOffset, u64& outputPieceIdOffset, u64& sliceSizeOnePiece,
    u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 xySOffset[][MAX_STEP_NUM_SC], u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, uint32_t root, u64 osn, u64 isn, u64 innerStepNum, u64 pieceId,
    const ScatterTopoInfo& topo);
ScatterTopoInfo InitScatterTopoInfo(OmniPipeSliceParam& omniPipeSliceParam, uint32_t root);
void InitScatterStepFlags(ScatterStepState& state, const ScatterTopoInfo& topo);
std::vector<u64> OmniPipeSplitScatterData(u64 rankSize, u64 count, u64 dataTypeSize, u64 root);
} // namespace ops_hccl
#endif
