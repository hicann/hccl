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
constexpr u64 MAX_RANK_SIZE = 2048;

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
void CalcScatterCornerStep(
    u64 innerStepNum, u64 outerStepNum, double xB, double yB, u64 zCornerStep, u64& xyCornerStep, u64& xInCornerStep,
    u64& yInCornerStep);
void PushScatterZStepSize(
    std::vector<std::vector<u64>>& scatterStepDataSize, u64* zScatterDataSize, u64 zCornerStep, u64 outerStepNum,
    u64 xRankSize, u64 yRankSize);
void PushScatterAxisStepSize(
    std::vector<std::vector<u64>>& scatterStepDataSize, u64* axisScatterDataSize, u64 axisInCornerStep,
    u64 innerStepNum, u64 xyCornerStep, u64 outerStepNum, u64 maxStepNum, u64 crossAxisRankSize, u64 zRankSize,
    int axisLevel);
std::vector<std::vector<u64>> CalScatterDataSizeStep(
    u64* xScatterDataSize, u64* yScatterDataSize, u64* zScatterDataSize, std::vector<u64> levelRankSize, u64 cornerStep,
    u64 outerStepNum, u64 innerStepNum, u64 maxStepNum, double xB, double yB);
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
std::vector<u64> CalcScatterScratchSize(
    u64* xSDataSize, u64* ySDataSize, u64* zSDataSize, std::vector<u64> levelRankSize, u64 cornerStep, u64 outerStepNum,
    u64 innerStepNum, u64 maxStepNum, std::vector<u64> levelAlgType, CommEngine engine, double xB, double yB);
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
    u64 rankSize, u64 zSDataSize[][MAX_STEP_NUM], u64 xySDataSize[][MAX_STEP_NUM],
    u64 xSDataSize[][MAX_STEP_NUM][MAX_STEP_NUM], u64 ySDataSize[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 zSOffset[][MAX_STEP_NUM], u64 xSOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 ySOffset[][MAX_STEP_NUM][MAX_STEP_NUM], u64 xySOffset[][MAX_STEP_NUM]);
void CalcScatterOneRankDataSize(
    const ScatterTopoInfo& topo, ScatterStepState& state, u64 rs, u64 finStepMark, double slowBw, double fastBw,
    u64 slowRankSize, u64 fastRankSize, u64 zSDataSize[][MAX_STEP_NUM], u64 xySDataSize[][MAX_STEP_NUM],
    u64 xSDataSize[][MAX_STEP_NUM][MAX_STEP_NUM], u64 ySDataSize[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 zSOffset[][MAX_STEP_NUM], u64 xSOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 ySOffset[][MAX_STEP_NUM][MAX_STEP_NUM], u64 xySOffset[][MAX_STEP_NUM],
    const std::vector<OmniPipeSplitSliceInfo>& omniPipeSplitSliceInfoListPerLoop);
void CalcScatterOuterCornerStep(ScatterStepState& state, u64 finStepMark);
void CalcScatterAllRankDataSize(
    const ScatterTopoInfo& topo, ScatterStepState& state, uint32_t root, u64 finStepMark,
    u64 zSDataSize[][MAX_STEP_NUM], u64 xySDataSize[][MAX_STEP_NUM], u64 xSDataSize[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 ySDataSize[][MAX_STEP_NUM][MAX_STEP_NUM], u64 zSOffset[][MAX_STEP_NUM],
    u64 xSOffset[][MAX_STEP_NUM][MAX_STEP_NUM], u64 ySOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 xySOffset[][MAX_STEP_NUM], const std::vector<OmniPipeSplitSliceInfo>& omniPipeSplitSliceInfoListPerLoop);
void PushScatterXAllSteps(
    std::vector<StepSliceInfo>& dataSliceLevelx, const ScatterTopoInfo& topo, const ScatterStepState& state,
    u64 xSDataSize[][MAX_STEP_NUM][MAX_STEP_NUM], u64 ySDataSize[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 xySOffset[][MAX_STEP_NUM], u64 xSOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 ySOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    const std::vector<OmniPipeSplitSliceInfo>& omniPipeSplitSliceInfoListPerLoop,
    const std::vector<OmniPipeSplitSliceInfo>& omniPipeSplitSliceInfoListTotal, uint32_t root, u64 xCclBufferBaseOff,
    u64 yCclBufferBaseOff, const std::vector<u64>& dataSizePerLoop);
void PushScatterYAllSteps(
    std::vector<StepSliceInfo>& dataSliceLevely, const ScatterTopoInfo& topo, const ScatterStepState& state,
    u64 xSDataSize[][MAX_STEP_NUM][MAX_STEP_NUM], u64 ySDataSize[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 xySOffset[][MAX_STEP_NUM], u64 xSOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 ySOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    const std::vector<OmniPipeSplitSliceInfo>& omniPipeSplitSliceInfoListPerLoop,
    const std::vector<OmniPipeSplitSliceInfo>& omniPipeSplitSliceInfoListTotal, uint32_t root, u64 xCclBufferBaseOff,
    u64 yCclBufferBaseOff, const std::vector<u64>& dataSizePerLoop);
void PrepareScatterBuffersAndPushZ(
    std::vector<StepSliceInfo>& dataSliceLevelz, u64& xCclBufferBaseOff, u64& yCclBufferBaseOff, u64& zCclBufferBaseOff,
    const ScatterTopoInfo& topo, const ScatterStepState& state, u64 xSDataSize[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 ySDataSize[][MAX_STEP_NUM][MAX_STEP_NUM], u64 zSDataSize[][MAX_STEP_NUM], u64 zSOffset[][MAX_STEP_NUM],
    const std::vector<OmniPipeSplitSliceInfo>& omniPipeSplitSliceInfoListPerLoop,
    const std::vector<OmniPipeSplitSliceInfo>& omniPipeSplitSliceInfoListTotal,
    const OmniPipeSliceParam& omniPipeSliceParam);
OmniPipeSliceInfo CalcScatterOmniPipeSliceInfo(OmniPipeSliceParam& omniPipeSliceParam, uint32_t root);
void PushScatterZDiagSteps(
    std::vector<StepSliceInfo>& dataSliceLevelz, u64 zSDataSize[][MAX_STEP_NUM], u64 zSOffset[][MAX_STEP_NUM],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    u64 dataTypeSize, u64 maxDataPieceId, u64 xRankSize, u64 yRankSize, u64 zRankSize, u64 xAxis, u64 yAxis,
    u64 zCclBufferBaseOff, u64 zCornerStep, const std::vector<std::vector<u64>>& xyzDataSizeStep);
void PushScatterZSameAxisSteps(
    std::vector<StepSliceInfo>& dataSliceLevelz, u64 zSDataSize[][MAX_STEP_NUM], u64 zSOffset[][MAX_STEP_NUM],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    u64 dataTypeSize, u64 maxDataPieceId, u64 xRankSize, u64 yRankSize, u64 zRankSize, u64 xAxis, u64 yAxis,
    u64 zCclBufferBaseOff, u64 zCornerStep, u64 outerStepNum, const std::vector<std::vector<u64>>& xyzDataSizeStep);
void PushScatterXInnerCornerOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevelx, u64 osn, u64 xSDataSize[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 xySOffset[][MAX_STEP_NUM], u64 xSOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    u64 dataTypeSize, u64 maxDataPieceId, u64 xRankSize, u64 yRankSize, u64 zRankSize, u64 yAxis, u64 zAxis,
    const std::vector<u64>& dataSizePerLoop, u64 xCclBufferBaseOff, u64 xInCornerStep);
void PushScatterXInnerSameAxisOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevelx, u64 osn, u64 xSDataSize[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 xySOffset[][MAX_STEP_NUM], u64 xSOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    u64 dataTypeSize, u64 maxDataPieceId, u64 xRankSize, u64 yRankSize, u64 zRankSize, u64 yAxis, u64 zAxis,
    const std::vector<u64>& dataSizePerLoop, u64 xCclBufferBaseOff, u64 xInCornerStep, u64 innerStepNum);
void PushScatterXOuterLECornerOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevelx, u64 osn, u64 xSDataSize[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 xySOffset[][MAX_STEP_NUM], u64 xSOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    u64 dataTypeSize, uint32_t root, u64 xRankSize, u64 yRankSize, u64 rootx, u64 rooty, u64 rootz,
    u64 xCclBufferBaseOff, u64 xInCornerStep);
void PushScatterXOuterLEFwdOneRank(
    StepSliceInfo& stepSliceInfotmp, u64 xSDataSize[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 ySDataSize[][MAX_STEP_NUM][MAX_STEP_NUM], u64 xySOffset[][MAX_STEP_NUM],
    u64 xSOffset[][MAX_STEP_NUM][MAX_STEP_NUM], u64 ySOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, uint32_t root, u64 osn, u64 isn, u64 innerStepNum,
    u64 xRankSize, u64 yRankSize, u64 rootx, u64 rootz, u64 oneDid, u64 dataTypeSize,
    const std::vector<u64>& dataSizePerLoop);
void PushScatterXOuterLESameAxisOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevelx, u64 osn, u64 xSDataSize[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 ySDataSize[][MAX_STEP_NUM][MAX_STEP_NUM], u64 xySOffset[][MAX_STEP_NUM],
    u64 xSOffset[][MAX_STEP_NUM][MAX_STEP_NUM], u64 ySOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    u64 dataTypeSize, uint32_t root, u64 xRankSize, u64 yRankSize, u64 rootx, u64 rooty, u64 rootz,
    u64 xCclBufferBaseOff, u64 xInCornerStep, u64 innerStepNum, const std::vector<u64>& dataSizePerLoop);
void PushScatterXOuterGTCornerOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevelx, u64 osn, u64 xSDataSize[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 xySOffset[][MAX_STEP_NUM], u64 xSOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    u64 dataTypeSize, uint32_t root, u64 xRankSize, u64 yRankSize, u64 rootx, u64 rooty, u64 rootz,
    u64 yCclBufferBaseOff, u64 xInCornerStep);
void PushScatterXOuterGTFwdOneRank(
    StepSliceInfo& stepSliceInfotmp, u64 ySDataSize[][MAX_STEP_NUM][MAX_STEP_NUM], u64 xySOffset[][MAX_STEP_NUM],
    u64 ySOffset[][MAX_STEP_NUM][MAX_STEP_NUM], const std::vector<OmniPipeSplitSliceInfo>& perLoop, uint32_t root,
    u64 osn, u64 xRankSize, u64 yRankSize, u64 rootx, u64 rootz, u64 one, u64 dataTypeSize);
void PushScatterXOuterGTSameAxisOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevelx, u64 osn, u64 xSDataSize[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 ySDataSize[][MAX_STEP_NUM][MAX_STEP_NUM], u64 xySOffset[][MAX_STEP_NUM],
    u64 xSOffset[][MAX_STEP_NUM][MAX_STEP_NUM], u64 ySOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    u64 dataTypeSize, uint32_t root, u64 xRankSize, u64 yRankSize, u64 rootx, u64 rooty, u64 rootz,
    u64 yCclBufferBaseOff, u64 xInCornerStep, u64 innerStepNum);
void PushScatterYInnerCornerOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevely, u64 osn, u64 ySDataSize[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 xySOffset[][MAX_STEP_NUM], u64 ySOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    u64 dataTypeSize, u64 maxDataPieceId, u64 xRankSize, u64 yRankSize, u64 zRankSize, u64 xAxis, u64 zAxis,
    u64 yCclBufferBaseOff, u64 yInCornerStep);
void PushScatterYInnerSameAxisOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevely, u64 osn, u64 ySDataSize[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 xySOffset[][MAX_STEP_NUM], u64 ySOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    u64 dataTypeSize, u64 maxDataPieceId, u64 xRankSize, u64 yRankSize, u64 zRankSize, u64 xAxis, u64 zAxis,
    const std::vector<u64>& dataSizePerLoop, u64 yCclBufferBaseOff, u64 yInCornerStep, u64 innerStepNum);
void PushScatterYOuterLECornerOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevely, u64 osn, u64 ySDataSize[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 xySOffset[][MAX_STEP_NUM], u64 ySOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    u64 dataTypeSize, uint32_t root, u64 xRankSize, u64 yRankSize, u64 rootx, u64 rooty, u64 zAxis,
    u64 yCclBufferBaseOff, u64 yInCornerStep);
void PushScatterYOuterLEFwdOneRank(
    StepSliceInfo& stepSliceInfotmp, u64 xSDataSize[][MAX_STEP_NUM][MAX_STEP_NUM], u64 xySOffset[][MAX_STEP_NUM],
    u64 xSOffset[][MAX_STEP_NUM][MAX_STEP_NUM], const std::vector<OmniPipeSplitSliceInfo>& perLoop, uint32_t root,
    u64 osn, u64 xRankSize, u64 yRankSize, u64 rooty, u64 rootz, u64 oneDid, u64 dataTypeSize);
void PushScatterYOuterLESameAxisOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevely, u64 osn, u64 xSDataSize[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 ySDataSize[][MAX_STEP_NUM][MAX_STEP_NUM], u64 xySOffset[][MAX_STEP_NUM],
    u64 xSOffset[][MAX_STEP_NUM][MAX_STEP_NUM], u64 ySOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    u64 dataTypeSize, uint32_t root, u64 xRankSize, u64 yRankSize, u64 rootx, u64 rooty, u64 rootz,
    u64 yCclBufferBaseOff, u64 yInCornerStep, u64 innerStepNum);
void PushScatterYOuterGTCornerOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevely, u64 osn, u64 ySDataSize[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 xySOffset[][MAX_STEP_NUM], u64 ySOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    u64 dataTypeSize, uint32_t root, u64 xRankSize, u64 yRankSize, u64 rootx, u64 rooty, u64 rootz,
    u64 xCclBufferBaseOff, u64 yInCornerStep);
void PushScatterYOuterGTFwdOneRank(
    StepSliceInfo& stepSliceInfotmp, u64 xSDataSize[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 ySDataSize[][MAX_STEP_NUM][MAX_STEP_NUM], u64 xySOffset[][MAX_STEP_NUM],
    u64 xSOffset[][MAX_STEP_NUM][MAX_STEP_NUM], u64 ySOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, uint32_t root, u64 osn, u64 isn, u64 innerStepNum,
    u64 xRankSize, u64 yRankSize, u64 rooty, u64 rootz, u64 rankx, u64 dataTypeSize,
    const std::vector<u64>& dataSizePerLoop);
void PushScatterYOuterGTSameAxisOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevely, u64 osn, u64 xSDataSize[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 ySDataSize[][MAX_STEP_NUM][MAX_STEP_NUM], u64 xySOffset[][MAX_STEP_NUM],
    u64 xSOffset[][MAX_STEP_NUM][MAX_STEP_NUM], u64 ySOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    u64 dataTypeSize, uint32_t root, u64 xRankSize, u64 yRankSize, u64 rootx, u64 rooty, u64 rootz,
    u64 xCclBufferBaseOff, u64 yInCornerStep, u64 innerStepNum, const std::vector<u64>& dataSizePerLoop);
void PushScatterXInnerCornerOneDiag(
    std::vector<u64>& sliceSizeMultRankPiece, std::vector<u64>& sliceCountMultRankPiece,
    std::vector<u64>& inputOmniPipeSliceStrideMultRankPiece, std::vector<u64>& outputOmniPipeSliceStrideMultRankPiece,
    u64 osn, u64 isn, u64 xSDataSize[][MAX_STEP_NUM][MAX_STEP_NUM], u64 xySOffset[][MAX_STEP_NUM],
    u64 xSOffset[][MAX_STEP_NUM][MAX_STEP_NUM], const std::vector<OmniPipeSplitSliceInfo>& perLoop,
    const std::vector<OmniPipeSplitSliceInfo>& total, u64 dataTypeSize, u64 maxDataPieceId, u64 xRankSize,
    u64 yRankSize, u64 zRankSize, u64 yAxis, u64 zAxis, const std::vector<u64>& dataSizePerLoop, u64 oneDid);
void CalcScatterXOuterLEOffset(
    u64& inputPieceIdOffset, u64& outputPieceIdOffset, u64& sliceSizeOnePiece,
    u64 xSDataSize[][MAX_STEP_NUM][MAX_STEP_NUM], u64 ySDataSize[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 xySOffset[][MAX_STEP_NUM], u64 ySOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, uint32_t root, u64 osn, u64 isn, u64 innerStepNum, u64 pieceId);
void PushScatterYInnerCornerOneDiag(
    std::vector<u64>& sliceSizeMultRankPiece, std::vector<u64>& sliceCountMultRankPiece,
    std::vector<u64>& inputOmniPipeSliceStrideMultRankPiece, std::vector<u64>& outputOmniPipeSliceStrideMultRankPiece,
    u64 osn, u64 isn, u64 ySDataSize[][MAX_STEP_NUM][MAX_STEP_NUM], u64 xySOffset[][MAX_STEP_NUM],
    u64 ySOffset[][MAX_STEP_NUM][MAX_STEP_NUM], const std::vector<OmniPipeSplitSliceInfo>& perLoop,
    const std::vector<OmniPipeSplitSliceInfo>& total, u64 dataTypeSize, u64 maxDataPieceId, u64 xRankSize,
    u64 yRankSize, u64 zRankSize, u64 xAxis, u64 zAxis, u64 oneDid);
void CalcScatterYOuterGTOffset(
    u64& inputPieceIdOffset, u64& outputPieceIdOffset, u64& sliceSizeOnePiece,
    u64 xSDataSize[][MAX_STEP_NUM][MAX_STEP_NUM], u64 ySDataSize[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 xySOffset[][MAX_STEP_NUM], u64 xSOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, uint32_t root, u64 osn, u64 isn, u64 innerStepNum, u64 pieceId);
ScatterTopoInfo InitScatterTopoInfo(OmniPipeSliceParam& omniPipeSliceParam, uint32_t root);
void InitScatterStepFlags(ScatterStepState& state, const ScatterTopoInfo& topo);
std::vector<u64> OmniPipeSplitScatterData(u64 rankSize, u64 count, u64 dataTypeSize, u64 root);
} // namespace ops_hccl
#endif
