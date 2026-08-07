/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "omnipipe_scatter_data_slice_calc.h"
#include "comm_engine_utils.h"
#include "utils.h"

namespace ops_hccl {
// 2D scatter发送数据片偏移计算,y轴快
// 参数,xSOffset x轴偏移，ySOffset y轴偏移，stepNum步数，xRankSize x轴大小，yRankSize y轴大小
// x轴每步每一片数据大小，y轴每步每一片数据大小 scatter不需要最后一步拆成两步
void CalScatter2DOffset(
    u64* xSOffset, u64* ySOffset, u64 stepNum, u64 xRankSize, u64 yRankSize, u64* xSDataSize, u64* ySDataSize)
{
    HCCL_DEBUG("[CalScatter2DOffset] start");
    xSOffset[0] = 0; // 第一步发斜对角，偏移为0
    ySOffset[0] = 0;

    if (stepNum > 1) {
        // 第二步开始发同轴数据，偏移也从0开始
        xSOffset[1] = 0;
        // y轴前n-1步发斜对角
        ySOffset[0] = xSOffset[0] + xSDataSize[0];

        // 第二步及以后，偏移接着上一步
        for (u64 sn = 1; sn < stepNum - 1; sn++) {
            ySOffset[sn] = ySOffset[sn - 1] + ySDataSize[sn - 1];
        }

        // 第二步及以后，偏移接着上一步
        for (u64 sn = 2; sn < stepNum - 1; sn++) {
            xSOffset[sn] = xSOffset[sn - 1] + xSDataSize[sn - 1];
        }
        // 最后一步
        if (stepNum > 2) {
            xSOffset[stepNum - 1] = xSOffset[stepNum - 1 - 1] + xSDataSize[stepNum - 1 - 1];
        }
        // 最后一步发送y轴同轴数据，所以offset是0
        ySOffset[stepNum - 1] = 0;
    }

    for (u64 i = 0; i < stepNum; i++) {
        HCCL_DEBUG("[CalScatter2DOffset] xSOffset[%llu]=[%llu],ySOffset[%llu]=[%llu]", i, xSOffset[i], i, ySOffset[i]);
    }
    HCCL_DEBUG("[CalScatter2DOffset] end");
}

// 计算2D scatter每步数据片大小存进数组，返回通信步数,y轴快,数据需要整除对齐，注意是每一小片的大小。
// scatter不需要最后一步拆成两步
void CalcScatterStepAndScale(
    double bandwidthRatio, double omniPipeRatio, u64 xRankSize, u64 maxStep, u64& step, double& scale)
{
    step = maxStep;
    if (xRankSize - bandwidthRatio > 0) {
        if (IsDoubleEqual(omniPipeRatio, 1.0)) {
            step = bandwidthRatio + 1;
        } else {
            step = ceil(std::log(xRankSize - bandwidthRatio) / std::log(omniPipeRatio)) + 1;
        }
        if (step <= maxStep) {
            scale = 1;
        } else {
            step = maxStep;
        }
    }
}

void CalcScatterFirstStepSize(
    u64* xStepP2pDataSize, u64* yStepP2pDataSize, double bandwidthRatio, u64 xRankSize, u64 yRankSize,
    u64 dataSizeEachRank, double scale, u64 step)
{
    u64 justifyLen = HCCL_MIN_SLICE_ALIGN;
    if (scale > 1) {
        xStepP2pDataSize[0]
            = dataSizeEachRank * scale * std::pow(xRankSize - 1, step - 1)
              / (((yRankSize - 1) * bandwidthRatio + xRankSize - 1) * std::pow(bandwidthRatio, step - 1));
    } else {
        xStepP2pDataSize[0]
            = (xRankSize - bandwidthRatio) * dataSizeEachRank / ((yRankSize - 1) * bandwidthRatio + xRankSize - 1);
    }
    xStepP2pDataSize[0] = xStepP2pDataSize[0] / justifyLen * justifyLen;
    if (step == 2) {
        yStepP2pDataSize[0] = dataSizeEachRank - xStepP2pDataSize[0];
    } else {
        yStepP2pDataSize[0] = xStepP2pDataSize[0] * bandwidthRatio * (yRankSize - 1) / (xRankSize - 1);
        yStepP2pDataSize[0] = yStepP2pDataSize[0] / justifyLen * justifyLen;
    }
}

void CalcScatterMidStepsSize(
    u64* xStepP2pDataSize, u64* yStepP2pDataSize, double bandwidthRatio, u64 xRankSize, u64 dataSizeEachRank, u64 step,
    u64& sumXDataSize, u64& sumYDataSize)
{
    u64 justifyLen = HCCL_MIN_SLICE_ALIGN;
    for (u64 index = 1; index < step - 1; index++) {
        if (index == step - 2) {
            yStepP2pDataSize[index] = dataSizeEachRank - sumYDataSize;
            xStepP2pDataSize[index] = yStepP2pDataSize[index] * (xRankSize - 1) / bandwidthRatio;
            if (index == 1 && xStepP2pDataSize[index] > sumYDataSize) {
                xStepP2pDataSize[index] = sumYDataSize;
            } else if (xStepP2pDataSize[index] > yStepP2pDataSize[index - 1]) {
                xStepP2pDataSize[index] = yStepP2pDataSize[index - 1];
            }
            xStepP2pDataSize[index] = xStepP2pDataSize[index] / justifyLen * justifyLen;
        } else {
            if (index == 1) {
                xStepP2pDataSize[index] = sumYDataSize;
            } else {
                xStepP2pDataSize[index] = yStepP2pDataSize[index - 1];
            }
            yStepP2pDataSize[index] = xStepP2pDataSize[index] * bandwidthRatio / (xRankSize - 1);
            yStepP2pDataSize[index] = yStepP2pDataSize[index] / justifyLen * justifyLen;
        }
        sumXDataSize += xStepP2pDataSize[index];
        sumYDataSize += yStepP2pDataSize[index];
    }
}

u64 CalScatterDataSize2D(
    u64* xStepP2pDataSize, u64* yStepP2pDataSize, double xB, double yB, u64 xRankSize, u64 yRankSize,
    u64 dataSizeEachRank, u64 maxStep)
{
    HCCL_DEBUG("[CalScatterDataSize2D] start");
    u64 step = 1;
    if (yRankSize == 1) {
        xStepP2pDataSize[0] = dataSizeEachRank;
    } else if (xRankSize == 1) {
        yStepP2pDataSize[0] = dataSizeEachRank;
    } else {
        double bandwidthRatio = yB / xB; // 带宽比例
        // 计算放大系数
        double scale = 0;
        // 计算斜对角等比
        double omniPipeRatio = (xRankSize - 1) / bandwidthRatio;
        for (u64 t = 0; t < maxStep - 1; t++) {
            scale = scale + std::pow(omniPipeRatio, t);
        }
        scale = bandwidthRatio / scale;
        CalcScatterStepAndScale(bandwidthRatio, omniPipeRatio, xRankSize, maxStep, step, scale);
        HCCL_DEBUG(
            "[CalScatterDataSize2D] bandwidthRatio=[%f],omniPipeRatio=[%f],scale=[%f],step=[%llu]", bandwidthRatio,
            omniPipeRatio, scale, step);
        // 1. 计算第一步的通信数据 (斜对角数据)
        CalcScatterFirstStepSize(
            xStepP2pDataSize, yStepP2pDataSize, bandwidthRatio, xRankSize, yRankSize, dataSizeEachRank, scale, step);

        u64 sumXDataSize = 0;
        u64 sumYDataSize = yStepP2pDataSize[0] + xStepP2pDataSize[0];

        // 2. 计算中间步骤的通信数据
        CalcScatterMidStepsSize(
            xStepP2pDataSize, yStepP2pDataSize, bandwidthRatio, xRankSize, dataSizeEachRank, step, sumXDataSize,
            sumYDataSize);

        // 3. 最后一步的通信数据，不需要拆成两步也不需要对齐了
        xStepP2pDataSize[step - 1] = dataSizeEachRank - sumXDataSize;
        yStepP2pDataSize[step - 1] = dataSizeEachRank;
    }

    HCCL_DEBUG("[CalScatterDataSize2D] step=[%llu]", step);
    for (u64 i = 0; i < step; i++) {
        HCCL_DEBUG(
            "[CalScatterDataSize2D] xStepP2pDataSize[%llu]=[%llu],yStepP2pDataSize[%llu]=[%llu]", i,
            xStepP2pDataSize[i], i, yStepP2pDataSize[i]);
    }
    HCCL_DEBUG("[CalScatterDataSize2D] end");
    return step;
}

std::vector<u64> CalcScatterScratchSize(
    u64* xSDataSize, u64* ySDataSize, u64* zSDataSize, std::vector<u64> levelRankSize, u64 cornerStep, u64 outerStepNum,
    u64 innerStepNum, u64 maxStepNum, std::vector<u64> levelAlgType, CommEngine engine, double xB, double yB)
{
    HCCL_DEBUG("[CalcScatterScratchSize] start");
    // 返回3个值，xBuffer,yBuffer,zBuffer,大小
    std::vector<u64> scratchSize = {0, 0, 0};
    u64 xRankSize = levelRankSize[OmniPipeLevel::OMNIPIPE_LEVEL0]; // x轴卡数
    u64 yRankSize = levelRankSize[OmniPipeLevel::OMNIPIPE_LEVEL1]; // y轴卡数
    u64 zRankSize = levelRankSize[OmniPipeLevel::OMNIPIPE_LEVEL2]; // z轴卡数
    HCCL_DEBUG(
        "[CalcScatterScratchSize] xRankSize=[%llu],yRankSize=[%llu],zRankSize=[%llu],", xRankSize, yRankSize,
        zRankSize);

    u64 xTopo = levelAlgType[OmniPipeLevel::OMNIPIPE_LEVEL0];
    u64 yTopo = levelAlgType[OmniPipeLevel::OMNIPIPE_LEVEL1];
    u64 zTopo = levelAlgType[OmniPipeLevel::OMNIPIPE_LEVEL2];
    HCCL_DEBUG("[CalcScatterScratchSize] xTopo=[%llu],yTopo=[%llu],zTopo=[%llu],", xTopo, yTopo, zTopo);

    std::vector<std::vector<u64>> sStepDataSize = CalScatterDataSizeStep(
        xSDataSize, ySDataSize, zSDataSize, levelRankSize, cornerStep, outerStepNum, innerStepNum, maxStepNum, xB, yB);

    for (u64 axis = 0; axis < levelAlgType.size(); axis++) {
        // 判断是不是aicpu+mesh，是的话需要预留scratch
        if (levelAlgType[axis] > 0
            && (engine == CommEngine::COMM_ENGINE_AICPU_TS || engine == CommEngine::COMM_ENGINE_CPU)) {
            for (u64 i = 0; i < sStepDataSize[axis].size(); i++) {
                if (scratchSize[axis] < sStepDataSize[axis][i] * levelRankSize[axis] && levelRankSize[axis] > 1) {
                    scratchSize[axis] = sStepDataSize[axis][i] * levelRankSize[axis];
                }
            }
        }
    }
    HCCL_DEBUG(
        "[CalcScatterScratchSize] scratchSize[0]=[%llu],scratchSize[1]=[%llu],scratchSize[2]=[%llu]",
        scratchSize[OmniPipeLevel::OMNIPIPE_LEVEL0], scratchSize[OmniPipeLevel::OMNIPIPE_LEVEL1],
        scratchSize[OmniPipeLevel::OMNIPIPE_LEVEL2]);
    HCCL_DEBUG("[CalcScatterScratchSize] end");
    return scratchSize;
}

void CalcScatterCornerStep(
    u64 innerStepNum, u64 outerStepNum, double xB, double yB, u64 zCornerStep, u64& xyCornerStep, u64& xInCornerStep,
    u64& yInCornerStep)
{
    u64 finStepMark = 1;
    if (innerStepNum > finStepMark) {
        if (yB >= xB) {
            xInCornerStep = 1;
            yInCornerStep = innerStepNum - finStepMark;
        } else {
            yInCornerStep = 1;
            xInCornerStep = innerStepNum - finStepMark;
        }
    }
    if (outerStepNum > finStepMark) {
        xyCornerStep = outerStepNum - zCornerStep - 1;
    }
}

void PushScatterZStepSize(
    std::vector<std::vector<u64>>& scatterStepDataSize, u64* zScatterDataSize, u64 zCornerStep, u64 outerStepNum,
    u64 xRankSize, u64 yRankSize)
{
    for (u64 osn = 0; osn < zCornerStep; osn++) {
        scatterStepDataSize[OmniPipeLevel::OMNIPIPE_LEVEL2].push_back(
            zScatterDataSize[osn] * (xRankSize * yRankSize - 1));
    }
    for (u64 osn = zCornerStep; osn < outerStepNum; osn++) {
        scatterStepDataSize[OmniPipeLevel::OMNIPIPE_LEVEL2].push_back(zScatterDataSize[osn]);
    }
}

void PushScatterAxisStepSize(
    std::vector<std::vector<u64>>& scatterStepDataSize, u64* axisScatterDataSize, u64 axisInCornerStep,
    u64 innerStepNum, u64 xyCornerStep, u64 outerStepNum, u64 maxStepNum, u64 crossAxisRankSize, u64 zRankSize,
    int axisLevel)
{
    for (u64 osn = 0; osn < xyCornerStep; osn++) {
        for (u64 isn = 0; isn < axisInCornerStep; isn++) {
            if (crossAxisRankSize > 1) {
                scatterStepDataSize[axisLevel].push_back(
                    axisScatterDataSize[osn * maxStepNum + isn] * (zRankSize - 1) * (crossAxisRankSize - 1));
            } else {
                scatterStepDataSize[axisLevel].push_back(axisScatterDataSize[osn * maxStepNum + isn] * (zRankSize - 1));
            }
        }
        for (u64 isn = axisInCornerStep; isn < innerStepNum; isn++) {
            if (crossAxisRankSize > 1) {
                scatterStepDataSize[axisLevel].push_back(axisScatterDataSize[osn * maxStepNum + isn] * (zRankSize - 1));
            } else {
                scatterStepDataSize[axisLevel].push_back(axisScatterDataSize[osn * maxStepNum + isn] * (zRankSize - 1));
            }
        }
    }
    for (u64 osn = xyCornerStep; osn < outerStepNum; osn++) {
        for (u64 isn = 0; isn < axisInCornerStep; isn++) {
            if (crossAxisRankSize > 1) {
                scatterStepDataSize[axisLevel].push_back(
                    axisScatterDataSize[osn * maxStepNum + isn] * (crossAxisRankSize - 1));
            } else {
                scatterStepDataSize[axisLevel].push_back(axisScatterDataSize[osn * maxStepNum + isn]);
            }
        }
        for (u64 isn = axisInCornerStep; isn < innerStepNum; isn++) {
            if (crossAxisRankSize > 1) {
                scatterStepDataSize[axisLevel].push_back(axisScatterDataSize[osn * maxStepNum + isn]);
            } else {
                scatterStepDataSize[axisLevel].push_back(axisScatterDataSize[osn * maxStepNum + isn]);
            }
        }
    }
}

// 根据数据片大小得到Scatter每步数据量
std::vector<std::vector<u64>> CalScatterDataSizeStep(
    u64* xScatterDataSize, u64* yScatterDataSize, u64* zScatterDataSize, std::vector<u64> levelRankSize, u64 cornerStep,
    u64 outerStepNum, u64 innerStepNum, u64 maxStepNum, double xB, double yB)
{
    HCCL_DEBUG("[CalScatterDataSizeStep] start");
    std::vector<std::vector<u64>> scatterStepDataSize = {};
    std::vector<u64> xSize = {};
    scatterStepDataSize.push_back(xSize);
    std::vector<u64> ySize = {};
    scatterStepDataSize.push_back(ySize);
    std::vector<u64> zSize = {};
    scatterStepDataSize.push_back(zSize);
    u64 xRankSize = levelRankSize[OmniPipeLevel::OMNIPIPE_LEVEL0]; // x轴卡数
    u64 yRankSize = levelRankSize[OmniPipeLevel::OMNIPIPE_LEVEL1]; // y轴卡数
    u64 zRankSize = levelRankSize[OmniPipeLevel::OMNIPIPE_LEVEL2]; // z轴卡数
    HCCL_DEBUG(
        "[CalScatterDataSizeStep] xRankSize=[%llu],yRankSize=[%llu],zRankSize=[%llu],", xRankSize, yRankSize,
        zRankSize);
    u64 zCornerStep = cornerStep;
    u64 xyCornerStep = 0;
    u64 xInCornerStep = 0;
    u64 yInCornerStep = 0;
    CalcScatterCornerStep(innerStepNum, outerStepNum, xB, yB, zCornerStep, xyCornerStep, xInCornerStep, yInCornerStep);

    HCCL_DEBUG(
        "[CalScatterDataSizeStep] xInCornerStep=[%llu],yInCornerStep=[%llu],cornerStep=[%llu],", xInCornerStep,
        yInCornerStep, cornerStep);

    PushScatterZStepSize(scatterStepDataSize, zScatterDataSize, zCornerStep, outerStepNum, xRankSize, yRankSize);
    PushScatterAxisStepSize(
        scatterStepDataSize, xScatterDataSize, xInCornerStep, innerStepNum, xyCornerStep, outerStepNum, maxStepNum,
        yRankSize, zRankSize, OmniPipeLevel::OMNIPIPE_LEVEL0);
    PushScatterAxisStepSize(
        scatterStepDataSize, yScatterDataSize, yInCornerStep, innerStepNum, xyCornerStep, outerStepNum, maxStepNum,
        xRankSize, zRankSize, OmniPipeLevel::OMNIPIPE_LEVEL1);

    for (u64 i = 0; i < outerStepNum; i++) {
        HCCL_DEBUG(
            "[CalScatterDataSizeStep] scatterStepDataSize[2][%llu]=[%llu],", i,
            scatterStepDataSize[OmniPipeLevel::OMNIPIPE_LEVEL2][i]);
    }
    for (u64 i = 0; i < outerStepNum * innerStepNum; i++) {
        HCCL_DEBUG(
            "[CalScatterDataSizeStep] scatterStepDataSize[0][%llu]=[%llu],", i,
            scatterStepDataSize[OmniPipeLevel::OMNIPIPE_LEVEL0][i]);
    }
    for (u64 i = 0; i < outerStepNum * innerStepNum; i++) {
        HCCL_DEBUG(
            "[CalScatterDataSizeStep] scatterStepDataSize[1][%llu]=[%llu],", i,
            scatterStepDataSize[OmniPipeLevel::OMNIPIPE_LEVEL1][i]);
    }
    HCCL_DEBUG("[CalScatterDataSizeStep] end");
    return scatterStepDataSize;
}

void CheckRootOrSameAxisAsRoot(
    u64 xRankSize, u64 yRankSize, u64 zRankSize, uint32_t root, uint32_t rankId, bool& ifRoot, bool& ifSameAxisAsRoot)
{
    ifRoot = (rankId == root);
    // 计算root节点在三维中的坐标
    u64 rootx = root % xRankSize;
    u64 rooty = (root / xRankSize) % yRankSize;
    u64 rootz = root / (xRankSize * yRankSize);
    u64 current_x = rankId % xRankSize;
    u64 current_y = (rankId / xRankSize) % yRankSize;
    u64 current_z = rankId / (xRankSize * yRankSize);
    if (zRankSize > 1) {
        ifSameAxisAsRoot = (current_x == rootx || current_y == rooty || current_z == rootz) && !ifRoot;
    } else {
        ifSameAxisAsRoot = (current_x == rootx || current_y == rooty) && !ifRoot;
    }
}

// 把一步的6个字段推入 stepSliceInfo（真数据分支，用副本因同一root向量可能被复用）
void PushStepFields(
    StepSliceInfo& s, const std::vector<u64>& sz, const std::vector<u64>& cnt, const std::vector<u64>& in,
    const std::vector<u64>& out, u64 inStride, u64 outStride)
{
    s.stepSliceSize.push_back(sz);
    s.stepCount.push_back(cnt);
    s.inputOmniPipeSliceStride.push_back(in);
    s.outputOmniPipeSliceStride.push_back(out);
    s.stepInputSliceStride.push_back(inStride);
    s.stepOutputSliceStride.push_back(outStride);
}

// 把等长零推入 stepSliceInfo（非root分支）
void PushStepZeros(StepSliceInfo& s, u64 n, u64 inStride, u64 outStride)
{
    std::vector<u64> z(n, 0);
    s.stepSliceSize.push_back(z);
    s.stepCount.push_back(z);
    s.inputOmniPipeSliceStride.push_back(z);
    s.outputOmniPipeSliceStride.push_back(z);
    s.stepInputSliceStride.push_back(inStride);
    s.stepOutputSliceStride.push_back(outStride);
}

// 根据 peerIdx 是否为 peerRoot，选择推真数据或等长零
void PushRootOrZeros(
    StepSliceInfo& s, const std::vector<u64>& sz, const std::vector<u64>& cnt, const std::vector<u64>& in,
    const std::vector<u64>& out, u64 peerIdx, u64 peerRoot, u64 outStride)
{
    if (peerIdx == peerRoot) {
        PushStepFields(s, sz, cnt, in, out, 0, outStride);
    } else {
        PushStepZeros(s, sz.size(), 0, 0);
    }
}

// 计算单个 piece 的 size/count/inputOffset/outputOffset 并 push 入四个 vector
// xyBaseOffset 为 xy 偏移基准，sDataSize 为本步该轴切片大小；input 用 total.offset，output 用 perLoop.offset
void CalcAndPushPiece(
    u64 pieceId, u64 xyBaseOffset, u64 sDataSize, const std::vector<OmniPipeSplitSliceInfo>& perLoop,
    const std::vector<OmniPipeSplitSliceInfo>& total, u64 dataTypeSize, std::vector<u64>& sz, std::vector<u64>& cnt,
    std::vector<u64>& in, std::vector<u64>& out)
{
    u64 sliceSizeOnePiece = DataSliceCut(sDataSize, xyBaseOffset, perLoop[pieceId].size);
    u64 inputPieceIdOffset = sliceOffsetCut(xyBaseOffset, perLoop[pieceId].size) + total[pieceId].offset;
    u64 outputPieceIdOffset = sliceOffsetCut(xyBaseOffset, perLoop[pieceId].size) + perLoop[pieceId].offset;
    sz.push_back(sliceSizeOnePiece);
    cnt.push_back(sliceSizeOnePiece / dataTypeSize);
    in.push_back(inputPieceIdOffset);
    out.push_back(outputPieceIdOffset);
}

void PushScatterZDiagSteps(
    std::vector<StepSliceInfo>& dataSliceLevelz, u64 zSDataSize[][MAX_STEP_NUM], u64 zSOffset[][MAX_STEP_NUM],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    u64 dataTypeSize, u64 maxDataPieceId, u64 xRankSize, u64 yRankSize, u64 zRankSize, u64 xAxis, u64 yAxis,
    u64 zCclBufferBaseOff, u64 zCornerStep, const std::vector<std::vector<u64>>& xyzDataSizeStep)
{
    HCCL_DEBUG("[PushScatterZDiagSteps] start push scatter z diag steps");
    for (u64 osn = 0; osn < zCornerStep; osn++) {
        struct BuffInfo bitmp;
        struct StepSliceInfo stepSliceInfotmp;
        BuffInfoAssign(bitmp, 0, 0, zCclBufferBaseOff);
        stepSliceInfotmp.buffInfo = bitmp;
        for (u64 oneDid = 0; oneDid < zRankSize; oneDid++) {
            u64 outputslicestride = 0;
            std::vector<u64> sliceCountMultRankPiece;
            std::vector<u64> sliceSizeMultRankPiece;
            std::vector<u64> outputOmniPipeSliceStrideMultRankPiece;
            std::vector<u64> inputOmniPipeSliceStrideMultRankPiece;
            for (u64 cornerDataSlice = 0; cornerDataSlice < xRankSize * yRankSize; cornerDataSlice++) {
                u64 currentDataSliceId = oneDid * xRankSize * yRankSize + cornerDataSlice;
                if (cornerDataSlice != yAxis * xRankSize + xAxis) {
                    u64 pieceId = currentDataSliceId;
                    u64 sliceSizeOnePiece = DataSliceCut(
                        zSDataSize[maxDataPieceId][osn], zSOffset[maxDataPieceId][osn], perLoop[pieceId].size);
                    u64 inputPieceIdOffset
                        = sliceOffsetCut(zSOffset[maxDataPieceId][osn], perLoop[pieceId].size) + total[pieceId].offset;
                    sliceSizeMultRankPiece.push_back(sliceSizeOnePiece);
                    sliceCountMultRankPiece.push_back(sliceSizeOnePiece / dataTypeSize);
                    inputOmniPipeSliceStrideMultRankPiece.push_back(inputPieceIdOffset);
                    outputOmniPipeSliceStrideMultRankPiece.push_back(outputslicestride);
                    outputslicestride += zSDataSize[maxDataPieceId][osn];
                }
            }
            stepSliceInfotmp.stepOutputSliceStride.push_back(xyzDataSizeStep[OMNIPIPE_LEVEL2][osn] * oneDid);
            stepSliceInfotmp.outputOmniPipeSliceStride.push_back(outputOmniPipeSliceStrideMultRankPiece);
            stepSliceInfotmp.inputOmniPipeSliceStride.push_back(inputOmniPipeSliceStrideMultRankPiece);
            stepSliceInfotmp.stepCount.push_back(sliceCountMultRankPiece);
            stepSliceInfotmp.stepSliceSize.push_back(sliceSizeMultRankPiece);
            stepSliceInfotmp.stepInputSliceStride.push_back(0);
        }
        dataSliceLevelz.insert(dataSliceLevelz.end(), stepSliceInfotmp);
    }
}

void PushScatterZSameAxisSteps(
    std::vector<StepSliceInfo>& dataSliceLevelz, u64 zSDataSize[][MAX_STEP_NUM], u64 zSOffset[][MAX_STEP_NUM],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    u64 dataTypeSize, u64 maxDataPieceId, u64 xRankSize, u64 yRankSize, u64 zRankSize, u64 xAxis, u64 yAxis,
    u64 zCclBufferBaseOff, u64 zCornerStep, u64 outerStepNum, const std::vector<std::vector<u64>>& xyzDataSizeStep)
{
    HCCL_DEBUG("[PushScatterZSameAxisSteps] start push scatter z same axis steps");
    for (u64 osn = zCornerStep; osn < outerStepNum; osn++) {
        struct BuffInfo bitmp;
        struct StepSliceInfo stepSliceInfotmp;
        BuffInfoAssign(bitmp, 0, 0, zCclBufferBaseOff);
        stepSliceInfotmp.buffInfo = bitmp;
        for (u64 oneDid = 0; oneDid < zRankSize; oneDid++) {
            std::vector<u64> sliceSizeMultRankPiece;
            std::vector<u64> sliceCountMultRankPiece;
            std::vector<u64> outputOmniPipeSliceStrideMultRankPiece;
            std::vector<u64> inputOmniPipeSliceStrideMultRankPiece;
            u64 pieceId = oneDid * xRankSize * yRankSize + yAxis * xRankSize + xAxis;
            u64 sliceSizeOnePiece
                = DataSliceCut(zSDataSize[maxDataPieceId][osn], zSOffset[maxDataPieceId][osn], perLoop[pieceId].size);
            u64 inputPieceIdOffset
                = sliceOffsetCut(zSOffset[maxDataPieceId][osn], perLoop[pieceId].size) + total[pieceId].offset;
            sliceCountMultRankPiece.push_back(sliceSizeOnePiece / dataTypeSize);
            sliceSizeMultRankPiece.push_back(sliceSizeOnePiece);
            inputOmniPipeSliceStrideMultRankPiece.push_back(inputPieceIdOffset);
            outputOmniPipeSliceStrideMultRankPiece.push_back(0);
            stepSliceInfotmp.stepInputSliceStride.push_back(0);
            stepSliceInfotmp.stepOutputSliceStride.push_back(xyzDataSizeStep[OMNIPIPE_LEVEL2][osn] * oneDid);
            stepSliceInfotmp.outputOmniPipeSliceStride.push_back(outputOmniPipeSliceStrideMultRankPiece);
            stepSliceInfotmp.inputOmniPipeSliceStride.push_back(inputOmniPipeSliceStrideMultRankPiece);
            stepSliceInfotmp.stepCount.push_back(sliceCountMultRankPiece);
            stepSliceInfotmp.stepSliceSize.push_back(sliceSizeMultRankPiece);
        }
        dataSliceLevelz.insert(dataSliceLevelz.end(), stepSliceInfotmp);
    }
}

void PushScatterXInnerCornerOneDiag(
    std::vector<u64>& sliceSizeMultRankPiece, std::vector<u64>& sliceCountMultRankPiece,
    std::vector<u64>& inputOmniPipeSliceStrideMultRankPiece, std::vector<u64>& outputOmniPipeSliceStrideMultRankPiece,
    u64 osn, u64 isn, u64 xSDataSize[][MAX_STEP_NUM][MAX_STEP_NUM], u64 xySOffset[][MAX_STEP_NUM],
    u64 xSOffset[][MAX_STEP_NUM][MAX_STEP_NUM], const std::vector<OmniPipeSplitSliceInfo>& perLoop,
    const std::vector<OmniPipeSplitSliceInfo>& total, u64 dataTypeSize, u64 maxDataPieceId, u64 xRankSize,
    u64 yRankSize, u64 zRankSize, u64 yAxis, u64 zAxis, const std::vector<u64>& dataSizePerLoop, u64 oneDid)
{
    u64 outputslicestride = 0;
    for (u64 outSliceNum = 0; outSliceNum < zRankSize; outSliceNum++) {
        if (outSliceNum != zAxis) {
            for (u64 cornerDataSlice = 0; cornerDataSlice < yRankSize; cornerDataSlice++) {
                u64 currentInnerStepDataSliceId
                    = outSliceNum * xRankSize * yRankSize + cornerDataSlice * xRankSize + oneDid;
                if (cornerDataSlice != yAxis && yRankSize > 1) {
                    u64 pieceId = currentInnerStepDataSliceId;
                    u64 sliceSizeOnePiece = DataSliceCut(
                        xSDataSize[maxDataPieceId][osn][isn],
                        xySOffset[maxDataPieceId][osn] + xSOffset[maxDataPieceId][osn][isn], perLoop[pieceId].size);
                    u64 inputPieceIdOffset = sliceOffsetCut(
                                                 xySOffset[maxDataPieceId][osn] + xSOffset[maxDataPieceId][osn][isn],
                                                 perLoop[pieceId].size)
                                             + total[pieceId].offset;
                    outputslicestride
                        = cornerDataSlice * dataSizePerLoop[maxDataPieceId] + xSOffset[maxDataPieceId][osn][isn];
                    sliceSizeMultRankPiece.push_back(sliceSizeOnePiece);
                    sliceCountMultRankPiece.push_back(sliceSizeOnePiece / dataTypeSize);
                    inputOmniPipeSliceStrideMultRankPiece.push_back(inputPieceIdOffset);
                    outputOmniPipeSliceStrideMultRankPiece.push_back(outputslicestride);
                    outputslicestride += xSDataSize[maxDataPieceId][osn][isn];
                }
            }
        }
    }
}

void PushScatterXInnerCornerOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevelx, u64 osn, u64 xSDataSize[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 xySOffset[][MAX_STEP_NUM], u64 xSOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    u64 dataTypeSize, u64 maxDataPieceId, u64 xRankSize, u64 yRankSize, u64 zRankSize, u64 yAxis, u64 zAxis,
    const std::vector<u64>& dataSizePerLoop, u64 xCclBufferBaseOff, u64 xInCornerStep)
{
    for (u64 isn = 0; isn < xInCornerStep; isn++) {
        struct BuffInfo bitmp;
        struct StepSliceInfo stepSliceInfotmp;
        BuffInfoAssign(bitmp, 0, 0, xCclBufferBaseOff);
        stepSliceInfotmp.buffInfo = bitmp;
        for (u64 oneDid = 0; oneDid < xRankSize; oneDid++) {
            std::vector<u64> sliceSizeMultRankPiece;
            std::vector<u64> inputOmniPipeSliceStrideMultRankPiece;
            std::vector<u64> sliceCountMultRankPiece;
            std::vector<u64> outputOmniPipeSliceStrideMultRankPiece;
            PushScatterXInnerCornerOneDiag(
                sliceSizeMultRankPiece, sliceCountMultRankPiece, inputOmniPipeSliceStrideMultRankPiece,
                outputOmniPipeSliceStrideMultRankPiece, osn, isn, xSDataSize, xySOffset, xSOffset, perLoop, total,
                dataTypeSize, maxDataPieceId, xRankSize, yRankSize, zRankSize, yAxis, zAxis, dataSizePerLoop, oneDid);
            PushStepFields(
                stepSliceInfotmp, sliceSizeMultRankPiece, sliceCountMultRankPiece,
                inputOmniPipeSliceStrideMultRankPiece, outputOmniPipeSliceStrideMultRankPiece, 0, 0);
        }
        dataSliceLevelx.insert(dataSliceLevelx.end(), stepSliceInfotmp);
    }
}

void PushScatterXInnerSameAxisOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevelx, u64 osn, u64 xSDataSize[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 xySOffset[][MAX_STEP_NUM], u64 xSOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    u64 dataTypeSize, u64 maxDataPieceId, u64 xRankSize, u64 yRankSize, u64 zRankSize, u64 yAxis, u64 zAxis,
    const std::vector<u64>& dataSizePerLoop, u64 xCclBufferBaseOff, u64 xInCornerStep, u64 innerStepNum)
{
    for (u64 isn = xInCornerStep; isn < innerStepNum; isn++) {
        struct BuffInfo bitmp;
        struct StepSliceInfo stepSliceInfotmp;
        BuffInfoAssign(bitmp, 0, 0, xCclBufferBaseOff);
        stepSliceInfotmp.buffInfo = bitmp;
        for (u64 oneDid = 0; oneDid < xRankSize; oneDid++) {
            std::vector<u64> inputOmniPipeSliceStrideMultRankPiece;
            std::vector<u64> sliceSizeMultRankPiece;
            std::vector<u64> sliceCountMultRankPiece;
            std::vector<u64> outputOmniPipeSliceStrideMultRankPiece;
            u64 outputslicestride = 0;
            for (u64 outSliceNum = 0; outSliceNum < zRankSize; outSliceNum++) {
                u64 currentDataSliceId = outSliceNum * xRankSize * yRankSize + yAxis * xRankSize + oneDid;
                if (outSliceNum != zAxis) {
                    u64 pieceId = currentDataSliceId;
                    u64 sliceSizeOnePiece = DataSliceCut(
                        xSDataSize[maxDataPieceId][osn][isn],
                        xySOffset[maxDataPieceId][osn] + xSOffset[maxDataPieceId][osn][isn], perLoop[pieceId].size);
                    u64 inputPieceIdOffset = sliceOffsetCut(
                                                 xySOffset[maxDataPieceId][osn] + xSOffset[maxDataPieceId][osn][isn],
                                                 perLoop[pieceId].size)
                                             + total[pieceId].offset;
                    outputslicestride
                        = outSliceNum * dataSizePerLoop[maxDataPieceId] + xSOffset[maxDataPieceId][osn][isn];
                    sliceSizeMultRankPiece.push_back(sliceSizeOnePiece);
                    sliceCountMultRankPiece.push_back(sliceSizeOnePiece / dataTypeSize);
                    inputOmniPipeSliceStrideMultRankPiece.push_back(inputPieceIdOffset);
                    outputOmniPipeSliceStrideMultRankPiece.push_back(outputslicestride);
                    outputslicestride += xSDataSize[maxDataPieceId][osn][isn];
                }
            }
            PushStepFields(
                stepSliceInfotmp, sliceSizeMultRankPiece, sliceCountMultRankPiece,
                inputOmniPipeSliceStrideMultRankPiece, outputOmniPipeSliceStrideMultRankPiece, 0, 0);
        }
        dataSliceLevelx.insert(dataSliceLevelx.end(), stepSliceInfotmp);
    }
}

// scatter x轴外层 xB<=yB
// 斜对角段（单个osn，isn∈[0,xInCornerStep)）：root发斜对角数据，root数据放index=rootx，其余x轴rank塞0
void PushScatterXOuterLECornerOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevelx, u64 osn, u64 xSDataSize[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 xySOffset[][MAX_STEP_NUM], u64 xSOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    u64 dataTypeSize, uint32_t root, u64 xRankSize, u64 yRankSize, u64 rootx, u64 rooty, u64 rootz,
    u64 xCclBufferBaseOff, u64 xInCornerStep)
{
    HCCL_DEBUG("[PushScatterXOuterLECornerOneOsn] start push scatter x outer le corner one osn");
    for (u64 isn = 0; isn < xInCornerStep; isn++) {
        struct StepSliceInfo stepSliceInfotmp;
        struct BuffInfo bitmp;
        BuffInfoAssign(bitmp, 0, 0, xCclBufferBaseOff);
        stepSliceInfotmp.buffInfo = bitmp;
        std::vector<u64> outputOmniPipeSliceStrideMultRankPiece;
        std::vector<u64> sliceCountMultRankPiece;
        std::vector<u64> sliceSizeMultRankPiece;
        std::vector<u64> inputOmniPipeSliceStrideMultRankPiece;
        for (u64 oneDid = 0; oneDid < xRankSize; oneDid++) {
            if (oneDid == rootx) {
                continue;
            }
            for (u64 cornerDataSlice = 0; cornerDataSlice < yRankSize; cornerDataSlice++) {
                u64 currentDataSliceId = rootz * xRankSize * yRankSize + cornerDataSlice * xRankSize + oneDid;
                if (cornerDataSlice != rooty) {
                    CalcAndPushPiece(
                        currentDataSliceId, xySOffset[root][osn] + xSOffset[root][osn][isn], xSDataSize[root][osn][isn],
                        perLoop, total, dataTypeSize, sliceSizeMultRankPiece, sliceCountMultRankPiece,
                        inputOmniPipeSliceStrideMultRankPiece, outputOmniPipeSliceStrideMultRankPiece);
                }
            }
        }
        for (u64 oneDid = 0; oneDid < yRankSize; oneDid++) {
            PushRootOrZeros(
                stepSliceInfotmp, sliceSizeMultRankPiece, sliceCountMultRankPiece,
                inputOmniPipeSliceStrideMultRankPiece, outputOmniPipeSliceStrideMultRankPiece, oneDid, rooty, 0);
        }
        dataSliceLevelx.insert(dataSliceLevelx.end(), stepSliceInfotmp);
    }
}

// scatter x轴外层 xB<=yB
// 同轴转发段（单个osn，isn∈[xInCornerStep,innerStepNum)）：root发同x轴数据，同y轴非root节点转发step1收到的对角数据
void CalcScatterXOuterLEOffset(
    u64& inputPieceIdOffset, u64& outputPieceIdOffset, u64& sliceSizeOnePiece,
    u64 xSDataSize[][MAX_STEP_NUM][MAX_STEP_NUM], u64 ySDataSize[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 xySOffset[][MAX_STEP_NUM], u64 ySOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, uint32_t root, u64 osn, u64 isn, u64 innerStepNum, u64 pieceId)
{
    HCCL_DEBUG("[CalcScatterXOuterLEOffset] start calc scatter x outer le offset");
    if (innerStepNum == 2) {
        inputPieceIdOffset = sliceOffsetCut(xySOffset[root][osn] + ySOffset[root][osn][isn - 1], perLoop[pieceId].size)
                             + perLoop[pieceId].offset;
        outputPieceIdOffset = sliceOffsetCut(xySOffset[root][osn] + ySOffset[root][osn][isn - 1], perLoop[pieceId].size)
                              + perLoop[pieceId].offset;
    } else {
        if (isn == innerStepNum - 1) {
            if (xSDataSize[root][osn][isn - 1] < ySDataSize[root][osn][isn - 2]) {
                inputPieceIdOffset
                    = sliceOffsetCut(
                          xySOffset[root][osn] + ySOffset[root][osn][isn - 2] + xSDataSize[root][osn][isn - 1],
                          perLoop[pieceId].size)
                      + perLoop[pieceId].offset;
                outputPieceIdOffset
                    = sliceOffsetCut(
                          xySOffset[root][osn] + ySOffset[root][osn][isn - 2] + xSDataSize[root][osn][isn - 1],
                          perLoop[pieceId].size)
                      + perLoop[pieceId].offset;
            } else {
                inputPieceIdOffset
                    = sliceOffsetCut(
                          xySOffset[root][osn] + ySOffset[root][osn][isn - 2] + ySDataSize[root][osn][isn - 2],
                          perLoop[pieceId].size)
                      + perLoop[pieceId].offset;
                outputPieceIdOffset
                    = sliceOffsetCut(
                          xySOffset[root][osn] + ySOffset[root][osn][isn - 2] + ySDataSize[root][osn][isn - 2],
                          perLoop[pieceId].size)
                      + perLoop[pieceId].offset;
            }
        } else {
            if (sliceSizeOnePiece > ySDataSize[root][osn][isn - 1]) {
                sliceSizeOnePiece = ySDataSize[root][osn][isn - 1];
            }
            inputPieceIdOffset
                = sliceOffsetCut(xySOffset[root][osn] + ySOffset[root][osn][isn - 1], perLoop[pieceId].size)
                  + perLoop[pieceId].offset;
            outputPieceIdOffset
                = sliceOffsetCut(xySOffset[root][osn] + ySOffset[root][osn][isn - 1], perLoop[pieceId].size)
                  + perLoop[pieceId].offset;
        }
    }
}

// scatter x轴外层 xB<=yB 同轴转发段：为非rooty的y轴节点构建转发piece并Push
void PushScatterXOuterLEFwdOneRank(
    StepSliceInfo& stepSliceInfotmp, u64 xSDataSize[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 ySDataSize[][MAX_STEP_NUM][MAX_STEP_NUM], u64 xySOffset[][MAX_STEP_NUM],
    u64 xSOffset[][MAX_STEP_NUM][MAX_STEP_NUM], u64 ySOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, uint32_t root, u64 osn, u64 isn, u64 innerStepNum,
    u64 xRankSize, u64 yRankSize, u64 rootx, u64 rootz, u64 oneDid, u64 dataTypeSize,
    const std::vector<u64>& dataSizePerLoop)
{
    HCCL_DEBUG("[PushScatterXOuterLEFwdOneRank] start push scatter x outer le fwd one rank");
    std::vector<u64> sliceSizeMultRankPiece;
    std::vector<u64> inputOmniPipeSliceStrideMultRankPiece;
    std::vector<u64> sliceCountMultRankPiece;
    std::vector<u64> outputOmniPipeSliceStrideMultRankPiece;
    for (u64 cornerDataSlice = 0; cornerDataSlice < xRankSize; cornerDataSlice++) {
        if (cornerDataSlice == rootx)
            continue;
        u64 currentDataSliceId = rootz * xRankSize * yRankSize + oneDid * xRankSize + cornerDataSlice;
        u64 pieceId = currentDataSliceId;
        u64 sliceSizeOnePiece = DataSliceCut(
            xSDataSize[root][osn][isn], xySOffset[root][osn] + xSOffset[root][osn][isn], perLoop[pieceId].size);
        u64 inputPieceIdOffset = 0;
        u64 outputPieceIdOffset = 0;
        CalcScatterXOuterLEOffset(
            inputPieceIdOffset, outputPieceIdOffset, sliceSizeOnePiece, xSDataSize, ySDataSize, xySOffset, ySOffset,
            perLoop, root, osn, isn, innerStepNum, pieceId);
        if (inputPieceIdOffset + sliceSizeOnePiece > perLoop[pieceId].offset + dataSizePerLoop[root]) {
            sliceSizeOnePiece = perLoop[pieceId].offset + dataSizePerLoop[root] - inputPieceIdOffset;
        }
        sliceSizeMultRankPiece.push_back(sliceSizeOnePiece);
        sliceCountMultRankPiece.push_back(sliceSizeOnePiece / dataTypeSize);
        inputOmniPipeSliceStrideMultRankPiece.push_back(inputPieceIdOffset);
        outputOmniPipeSliceStrideMultRankPiece.push_back(outputPieceIdOffset);
    }
    PushStepFields(
        stepSliceInfotmp, sliceSizeMultRankPiece, sliceCountMultRankPiece, inputOmniPipeSliceStrideMultRankPiece,
        outputOmniPipeSliceStrideMultRankPiece, 0, 0);
}

void PushScatterXOuterLESameAxisOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevelx, u64 osn, u64 xSDataSize[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 ySDataSize[][MAX_STEP_NUM][MAX_STEP_NUM], u64 xySOffset[][MAX_STEP_NUM],
    u64 xSOffset[][MAX_STEP_NUM][MAX_STEP_NUM], u64 ySOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    u64 dataTypeSize, uint32_t root, u64 xRankSize, u64 yRankSize, u64 rootx, u64 rooty, u64 rootz,
    u64 xCclBufferBaseOff, u64 xInCornerStep, u64 innerStepNum, const std::vector<u64>& dataSizePerLoop)
{
    HCCL_DEBUG("[PushScatterXOuterLESameAxisOneOsn] start push scatter x outer le same axis one osn");
    for (u64 isn = xInCornerStep; isn < innerStepNum; isn++) {
        struct StepSliceInfo stepSliceInfotmp;
        struct BuffInfo bitmp;
        BuffInfoAssign(bitmp, 0, 0, xCclBufferBaseOff);
        stepSliceInfotmp.buffInfo = bitmp;
        std::vector<u64> sliceSizeMultRankPiece;
        std::vector<u64> sliceCountMultRankPiece;
        std::vector<u64> outputOmniPipeSliceStrideMultRankPiece;
        std::vector<u64> inputOmniPipeSliceStrideMultRankPiece;
        for (u64 oneDid = 0; oneDid < xRankSize; oneDid++) {
            if (oneDid == rootx)
                continue;
            u64 pieceId = rootz * xRankSize * yRankSize + rooty * xRankSize + oneDid;
            CalcAndPushPiece(
                pieceId, xySOffset[root][osn] + xSOffset[root][osn][isn], xSDataSize[root][osn][isn], perLoop, total,
                dataTypeSize, sliceSizeMultRankPiece, sliceCountMultRankPiece, inputOmniPipeSliceStrideMultRankPiece,
                outputOmniPipeSliceStrideMultRankPiece);
        }
        for (u64 oneDid = 0; oneDid < yRankSize; oneDid++) {
            if (oneDid == rooty) {
                PushStepFields(
                    stepSliceInfotmp, sliceSizeMultRankPiece, sliceCountMultRankPiece,
                    inputOmniPipeSliceStrideMultRankPiece, outputOmniPipeSliceStrideMultRankPiece, 0, 0);
            } else {
                PushScatterXOuterLEFwdOneRank(
                    stepSliceInfotmp, xSDataSize, ySDataSize, xySOffset, xSOffset, ySOffset, perLoop, root, osn, isn,
                    innerStepNum, xRankSize, yRankSize, rootx, rootz, oneDid, dataTypeSize, dataSizePerLoop);
            }
        }
        dataSliceLevelx.insert(dataSliceLevelx.end(), stepSliceInfotmp);
    }
}

// scatter x轴外层 xB>yB
// 斜对角转发段（单个osn，isn∈[0,xInCornerStep+1)）：前两步都是转发对角数据，buffer用yCclBufferBaseOff
void PushScatterXOuterGTCornerOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevelx, u64 osn, u64 xSDataSize[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 xySOffset[][MAX_STEP_NUM], u64 xSOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    u64 dataTypeSize, uint32_t root, u64 xRankSize, u64 yRankSize, u64 rootx, u64 rooty, u64 rootz,
    u64 yCclBufferBaseOff, u64 xInCornerStep)
{
    HCCL_DEBUG("[PushScatterXOuterGTCornerOneOsn] start push scatter x outer gt corner one osn");
    for (u64 isn = 0; isn < xInCornerStep + 1; isn++) {
        struct StepSliceInfo stepSliceInfotmp;
        struct BuffInfo bitmp;
        BuffInfoAssign(bitmp, 0, 0, yCclBufferBaseOff);
        stepSliceInfotmp.buffInfo = bitmp;
        std::vector<u64> sliceSizeMultRankPiece;
        std::vector<u64> sliceCountMultRankPiece;
        std::vector<u64> inputOmniPipeSliceStrideMultRankPiece;
        std::vector<u64> outputOmniPipeSliceStrideMultRankPiece;
        for (u64 oneDid = 0; oneDid < xRankSize; oneDid++) {
            if (oneDid == rootx)
                continue;
            for (u64 cornerSlice = 0; cornerSlice < yRankSize; cornerSlice++) {
                u64 currentDataSliceId = rootz * xRankSize * yRankSize + cornerSlice * xRankSize + oneDid;
                if (cornerSlice != rooty) {
                    CalcAndPushPiece(
                        currentDataSliceId, xySOffset[root][osn] + xSOffset[root][osn][isn], xSDataSize[root][osn][isn],
                        perLoop, total, dataTypeSize, sliceSizeMultRankPiece, sliceCountMultRankPiece,
                        inputOmniPipeSliceStrideMultRankPiece, outputOmniPipeSliceStrideMultRankPiece);
                }
            }
        }
        for (u64 one = 0; one < yRankSize; one++) {
            PushRootOrZeros(
                stepSliceInfotmp, sliceSizeMultRankPiece, sliceCountMultRankPiece,
                inputOmniPipeSliceStrideMultRankPiece, outputOmniPipeSliceStrideMultRankPiece, one, rooty, 0);
        }
        dataSliceLevelx.insert(dataSliceLevelx.end(), stepSliceInfotmp);
    }
}

// scatter x轴外层 xB>yB 同轴转发段：为非rooty的y轴节点构建转发piece并Push
void PushScatterXOuterGTFwdOneRank(
    StepSliceInfo& stepSliceInfotmp, u64 ySDataSize[][MAX_STEP_NUM][MAX_STEP_NUM], u64 xySOffset[][MAX_STEP_NUM],
    u64 ySOffset[][MAX_STEP_NUM][MAX_STEP_NUM], const std::vector<OmniPipeSplitSliceInfo>& perLoop, uint32_t root,
    u64 osn, u64 xRankSize, u64 yRankSize, u64 rootx, u64 rootz, u64 one, u64 dataTypeSize)
{
    std::vector<u64> sliceCountMultRankPiece1;
    std::vector<u64> sliceSizeMultRankPiece1;
    std::vector<u64> inputOmniPipeSliceStrideMultRankPiece1;
    std::vector<u64> outputOmniPipeSliceStrideMultRankPiece1;
    for (u64 cornerDataSlice = 0; cornerDataSlice < xRankSize; cornerDataSlice++) {
        if (cornerDataSlice == rootx)
            continue;
        u64 currentDataSliceId = rootz * xRankSize * yRankSize + one * xRankSize + cornerDataSlice;
        u64 pieceId = currentDataSliceId;
        CalcAndPushPiece(
            pieceId, xySOffset[root][0] + ySOffset[root][osn][0], ySDataSize[root][osn][0], perLoop, perLoop,
            dataTypeSize, sliceSizeMultRankPiece1, sliceCountMultRankPiece1, inputOmniPipeSliceStrideMultRankPiece1,
            outputOmniPipeSliceStrideMultRankPiece1);
    }
    PushStepFields(
        stepSliceInfotmp, sliceSizeMultRankPiece1, sliceCountMultRankPiece1, inputOmniPipeSliceStrideMultRankPiece1,
        outputOmniPipeSliceStrideMultRankPiece1, 0, 0);
}

void PushScatterXOuterGTSameAxisOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevelx, u64 osn, u64 xSDataSize[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 ySDataSize[][MAX_STEP_NUM][MAX_STEP_NUM], u64 xySOffset[][MAX_STEP_NUM],
    u64 xSOffset[][MAX_STEP_NUM][MAX_STEP_NUM], u64 ySOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    u64 dataTypeSize, uint32_t root, u64 xRankSize, u64 yRankSize, u64 rootx, u64 rooty, u64 rootz,
    u64 yCclBufferBaseOff, u64 xInCornerStep, u64 innerStepNum)
{
    HCCL_DEBUG("[PushScatterXOuterGTSameAxisOneOsn] start push scatter x outer gt same axis one osn");
    for (u64 isn = xInCornerStep + 1; isn < innerStepNum; isn++) {
        struct StepSliceInfo stepSliceInfotmp;
        struct BuffInfo bitmp;
        BuffInfoAssign(bitmp, 0, 0, yCclBufferBaseOff);
        stepSliceInfotmp.buffInfo = bitmp;
        std::vector<u64> sliceSizeMultRankPiece;
        std::vector<u64> inputOmniPipeSliceStrideMultRankPiece;
        std::vector<u64> outputOmniPipeSliceStrideMultRankPiece;
        std::vector<u64> sliceCountMultRankPiece;
        for (u64 one = 0; one < xRankSize; one++) {
            if (one == rootx)
                continue;
            u64 pieceId = rootz * xRankSize * yRankSize + rooty * xRankSize + one;
            CalcAndPushPiece(
                pieceId, xySOffset[root][osn] + xSOffset[root][osn][isn], xSDataSize[root][osn][isn], perLoop, total,
                dataTypeSize, sliceSizeMultRankPiece, sliceCountMultRankPiece, inputOmniPipeSliceStrideMultRankPiece,
                outputOmniPipeSliceStrideMultRankPiece);
        }
        for (u64 one = 0; one < yRankSize; one++) {
            if (one == rooty) {
                PushStepFields(
                    stepSliceInfotmp, sliceSizeMultRankPiece, sliceCountMultRankPiece,
                    inputOmniPipeSliceStrideMultRankPiece, outputOmniPipeSliceStrideMultRankPiece, 0, 0);
            } else {
                PushScatterXOuterGTFwdOneRank(
                    stepSliceInfotmp, ySDataSize, xySOffset, ySOffset, perLoop, root, osn, xRankSize, yRankSize, rootx,
                    rootz, one, dataTypeSize);
            }
        }
        dataSliceLevelx.insert(dataSliceLevelx.end(), stepSliceInfotmp);
    }
}

// scatter y轴内层斜对角段（单个osn，isn∈[0,yInCornerStep)）：root和同轴线节点处理y轴斜对角通信
void PushScatterYInnerCornerOneDiag(
    std::vector<u64>& sliceSizeMultRankPiece, std::vector<u64>& sliceCountMultRankPiece,
    std::vector<u64>& inputOmniPipeSliceStrideMultRankPiece, std::vector<u64>& outputOmniPipeSliceStrideMultRankPiece,
    u64 osn, u64 isn, u64 ySDataSize[][MAX_STEP_NUM][MAX_STEP_NUM], u64 xySOffset[][MAX_STEP_NUM],
    u64 ySOffset[][MAX_STEP_NUM][MAX_STEP_NUM], const std::vector<OmniPipeSplitSliceInfo>& perLoop,
    const std::vector<OmniPipeSplitSliceInfo>& total, u64 dataTypeSize, u64 maxDataPieceId, u64 xRankSize,
    u64 yRankSize, u64 zRankSize, u64 xAxis, u64 zAxis, u64 oneDid)
{
    HCCL_DEBUG("[PushScatterYInnerCornerOneDiag] start push scatter y inner corner one diag");
    u64 outputslicestride = 0;
    for (u64 outSliceNum = 0; outSliceNum < zRankSize; outSliceNum++) {
        if (outSliceNum != zAxis) {
            for (u64 cornerDataSlice = 0; cornerDataSlice < xRankSize; cornerDataSlice++) {
                u64 currentInnerStepDataSliceId
                    = outSliceNum * xRankSize * yRankSize + oneDid * xRankSize + cornerDataSlice;
                if (cornerDataSlice != xAxis || xRankSize == 1) {
                    u64 pieceId = currentInnerStepDataSliceId;
                    u64 sliceSizeOnePiece = DataSliceCut(
                        ySDataSize[maxDataPieceId][osn][isn],
                        xySOffset[maxDataPieceId][osn] + ySOffset[maxDataPieceId][osn][isn], perLoop[pieceId].size);
                    u64 inputPieceIdOffset = sliceOffsetCut(
                                                 xySOffset[maxDataPieceId][osn] + ySOffset[maxDataPieceId][osn][isn],
                                                 perLoop[pieceId].size)
                                             + total[pieceId].offset;
                    sliceSizeMultRankPiece.push_back(sliceSizeOnePiece);
                    sliceCountMultRankPiece.push_back(sliceSizeOnePiece / dataTypeSize);
                    inputOmniPipeSliceStrideMultRankPiece.push_back(inputPieceIdOffset);
                    outputOmniPipeSliceStrideMultRankPiece.push_back(outputslicestride);
                    outputslicestride += ySDataSize[maxDataPieceId][osn][isn];
                }
            }
        }
    }
}

void PushScatterYInnerCornerOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevely, u64 osn, u64 ySDataSize[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 xySOffset[][MAX_STEP_NUM], u64 ySOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    u64 dataTypeSize, u64 maxDataPieceId, u64 xRankSize, u64 yRankSize, u64 zRankSize, u64 xAxis, u64 zAxis,
    u64 yCclBufferBaseOff, u64 yInCornerStep)
{
    HCCL_DEBUG("[PushScatterYInnerCornerOneOsn] start push scatter y inner corner one osn");
    for (u64 isn = 0; isn < yInCornerStep; isn++) {
        struct BuffInfo bitmp;
        struct StepSliceInfo stepSliceInfotmp;
        BuffInfoAssign(bitmp, 0, 0, yCclBufferBaseOff);
        stepSliceInfotmp.buffInfo = bitmp;
        std::vector<u64> sliceSizeMultRankPiece;
        std::vector<u64> outputOmniPipeSliceStrideMultRankPiece;
        std::vector<u64> sliceCountMultRankPiece;
        std::vector<u64> inputOmniPipeSliceStrideMultRankPiece;
        for (u64 oneDid = 0; oneDid < yRankSize; oneDid++) {
            PushScatterYInnerCornerOneDiag(
                sliceSizeMultRankPiece, sliceCountMultRankPiece, inputOmniPipeSliceStrideMultRankPiece,
                outputOmniPipeSliceStrideMultRankPiece, osn, isn, ySDataSize, xySOffset, ySOffset, perLoop, total,
                dataTypeSize, maxDataPieceId, xRankSize, yRankSize, zRankSize, xAxis, zAxis, oneDid);
            PushStepFields(
                stepSliceInfotmp, sliceSizeMultRankPiece, sliceCountMultRankPiece,
                inputOmniPipeSliceStrideMultRankPiece, outputOmniPipeSliceStrideMultRankPiece, 0, 0);
        }
        dataSliceLevely.insert(dataSliceLevely.end(), stepSliceInfotmp);
    }
}

// scatter y轴内层同轴段（单个osn，isn∈[yInCornerStep,innerStepNum)）：root和同轴线节点处理y轴同轴通信
void PushScatterYInnerSameAxisOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevely, u64 osn, u64 ySDataSize[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 xySOffset[][MAX_STEP_NUM], u64 ySOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    u64 dataTypeSize, u64 maxDataPieceId, u64 xRankSize, u64 yRankSize, u64 zRankSize, u64 xAxis, u64 zAxis,
    const std::vector<u64>& dataSizePerLoop, u64 yCclBufferBaseOff, u64 yInCornerStep, u64 innerStepNum)
{
    HCCL_DEBUG("[PushScatterYInnerSameAxisOneOsn] start push scatter y inner same axis one osn");
    for (u64 isn = yInCornerStep; isn < innerStepNum; isn++) {
        struct StepSliceInfo stepSliceInfotmp;
        struct BuffInfo bitmp;
        BuffInfoAssign(bitmp, 0, 0, yCclBufferBaseOff);
        stepSliceInfotmp.buffInfo = bitmp;
        for (u64 oneDid = 0; oneDid < yRankSize; oneDid++) {
            std::vector<u64> outputOmniPipeSliceStrideMultRankPiece;
            std::vector<u64> sliceSizeMultRankPiece;
            std::vector<u64> sliceCountMultRankPiece;
            std::vector<u64> inputOmniPipeSliceStrideMultRankPiece;
            u64 outputslicestride = 0;
            for (u64 outSliceNum = 0; outSliceNum < zRankSize; outSliceNum++) {
                u64 currentDataSliceId = outSliceNum * xRankSize * yRankSize + oneDid * xRankSize + xAxis;
                if (outSliceNum != zAxis) {
                    u64 pieceId = currentDataSliceId;
                    u64 sliceSizeOnePiece = DataSliceCut(
                        ySDataSize[maxDataPieceId][osn][isn],
                        xySOffset[maxDataPieceId][osn] + ySOffset[maxDataPieceId][osn][isn], perLoop[pieceId].size);
                    u64 inputPieceIdOffset = sliceOffsetCut(
                                                 xySOffset[maxDataPieceId][osn] + ySOffset[maxDataPieceId][osn][isn],
                                                 perLoop[pieceId].size)
                                             + total[pieceId].offset;
                    outputslicestride
                        = outSliceNum * dataSizePerLoop[maxDataPieceId] + ySOffset[maxDataPieceId][osn][isn];
                    sliceSizeMultRankPiece.push_back(sliceSizeOnePiece);
                    sliceCountMultRankPiece.push_back(sliceSizeOnePiece / dataTypeSize);
                    inputOmniPipeSliceStrideMultRankPiece.push_back(inputPieceIdOffset);
                    outputOmniPipeSliceStrideMultRankPiece.push_back(outputslicestride);
                    outputslicestride += ySDataSize[maxDataPieceId][osn][isn];
                }
            }
            PushStepFields(
                stepSliceInfotmp, sliceSizeMultRankPiece, sliceCountMultRankPiece,
                inputOmniPipeSliceStrideMultRankPiece, outputOmniPipeSliceStrideMultRankPiece, 0, 0);
        }
        dataSliceLevely.insert(dataSliceLevely.end(), stepSliceInfotmp);
    }
}

// scatter y轴外层 xB<=yB
// 斜对角段（单个osn，isn∈[0,yInCornerStep+1)）：root发斜对角数据，root数据放index=rootx，其余x轴rank塞0
void PushScatterYOuterLECornerOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevely, u64 osn, u64 ySDataSize[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 xySOffset[][MAX_STEP_NUM], u64 ySOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    u64 dataTypeSize, uint32_t root, u64 xRankSize, u64 yRankSize, u64 rootx, u64 rooty, u64 zAxis,
    u64 yCclBufferBaseOff, u64 yInCornerStep)
{
    HCCL_DEBUG("[PushScatterYOuterLECornerOneOsn] start push scatter y outer le corner one osn");
    for (u64 isn = 0; isn < yInCornerStep + 1; isn++) {
        struct StepSliceInfo stepSliceInfotmp;
        struct BuffInfo bitmp;
        BuffInfoAssign(bitmp, 0, 0, yCclBufferBaseOff);
        stepSliceInfotmp.buffInfo = bitmp;
        std::vector<u64> sliceSizeMultRankPiece;
        std::vector<u64> sliceCountMultRankPiece;
        std::vector<u64> inputOmniPipeSliceStrideMultRankPiece;
        std::vector<u64> outputOmniPipeSliceStrideMultRankPiece;
        for (u64 oneDid = 0; oneDid < yRankSize; oneDid++) {
            if (oneDid == rooty)
                continue;
            for (u64 cornerDataSlice = 0; cornerDataSlice < xRankSize; cornerDataSlice++) {
                u64 currentDataSliceId = zAxis * xRankSize * yRankSize + oneDid * xRankSize + cornerDataSlice;
                if (cornerDataSlice != rootx) {
                    CalcAndPushPiece(
                        currentDataSliceId, xySOffset[root][osn] + ySOffset[root][osn][isn], ySDataSize[root][osn][isn],
                        perLoop, total, dataTypeSize, sliceSizeMultRankPiece, sliceCountMultRankPiece,
                        inputOmniPipeSliceStrideMultRankPiece, outputOmniPipeSliceStrideMultRankPiece);
                }
            }
        }
        for (u64 oneDid = 0; oneDid < xRankSize; oneDid++) {
            PushRootOrZeros(
                stepSliceInfotmp, sliceSizeMultRankPiece, sliceCountMultRankPiece,
                inputOmniPipeSliceStrideMultRankPiece, outputOmniPipeSliceStrideMultRankPiece, oneDid, rootx, 0);
        }
        dataSliceLevely.insert(dataSliceLevely.end(), stepSliceInfotmp);
    }
}

// scatter y轴外层 xB<=yB 同轴转发段：为非rootx的x轴节点构建转发piece并Push
void PushScatterYOuterLEFwdOneRank(
    StepSliceInfo& stepSliceInfotmp, u64 xSDataSize[][MAX_STEP_NUM][MAX_STEP_NUM], u64 xySOffset[][MAX_STEP_NUM],
    u64 xSOffset[][MAX_STEP_NUM][MAX_STEP_NUM], const std::vector<OmniPipeSplitSliceInfo>& perLoop, uint32_t root,
    u64 osn, u64 xRankSize, u64 yRankSize, u64 rooty, u64 rootz, u64 oneDid, u64 dataTypeSize)
{
    HCCL_DEBUG("[PushScatterYOuterLEFwdOneRank] start push scatter y outer le fwd one rank");
    std::vector<u64> sliceSizeMultRankPiece1;
    std::vector<u64> sliceCountMultRankPiece1;
    std::vector<u64> inputOmniPipeSliceStrideMultRankPiece1;
    std::vector<u64> outputOmniPipeSliceStrideMultRankPiece1;
    for (u64 cornerDataSlice = 0; cornerDataSlice < yRankSize; cornerDataSlice++) {
        if (cornerDataSlice == rooty)
            continue;
        u64 currentDataSliceId = rootz * xRankSize * yRankSize + cornerDataSlice * xRankSize + oneDid;
        CalcAndPushPiece(
            currentDataSliceId, xySOffset[root][0] + xSOffset[root][osn][0], xSDataSize[root][osn][0], perLoop, perLoop,
            dataTypeSize, sliceSizeMultRankPiece1, sliceCountMultRankPiece1, inputOmniPipeSliceStrideMultRankPiece1,
            outputOmniPipeSliceStrideMultRankPiece1);
    }
    PushStepFields(
        stepSliceInfotmp, sliceSizeMultRankPiece1, sliceCountMultRankPiece1, inputOmniPipeSliceStrideMultRankPiece1,
        outputOmniPipeSliceStrideMultRankPiece1, 0, 0);
}

// scatter y轴外层 xB<=yB
// 同轴转发段（单个osn，isn∈[yInCornerStep+1,innerStepNum)）：root发同y轴数据，同x轴非root节点转发step1收到的对角数据
void PushScatterYOuterLESameAxisOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevely, u64 osn, u64 xSDataSize[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 ySDataSize[][MAX_STEP_NUM][MAX_STEP_NUM], u64 xySOffset[][MAX_STEP_NUM],
    u64 xSOffset[][MAX_STEP_NUM][MAX_STEP_NUM], u64 ySOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    u64 dataTypeSize, uint32_t root, u64 xRankSize, u64 yRankSize, u64 rootx, u64 rooty, u64 rootz,
    u64 yCclBufferBaseOff, u64 yInCornerStep, u64 innerStepNum)
{
    HCCL_DEBUG("[PushScatterYOuterLESameAxisOneOsn] start push scatter y outer le same axis one osn");
    for (u64 isn = yInCornerStep + 1; isn < innerStepNum; isn++) {
        struct StepSliceInfo stepSliceInfotmp;
        struct BuffInfo bitmp;
        BuffInfoAssign(bitmp, 0, 0, yCclBufferBaseOff);
        stepSliceInfotmp.buffInfo = bitmp;
        std::vector<u64> sliceSizeMultRankPiece;
        std::vector<u64> inputOmniPipeSliceStrideMultRankPiece;
        std::vector<u64> outputOmniPipeSliceStrideMultRankPiece;
        std::vector<u64> sliceCountMultRankPiece;
        for (u64 oneDid = 0; oneDid < yRankSize; oneDid++) {
            if (oneDid == rooty)
                continue;
            u64 pieceId = rootz * xRankSize * yRankSize + oneDid * xRankSize + rootx;
            CalcAndPushPiece(
                pieceId, xySOffset[root][osn] + ySOffset[root][osn][isn], ySDataSize[root][osn][isn], perLoop, total,
                dataTypeSize, sliceSizeMultRankPiece, sliceCountMultRankPiece, inputOmniPipeSliceStrideMultRankPiece,
                outputOmniPipeSliceStrideMultRankPiece);
        }
        for (u64 oneDid = 0; oneDid < xRankSize; oneDid++) {
            if (oneDid == rootx) {
                PushStepFields(
                    stepSliceInfotmp, sliceSizeMultRankPiece, sliceCountMultRankPiece,
                    inputOmniPipeSliceStrideMultRankPiece, outputOmniPipeSliceStrideMultRankPiece, 0, 0);
            } else {
                PushScatterYOuterLEFwdOneRank(
                    stepSliceInfotmp, xSDataSize, xySOffset, xSOffset, perLoop, root, osn, xRankSize, yRankSize, rooty,
                    rootz, oneDid, dataTypeSize);
            }
        }
        dataSliceLevely.insert(dataSliceLevely.end(), stepSliceInfotmp);
    }
}

// scatter y轴外层 xB>yB
// 斜对角段（单个osn，isn∈[0,yInCornerStep)）：第一步只有root发斜对角数据，buffer用xCclBufferBaseOff
void PushScatterYOuterGTCornerOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevely, u64 osn, u64 ySDataSize[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 xySOffset[][MAX_STEP_NUM], u64 ySOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    u64 dataTypeSize, uint32_t root, u64 xRankSize, u64 yRankSize, u64 rootx, u64 rooty, u64 rootz,
    u64 xCclBufferBaseOff, u64 yInCornerStep)
{
    HCCL_DEBUG("[PushScatterYOuterGTCornerOneOsn] start push scatter y outer gt corner one osn");
    for (u64 isn = 0; isn < yInCornerStep; isn++) {
        struct StepSliceInfo stepSliceInfotmp;
        struct BuffInfo bitmp;
        BuffInfoAssign(bitmp, 0, 0, xCclBufferBaseOff);
        stepSliceInfotmp.buffInfo = bitmp;
        std::vector<u64> outputOmniPipeSliceStrideMultRankPiece;
        std::vector<u64> sliceCountMultRankPiece;
        std::vector<u64> sliceSizeMultRankPiece;
        std::vector<u64> inputOmniPipeSliceStrideMultRankPiece;
        for (u64 oneDid = 0; oneDid < yRankSize; oneDid++) {
            if (oneDid == rooty) {
                continue;
            }
            for (u64 oneDataSlice = 0; oneDataSlice < xRankSize; oneDataSlice++) {
                u64 currentDataSliceId = rootz * xRankSize * yRankSize + oneDid * xRankSize + oneDataSlice;
                if (oneDataSlice != rootx) {
                    CalcAndPushPiece(
                        currentDataSliceId, xySOffset[root][osn] + ySOffset[root][osn][isn], ySDataSize[root][osn][isn],
                        perLoop, total, dataTypeSize, sliceSizeMultRankPiece, sliceCountMultRankPiece,
                        inputOmniPipeSliceStrideMultRankPiece, outputOmniPipeSliceStrideMultRankPiece);
                }
            }
        }
        for (u64 oneRank = 0; oneRank < xRankSize; oneRank++) {
            PushRootOrZeros(
                stepSliceInfotmp, sliceSizeMultRankPiece, sliceCountMultRankPiece,
                inputOmniPipeSliceStrideMultRankPiece, outputOmniPipeSliceStrideMultRankPiece, oneRank, rootx, 0);
        }
        dataSliceLevely.insert(dataSliceLevely.end(), stepSliceInfotmp);
    }
}

// scatter y轴外层 xB>yB
// 同轴转发段（单个osn，isn∈[yInCornerStep,innerStepNum)）：root发同y轴数据，同x轴非root节点转发step1收到的对角数据
void CalcScatterYOuterGTOffset(
    u64& inputPieceIdOffset, u64& outputPieceIdOffset, u64& sliceSizeOnePiece,
    u64 xSDataSize[][MAX_STEP_NUM][MAX_STEP_NUM], u64 ySDataSize[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 xySOffset[][MAX_STEP_NUM], u64 xSOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, uint32_t root, u64 osn, u64 isn, u64 innerStepNum, u64 pieceId)
{
    HCCL_DEBUG("[CalcScatterYOuterGTOffset] start calc scatter y outer gt offset");
    if (innerStepNum == 2) {
        inputPieceIdOffset = sliceOffsetCut(xySOffset[root][osn] + xSOffset[root][osn][isn - 1], perLoop[pieceId].size)
                             + perLoop[pieceId].offset;
        outputPieceIdOffset = sliceOffsetCut(xySOffset[root][osn] + xSOffset[root][osn][isn - 1], perLoop[pieceId].size)
                              + perLoop[pieceId].offset;
    } else {
        if (isn == innerStepNum - 1) {
            if (ySDataSize[root][osn][isn - 1] < xSDataSize[root][osn][isn - 2]) {
                inputPieceIdOffset
                    = sliceOffsetCut(
                          xySOffset[root][osn] + xSOffset[root][osn][isn - 2] + ySDataSize[root][osn][isn - 1],
                          perLoop[pieceId].size)
                      + perLoop[pieceId].offset;
                outputPieceIdOffset
                    = sliceOffsetCut(
                          xySOffset[root][osn] + xSOffset[root][osn][isn - 2] + ySDataSize[root][osn][isn - 1],
                          perLoop[pieceId].size)
                      + perLoop[pieceId].offset;
            } else {
                inputPieceIdOffset
                    = sliceOffsetCut(
                          xySOffset[root][osn] + xSOffset[root][osn][isn - 2] + xSDataSize[root][osn][isn - 2],
                          perLoop[pieceId].size)
                      + perLoop[pieceId].offset;
                outputPieceIdOffset
                    = sliceOffsetCut(
                          xySOffset[root][osn] + xSOffset[root][osn][isn - 2] + xSDataSize[root][osn][isn - 2],
                          perLoop[pieceId].size)
                      + perLoop[pieceId].offset;
            }
        } else {
            if (sliceSizeOnePiece > xSDataSize[root][osn][isn - 1]) {
                sliceSizeOnePiece = xSDataSize[root][osn][isn - 1];
            }
            inputPieceIdOffset
                = sliceOffsetCut(xySOffset[root][osn] + xSOffset[root][osn][isn - 1], perLoop[pieceId].size)
                  + perLoop[pieceId].offset;
            outputPieceIdOffset
                = sliceOffsetCut(xySOffset[root][osn] + xSOffset[root][osn][isn - 1], perLoop[pieceId].size)
                  + perLoop[pieceId].offset;
        }
    }
}

// scatter y轴外层 xB>yB 同轴转发段：为非rootx的x轴节点构建转发piece并Push
void PushScatterYOuterGTFwdOneRank(
    StepSliceInfo& stepSliceInfotmp, u64 xSDataSize[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 ySDataSize[][MAX_STEP_NUM][MAX_STEP_NUM], u64 xySOffset[][MAX_STEP_NUM],
    u64 xSOffset[][MAX_STEP_NUM][MAX_STEP_NUM], u64 ySOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, uint32_t root, u64 osn, u64 isn, u64 innerStepNum,
    u64 xRankSize, u64 yRankSize, u64 rooty, u64 rootz, u64 rankx, u64 dataTypeSize,
    const std::vector<u64>& dataSizePerLoop)
{
    HCCL_DEBUG("[PushScatterYOuterGTFwdOneRank] start push scatter y outer gt fwd one rank");
    std::vector<u64> sliceSizeMultRankPiece;
    std::vector<u64> inputOmniPipeSliceStrideMultRankPiece;
    std::vector<u64> sliceCountMultRankPiece;
    std::vector<u64> outputOmniPipeSliceStrideMultRankPiece;
    for (u64 cornerDataSlice = 0; cornerDataSlice < yRankSize; cornerDataSlice++) {
        if (cornerDataSlice == rooty)
            continue;
        u64 currentDataSliceId = rootz * xRankSize * yRankSize + cornerDataSlice * xRankSize + rankx;
        u64 pieceId = currentDataSliceId;
        u64 sliceSizeOnePiece = DataSliceCut(
            ySDataSize[root][osn][isn], xySOffset[root][osn] + ySOffset[root][osn][isn], perLoop[pieceId].size);
        u64 inputPieceIdOffset = 0;
        u64 outputPieceIdOffset = 0;
        CalcScatterYOuterGTOffset(
            inputPieceIdOffset, outputPieceIdOffset, sliceSizeOnePiece, xSDataSize, ySDataSize, xySOffset, xSOffset,
            perLoop, root, osn, isn, innerStepNum, pieceId);
        if (inputPieceIdOffset + sliceSizeOnePiece > perLoop[pieceId].offset + dataSizePerLoop[root]) {
            sliceSizeOnePiece = perLoop[pieceId].offset + dataSizePerLoop[root] - inputPieceIdOffset;
        }
        sliceCountMultRankPiece.push_back(sliceSizeOnePiece / dataTypeSize);
        sliceSizeMultRankPiece.push_back(sliceSizeOnePiece);
        outputOmniPipeSliceStrideMultRankPiece.push_back(outputPieceIdOffset);
        inputOmniPipeSliceStrideMultRankPiece.push_back(inputPieceIdOffset);
    }
    PushStepFields(
        stepSliceInfotmp, sliceSizeMultRankPiece, sliceCountMultRankPiece, inputOmniPipeSliceStrideMultRankPiece,
        outputOmniPipeSliceStrideMultRankPiece, 0, 0);
}

void PushScatterYOuterGTSameAxisOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevely, u64 osn, u64 xSDataSize[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 ySDataSize[][MAX_STEP_NUM][MAX_STEP_NUM], u64 xySOffset[][MAX_STEP_NUM],
    u64 xSOffset[][MAX_STEP_NUM][MAX_STEP_NUM], u64 ySOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    u64 dataTypeSize, uint32_t root, u64 xRankSize, u64 yRankSize, u64 rootx, u64 rooty, u64 rootz,
    u64 xCclBufferBaseOff, u64 yInCornerStep, u64 innerStepNum, const std::vector<u64>& dataSizePerLoop)
{
    HCCL_DEBUG("[PushScatterYOuterGTSameAxisOneOsn] start push scatter y outer gt same axis one osn");
    for (u64 isn = yInCornerStep; isn < innerStepNum; isn++) {
        struct StepSliceInfo stepSliceInfotmp;
        struct BuffInfo bitmp;
        BuffInfoAssign(bitmp, 0, 0, xCclBufferBaseOff);
        stepSliceInfotmp.buffInfo = bitmp;
        std::vector<u64> sliceSizeMultRankPiece;
        std::vector<u64> sliceCountMultRankPiece;
        std::vector<u64> outputOmniPipeSliceStrideMultRankPiece;
        std::vector<u64> inputOmniPipeSliceStrideMultRankPiece;
        for (u64 oneDid = 0; oneDid < yRankSize; oneDid++) {
            if (oneDid == rooty)
                continue;
            u64 pieceId = rootz * xRankSize * yRankSize + oneDid * xRankSize + rootx;
            CalcAndPushPiece(
                pieceId, xySOffset[root][osn] + ySOffset[root][osn][isn], ySDataSize[root][osn][isn], perLoop, total,
                dataTypeSize, sliceSizeMultRankPiece, sliceCountMultRankPiece, inputOmniPipeSliceStrideMultRankPiece,
                outputOmniPipeSliceStrideMultRankPiece);
        }
        for (u64 rankx = 0; rankx < xRankSize; rankx++) {
            if (rankx == rootx) {
                PushStepFields(
                    stepSliceInfotmp, sliceSizeMultRankPiece, sliceCountMultRankPiece,
                    inputOmniPipeSliceStrideMultRankPiece, outputOmniPipeSliceStrideMultRankPiece, 0, 0);
            } else {
                PushScatterYOuterGTFwdOneRank(
                    stepSliceInfotmp, xSDataSize, ySDataSize, xySOffset, xSOffset, ySOffset, perLoop, root, osn, isn,
                    innerStepNum, xRankSize, yRankSize, rooty, rootz, rankx, dataTypeSize, dataSizePerLoop);
            }
        }
        dataSliceLevely.insert(dataSliceLevely.end(), stepSliceInfotmp);
    }
}

ScatterTopoInfo InitScatterTopoInfo(OmniPipeSliceParam& omniPipeSliceParam, uint32_t root)
{
    HCCL_DEBUG("[InitScatterTopoInfo] start init some scatter topo info");
    ScatterTopoInfo info;
    info.processedDataEachRank = 0;
    std::vector<u64> levelRankSize = omniPipeSliceParam.levelRankSize;
    std::vector<u64> dataSize = omniPipeSliceParam.dataWholeSize;
    info.maxDataPieceId = 0;
    for (u64 i = 0; i < dataSize.size(); i++) {
        if (dataSize[info.maxDataPieceId] < dataSize[i]) {
            info.maxDataPieceId = i;
        }
    }
    std::vector<double> endpointAttrBw = omniPipeSliceParam.endpointAttrBw;
    info.dataTypeSize = omniPipeSliceParam.dataTypeSize;
    std::vector<u64> levelRankId = omniPipeSliceParam.levelRankId;
    info.xRankSize = levelRankSize[OmniPipeLevel::OMNIPIPE_LEVEL0];
    info.yRankSize = levelRankSize[OmniPipeLevel::OMNIPIPE_LEVEL1];
    info.zRankSize = levelRankSize[OmniPipeLevel::OMNIPIPE_LEVEL2];
    info.rankSize = info.zRankSize * info.yRankSize * info.xRankSize;
    info.xB = endpointAttrBw[OmniPipeLevel::OMNIPIPE_LEVEL0];
    info.yB = endpointAttrBw[OmniPipeLevel::OMNIPIPE_LEVEL1];
    info.zB = endpointAttrBw[OmniPipeLevel::OMNIPIPE_LEVEL2];
    info.xyB = info.xB;
    if (info.yB >= info.xB) {
        info.xyB = CalcBandwidth2D(info.xB, info.yB, info.xRankSize, info.yRankSize, MAX_STEP_NUM);
    } else {
        info.xyB = CalcBandwidth2D(info.yB, info.xB, info.yRankSize, info.xRankSize, MAX_STEP_NUM);
    }
    info.xAxis = levelRankId[OmniPipeLevel::OMNIPIPE_LEVEL0];
    info.yAxis = levelRankId[OmniPipeLevel::OMNIPIPE_LEVEL1];
    info.zAxis = levelRankId[OmniPipeLevel::OMNIPIPE_LEVEL2];
    info.rankid = info.xAxis + info.yAxis * info.xRankSize + info.zAxis * info.xRankSize * info.yRankSize;
    info.rootx = root % info.xRankSize;
    info.rooty = (root / info.xRankSize) % info.yRankSize;
    info.rootz = root / (info.xRankSize * info.yRankSize);
    return info;
}

void InitScatterStepFlags(ScatterStepState& state, const ScatterTopoInfo& topo)
{
    HCCL_DEBUG("[InitScatterStepFlags] start init scatter step flags");
    state.outerStepNum = 0;
    state.innerStepNum = 0;
    state.xyCornerStep = 0;
    state.xInCornerStep = 1;
    state.yInCornerStep = 0;
    state.zCornerStep = 0;
    state.isZSlowAxis = (topo.xyB > topo.zB);
    state.isXSlowAxis = (topo.yB < topo.xB);
}

// 零初始化scatter数据大小与偏移数组
void ZeroInitScatterDataArrays(
    u64 rankSize, u64 zSDataSize[][MAX_STEP_NUM], u64 xySDataSize[][MAX_STEP_NUM],
    u64 xSDataSize[][MAX_STEP_NUM][MAX_STEP_NUM], u64 ySDataSize[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 zSOffset[][MAX_STEP_NUM], u64 xSOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 ySOffset[][MAX_STEP_NUM][MAX_STEP_NUM], u64 xySOffset[][MAX_STEP_NUM])
{
    for (u64 rs = 0; rs < rankSize; rs++) {
        for (u64 i = 0; i < MAX_STEP_NUM; i++) {
            zSDataSize[rs][i] = 0;
            xySDataSize[rs][i] = 0;
            zSOffset[rs][i] = 0;
            xySOffset[rs][i] = 0;
            for (u64 j = 0; j < MAX_STEP_NUM; j++) {
                xSDataSize[rs][i][j] = 0;
                ySDataSize[rs][i][j] = 0;
                xSOffset[rs][i][j] = 0;
                ySOffset[rs][i][j] = 0;
            }
        }
    }
}

// 计算单个rank的scatter数据大小与偏移（isZSlowAxis决定外层轴选择，isXSlowAxis决定内层轴选择）
void CalcScatterOneRankDataSize(
    const ScatterTopoInfo& topo, ScatterStepState& state, u64 rs, u64 finStepMark, double slowBw, double fastBw,
    u64 slowRankSize, u64 fastRankSize, u64 zSDataSize[][MAX_STEP_NUM], u64 xySDataSize[][MAX_STEP_NUM],
    u64 xSDataSize[][MAX_STEP_NUM][MAX_STEP_NUM], u64 ySDataSize[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 zSOffset[][MAX_STEP_NUM], u64 xSOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 ySOffset[][MAX_STEP_NUM][MAX_STEP_NUM], u64 xySOffset[][MAX_STEP_NUM],
    const std::vector<OmniPipeSplitSliceInfo>& omniPipeSplitSliceInfoListPerLoop)
{
    u64* slowDataSize = state.isZSlowAxis ? zSDataSize[rs] : xySDataSize[rs];
    u64* fastDataSize = state.isZSlowAxis ? xySDataSize[rs] : zSDataSize[rs];
    u64* slowOffset = state.isZSlowAxis ? zSOffset[rs] : xySOffset[rs];
    u64* fastOffset = state.isZSlowAxis ? xySOffset[rs] : zSOffset[rs];

    state.outerStepNum = CalScatterDataSize2D(
        slowDataSize, fastDataSize, slowBw, fastBw, slowRankSize, fastRankSize,
        omniPipeSplitSliceInfoListPerLoop[rs].size, MAX_STEP_NUM);
    HCCL_DEBUG("[CalcScatterOneRankDataSize] outerStepNum: %llu", state.outerStepNum);

    double innerSlowBw = state.isXSlowAxis ? topo.yB : topo.xB;
    double innerFastBw = state.isXSlowAxis ? topo.xB : topo.yB;
    u64 innerSlowRankSize = state.isXSlowAxis ? topo.yRankSize : topo.xRankSize;
    u64 innerFastRankSize = state.isXSlowAxis ? topo.xRankSize : topo.yRankSize;

    for (u64 i = 0; i < state.outerStepNum; i++) {
        u64* innerSlowDataSize = state.isXSlowAxis ? ySDataSize[rs][i] : xSDataSize[rs][i];
        u64* innerFastDataSize = state.isXSlowAxis ? xSDataSize[rs][i] : ySDataSize[rs][i];
        u64* innerSlowOffset = state.isXSlowAxis ? ySOffset[rs][i] : xSOffset[rs][i];
        u64* innerFastOffset = state.isXSlowAxis ? xSOffset[rs][i] : ySOffset[rs][i];

        state.innerStepNum = CalScatterDataSize2D(
            innerSlowDataSize, innerFastDataSize, innerSlowBw, innerFastBw, innerSlowRankSize, innerFastRankSize,
            state.isZSlowAxis ? fastDataSize[i] : slowDataSize[i], MAX_STEP_NUM);
        HCCL_DEBUG("[CalcScatterOneRankDataSize] innerStepNum: %llu", state.innerStepNum);
        CalScatter2DOffset(
            innerSlowOffset, innerFastOffset, state.innerStepNum, innerSlowRankSize, innerFastRankSize,
            innerSlowDataSize, innerFastDataSize);
    }
    if (state.innerStepNum > finStepMark) {
        if (state.isXSlowAxis) {
            state.yInCornerStep = 1;
            state.xInCornerStep = state.innerStepNum - finStepMark;
        } else {
            state.xInCornerStep = 1;
            state.yInCornerStep = state.innerStepNum - finStepMark;
        }
    }

    CalScatter2DOffset(
        slowOffset, fastOffset, state.outerStepNum, slowRankSize, fastRankSize, slowDataSize, fastDataSize);
}

// 计算外层corner step（z轴与xy轴的对齐步数）
void CalcScatterOuterCornerStep(ScatterStepState& state, u64 finStepMark)
{
    HCCL_DEBUG("[CalcScatterOuterCornerStep] start calc scatter outer corner step");
    if (state.outerStepNum > finStepMark) {
        if (state.isZSlowAxis) {
            state.zCornerStep = 1;
            state.xyCornerStep = state.outerStepNum - finStepMark;
        } else {
            state.zCornerStep = state.outerStepNum - finStepMark;
            state.xyCornerStep = 1;
        }
    }
}

// 计算所有rank的scatter数据大小与偏移
void CalcScatterAllRankDataSize(
    const ScatterTopoInfo& topo, ScatterStepState& state, uint32_t root, u64 finStepMark,
    u64 zSDataSize[][MAX_STEP_NUM], u64 xySDataSize[][MAX_STEP_NUM], u64 xSDataSize[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 ySDataSize[][MAX_STEP_NUM][MAX_STEP_NUM], u64 zSOffset[][MAX_STEP_NUM],
    u64 xSOffset[][MAX_STEP_NUM][MAX_STEP_NUM], u64 ySOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 xySOffset[][MAX_STEP_NUM], const std::vector<OmniPipeSplitSliceInfo>& omniPipeSplitSliceInfoListPerLoop)
{
    HCCL_DEBUG("[CalcScatterAllRankDataSize] start calc scatter all rank data size");
    double slowBw = state.isZSlowAxis ? topo.zB : topo.xyB;
    double fastBw = state.isZSlowAxis ? topo.xyB : topo.zB;
    u64 slowRankSize = state.isZSlowAxis ? topo.zRankSize : (topo.xRankSize * topo.yRankSize);
    u64 fastRankSize = state.isZSlowAxis ? (topo.xRankSize * topo.yRankSize) : topo.zRankSize;
    for (u64 rs = 0; rs < topo.rankSize; rs++) {
        bool ifroot;
        bool isSameAxis;
        CheckRootOrSameAxisAsRoot(topo.xRankSize, topo.yRankSize, topo.zRankSize, root, rs, ifroot, isSameAxis);

        if (ifroot || isSameAxis) {
            CalcScatterOneRankDataSize(
                topo, state, rs, finStepMark, slowBw, fastBw, slowRankSize, fastRankSize, zSDataSize, xySDataSize,
                xSDataSize, ySDataSize, zSOffset, xSOffset, ySOffset, xySOffset, omniPipeSplitSliceInfoListPerLoop);
        }
    }
    CalcScatterOuterCornerStep(state, finStepMark);
}

// 构建X轴所有step的slice信息（inner corner+sameAxis + outer corner+sameAxis）
void PushScatterXAllSteps(
    std::vector<StepSliceInfo>& dataSliceLevelx, const ScatterTopoInfo& topo, const ScatterStepState& state,
    u64 xSDataSize[][MAX_STEP_NUM][MAX_STEP_NUM], u64 ySDataSize[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 xySOffset[][MAX_STEP_NUM], u64 xSOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 ySOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    const std::vector<OmniPipeSplitSliceInfo>& omniPipeSplitSliceInfoListPerLoop,
    const std::vector<OmniPipeSplitSliceInfo>& omniPipeSplitSliceInfoListTotal, uint32_t root, u64 xCclBufferBaseOff,
    u64 yCclBufferBaseOff, const std::vector<u64>& dataSizePerLoop)
{
    HCCL_DEBUG("[PushScatterXAllSteps] start push scatter X Axis all steps");
    for (u64 osn = 0; osn < state.xyCornerStep; osn++) {
        PushScatterXInnerCornerOneOsn(
            dataSliceLevelx, osn, xSDataSize, xySOffset, xSOffset, omniPipeSplitSliceInfoListPerLoop,
            omniPipeSplitSliceInfoListTotal, topo.dataTypeSize, topo.maxDataPieceId, topo.xRankSize, topo.yRankSize,
            topo.zRankSize, topo.yAxis, topo.zAxis, dataSizePerLoop, xCclBufferBaseOff, state.xInCornerStep);
        PushScatterXInnerSameAxisOneOsn(
            dataSliceLevelx, osn, xSDataSize, xySOffset, xSOffset, omniPipeSplitSliceInfoListPerLoop,
            omniPipeSplitSliceInfoListTotal, topo.dataTypeSize, topo.maxDataPieceId, topo.xRankSize, topo.yRankSize,
            topo.zRankSize, topo.yAxis, topo.zAxis, dataSizePerLoop, xCclBufferBaseOff, state.xInCornerStep,
            state.innerStepNum);
    }
    if (topo.xB <= topo.yB) {
        for (u64 osn = state.xyCornerStep; osn < state.outerStepNum; osn++) {
            PushScatterXOuterLECornerOneOsn(
                dataSliceLevelx, osn, xSDataSize, xySOffset, xSOffset, omniPipeSplitSliceInfoListPerLoop,
                omniPipeSplitSliceInfoListTotal, topo.dataTypeSize, root, topo.xRankSize, topo.yRankSize, topo.rootx,
                topo.rooty, topo.rootz, xCclBufferBaseOff, state.xInCornerStep);
            PushScatterXOuterLESameAxisOneOsn(
                dataSliceLevelx, osn, xSDataSize, ySDataSize, xySOffset, xSOffset, ySOffset,
                omniPipeSplitSliceInfoListPerLoop, omniPipeSplitSliceInfoListTotal, topo.dataTypeSize, root,
                topo.xRankSize, topo.yRankSize, topo.rootx, topo.rooty, topo.rootz, xCclBufferBaseOff,
                state.xInCornerStep, state.innerStepNum, dataSizePerLoop);
        }
    } else {
        for (u64 osn = state.xyCornerStep; osn < state.outerStepNum; osn++) {
            PushScatterXOuterGTCornerOneOsn(
                dataSliceLevelx, osn, xSDataSize, xySOffset, xSOffset, omniPipeSplitSliceInfoListPerLoop,
                omniPipeSplitSliceInfoListTotal, topo.dataTypeSize, root, topo.xRankSize, topo.yRankSize, topo.rootx,
                topo.rooty, topo.rootz, yCclBufferBaseOff, state.xInCornerStep);
            PushScatterXOuterGTSameAxisOneOsn(
                dataSliceLevelx, osn, xSDataSize, ySDataSize, xySOffset, xSOffset, ySOffset,
                omniPipeSplitSliceInfoListPerLoop, omniPipeSplitSliceInfoListTotal, topo.dataTypeSize, root,
                topo.xRankSize, topo.yRankSize, topo.rootx, topo.rooty, topo.rootz, yCclBufferBaseOff,
                state.xInCornerStep, state.innerStepNum);
        }
    }
}

// 构建Y轴所有step的slice信息（inner corner+sameAxis + outer corner+sameAxis）
void PushScatterYAllSteps(
    std::vector<StepSliceInfo>& dataSliceLevely, const ScatterTopoInfo& topo, const ScatterStepState& state,
    u64 xSDataSize[][MAX_STEP_NUM][MAX_STEP_NUM], u64 ySDataSize[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 xySOffset[][MAX_STEP_NUM], u64 xSOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 ySOffset[][MAX_STEP_NUM][MAX_STEP_NUM],
    const std::vector<OmniPipeSplitSliceInfo>& omniPipeSplitSliceInfoListPerLoop,
    const std::vector<OmniPipeSplitSliceInfo>& omniPipeSplitSliceInfoListTotal, uint32_t root, u64 xCclBufferBaseOff,
    u64 yCclBufferBaseOff, const std::vector<u64>& dataSizePerLoop)
{
    HCCL_DEBUG("[PushScatterYAllSteps] start push scatter Y Axis all steps");
    for (u64 osn = 0; osn < state.xyCornerStep; osn++) {
        PushScatterYInnerCornerOneOsn(
            dataSliceLevely, osn, ySDataSize, xySOffset, ySOffset, omniPipeSplitSliceInfoListPerLoop,
            omniPipeSplitSliceInfoListTotal, topo.dataTypeSize, topo.maxDataPieceId, topo.xRankSize, topo.yRankSize,
            topo.zRankSize, topo.xAxis, topo.zAxis, yCclBufferBaseOff, state.yInCornerStep);
        PushScatterYInnerSameAxisOneOsn(
            dataSliceLevely, osn, ySDataSize, xySOffset, ySOffset, omniPipeSplitSliceInfoListPerLoop,
            omniPipeSplitSliceInfoListTotal, topo.dataTypeSize, topo.maxDataPieceId, topo.xRankSize, topo.yRankSize,
            topo.zRankSize, topo.xAxis, topo.zAxis, dataSizePerLoop, yCclBufferBaseOff, state.yInCornerStep,
            state.innerStepNum);
    }
    if (topo.xB <= topo.yB) {
        HCCL_DEBUG("xB <= yB");
        for (u64 osn = state.xyCornerStep; osn < state.outerStepNum; osn++) {
            PushScatterYOuterLECornerOneOsn(
                dataSliceLevely, osn, ySDataSize, xySOffset, ySOffset, omniPipeSplitSliceInfoListPerLoop,
                omniPipeSplitSliceInfoListTotal, topo.dataTypeSize, root, topo.xRankSize, topo.yRankSize, topo.rootx,
                topo.rooty, topo.zAxis, yCclBufferBaseOff, state.yInCornerStep);
            PushScatterYOuterLESameAxisOneOsn(
                dataSliceLevely, osn, xSDataSize, ySDataSize, xySOffset, xSOffset, ySOffset,
                omniPipeSplitSliceInfoListPerLoop, omniPipeSplitSliceInfoListTotal, topo.dataTypeSize, root,
                topo.xRankSize, topo.yRankSize, topo.rootx, topo.rooty, topo.rootz, yCclBufferBaseOff,
                state.yInCornerStep, state.innerStepNum);
        }
    } else {
        HCCL_DEBUG("xB > yB");
        for (u64 osn = state.xyCornerStep; osn < state.outerStepNum; osn++) {
            PushScatterYOuterGTCornerOneOsn(
                dataSliceLevely, osn, ySDataSize, xySOffset, ySOffset, omniPipeSplitSliceInfoListPerLoop,
                omniPipeSplitSliceInfoListTotal, topo.dataTypeSize, root, topo.xRankSize, topo.yRankSize, topo.rootx,
                topo.rooty, topo.rootz, xCclBufferBaseOff, state.yInCornerStep);
            PushScatterYOuterGTSameAxisOneOsn(
                dataSliceLevely, osn, xSDataSize, ySDataSize, xySOffset, xSOffset, ySOffset,
                omniPipeSplitSliceInfoListPerLoop, omniPipeSplitSliceInfoListTotal, topo.dataTypeSize, root,
                topo.xRankSize, topo.yRankSize, topo.rootx, topo.rooty, topo.rootz, xCclBufferBaseOff,
                state.yInCornerStep, state.innerStepNum, dataSizePerLoop);
        }
    }
}

// 计算scratch size与buffer偏移，并构建Z轴所有step的slice信息
void PrepareScatterBuffersAndPushZ(
    std::vector<StepSliceInfo>& dataSliceLevelz, u64& xCclBufferBaseOff, u64& yCclBufferBaseOff, u64& zCclBufferBaseOff,
    const ScatterTopoInfo& topo, const ScatterStepState& state, u64 xSDataSize[][MAX_STEP_NUM][MAX_STEP_NUM],
    u64 ySDataSize[][MAX_STEP_NUM][MAX_STEP_NUM], u64 zSDataSize[][MAX_STEP_NUM], u64 zSOffset[][MAX_STEP_NUM],
    const std::vector<OmniPipeSplitSliceInfo>& omniPipeSplitSliceInfoListPerLoop,
    const std::vector<OmniPipeSplitSliceInfo>& omniPipeSplitSliceInfoListTotal,
    const OmniPipeSliceParam& omniPipeSliceParam)
{
    HCCL_DEBUG("[PrepareScatterBuffersAndPushZ] start prepare scatter buffers and push Z Axis");
    std::vector<u64> levelRankSize = omniPipeSliceParam.levelRankSize;
    std::vector<u64> scratchSizexyz = CalcScatterScratchSize(
        (u64*)xSDataSize[topo.maxDataPieceId], (u64*)ySDataSize[topo.maxDataPieceId], zSDataSize[topo.maxDataPieceId],
        levelRankSize, state.zCornerStep, state.outerStepNum, state.innerStepNum, MAX_STEP_NUM,
        omniPipeSliceParam.levelAlgType, omniPipeSliceParam.engine, topo.xB, topo.yB);
    std::vector<std::vector<u64>> xyzDataSizeStep = CalScatterDataSizeStep(
        (u64*)xSDataSize[topo.maxDataPieceId], (u64*)ySDataSize[topo.maxDataPieceId], zSDataSize[topo.maxDataPieceId],
        levelRankSize, state.zCornerStep, state.outerStepNum, state.innerStepNum, MAX_STEP_NUM, topo.xB, topo.yB);

    xCclBufferBaseOff = 0;
    yCclBufferBaseOff = xCclBufferBaseOff + scratchSizexyz[OmniPipeLevel::OMNIPIPE_LEVEL0];
    zCclBufferBaseOff = yCclBufferBaseOff + scratchSizexyz[OmniPipeLevel::OMNIPIPE_LEVEL1];

    HCCL_DEBUG(
        "zCornerStep[%llu] outerStepNum[%llu] xyCornerStep[%llu] xInCornerStep[%llu] yInCornerStep[%llu] "
        "innerStepNum[%llu]",
        state.zCornerStep, state.outerStepNum, state.xyCornerStep, state.xInCornerStep, state.yInCornerStep,
        state.innerStepNum);
    PushScatterZDiagSteps(
        dataSliceLevelz, zSDataSize, zSOffset, omniPipeSplitSliceInfoListPerLoop, omniPipeSplitSliceInfoListTotal,
        topo.dataTypeSize, topo.maxDataPieceId, topo.xRankSize, topo.yRankSize, topo.zRankSize, topo.xAxis, topo.yAxis,
        zCclBufferBaseOff, state.zCornerStep, xyzDataSizeStep);
    PushScatterZSameAxisSteps(
        dataSliceLevelz, zSDataSize, zSOffset, omniPipeSplitSliceInfoListPerLoop, omniPipeSplitSliceInfoListTotal,
        topo.dataTypeSize, topo.maxDataPieceId, topo.xRankSize, topo.yRankSize, topo.zRankSize, topo.xAxis, topo.yAxis,
        zCclBufferBaseOff, state.zCornerStep, state.outerStepNum, xyzDataSizeStep);
}

// 计算scatter omnipipe slice info的主函数
OmniPipeSliceInfo CalcScatterOmniPipeSliceInfo(OmniPipeSliceParam& omniPipeSliceParam, uint32_t root)
{
    ScatterTopoInfo topo = InitScatterTopoInfo(omniPipeSliceParam, root);
    if (topo.rankSize > MAX_RANK_SIZE) {
        HCCL_ERROR("rankSize[%d] is larger than MAX_RANK_SIZE[%d]", topo.rankSize, MAX_RANK_SIZE);
        return {};
    }
    std::vector<OmniPipeSplitSliceInfo> omniPipeSplitSliceInfoListPerLoop
        = OmniPipeSplitSliceInfoListAssign(omniPipeSliceParam.dataSizePerLoop, topo.rankSize, topo.dataTypeSize);
    std::vector<OmniPipeSplitSliceInfo> omniPipeSplitSliceInfoListTotal
        = OmniPipeSplitSliceInfoListAssign(omniPipeSliceParam.dataWholeSize, topo.rankSize, topo.dataTypeSize);

    u64 zSDataSize[topo.rankSize][MAX_STEP_NUM];
    u64 xySDataSize[topo.rankSize][MAX_STEP_NUM];
    u64 xSDataSize[topo.rankSize][MAX_STEP_NUM][MAX_STEP_NUM];
    u64 ySDataSize[topo.rankSize][MAX_STEP_NUM][MAX_STEP_NUM];
    u64 zSOffset[topo.rankSize][MAX_STEP_NUM];
    u64 xSOffset[topo.rankSize][MAX_STEP_NUM][MAX_STEP_NUM];
    u64 ySOffset[topo.rankSize][MAX_STEP_NUM][MAX_STEP_NUM];
    u64 xySOffset[topo.rankSize][MAX_STEP_NUM];
    ZeroInitScatterDataArrays(
        topo.rankSize, zSDataSize, xySDataSize, xSDataSize, ySDataSize, zSOffset, xSOffset, ySOffset, xySOffset);
    u64 xCclBufferBaseOff = 0;
    u64 yCclBufferBaseOff = 0;
    u64 zCclBufferBaseOff = 0;
    ScatterStepState state;
    InitScatterStepFlags(state, topo);
    u64 finStepMark = 2;

    CalcScatterAllRankDataSize(
        topo, state, root, finStepMark, zSDataSize, xySDataSize, xSDataSize, ySDataSize, zSOffset, xSOffset, ySOffset,
        xySOffset, omniPipeSplitSliceInfoListPerLoop);
    std::vector<StepSliceInfo> dataSliceLevelz;
    PrepareScatterBuffersAndPushZ(
        dataSliceLevelz, xCclBufferBaseOff, yCclBufferBaseOff, zCclBufferBaseOff, topo, state, xSDataSize, ySDataSize,
        zSDataSize, zSOffset, omniPipeSplitSliceInfoListPerLoop, omniPipeSplitSliceInfoListTotal, omniPipeSliceParam);

    std::vector<StepSliceInfo> dataSliceLevelx;
    PushScatterXAllSteps(
        dataSliceLevelx, topo, state, xSDataSize, ySDataSize, xySOffset, xSOffset, ySOffset,
        omniPipeSplitSliceInfoListPerLoop, omniPipeSplitSliceInfoListTotal, root, xCclBufferBaseOff, yCclBufferBaseOff,
        omniPipeSliceParam.dataSizePerLoop);
    std::vector<StepSliceInfo> dataSliceLevely;
    PushScatterYAllSteps(
        dataSliceLevely, topo, state, xSDataSize, ySDataSize, xySOffset, xSOffset, ySOffset,
        omniPipeSplitSliceInfoListPerLoop, omniPipeSplitSliceInfoListTotal, root, xCclBufferBaseOff, yCclBufferBaseOff,
        omniPipeSliceParam.dataSizePerLoop);

    struct OmniPipeSliceInfo dataSliceInfoxyz;
    dataSliceInfoxyz.dataSliceLevel2 = dataSliceLevelz;
    dataSliceInfoxyz.dataSliceLevel0 = dataSliceLevelx;
    dataSliceInfoxyz.dataSliceLevel1 = dataSliceLevely;

    return dataSliceInfoxyz;
}

std::vector<u64> OmniPipeSplitScatterData(u64 rankSize, u64 count, u64 dataTypeSize, u64 root)
{
    (void)dataTypeSize;
    if (rankSize == 0 || root >= rankSize) {
        HCCL_ERROR("[OmniPipeSplitScatterData] invalid rankSize[%llu] or root[%llu]", rankSize, root);
        return {};
    }

    std::vector<u64> omniPipeSplitSliceInfoList(rankSize, 0);
    const u64 sliceCount = RoundUp(count, rankSize);
    u64 remainingCount = count;

    // Allocate full slices beginning at root and then walk ranks cyclically.
    for (u64 order = 0; order < rankSize && remainingCount > 0; ++order) {
        const u64 rankIdx = (root + order) % rankSize;
        const u64 curSliceCount = std::min(sliceCount, remainingCount);
        omniPipeSplitSliceInfoList[rankIdx] = curSliceCount;
        remainingCount -= curSliceCount;
    }
    return omniPipeSplitSliceInfoList;
}

} // namespace ops_hccl
