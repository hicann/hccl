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
    if (xStepP2pDataSize == nullptr || yStepP2pDataSize == nullptr) {
        return;
    }
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

// 把一步的6个字段推入 stepSliceInfo
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

namespace {
    // 转调 PushStepFields
    void PushPieces(StepSliceInfo& s, const ScatterPieceVecs& p, u64 inStride, u64 outStride)
    {
        PushStepFields(s, p.sz, p.cnt, p.in, p.out, inStride, outStride);
    }

    // 统一 StepSliceInfo + BuffInfo 初始化
    StepSliceInfo MakeStepSliceInfo(u64 cclBufferBaseOff)
    {
        StepSliceInfo s;
        BuffInfo bi;
        BuffInfoAssign(bi, 0, 0, cclBufferBaseOff);
        s.buffInfo = bi;
        return s;
    }

    // 根据是否与root同z轴，选择 xySOffset[root][osn-1] 或 xySOffset[root][osn]
    u64 GetXyOffset(const ScatterTopoInfo& topo, u64 xySOffset[], u64 osn)
    {
        bool sameZAxis = (topo.rootz != topo.zAxis);
        return sameZAxis ? xySOffset[osn - 1] : xySOffset[osn];
    }
} // namespace

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

// ScatterPieceVecs 重载：调用处从 4-vector 参数压到 1 个 pieces
void PushRootOrZeros(StepSliceInfo& s, const ScatterPieceVecs& p, u64 peerIdx, u64 peerRoot, u64 outStride)
{
    PushRootOrZeros(s, p.sz, p.cnt, p.in, p.out, peerIdx, peerRoot, outStride);
}

// ScatterPieceVecs 重载：调用处从 4-vector 参数压到 1 个 pieces
void CalcAndPushPiece(
    u64 pieceId, u64 xyBaseOffset, u64 sDataSize, const std::vector<OmniPipeSplitSliceInfo>& perLoop,
    const std::vector<OmniPipeSplitSliceInfo>& total, u64 dataTypeSize, ScatterPieceVecs& p)
{
    CalcAndPushPiece(pieceId, xyBaseOffset, sDataSize, perLoop, total, dataTypeSize, p.sz, p.cnt, p.in, p.out);
}

static void PushScatterZDiagStepsImpl(
    std::vector<StepSliceInfo>& dataSliceLevelz, u64 zSDataSize[][MAX_STEP_NUM_SC], u64 zSOffset[][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    const ScatterTopoInfo& topo, u64 zCclBufferBaseOff, u64 stepUpperBound, uint32_t root)
{
    for (u64 osn = 0; osn < stepUpperBound; osn++) {
        StepSliceInfo stepSliceInfotmp = MakeStepSliceInfo(zCclBufferBaseOff);
        u64 xRankSize = topo.xRankSize;
        u64 yRankSize = topo.yRankSize;
        u64 zRankSize = topo.zRankSize;
        ScatterPieceVecs pieces;
        for (u64 oneDid = 0; oneDid < zRankSize; oneDid++) {
            if (oneDid == topo.rootz) {
                continue;
            }
            for (u64 cornerDataSlice = 0; cornerDataSlice < xRankSize * yRankSize; cornerDataSlice++) {
                // 算一下从z轴看 root的跨平面的对角节点，cornerDataSlice是xy平面2D索引，跳过root所在xy位置
                u64 currentDataSliceId = oneDid * xRankSize * yRankSize + cornerDataSlice;
                if (cornerDataSlice != topo.rooty * xRankSize + topo.rootx) {
                    CalcAndPushPiece(
                        currentDataSliceId, zSOffset[root][osn], zSDataSize[root][osn], perLoop, total,
                        topo.dataTypeSize, pieces);
                }
            }
        }
        for (u64 oneDid = 0; oneDid < xRankSize * yRankSize; oneDid++) {
            PushRootOrZeros(stepSliceInfotmp, pieces, oneDid, topo.rooty * xRankSize + topo.rootx, 0);
        }
        dataSliceLevelz.insert(dataSliceLevelz.end(), stepSliceInfotmp);
    }
}

void PushScatterZDiagSteps(
    std::vector<StepSliceInfo>& dataSliceLevelz, u64 zSDataSize[][MAX_STEP_NUM_SC], u64 zSOffset[][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    const ScatterTopoInfo& topo, u64 zCclBufferBaseOff, u64 zCornerStep, uint32_t root)
{
    HCCL_DEBUG("[PushScatterZDiagSteps] start push scatter z diag steps");
    PushScatterZDiagStepsImpl(
        dataSliceLevelz, zSDataSize, zSOffset, perLoop, total, topo, zCclBufferBaseOff, zCornerStep, root);
}

void PushScatterZDiagStepsZgXY(
    std::vector<StepSliceInfo>& dataSliceLevelz, u64 zSDataSize[][MAX_STEP_NUM_SC], u64 zSOffset[][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    const ScatterTopoInfo& topo, u64 zCclBufferBaseOff, u64 zCornerStep, uint32_t root)
{
    HCCL_DEBUG(
        "[PushScatterZDiagStepsZgXY] start push scatter z diag steps when z bandwidth greater than xy bandwidth");
    PushScatterZDiagStepsImpl(
        dataSliceLevelz, zSDataSize, zSOffset, perLoop, total, topo, zCclBufferBaseOff, zCornerStep, root);
}

// 收集z轴同轴段root对角piece：遍历非rootz的z轴rank，计算每片size/count/inputOffset/outputOffset并push
static void PushScatterZSameAxisDiagPieces(
    u64 osn, u64 zSDataSize[][MAX_STEP_NUM_SC], u64 zSOffset[][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    const ScatterTopoInfo& topo, uint32_t root, ScatterPieceVecs& pieces)
{
    u64 xRankSize = topo.xRankSize;
    u64 yRankSize = topo.yRankSize;
    u64 zRankSize = topo.zRankSize;
    for (u64 oneDid = 0; oneDid < zRankSize; oneDid++) {
        if (oneDid == topo.rootz) {
            continue;
        }
        u64 pieceId = oneDid * xRankSize * yRankSize + topo.rooty * xRankSize + topo.rootx;
        CalcAndPushPiece(
            pieceId, zSOffset[root][osn], zSDataSize[root][osn], perLoop, total, topo.dataTypeSize, pieces);
    }
}

// 计算z轴同轴段转发piece的offset：根据outerStepNum和osn位置选择offset计算分支
static void CalcScatterZSameAxisFwdOffset(
    u64 osn, u64 outerStepNum, u64 zSDataSize[][MAX_STEP_NUM_SC], u64 xySDataSize[][MAX_STEP_NUM_SC],
    u64 xySOffset[][MAX_STEP_NUM_SC], const std::vector<OmniPipeSplitSliceInfo>& perLoop, uint32_t root, u64 pieceId,
    u64& sliceSizeOnePiece, u64& inputPieceIdOffset, u64& outputPieceIdOffset)
{
    if (outerStepNum == 2) {
        inputPieceIdOffset = sliceOffsetCut(xySOffset[root][osn - 1], perLoop[pieceId].size) + perLoop[pieceId].offset;
        outputPieceIdOffset = sliceOffsetCut(xySOffset[root][osn - 1], perLoop[pieceId].size) + perLoop[pieceId].offset;
        return;
    }
    if (osn == outerStepNum - 1) {
        u64 xyBase = xySOffset[root][osn - 2];
        u64 extra = (zSDataSize[root][osn - 1] < xySDataSize[root][osn - 2]) ? zSDataSize[root][osn - 1] :
                                                                               xySDataSize[root][osn - 2];
        inputPieceIdOffset = sliceOffsetCut(xyBase + extra, perLoop[pieceId].size) + perLoop[pieceId].offset;
        outputPieceIdOffset = sliceOffsetCut(xyBase + extra, perLoop[pieceId].size) + perLoop[pieceId].offset;
        return;
    }
    if (sliceSizeOnePiece > xySDataSize[root][osn - 1]) {
        sliceSizeOnePiece = xySDataSize[root][osn - 1];
    }
    inputPieceIdOffset = sliceOffsetCut(xySOffset[root][osn - 1], perLoop[pieceId].size) + perLoop[pieceId].offset;
    outputPieceIdOffset = sliceOffsetCut(xySOffset[root][osn - 1], perLoop[pieceId].size) + perLoop[pieceId].offset;
}

// 计算z轴同轴段非rootz节点的转发piece：根据outerStepNum和osn位置选择offset计算分支
// 注意：oneDid 是 xy 平面 2D 索引（范围 [0, xRankSize*yRankSize)），pieceId 不应再叠加 rooty*xRankSize
static void CalcScatterZSameAxisFwdPieces(
    u64 osn, u64 outerStepNum, u64 zSDataSize[][MAX_STEP_NUM_SC], u64 xySDataSize[][MAX_STEP_NUM_SC],
    u64 zSOffset[][MAX_STEP_NUM_SC], u64 xySOffset[][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const ScatterTopoInfo& topo, uint32_t root, u64 oneDid,
    ScatterPieceVecs& pieces)
{
    u64 xRankSize = topo.xRankSize;
    u64 yRankSize = topo.yRankSize;
    u64 zRankSize = topo.zRankSize;
    for (u64 sameAxisDataSlice = 0; sameAxisDataSlice < zRankSize; sameAxisDataSlice++) {
        if (sameAxisDataSlice == topo.rootz) {
            continue;
        }
        u64 pieceId = sameAxisDataSlice * xRankSize * yRankSize + oneDid;
        u64 sliceSizeOnePiece = DataSliceCut(zSDataSize[root][osn], zSOffset[root][osn], perLoop[pieceId].size);
        u64 inputPieceIdOffset = 0;
        u64 outputPieceIdOffset = 0;
        CalcScatterZSameAxisFwdOffset(
            osn, outerStepNum, zSDataSize, xySDataSize, xySOffset, perLoop, root, pieceId, sliceSizeOnePiece,
            inputPieceIdOffset, outputPieceIdOffset);
        if (inputPieceIdOffset + sliceSizeOnePiece > perLoop[pieceId].offset + perLoop[pieceId].size) {
            sliceSizeOnePiece = perLoop[pieceId].offset + perLoop[pieceId].size - inputPieceIdOffset;
        }
        pieces.cnt.push_back(sliceSizeOnePiece / topo.dataTypeSize);
        pieces.in.push_back(inputPieceIdOffset);
        pieces.out.push_back(outputPieceIdOffset);
        pieces.sz.push_back(sliceSizeOnePiece);
    }
}

void PushScatterZSameAxisSteps(
    std::vector<StepSliceInfo>& dataSliceLevelz, u64 zSDataSize[][MAX_STEP_NUM_SC], u64 xySDataSize[][MAX_STEP_NUM_SC],
    u64 zSOffset[][MAX_STEP_NUM_SC], u64 xySOffset[][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    const ScatterTopoInfo& topo, u64 zCclBufferBaseOff, u64 zCornerStep, u64 outerStepNum, uint32_t root)
{
    HCCL_DEBUG("[PushScatterZSameAxisSteps] start push scatter z same axis steps");
    for (u64 osn = zCornerStep; osn < outerStepNum; osn++) {
        StepSliceInfo stepSliceInfotmp = MakeStepSliceInfo(zCclBufferBaseOff);
        u64 xRankSize = topo.xRankSize;
        u64 yRankSize = topo.yRankSize;
        ScatterPieceVecs pieces;
        PushScatterZSameAxisDiagPieces(osn, zSDataSize, zSOffset, perLoop, total, topo, root, pieces);
        for (u64 oneDid = 0; oneDid < xRankSize * yRankSize; oneDid++) {
            if (oneDid == topo.rooty * xRankSize + topo.rootx) {
                PushPieces(stepSliceInfotmp, pieces, 0, 0);
            } else {
                ScatterPieceVecs pieces1;
                CalcScatterZSameAxisFwdPieces(
                    osn, outerStepNum, zSDataSize, xySDataSize, zSOffset, xySOffset, perLoop, topo, root, oneDid,
                    pieces1);
                PushPieces(stepSliceInfotmp, pieces1, 0, 0);
            }
        }
        dataSliceLevelz.insert(dataSliceLevelz.end(), stepSliceInfotmp);
    }
}

// 计算z轴同轴段(z带宽>xy)非rootz节点的转发piece：sliceSize和offset都从xy轴取
// 注意：oneDid 是 xy 平面 2D 索引（范围 [0, xRankSize*yRankSize)），pieceId 不应再叠加 rooty*xRankSize
static void CalcScatterZSameAxisZgXYFwdPieces(
    u64 xySDataSize[][MAX_STEP_NUM_SC], u64 xySOffset[][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const ScatterTopoInfo& topo, uint32_t root, u64 oneDid,
    ScatterPieceVecs& pieces)
{
    u64 yRankSize = topo.yRankSize;
    u64 xRankSize = topo.xRankSize;
    u64 zRankSize = topo.zRankSize;
    for (u64 sameAxisDataSlice = 0; sameAxisDataSlice < zRankSize; sameAxisDataSlice++) {
        if (sameAxisDataSlice == topo.rootz) {
            continue;
        }
        u64 pieceId = sameAxisDataSlice * xRankSize * yRankSize + oneDid;
        u64 sliceSizeOnePiece = DataSliceCut(xySDataSize[root][0], xySOffset[root][0], perLoop[pieceId].size);
        u64 inputPieceIdOffset = sliceOffsetCut(xySOffset[root][0], perLoop[pieceId].size) + perLoop[pieceId].offset;
        u64 outputPieceIdOffset = sliceOffsetCut(xySOffset[root][0], perLoop[pieceId].size) + perLoop[pieceId].offset;
        pieces.cnt.push_back(sliceSizeOnePiece / topo.dataTypeSize);
        pieces.sz.push_back(sliceSizeOnePiece);
        pieces.in.push_back(inputPieceIdOffset);
        pieces.out.push_back(outputPieceIdOffset);
    }
}

void PushScatterZSameAxisStepsZgXY(
    std::vector<StepSliceInfo>& dataSliceLevelz, u64 zSDataSize[][MAX_STEP_NUM_SC], u64 xySDataSize[][MAX_STEP_NUM_SC],
    u64 zSOffset[][MAX_STEP_NUM_SC], u64 xySOffset[][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    const ScatterTopoInfo& topo, u64 zCclBufferBaseOff, u64 zCornerStep, u64 outerStepNum, uint32_t root)
{
    HCCL_DEBUG("[PushScatterZSameAxisStepsZgXY] start push scatter z same axis steps when z bandwidth greater than xy "
               "bandwidth");
    for (u64 osn = zCornerStep; osn < outerStepNum; osn++) {
        StepSliceInfo stepSliceInfotmp = MakeStepSliceInfo(zCclBufferBaseOff);
        ScatterPieceVecs pieces;
        u64 xRankSize = topo.xRankSize;
        u64 yRankSize = topo.yRankSize;
        PushScatterZSameAxisDiagPieces(osn, zSDataSize, zSOffset, perLoop, total, topo, root, pieces);
        for (u64 oneDid = 0; oneDid < xRankSize * yRankSize; oneDid++) {
            if (oneDid == topo.rooty * xRankSize + topo.rootx) {
                PushPieces(stepSliceInfotmp, pieces, 0, 0);
            } else {
                ScatterPieceVecs pieces1;
                CalcScatterZSameAxisZgXYFwdPieces(xySDataSize, xySOffset, perLoop, topo, root, oneDid, pieces1);
                PushPieces(stepSliceInfotmp, pieces1, 0, 0);
            }
        }
        dataSliceLevelz.insert(dataSliceLevelz.end(), stepSliceInfotmp);
    }
}

void PushScatterXInnerCornerOneDiag(
    ScatterPieceVecs& pieces, u64 osn, u64 isn, u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 xySOffset[][MAX_STEP_NUM_SC], u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    const ScatterTopoInfo& topo, const std::vector<u64>& dataSizePerLoop, u64 oneDid, uint32_t root)
{
    u64 xRankSize = topo.xRankSize;
    u64 yRankSize = topo.yRankSize;
    u64 zRankSize = topo.zRankSize;
    for (u64 outSliceNum = 0; outSliceNum < zRankSize; outSliceNum++) {
        if (outSliceNum != topo.rootz) {
            for (u64 cornerDataSlice = 0; cornerDataSlice < yRankSize; cornerDataSlice++) {
                u64 currentInnerStepDataSliceId
                    = outSliceNum * xRankSize * yRankSize + cornerDataSlice * xRankSize + oneDid;
                if (cornerDataSlice != topo.rooty && yRankSize > 1) {
                    u64 pieceId = currentInnerStepDataSliceId;
                    u64 sliceSizeOnePiece = DataSliceCut(
                        xSDataSize[root][osn][isn], xySOffset[root][osn] + xSOffset[root][osn][isn],
                        perLoop[pieceId].size);
                    u64 inputPieceIdOffset
                        = sliceOffsetCut(xySOffset[root][osn] + xSOffset[root][osn][isn], perLoop[pieceId].size)
                          + total[pieceId].offset;
                    u64 outputPieceIdOffset
                        = sliceOffsetCut(xySOffset[root][osn] + xSOffset[root][osn][isn], perLoop[pieceId].size)
                          + perLoop[pieceId].offset;
                    pieces.sz.push_back(sliceSizeOnePiece);
                    pieces.cnt.push_back(sliceSizeOnePiece / topo.dataTypeSize);
                    pieces.in.push_back(inputPieceIdOffset);
                    pieces.out.push_back(outputPieceIdOffset);
                }
            }
        }
    }
}

void PushScatterXInnerCornerOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevelx, u64 osn, u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 xySOffset[][MAX_STEP_NUM_SC], u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    const ScatterTopoInfo& topo, const std::vector<u64>& dataSizePerLoop, u64 xCclBufferBaseOff, u64 xInCornerStep,
    uint32_t root)
{
    for (u64 isn = 0; isn < xInCornerStep; isn++) {
        StepSliceInfo stepSliceInfotmp = MakeStepSliceInfo(xCclBufferBaseOff);
        ScatterPieceVecs pieces;
        for (u64 oneDid = 0; oneDid < topo.xRankSize; oneDid++) {
            if (oneDid == topo.rootx) {
                continue;
            }
            PushScatterXInnerCornerOneDiag(
                pieces, osn, isn, xSDataSize, xySOffset, xSOffset, perLoop, total, topo, dataSizePerLoop, oneDid, root);
        }
        for (u64 oneDid = 0; oneDid < topo.yRankSize; oneDid++) {
            PushRootOrZeros(stepSliceInfotmp, pieces, oneDid, topo.rooty, 0);
        }
        dataSliceLevelx.insert(dataSliceLevelx.end(), stepSliceInfotmp);
    }
}

static void CalcScatterXInnerDiagPieces(
    u64 osn, u64 isn, u64 xySDataSize[][MAX_STEP_NUM_SC], u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 xySOffset[][MAX_STEP_NUM_SC], u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    const ScatterTopoInfo& topo, uint32_t root, ScatterPieceVecs& pieces)
{
    u64 xRankSize = topo.xRankSize;
    u64 yRankSize = topo.yRankSize;
    u64 zRankSize = topo.zRankSize;
    for (u64 oneDid = 0; oneDid < xRankSize; oneDid++) {
        if (oneDid == topo.rootx) {
            continue;
        }
        for (u64 outSliceNum = 0; outSliceNum < zRankSize; outSliceNum++) {
            u64 currentDataSliceId = outSliceNum * xRankSize * yRankSize + topo.rooty * xRankSize + oneDid;
            if (outSliceNum != topo.rootz) {
                u64 pieceId = currentDataSliceId;
                u64 sliceSizeOnePiece = DataSliceCut(
                    xSDataSize[root][osn][isn], xySOffset[root][osn] + xSOffset[root][osn][isn], perLoop[pieceId].size);
                u64 inputPieceIdOffset
                    = sliceOffsetCut(xySOffset[root][osn] + xSOffset[root][osn][isn], perLoop[pieceId].size)
                      + total[pieceId].offset;
                u64 outputPieceIdOffset
                    = sliceOffsetCut(xySOffset[root][osn] + xSOffset[root][osn][isn], perLoop[pieceId].size)
                      + perLoop[pieceId].offset;
                pieces.cnt.push_back(sliceSizeOnePiece / topo.dataTypeSize);
                pieces.sz.push_back(sliceSizeOnePiece);
                pieces.in.push_back(inputPieceIdOffset);
                pieces.out.push_back(outputPieceIdOffset);
            }
        }
    }
}

static void CalcScatterXInnerSameAxisPieceOffset(
    u64 osn, u64 isn, u64 innerStepNum, u64 xySDataSize[][MAX_STEP_NUM_SC],
    u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 xySOffset[][MAX_STEP_NUM_SC], u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, uint32_t root, u64 pieceId, u64& sliceSizeOnePiece,
    u64& inputPieceIdOffset, u64& outputPieceIdOffset)
{
    u64 xyBaseOff = xySOffset[root][osn];
    if (innerStepNum == 2) {
        inputPieceIdOffset
            = sliceOffsetCut(xyBaseOff + ySOffset[root][osn][isn - 1], perLoop[pieceId].size) + perLoop[pieceId].offset;
        outputPieceIdOffset
            = sliceOffsetCut(xyBaseOff + ySOffset[root][osn][isn - 1], perLoop[pieceId].size) + perLoop[pieceId].offset;
    } else {
        if (isn == innerStepNum - 1) {
            if (xSDataSize[root][osn][isn - 1] < ySDataSize[root][osn][isn - 2]) {
                inputPieceIdOffset = sliceOffsetCut(
                                         xyBaseOff + ySOffset[root][osn][isn - 2] + xSDataSize[root][osn][isn - 1],
                                         perLoop[pieceId].size)
                                     + perLoop[pieceId].offset;
                outputPieceIdOffset = sliceOffsetCut(
                                          xyBaseOff + ySOffset[root][osn][isn - 2] + xSDataSize[root][osn][isn - 1],
                                          perLoop[pieceId].size)
                                      + perLoop[pieceId].offset;
            } else {
                inputPieceIdOffset = sliceOffsetCut(
                                         xyBaseOff + ySOffset[root][osn][isn - 2] + ySDataSize[root][osn][isn - 2],
                                         perLoop[pieceId].size)
                                     + perLoop[pieceId].offset;
                outputPieceIdOffset = sliceOffsetCut(
                                          xyBaseOff + ySOffset[root][osn][isn - 2] + ySDataSize[root][osn][isn - 2],
                                          perLoop[pieceId].size)
                                      + perLoop[pieceId].offset;
            }
        } else {
            if (sliceSizeOnePiece > ySDataSize[root][osn][isn - 1]) {
                sliceSizeOnePiece = ySDataSize[root][osn][isn - 1];
            }
            inputPieceIdOffset = sliceOffsetCut(xyBaseOff + ySOffset[root][osn][isn - 1], perLoop[pieceId].size)
                                 + perLoop[pieceId].offset;
            outputPieceIdOffset = sliceOffsetCut(xyBaseOff + ySOffset[root][osn][isn - 1], perLoop[pieceId].size)
                                  + perLoop[pieceId].offset;
        }
    }
    if (inputPieceIdOffset + sliceSizeOnePiece > perLoop[pieceId].offset + xySDataSize[root][osn] + xyBaseOff) {
        sliceSizeOnePiece = perLoop[pieceId].offset + xySDataSize[root][osn] + xyBaseOff - inputPieceIdOffset;
    }
}

// 计算x轴内层同轴段非rooty节点的转发piece：遍历非rootx的x轴rank和非rootz的z轴rank
static void CalcScatterXInnerSameAxisFwdPieces(
    u64 osn, u64 isn, u64 innerStepNum, u64 xySDataSize[][MAX_STEP_NUM_SC],
    u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 xySOffset[][MAX_STEP_NUM_SC], u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], const std::vector<OmniPipeSplitSliceInfo>& perLoop,
    const ScatterTopoInfo& topo, uint32_t root, u64 oneDid, ScatterPieceVecs& pieces)
{
    u64 zRankSize = topo.zRankSize;
    u64 xRankSize = topo.xRankSize;
    u64 yRankSize = topo.yRankSize;
    for (u64 sameAxisDataSlice = 0; sameAxisDataSlice < xRankSize; sameAxisDataSlice++) {
        if (sameAxisDataSlice == topo.rootx) {
            continue;
        }
        for (u64 outSliceNum = 0; outSliceNum < zRankSize; outSliceNum++) {
            if (outSliceNum == topo.rootz) {
                continue;
            }
            u64 pieceId = outSliceNum * xRankSize * yRankSize + oneDid * xRankSize + sameAxisDataSlice;
            u64 sliceSizeOnePiece = DataSliceCut(
                xSDataSize[root][osn][isn], xySOffset[root][osn] + xSOffset[root][osn][isn], perLoop[pieceId].size);
            u64 inputPieceIdOffset = 0;
            u64 outputPieceIdOffset = 0;
            CalcScatterXInnerSameAxisPieceOffset(
                osn, isn, innerStepNum, xySDataSize, xSDataSize, ySDataSize, xySOffset, ySOffset, perLoop, root,
                pieceId, sliceSizeOnePiece, inputPieceIdOffset, outputPieceIdOffset);
            pieces.sz.push_back(sliceSizeOnePiece);
            pieces.cnt.push_back(sliceSizeOnePiece / topo.dataTypeSize);
            pieces.in.push_back(inputPieceIdOffset);
            pieces.out.push_back(outputPieceIdOffset);
        }
    }
}

void PushScatterXInnerSameAxisOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevelx, u64 osn, u64 xySDataSize[][MAX_STEP_NUM_SC],
    u64 zSDataSize[][MAX_STEP_NUM_SC], u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 xySOffset[][MAX_STEP_NUM_SC],
    u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    const ScatterTopoInfo& topo, const std::vector<u64>& dataSizePerLoop, u64 xCclBufferBaseOff, u64 xInCornerStep,
    u64 innerStepNum, uint32_t root)
{
    for (u64 isn = xInCornerStep; isn < innerStepNum; isn++) {
        StepSliceInfo stepSliceInfotmp = MakeStepSliceInfo(xCclBufferBaseOff);
        ScatterPieceVecs pieces;
        CalcScatterXInnerDiagPieces(
            osn, isn, xySDataSize, xSDataSize, xySOffset, xSOffset, perLoop, total, topo, root, pieces);
        for (u64 oneDid = 0; oneDid < topo.yRankSize; oneDid++) {
            if (oneDid == topo.rooty) {
                PushPieces(stepSliceInfotmp, pieces, 0, 0);
            } else {
                ScatterPieceVecs pieces1;
                CalcScatterXInnerSameAxisFwdPieces(
                    osn, isn, innerStepNum, xySDataSize, xSDataSize, ySDataSize, xySOffset, xSOffset, ySOffset, perLoop,
                    topo, root, oneDid, pieces1);
                PushPieces(stepSliceInfotmp, pieces1, 0, 0);
            }
        }
        dataSliceLevelx.insert(dataSliceLevelx.end(), stepSliceInfotmp);
    }
}

static void CalcScatterXOverSameAxisPieces(
    u64 osn, u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 xySOffset[][MAX_STEP_NUM_SC],
    u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const ScatterTopoInfo& topo, uint32_t root, u64 one,
    ScatterPieceVecs& pieces)
{
    u64 xRankSize = topo.xRankSize;
    u64 yRankSize = topo.yRankSize;
    u64 zRankSize = topo.zRankSize;
    for (u64 sameAxisDataSlice = 0; sameAxisDataSlice < xRankSize; sameAxisDataSlice++) {
        if (sameAxisDataSlice == topo.rootx) {
            continue;
        }
        for (u64 outSliceNum = 0; outSliceNum < zRankSize; outSliceNum++) {
            if (outSliceNum == topo.rootz) {
                continue;
            }
            u64 pieceId = outSliceNum * xRankSize * yRankSize + one * xRankSize + sameAxisDataSlice;
            u64 sliceSizeOnePiece = DataSliceCut(
                ySDataSize[root][osn][0], xySOffset[root][osn] + xSOffset[root][osn][0], perLoop[pieceId].size);
            u64 inputPieceIdOffset
                = sliceOffsetCut(xySOffset[root][osn] + ySOffset[root][osn][0], perLoop[pieceId].size)
                  + perLoop[pieceId].offset;
            u64 outputPieceIdOffset
                = sliceOffsetCut(xySOffset[root][osn] + ySOffset[root][osn][0], perLoop[pieceId].size)
                  + perLoop[pieceId].offset;
            pieces.cnt.push_back(sliceSizeOnePiece / topo.dataTypeSize);
            pieces.sz.push_back(sliceSizeOnePiece);
            pieces.out.push_back(outputPieceIdOffset);
            pieces.in.push_back(inputPieceIdOffset);
        }
    }
}

void PushScatterXOverSameAxisOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevelx, u64 osn, u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 xySOffset[][MAX_STEP_NUM_SC],
    u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    const ScatterTopoInfo& topo, const std::vector<u64>& dataSizePerLoop, u64 xCclBufferBaseOff, u64 xInCornerStep,
    u64 innerStepNum, uint32_t root)
{
    HCCL_DEBUG("[PushScatterXOverSameAxisOneOsn] start push scatter x over same axis one osn");
    for (u64 isn = xInCornerStep + 1; isn < innerStepNum; isn++) {
        StepSliceInfo stepSliceInfotmp = MakeStepSliceInfo(xCclBufferBaseOff);
        u64 xRankSize = topo.xRankSize;
        u64 yRankSize = topo.yRankSize;
        u64 zRankSize = topo.zRankSize;
        ScatterPieceVecs pieces;
        for (u64 one = 0; one < xRankSize; one++) {
            if (one == topo.rootx)
                continue;
            for (u64 outSliceNum = 0; outSliceNum < zRankSize; outSliceNum++) {
                if (outSliceNum == topo.rootz) {
                    continue;
                }
                u64 pieceId = outSliceNum * xRankSize * yRankSize + topo.rooty * xRankSize + one;
                CalcAndPushPiece(
                    pieceId, xySOffset[root][osn] + xSOffset[root][osn][isn], xSDataSize[root][osn][isn], perLoop,
                    total, topo.dataTypeSize, pieces);
            }
        }
        for (u64 one = 0; one < yRankSize; one++) {
            if (one == topo.rooty) {
                PushPieces(stepSliceInfotmp, pieces, 0, 0);
            } else {
                ScatterPieceVecs pieces1;
                CalcScatterXOverSameAxisPieces(
                    osn, ySDataSize, xySOffset, xSOffset, ySOffset, perLoop, topo, root, one, pieces1);
                PushPieces(stepSliceInfotmp, pieces1, 0, 0);
            }
        }
        dataSliceLevelx.insert(dataSliceLevelx.end(), stepSliceInfotmp);
    }
}

// 收集x轴外层斜对角piece：遍历非rootx的x轴rank和非rooty的y轴rank，计算xyoffset并调用CalcAndPushPiece
static void PushScatterXOuterCornerDiagPieces(
    u64 osn, u64 isn, u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 xySOffset[][MAX_STEP_NUM_SC],
    u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], const std::vector<OmniPipeSplitSliceInfo>& perLoop,
    const std::vector<OmniPipeSplitSliceInfo>& total, const ScatterTopoInfo& topo, uint32_t root, u64 dataTypeSize,
    ScatterPieceVecs& pieces)
{
    u64 xRankSize = topo.xRankSize;
    u64 yRankSize = topo.yRankSize;
    u64 rootx = topo.rootx;
    u64 rooty = topo.rooty;
    for (u64 oneDid = 0; oneDid < xRankSize; oneDid++) {
        if (oneDid == rootx) {
            continue;
        }
        for (u64 cornerDataSlice = 0; cornerDataSlice < yRankSize; cornerDataSlice++) {
            if (cornerDataSlice == rooty) {
                continue;
            }
            u64 currentDataSliceId = topo.zAxis * xRankSize * yRankSize + cornerDataSlice * xRankSize + oneDid;
            u64 xyoffset = GetXyOffset(topo, xySOffset[root], osn);
            CalcAndPushPiece(
                currentDataSliceId, xyoffset + xSOffset[root][osn][isn], xSDataSize[root][osn][isn], perLoop, total,
                dataTypeSize, pieces);
        }
    }
}

// scatter x轴外层 xB<=yB
// 斜对角段（单个osn，isn∈[0,xInCornerStep)）：root发斜对角数据，root数据放index=rootx，其余x轴rank塞0
void PushScatterXOuterLECornerOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevelx, u64 osn, u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 xySOffset[][MAX_STEP_NUM_SC], u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    const ScatterTopoInfo& topo, uint32_t root, u64 xCclBufferBaseOff, u64 xInCornerStep)
{
    HCCL_DEBUG("[PushScatterXOuterLECornerOneOsn] start push scatter x outer le corner one osn");
    for (u64 isn = 0; isn < xInCornerStep; isn++) {
        StepSliceInfo stepSliceInfotmp = MakeStepSliceInfo(xCclBufferBaseOff);
        u64 xRankSize = topo.xRankSize;
        u64 yRankSize = topo.yRankSize;
        u64 dataTypeSize = topo.dataTypeSize;
        u64 rootx = topo.rootx;
        u64 rooty = topo.rooty;
        ScatterPieceVecs pieces;
        for (u64 oneDid = 0; oneDid < xRankSize; oneDid++) {
            if (oneDid == rootx) {
                continue;
            }
            PushScatterXOuterCornerDiagPieces(
                osn, isn, xSDataSize, xySOffset, xSOffset, perLoop, total, topo, root, dataTypeSize, pieces);
        }
        for (u64 oneDid = 0; oneDid < yRankSize; oneDid++) {
            PushRootOrZeros(stepSliceInfotmp, pieces, oneDid, rooty, 0);
        }
        dataSliceLevelx.insert(dataSliceLevelx.end(), stepSliceInfotmp);
    }
}

// scatter x轴外层 xB<=yB
// 同轴转发段（单个osn，isn∈[xInCornerStep,innerStepNum)）：root发同x轴数据，同y轴非root节点转发step1收到的对角数据
void CalcScatterXOuterLEOffset(
    u64& inputPieceIdOffset, u64& outputPieceIdOffset, u64& sliceSizeOnePiece,
    u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 xySOffset[][MAX_STEP_NUM_SC], u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, uint32_t root, u64 osn, u64 isn, u64 innerStepNum, u64 pieceId,
    const ScatterTopoInfo& topo)
{
    HCCL_DEBUG("[CalcScatterXOuterLEOffset] start calc scatter x outer le offset");
    u64 xyoffset = GetXyOffset(topo, xySOffset[root], osn);
    if (innerStepNum == 2) {
        inputPieceIdOffset
            = sliceOffsetCut(xyoffset + ySOffset[root][osn][isn - 1], perLoop[pieceId].size) + perLoop[pieceId].offset;
        outputPieceIdOffset
            = sliceOffsetCut(xyoffset + ySOffset[root][osn][isn - 1], perLoop[pieceId].size) + perLoop[pieceId].offset;
    } else {
        if (isn == innerStepNum - 1) {
            if (xSDataSize[root][osn][isn - 1] < ySDataSize[root][osn][isn - 2]) {
                inputPieceIdOffset = sliceOffsetCut(
                                         xyoffset + ySOffset[root][osn][isn - 2] + xSDataSize[root][osn][isn - 1],
                                         perLoop[pieceId].size)
                                     + perLoop[pieceId].offset;
                outputPieceIdOffset = sliceOffsetCut(
                                          xyoffset + ySOffset[root][osn][isn - 2] + xSDataSize[root][osn][isn - 1],
                                          perLoop[pieceId].size)
                                      + perLoop[pieceId].offset;
            } else {
                inputPieceIdOffset = sliceOffsetCut(
                                         xyoffset + ySOffset[root][osn][isn - 2] + ySDataSize[root][osn][isn - 2],
                                         perLoop[pieceId].size)
                                     + perLoop[pieceId].offset;
                outputPieceIdOffset = sliceOffsetCut(
                                          xyoffset + ySOffset[root][osn][isn - 2] + ySDataSize[root][osn][isn - 2],
                                          perLoop[pieceId].size)
                                      + perLoop[pieceId].offset;
            }
        } else {
            if (sliceSizeOnePiece > ySDataSize[root][osn][isn - 1]) {
                sliceSizeOnePiece = ySDataSize[root][osn][isn - 1];
            }
            inputPieceIdOffset = sliceOffsetCut(xyoffset + ySOffset[root][osn][isn - 1], perLoop[pieceId].size)
                                 + perLoop[pieceId].offset;
            outputPieceIdOffset = sliceOffsetCut(xyoffset + ySOffset[root][osn][isn - 1], perLoop[pieceId].size)
                                  + perLoop[pieceId].offset;
        }
    }
}

// scatter x轴外层 xB<=yB 同轴转发段：为非rooty的y轴节点构建转发piece并Push
void PushScatterXOuterLEFwdOneRank(
    StepSliceInfo& stepSliceInfotmp, u64 xySDataSize[][MAX_STEP_NUM_SC],
    u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 xySOffset[][MAX_STEP_NUM_SC], u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], const std::vector<OmniPipeSplitSliceInfo>& perLoop, uint32_t root,
    u64 osn, u64 isn, u64 innerStepNum, const ScatterTopoInfo& topo, u64 oneDid,
    const std::vector<u64>& dataSizePerLoop)
{
    HCCL_DEBUG("[PushScatterXOuterLEFwdOneRank] start push scatter x outer le fwd one rank");
    ScatterPieceVecs pieces;
    u64 rootx = topo.rootx;
    u64 xRankSize = topo.xRankSize;
    u64 yRankSize = topo.yRankSize;
    u64 zRankSize = topo.zRankSize;
    u64 dataTypeSize = topo.dataTypeSize;
    bool sameZAxis = (topo.rootz != topo.zAxis);
    for (u64 cornerDataSlice = 0; cornerDataSlice < xRankSize; cornerDataSlice++) {
        if (cornerDataSlice == rootx)
            continue;
        u64 pieceId = topo.zAxis * xRankSize * yRankSize + oneDid * xRankSize + cornerDataSlice;
        u64 xysize = sameZAxis ? xySDataSize[root][osn - 1] : xySDataSize[root][osn];
        u64 xyoffset = sameZAxis ? xySOffset[root][osn - 1] : xySOffset[root][osn];
        u64 sliceSizeOnePiece
            = DataSliceCut(xSDataSize[root][osn][isn], xyoffset + xSOffset[root][osn][isn], perLoop[pieceId].size);

        u64 inputPieceIdOffset = 0;
        u64 outputPieceIdOffset = 0;
        CalcScatterXOuterLEOffset(
            inputPieceIdOffset, outputPieceIdOffset, sliceSizeOnePiece, xSDataSize, ySDataSize, xySOffset, ySOffset,
            perLoop, root, osn, isn, innerStepNum, pieceId, topo);
        if (zRankSize <= 1) {
            if (inputPieceIdOffset + sliceSizeOnePiece > perLoop[pieceId].offset + dataSizePerLoop[root]) {
                sliceSizeOnePiece = perLoop[pieceId].offset + dataSizePerLoop[root] - inputPieceIdOffset;
            }
        } else {
            if (inputPieceIdOffset + sliceSizeOnePiece > perLoop[pieceId].offset + xysize + xyoffset) {
                sliceSizeOnePiece = perLoop[pieceId].offset + xysize + xyoffset - inputPieceIdOffset;
            }
        }

        pieces.sz.push_back(sliceSizeOnePiece);
        pieces.cnt.push_back(sliceSizeOnePiece / dataTypeSize);
        pieces.in.push_back(inputPieceIdOffset);
        pieces.out.push_back(outputPieceIdOffset);
    }
    PushPieces(stepSliceInfotmp, pieces, 0, 0);
}

void PushScatterXOuterLESameAxisOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevelx, u64 osn, u64 xySDataSize[][MAX_STEP_NUM_SC],
    u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 xySOffset[][MAX_STEP_NUM_SC], u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], const std::vector<OmniPipeSplitSliceInfo>& perLoop,
    const std::vector<OmniPipeSplitSliceInfo>& total, const ScatterTopoInfo& topo, uint32_t root, u64 xCclBufferBaseOff,
    u64 xInCornerStep, u64 innerStepNum, const std::vector<u64>& dataSizePerLoop)
{
    HCCL_DEBUG("[PushScatterXOuterLESameAxisOneOsn] start push scatter x outer le same axis one osn");
    u64 xRankSize = topo.xRankSize;
    u64 yRankSize = topo.yRankSize;
    u64 dataTypeSize = topo.dataTypeSize;
    u64 rootx = topo.rootx;
    u64 rooty = topo.rooty;
    for (u64 isn = xInCornerStep; isn < innerStepNum; isn++) {
        StepSliceInfo stepSliceInfotmp = MakeStepSliceInfo(xCclBufferBaseOff);
        ScatterPieceVecs pieces;
        for (u64 oneDid = 0; oneDid < xRankSize; oneDid++) {
            if (oneDid == rootx)
                continue;
            u64 pieceId = topo.zAxis * xRankSize * yRankSize + rooty * xRankSize + oneDid;
            u64 xyoffset = GetXyOffset(topo, xySOffset[root], osn);
            CalcAndPushPiece(
                pieceId, xyoffset + xSOffset[root][osn][isn], xSDataSize[root][osn][isn], perLoop, total, dataTypeSize,
                pieces);
        }
        for (u64 oneDid = 0; oneDid < yRankSize; oneDid++) {
            if (oneDid == rooty) {
                PushPieces(stepSliceInfotmp, pieces, 0, 0);
            } else {
                PushScatterXOuterLEFwdOneRank(
                    stepSliceInfotmp, xySDataSize, xSDataSize, ySDataSize, xySOffset, xSOffset, ySOffset, perLoop, root,
                    osn, isn, innerStepNum, topo, oneDid, dataSizePerLoop);
            }
        }
        dataSliceLevelx.insert(dataSliceLevelx.end(), stepSliceInfotmp);
    }
}

// scatter x轴外层 xB>yB
// 斜对角转发段（单个osn，isn∈[0,xInCornerStep+1)）：前两步都是转发对角数据，buffer用yCclBufferBaseOff
void PushScatterXOuterGTCornerOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevelx, u64 osn, u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 xySOffset[][MAX_STEP_NUM_SC], u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    const ScatterTopoInfo& topo, uint32_t root, u64 yCclBufferBaseOff, u64 xInCornerStep)
{
    HCCL_DEBUG("[PushScatterXOuterGTCornerOneOsn] start push scatter x outer gt corner one osn");
    u64 rootx = topo.rootx;
    u64 rooty = topo.rooty;
    u64 xRankSize = topo.xRankSize;
    u64 yRankSize = topo.yRankSize;
    u64 dataTypeSize = topo.dataTypeSize;
    for (u64 isn = 0; isn < xInCornerStep + 1; isn++) {
        StepSliceInfo stepSliceInfotmp = MakeStepSliceInfo(yCclBufferBaseOff);
        ScatterPieceVecs pieces;
        for (u64 oneDid = 0; oneDid < xRankSize; oneDid++) {
            if (oneDid == rootx)
                continue;
            PushScatterXOuterCornerDiagPieces(
                osn, isn, xSDataSize, xySOffset, xSOffset, perLoop, total, topo, root, dataTypeSize, pieces);
        }
        for (u64 one = 0; one < yRankSize; one++) {
            PushRootOrZeros(stepSliceInfotmp, pieces, one, rooty, 0);
        }
        dataSliceLevelx.insert(dataSliceLevelx.end(), stepSliceInfotmp);
    }
}

// scatter x轴外层 xB>yB 同轴转发段：为非rooty的y轴节点构建转发piece并Push
void PushScatterXOuterGTFwdOneRank(
    StepSliceInfo& stepSliceInfotmp, u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 xySOffset[][MAX_STEP_NUM_SC], u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, uint32_t root, u64 osn, const ScatterTopoInfo& topo, u64 one,
    u64 dataTypeSize)
{
    ScatterPieceVecs pieces1;
    u64 xRankSize = topo.xRankSize;
    u64 yRankSize = topo.yRankSize;
    for (u64 cornerDataSlice = 0; cornerDataSlice < xRankSize; cornerDataSlice++) {
        if (cornerDataSlice == topo.rootx)
            continue;
        u64 currentDataSliceId = topo.zAxis * xRankSize * yRankSize + one * xRankSize + cornerDataSlice;
        u64 pieceId = currentDataSliceId;
        u64 xyoffset = GetXyOffset(topo, xySOffset[root], osn);
        CalcAndPushPiece(
            pieceId, xyoffset + ySOffset[root][osn][0], ySDataSize[root][osn][0], perLoop, perLoop, dataTypeSize,
            pieces1);
    }
    PushPieces(stepSliceInfotmp, pieces1, 0, 0);
}

void PushScatterXOuterGTSameAxisOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevelx, u64 osn, u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 xySOffset[][MAX_STEP_NUM_SC],
    u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    const ScatterTopoInfo& topo, uint32_t root, u64 yCclBufferBaseOff, u64 xInCornerStep, u64 innerStepNum)
{
    HCCL_DEBUG("[PushScatterXOuterGTSameAxisOneOsn] start push scatter x outer gt same axis one osn");
    u64 dataTypeSize = topo.dataTypeSize;
    u64 rootx = topo.rootx;
    u64 rooty = topo.rooty;
    u64 xRankSize = topo.xRankSize;
    u64 yRankSize = topo.yRankSize;
    for (u64 isn = xInCornerStep + 1; isn < innerStepNum; isn++) {
        StepSliceInfo stepSliceInfotmp = MakeStepSliceInfo(yCclBufferBaseOff);
        ScatterPieceVecs pieces;
        for (u64 one = 0; one < xRankSize; one++) {
            if (one == rootx)
                continue;
            u64 pieceId = topo.zAxis * xRankSize * yRankSize + rooty * xRankSize + one;
            u64 xyoffset = GetXyOffset(topo, xySOffset[root], osn);
            CalcAndPushPiece(
                pieceId, xyoffset + xSOffset[root][osn][isn], xSDataSize[root][osn][isn], perLoop, total, dataTypeSize,
                pieces);
        }
        for (u64 one = 0; one < yRankSize; one++) {
            if (one == rooty) {
                PushPieces(stepSliceInfotmp, pieces, 0, 0);
            } else {
                PushScatterXOuterGTFwdOneRank(
                    stepSliceInfotmp, ySDataSize, xySOffset, ySOffset, perLoop, root, osn, topo, one, dataTypeSize);
            }
        }
        dataSliceLevelx.insert(dataSliceLevelx.end(), stepSliceInfotmp);
    }
}

// scatter y轴内层斜对角段（单个osn，isn∈[0,yInCornerStep)）：root和同轴线节点处理y轴斜对角通信
void PushScatterYInnerCornerOneDiag(
    ScatterPieceVecs& pieces, u64 osn, u64 isn, u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 xySOffset[][MAX_STEP_NUM_SC], u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    const ScatterTopoInfo& topo, u64 oneDid, uint32_t root)
{
    HCCL_DEBUG("[PushScatterYInnerCornerOneDiag] start push scatter y inner corner one diag");
    u64 xRankSize = topo.xRankSize;
    u64 yRankSize = topo.yRankSize;
    u64 zRankSize = topo.zRankSize;
    u64 dataTypeSize = topo.dataTypeSize;
    for (u64 outSliceNum = 0; outSliceNum < zRankSize; outSliceNum++) {
        if (outSliceNum != topo.rootz) {
            for (u64 cornerDataSlice = 0; cornerDataSlice < xRankSize; cornerDataSlice++) {
                u64 currentInnerStepDataSliceId
                    = outSliceNum * xRankSize * yRankSize + oneDid * xRankSize + cornerDataSlice;
                if (cornerDataSlice != topo.rootx) {
                    u64 pieceId = currentInnerStepDataSliceId;
                    u64 sliceSizeOnePiece = DataSliceCut(
                        ySDataSize[root][osn][isn], xySOffset[root][osn] + ySOffset[root][osn][isn],
                        perLoop[pieceId].size);
                    u64 inputPieceIdOffset
                        = sliceOffsetCut(xySOffset[root][osn] + ySOffset[root][osn][isn], perLoop[pieceId].size)
                          + total[pieceId].offset;
                    u64 outputPieceIdOffset
                        = sliceOffsetCut(xySOffset[root][osn] + ySOffset[root][osn][isn], perLoop[pieceId].size)
                          + perLoop[pieceId].offset;
                    pieces.cnt.push_back(sliceSizeOnePiece / dataTypeSize);
                    pieces.sz.push_back(sliceSizeOnePiece);
                    pieces.in.push_back(inputPieceIdOffset);
                    pieces.out.push_back(outputPieceIdOffset);
                }
            }
        }
    }
}

void PushScatterYInnerCornerOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevely, u64 osn, u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 xySOffset[][MAX_STEP_NUM_SC], u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    const ScatterTopoInfo& topo, u64 yCclBufferBaseOff, u64 yInCornerStep, uint32_t root)
{
    HCCL_DEBUG("[PushScatterYInnerCornerOneOsn] start push scatter y inner corner one osn");
    for (u64 isn = 0; isn < yInCornerStep; isn++) {
        StepSliceInfo stepSliceInfotmp = MakeStepSliceInfo(yCclBufferBaseOff);
        ScatterPieceVecs pieces;
        for (u64 oneDid = 0; oneDid < topo.yRankSize; oneDid++) {
            if (oneDid == topo.rooty)
                continue;
            PushScatterYInnerCornerOneDiag(
                pieces, osn, isn, ySDataSize, xySOffset, ySOffset, perLoop, total, topo, oneDid, root);
        }
        for (u64 one = 0; one < topo.xRankSize; one++) {
            PushRootOrZeros(stepSliceInfotmp, pieces, one, topo.rootx, 0);
        }
        dataSliceLevely.insert(dataSliceLevely.end(), stepSliceInfotmp);
    }
}

static void CalcScatterYOverDiagPieces(
    u64 osn, u64 isn, u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 xySOffset[][MAX_STEP_NUM_SC],
    u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], const std::vector<OmniPipeSplitSliceInfo>& perLoop,
    const std::vector<OmniPipeSplitSliceInfo>& total, const ScatterTopoInfo& topo, uint32_t root,
    ScatterPieceVecs& pieces)
{
    u64 xRankSize = topo.xRankSize;
    u64 yRankSize = topo.yRankSize;
    u64 zRankSize = topo.zRankSize;
    for (u64 oneDid = 0; oneDid < yRankSize; oneDid++) {
        if (oneDid == topo.rooty) {
            continue;
        }
        for (u64 outSliceNum = 0; outSliceNum < zRankSize; outSliceNum++) {
            // oneDid 是 y 轴索引，pieceId = z*(xR*yR) + y*xR + x，这里 y=oneDid, x=rootx
            u64 currentDataSliceId = outSliceNum * xRankSize * yRankSize + oneDid * xRankSize + topo.rootx;
            if (outSliceNum != topo.rootz) {
                u64 pieceId = currentDataSliceId;
                u64 inputPieceIdOffset
                    = sliceOffsetCut(xySOffset[root][osn] + ySOffset[root][osn][isn], perLoop[pieceId].size)
                      + total[pieceId].offset;
                u64 sliceSizeOnePiece = DataSliceCut(
                    ySDataSize[root][osn][isn], xySOffset[root][osn] + ySOffset[root][osn][isn], perLoop[pieceId].size);
                u64 outputPieceIdOffset
                    = sliceOffsetCut(xySOffset[root][osn] + ySOffset[root][osn][isn], perLoop[pieceId].size)
                      + perLoop[pieceId].offset;
                pieces.sz.push_back(sliceSizeOnePiece);
                pieces.cnt.push_back(sliceSizeOnePiece / topo.dataTypeSize);
                pieces.in.push_back(inputPieceIdOffset);
                pieces.out.push_back(outputPieceIdOffset);
            }
        }
    }
}

static void CalcScatterYOverSameAxisPieceOffset(
    u64 osn, u64 isn, u64 innerStepNum, u64 xySDataSize[][MAX_STEP_NUM_SC],
    u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 xySOffset[][MAX_STEP_NUM_SC], u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, uint32_t root, u64 pieceId, u64& sliceSizeOnePiece,
    u64& inputPieceIdOffset, u64& outputPieceIdOffset)
{
    u64 xyBaseOff = xySOffset[root][osn];
    if (innerStepNum == 2) {
        inputPieceIdOffset
            = sliceOffsetCut(xyBaseOff + xSOffset[root][osn][isn - 1], perLoop[pieceId].size) + perLoop[pieceId].offset;
        outputPieceIdOffset
            = sliceOffsetCut(xyBaseOff + xSOffset[root][osn][isn - 1], perLoop[pieceId].size) + perLoop[pieceId].offset;
    } else {
        if (isn == innerStepNum - 1) {
            if (ySDataSize[root][osn][isn - 1] < xSDataSize[root][osn][isn - 2]) {
                inputPieceIdOffset = sliceOffsetCut(
                                         xyBaseOff + xSOffset[root][osn][isn - 2] + ySDataSize[root][osn][isn - 1],
                                         perLoop[pieceId].size)
                                     + perLoop[pieceId].offset;
                outputPieceIdOffset = sliceOffsetCut(
                                          xyBaseOff + xSOffset[root][osn][isn - 2] + ySDataSize[root][osn][isn - 1],
                                          perLoop[pieceId].size)
                                      + perLoop[pieceId].offset;
            } else {
                inputPieceIdOffset = sliceOffsetCut(
                                         xyBaseOff + xSOffset[root][osn][isn - 2] + xSDataSize[root][osn][isn - 2],
                                         perLoop[pieceId].size)
                                     + perLoop[pieceId].offset;
                outputPieceIdOffset = sliceOffsetCut(
                                          xyBaseOff + xSOffset[root][osn][isn - 2] + xSDataSize[root][osn][isn - 2],
                                          perLoop[pieceId].size)
                                      + perLoop[pieceId].offset;
            }
        } else {
            if (sliceSizeOnePiece > xSDataSize[root][osn][isn - 1]) {
                sliceSizeOnePiece = xSDataSize[root][osn][isn - 1];
            }
            inputPieceIdOffset = sliceOffsetCut(xyBaseOff + xSOffset[root][osn][isn - 1], perLoop[pieceId].size)
                                 + perLoop[pieceId].offset;
            outputPieceIdOffset = sliceOffsetCut(xyBaseOff + xSOffset[root][osn][isn - 1], perLoop[pieceId].size)
                                  + perLoop[pieceId].offset;
        }
    }
    if (inputPieceIdOffset + sliceSizeOnePiece > perLoop[pieceId].offset + xySDataSize[root][osn] + xyBaseOff) {
        sliceSizeOnePiece = perLoop[pieceId].offset + xySDataSize[root][osn] + xyBaseOff - inputPieceIdOffset;
    }
}

// 计算y轴外层同轴段非rootx节点的转发piece：遍历非rooty的y轴rank和非rootz的z轴rank
static void CalcScatterYOverSameAxisFwdPieces(
    u64 osn, u64 isn, u64 innerStepNum, u64 xySDataSize[][MAX_STEP_NUM_SC],
    u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 xySOffset[][MAX_STEP_NUM_SC], u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], const std::vector<OmniPipeSplitSliceInfo>& perLoop,
    const ScatterTopoInfo& topo, uint32_t root, u64 oneDid, ScatterPieceVecs& pieces)
{
    u64 xRankSize = topo.xRankSize;
    u64 yRankSize = topo.yRankSize;
    u64 zRankSize = topo.zRankSize;
    for (u64 sameAxisDataSlice = 0; sameAxisDataSlice < yRankSize; sameAxisDataSlice++) {
        if (sameAxisDataSlice == topo.rooty) {
            continue;
        }
        for (u64 outSliceNum = 0; outSliceNum < zRankSize; outSliceNum++) {
            if (outSliceNum == topo.rootz) {
                continue;
            }
            u64 pieceId = outSliceNum * xRankSize * yRankSize + oneDid * xRankSize + sameAxisDataSlice;
            u64 sliceSizeOnePiece = DataSliceCut(
                ySDataSize[root][osn][isn], xySOffset[root][osn] + ySOffset[root][osn][isn], perLoop[pieceId].size);
            u64 inputPieceIdOffset = 0;
            u64 outputPieceIdOffset = 0;
            CalcScatterYOverSameAxisPieceOffset(
                osn, isn, innerStepNum, xySDataSize, xSDataSize, ySDataSize, xySOffset, xSOffset, perLoop, root,
                pieceId, sliceSizeOnePiece, inputPieceIdOffset, outputPieceIdOffset);
            pieces.cnt.push_back(sliceSizeOnePiece / topo.dataTypeSize);
            pieces.sz.push_back(sliceSizeOnePiece);
            pieces.in.push_back(inputPieceIdOffset);
            pieces.out.push_back(outputPieceIdOffset);
        }
    }
}

void PushScatterYOverSameAxisOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevely, u64 osn, u64 xySDataSize[][MAX_STEP_NUM_SC],
    u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 xySOffset[][MAX_STEP_NUM_SC], u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], const std::vector<OmniPipeSplitSliceInfo>& perLoop,
    const std::vector<OmniPipeSplitSliceInfo>& total, const ScatterTopoInfo& topo,
    const std::vector<u64>& dataSizePerLoop, u64 yCclBufferBaseOff, u64 yInCornerStep, u64 innerStepNum, uint32_t root)
{
    for (u64 isn = yInCornerStep; isn < innerStepNum; isn++) {
        StepSliceInfo stepSliceInfotmp = MakeStepSliceInfo(yCclBufferBaseOff);
        ScatterPieceVecs pieces;
        CalcScatterYOverDiagPieces(osn, isn, ySDataSize, xySOffset, ySOffset, perLoop, total, topo, root, pieces);
        for (u64 oneDid = 0; oneDid < topo.xRankSize; oneDid++) {
            if (oneDid == topo.rootx) {
                PushPieces(stepSliceInfotmp, pieces, 0, 0);
            } else {
                ScatterPieceVecs pieces1;
                CalcScatterYOverSameAxisFwdPieces(
                    osn, isn, innerStepNum, xySDataSize, xSDataSize, ySDataSize, xySOffset, xSOffset, ySOffset, perLoop,
                    topo, root, oneDid, pieces1);
                PushPieces(stepSliceInfotmp, pieces1, 0, 0);
            }
        }
        dataSliceLevely.insert(dataSliceLevely.end(), stepSliceInfotmp);
    }
}

// scatter y轴内层同轴段（单个osn，isn∈[yInCornerStep,innerStepNum)）：root和同轴线节点处理y轴同轴通信
static void CalcScatterYInnerDiagPieces(
    u64 osn, u64 isn, u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 xySOffset[][MAX_STEP_NUM_SC],
    u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], const std::vector<OmniPipeSplitSliceInfo>& perLoop,
    const std::vector<OmniPipeSplitSliceInfo>& total, const ScatterTopoInfo& topo, uint32_t root,
    ScatterPieceVecs& pieces)
{
    u64 xRankSize = topo.xRankSize;
    u64 yRankSize = topo.yRankSize;
    u64 zRankSize = topo.zRankSize;
    u64 dataTypeSize = topo.dataTypeSize;
    for (u64 oneDid = 0; oneDid < yRankSize; oneDid++) {
        if (oneDid == topo.rooty)
            continue;
        for (u64 outSliceNum = 0; outSliceNum < zRankSize; outSliceNum++) {
            u64 currentDataSliceId = outSliceNum * xRankSize * yRankSize + oneDid * xRankSize + topo.rootx;
            if (outSliceNum != topo.rootz) {
                u64 pieceId = currentDataSliceId;
                u64 sliceSizeOnePiece = DataSliceCut(
                    ySDataSize[root][osn][isn], xySOffset[root][osn] + ySOffset[root][osn][isn], perLoop[pieceId].size);
                u64 inputPieceIdOffset
                    = sliceOffsetCut(xySOffset[root][osn] + ySOffset[root][osn][isn], perLoop[pieceId].size)
                      + total[pieceId].offset;
                u64 outputPieceIdOffset
                    = sliceOffsetCut(xySOffset[root][osn] + ySOffset[root][osn][isn], perLoop[pieceId].size)
                      + perLoop[pieceId].offset;
                pieces.sz.push_back(sliceSizeOnePiece);
                pieces.cnt.push_back(sliceSizeOnePiece / dataTypeSize);
                pieces.in.push_back(inputPieceIdOffset);
                pieces.out.push_back(outputPieceIdOffset);
            }
        }
    }
}

static void CalcScatterYInnerSameAxisPieces(
    u64 osn, u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 xySOffset[][MAX_STEP_NUM_SC],
    u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const ScatterTopoInfo& topo, uint32_t root, u64 one,
    ScatterPieceVecs& pieces)
{
    u64 xRankSize = topo.xRankSize;
    u64 yRankSize = topo.yRankSize;
    u64 zRankSize = topo.zRankSize;
    u64 dataTypeSize = topo.dataTypeSize;
    for (u64 sameAxisDataSlice = 0; sameAxisDataSlice < yRankSize; sameAxisDataSlice++) {
        if (sameAxisDataSlice == topo.rooty)
            continue;
        for (u64 outSliceNum = 0; outSliceNum < zRankSize; outSliceNum++) {
            if (outSliceNum == topo.rootz)
                continue;
            // sameAxisDataSlice 是 y 轴索引，one 是 x 轴索引，pieceId = z*(xR*yR) + y*xR + x
            u64 pieceId = outSliceNum * xRankSize * yRankSize + sameAxisDataSlice * xRankSize + one;
            u64 sliceSizeOnePiece = DataSliceCut(
                xSDataSize[root][osn][0], xySOffset[root][osn] + ySOffset[root][osn][0], perLoop[pieceId].size);
            u64 inputPieceIdOffset
                = sliceOffsetCut(xySOffset[root][osn] + xSOffset[root][osn][0], perLoop[pieceId].size)
                  + perLoop[pieceId].offset;
            u64 outputPieceIdOffset
                = sliceOffsetCut(xySOffset[root][osn] + xSOffset[root][osn][0], perLoop[pieceId].size)
                  + perLoop[pieceId].offset;
            pieces.sz.push_back(sliceSizeOnePiece);
            pieces.cnt.push_back(sliceSizeOnePiece / dataTypeSize);
            pieces.in.push_back(inputPieceIdOffset);
            pieces.out.push_back(outputPieceIdOffset);
        }
    }
}

void PushScatterYInnerSameAxisOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevely, u64 osn, u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 xySOffset[][MAX_STEP_NUM_SC],
    u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    const ScatterTopoInfo& topo, const std::vector<u64>& dataSizePerLoop, u64 yCclBufferBaseOff, u64 yInCornerStep,
    u64 innerStepNum, uint32_t root)
{
    HCCL_DEBUG("[PushScatterYInnerSameAxisOneOsn] start push scatter y inner same axis one osn");
    for (u64 isn = yInCornerStep; isn < innerStepNum; isn++) {
        StepSliceInfo stepSliceInfotmp = MakeStepSliceInfo(yCclBufferBaseOff);
        u64 xRankSize = topo.xRankSize;
        ScatterPieceVecs pieces;
        CalcScatterYInnerDiagPieces(osn, isn, ySDataSize, xySOffset, ySOffset, perLoop, total, topo, root, pieces);
        for (u64 one = 0; one < xRankSize; one++) {
            if (one == topo.rootx) {
                PushPieces(stepSliceInfotmp, pieces, 0, 0);
            } else {
                ScatterPieceVecs pieces1;
                CalcScatterYInnerSameAxisPieces(
                    osn, xSDataSize, xySOffset, xSOffset, ySOffset, perLoop, topo, root, one, pieces1);
                PushPieces(stepSliceInfotmp, pieces1, 0, 0);
            }
        }
        dataSliceLevely.insert(dataSliceLevely.end(), stepSliceInfotmp);
    }
}

// 收集y轴外层斜对角piece：遍历非rooty的y轴rank和非rootx的x轴rank，计算xyoffset并调用CalcAndPushPiece
static void PushScatterYOuterCornerDiagPieces(
    u64 osn, u64 isn, u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 xySOffset[][MAX_STEP_NUM_SC],
    u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], const std::vector<OmniPipeSplitSliceInfo>& perLoop,
    const std::vector<OmniPipeSplitSliceInfo>& total, const ScatterTopoInfo& topo, uint32_t root, u64 dataTypeSize,
    ScatterPieceVecs& pieces)
{
    u64 xRankSize = topo.xRankSize;
    u64 yRankSize = topo.yRankSize;
    u64 rootx = topo.rootx;
    u64 rooty = topo.rooty;
    for (u64 oneDid = 0; oneDid < yRankSize; oneDid++) {
        if (oneDid == rooty) {
            continue;
        }
        for (u64 cornerDataSlice = 0; cornerDataSlice < xRankSize; cornerDataSlice++) {
            if (cornerDataSlice == rootx) {
                continue;
            }
            u64 currentDataSliceId = topo.zAxis * xRankSize * yRankSize + oneDid * xRankSize + cornerDataSlice;
            u64 xyoffset = GetXyOffset(topo, xySOffset[root], osn);
            CalcAndPushPiece(
                currentDataSliceId, xyoffset + ySOffset[root][osn][isn], ySDataSize[root][osn][isn], perLoop, total,
                dataTypeSize, pieces);
        }
    }
}

// scatter y轴外层 xB<=yB
// 斜对角段（单个osn，isn∈[0,yInCornerStep+1)）：root发斜对角数据，root数据放index=rootx，其余x轴rank塞0
void PushScatterYOuterLECornerOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevely, u64 osn, u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 xySOffset[][MAX_STEP_NUM_SC], u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    const ScatterTopoInfo& topo, uint32_t root, u64 yCclBufferBaseOff, u64 yInCornerStep)
{
    HCCL_DEBUG("[PushScatterYOuterLECornerOneOsn] start push scatter y outer le corner one osn");
    for (u64 isn = 0; isn < yInCornerStep + 1; isn++) {
        StepSliceInfo stepSliceInfotmp = MakeStepSliceInfo(yCclBufferBaseOff);
        u64 xRankSize = topo.xRankSize;
        u64 yRankSize = topo.yRankSize;
        u64 dataTypeSize = topo.dataTypeSize;
        ScatterPieceVecs pieces;
        PushScatterYOuterCornerDiagPieces(
            osn, isn, ySDataSize, xySOffset, ySOffset, perLoop, total, topo, root, dataTypeSize, pieces);
        for (u64 oneDid = 0; oneDid < xRankSize; oneDid++) {
            PushRootOrZeros(stepSliceInfotmp, pieces, oneDid, topo.rootx, 0);
        }
        dataSliceLevely.insert(dataSliceLevely.end(), stepSliceInfotmp);
    }
}

// scatter y轴外层 xB<=yB 同轴转发段：为非rootx的x轴节点构建转发piece并Push
void PushScatterYOuterLEFwdOneRank(
    StepSliceInfo& stepSliceInfotmp, u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 xySOffset[][MAX_STEP_NUM_SC], u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, uint32_t root, u64 osn, const ScatterTopoInfo& topo, u64 oneDid)
{
    HCCL_DEBUG("[PushScatterYOuterLEFwdOneRank] start push scatter y outer le fwd one rank");
    u64 xRankSize = topo.xRankSize;
    u64 yRankSize = topo.yRankSize;
    u64 dataTypeSize = topo.dataTypeSize;
    ScatterPieceVecs pieces1;
    for (u64 cornerDataSlice = 0; cornerDataSlice < yRankSize; cornerDataSlice++) {
        if (cornerDataSlice == topo.rooty)
            continue;
        u64 currentDataSliceId = topo.zAxis * xRankSize * yRankSize + cornerDataSlice * xRankSize + oneDid;
        u64 xyoffset = GetXyOffset(topo, xySOffset[root], osn);
        CalcAndPushPiece(
            currentDataSliceId, xyoffset + xSOffset[root][osn][0], xSDataSize[root][osn][0], perLoop, perLoop,
            dataTypeSize, pieces1);
    }
    PushPieces(stepSliceInfotmp, pieces1, 0, 0);
}

// scatter y轴外层 xB<=yB
// 同轴转发段（单个osn，isn∈[yInCornerStep+1,innerStepNum)）：root发同y轴数据，同x轴非root节点转发step1收到的对角数据
void PushScatterYOuterLESameAxisOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevely, u64 osn, u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 xySOffset[][MAX_STEP_NUM_SC],
    u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    const ScatterTopoInfo& topo, uint32_t root, u64 yCclBufferBaseOff, u64 yInCornerStep, u64 innerStepNum)
{
    HCCL_DEBUG("[PushScatterYOuterLESameAxisOneOsn] start push scatter y outer le same axis one osn");
    for (u64 isn = yInCornerStep + 1; isn < innerStepNum; isn++) {
        StepSliceInfo stepSliceInfotmp = MakeStepSliceInfo(yCclBufferBaseOff);
        u64 xRankSize = topo.xRankSize;
        u64 yRankSize = topo.yRankSize;
        ScatterPieceVecs pieces;
        for (u64 oneDid = 0; oneDid < yRankSize; oneDid++) {
            if (oneDid == topo.rooty)
                continue;
            u64 pieceId = topo.zAxis * xRankSize * yRankSize + oneDid * xRankSize + topo.rootx;
            u64 xyoffset = GetXyOffset(topo, xySOffset[root], osn);
            CalcAndPushPiece(
                pieceId, xyoffset + ySOffset[root][osn][isn], ySDataSize[root][osn][isn], perLoop, total,
                topo.dataTypeSize, pieces);
        }
        for (u64 oneDid = 0; oneDid < xRankSize; oneDid++) {
            if (oneDid == topo.rootx) {
                PushPieces(stepSliceInfotmp, pieces, 0, 0);
            } else {
                PushScatterYOuterLEFwdOneRank(
                    stepSliceInfotmp, xSDataSize, xySOffset, xSOffset, perLoop, root, osn, topo, oneDid);
            }
        }
        dataSliceLevely.insert(dataSliceLevely.end(), stepSliceInfotmp);
    }
}

// scatter y轴外层 xB>yB
// 斜对角段（单个osn，isn∈[0,yInCornerStep)）：第一步只有root发斜对角数据，buffer用xCclBufferBaseOff
void PushScatterYOuterGTCornerOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevely, u64 osn, u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 xySOffset[][MAX_STEP_NUM_SC], u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, const std::vector<OmniPipeSplitSliceInfo>& total,
    const ScatterTopoInfo& topo, uint32_t root, u64 xCclBufferBaseOff, u64 yInCornerStep)
{
    HCCL_DEBUG("[PushScatterYOuterGTCornerOneOsn] start push scatter y outer gt corner one osn");
    for (u64 isn = 0; isn < yInCornerStep; isn++) {
        StepSliceInfo stepSliceInfotmp = MakeStepSliceInfo(xCclBufferBaseOff);
        u64 xRankSize = topo.xRankSize;
        u64 yRankSize = topo.yRankSize;
        u64 dataTypeSize = topo.dataTypeSize;
        ScatterPieceVecs pieces;
        for (u64 oneDid = 0; oneDid < yRankSize; oneDid++) {
            if (oneDid == topo.rooty) {
                continue;
            }
            PushScatterYOuterCornerDiagPieces(
                osn, isn, ySDataSize, xySOffset, ySOffset, perLoop, total, topo, root, dataTypeSize, pieces);
        }
        for (u64 oneRank = 0; oneRank < xRankSize; oneRank++) {
            PushRootOrZeros(stepSliceInfotmp, pieces, oneRank, topo.rootx, 0);
        }
        dataSliceLevely.insert(dataSliceLevely.end(), stepSliceInfotmp);
    }
}

// scatter y轴外层 xB>yB
// 同轴转发段（单个osn，isn∈[yInCornerStep,innerStepNum)）：root发同y轴数据，同x轴非root节点转发step1收到的对角数据
void CalcScatterYOuterGTOffset(
    u64& inputPieceIdOffset, u64& outputPieceIdOffset, u64& sliceSizeOnePiece,
    u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 xySOffset[][MAX_STEP_NUM_SC], u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& perLoop, uint32_t root, u64 osn, u64 isn, u64 innerStepNum, u64 pieceId,
    const ScatterTopoInfo& topo)
{
    HCCL_DEBUG("[CalcScatterYOuterGTOffset] start calc scatter y outer gt offset");
    u64 xyoffset = GetXyOffset(topo, xySOffset[root], osn);
    if (innerStepNum == 2) {
        inputPieceIdOffset
            = sliceOffsetCut(xyoffset + xSOffset[root][osn][isn - 1], perLoop[pieceId].size) + perLoop[pieceId].offset;
        outputPieceIdOffset
            = sliceOffsetCut(xyoffset + xSOffset[root][osn][isn - 1], perLoop[pieceId].size) + perLoop[pieceId].offset;
    } else {
        if (isn == innerStepNum - 1) {
            if (ySDataSize[root][osn][isn - 1] < xSDataSize[root][osn][isn - 2]) {
                inputPieceIdOffset = sliceOffsetCut(
                                         xyoffset + xSOffset[root][osn][isn - 2] + ySDataSize[root][osn][isn - 1],
                                         perLoop[pieceId].size)
                                     + perLoop[pieceId].offset;
                outputPieceIdOffset = sliceOffsetCut(
                                          xyoffset + xSOffset[root][osn][isn - 2] + ySDataSize[root][osn][isn - 1],
                                          perLoop[pieceId].size)
                                      + perLoop[pieceId].offset;
            } else {
                inputPieceIdOffset = sliceOffsetCut(
                                         xyoffset + xSOffset[root][osn][isn - 2] + xSDataSize[root][osn][isn - 2],
                                         perLoop[pieceId].size)
                                     + perLoop[pieceId].offset;
                outputPieceIdOffset = sliceOffsetCut(
                                          xyoffset + xSOffset[root][osn][isn - 2] + xSDataSize[root][osn][isn - 2],
                                          perLoop[pieceId].size)
                                      + perLoop[pieceId].offset;
            }
        } else {
            if (sliceSizeOnePiece > xSDataSize[root][osn][isn - 1]) {
                sliceSizeOnePiece = xSDataSize[root][osn][isn - 1];
            }
            inputPieceIdOffset = sliceOffsetCut(xyoffset + xSOffset[root][osn][isn - 1], perLoop[pieceId].size)
                                 + perLoop[pieceId].offset;
            outputPieceIdOffset = sliceOffsetCut(xyoffset + xSOffset[root][osn][isn - 1], perLoop[pieceId].size)
                                  + perLoop[pieceId].offset;
        }
    }
}

// scatter y轴外层 xB>yB 同轴转发段：为非rootx的x轴节点构建转发piece并Push
void PushScatterYOuterGTFwdOneRank(
    StepSliceInfo& stepSliceInfotmp, u64 xySDataSize[][MAX_STEP_NUM_SC],
    u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 xySOffset[][MAX_STEP_NUM_SC], u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], const std::vector<OmniPipeSplitSliceInfo>& perLoop, uint32_t root,
    u64 osn, u64 isn, u64 innerStepNum, const ScatterTopoInfo& topo, const std::vector<u64>& dataSizePerLoop)
{
    HCCL_DEBUG("[PushScatterYOuterGTFwdOneRank] start push scatter y outer gt fwd one rank");
    u64 xRankSize = topo.xRankSize;
    u64 yRankSize = topo.yRankSize;
    u64 dataTypeSize = topo.dataTypeSize;
    ScatterPieceVecs pieces;
    bool sameZAxis = (topo.rootz != topo.zAxis);
    for (u64 cornerDataSlice = 0; cornerDataSlice < yRankSize; cornerDataSlice++) {
        if (cornerDataSlice == topo.rooty)
            continue;
        u64 pieceId = topo.zAxis * xRankSize * yRankSize + cornerDataSlice * xRankSize + topo.rootx;
        u64 xyoffset = sameZAxis ? xySOffset[root][osn - 1] : xySOffset[root][osn];
        u64 xysize = sameZAxis ? xySDataSize[root][osn - 1] : xySDataSize[root][osn];
        u64 sliceSizeOnePiece
            = DataSliceCut(ySDataSize[root][osn][isn], xyoffset + ySOffset[root][osn][isn], perLoop[pieceId].size);
        u64 inputPieceIdOffset = 0;
        u64 outputPieceIdOffset = 0;
        CalcScatterYOuterGTOffset(
            inputPieceIdOffset, outputPieceIdOffset, sliceSizeOnePiece, xSDataSize, ySDataSize, xySOffset, xSOffset,
            perLoop, root, osn, isn, innerStepNum, pieceId, topo);
        if (topo.zRankSize <= 1) {
            if (inputPieceIdOffset + sliceSizeOnePiece > perLoop[pieceId].offset + dataSizePerLoop[root]) {
                sliceSizeOnePiece = perLoop[pieceId].offset + dataSizePerLoop[root] - inputPieceIdOffset;
            }
        } else {
            if (inputPieceIdOffset + sliceSizeOnePiece > perLoop[pieceId].offset + xysize + xyoffset) {
                sliceSizeOnePiece = perLoop[pieceId].offset + xysize + xyoffset - inputPieceIdOffset;
            }
        }

        pieces.cnt.push_back(sliceSizeOnePiece / dataTypeSize);
        pieces.sz.push_back(sliceSizeOnePiece);
        pieces.out.push_back(outputPieceIdOffset);
        pieces.in.push_back(inputPieceIdOffset);
    }
    PushPieces(stepSliceInfotmp, pieces, 0, 0);
}

void PushScatterYOuterGTSameAxisOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevely, u64 osn, u64 xySDataSize[][MAX_STEP_NUM_SC],
    u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 xySOffset[][MAX_STEP_NUM_SC], u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], const std::vector<OmniPipeSplitSliceInfo>& perLoop,
    const std::vector<OmniPipeSplitSliceInfo>& total, const ScatterTopoInfo& topo, uint32_t root, u64 xCclBufferBaseOff,
    u64 yInCornerStep, u64 innerStepNum, const std::vector<u64>& dataSizePerLoop)
{
    HCCL_DEBUG("[PushScatterYOuterGTSameAxisOneOsn] start push scatter y outer gt same axis one osn");
    u64 xRankSize = topo.xRankSize;
    u64 yRankSize = topo.yRankSize;
    u64 rootx = topo.rootx;
    u64 rooty = topo.rooty;
    for (u64 isn = yInCornerStep; isn < innerStepNum; isn++) {
        StepSliceInfo stepSliceInfotmp = MakeStepSliceInfo(xCclBufferBaseOff);
        ScatterPieceVecs pieces;
        for (u64 oneDid = 0; oneDid < yRankSize; oneDid++) {
            if (oneDid == rooty)
                continue;
            u64 pieceId = topo.zAxis * xRankSize * yRankSize + oneDid * xRankSize + rootx;
            u64 xyoffsets = GetXyOffset(topo, xySOffset[root], osn);
            CalcAndPushPiece(
                pieceId, xyoffsets + ySOffset[root][osn][isn], ySDataSize[root][osn][isn], perLoop, total,
                topo.dataTypeSize, pieces);
        }
        for (u64 rankx = 0; rankx < xRankSize; rankx++) {
            if (rankx == rootx) {
                PushPieces(stepSliceInfotmp, pieces, 0, 0);
            } else {
                PushScatterYOuterGTFwdOneRank(
                    stepSliceInfotmp, xySDataSize, xSDataSize, ySDataSize, xySOffset, xSOffset, ySOffset, perLoop, root,
                    osn, isn, innerStepNum, topo, dataSizePerLoop);
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
        info.xyB = CalcBandwidth2D(info.xB, info.yB, info.xRankSize, info.yRankSize, MAX_STEP_NUM_SC);
    } else {
        info.xyB = CalcBandwidth2D(info.yB, info.xB, info.yRankSize, info.xRankSize, MAX_STEP_NUM_SC);
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
    u64 rankSize, u64 zSDataSize[][MAX_STEP_NUM_SC], u64 xySDataSize[][MAX_STEP_NUM_SC],
    u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 zSOffset[][MAX_STEP_NUM_SC], u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 xySOffset[][MAX_STEP_NUM_SC])
{
    for (u64 rs = 0; rs < rankSize; rs++) {
        for (u64 i = 0; i < MAX_STEP_NUM_SC; i++) {
            zSDataSize[rs][i] = 0;
            xySDataSize[rs][i] = 0;
            zSOffset[rs][i] = 0;
            xySOffset[rs][i] = 0;
            for (u64 j = 0; j < MAX_STEP_NUM_SC; j++) {
                xSDataSize[rs][i][j] = 0;
                ySDataSize[rs][i][j] = 0;
                xSOffset[rs][i][j] = 0;
                ySOffset[rs][i][j] = 0;
            }
        }
    }
}

// 计算单个rank的scatter数据大小与偏移（isZSlowAxis决定外层轴选择，isXSlowAxis决定内层轴选择）
static void CalcScatterInnerStepOnce(
    u64 rs, u64 i, const ScatterTopoInfo& topo, ScatterStepState& state, double innerSlowBw, double innerFastBw,
    u64 innerSlowRankSize, u64 innerFastRankSize, u64 prevStepDataSize,
    u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC])
{
    u64* innerSlowDataSize = state.isXSlowAxis ? ySDataSize[rs][i] : xSDataSize[rs][i];
    u64* innerFastDataSize = state.isXSlowAxis ? xSDataSize[rs][i] : ySDataSize[rs][i];
    u64* innerSlowOffset = state.isXSlowAxis ? ySOffset[rs][i] : xSOffset[rs][i];
    u64* innerFastOffset = state.isXSlowAxis ? xSOffset[rs][i] : ySOffset[rs][i];

    state.innerStepNum = CalScatterDataSize2D(
        innerSlowDataSize, innerFastDataSize, innerSlowBw, innerFastBw, innerSlowRankSize, innerFastRankSize,
        prevStepDataSize, MAX_STEP_NUM_SC);
    HCCL_DEBUG("[CalcScatterOneRankDataSize] innerStepNum: %llu", state.innerStepNum);
    CalScatter2DOffset(
        innerSlowOffset, innerFastOffset, state.innerStepNum, innerSlowRankSize, innerFastRankSize, innerSlowDataSize,
        innerFastDataSize);
}

// 处理与root同z轴节点所在机器xy平面的内层步计算：zB>=xyB时从第2大步开始会有多次发同z轴数据，否则固定在第2大步发一次
static void CalcScatterRootSameAxisInnerSteps(
    u64 rs, const ScatterTopoInfo& topo, ScatterStepState& state, double innerSlowBw, double innerFastBw,
    u64 innerSlowRankSize, u64 innerFastRankSize, u64* slowDataSize, u64* fastDataSize,
    u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC])
{
    if (topo.zB >= topo.xyB) {
        for (u64 i = 1; i < state.outerStepNum; i++) {
            CalcScatterInnerStepOnce(
                rs, i, topo, state, innerSlowBw, innerFastBw, innerSlowRankSize, innerFastRankSize, fastDataSize[i - 1],
                xSDataSize, ySDataSize, xSOffset, ySOffset);
        }
    } else {
        CalcScatterInnerStepOnce(
            rs, 1, topo, state, innerSlowBw, innerFastBw, innerSlowRankSize, innerFastRankSize, slowDataSize[0],
            xSDataSize, ySDataSize, xSOffset, ySOffset);
    }
}

// 处理非root同z轴节点的内层步计算：遍历所有外层步，按isZSlowAxis选择prevStepDataSize
static void CalcScatterNonRootSameAxisInnerSteps(
    u64 rs, const ScatterTopoInfo& topo, ScatterStepState& state, double innerSlowBw, double innerFastBw,
    u64 innerSlowRankSize, u64 innerFastRankSize, u64* slowDataSize, u64* fastDataSize,
    u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC])
{
    for (u64 i = 0; i < state.outerStepNum; i++) {
        u64 prevStepDataSize = state.isZSlowAxis ? fastDataSize[i] : slowDataSize[i];
        CalcScatterInnerStepOnce(
            rs, i, topo, state, innerSlowBw, innerFastBw, innerSlowRankSize, innerFastRankSize, prevStepDataSize,
            xSDataSize, ySDataSize, xSOffset, ySOffset);
    }
}

// 根据innerStepNum与finStepMark的关系计算InCornerStep：innerStepNum>finStepMark时慢轴InCornerStep=1，快轴取差值
static void CalcScatterInCornerStep(ScatterStepState& state, u64 finStepMark)
{
    if (state.innerStepNum <= finStepMark) {
        return;
    }
    if (state.isXSlowAxis) {
        state.yInCornerStep = 1;
        state.xInCornerStep = state.innerStepNum - finStepMark;
    } else {
        state.xInCornerStep = 1;
        state.yInCornerStep = state.innerStepNum - finStepMark;
    }
}

void CalcScatterOneRankDataSize(
    const ScatterTopoInfo& topo, ScatterStepState& state, u64 rs, u64 finStepMark, double slowBw, double fastBw,
    u64 slowRankSize, u64 fastRankSize, u64 zSDataSize[][MAX_STEP_NUM_SC], u64 xySDataSize[][MAX_STEP_NUM_SC],
    u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 zSOffset[][MAX_STEP_NUM_SC], u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 xySOffset[][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& omniPipeSplitSliceInfoListPerLoop)
{
    u64* slowDataSize = state.isZSlowAxis ? zSDataSize[rs] : xySDataSize[rs];
    u64* fastDataSize = state.isZSlowAxis ? xySDataSize[rs] : zSDataSize[rs];
    u64* slowOffset = state.isZSlowAxis ? zSOffset[rs] : xySOffset[rs];
    u64* fastOffset = state.isZSlowAxis ? xySOffset[rs] : zSOffset[rs];

    state.outerStepNum = CalScatterDataSize2D(
        slowDataSize, fastDataSize, slowBw, fastBw, slowRankSize, fastRankSize,
        omniPipeSplitSliceInfoListPerLoop[rs].size, MAX_STEP_NUM_SC - 1);
    HCCL_DEBUG("[CalcScatterOneRankDataSize] outerStepNum: %llu", state.outerStepNum);

    double innerSlowBw = state.isXSlowAxis ? topo.yB : topo.xB;
    double innerFastBw = state.isXSlowAxis ? topo.xB : topo.yB;
    u64 innerSlowRankSize = state.isXSlowAxis ? topo.yRankSize : topo.xRankSize;
    u64 innerFastRankSize = state.isXSlowAxis ? topo.xRankSize : topo.yRankSize;

    if (topo.rootz != topo.zAxis) {
        // 处理与root同z轴节点所在机器xy平面的内层计算
        CalcScatterRootSameAxisInnerSteps(
            rs, topo, state, innerSlowBw, innerFastBw, innerSlowRankSize, innerFastRankSize, slowDataSize, fastDataSize,
            xSDataSize, ySDataSize, xSOffset, ySOffset);
    } else {
        // 处理root节点所在机器xy平面的内层计算
        CalcScatterNonRootSameAxisInnerSteps(
            rs, topo, state, innerSlowBw, innerFastBw, innerSlowRankSize, innerFastRankSize, slowDataSize, fastDataSize,
            xSDataSize, ySDataSize, xSOffset, ySOffset);
    }

    CalcScatterInCornerStep(state, finStepMark);

    CalScatter2DOffset(
        slowOffset, fastOffset, state.outerStepNum, slowRankSize, fastRankSize, slowDataSize, fastDataSize);
}

// 计算外层corner step（z轴与xy轴的对齐步数）
void CalcScatterOuterCornerStep(const ScatterTopoInfo& topo, ScatterStepState& state, u64 finStepMark)
{
    HCCL_DEBUG("[CalcScatterOuterCornerStep] start calc scatter outer corner step");
    if (state.outerStepNum > 1) {
        if (state.isZSlowAxis) {
            state.zCornerStep = 1;
            state.xyCornerStep = (state.outerStepNum == 2 ? 1 : state.outerStepNum - finStepMark);
        } else {
            state.zCornerStep = (state.outerStepNum == 2 ? 1 : state.outerStepNum - finStepMark);
            state.xyCornerStep = 1;
        }
    }
    if (topo.rootz != topo.zAxis) {
        state.xyCornerStep = 0;
    }
}

// 计算所有rank的scatter数据大小与偏移
void CalcScatterAllRankDataSize(
    const ScatterTopoInfo& topo, ScatterStepState& state, uint32_t root, u64 finStepMark,
    u64 zSDataSize[][MAX_STEP_NUM_SC], u64 xySDataSize[][MAX_STEP_NUM_SC],
    u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 zSOffset[][MAX_STEP_NUM_SC], u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 xySOffset[][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& omniPipeSplitSliceInfoListPerLoop)
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
    CalcScatterOuterCornerStep(topo, state, finStepMark);
}

// 构建X轴所有step的slice信息（inner corner+sameAxis + outer corner+sameAxis）
static void PushScatterXCornerLEOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevelx, u64 osn, u64 xySDataSize[][MAX_STEP_NUM_SC],
    u64 zSDataSize[][MAX_STEP_NUM_SC], u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 xySOffset[][MAX_STEP_NUM_SC],
    u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& omniPipeSplitSliceInfoListPerLoop,
    const std::vector<OmniPipeSplitSliceInfo>& omniPipeSplitSliceInfoListTotal, const ScatterTopoInfo& topo,
    const std::vector<u64>& dataSizePerLoop, u64 xCclBufferBaseOff, u64 xInCornerStep, u64 innerStepNum, uint32_t root)
{
    PushScatterXInnerCornerOneOsn(
        dataSliceLevelx, osn, xSDataSize, xySOffset, xSOffset, omniPipeSplitSliceInfoListPerLoop,
        omniPipeSplitSliceInfoListTotal, topo, dataSizePerLoop, xCclBufferBaseOff, xInCornerStep, root);
    PushScatterXInnerSameAxisOneOsn(
        dataSliceLevelx, osn, xySDataSize, zSDataSize, xSDataSize, ySDataSize, xySOffset, xSOffset, ySOffset,
        omniPipeSplitSliceInfoListPerLoop, omniPipeSplitSliceInfoListTotal, topo, dataSizePerLoop, xCclBufferBaseOff,
        xInCornerStep, innerStepNum, root);
}

static void PushScatterXCornerGtOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevelx, u64 osn, u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 xySOffset[][MAX_STEP_NUM_SC],
    u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& omniPipeSplitSliceInfoListPerLoop,
    const std::vector<OmniPipeSplitSliceInfo>& omniPipeSplitSliceInfoListTotal, const ScatterTopoInfo& topo,
    const std::vector<u64>& dataSizePerLoop, u64 xCclBufferBaseOff, u64 xInCornerStep, u64 innerStepNum, uint32_t root)
{
    PushScatterXInnerCornerOneOsn(
        dataSliceLevelx, osn, xSDataSize, xySOffset, xSOffset, omniPipeSplitSliceInfoListPerLoop,
        omniPipeSplitSliceInfoListTotal, topo, dataSizePerLoop, xCclBufferBaseOff, xInCornerStep + 1, root);
    PushScatterXOverSameAxisOneOsn(
        dataSliceLevelx, osn, xSDataSize, ySDataSize, xySOffset, xSOffset, ySOffset, omniPipeSplitSliceInfoListPerLoop,
        omniPipeSplitSliceInfoListTotal, topo, dataSizePerLoop, xCclBufferBaseOff, xInCornerStep, innerStepNum, root);
}

void PushScatterXAllSteps(
    std::vector<StepSliceInfo>& dataSliceLevelx, const ScatterTopoInfo& topo, const ScatterStepState& state,
    u64 zSDataSize[][MAX_STEP_NUM_SC], u64 xySDataSize[][MAX_STEP_NUM_SC],
    u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 zSOffset[][MAX_STEP_NUM_SC], u64 xySOffset[][MAX_STEP_NUM_SC], u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& omniPipeSplitSliceInfoListPerLoop,
    const std::vector<OmniPipeSplitSliceInfo>& omniPipeSplitSliceInfoListTotal, uint32_t root, u64 xCclBufferBaseOff,
    u64 yCclBufferBaseOff, const std::vector<u64>& dataSizePerLoop)
{
    HCCL_DEBUG("[PushScatterXAllSteps] start push scatter X Axis all steps");
    // 针对x轴，处理xy平面跨平面的对角节点
    if (state.xyCornerStep > 0) {
        if (topo.xB <= topo.yB) {
            // 当x轴带宽小于y轴带宽时，在x轴方向上第一步发的跨平面的最远的斜对角节点，后续发的是近的跨平面的对角
            for (u64 osn = 0; osn < state.xyCornerStep; osn++) {
                PushScatterXCornerLEOneOsn(
                    dataSliceLevelx, osn, xySDataSize, zSDataSize, xSDataSize, ySDataSize, xySOffset, xSOffset,
                    ySOffset, omniPipeSplitSliceInfoListPerLoop, omniPipeSplitSliceInfoListTotal, topo, dataSizePerLoop,
                    xCclBufferBaseOff, state.xInCornerStep, state.innerStepNum, root);
            }
        } else {
            // 当x轴带宽大于y轴带宽时，在x轴方向上xInCornerStep+1步发的跨平面的最远的斜对角节点，最后一步发的是近的跨平面的对角
            for (u64 osn = 0; osn < state.xyCornerStep; osn++) {
                PushScatterXCornerGtOneOsn(
                    dataSliceLevelx, osn, xSDataSize, ySDataSize, xySOffset, xSOffset, ySOffset,
                    omniPipeSplitSliceInfoListPerLoop, omniPipeSplitSliceInfoListTotal, topo, dataSizePerLoop,
                    xCclBufferBaseOff, state.xInCornerStep, state.innerStepNum, root);
            }
        }
    }

    bool sameZAxis = (topo.rootz != topo.zAxis);
    const std::vector<OmniPipeSplitSliceInfo>& omniPipeSplitSliceInfoList
        = sameZAxis ? omniPipeSplitSliceInfoListPerLoop : omniPipeSplitSliceInfoListTotal;
    // 处理本平面的对角节点的x轴
    if (topo.xB <= topo.yB) {
        for (u64 osn = state.xyCornerStep; osn < state.outerStepNum; osn++) {
            PushScatterXOuterLECornerOneOsn(
                dataSliceLevelx, osn, xSDataSize, sameZAxis ? zSOffset : xySOffset, xSOffset,
                omniPipeSplitSliceInfoListPerLoop, omniPipeSplitSliceInfoList, topo, root, xCclBufferBaseOff,
                state.xInCornerStep);
            PushScatterXOuterLESameAxisOneOsn(
                dataSliceLevelx, osn, sameZAxis ? zSDataSize : xySDataSize, xSDataSize, ySDataSize,
                sameZAxis ? zSOffset : xySOffset, xSOffset, ySOffset, omniPipeSplitSliceInfoListPerLoop,
                omniPipeSplitSliceInfoList, topo, root, xCclBufferBaseOff, state.xInCornerStep, state.innerStepNum,
                dataSizePerLoop);
        }
    } else {
        for (u64 osn = state.xyCornerStep; osn < state.outerStepNum; osn++) {
            PushScatterXOuterGTCornerOneOsn(
                dataSliceLevelx, osn, xSDataSize, sameZAxis ? zSOffset : xySOffset, xSOffset,
                omniPipeSplitSliceInfoListPerLoop, omniPipeSplitSliceInfoList, topo, root, yCclBufferBaseOff,
                state.xInCornerStep);
            PushScatterXOuterGTSameAxisOneOsn(
                dataSliceLevelx, osn, xSDataSize, ySDataSize, sameZAxis ? zSOffset : xySOffset, xSOffset, ySOffset,
                omniPipeSplitSliceInfoListPerLoop, omniPipeSplitSliceInfoList, topo, root, yCclBufferBaseOff,
                state.xInCornerStep, state.innerStepNum);
        }
    }
}

// 构建Y轴所有step的slice信息（inner corner+sameAxis + outer corner+sameAxis）
static void PushScatterYCornerLEOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevely, u64 osn, u64 xySDataSize[][MAX_STEP_NUM_SC],
    u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 xySOffset[][MAX_STEP_NUM_SC], u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& omniPipeSplitSliceInfoListPerLoop,
    const std::vector<OmniPipeSplitSliceInfo>& omniPipeSplitSliceInfoListTotal, const ScatterTopoInfo& topo,
    const std::vector<u64>& dataSizePerLoop, u64 yCclBufferBaseOff, u64 yInCornerStep, u64 innerStepNum, uint32_t root)
{
    PushScatterYInnerCornerOneOsn(
        dataSliceLevely, osn, ySDataSize, xySOffset, ySOffset, omniPipeSplitSliceInfoListPerLoop,
        omniPipeSplitSliceInfoListTotal, topo, yCclBufferBaseOff, yInCornerStep + 1, root);
    PushScatterYInnerSameAxisOneOsn(
        dataSliceLevely, osn, xSDataSize, ySDataSize, xySOffset, xSOffset, ySOffset, omniPipeSplitSliceInfoListPerLoop,
        omniPipeSplitSliceInfoListTotal, topo, dataSizePerLoop, yCclBufferBaseOff, yInCornerStep + 1, innerStepNum,
        root);
}

static void PushScatterYCornerGtOneOsn(
    std::vector<StepSliceInfo>& dataSliceLevely, u64 osn, u64 innerStepNum, u64 xySDataSize[][MAX_STEP_NUM_SC],
    u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 xySOffset[][MAX_STEP_NUM_SC], u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& omniPipeSplitSliceInfoListPerLoop,
    const std::vector<OmniPipeSplitSliceInfo>& omniPipeSplitSliceInfoListTotal, const ScatterTopoInfo& topo,
    const std::vector<u64>& dataSizePerLoop, u64 yCclBufferBaseOff, u64 yInCornerStep, uint32_t root)
{
    PushScatterYInnerCornerOneOsn(
        dataSliceLevely, osn, ySDataSize, xySOffset, ySOffset, omniPipeSplitSliceInfoListPerLoop,
        omniPipeSplitSliceInfoListTotal, topo, yCclBufferBaseOff, yInCornerStep, root);
    PushScatterYOverSameAxisOneOsn(
        dataSliceLevely, osn, xySDataSize, xSDataSize, ySDataSize, xySOffset, xSOffset, ySOffset,
        omniPipeSplitSliceInfoListPerLoop, omniPipeSplitSliceInfoListTotal, topo, dataSizePerLoop, yCclBufferBaseOff,
        yInCornerStep, innerStepNum, root);
}

void PushScatterYAllSteps(
    std::vector<StepSliceInfo>& dataSliceLevely, const ScatterTopoInfo& topo, const ScatterStepState& state,
    u64 zSDataSize[][MAX_STEP_NUM_SC], u64 xySDataSize[][MAX_STEP_NUM_SC],
    u64 xSDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC], u64 ySDataSize[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 zSOffset[][MAX_STEP_NUM_SC], u64 xySOffset[][MAX_STEP_NUM_SC], u64 xSOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    u64 ySOffset[][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& omniPipeSplitSliceInfoListPerLoop,
    const std::vector<OmniPipeSplitSliceInfo>& omniPipeSplitSliceInfoListTotal, uint32_t root, u64 xCclBufferBaseOff,
    u64 yCclBufferBaseOff, const std::vector<u64>& dataSizePerLoop)
{
    HCCL_DEBUG("[PushScatterYAllSteps] start push scatter Y Axis all steps");
    // 针对y轴，处理xy平面跨平面的对角节点
    if (state.xyCornerStep > 0) {
        if (topo.xB <= topo.yB) {
            // 当x轴带宽小于y轴带宽时，在y轴方向上yInCornerStep+1步发的跨平面的远的斜对角节点，最后一步发的是近的跨平面的对角
            for (u64 osn = 0; osn < state.xyCornerStep; osn++) {
                PushScatterYCornerLEOneOsn(
                    dataSliceLevely, osn, xySDataSize, xSDataSize, ySDataSize, xySOffset, xSOffset, ySOffset,
                    omniPipeSplitSliceInfoListPerLoop, omniPipeSplitSliceInfoListTotal, topo, dataSizePerLoop,
                    yCclBufferBaseOff, state.yInCornerStep, state.innerStepNum, root);
            }
        } else {
            // 当x轴带宽大于y轴带宽时，在y轴方向上只有第1步发的跨平面的远的斜对角节点，后续发的是近的跨平面的对角
            for (u64 osn = 0; osn < state.xyCornerStep; osn++) {
                PushScatterYCornerGtOneOsn(
                    dataSliceLevely, osn, state.innerStepNum, xySDataSize, xSDataSize, ySDataSize, xySOffset, xSOffset,
                    ySOffset, omniPipeSplitSliceInfoListPerLoop, omniPipeSplitSliceInfoListTotal, topo, dataSizePerLoop,
                    yCclBufferBaseOff, state.yInCornerStep, root);
            }
        }
    }
    bool sameZAxis = (topo.rootz != topo.zAxis);
    const std::vector<OmniPipeSplitSliceInfo>& omniPipeSplitSliceInfoList
        = sameZAxis ? omniPipeSplitSliceInfoListPerLoop : omniPipeSplitSliceInfoListTotal;
    // 处理本平面内的的对角节点的y轴
    if (topo.xB <= topo.yB) {
        for (u64 osn = state.xyCornerStep; osn < state.outerStepNum; osn++) {
            PushScatterYOuterLECornerOneOsn(
                dataSliceLevely, osn, ySDataSize, sameZAxis ? zSOffset : xySOffset, ySOffset,
                omniPipeSplitSliceInfoListPerLoop, omniPipeSplitSliceInfoList, topo, root, yCclBufferBaseOff,
                state.yInCornerStep);
            PushScatterYOuterLESameAxisOneOsn(
                dataSliceLevely, osn, xSDataSize, ySDataSize, sameZAxis ? zSOffset : xySOffset, xSOffset, ySOffset,
                omniPipeSplitSliceInfoListPerLoop, omniPipeSplitSliceInfoList, topo, root, yCclBufferBaseOff,
                state.yInCornerStep, state.innerStepNum);
        }
    } else {
        for (u64 osn = state.xyCornerStep; osn < state.outerStepNum; osn++) {
            PushScatterYOuterGTCornerOneOsn(
                dataSliceLevely, osn, ySDataSize, sameZAxis ? zSOffset : xySOffset, ySOffset,
                omniPipeSplitSliceInfoListPerLoop, omniPipeSplitSliceInfoList, topo, root, xCclBufferBaseOff,
                state.yInCornerStep);
            PushScatterYOuterGTSameAxisOneOsn(
                dataSliceLevely, osn, sameZAxis ? zSDataSize : xySDataSize, xSDataSize, ySDataSize,
                sameZAxis ? zSOffset : xySOffset, xSOffset, ySOffset, omniPipeSplitSliceInfoListPerLoop,
                omniPipeSplitSliceInfoList, topo, root, xCclBufferBaseOff, state.yInCornerStep, state.innerStepNum,
                dataSizePerLoop);
        }
    }
}

static void PushScatterZAllSteps(
    std::vector<StepSliceInfo>& dataSliceLevelz, u64 zSDataSize[][MAX_STEP_NUM_SC], u64 xySDataSize[][MAX_STEP_NUM_SC],
    u64 zSOffset[][MAX_STEP_NUM_SC], u64 xySOffset[][MAX_STEP_NUM_SC],
    const std::vector<OmniPipeSplitSliceInfo>& omniPipeSplitSliceInfoListPerLoop,
    const std::vector<OmniPipeSplitSliceInfo>& omniPipeSplitSliceInfoListTotal, const ScatterTopoInfo& topo,
    const ScatterStepState& state, u64 zCclBufferBaseOff, uint32_t root)
{
    HCCL_DEBUG(
        "[PushScatterZAllSteps] zCornerStep[%llu] outerStepNum[%llu] xyCornerStep[%llu] xInCornerStep[%llu] "
        "yInCornerStep[%llu] innerStepNum[%llu]",
        state.zCornerStep, state.outerStepNum, state.xyCornerStep, state.xInCornerStep, state.yInCornerStep,
        state.innerStepNum);
    if (topo.zB >= topo.xyB) {
        // 3D场景下，当z轴带宽大于等于xy轴打平带宽时，z轴最后一步才会发同Z轴数据，前面步骤都发的root的跨平面对角数据
        PushScatterZDiagStepsZgXY(
            dataSliceLevelz, zSDataSize, zSOffset, omniPipeSplitSliceInfoListPerLoop, omniPipeSplitSliceInfoListTotal,
            topo, zCclBufferBaseOff, state.zCornerStep, root);
        PushScatterZSameAxisStepsZgXY(
            dataSliceLevelz, zSDataSize, xySDataSize, zSOffset, xySOffset, omniPipeSplitSliceInfoListPerLoop,
            omniPipeSplitSliceInfoListTotal, topo, zCclBufferBaseOff, state.zCornerStep, state.outerStepNum, root);
    } else {
        // 3D场景下，当z轴带宽小于xy轴打平带宽时，z轴只有第一步发的root的跨平面对角数据
        PushScatterZDiagSteps(
            dataSliceLevelz, zSDataSize, zSOffset, omniPipeSplitSliceInfoListPerLoop, omniPipeSplitSliceInfoListTotal,
            topo, zCclBufferBaseOff, state.zCornerStep, root);
        PushScatterZSameAxisSteps(
            dataSliceLevelz, zSDataSize, xySDataSize, zSOffset, xySOffset, omniPipeSplitSliceInfoListPerLoop,
            omniPipeSplitSliceInfoListTotal, topo, zCclBufferBaseOff, state.zCornerStep, state.outerStepNum, root);
    }
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

    u64 zSDataSize[topo.rankSize][MAX_STEP_NUM_SC];
    u64 xySDataSize[topo.rankSize][MAX_STEP_NUM_SC];
    u64 xSDataSize[topo.rankSize][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC];
    u64 ySDataSize[topo.rankSize][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC];
    u64 zSOffset[topo.rankSize][MAX_STEP_NUM_SC];
    u64 xSOffset[topo.rankSize][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC];
    u64 ySOffset[topo.rankSize][MAX_STEP_NUM_SC][MAX_STEP_NUM_SC];
    u64 xySOffset[topo.rankSize][MAX_STEP_NUM_SC];
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
    PushScatterZAllSteps(
        dataSliceLevelz, zSDataSize, xySDataSize, zSOffset, xySOffset, omniPipeSplitSliceInfoListPerLoop,
        omniPipeSplitSliceInfoListTotal, topo, state, zCclBufferBaseOff, root);

    std::vector<StepSliceInfo> dataSliceLevelx;
    PushScatterXAllSteps(
        dataSliceLevelx, topo, state, zSDataSize, xySDataSize, xSDataSize, ySDataSize, zSOffset, xySOffset, xSOffset,
        ySOffset, omniPipeSplitSliceInfoListPerLoop, omniPipeSplitSliceInfoListTotal, root, xCclBufferBaseOff,
        yCclBufferBaseOff, omniPipeSliceParam.dataSizePerLoop);
    std::vector<StepSliceInfo> dataSliceLevely;
    PushScatterYAllSteps(
        dataSliceLevely, topo, state, zSDataSize, xySDataSize, xSDataSize, ySDataSize, zSOffset, xySOffset, xSOffset,
        ySOffset, omniPipeSplitSliceInfoListPerLoop, omniPipeSplitSliceInfoListTotal, root, xCclBufferBaseOff,
        yCclBufferBaseOff, omniPipeSliceParam.dataSizePerLoop);

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
