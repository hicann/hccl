/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "omnipipe_gather_data_slice_calc.h"
#include "comm_engine_utils.h"

namespace ops_hccl {

// 初始化一个 StepSliceInfo：in/out 偏移为 0，hcclBuff 偏移为 cclBufOff
StepSliceInfo MakeGatherStep(u64 cclBufOff)
{
    StepSliceInfo s;
    BuffInfo bi;
    BuffInfoAssign(bi, 0, 0, cclBufOff);
    s.buffInfo = bi;
    return s;
}

// 把一个 rank 的一组 piece 推入 step 的六个字段；count 由 size/dataTypeSize 推导
void PushGatherRankEntry(StepSliceInfo &s, u64 dataTypeSize, u64 inStride, u64 outStride,
    std::vector<u64> sz, std::vector<u64> inOff, std::vector<u64> outOff)
{
    std::vector<u64> cnt;
    cnt.reserve(sz.size());
    for (u64 v : sz) {
        cnt.push_back(v / dataTypeSize);
    }
    s.stepSliceSize.push_back(std::move(sz));
    s.stepCount.push_back(std::move(cnt));
    s.inputOmniPipeSliceStride.push_back(std::move(inOff));
    s.outputOmniPipeSliceStride.push_back(std::move(outOff));
    s.stepInputSliceStride.push_back(inStride);
    s.stepOutputSliceStride.push_back(outStride);
}

// 分配 z/xy/x/y 的数据大小与偏移数组，零初始化
void InitGatherDataArrays(GatherSliceContext &ctx)
{
    std::vector<u64> d1(ctx.maxStepNum, 0);
    std::vector<std::vector<u64>> d2(ctx.maxStepNum, d1);
    ctx.zGDS.assign(ctx.rankSize, d1);
    ctx.xyGDS.assign(ctx.rankSize, d1);
    ctx.zGOff.assign(ctx.rankSize, d1);
    ctx.xyGOff.assign(ctx.rankSize, d1);
    ctx.xGDS.assign(ctx.rankSize, d2);
    ctx.yGDS.assign(ctx.rankSize, d2);
    ctx.xGOff.assign(ctx.rankSize, d2);
    ctx.yGOff.assign(ctx.rankSize, d2);
}

// 计算外层(z/xy)与内层(x/y)步数、每步数据大小及 2D 偏移，合并 xyB>zB / xyB<=zB 两分支
void CalcGatherStepDataAndOffset(GatherSliceContext &ctx)
{
    bool xyGtZ = (ctx.xyB > ctx.zB);
    u64 xyRankSize = ctx.xRankSize * ctx.yRankSize;
    for (u64 rs = 0; rs < ctx.rankSize; rs++) {
        if (xyGtZ) {
            ctx.outerStepNum = CalAllgatherDataSize2D(ctx.zGDS[rs].data(), ctx.xyGDS[rs].data(), ctx.zB, ctx.xyB,
                ctx.zRankSize, xyRankSize, ctx.perLoop[rs].size, ctx.maxStepNum, ctx.engine);
        } else {
            ctx.outerStepNum = CalAllgatherDataSize2D(ctx.xyGDS[rs].data(), ctx.zGDS[rs].data(), ctx.xyB, ctx.zB,
                xyRankSize, ctx.zRankSize, ctx.perLoop[rs].size, ctx.maxStepNum, ctx.engine);
        }
        for (u64 i = 0; i < ctx.outerStepNum; i++) {
            if (ctx.yGeX) {
                ctx.innerStepNum = CalAllgatherDataSize2D(ctx.xGDS[rs][i].data(), ctx.yGDS[rs][i].data(), ctx.xB,
                ctx.yB, ctx.xRankSize, ctx.yRankSize, ctx.xyGDS[rs][i], ctx.maxStepNum, ctx.engine);
            } else {
                ctx.innerStepNum = CalAllgatherDataSize2D(ctx.yGDS[rs][i].data(), ctx.xGDS[rs][i].data(), ctx.yB,
                ctx.xB, ctx.yRankSize, ctx.xRankSize, ctx.xyGDS[rs][i], ctx.maxStepNum, ctx.engine);
            }
        }
        if (ctx.yGeX) {
            if (ctx.innerStepNum > 1) { ctx.xInCornerStep = ctx.innerStepNum - 1; }
        } else {
            if (ctx.innerStepNum > 1) { ctx.yInCornerStep = ctx.innerStepNum - 1; }
        }
        if (xyGtZ) {
            CalAllgather2DOffset(ctx.zGOff[rs].data(), ctx.xyGOff[rs].data(), ctx.outerStepNum,
                ctx.zRankSize, xyRankSize, ctx.zGDS[rs].data(), ctx.xyGDS[rs].data());
        } else {
            CalAllgather2DOffset(ctx.xyGOff[rs].data(), ctx.zGOff[rs].data(), ctx.outerStepNum, xyRankSize,
                ctx.zRankSize, ctx.xyGDS[rs].data(), ctx.zGDS[rs].data());
        }
    }
    if (xyGtZ) {
        if (ctx.outerStepNum > 1) { ctx.zCornerStep = ctx.outerStepNum - 1; }
    } else {
        if (ctx.outerStepNum > 1) { ctx.xyCornerStep = ctx.outerStepNum - 1; }
    }
    HCCL_INFO("[CalcGatherOmniPipeSliceInfo] xInCornerStep=[%d],yInCornerStep=[%d],zConnerStep=[%d]",
        ctx.xInCornerStep, ctx.yInCornerStep, ctx.zCornerStep);
}

// 重算 x/y 轴 2D 偏移，合并 yB>=xB / else 两分支
void CalcGatherXY2DOffset(GatherSliceContext &ctx)
{
    for (u64 rs = 0; rs < ctx.rankSize; rs++) {
        for (u64 osn = 0; osn < ctx.outerStepNum; osn++) {
            if (ctx.yGeX) {
                CalAllgather2DOffset(ctx.xGOff[rs][osn].data(), ctx.yGOff[rs][osn].data(), ctx.innerStepNum,
                    ctx.xRankSize, ctx.yRankSize, ctx.xGDS[rs][osn].data(), ctx.yGDS[rs][osn].data());
            } else {
                CalAllgather2DOffset(ctx.yGOff[rs][osn].data(), ctx.xGOff[rs][osn].data(), ctx.innerStepNum,
                    ctx.yRankSize, ctx.xRankSize, ctx.yGDS[rs][osn].data(), ctx.xGDS[rs][osn].data());
            }
        }
    }
}

// 构造 z 轴 step：前 zCornerStep 步同轴，其后斜对角
std::vector<StepSliceInfo> BuildGatherZSteps(GatherSliceContext &ctx)
{
    std::vector<StepSliceInfo> dataSliceLevelz;
    for (u64 osn = 0; osn < static_cast<u64>(ctx.zCornerStep); osn++) {
        StepSliceInfo s = MakeGatherStep(ctx.zCclBufOff);
        for (u64 oneDid = 0; oneDid < ctx.zRankSize; oneDid++) {
            u64 pieceId = oneDid * ctx.xRankSize * ctx.yRankSize + ctx.yAxis * ctx.xRankSize + ctx.xAxis;
            u64 sliceSize = ctx.zGDS[pieceId][osn];
            u64 off = ctx.zGOff[pieceId][osn];
            u64 stride = ctx.total[pieceId].offset;
            PushGatherRankEntry(s, ctx.dataTypeSize, stride, stride, {sliceSize}, {off}, {off});
        }
        dataSliceLevelz.push_back(std::move(s));
    }
    for (u64 osn = static_cast<u64>(ctx.zCornerStep); osn < ctx.outerStepNum; osn++) {
        StepSliceInfo s = MakeGatherStep(ctx.zCclBufOff);
        for (u64 oneDid = 0; oneDid < ctx.zRankSize; oneDid++) {
            std::vector<u64> sz;
            std::vector<u64> inOff;
            std::vector<u64> outOff;
            for (u64 cds = 0; cds < ctx.xRankSize * ctx.yRankSize; cds++) {
                u64 curId = oneDid * ctx.xRankSize * ctx.yRankSize + cds;
                if (cds != ctx.yAxis * ctx.xRankSize + ctx.xAxis) {
                    u64 pieceId = curId;
                    u64 sliceSize = ctx.zGDS[pieceId][osn];
                    u64 off = ctx.zGOff[pieceId][osn] + ctx.total[pieceId].offset;
                    sz.push_back(sliceSize);
                    inOff.push_back(off);
                    outOff.push_back(off);
                }
            }
            PushGatherRankEntry(s, ctx.dataTypeSize, 0, 0, std::move(sz), std::move(inOff), std::move(outOff));
        }
        dataSliceLevelz.push_back(std::move(s));
    }
    return dataSliceLevelz;
}

// 为单个 oneDid 收集机内斜对角 pieces（X 遍历 yRankSize 过滤 yAxis；Y 遍历 xRankSize 过滤 xAxis）
void CollectGatherInnerCornerPieces(const GatherSliceContext &ctx, u64 osn, u64 isn, u64 oneDid,
    bool isX, std::vector<u64> &sz, std::vector<u64> &inOff, std::vector<u64> &outOff)
{
    const auto &gds = isX ? ctx.xGDS : ctx.yGDS;
    const auto &goff = isX ? ctx.xGOff : ctx.yGOff;
    u64 range = isX ? ctx.yRankSize : ctx.xRankSize;
    u64 selfCoord = isX ? ctx.yAxis : ctx.xAxis;
    u64 crossStride = isX ? ctx.xRankSize : 1;
    u64 oneDidStride = isX ? 1 : ctx.xRankSize;
    u64 base = ctx.zAxis * ctx.xRankSize * ctx.yRankSize;
    for (u64 cds = 0; cds < range; cds++) {
        if (cds != selfCoord) {
            u64 pieceId = base + cds * crossStride + oneDid * oneDidStride;
            u64 sliceSize = gds[pieceId][osn][isn];
            u64 off = goff[pieceId][osn][isn] + ctx.perLoop[pieceId].offset;
            sz.push_back(sliceSize);
            inOff.push_back(off);
            outOff.push_back(off);
        }
    }
}

// 为单个 oneDid 收集机间同轴 pieces（遍历 zRankSize 过滤 zAxis）
void CollectGatherOuterSameAxisPieces(const GatherSliceContext &ctx, u64 osn, u64 isn, u64 oneDid,
    bool isX, std::vector<u64> &sz, std::vector<u64> &inOff, std::vector<u64> &outOff)
{
    const auto &gds = isX ? ctx.xGDS : ctx.yGDS;
    const auto &goff = isX ? ctx.xGOff : ctx.yGOff;
    u64 xy = ctx.xRankSize * ctx.yRankSize;
    u64 fixedPart = isX ? (ctx.yAxis * ctx.xRankSize + oneDid) : (oneDid * ctx.xRankSize + ctx.xAxis);
    for (u64 osn2 = 0; osn2 < ctx.zRankSize; osn2++) {
        if (osn2 != ctx.zAxis) {
            u64 pieceId = osn2 * xy + fixedPart;
            u64 sliceSize = gds[pieceId][osn][isn];
            u64 off = ctx.xyGOff[pieceId][osn] + goff[pieceId][osn][isn] + ctx.total[pieceId].offset;
            sz.push_back(sliceSize);
            inOff.push_back(off);
            outOff.push_back(off);
        }
    }
}

// 为单个 oneDid 收集机间机内双重斜对角 pieces
void CollectGatherOuterCornerPieces(const GatherSliceContext &ctx, u64 osn, u64 isn, u64 oneDid,
    bool isX, std::vector<u64> &sz, std::vector<u64> &inOff, std::vector<u64> &outOff)
{
    const auto &gds = isX ? ctx.xGDS : ctx.yGDS;
    const auto &goff = isX ? ctx.xGOff : ctx.yGOff;
    u64 xy = ctx.xRankSize * ctx.yRankSize;
    u64 innerRange = isX ? ctx.yRankSize : ctx.xRankSize;
    u64 innerSelf = isX ? ctx.yAxis : ctx.xAxis;
    u64 innerStride = isX ? ctx.xRankSize : 1;
    u64 oneDidStride = isX ? 1 : ctx.xRankSize;
    for (u64 osn2 = 0; osn2 < ctx.zRankSize; osn2++) {
        if (osn2 != ctx.zAxis) {
            for (u64 cds = 0; cds < innerRange; cds++) {
                if (cds != innerSelf && innerRange > 1) {
                    u64 pieceId = osn2 * xy + cds * innerStride + oneDid * oneDidStride;
                    u64 sliceSize = gds[pieceId][osn][isn];
                    u64 off = ctx.xyGOff[pieceId][osn] + goff[pieceId][osn][isn] + ctx.total[pieceId].offset;
                    sz.push_back(sliceSize);
                    inOff.push_back(off);
                    outOff.push_back(off);
                }
            }
        }
    }
}

// x 轴机内 step（osn < xyCornerStep）：前 xInCornerStep 步同轴，其后机内斜对角
void BuildGatherXInnerSteps(GatherSliceContext &ctx, std::vector<StepSliceInfo> &out)
{
    for (u64 osn = 0; osn < static_cast<u64>(ctx.xyCornerStep); osn++) {
        for (u64 isn = 0; isn < static_cast<u64>(ctx.xInCornerStep); isn++) {
            StepSliceInfo s = MakeGatherStep(ctx.xCclBufOff);
            for (u64 oneDid = 0; oneDid < ctx.xRankSize; oneDid++) {
                u64 pieceId = ctx.zAxis * ctx.xRankSize * ctx.yRankSize + ctx.yAxis * ctx.xRankSize + oneDid;
                u64 sliceSize = ctx.xGDS[pieceId][osn][isn];
                u64 inOff = pieceId * ctx.dataSize[ctx.maxDataPieceId] + ctx.xGOff[pieceId][osn][isn];
                u64 outOff = pieceId * ctx.dataSizePerLoop[ctx.maxDataPieceId] + ctx.xGOff[pieceId][osn][isn];
                u64 stride = ctx.total[pieceId].offset;
                PushGatherRankEntry(s, ctx.dataTypeSize, stride, stride, {sliceSize}, {inOff}, {outOff});
            }
            out.push_back(std::move(s));
        }
        for (u64 isn = static_cast<u64>(ctx.xInCornerStep); isn < ctx.innerStepNum; isn++) {
            StepSliceInfo s = MakeGatherStep(ctx.xCclBufOff);
            for (u64 oneDid = 0; oneDid < ctx.xRankSize; oneDid++) {
                std::vector<u64> sz;
                std::vector<u64> inOff;
                std::vector<u64> outOff;
                CollectGatherInnerCornerPieces(ctx, osn, isn, oneDid, true, sz, inOff, outOff);
                PushGatherRankEntry(s, ctx.dataTypeSize, 0, 0, std::move(sz), std::move(inOff), std::move(outOff));
            }
            out.push_back(std::move(s));
        }
    }
}

// x 轴机间 step（osn >= xyCornerStep）：同轴片 + 机间机内双重斜对角
void BuildGatherXOuterSteps(GatherSliceContext &ctx, std::vector<StepSliceInfo> &out)
{
    for (u64 osn = static_cast<u64>(ctx.xyCornerStep); osn < ctx.outerStepNum; osn++) {
        for (u64 isn = 0; isn < static_cast<u64>(ctx.xInCornerStep); isn++) {
            StepSliceInfo s = MakeGatherStep(ctx.xCclBufOff);
            for (u64 oneDid = 0; oneDid < ctx.xRankSize; oneDid++) {
                std::vector<u64> sz;
                std::vector<u64> inOff;
                std::vector<u64> outOff;
                CollectGatherOuterSameAxisPieces(ctx, osn, isn, oneDid, true, sz, inOff, outOff);
                PushGatherRankEntry(s, ctx.dataTypeSize, 0, 0, std::move(sz), std::move(inOff), std::move(outOff));
            }
            out.push_back(std::move(s));
        }
        for (u64 isn = static_cast<u64>(ctx.xInCornerStep); isn < ctx.innerStepNum; isn++) {
            StepSliceInfo s = MakeGatherStep(ctx.xCclBufOff);
            for (u64 oneDid = 0; oneDid < ctx.xRankSize; oneDid++) {
                std::vector<u64> sz;
                std::vector<u64> inOff;
                std::vector<u64> outOff;
                CollectGatherOuterCornerPieces(ctx, osn, isn, oneDid, true, sz, inOff, outOff);
                PushGatherRankEntry(s, ctx.dataTypeSize, 0, 0, std::move(sz), std::move(inOff), std::move(outOff));
            }
            out.push_back(std::move(s));
        }
    }
}

// y 轴机内 step（osn < xyCornerStep）：前 yInCornerStep 步同轴，其后机内斜对角
void BuildGatherYInnerSteps(GatherSliceContext &ctx, std::vector<StepSliceInfo> &out)
{
    for (u64 osn = 0; osn < static_cast<u64>(ctx.xyCornerStep); osn++) {
        for (u64 isn = 0; isn < static_cast<u64>(ctx.yInCornerStep); isn++) {
            StepSliceInfo s = MakeGatherStep(ctx.yCclBufOff);
            for (u64 oneDid = 0; oneDid < ctx.yRankSize; oneDid++) {
                u64 pieceId = ctx.zAxis * ctx.xRankSize * ctx.yRankSize + oneDid * ctx.xRankSize + ctx.xAxis;
                u64 sliceSize = ctx.yGDS[pieceId][osn][isn];
                u64 inOff = pieceId * ctx.dataSize[ctx.maxDataPieceId] + ctx.yGOff[pieceId][osn][isn];
                u64 outOff = pieceId * ctx.dataSizePerLoop[ctx.maxDataPieceId] + ctx.yGOff[pieceId][osn][isn];
                u64 stride = ctx.total[pieceId].offset;
                PushGatherRankEntry(s, ctx.dataTypeSize, stride, stride, {sliceSize}, {inOff}, {outOff});
            }
            out.push_back(std::move(s));
        }
        for (u64 isn = static_cast<u64>(ctx.yInCornerStep); isn < ctx.innerStepNum; isn++) {
            StepSliceInfo s = MakeGatherStep(ctx.yCclBufOff);
            for (u64 oneDid = 0; oneDid < ctx.yRankSize; oneDid++) {
                std::vector<u64> sz;
                std::vector<u64> inOff;
                std::vector<u64> outOff;
                CollectGatherInnerCornerPieces(ctx, osn, isn, oneDid, false, sz, inOff, outOff);
                PushGatherRankEntry(s, ctx.dataTypeSize, 0, 0, std::move(sz), std::move(inOff), std::move(outOff));
            }
            out.push_back(std::move(s));
        }
    }
}

// y 轴机间 step（osn >= xyCornerStep）：同轴片 + 机间机内双重斜对角
void BuildGatherYOuterSteps(GatherSliceContext &ctx, std::vector<StepSliceInfo> &out)
{
    for (u64 osn = static_cast<u64>(ctx.xyCornerStep); osn < ctx.outerStepNum; osn++) {
        for (u64 isn = 0; isn < static_cast<u64>(ctx.yInCornerStep); isn++) {
            StepSliceInfo s = MakeGatherStep(ctx.yCclBufOff);
            for (u64 oneDid = 0; oneDid < ctx.yRankSize; oneDid++) {
                std::vector<u64> sz;
                std::vector<u64> inOff;
                std::vector<u64> outOff;
                CollectGatherOuterSameAxisPieces(ctx, osn, isn, oneDid, false, sz, inOff, outOff);
                PushGatherRankEntry(s, ctx.dataTypeSize, 0, 0, std::move(sz), std::move(inOff), std::move(outOff));
            }
            out.push_back(std::move(s));
        }
        for (u64 isn = static_cast<u64>(ctx.yInCornerStep); isn < ctx.innerStepNum; isn++) {
            StepSliceInfo s = MakeGatherStep(ctx.yCclBufOff);
            for (u64 oneDid = 0; oneDid < ctx.yRankSize; oneDid++) {
                std::vector<u64> sz;
                std::vector<u64> inOff;
                std::vector<u64> outOff;
                CollectGatherOuterCornerPieces(ctx, osn, isn, oneDid, false, sz, inOff, outOff);
                PushGatherRankEntry(s, ctx.dataTypeSize, 0, 0, std::move(sz), std::move(inOff), std::move(outOff));
            }
            out.push_back(std::move(s));
        }
    }
}

OmniPipeSliceInfo CalcGatherOmniPipeSliceInfo(OmniPipeSliceParam &omniPipeSliceParam)
{
    HCCL_INFO("[CalcGatherOmniPipeSliceInfo] Run start");
    GatherSliceContext ctx;
    ctx.maxStepNum = MAX_STEP_NUM;
    ctx.xRankSize = omniPipeSliceParam.levelRankSize[OmniPipeLevel::OMNIPIPE_LEVEL0];
    ctx.yRankSize = omniPipeSliceParam.levelRankSize[OmniPipeLevel::OMNIPIPE_LEVEL1];
    ctx.zRankSize = omniPipeSliceParam.levelRankSize[OmniPipeLevel::OMNIPIPE_LEVEL2];
    ctx.rankSize = ctx.xRankSize * ctx.yRankSize * ctx.zRankSize;
    ctx.xB = omniPipeSliceParam.endpointAttrBw[OmniPipeLevel::OMNIPIPE_LEVEL0] * 1.0;
    ctx.yB = omniPipeSliceParam.endpointAttrBw[OmniPipeLevel::OMNIPIPE_LEVEL1] * 1.0;
    ctx.zB = omniPipeSliceParam.endpointAttrBw[OmniPipeLevel::OMNIPIPE_LEVEL2] * 1.0;
    ctx.yGeX = (ctx.yB >= ctx.xB);
    ctx.xyB = ctx.yGeX ? CalcBandwidth2D(ctx.xB, ctx.yB, ctx.xRankSize, ctx.yRankSize, ctx.maxStepNum)
                       : CalcBandwidth2D(ctx.yB, ctx.xB, ctx.yRankSize, ctx.xRankSize, ctx.maxStepNum);
    ctx.xAxis = omniPipeSliceParam.levelRankId[OmniPipeLevel::OMNIPIPE_LEVEL0];
    ctx.yAxis = omniPipeSliceParam.levelRankId[OmniPipeLevel::OMNIPIPE_LEVEL1];
    ctx.zAxis = omniPipeSliceParam.levelRankId[OmniPipeLevel::OMNIPIPE_LEVEL2];
    u64 rankid = ctx.xAxis + ctx.yAxis * ctx.xRankSize + ctx.zAxis * ctx.xRankSize * ctx.yRankSize;
    HCCL_INFO("[CalcGatherOmniPipeSliceInfo] xRankSize=[%llu],yRankSize=[%llu],zRankSize=[%llu],",
        ctx.xRankSize, ctx.yRankSize, ctx.zRankSize);
    HCCL_INFO("[CalcGatherOmniPipeSliceInfo] xB=[%f],yB=[%f],zB=[%f],xyB=[%f]", ctx.xB, ctx.yB, ctx.zB, ctx.xyB);
    HCCL_INFO("[CalcGatherOmniPipeSliceInfo] xAxis=[%llu],yAxis=[%llu],zAxis=[%llu],rankid=[%llu]",
        ctx.xAxis, ctx.yAxis, ctx.zAxis, rankid);
    ctx.engine = omniPipeSliceParam.engine;
    ctx.dataTypeSize = omniPipeSliceParam.dataTypeSize;
    ctx.dataSize = omniPipeSliceParam.dataWholeSize;
    ctx.dataSizePerLoop = omniPipeSliceParam.dataSizePerLoop;
    ctx.perLoop = OmniPipeSplitSliceInfoListAssign(ctx.dataSizePerLoop, ctx.rankSize, ctx.dataTypeSize);
    ctx.total = OmniPipeSplitSliceInfoListAssign(ctx.dataSize, ctx.rankSize, ctx.dataTypeSize);
    for (size_t i = 0; i < ctx.dataSize.size(); i++) {
        if (ctx.dataSize[ctx.maxDataPieceId] < ctx.dataSize[i]) {
            ctx.maxDataPieceId = i;
        }
    }
    InitGatherDataArrays(ctx);
    CalcGatherStepDataAndOffset(ctx);
    ctx.yCclBufOff = ctx.xCclBufOff + ctx.dataSizePerLoop[ctx.maxDataPieceId] * ctx.xRankSize;
    ctx.zCclBufOff = ctx.yCclBufOff + ctx.dataSizePerLoop[ctx.maxDataPieceId] * ctx.yRankSize;
    std::vector<StepSliceInfo> dataSliceLevelz = BuildGatherZSteps(ctx);
    CalcGatherXY2DOffset(ctx);
    std::vector<StepSliceInfo> dataSliceLevelx;
    BuildGatherXInnerSteps(ctx, dataSliceLevelx);
    BuildGatherXOuterSteps(ctx, dataSliceLevelx);
    std::vector<StepSliceInfo> dataSliceLevely;
    BuildGatherYInnerSteps(ctx, dataSliceLevely);
    BuildGatherYOuterSteps(ctx, dataSliceLevely);
    return {std::move(dataSliceLevelx), std::move(dataSliceLevely), std::move(dataSliceLevelz)};
}

}  // namespace ops_hccl