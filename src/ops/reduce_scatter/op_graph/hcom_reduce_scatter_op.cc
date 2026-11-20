/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 调用的 hcom.h (pkg_inc/hccl/hcom.h) 接口:
//   HcomGetCommHandleByGroup — 按 group 获取通信域 handle (hccl_custom_op.cc)
//   HcomGetCommCCLBufferSize — 获取 CCL buffer 大小 (CalcResourcesStatic)
//   HcomCreateCommCCLbuffer  — 创建通信域 CCL buffer (hccl_custom_op.cc GetOrCreateInCCLbuf/OutCCLbuf)
//   HcomGetInCCLbuffer       — 获取输入 CCL buffer (hccl_custom_op.cc GetOrCreateInCCLbuf)
//   HcomGetOutCCLbuffer      — 获取输出 CCL buffer (hccl_custom_op.cc GetOrCreateOutCCLbuf)
// 另调用 hccl_comm.h: HcclGetRankSize (CalcResourcesStatic)
// 另使用 hcom.h 常量: HCCL_KERNEL_OP_TYPE_REDUCESCATTER, HCCL_CMD_REDUCE_SCATTER
// 通信调用: 静态路径 HcclReduceScatterGraphMode — 本仓 src/ops/reduce_scatter/reduce_scatter.h, hccl.map
//           导出, extern "C" 前向声明, 无 root 参数 (区别于 HcclReduceGraphMode);
//           动态路径 HcclReduceScatter 单算子接口 — include/hccl.h (comm 句柄, 无 tag/streams/scratch);
//           AIV 算力上限统一用前向声明的 HcclSetAivCoreLimitGraphMode
// 公共工具 (CalcLoopMaxTime/BuildTagVec/CalcOnlineResources/CrackCleaner) 见 hccl_op_utils.h/.cc

#include "hcom_reduce_scatter_op.h"

#include <cstddef>

#include "acl/acl_rt.h"
#include "hccl.h" // 本仓源头 include/hccl.h (HcclReduceScatter 单算子接口), 非 CANN 安装产物
#include "hccl/hccl_comm.h"
#include "hccl/hcom.h"
#include "hccl_op_utils.h"
#include "hccl_rt_utils.h"
#include "log.h"

// 前向声明, 避免引入内部头文件依赖链 (签名对照本仓 src/ops/reduce_scatter/reduce_scatter.h)
extern "C" {
HcclResult HcclSetAivCoreLimitGraphMode(const char* group, uint32_t aivCoreLimit);
HcclResult HcclReduceScatterGraphMode(
    void* sendBuf, void* recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, const char* group,
    aclrtStream stream, const char* tag, void** streams, size_t streamCount, void* scratchMemAddr,
    uint64_t scratchMemSize);
}

namespace {
// SplitV 融合形态的 fusion attr 值 (fusion=2 即 HCOM_ATTR_FUSION_BY_FUSION_ID, 按 fusion_id 分组,
// 对照 GE op_fusion_base_pub.h); ExtractParamsStatic 与 InferShape 共用
constexpr int64_t FUSION_BY_FUSION_ID = 2;
} // namespace

ge::graphStatus HcomReduceScatterOp::ExtractCommonAttrs(hccl::HcclOpState& st, const char*& group, int64_t& rankSizeI)
{
    const gert::Tensor* firstInput = st.ctx->GetInputTensor(INPUT_INDEX);
    if (firstInput == nullptr) {
        HCCL_ERROR("[HcomReduceScatterOp][ExtractCommonAttrs] input tensor is null.");
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs* attrs = st.ctx->GetAttrs();
    if (attrs == nullptr) {
        HCCL_ERROR("[HcomReduceScatterOp][ExtractCommonAttrs] attrs is null.");
        return ge::GRAPH_FAILED;
    }
    const char* reduction = attrs->GetStr(ATTR_REDUCTION);
    group = attrs->GetStr(ATTR_GROUP);
    if (group == nullptr) {
        HCCL_ERROR("[HcomReduceScatterOp][ExtractCommonAttrs] group attr is null.");
        return ge::GRAPH_FAILED;
    }
    // attr rank_size (index 4): 编译期 rankSize, InferShape 与静态 count 计算用;
    // 运行期膨胀搬运/loopMaxTime 用通信域 rankSize (CalcResourcesStatic 存 st.rankSize), 不一致时 WARNING
    const int64_t* attrRankSize = attrs->GetInt(ATTR_RANK_SIZE);
    if (attrRankSize == nullptr) {
        HCCL_ERROR("[HcomReduceScatterOp][ExtractCommonAttrs] rank_size attr is missing.");
        return ge::GRAPH_FAILED;
    }
    if (*attrRankSize <= 0) {
        HCCL_ERROR("[HcomReduceScatterOp][ExtractCommonAttrs] invalid rank_size %ld.", *attrRankSize);
        return ge::GRAPH_FAILED;
    }
    rankSizeI = *attrRankSize;

    st.dataType = hccl::GeDataTypeToHccl(firstInput->GetDataType());
    if (st.dataType == HCCL_DATA_TYPE_RESERVED) {
        HCCL_ERROR(
            "[HcomReduceScatterOp][ExtractCommonAttrs] unsupported data type %d.",
            static_cast<int>(firstInput->GetDataType()));
        return ge::GRAPH_FAILED;
    }

    st.reduceOp = hccl::StringToReduceOp(reduction);
    if (st.reduceOp == HCCL_REDUCE_RESERVED) {
        HCCL_ERROR(
            "[HcomReduceScatterOp][ExtractCommonAttrs] unsupported reduction '%s'.", reduction ? reduction : "(null)");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

gert::Tensor*
HcomReduceScatterOp::AllocRsOutput(hccl::HcclOpState& st, size_t index, const gert::Tensor* inTensor, int64_t rankSize)
{
    // RS 输出 shape = 输入 dim0 / rank_size (对照 InferShape 契约); UNKNOWN_DIM 直通
    gert::StorageShape outShape = inTensor->GetShape();
    int64_t dim0 = outShape.GetStorageShape().GetDim(0);
    if (dim0 != ge::UNKNOWN_DIM) {
        if (dim0 % rankSize != 0) {
            HCCL_ERROR(
                "[HcomReduceScatterOp][AllocRsOutput] input[%zu] dim0=%ld not divisible by rank_size=%ld.", index, dim0,
                rankSize);
            return nullptr;
        }
        outShape.MutableStorageShape().SetDim(0, dim0 / rankSize);
        outShape.MutableOriginShape().SetDim(0, dim0 / rankSize);
    }
    // 空输出 (输入空 → 输出 0 元素): GE eager Malloc 不支持 0 字节申请, 只定形不分配;
    // HCCL 对 recvCount==0 提前成功返回 (HcclReduceScatter), 输出地址为空合法
    return (outShape.GetStorageShape().GetShapeSize() == 0) ?
               hccl::SetEmptyOutputTensor(st.ctx, index, outShape, inTensor->GetFormat(), inTensor->GetDataType()) :
               st.ctx->MallocOutputTensor(index, outShape, inTensor->GetFormat(), inTensor->GetDataType());
}

void HcomReduceScatterOp::AccumulateCrackInfo(uint64_t inputSize, uint64_t& crackTotalSize, hccl::HcclOpState& st)
{
    constexpr uint64_t ALIGNED_SIZE = 512;
    // 计算融合缝隙（crack）：对应 InfoStore Builder GetCrackParamsInfo (GE auto-padding → 512B 对齐)
    // crackOffset = 当前 tensor 在融合 buffer 中的偏移 + 实际数据大小
    // crackSize = 512B 对齐后大小 - 实际数据大小（padding 缝隙）
    uint64_t crackBlockSize = (inputSize + ALIGNED_SIZE - 1) / ALIGNED_SIZE * ALIGNED_SIZE;
    if (crackBlockSize > inputSize) {
        st.crackOffsets.push_back(crackTotalSize + inputSize);
        st.crackSizes.push_back(crackBlockSize - inputSize);
    }
    crackTotalSize += crackBlockSize;
}

bool HcomReduceScatterOp::IsFusedSplitvForm(hccl::HcclOpState& st, uint64_t attrRankSizeU, size_t& inputNum)
{
    // 预扫描输入个数 (GetInputTensor 越界返回 nullptr)
    inputNum = 0;
    while (st.ctx->GetInputTensor(inputNum) != nullptr) {
        inputNum++;
    }
    // fusion=2 (HCOM_ATTR_FUSION_BY_FUSION_ID) 且 rankSize > 1 且输入数为 rankSize 整数倍即融合节点;
    // rankSize==1 时 fusedOps==inputNum, CalcUniformSlotBytes 会读 inAddrs[inputNum] 尾后越界,
    // 与 InferShape 融合分支的 rankSize > 1 判定对齐
    const gert::RuntimeAttrs* attrs = st.ctx->GetAttrs();
    const int64_t* fusion = attrs->GetInt(ATTR_FUSION);
    return (fusion != nullptr) && (*fusion == FUSION_BY_FUSION_ID) && (attrRankSizeU > 1) && (inputNum > 1)
           && (inputNum % attrRankSizeU == 0);
}

ge::graphStatus HcomReduceScatterOp::ExtractParamsStatic(hccl::HcclOpState& st)
{
    const char* group = nullptr;
    int64_t rankSizeI = 0;
    HCCL_GE_CHK_RET(ExtractCommonAttrs(st, group, rankSizeI));
    const uint64_t attrRankSizeU = static_cast<uint64_t>(rankSizeI);

    const gert::RuntimeAttrs* attrs = st.ctx->GetAttrs();
    const int64_t* fusion = attrs->GetInt(ATTR_FUSION);
    const int64_t* fusionId = attrs->GetInt(ATTR_FUSION_ID);

    // SplitV 融合形态判定: GE RS fusion pass (fusion=2, 按 fusion_id 分组) 将 N 个原算子重写为
    // N×rankSize 输入/N 输出的融合节点; 其余形态走下方原逻辑
    size_t inputNum = 0;
    if (IsFusedSplitvForm(st, attrRankSizeU, inputNum)) {
        return ExtractParamsStaticFused(st, inputNum, attrRankSizeU, group);
    }

    // 遍历所有 input（非融合形态: 单输入或前端多 input 直连）。
    // count = 每个 input 的真实 per-rank 输出元素数累加 (= outTensor 元素数 = inputShapeSize/rankSize)。
    // 注意: 不复刻 InfoStore 的 +1024B padding —— 那是"旧路径输入 buffer 带 mindspore 补齐"的口径;
    // eager 路径喂给算法的是用户原始张量、无该补齐, 加 1024 会让 count 偏大 → 结果错。
    const uint64_t unitSize = hccl::GetHcclDataTypeSize(st.dataType);
    st.count = 0;
    uint64_t crackTotalSize = 0; // 512B 对齐累加 (crack 计算口径独立, 与 AllReduce 同款)
    size_t inputCount = 0;
    for (size_t i = 0;; i++) {
        const gert::Tensor* inTensor = st.ctx->GetInputTensor(i);
        if (inTensor == nullptr) {
            break;
        }
        inputCount++;

        gert::Tensor* outTensor = AllocRsOutput(st, i, inTensor, rankSizeI);
        if (outTensor == nullptr) {
            HCCL_ERROR("[HcomReduceScatterOp][ExtractParamsStatic] failed to allocate output tensor %zu.", i);
            return ge::GRAPH_FAILED;
        }
        uint64_t shapeSize = static_cast<uint64_t>(inTensor->GetShapeSize());
        uint64_t inputSize = shapeSize * unitSize;
        // count 累加真实 per-rank 输出元素数 (outTensor 元素数), 不加 InfoStore 的 +1024 padding
        st.count += static_cast<uint64_t>(outTensor->GetShapeSize());
        AccumulateCrackInfo(inputSize, crackTotalSize, st);
        st.inputPtrs.push_back(const_cast<void*>(inTensor->GetAddr()));
        st.outputPtrs.push_back(outTensor->GetAddr());

        HCCL_INFO(
            "[HcomReduceScatterOp][ExtractParamsStatic] input[%zu] addr=%p output addr=%p shapeSize=%lu "
            "inputSize=%lu(B).",
            i, inTensor->GetAddr(), outTensor->GetAddr(), shapeSize, inputSize);
    }

    if (inputCount == 0) {
        HCCL_ERROR("[HcomReduceScatterOp][ExtractParamsStatic] no input tensors.");
        return ge::GRAPH_FAILED;
    }

    // st.count 已在循环内按 outTensor 元素数累加 (per-rank output count 语义)
    st.group = group;
    st.stream = st.ctx->GetStream();
    LogExtractParamsStaticSummary(st, inputCount, attrRankSizeU, fusion, fusionId);
    return ge::GRAPH_SUCCESS;
}

void HcomReduceScatterOp::LogExtractParamsStaticSummary(
    const hccl::HcclOpState& st, size_t inputCount, uint64_t attrRankSizeU, const int64_t* fusion,
    const int64_t* fusionId)
{
    HCCL_INFO(
        "[HcomReduceScatterOp][ExtractParamsStatic] inputCount=%zu input[0]=%p output[0]=%p count=%lu "
        "dataType=%d reduceOp=%d group=%s stream=%p attrRankSize=%lu fusion=%ld fusionId=%ld.",
        inputCount, st.inputPtrs[0], st.outputPtrs[0], st.count, st.dataType, st.reduceOp,
        st.group ? st.group : "(null)", st.stream, attrRankSizeU, fusion ? *fusion : -1, fusionId ? *fusionId : -1);
}

ge::graphStatus
HcomReduceScatterOp::ValidateFusedSliceSizes(const std::vector<uint64_t>& inBytes, size_t fusedOps, uint64_t rankSize)
{
    // 校验 1: 同一 op 的各槽位切片 shape 一致 (input[r*N+op] ≡ input[op])
    for (size_t op = 0; op < fusedOps; op++) {
        for (uint64_t r = 1; r < rankSize; r++) {
            if (inBytes[static_cast<size_t>(r) * fusedOps + op] != inBytes[op]) {
                HCCL_ERROR(
                    "[HcomReduceScatterOp][ValidateFusedSliceSizes] slice size mismatch, op=%zu slot0=%lu(B) "
                    "slot%lu=%lu(B).",
                    op, inBytes[op], r, inBytes[static_cast<size_t>(r) * fusedOps + op]);
                return ge::GRAPH_FAILED;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus
HcomReduceScatterOp::ValidateFusedContiguity(const std::vector<void*>& inAddrs, const std::vector<uint64_t>& inBytes)
{
    // 校验 2: 地址连续升序不重叠
    for (size_t i = 0; i + 1 < inAddrs.size(); i++) {
        if (static_cast<char*>(inAddrs[i + 1]) < static_cast<char*>(inAddrs[i]) + inBytes[i]) {
            HCCL_ERROR(
                "[HcomReduceScatterOp][ValidateFusedContiguity] input[%zu] addr=%p overlaps previous end=%p "
                "- non-contiguous layout, not supported.",
                i + 1, inAddrs[i + 1], static_cast<void*>(static_cast<char*>(inAddrs[i]) + inBytes[i]));
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomReduceScatterOp::CalcUniformSlotBytes(
    const std::vector<void*>& inAddrs, size_t fusedOps, uint64_t rankSize, uint64_t unitSize, uint64_t& slotBytes)
{
    // 校验 3: 槽位均匀 (slotBytes = input[N] - input[0], 各槽位字节步长相等)
    slotBytes = static_cast<uint64_t>(static_cast<char*>(inAddrs[fusedOps]) - static_cast<char*>(inAddrs[0]));
    for (uint64_t r = 1; r + 1 < rankSize; r++) {
        uint64_t stride = static_cast<uint64_t>(
            static_cast<char*>(inAddrs[(r + 1) * fusedOps]) - static_cast<char*>(inAddrs[r * fusedOps]));
        if (stride != slotBytes) {
            HCCL_ERROR(
                "[HcomReduceScatterOp][CalcUniformSlotBytes] slot stride mismatch, slot0=%lu(B) "
                "slot%lu=%lu(B): slots not uniform, not supported.",
                slotBytes, r, stride);
            return ge::GRAPH_FAILED;
        }
    }
    if (slotBytes % unitSize != 0) {
        HCCL_ERROR(
            "[HcomReduceScatterOp][CalcUniformSlotBytes] slotBytes=%lu not aligned to unitSize=%lu.", slotBytes,
            unitSize);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus
HcomReduceScatterOp::AllocFusedOutputs(hccl::HcclOpState& st, size_t fusedOps, std::vector<void*>& outAddrs)
{
    // 分配 N 个输出: shape = 切片 shape (不再除 rankSize), 收集输出地址
    for (size_t op = 0; op < fusedOps; op++) {
        const gert::Tensor* inTensor = st.ctx->GetInputTensor(op);
        gert::Tensor* outTensor
            = st.ctx->MallocOutputTensor(op, inTensor->GetShape(), inTensor->GetFormat(), inTensor->GetDataType());
        if (outTensor == nullptr) {
            HCCL_ERROR("[HcomReduceScatterOp][AllocFusedOutputs] failed to allocate output %zu.", op);
            return ge::GRAPH_FAILED;
        }
        outAddrs[op] = outTensor->GetAddr();
        st.outputPtrs.push_back(outAddrs[op]);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomReduceScatterOp::ValidateFusedOutputMirror(
    const std::vector<void*>& inAddrs, const std::vector<void*>& outAddrs, size_t fusedOps)
{
    // 校验 4: 输出与输入槽位布局镜像 (各 op 输出偏移 == 槽内输入偏移)。
    // 容量说明: GE eager 的 MallocOutputTensor 按 RoundUp(shape×dt, 512) 分配 (kMemAlignment=512,
    // eager_op_execution_context.cc), 槽内 op 间距 == 输出槽距; 镜像校验通过即输出块容量
    // = Σ输出槽距 = Σ输入槽内间距 = slotBytes, 恰好覆盖通信写入范围 —— 故无需再做容量判定
    // (旧 outSpan 判据把"数据覆盖"误当容量, 小数据时 outSpan 必小于 slotBytes, 属误报)。
    for (size_t op = 1; op < fusedOps; op++) {
        int64_t inOff = static_cast<int64_t>(static_cast<char*>(inAddrs[op]) - static_cast<char*>(inAddrs[0]));
        int64_t outOff = static_cast<int64_t>(static_cast<char*>(outAddrs[op]) - static_cast<char*>(outAddrs[0]));
        if (inOff != outOff) {
            HCCL_ERROR(
                "[HcomReduceScatterOp][ValidateFusedOutputMirror] output[%zu] offset %ld != input offset %ld "
                "(output does not mirror slot layout), not supported.",
                op, outOff, inOff);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

void HcomReduceScatterOp::DeriveFusedCracks(
    const std::vector<void*>& inAddrs, const std::vector<uint64_t>& inBytes, size_t inputNum, uint64_t slotBytes,
    uint64_t rankSize, hccl::HcclOpState& st)
{
    // crack 由实际地址间隙推导: 相邻输入间隙 + 尾隙 (最后一个输入末尾到总覆盖末尾)
    for (size_t i = 0; i + 1 < inputNum; i++) {
        uint64_t gap
            = static_cast<uint64_t>(static_cast<char*>(inAddrs[i + 1]) - (static_cast<char*>(inAddrs[i]) + inBytes[i]));
        if (gap > 0) {
            st.crackOffsets.push_back(
                static_cast<uint64_t>((static_cast<char*>(inAddrs[i]) + inBytes[i]) - static_cast<char*>(inAddrs[0])));
            st.crackSizes.push_back(gap);
        }
    }
    const uint64_t totalCovered = slotBytes * rankSize;
    const uint64_t lastEnd = static_cast<uint64_t>(
        (static_cast<char*>(inAddrs[inputNum - 1]) + inBytes[inputNum - 1]) - static_cast<char*>(inAddrs[0]));
    if (lastEnd < totalCovered) {
        st.crackOffsets.push_back(lastEnd);
        st.crackSizes.push_back(totalCovered - lastEnd);
    }
}

ge::graphStatus HcomReduceScatterOp::ExtractParamsStaticFused(
    hccl::HcclOpState& st, size_t inputNum, uint64_t rankSize, const char* group)
{
    // SplitV 融合契约 (对照 GE AddReduceScatterNode): input[r*N + op] = op 的 rank r 切片 (槽位优先排列),
    // 输出 N 个, shape = 切片 shape (原 dim0/rankSize 已由 SplitV 完成, 此处不再除).
    // 扁平 RS 语义依赖的布局事实全部用实际地址推导验证, 不假设 GE 对齐规则 (区别于 InfoStore 的
    // +1024B padding 估计口径); 校验 1-4 见各 Validate*/Calc* 函数头注释。
    // count = 槽位元素数 (含槽内缝隙): rank r 的 chunk [r*slotBytes, (r+1)*slotBytes) 恰为槽位 r,
    // 缝隙由 crack 清零兜底; Direct/Refresh 主流程与非融合形态同构, 无需改动
    const size_t fusedOps = inputNum / static_cast<size_t>(rankSize);
    const uint64_t unitSize = hccl::GetHcclDataTypeSize(st.dataType);

    std::vector<void*> inAddrs(inputNum, nullptr);
    std::vector<uint64_t> inBytes(inputNum, 0);
    for (size_t i = 0; i < inputNum; i++) {
        const gert::Tensor* inTensor = st.ctx->GetInputTensor(i);
        if (inTensor == nullptr) { // 防御: 与预扫描不一致
            HCCL_ERROR("[HcomReduceScatterOp][ExtractParamsStaticFused] input[%zu] is null.", i);
            return ge::GRAPH_FAILED;
        }
        uint64_t shapeSize = static_cast<uint64_t>(inTensor->GetShapeSize());
        if (shapeSize == 0) {
            HCCL_ERROR(
                "[HcomReduceScatterOp][ExtractParamsStaticFused] input[%zu] is empty: SplitV integer-division "
                "truncation produced"
                "an empty slice (dim0 not divisible by rankSize), not supported.",
                i);
            return ge::GRAPH_FAILED;
        }
        inAddrs[i] = const_cast<void*>(inTensor->GetAddr());
        inBytes[i] = shapeSize * unitSize;
    }

    HCCL_GE_CHK_RET(ValidateFusedSliceSizes(inBytes, fusedOps, rankSize));
    HCCL_GE_CHK_RET(ValidateFusedContiguity(inAddrs, inBytes));
    uint64_t slotBytes = 0;
    HCCL_GE_CHK_RET(CalcUniformSlotBytes(inAddrs, fusedOps, rankSize, unitSize, slotBytes));

    // count = 槽位元素数 (含槽内缝隙, 与 GenerateTagVec/LaunchRefresh 的 count*rankSize 膨胀语义同构)
    st.count = slotBytes / unitSize;

    std::vector<void*> outAddrs(fusedOps, nullptr);
    HCCL_GE_CHK_RET(AllocFusedOutputs(st, fusedOps, outAddrs));
    HCCL_GE_CHK_RET(ValidateFusedOutputMirror(inAddrs, outAddrs, fusedOps));
    DeriveFusedCracks(inAddrs, inBytes, inputNum, slotBytes, rankSize, st);

    st.inputPtrs.assign(inAddrs.begin(), inAddrs.end());
    st.group = group;
    st.stream = st.ctx->GetStream();

    HCCL_INFO(
        "[HcomReduceScatterOp][ExtractParamsStaticFused] fusedOps=%zu rankSize=%lu inputNum=%zu slotBytes=%lu "
        "count=%lu crackNum=%zu input[0]=%p output[0]=%p group=%s stream=%p.",
        fusedOps, rankSize, inputNum, slotBytes, st.count, st.crackOffsets.size(), st.inputPtrs[0], st.outputPtrs[0],
        group ? group : "(null)", st.stream);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomReduceScatterOp::ExtractParamsDynamic(hccl::HcclOpState& st)
{
    // 动态路径 (lowering-style): count 取 outputShape (GetCountByShape 口径, 不 512B 对齐), 不计算 crack
    // 对照 lowering HcomReduceScatterKernelV2: 强制 inputNum == 1 (多 input 报错), count 从 outputShapes[0]
    const char* group = nullptr;
    int64_t rankSizeI = 0;
    HCCL_GE_CHK_RET(ExtractCommonAttrs(st, group, rankSizeI));

    // 遍历所有 input (统计 inputCount 用于单 input 校验); 输出 shape 变换与静态路径同契约
    // (dim0/rankSize, storage/origin 双 shape 同步; 只改 origin 会让 count 仍被放大 rankSize 倍)
    size_t inputCount = 0;
    for (size_t i = 0;; i++) {
        const gert::Tensor* inTensor = st.ctx->GetInputTensor(i);
        if (inTensor == nullptr) {
            break;
        }
        inputCount++;

        gert::Tensor* outTensor = AllocRsOutput(st, i, inTensor, rankSizeI);
        if (outTensor == nullptr) {
            HCCL_ERROR("[HcomReduceScatterOp][ExtractParamsDynamic] failed to allocate output tensor %zu.", i);
            return ge::GRAPH_FAILED;
        }

        uint64_t count = static_cast<uint64_t>(outTensor->GetShapeSize()); // count 取 outputShape
        uint64_t inCount = static_cast<uint64_t>(inTensor->GetShapeSize());
        HCCL_INFO(
            "[HcomReduceScatterOp][ExtractParamsDynamic] input[%zu] addr=%p output[%zu] addr=%p "
            "inCount=%lu outCount=%lu (RS output = input dim0/rankSize).",
            i, inTensor->GetAddr(), i, outTensor->GetAddr(), inCount, count);
        st.inputCounts.push_back(count);
        st.inputPtrs.push_back(const_cast<void*>(inTensor->GetAddr()));
        st.outputPtrs.push_back(outTensor->GetAddr());
    }

    if (inputCount == 0) {
        HCCL_ERROR("[HcomReduceScatterOp][ExtractParamsDynamic] no input tensors.");
        return ge::GRAPH_FAILED;
    }
    // 对照 lowering HcomReduceScatterKernelV2: inputNum != 1 报错 (RS 动态路径强制单 input)
    if (inputCount != 1) {
        HCCL_ERROR(
            "[HcomReduceScatterOp][ExtractParamsDynamic] inputNum=%zu != 1 (ReduceScatter dynamic path "
            "requires single input).",
            inputCount);
        return ge::GRAPH_FAILED;
    }

    st.count = st.inputCounts[0];
    st.group = group;
    st.stream = st.ctx->GetStream();

    HCCL_INFO(
        "[HcomReduceScatterOp][ExtractParamsDynamic] inputCount=%zu count(outputShape)=%lu dataType=%d "
        "reduceOp=%d group=%s stream=%p.",
        inputCount, st.count, st.dataType, st.reduceOp, st.group ? st.group : "(null)", st.stream);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomReduceScatterOp::CalcResourcesStatic(hccl::HcclOpState& st)
{
    uint32_t rankSize = 0;
    HcclResult ret = HcclGetRankSize(st.comm, &rankSize);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HcomReduceScatterOp][CalcResourcesStatic] HcclGetRankSize failed, ret=%d.", static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }
    st.rankSize = rankSize;

    // attr rank_size 与通信域 rankSize 一致性校验: 不一致仅告警, 膨胀搬运/loopMaxTime 用通信域值
    const gert::RuntimeAttrs* attrs = st.ctx->GetAttrs();
    if (attrs != nullptr) {
        const int64_t* attrRankSize = attrs->GetInt(ATTR_RANK_SIZE);
        if (attrRankSize != nullptr && static_cast<uint64_t>(*attrRankSize) != static_cast<uint64_t>(rankSize)) {
            HCCL_WARNING(
                "[HcomReduceScatterOp][CalcResourcesStatic] attr rank_size=%ld != communicator rankSize=%u, "
                "using communicator value for inflated copy and loopMaxTime.",
                *attrRankSize, rankSize);
        }
    }

    // [收敛] st.cclBuffSize (实际 in-CCL buffer 尺寸) 仅 Refresh 路径消费 (GenerateTagVec
    // 的 loopMaxTime 与 LaunchRefresh 的 maxCountPerLoop, 均门控 needRefresh) — Direct 路径直传用户
    // buffer 不经 CCL 中转, 不再创建 (原无条件 get-or-create 在 Direct 下白创建 in-CCL buffer)。
    // get-or-create 语义对照 InfoStore GetHcomInCCLbufferSize (查→建→重查)
    if (st.needRefresh) {
        void* commInputPtr = nullptr;
        HCCL_GE_CHK_RET(GetOrCreateInCCLbuf(st.group, commInputPtr, st.cclBuffSize));
        if (st.cclBuffSize == 0) {
            HCCL_ERROR(
                "[HcomReduceScatterOp][CalcResourcesStatic] CCL buffer size is 0 for group '%s'.",
                st.group ? st.group : "(null)");
            return ge::GRAPH_FAILED;
        }
    }

    // 在线计算输入: 环境推导 available size (对照旧路径 kernel_builder 的
    // GetCCLBufferAvailableSize, 与 AllReduce/Reduce 同源), 与实际 buffer 尺寸是两个来源
    uint64_t cclBufferAvailSize = 0;
    HCCL_GE_CHK_RET(hccl::GetCCLBufferAvailableSize(cclBufferAvailSize));
    return hccl::CalcOnlineResources(
        st, rankSize, cclBufferAvailSize, HCCL_KERNEL_OP_TYPE_REDUCESCATTER, HCCL_CMD_REDUCE_SCATTER);
}

ge::graphStatus HcomReduceScatterOp::GenerateTagVec(hccl::HcclOpState& st)
{
    // 对应 InfoStore GenerateOpTagFromTaskInfo: needRefresh 时算 loopMaxTime
    // RS 属组 2 公式 (膨胀, maxCountPerLoop = cclBuffSize / (rankSize*unitSize)), rankSize 用通信域值
    // (st.rankSize, CalcResourcesStatic 已存), 与 LaunchRefresh 循环同源, tag 数恒等于实际迭代数
    // Direct 路径用裸 tag (loopMaxTime=0); Refresh 路径算 loopMaxTime (tag 数必须等于实际迭代数)
    uint32_t loopMaxTime = 0;
    if (st.needRefresh) {
        uint64_t unitSize = hccl::GetHcclDataTypeSize(st.dataType);
        if (st.cclBuffSize / (st.rankSize * unitSize) == 0) {
            HCCL_ERROR(
                "[HcomReduceScatterOp][GenerateTagVec] cclBuffSize=%lu cannot hold one inflated element "
                "(rankSize=%u, unitSize=%lu).",
                st.cclBuffSize, st.rankSize, unitSize);
            return ge::GRAPH_FAILED;
        }
        HCCL_GE_CHK_RET(hccl::CalcLoopMaxTime(
            HCCL_KERNEL_OP_TYPE_REDUCESCATTER, st.count, unitSize, st.cclBuffSize, st.rankSize, loopMaxTime));
    }

    // 对应 InfoStore GetTagVectorInfo: loopMaxTime==0 → {tag}; loopMaxTime>0 → {tag + "_looptime_" + i}
    return hccl::BuildTagVec(HCCL_KERNEL_OP_TYPE_REDUCESCATTER, st.group, loopMaxTime, st.tags);
}

ge::graphStatus HcomReduceScatterOp::LaunchHcclOpStatic(hccl::HcclOpState& st)
{
    // 静态路径 (InfoStore-style): needRefresh 判定已收编基类 Execute, 此处只消费 st.needRefresh
    if (st.scratchMemSize > 0) {
        st.scratchMem = st.ctx->MallocWorkSpace(st.scratchMemSize);
        if (st.scratchMem == nullptr) {
            HCCL_ERROR(
                "[HcomReduceScatterOp][LaunchHcclOpStatic] MallocWorkSpace failed, size=%lu.", st.scratchMemSize);
            return ge::GRAPH_FAILED;
        }
    }

    HcclResult ret = HcclSetAivCoreLimitGraphMode(st.group, st.aivCoreLimit);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomReduceScatterOp][LaunchHcclOpStatic] HcclSetAivCoreLimitGraphMode failed, ret=%d.",
            static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }

    HCCL_GE_CHK_RET(GenerateTagVec(st));

    // attach 流申请 + 交付 (双流: graph/group 粒度, 对照旧 InfoStore SetAttachedStream 每 op 一次;
    // AIV 跳过)
    // follow 流申请 (按 st.streamNum, 复刻旧 main_stream_id 池化; AIV/流数 0 自动跳过)
    HCCL_GE_CHK_RET(RequestFollowStreams(st));
    HCCL_GE_CHK_RET(RequestAttachStream(st));
    HCCL_GE_CHK_RET(RegisterUnfoldStreamToModel(st));
    if (!st.needRefresh) {
        return LaunchDirect(st);
    }
    return LaunchRefresh(st);
}

ge::graphStatus HcomReduceScatterOp::LaunchHcclOpDynamic(hccl::HcclOpState& st)
{
    // 动态路径 (lowering-style): 对照 lowering HcomReduceScatterKernelV2 —
    // 单 input (ExtractParamsDynamic 已强制), count 取 outputShape, 直传用户 buffer, 无 crack;
    // 通信调用用单算子接口 HcclReduceScatter (comm 句柄, 无 tag/streams/scratch, 与 AllReduce/Reduce
    // 动态路径同款), AIV 算力上限用 GraphMode 版 HcclSetAivCoreLimitGraphMode
    HcclResult ret = HcclSetAivCoreLimitGraphMode(st.group, st.aivCoreLimit);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomReduceScatterOp][LaunchHcclOpDynamic] HcclSetAivCoreLimitGraphMode failed, ret=%d.",
            static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }

    HCCL_INFO(
        "[HcomReduceScatterOp][LaunchHcclOpDynamic] input=%p output=%p count=%lu (direct, user buffers).",
        st.inputPtrs[0], st.outputPtrs[0], st.count);

    ret = HcclReduceScatter(
        st.inputPtrs[0], st.outputPtrs[0], st.count, st.dataType, st.reduceOp, st.comm,
        static_cast<aclrtStream>(st.stream));
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomReduceScatterOp][LaunchHcclOpDynamic] HcclReduceScatter failed, ret=%d.", static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomReduceScatterOp::LaunchDirect(hccl::HcclOpState& st)
{
    // Direct 直传用户 buffer (不经 CCL buf 中转, 数据尺寸与 cclBuffSize 无关, 不设超限拦截)
    HCCL_INFO(
        "[HcomReduceScatterOp][LaunchDirect] input=%p output=%p count=%lu crackNum=%zu (non-refresh, user "
        "buffers directly).",
        st.inputPtrs[0], st.outputPtrs[0], st.count, st.crackOffsets.size());

    // Direct 路径: crack 地址 = 用户 buffer 基址 + crackOffset（inputOffset=0）
    // RS 的清零 gate 含 tensorNum > 1 (对照 InfoStore HcomReduceScatterLoop)
    if (st.inputPtrs.size() > 1) {
        std::vector<void*> crackAddrs;
        std::vector<uint64_t> crackSizes;
        for (size_t i = 0; i < st.crackOffsets.size(); i++) {
            crackAddrs.push_back(static_cast<char*>(st.inputPtrs[0]) + st.crackOffsets[i]);
            crackSizes.push_back(st.crackSizes[i]);
        }
        HCCL_GE_CHK_RET(crackCleaner_.Clean(st, crackAddrs, crackSizes));
    }

    HcclResult ret = HcclReduceScatterGraphMode(
        st.inputPtrs[0], st.outputPtrs[0], st.count, st.dataType, st.reduceOp, st.group,
        static_cast<aclrtStream>(st.stream), st.tags[0].c_str(),
        st.followStreams.empty() ? nullptr : st.followStreams.data(), st.followStreams.size(), st.scratchMem,
        st.scratchMemSize);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomReduceScatterOp][LaunchDirect] HcclReduceScatterGraphMode failed, ret=%d.", static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomReduceScatterOp::RefreshInputToCclOnce(
    const hccl::HcclOpState& st, void* devCclInputPtr, uint64_t commInputSize, void* devInputAddr, uint64_t curSize,
    uint64_t inputOffset, uint64_t loopTime)
{
    // 单迭代: 一次连续膨胀拷贝 curSize*rankSize (对照 InfoStore 非 legacy 单次分支)
    //   src = devInputAddr    (二级指针 → 解引用得用户 input 地址)
    //   dst = devCclInputPtr  (二级指针 → 解引用得 CCL buf 地址)
    HcclResult hrtRet = hccl::HcclRtMemAddrAsyncCopyWithOffset(
        devCclInputPtr, commInputSize, 0, devInputAddr, curSize * st.rankSize, inputOffset,
        static_cast<aclrtStream>(st.stream));
    if (hrtRet != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomReduceScatterOp][RefreshInputToCclOnce] D2D ADDR input->CCL (contiguous) failed, "
            "loopTime=%lu, ret=%d.",
            loopTime, static_cast<int>(hrtRet));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomReduceScatterOp::RefreshInputToCclPerRank(
    const hccl::HcclOpState& st, void* devCclInputPtr, uint64_t commInputSize, void* devInputAddr, uint64_t curSize,
    uint64_t unitSize, uint64_t inputOffset, uint64_t loopTime)
{
    // 多迭代: per-rank 分段拷贝 — 用户 input 按 rank 分块 (rank r 的 chunk 位于 r*count*unitSize),
    // 每迭代从各 rank chunk 取 [inputOffset, inputOffset+curSize) 段放入 CCL buf 的 r*curSize 处
    // (对照 InfoStore RefreshReduceScatterInputAddr legacy 分支: userInMemOffset = inputOffset +
    //  count*unitSize*i, indirectInCCLBufOffset = curSize*i, size = curSize)
    for (uint32_t r = 0; r < st.rankSize; r++) {
        HcclResult hrtRet = hccl::HcclRtMemAddrAsyncCopyWithOffset(
            devCclInputPtr, commInputSize, static_cast<uint64_t>(r) * curSize, devInputAddr, curSize,
            inputOffset + static_cast<uint64_t>(r) * st.count * unitSize, static_cast<aclrtStream>(st.stream));
        if (hrtRet != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[HcomReduceScatterOp][RefreshInputToCclPerRank] D2D ADDR input->CCL (per-rank r=%u) failed, "
                "loopTime=%lu, ret=%d.",
                r, loopTime, static_cast<int>(hrtRet));
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomReduceScatterOp::CleanFusedCracksForLoop(
    hccl::HcclOpState& st, void* commInputPtr, uint64_t curSize, uint64_t curCount, uint64_t countLeft,
    uint64_t unitSize, uint64_t inputOffset, bool singleIteration)
{
    // 缝隙清理 gate (对照 InfoStore HcomReduceScatterLoop): 仅末迭代 (curCount == countLeft)
    // + tensorNum > 1 + 非空 crack; 窗口按 RS 膨胀布局:
    //   单迭代: 窗口 [0, curSize*rankSize) 按逐字节布局直译 (地址 = commInputPtr + crackOffset,
    //     与 InfoStore 单迭代无过滤平移等价)
    //   多迭代: per-rank 映射 — crack 与 rank r 窗口 [inputOffset + r*count*unitSize, +curSize) 取交
    //     集分段清, 地址 = commInputPtr + r*curSize + (交集起点 - 窗口左界), 不越过窗口右界
    if (curCount != countLeft || st.inputPtrs.size() <= 1 || st.crackOffsets.empty()) {
        return ge::GRAPH_SUCCESS;
    }
    std::vector<void*> crackAddrs;
    std::vector<uint64_t> crackSizes;
    if (singleIteration) {
        for (size_t i = 0; i < st.crackOffsets.size(); i++) {
            if (st.crackOffsets[i] < curSize * st.rankSize) {
                crackAddrs.push_back(static_cast<char*>(commInputPtr) + st.crackOffsets[i]);
                crackSizes.push_back(st.crackSizes[i]);
            }
        }
    } else {
        // crack 与 rank r 窗口取交集分段清: 右界不截断会越过 (r+1)*curSize 清掉相邻 rank 已拷贝
        // 数据 (最末 rank 越过 CCL buf 界); 跨界 crack 的伸入段由相邻窗口的交集补齐
        for (size_t i = 0; i < st.crackOffsets.size(); i++) {
            uint64_t crackEnd = st.crackOffsets[i] + st.crackSizes[i];
            for (uint32_t r = 0; r < st.rankSize; r++) {
                uint64_t winLeft = static_cast<uint64_t>(r) * st.count * unitSize + inputOffset;
                uint64_t cleanStart = (st.crackOffsets[i] > winLeft) ? st.crackOffsets[i] : winLeft;
                uint64_t cleanEnd = (crackEnd < winLeft + curSize) ? crackEnd : winLeft + curSize;
                if (cleanStart < cleanEnd) {
                    crackAddrs.push_back(
                        static_cast<char*>(commInputPtr) + static_cast<uint64_t>(r) * curSize + (cleanStart - winLeft));
                    crackSizes.push_back(cleanEnd - cleanStart);
                }
            }
        }
    }
    // Clean 内部按需经 MallocWorkSpace 申请零缓冲; 空向量时 no-op
    return crackCleaner_.Clean(st, crackAddrs, crackSizes);
}

ge::graphStatus HcomReduceScatterOp::RefreshOutputFromCcl(
    const hccl::HcclOpState& st, void* devCclOutputPtr, void* devOutputAddr, uint64_t outputMaxSize, uint64_t curSize,
    uint64_t outputOffset, uint64_t loopTime)
{
    // ⑤ D2D ADDR 带偏移: CCL output buf → 用户 output (dstOffset=outputOffset, srcOffset=0)
    //   src = devCclOutputPtr  (二级指针 → 解引用得 CCL buf 地址)
    //   dst = devOutputAddr    (二级指针 → 解引用得用户 output 地址)
    // RS 输出为 per-rank 等长 curSize, 全 rank 回写 (无 Reduce 的 root-only 判定)
    HcclResult hrtRet = hccl::HcclRtMemAddrAsyncCopyWithOffset(
        devOutputAddr, outputMaxSize, outputOffset, devCclOutputPtr, curSize, 0, static_cast<aclrtStream>(st.stream));
    if (hrtRet != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomReduceScatterOp][RefreshOutputFromCcl] D2D ADDR CCL->output failed, loopTime=%lu, ret=%d.", loopTime,
            static_cast<int>(hrtRet));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomReduceScatterOp::GetRefreshCclBufs(
    const char* group, void*& commInputPtr, uint64_t& commInputSize, void*& commOutputPtr, uint64_t& commOutputSize)
{
    HCCL_GE_CHK_RET(GetOrCreateInCCLbuf(group, commInputPtr, commInputSize));
    uint64_t outCclBufCapacity = 0;
    HCCL_GE_CHK_RET(GetOrCreateOutCCLbuf(group, commOutputPtr, outCclBufCapacity));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus
HcomReduceScatterOp::ResolveRefreshLoopConfig(const hccl::HcclOpState& st, uint64_t unitSize, RsRefreshLoopConfig& cfg)
{
    // maxCountPerLoop 与 GenerateTagVec 的 loopMaxTime 同源 (st.cclBuffSize + 通信域 rankSize 单一来源)
    cfg.maxCountPerLoop = st.cclBuffSize / (st.rankSize * unitSize);
    if (cfg.maxCountPerLoop == 0) {
        HCCL_ERROR(
            "[HcomReduceScatterOp][ResolveRefreshLoopConfig] cclBuffSize=%lu cannot hold one inflated element "
            "(rankSize=%u, unitSize=%lu).",
            st.cclBuffSize, st.rankSize, unitSize);
        return ge::GRAPH_FAILED;
    }
    // ⑤ 侧 dstMax = 输出全量 count*unitSize (RS 输出为 per-rank 等长)
    cfg.outputMaxSize = st.count * unitSize;
    cfg.singleIteration = (st.count * unitSize * st.rankSize <= st.cclBuffSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomReduceScatterOp::SetupRefreshDevArgs(
    const hccl::HcclOpState& st, void* commInputPtr, void* commOutputPtr, RsRefreshDevArgs& out)
{
    HcomReduceScatterArgs args = {commInputPtr, commOutputPtr, st.inputPtrs[0], st.outputPtrs[0]};
    out.devArgs = st.ctx->MallocReadOnlyDevArgs(&args, sizeof(args));
    if (out.devArgs == nullptr || out.devArgs->args_data == nullptr) {
        HCCL_ERROR("[HcomReduceScatterOp][SetupRefreshDevArgs] MallocReadOnlyDevArgs failed.");
        return ge::GRAPH_FAILED;
    }
    // devArgs->args_data 是 device 侧 args 起始地址; 字段偏移用 offsetof 推导
    // (cclInputPtr=0, cclOutputPtr=8, inputAddr=16, outputAddr=24); D2D ADDR task 以字段
    // device 地址为二级指针间接寻址 (解引用得真实 buf/用户地址, UpdateHostArgs 刷新后自动生效)
    char* devArgsBase = static_cast<char*>(out.devArgs->args_data);
    out.devCclInputPtr = devArgsBase + offsetof(HcomReduceScatterArgs, cclInputPtr);
    out.devCclOutputPtr = devArgsBase + offsetof(HcomReduceScatterArgs, cclOutputPtr);
    out.devInputAddr = devArgsBase + offsetof(HcomReduceScatterArgs, inputAddr);
    out.devOutputAddr = devArgsBase + offsetof(HcomReduceScatterArgs, outputAddr);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomReduceScatterOp::CheckRefreshTagBounds(const hccl::HcclOpState& st, uint64_t loopTime)
{
    if (loopTime >= st.tags.size()) {
        HCCL_ERROR(
            "[HcomReduceScatterOp][CheckRefreshTagBounds] tagVec access out-of-bounds, loopTime=%lu "
            "tags.size=%zu.",
            loopTime, st.tags.size());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomReduceScatterOp::LaunchRefreshIterationKernel(
    hccl::HcclOpState& st, void* commInputPtr, void* commOutputPtr, uint64_t curCount, uint64_t loopTime)
{
    // ③ HcclReduceScatterGraphMode (每迭代独立 tag: _looptime_N; 无 root 参数)
    HcclResult ret = HcclReduceScatterGraphMode(
        commInputPtr, commOutputPtr, curCount, st.dataType, st.reduceOp, st.group, static_cast<aclrtStream>(st.stream),
        st.tags[loopTime].c_str(), st.followStreams.empty() ? nullptr : st.followStreams.data(),
        st.followStreams.size(), st.scratchMem, st.scratchMemSize);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomReduceScatterOp][LaunchRefreshIterationKernel] HcclReduceScatterGraphMode failed, "
            "loopTime=%lu, ret=%d.",
            loopTime, static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomReduceScatterOp::LaunchRefresh(hccl::HcclOpState& st)
{
    // 地址刷新路径: args struct + ArgsUpdater, GE 统一 H2D
    // 多迭代 loop: 结构对照 InfoStore HcomReduceScatterLoop, 超限数据分迭代执行 (input 膨胀系数 rankSize)
    // 与 AllReduce LaunchRefresh 的差异:
    //   ② 双形态: 单迭代 (膨胀总量不超 CCL buf) 一次连续膨胀拷贝 curSize*rankSize;
    //      多迭代 per-rank 分段拷贝 (RefreshInputToCclOnce / RefreshInputToCclPerRank)
    //   ③ HcclReduceScatterGraphMode 无 root 参数 (LaunchRefreshIterationKernel)
    //   ⑤ 等长 curSize 全 rank 回写 (RefreshOutputFromCcl)
    //   crack 清零: 末迭代 + tensorNum>1 gate, 窗口按 RS 膨胀布局 (CleanFusedCracksForLoop)
    // 对应 InfoStore RefreshInputAddr/RefreshOutputAddr 的 4 次拷贝:
    //   ①④ H2D (写 CCL buf 地址) → MallocReadOnlyDevArgs 由 GE 统一同步
    //   ②  D2D ADDR 带偏移 (用户数据 → CCL input buf) → 每迭代第 1 组拷贝
    //   ⑤  D2D ADDR 带偏移 (CCL output buf → 用户 output) → 每迭代第 2 组拷贝

    void* commInputPtr = nullptr;
    uint64_t commInputSize = 0;
    void* commOutputPtr = nullptr;
    uint64_t commOutputSize = 0;
    HCCL_GE_CHK_RET(GetRefreshCclBufs(st.group, commInputPtr, commInputSize, commOutputPtr, commOutputSize));

    const uint64_t unitSize = hccl::GetHcclDataTypeSize(st.dataType);
    RsRefreshLoopConfig cfg;
    HCCL_GE_CHK_RET(ResolveRefreshLoopConfig(st, unitSize, cfg));

    RsRefreshDevArgs devArgs;
    HCCL_GE_CHK_RET(SetupRefreshDevArgs(st, commInputPtr, commOutputPtr, devArgs));

    // 多迭代循环: countLeft 递减 curCount, inputOffset/outputOffset 步进 curSize, loopTime 对 tag 索引
    uint64_t curCount = 0;
    uint64_t countLeft = st.count;
    uint64_t inputOffset = 0;
    uint64_t outputOffset = 0;
    uint64_t loopTime = 0;
    for (; countLeft > 0; countLeft -= curCount) {
        HCCL_INFO(
            "[HcomReduceScatterOp][LaunchRefresh] loopTime=%lu inputOffset=%lu outputOffset=%lu countLeft=%lu "
            "maxCountPerLoop=%lu.",
            loopTime, inputOffset, outputOffset, countLeft, cfg.maxCountPerLoop);

        // tag 越界守卫 + 本迭代 count 计算 (大于 cclbuffer 时取最大支持 count, 膨胀系数 rankSize)
        HCCL_GE_CHK_RET(CheckRefreshTagBounds(st, loopTime));
        curCount = (countLeft > cfg.maxCountPerLoop) ? cfg.maxCountPerLoop : countLeft;
        uint64_t curSize = curCount * unitSize;

        // ② D2D ADDR 带偏移: 用户 input → CCL input buf (双形态)
        if (cfg.singleIteration) {
            HCCL_GE_CHK_RET(RefreshInputToCclOnce(
                st, devArgs.devCclInputPtr, commInputSize, devArgs.devInputAddr, curSize, inputOffset, loopTime));
        } else {
            HCCL_GE_CHK_RET(RefreshInputToCclPerRank(
                st, devArgs.devCclInputPtr, commInputSize, devArgs.devInputAddr, curSize, unitSize, inputOffset,
                loopTime));
        }

        // 缝隙清理: 仅末迭代 + tensorNum > 1 gate (窗口按 RS 膨胀布局)
        HCCL_GE_CHK_RET(CleanFusedCracksForLoop(
            st, commInputPtr, curSize, curCount, countLeft, unitSize, inputOffset, cfg.singleIteration));

        // ③ 每迭代 kernel launch (独立 tag)
        HCCL_GE_CHK_RET(LaunchRefreshIterationKernel(st, commInputPtr, commOutputPtr, curCount, loopTime));

        // ⑤ D2D ADDR 带偏移: CCL output buf → 用户 output (等长 curSize 全 rank 回写)
        HCCL_GE_CHK_RET(RefreshOutputFromCcl(
            st, devArgs.devCclOutputPtr, devArgs.devOutputAddr, cfg.outputMaxSize, curSize, outputOffset, loopTime));

        inputOffset += curSize;
        outputOffset += curSize;
        loopTime++;
    }

    HCCL_INFO(
        "[HcomReduceScatterOp][LaunchRefresh] devArgs=%p count=%lu maxCountPerLoop=%lu rankSize=%u "
        "cclInput=%p cclOutput=%p inputAddr=%p outputAddr=%p.",
        devArgs.devArgs->args_data, st.count, cfg.maxCountPerLoop, st.rankSize, commInputPtr, commOutputPtr,
        st.inputPtrs[0], st.outputPtrs[0]);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomReduceScatterOp::UpdateHostArgs(gert::UpdateArgsContext* ctx)
{
    if (ctx == nullptr) {
        HCCL_ERROR("[HcomReduceScatterOp][UpdateHostArgs] update args context is null.");
        return ge::GRAPH_FAILED;
    }
    // GE 在 sink 模型每次执行前调: 刷新 args struct 中的 inputAddr/outputAddr
    // GE 统一 H2D 同步 host → device, D2D ADDR task 自动 dereference 新地址
    const gert::Tensor* inputTensor = ctx->GetInputTensor(INPUT_INDEX);
    const gert::Tensor* outputTensor = ctx->GetOutputTensor(OUTPUT_INDEX);
    if (inputTensor == nullptr || outputTensor == nullptr) {
        HCCL_ERROR("[HcomReduceScatterOp][UpdateHostArgs] input=%p output=%p.", inputTensor, outputTensor);
        return ge::GRAPH_FAILED;
    }

    const gert::KernelArgs* hostArgs = ctx->GetKernelArgs(gert::Placement::kPlacementHost, 0);
    if (hostArgs == nullptr || hostArgs->args_data == nullptr) {
        HCCL_ERROR("[HcomReduceScatterOp][UpdateHostArgs] hostArgs is null.");
        return ge::GRAPH_FAILED;
    }

    auto* args = static_cast<HcomReduceScatterArgs*>(hostArgs->args_data);
    void* prevIn = args->inputAddr; // Execute 阶段填入或上次刷新的地址
    void* prevOut = args->outputAddr;
    args->inputAddr = const_cast<void*>(inputTensor->GetAddr());
    args->outputAddr = const_cast<void*>(outputTensor->GetAddr());

    // DFX: 地址刷新链路对照 — 旧值 = Execute 阶段获取的地址(或上次刷新值), 新值 = 本次回调从 ctx 获取的地址
    HCCL_INFO(
        "[HcomReduceScatterOp][UpdateHostArgs] input %p -> %p, output %p -> %p.", prevIn, args->inputAddr, prevOut,
        args->outputAddr);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomReduceScatterOp::InferShapeFused(gert::InferShapeContext* ctx, size_t outputNum, int64_t rankSize)
{
    // SplitV 融合分支: 对照 ExtractParamsStaticFused 的校验 1 — 校验同一 op 的各槽位切片 shape 一致
    for (size_t i = 0; i < outputNum; i++) {
        const gert::Shape* inputShape = ctx->GetInputShape(i);
        gert::Shape* outputShape = ctx->GetOutputShape(i);
        if ((inputShape == nullptr) || (outputShape == nullptr)) {
            HCCL_ERROR("[HcomReduceScatterOp][InferShapeFused] fused shape[%zu] is null.", i);
            return ge::GRAPH_FAILED;
        }
        for (int64_t r = 1; r < rankSize; r++) {
            const gert::Shape* sliceShape = ctx->GetInputShape(static_cast<size_t>(r) * outputNum + i);
            if ((sliceShape == nullptr) || (sliceShape->GetShapeSize() != inputShape->GetShapeSize())) {
                HCCL_ERROR(
                    "[HcomReduceScatterOp][InferShapeFused] fused slice shape mismatch, output[%zu] vs slot "
                    "%ld.",
                    i, r);
                return ge::GRAPH_FAILED;
            }
        }
        *outputShape = *inputShape;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomReduceScatterOp::GetInferRankSize(gert::InferShapeContext* ctx, int64_t& rankSize)
{
    // rank_size 取自 IR attr index 4 (编译期无通信域), 对照 reduce_scatter_proto.cc (rankSizeIndex = 4)
    const gert::RuntimeAttrs* attrs = ctx->GetAttrs();
    if (attrs == nullptr) {
        HCCL_ERROR("[HcomReduceScatterOp][GetInferRankSize] attrs is null.");
        return ge::GRAPH_FAILED;
    }
    const int64_t* rankSizePtr = attrs->GetInt(ATTR_RANK_SIZE);
    if (rankSizePtr == nullptr) {
        HCCL_ERROR("[HcomReduceScatterOp][GetInferRankSize] rank_size attr is missing.");
        return ge::GRAPH_FAILED;
    }
    if (*rankSizePtr <= 0) {
        HCCL_ERROR("[HcomReduceScatterOp][GetInferRankSize] invalid rank_size %ld.", *rankSizePtr);
        return ge::GRAPH_FAILED;
    }
    rankSize = *rankSizePtr;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomReduceScatterOp::InferShape(gert::InferShapeContext* ctx)
{
    if (ctx == nullptr) {
        HCCL_ERROR("[HcomReduceScatterOp][InferShape] infer shape context is null.");
        return ge::GRAPH_FAILED;
    }
    size_t inputNum = ctx->GetComputeNodeInputNum();
    size_t outputNum = ctx->GetComputeNodeOutputNum();
    int64_t rankSize = 0;
    HCCL_GE_CHK_RET(GetInferRankSize(ctx, rankSize));

    // SplitV 融合分支 (fusion=2, 对照 ExtractParamsStaticFused): N×rankSize 输入切片/N 输出,
    // 输出 shape = 切片 shape 直通 (dim0/rankSize 已由 GE SplitV 完成, 不再除)
    const gert::RuntimeAttrs* attrs = ctx->GetAttrs();
    const int64_t* fusionAttr = attrs->GetInt(ATTR_FUSION);
    if ((fusionAttr != nullptr) && (*fusionAttr == FUSION_BY_FUSION_ID) && (rankSize > 1) && (outputNum >= 1)
        && (inputNum == outputNum * static_cast<size_t>(rankSize))) {
        return InferShapeFused(ctx, outputNum, rankSize);
    }

    if (inputNum != outputNum) {
        HCCL_ERROR("[HcomReduceScatterOp][InferShape] inputNum=%zu != outputNum=%zu.", inputNum, outputNum);
        return ge::GRAPH_FAILED;
    }

    for (size_t i = 0; i < inputNum; i++) {
        const gert::Shape* inputShape = ctx->GetInputShape(i);
        if (inputShape == nullptr) {
            HCCL_ERROR("[HcomReduceScatterOp][InferShape] input shape[%zu] is null.", i);
            return ge::GRAPH_FAILED;
        }
        gert::Shape* outputShape = ctx->GetOutputShape(i);
        if (outputShape == nullptr) {
            HCCL_ERROR("[HcomReduceScatterOp][InferShape] output shape[%zu] is null.", i);
            return ge::GRAPH_FAILED;
        }

        // 对照 reduce_scatter_proto.cc: 0 维输入首维非法, 先于 GetDim(0) 拦截 (GetDim 不校验 dim_num_)
        if (inputShape->GetDimNum() == 0) {
            HCCL_ERROR("[HcomReduceScatterOp][InferShape] input tensor's first dim is illegal, expected: > 0, "
                       "actual: 0.");
            return ge::GRAPH_FAILED;
        }
        *outputShape = *inputShape;
        int64_t dim0 = inputShape->GetDim(0);
        if (dim0 == ge::UNKNOWN_DIM) {
            continue; // 动态 shape: output = input 直通 (对照 reduce_scatter_proto.cc 的直通分支)
        }
        if (dim0 % rankSize != 0) {
            HCCL_ERROR(
                "[HcomReduceScatterOp][InferShape] input dim0=%ld not divisible by rank_size=%ld.", dim0, rankSize);
            return ge::GRAPH_FAILED;
        }
        outputShape->SetDim(0, dim0 / rankSize);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomReduceScatterOp::InferDataType(gert::InferDataTypeContext* ctx)
{
    if (ctx == nullptr) {
        HCCL_ERROR("[HcomReduceScatterOp][InferDataType] infer datatype context is null.");
        return ge::GRAPH_FAILED;
    }
    size_t inputNum = ctx->GetComputeNodeInputNum();
    size_t outputNum = ctx->GetComputeNodeOutputNum();
    // 输出 dtype 与对应输入一致; 非融合 inputNum == outputNum, SplitV 融合 inputNum 为 outputNum 的
    // rankSize 倍 (输出取槽位 0 中各 op 切片的 dtype), 故按输出循环并校验整除关系
    if ((outputNum == 0) || (inputNum % outputNum != 0)) {
        HCCL_ERROR(
            "[HcomReduceScatterOp][InferDataType] inputNum=%zu not compatible with outputNum=%zu.", inputNum,
            outputNum);
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < outputNum; i++) {
        ge::DataType dataType = ctx->GetInputDataType(i);
        if (dataType == ge::DT_UNDEFINED) {
            HCCL_ERROR("[HcomReduceScatterOp][InferDataType] input dtype[%zu] is undefined.", i);
            return ge::GRAPH_FAILED;
        }
        ge::graphStatus ret = ctx->SetOutputDataType(i, dataType);
        if (ret != ge::GRAPH_SUCCESS) {
            HCCL_ERROR("[HcomReduceScatterOp][InferDataType] SetOutputDataType[%zu] failed.", i);
            return ret;
        }
    }
    return ge::GRAPH_SUCCESS;
}

const char* HcomReduceScatterOp::GetOpTypeName() const { return HCCL_KERNEL_OP_TYPE_REDUCESCATTER.c_str(); }

static ge::BaseCustomOp* CreateHcomReduceScatterOp() { return new HcomReduceScatterOp(); }
// 条件注册: 目标平台 (950/960) 且 GE/RTS 新接口在位才注册接管, 否则回退原 hccl_engine 路径
// (hccl_custom_op.h CreateSocGatedRegister: SoC 判定 + GE/RTS 接口在位判定)
static ge::CustomOpCreatorRegister* g_hcom_reduce_scatter_register
    = hccl::CreateSocGatedRegister(HCCL_KERNEL_OP_TYPE_REDUCESCATTER.c_str(), CreateHcomReduceScatterOp);
