/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// V 变体算子参数解析链 + 运行期张量读取
// 从 hccl_op_utils.cc 拆出; 声明仍在 hccl_op_utils.h (同 namespace hccl, 调用方无感知)
#include <algorithm>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "exe_graph/runtime/runtime_tensor.h"
#include "graph/types.h"
#include "hccl/hccl_types.h"
#include "hccl/hcom.h"
#include "hccl_custom_op.h"
#include "hccl_op_utils.h"
#include "hccl_rt_utils.h"
#include "log.h"

namespace hccl {
// Tensor::GetSize() 是 TensorData buffer 实际字节数, 调用方可能分配大于 shape 的固定长度 buffer
// (如实测 shape=[2] 但 buffer 64 元素), 必须以 shape 为准 (标量 GetShapeSize()=1);
// buffer 小于 shape 声明则是数据不完整的真错误
static ge::graphStatus GetShapeDeclaredByteSize(const gert::Tensor* tensor, size_t& byteSize)
{
    const int64_t shapeSize = tensor->GetShape().GetStorageShape().GetShapeSize();
    // shapeSize < 0: 未知维 (UNKNOWN_DIM=-1 / UNKNOWN_DIM_NUM=-2), counts 必须静态形状 → 拒绝
    // shapeSize == 0: 空张量 (如未提供的可选输入 recv_displacements) 合法 → 读出空向量, 由调用方判定
    if (shapeSize < 0) {
        HCCL_ERROR("[GetShapeDeclaredByteSize] invalid shapeSize=%ld, counts shape must be static.", shapeSize);
        return ge::GRAPH_FAILED;
    }
    const ge::DataType dtype = tensor->GetDataType();
    size_t elemSize = 0;
    if (dtype == ge::DT_INT64) {
        elemSize = sizeof(int64_t);
    } else if (dtype == ge::DT_INT32) {
        elemSize = sizeof(int32_t);
    } else {
        HCCL_ERROR("[GetShapeDeclaredByteSize] unsupported dtype=%d, expect INT32|INT64.", static_cast<int>(dtype));
        return ge::GRAPH_FAILED;
    }
    byteSize = static_cast<size_t>(shapeSize) * elemSize;
    if (byteSize > tensor->GetSize()) {
        HCCL_ERROR(
            "[GetShapeDeclaredByteSize] shape declares %zu bytes > tensor buffer %zu bytes.", byteSize,
            tensor->GetSize());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// 按 dtype 从原始字节 buffer 转换为 u64 向量 (对应 vinputvec pass GetInt64VectorFromTensor 的双 dtype 兼容)
static ge::graphStatus
ConvertBytesToU64Vector(const void* data, size_t byteSize, ge::DataType dtype, std::vector<uint64_t>& values)
{
    values.clear();
    if (dtype == ge::DT_INT64) {
        const int64_t* ptr = reinterpret_cast<const int64_t*>(data);
        for (size_t i = 0; i < byteSize / sizeof(int64_t); i++) {
            // counts/displacements 语义上非负: 负值符号扩展成巨大 uint64 参与后续乘除是静默数据错误
            if (ptr[i] < 0) {
                HCCL_ERROR(
                    "[ConvertBytesToU64Vector] negative int64 value[%zu]=%ld, counts must be non-negative.", i, ptr[i]);
                return ge::GRAPH_FAILED;
            }
            values.push_back(static_cast<uint64_t>(ptr[i]));
        }
    } else if (dtype == ge::DT_INT32) {
        const int32_t* ptr = reinterpret_cast<const int32_t*>(data);
        for (size_t i = 0; i < byteSize / sizeof(int32_t); i++) {
            if (ptr[i] < 0) {
                HCCL_ERROR(
                    "[ConvertBytesToU64Vector] negative int32 value[%zu]=%d, counts must be non-negative.", i, ptr[i]);
                return ge::GRAPH_FAILED;
            }
            values.push_back(static_cast<uint64_t>(ptr[i]));
        }
    } else {
        HCCL_ERROR("[ConvertBytesToU64Vector] unsupported dtype=%d, expect INT32|INT64.", static_cast<int>(dtype));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReadHostU64Vector(const gert::Tensor* tensor, std::vector<uint64_t>& values)
{
    values.clear();
    if (tensor == nullptr || tensor->GetAddr() == nullptr) {
        HCCL_ERROR("[ReadHostU64Vector] input tensor or addr is null.");
        return ge::GRAPH_FAILED;
    }
    // host placement: kOnHost / kFollowing (tensor_data.h); kFollowing 的 GetAddr 已返回数据地址
    const gert::TensorPlacement placement = tensor->GetPlacement();
    if (placement != gert::kOnHost && placement != gert::kFollowing) {
        HCCL_ERROR("[ReadHostU64Vector] unexpected placement=%d, expect host.", static_cast<int>(placement));
        return ge::GRAPH_FAILED;
    }
    size_t byteSize = 0;
    if (GetShapeDeclaredByteSize(tensor, byteSize) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ConvertBytesToU64Vector(tensor->GetAddr(), byteSize, tensor->GetDataType(), values);
}

ge::graphStatus SyncStreamBeforeD2H(aclrtStream stream)
{
    if (stream == nullptr) {
        return ge::GRAPH_SUCCESS; // 无流 (host 直读场景): 无需同步
    }
    aclError aclRet = aclrtSynchronizeStream(stream);
    if (aclRet != ACL_SUCCESS) {
        // fail-fast: 同步失败继续 D2H 会读到未同步数据 (分配器 fill/旧值), 垃圾值可能
        // 通过 dtype/长度校验进入后续资源计算 — 显式失败优于静默错误值
        HCCL_ERROR("[SyncStreamBeforeD2H] aclrtSynchronizeStream failed, ret=%d.", static_cast<int>(aclRet));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReadU64VectorWithD2HNoSync(const gert::Tensor* tensor, std::vector<uint64_t>& values)
{
    values.clear();
    if (tensor == nullptr || tensor->GetAddr() == nullptr) {
        HCCL_ERROR("[ReadU64VectorWithD2H] input tensor or addr is null.");
        return ge::GRAPH_FAILED;
    }
    // host 直读失败分两类: placement 非 host (走下方 D2H) 与 shape/dtype 非法 (直接报错, 不可降级 D2H)
    const gert::TensorPlacement placement = tensor->GetPlacement();
    if (placement == gert::kOnHost || placement == gert::kFollowing) {
        return ReadHostU64Vector(tensor, values);
    }

    // device → host 同步拷贝: V 算子 counts 仅 rankSize×8B 量级, 同步开销可忽略
    size_t byteSize = 0;
    if (GetShapeDeclaredByteSize(tensor, byteSize) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (byteSize == 0) {
        return ge::GRAPH_SUCCESS; // 空张量: 无可读元素, 不发起 0 字节 D2H 拷贝
    }
    // 同步已由调用方完成 (ReadU64VectorWithD2H 包装 / 批量入口的 SyncStreamBeforeD2H)
    std::vector<uint8_t> hostBuf(byteSize);
    HcclResult ret = HcclRtMemSyncCopy(
        hostBuf.data(), byteSize, tensor->GetAddr(), byteSize, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[ReadU64VectorWithD2H] D2H copy failed, size=%zu, ret=%d.", byteSize, static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }
    return ConvertBytesToU64Vector(hostBuf.data(), byteSize, tensor->GetDataType(), values);
}

// 同步版入口 (单张量场景): 同步一次 → NoSync 读; 批量同流多张量场景调用方应改用
// SyncStreamBeforeD2H + NoSync 组合, 免去逐张量重复同步
ge::graphStatus ReadU64VectorWithD2H(const gert::Tensor* tensor, std::vector<uint64_t>& values, aclrtStream stream)
{
    HCCL_GE_CHK_RET(SyncStreamBeforeD2H(stream));
    return ReadU64VectorWithD2HNoSync(tensor, values);
}

std::string DumpU64TensorInfo(const gert::Tensor* tensor, const std::vector<uint64_t>& values)
{
    if (tensor == nullptr) {
        return "tensor=null";
    }
    std::ostringstream oss;
    const gert::Shape& shape = tensor->GetShape().GetStorageShape();
    oss << "placement=" << static_cast<int>(tensor->GetPlacement()) << " shape=[";
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        oss << (i == 0 ? "" : ",") << shape.GetDim(i);
    }
    oss << "] dtype=" << static_cast<int>(tensor->GetDataType()) << " elemCount=" << values.size() << " values=[";
    constexpr size_t DUMP_LIMIT = 16;
    const size_t dumpNum = std::min(values.size(), DUMP_LIMIT);
    for (size_t i = 0; i < dumpNum; i++) {
        oss << (i == 0 ? "" : ",") << values[i];
    }
    if (values.size() > DUMP_LIMIT) {
        oss << ",...";
    }
    oss << "]";
    return oss.str();
}

std::string BuildSendRecvBaseTag(uint32_t srTag, uint32_t srcRank, uint32_t destRank)
{
    return "SendRecv_" + std::to_string(srTag) + "_" + std::to_string(srcRank) + "_" + std::to_string(destRank);
}

gert::Tensor* SetEmptyOutputTensor(
    gert::EagerOpExecutionContext* ctx, size_t index, const gert::StorageShape& shape,
    const gert::StorageFormat& format, ge::DataType dtype)
{
    // GetOutputTensor 返回 const 指针; 对象本体非常量 (与 MallocOutputTensor 内部
    // GetOutputPointer<Tensor> 写回的是同一运行时对象), const_cast 后复刻其 SetTensorDesc 的字段写入
    gert::Tensor* outputTensor = const_cast<gert::Tensor*>(ctx->GetOutputTensor(index));
    if (outputTensor == nullptr) {
        HCCL_ERROR("[SetEmptyOutputTensor] output tensor %zu is null.", index);
        return nullptr;
    }
    outputTensor->MutableStorageShape() = shape.GetStorageShape();
    outputTensor->MutableOriginShape() = shape.GetOriginShape();
    outputTensor->SetStorageFormat(format.GetStorageFormat());
    outputTensor->SetOriginFormat(format.GetOriginFormat());
    outputTensor->SetDataType(dtype);
    HCCL_INFO(
        "[SetEmptyOutputTensor] output %zu set empty (dimNum=%zu), no memory allocated.", index,
        shape.GetStorageShape().GetDimNum());
    return outputTensor;
}

uint64_t CalcAlignedCount(uint64_t dataSize, uint64_t unitSize, uint64_t alignSize)
{
    if (unitSize == 0) {
        return 0; // 防御: 调用方须先完成 dataType 早校验 (RESERVED → 0)
    }
    if (alignSize == 0) {
        return 0; // 防御: alignSize 为除数, 0 值触发整型除零 UB
    }
    const uint64_t alignedSize = (dataSize + alignSize - 1) / alignSize * alignSize;
    return alignedSize / unitSize;
}

ge::graphStatus CalcOtherDimsFromShape(const gert::Shape& xShape, const char* opName, uint64_t& otherDims)
{
    if (xShape.GetDimNum() == 0) {
        HCCL_ERROR(
            "[%s][CalcOtherDimsFromShape] x tensor's first dim is illegal, dimNum=0.", opName ? opName : "unknown");
        return ge::GRAPH_FAILED;
    }
    otherDims = 1;
    for (size_t i = 1; i < xShape.GetDimNum(); i++) {
        const int64_t dim = xShape.GetDim(i);
        // 拦截零维与未知维 (UNKNOWN_DIM=-1/UNKNOWN_DIM_NUM=-2): 乘成 0 会静默清空 counts,
        // 负值 cast 到 uint64 变垃圾大数; 均在乘法前显式报错 (集群一致性要求 fail fast)
        if (dim <= 0) {
            HCCL_ERROR(
                "[%s][CalcOtherDimsFromShape] input shape dim[%zu]=%ld is zero or unknown.",
                opName ? opName : "unknown", i, dim);
            return ge::GRAPH_FAILED;
        }
        // 连乘溢出防护: otherDims *= dim 前检查回绕
        if (static_cast<uint64_t>(dim) > UINT64_MAX / otherDims) {
            HCCL_ERROR(
                "[%s][CalcOtherDimsFromShape] otherDims overflow at dim[%zu]=%ld, otherDims=%lu.",
                opName ? opName : "unknown", i, dim, otherDims);
            return ge::GRAPH_FAILED;
        }
        otherDims *= static_cast<uint64_t>(dim);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReadVCountPairWithCheck(
    const gert::Tensor* countTensor, const gert::Tensor* countsTensor, const char* opName, const char* countRole,
    const char* countsRole, uint32_t rankSize, std::vector<uint64_t>& rawCount, std::vector<uint64_t>& rawCounts,
    aclrtStream stream)
{
    const char* op = opName ? opName : "unknown";
    HCCL_GE_CHK_RET(ReadU64VectorWithD2H(countTensor, rawCount, stream));
    HCCL_GE_CHK_RET(ReadU64VectorWithD2H(countsTensor, rawCounts, stream));
    if (rawCount.empty() || rawCounts.size() != rankSize) {
        // 契约校验失败 (对应 proto recvCount.empty()/sendCounts 每 rank 一个的报错语义);
        // 打出调用方实际喂入的 counts 张量信息 (shape 超 rankSize 多为 padding/全局索引)
        HCCL_ERROR(
            "[%s][ReadVCountPairWithCheck] %s.size=%zu %s.size=%zu != rankSize=%u.", op, countRole, rawCount.size(),
            countsRole, rawCounts.size(), rankSize);
        HCCL_ERROR(
            "[%s][ReadVCountPairWithCheck] %s input: %s.", op, countRole,
            DumpU64TensorInfo(countTensor, rawCount).c_str());
        HCCL_ERROR(
            "[%s][ReadVCountPairWithCheck] %s input: %s.", op, countsRole,
            DumpU64TensorInfo(countsTensor, rawCounts).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReadOptionalVDispls(
    const gert::Tensor* displsTensor, const char* opName, const char* displsRole, uint32_t rankSize,
    std::vector<uint64_t>& rawDispls, bool& hasDispls, aclrtStream stream)
{
    hasDispls = false;
    if (displsTensor == nullptr) {
        return ge::GRAPH_SUCCESS; // 未接入 displs 输入: 回退按 counts 前缀和
    }
    // 张量存在但读取失败 (D2H 拷贝/形状非法) 不静默吞 —— 与"不存在"区分, fail fast
    HCCL_GE_CHK_RET(ReadU64VectorWithD2H(displsTensor, rawDispls, stream));
    if (rawDispls.size() == rankSize) {
        hasDispls = true;
    } else {
        HCCL_WARNING(
            "[%s][ReadOptionalVDispls] %s size=%zu != rankSize=%u, fallback to prefix-sum.",
            opName ? opName : "unknown", displsRole, rawDispls.size(), rankSize);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResolveVDispls(
    bool hasDispls, const std::vector<uint64_t>& rawDispls, const std::vector<uint64_t>& counts,
    std::vector<uint64_t>& displs)
{
    displs.clear();
    if (hasDispls) {
        displs = rawDispls;
        return ge::GRAPH_SUCCESS;
    }
    // 前缀和溢出防护: counts 每 rank 可达 2^63-1, 累加回绕产出静默错误 displs
    uint64_t sum = 0;
    for (uint64_t c : counts) {
        displs.push_back(sum);
        if (c > UINT64_MAX - sum) {
            displs.clear();
            HCCL_ERROR("[ResolveVDispls] prefix sum overflow at sum=%lu, c=%lu.", sum, c);
            return ge::GRAPH_FAILED;
        }
        sum += c;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConvertVCountsToElementUnit(
    const std::vector<uint64_t>& rawCounts, const std::vector<uint64_t>& dimDispls, uint64_t otherDims,
    std::vector<uint64_t>& counts, std::vector<uint64_t>& displs)
{
    counts.clear();
    displs.clear();
    if (dimDispls.size() < rawCounts.size()) {
        HCCL_ERROR(
            "[ConvertVCountsToElementUnit] dimDispls.size=%zu < rawCounts.size=%zu, caller assembly mismatch.",
            dimDispls.size(), rawCounts.size());
        return ge::GRAPH_FAILED;
    }
    if (otherDims == 0) {
        HCCL_ERROR("[ConvertVCountsToElementUnit] otherDims is 0, expect > 0.");
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < rawCounts.size(); i++) {
        // u64 溢出检查: ×otherDims 回绕会产生静默错误元素数
        if (rawCounts[i] > UINT64_MAX / otherDims || dimDispls[i] > UINT64_MAX / otherDims) {
            HCCL_ERROR(
                "[ConvertVCountsToElementUnit] overflow at i=%zu: counts=%lu displs=%lu otherDims=%lu.", i,
                rawCounts[i], dimDispls[i], otherDims);
            return ge::GRAPH_FAILED;
        }
        counts.push_back(rawCounts[i] * otherDims);
        displs.push_back(dimDispls[i] * otherDims);
    }
    return ge::GRAPH_SUCCESS;
}

gert::Tensor* AllocVOutputTensor(HcclOpState& st, const gert::Tensor* xTensor, int64_t outDim0)
{
    if (xTensor == nullptr) {
        HCCL_ERROR("[AllocVOutputTensor] xTensor is null.");
        return nullptr;
    }
    gert::StorageShape outShape = xTensor->GetShape();
    // SetDim(0, ...) 要求 x 至少一维: 标量 (dimNum == 0) 在 shape 容器上越界写
    if (outShape.GetStorageShape().GetDimNum() == 0 || outShape.GetOriginShape().GetDimNum() == 0) {
        HCCL_ERROR("[AllocVOutputTensor] x shape dimNum is 0, cannot set dim 0.");
        return nullptr;
    }
    outShape.MutableStorageShape().SetDim(0, outDim0);
    outShape.MutableOriginShape().SetDim(0, outDim0);
    // 空输出 (outDim0==0): GE eager Malloc 不支持 0 字节申请, 只定形不分配;
    // HCCL 零计数语义允许输出地址为空 (调用方按空操作/无校验路径放行)
    return (outDim0 == 0) ? SetEmptyOutputTensor(st.ctx, 0, outShape, xTensor->GetFormat(), xTensor->GetDataType()) :
                            st.ctx->MallocOutputTensor(0, outShape, xTensor->GetFormat(), xTensor->GetDataType());
}

ge::graphStatus GetVRankSizeByGroup(const char* group, const char* opName, uint32_t& rankSize)
{
    rankSize = 0;
    HcclResult ret = HcomGetRankSize(group, &rankSize);
    if (ret != HCCL_SUCCESS || rankSize == 0) {
        HCCL_ERROR(
            "[%s][GetVRankSizeByGroup] HcomGetRankSize failed for group '%s', ret=%d.", opName ? opName : "unknown",
            group ? group : "(null)", static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

uint32_t ParseAivCoreLimit(const char* valueStr)
{
    if (valueStr == nullptr || valueStr[0] == '\0') {
        return 0;
    }
    char* endPtr = nullptr;
    errno = 0;
    long value = std::strtol(valueStr, &endPtr, 10);
    if (endPtr == valueStr || *endPtr != '\0' || errno == ERANGE || value <= 0
        || value > static_cast<long>(UINT32_MAX)) {
        HCCL_WARNING("[ParseAivCoreLimit] invalid value '%s', keeping default.", valueStr);
        return 0;
    }
    return static_cast<uint32_t>(value);
}

uint32_t ParseVectorCoreCntFromHardwareInfo(const char* hardwareInfo)
{
    // 对照 InfoStore GetAivCoreLimit 的 hardwareInfo 回退 (hcom_op_utils.cc):
    // ge.hardwareInfo 格式 "key1:value1;vector_core_cnt:N;key3:value3", 取 N
    if (hardwareInfo == nullptr) {
        return 0;
    }
    std::string info(hardwareInfo);
    size_t pos = info.find("vector_core_cnt:");
    if (pos == std::string::npos) {
        return 0;
    }
    pos += strlen("vector_core_cnt:"); // 跳过 key
    size_t endPos = info.find(';', pos);
    if (endPos == std::string::npos) {
        endPos = info.size();
    }
    if (endPos <= pos) {
        return 0;
    }
    std::string valueStr = info.substr(pos, endPos - pos);
    if (valueStr.empty()) {
        return 0;
    }
    // InfoStore 对 hardwareInfo 分支: aivNum > 0 时覆盖, 否则保持缺省
    uint32_t value = ParseAivCoreLimit(valueStr.c_str());
    return value; // 0 = 保持缺省
}

std::string SummarizeVCounts(const std::vector<uint64_t>& counts, const std::vector<uint64_t>& displs)
{
    constexpr uint32_t MAX_LOG_RANK_NUM = 8;
    std::ostringstream oss;
    const size_t total = counts.size();
    // displs 为可选输入 (缺失时为空/短于 counts): 日志助手按两侧较短者为界, 不得越界读
    const size_t dumpNum = std::min({total, displs.size(), static_cast<size_t>(MAX_LOG_RANK_NUM)});
    for (size_t i = 0; i < dumpNum; i++) {
        oss << (i == 0 ? "" : "; ") << i << ":" << counts[i] << "/" << displs[i];
    }
    if (total > dumpNum) {
        oss << "; ... total " << total << " ranks";
    }
    return oss.str();
}

} // namespace hccl
