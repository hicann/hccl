/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file hcom_vinputvec_set_pass.cc
 * \brief 图优化 Pass：为需要 vInputVec 属性的算子节点设置该属性，
 *        补偿 V2 InferShape 中无法调用 SetAttr 的缺失。
 */

#include <vector>
#include <cstdint>
#include <algorithm>
#include "ge/fusion/pass/fusion_base_pass.h"
#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "graph/graph.h"
#include "graph/gnode.h"
#include "graph/tensor.h"
#include "graph/ascend_string.h"
#include "log.h"

namespace {
constexpr int32_t RECV_COUNT_INPUT_INDEX = 1;
constexpr int32_t SEND_COUNTS_INPUT_INDEX = 2;
constexpr int32_t SEND_DISPLS_INPUT_INDEX = 3;
constexpr int32_t X_INPUT_INDEX = 0;

// HcomAllGatherV input indices: x=0, send_count=1, recv_counts=2, recv_displacements=3(optional)
constexpr int32_t AGV_SEND_COUNT_INPUT_INDEX = 1;
constexpr int32_t AGV_RECV_COUNTS_INPUT_INDEX = 2;
constexpr int32_t AGV_RECV_DISPLS_INPUT_INDEX = 3;

// HcomAllToAllV input indices: send_data=0, send_counts=1, send_displacements=2, recv_counts=3, recv_displacements=4
constexpr int32_t A2AV_SEND_COUNTS_INPUT_INDEX = 1;
constexpr int32_t A2AV_SEND_DISPLS_INPUT_INDEX = 2;
constexpr int32_t A2AV_RECV_COUNTS_INPUT_INDEX = 3;
constexpr int32_t A2AV_RECV_DISPLS_INPUT_INDEX = 4;

ge::graphStatus GetInt64VectorFromTensor(const ge::Tensor& tensor, std::vector<int64_t>& values)
{
    auto dtype = tensor.GetDataType();
    const uint8_t* rawData = tensor.GetData();
    if (rawData == nullptr) {
        return ge::GRAPH_FAILED;
    }
    size_t byteSize = tensor.GetSize();
    if (dtype == ge::DT_INT64) {
        size_t elemCount = byteSize / sizeof(int64_t);
        const int64_t* ptr = reinterpret_cast<const int64_t*>(rawData);
        for (size_t i = 0; i < elemCount; ++i) {
            values.push_back(ptr[i]);
        }
    } else if (dtype == ge::DT_INT32) {
        size_t elemCount = byteSize / sizeof(int32_t);
        const int32_t* ptr = reinterpret_cast<const int32_t*>(rawData);
        for (size_t i = 0; i < elemCount; ++i) {
            values.push_back(static_cast<int64_t>(ptr[i]));
        }
    } else {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus WriteInt64VectorToTensor(ge::Tensor& tensor, const std::vector<int64_t>& values)
{
    size_t byteSize = values.size() * sizeof(int64_t);
    auto ret = tensor.SetData(reinterpret_cast<const uint8_t*>(values.data()), byteSize);
    return (ret == ge::GRAPH_SUCCESS) ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}

HcclResult GetOtherDims(ge::GNode& node, const ge::AscendString& nodeName, int64_t& otherDims)
{
    ge::TensorDesc xDesc;
    if (node.GetInputDesc(X_INPUT_INDEX, xDesc) != ge::GRAPH_SUCCESS) {
        HCCL_ERROR("[GetOtherDims] node[%s] get input x desc failed.", nodeName.GetString());
        return HCCL_E_INTERNAL;
    }
    auto inDims = xDesc.GetShape().GetDims();
    if (inDims.empty()) {
        HCCL_ERROR(
            "[GetOtherDims] node[%s] input tensor's first dim is illegal, expected: > 0, actual: 0.",
            nodeName.GetString());
        return HCCL_E_INTERNAL;
    }
    otherDims = 1;
    for (size_t i = 1; i < inDims.size(); i++) {
        otherDims *= inDims[i];
    }
    return HCCL_SUCCESS;
}

void MultiplyByOtherDims(std::vector<int64_t>& values, int64_t otherDims)
{
    std::transform(values.begin(), values.end(), values.begin(), [otherDims](int64_t num) {
        return num * otherDims;
    });
}

HcclResult WriteTensorsAndSetAttr(
    ge::GNode& node, const ge::AscendString& nodeName, const char* funcName, const char* attrName,
    std::vector<ge::Tensor>& tensors, const std::vector<std::vector<int64_t>*>& valueVecs)
{
    for (size_t i = 0; i < tensors.size(); i++) {
        if (WriteInt64VectorToTensor(tensors[i], *valueVecs[i]) != ge::GRAPH_SUCCESS) {
            HCCL_ERROR("[%s] node[%s] write tensor[%zu] failed.", funcName, nodeName.GetString(), i);
            return HCCL_E_INTERNAL;
        }
    }
    ge::AscendString attr(attrName);
    if (node.SetAttr(attr, tensors) != ge::GRAPH_SUCCESS) {
        HCCL_ERROR("[%s] node[%s] set %s failed.", funcName, nodeName.GetString(), attrName);
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO("[%s] node[%s] set %s success.", funcName, nodeName.GetString(), attrName);
    return HCCL_SUCCESS;
}

// 检查指定输入的 peer 节点是否为 Const/Constant, 避免 GetInputConstData 对非 const 输入打印错误日志
bool IsInputConst(const ge::GNode& node, int32_t inputIndex)
{
    auto inNodePair = node.GetInDataNodesAndPortIndexs(inputIndex);
    if (inNodePair.first == nullptr) {
        return false;
    }
    ge::AscendString inNodeType;
    inNodePair.first->GetType(inNodeType);
    return (inNodeType == "Const" || inNodeType == "Constant");
}

// 检查必选输入是否全部为 Const, 有非 const 则返回 false
bool CheckRequiredInputsConst(const ge::GNode& node, const std::vector<int32_t>& indices)
{
    for (auto idx : indices) {
        if (!IsInputConst(node, idx)) {
            return false;
        }
    }
    return true;
}

// 解析 optional displacement: 有 const 数据则用真值, 无则按 counts 连续计算
HcclResult CalcDisplacements(
    const ge::Tensor& displsTensor, bool hasDispls, const std::vector<int64_t>& counts, std::vector<int64_t>& displs,
    int64_t otherDims, const ge::AscendString& nodeName, const char* funcName)
{
    if (hasDispls) {
        if (GetInt64VectorFromTensor(displsTensor, displs) != ge::GRAPH_SUCCESS) {
            HCCL_ERROR("[%s] node[%s] parse displacements failed.", funcName, nodeName.GetString());
            return HCCL_E_INTERNAL;
        }
        MultiplyByOtherDims(displs, otherDims);
    } else {
        int64_t tmpCount = 0;
        for (size_t i = 0; i < counts.size(); i++) {
            displs.push_back(tmpCount);
            tmpCount += counts[i];
        }
    }
    return HCCL_SUCCESS;
}

HcclResult SetReduceScatterVVInputVec(ge::GNode& node)
{
    if (node.HasAttr("vInputVec")) {
        return HCCL_SUCCESS;
    }
    ge::AscendString nodeName;
    node.GetName(nodeName);
    if (!CheckRequiredInputsConst(node, {RECV_COUNT_INPUT_INDEX, SEND_COUNTS_INPUT_INDEX})) {
        HCCL_INFO("[SetReduceScatterVVInputVec] node[%s] required input not const, skip.", nodeName.GetString());
        return HCCL_SUCCESS;
    }
    ge::Tensor recvCountTensor;
    ge::Tensor sendCountsTensor;
    ge::Tensor sendDisplsTensor;
    if ((node.GetInputConstData(RECV_COUNT_INPUT_INDEX, recvCountTensor) != ge::GRAPH_SUCCESS)
        || (node.GetInputConstData(SEND_COUNTS_INPUT_INDEX, sendCountsTensor) != ge::GRAPH_SUCCESS)) {
        HCCL_INFO("[SetReduceScatterVVInputVec] node[%s] const data not available, skip.", nodeName.GetString());
        return HCCL_SUCCESS;
    }
    bool hasSendDispls = IsInputConst(node, SEND_DISPLS_INPUT_INDEX);
    if (hasSendDispls) {
        hasSendDispls = (node.GetInputConstData(SEND_DISPLS_INPUT_INDEX, sendDisplsTensor) == ge::GRAPH_SUCCESS);
    }
    std::vector<int64_t> recvCount;
    std::vector<int64_t> sendCounts;
    std::vector<int64_t> sendDispls;
    if (GetInt64VectorFromTensor(recvCountTensor, recvCount) != ge::GRAPH_SUCCESS
        || GetInt64VectorFromTensor(sendCountsTensor, sendCounts) != ge::GRAPH_SUCCESS) {
        HCCL_ERROR("[SetReduceScatterVVInputVec] node[%s] parse counts failed.", nodeName.GetString());
        return HCCL_E_INTERNAL;
    }
    int64_t otherDims = 1;
    CHK_RET(GetOtherDims(node, nodeName, otherDims));
    MultiplyByOtherDims(recvCount, otherDims);
    MultiplyByOtherDims(sendCounts, otherDims);
    CHK_RET(CalcDisplacements(
        sendDisplsTensor, hasSendDispls, sendCounts, sendDispls, otherDims, nodeName, "SetReduceScatterVVInputVec"));
    std::vector<ge::Tensor> tensors = {recvCountTensor, sendCountsTensor, sendDisplsTensor};
    std::vector<std::vector<int64_t>*> valueVecs = {&recvCount, &sendCounts, &sendDispls};
    return WriteTensorsAndSetAttr(node, nodeName, "SetReduceScatterVVInputVec", "vInputVec", tensors, valueVecs);
}

HcclResult SetAllGatherVVInputVec(ge::GNode& node)
{
    if (node.HasAttr("vInputVec")) {
        return HCCL_SUCCESS;
    }
    ge::AscendString nodeName;
    node.GetName(nodeName);
    if (!CheckRequiredInputsConst(node, {AGV_SEND_COUNT_INPUT_INDEX, AGV_RECV_COUNTS_INPUT_INDEX})) {
        HCCL_INFO("[SetAllGatherVVInputVec] node[%s] required input not const, skip.", nodeName.GetString());
        return HCCL_SUCCESS;
    }
    ge::Tensor sendCountTensor;
    ge::Tensor recvCountsTensor;
    ge::Tensor recvDispsTensor;
    if ((node.GetInputConstData(AGV_SEND_COUNT_INPUT_INDEX, sendCountTensor) != ge::GRAPH_SUCCESS)
        || (node.GetInputConstData(AGV_RECV_COUNTS_INPUT_INDEX, recvCountsTensor) != ge::GRAPH_SUCCESS)) {
        HCCL_INFO("[SetAllGatherVVInputVec] node[%s] const data not available, skip.", nodeName.GetString());
        return HCCL_SUCCESS;
    }
    bool hasRecvDispls = IsInputConst(node, AGV_RECV_DISPLS_INPUT_INDEX);
    if (hasRecvDispls) {
        hasRecvDispls = (node.GetInputConstData(AGV_RECV_DISPLS_INPUT_INDEX, recvDispsTensor) == ge::GRAPH_SUCCESS);
    }
    std::vector<int64_t> sendCount;
    std::vector<int64_t> recvCounts;
    std::vector<int64_t> recvDisp;
    if (GetInt64VectorFromTensor(sendCountTensor, sendCount) != ge::GRAPH_SUCCESS
        || GetInt64VectorFromTensor(recvCountsTensor, recvCounts) != ge::GRAPH_SUCCESS) {
        HCCL_ERROR("[SetAllGatherVVInputVec] node[%s] parse counts failed.", nodeName.GetString());
        return HCCL_E_INTERNAL;
    }
    if (sendCount.empty()) {
        HCCL_ERROR("[SetAllGatherVVInputVec] node[%s] send_count is empty.", nodeName.GetString());
        return HCCL_E_INTERNAL;
    }
    int64_t otherDims = 1;
    CHK_RET(GetOtherDims(node, nodeName, otherDims));
    MultiplyByOtherDims(recvCounts, otherDims);
    sendCount[0] *= otherDims;
    CHK_RET(CalcDisplacements(
        recvDispsTensor, hasRecvDispls, recvCounts, recvDisp, otherDims, nodeName, "SetAllGatherVVInputVec"));
    std::vector<ge::Tensor> tensors = {recvCountsTensor, recvDispsTensor, sendCountTensor};
    std::vector<std::vector<int64_t>*> valueVecs = {&recvCounts, &recvDisp, &sendCount};
    return WriteTensorsAndSetAttr(node, nodeName, "SetAllGatherVVInputVec", "vInputVec", tensors, valueVecs);
}

HcclResult SetAllToAllVInputVec(ge::GNode& node)
{
    if (node.HasAttr("alltoallvInputVec")) {
        return HCCL_SUCCESS;
    }

    ge::AscendString nodeName;
    node.GetName(nodeName);

    if (!CheckRequiredInputsConst(
            node, {A2AV_SEND_COUNTS_INPUT_INDEX, A2AV_SEND_DISPLS_INPUT_INDEX, A2AV_RECV_COUNTS_INPUT_INDEX,
                   A2AV_RECV_DISPLS_INPUT_INDEX})) {
        HCCL_INFO("[SetAllToAllVInputVec] node[%s] required input not const, skip.", nodeName.GetString());
        return HCCL_SUCCESS;
    }

    ge::Tensor sendCountsTensor;
    ge::Tensor sendDispTensor;
    ge::Tensor recvCountsTensor;
    ge::Tensor recvDispTensor;
    if ((node.GetInputConstData(A2AV_SEND_COUNTS_INPUT_INDEX, sendCountsTensor) != ge::GRAPH_SUCCESS)
        || (node.GetInputConstData(A2AV_SEND_DISPLS_INPUT_INDEX, sendDispTensor) != ge::GRAPH_SUCCESS)
        || (node.GetInputConstData(A2AV_RECV_COUNTS_INPUT_INDEX, recvCountsTensor) != ge::GRAPH_SUCCESS)
        || (node.GetInputConstData(A2AV_RECV_DISPLS_INPUT_INDEX, recvDispTensor) != ge::GRAPH_SUCCESS)) {
        HCCL_INFO("[SetAllToAllVInputVec] node[%s] const data not available, skip.", nodeName.GetString());
        return HCCL_SUCCESS;
    }

    std::vector<ge::Tensor> alltoallvInputVec;
    alltoallvInputVec.push_back(sendCountsTensor);
    alltoallvInputVec.push_back(sendDispTensor);
    alltoallvInputVec.push_back(recvCountsTensor);
    alltoallvInputVec.push_back(recvDispTensor);
    ge::AscendString attrName("alltoallvInputVec");
    if (node.SetAttr(attrName, alltoallvInputVec) != ge::GRAPH_SUCCESS) {
        HCCL_ERROR("[SetAllToAllVInputVec] node[%s] set alltoallvInputVec failed.", nodeName.GetString());
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO("[SetAllToAllVInputVec] node[%s] set alltoallvInputVec success.", nodeName.GetString());
    return HCCL_SUCCESS;
}
} // namespace

ge::Status HcomVInputVecSetPassFn(ge::GraphPtr& graph, ge::CustomPassContext& ctx)
{
    if (graph == nullptr) {
        HCCL_ERROR("[HcomVInputVecSetPassFn] graph is null.");
        return ge::FAILED;
    }

    auto nodes = graph->GetAllNodes();
    for (auto& node : nodes) {
        ge::AscendString nodeType;
        if (node.GetType(nodeType) != ge::GRAPH_SUCCESS) {
            continue;
        }
        if (nodeType == "HcomReduceScatterV") {
            auto ret = SetReduceScatterVVInputVec(node);
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR("[HcomVInputVecSetPassFn] SetReduceScatterVVInputVec failed, ret=%d.", ret);
                return ge::FAILED;
            }
        } else if (nodeType == "HcomAllGatherV") {
            auto ret = SetAllGatherVVInputVec(node);
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR("[HcomVInputVecSetPassFn] SetAllGatherVVInputVec failed, ret=%d.", ret);
                return ge::FAILED;
            }
        } else if (nodeType == "HcomAllToAllV") {
            auto ret = SetAllToAllVInputVec(node);
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR("[HcomVInputVecSetPassFn] SetAllToAllVInputVec failed, ret=%d.", ret);
                return ge::FAILED;
            }
        }
    }

    return ge::SUCCESS;
}

class HcomVInputVecSetPass : public ge::fusion::FusionBasePass {
public:
    ge::Status Run(ge::GraphPtr& graph, ge::CustomPassContext& ctx) override
    {
        return HcomVInputVecSetPassFn(graph, ctx);
    }
};

REG_FUSION_PASS(HcomVInputVecSetPass).Stage(ge::CustomPassStage::kAfterInferShape);
