/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CUSTOM_OP_GE_HCOM_ALL_GATHER_OP_H
#define HCCL_CUSTOM_OP_GE_HCOM_ALL_GATHER_OP_H

#include "acl/acl_rt.h"
#include "hccl_custom_op.h"
#include <cstdint>

// 地址刷新 args struct: MallocReadOnlyDevArgs 分配 host+device 双份, GE 统一 H2D 同步
// D2D ADDR task 用字段 device 地址做二级指针间接寻址
// AG 的膨胀 (output = input × rankSize) 体现在搬运 size 与 loopMaxTime, 不体现在字段数 (仍 4 字段)
struct HcomAllGatherArgs {
    void* cclInputPtr;  // CCL input buf device 地址 (Execute 写入, 不变)
    void* cclOutputPtr; // CCL output buf device 地址 (Execute 写入, 不变)
    void* inputAddr;    // 用户 input device 地址 (UpdateHostArgs 刷新)
    void* outputAddr;   // 用户 output device 地址 (UpdateHostArgs 刷新)
};

class HcomAllGatherOp : public hccl::HcclCustomOpBase, public ge::ArgsUpdater {
public:
    const char* GetOpTypeName() const override;
    static constexpr size_t INPUT_INDEX = 0;
    static constexpr size_t OUTPUT_INDEX = 0;
    // attrs 索引对照本仓 ops_proto_hccl.h 的 HcomAllGather IR 定义顺序:
    // REQUIRED_ATTR(rank_size, Int) / REQUIRED_ATTR(group, String) / ATTR(fusion) / ATTR(fusion_id)
    // 注意无 reduction (区别于 AllReduce/Reduce/RS), group 在 index 1 (区别于 RS 的 index 3)
    static constexpr size_t ATTR_RANK_SIZE = 0;
    static constexpr size_t ATTR_GROUP = 1;
    static constexpr size_t ATTR_FUSION = 2;
    static constexpr size_t ATTR_FUSION_ID = 3;

protected:
    ge::graphStatus ExtractParamsStatic(hccl::HcclOpState& st) override;
    ge::graphStatus CalcResourcesStatic(hccl::HcclOpState& st) override;
    ge::graphStatus LaunchHcclOpStatic(hccl::HcclOpState& st) override;
    ge::graphStatus InferShape(gert::InferShapeContext* ctx) override;
    ge::graphStatus InferDataType(gert::InferDataTypeContext* ctx) override;

    ge::graphStatus ExtractParamsDynamic(hccl::HcclOpState& st) override;
    ge::graphStatus LaunchHcclOpDynamic(hccl::HcclOpState& st) override;
    ge::graphStatus UpdateHostArgs(gert::UpdateArgsContext* ctx) override;

private:
    // 静态路径 (CalcOnlineResources/BuildTagVec 的公共实现见 hccl_op_utils)
    ge::graphStatus LaunchDirect(hccl::HcclOpState& st);
    ge::graphStatus LaunchRefresh(hccl::HcclOpState& st);
    ge::graphStatus GenerateTagVec(hccl::HcclOpState& st);

    // ---------------- 融合节点契约 (GE 融合 pass 产出的多输入 AG, 详见 .cc 各融合分支) ----------------
    // 分组: 同 label (optype + group + fusion_id + dtype) 的 N 个 HcomAllGather 合为 1 节点
    //       (引擎融合 pass GenerateFusionLabel), 不同 dtype 永不相融
    // 拓扑: N 输入 / N*rankSize 输出, anchor 映射 idx = r*N + i (rank r 持有的 op_i 数据 chunk)
    // desc: output[idx] 的 shape/dtype = input[idx % N] 原样 (不乘 R);
    //       ×R 语义由下游 Concat (AddRecvDataConCat 接收锚点 i 与 i+N) 的 dim_first 累加完成
    // count: 每输入 GetSize() 向上 512B 对齐累加 / unitSize (与 AR 融合同口径)
    // 内存: 融合 pass 对多输入 AG/RS 设 CONTINUOUS_INPUT/CONTINUOUS_OUTPUT
    //       (融合输入依赖首地址+连续布局假设, 运行期不校验, 详见 .cc)
    // 路径: refresh 放行融合 (GE-CUSTOMOP-REFRESH-IMMATURE 警告, LaunchHcclOpStatic),
    //       动态路径拒绝融合 (lowering 强制单输入, ExtractParamsDynamic 入口防御)
    // (前置: 调用方已完成 attrs 解析与校验, st.dataType 已设置)
    ge::graphStatus ExtractParamsStaticFused(
        hccl::HcclOpState& st, size_t inputCount, uint64_t attrRankSizeU, const char* group, const int64_t* fusion,
        const int64_t* fusionId);

    // AllGather 无 crack 清零 (InfoStore GenerateTask 的 crack tensor 列表不含 AG, tensorNum=0)
};

#endif // HCCL_CUSTOM_OP_GE_HCOM_ALL_GATHER_OP_H
