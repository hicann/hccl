/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CUSTOM_OP_GE_HCOM_ALLTOALLV_OP_H
#define HCCL_CUSTOM_OP_GE_HCOM_ALLTOALLV_OP_H

#include "hccl_custom_op.h"
#include <cstdint>
#include <vector>

// AllToAll 系列三算子 (HcomAllToAll/HcomAllToAllV/HcomAllToAllVC) 单文件收敛,
// 文件名对齐 HCCL 侧 src/ops/all_to_all_v (实现主体为 V 形态, AllToAll/VC 是其特例):
// HCCL 单算子侧三个 GraphMode 入口内部全走 AlltoAllVOutPlaceGraphMode (只传不同 HcclCMDType),
// custom op 层照此共享一个 Op 类 + kind 区分, 三入口仅参数形态不同:
//   AllToAll:   x → y, per-rank 均匀切分 (sendCount = recvCount = 总元素数 / rankSize)
//   AllToAllV:  send_data + send_counts/sdispls/recv_counts/rdispls (数组 device 地址) → recv_data
//               (output 1-D [max(rdispls[i]+recvCounts[i])], proto HcomAllToAllVInferShapeV2)
//   AllToAllVC: send_data + send_count_matrix (rankSize*rankSize 矩阵 device 地址) → recv_data
//               (output 1-D [Σ matrix[rankId+i*rankSize]] 本 rank 列和, GE InfoStore GetHcomAlltoallVCOpMemSize 口径)
// V/VC 收发不等长, output 不能镜像 send_data shape
enum class HcomAllToAllKind {
    kAllToAll,
    kAllToAllV,
    kAllToAllVC,
};

// 地址刷新 args struct (仅 V/VC 消费, AllToAll 等长形态不参与刷新 — 原路径 InfoStore
// HcomAllToAllOpKernel 不调 RefreshInputAddr/RefreshOutputAddr): MallocReadOnlyDevArgs 分配
// host+device 双份, GE 统一 H2D 同步, D2D ADDR task 用字段 device 地址二级寻址
struct HcomAllToAllArgs {
    void* cclInputPtr;  // CCL input buf device 地址 (Execute 写入, 不变)
    void* cclOutputPtr; // CCL output buf device 地址 (Execute 写入, 不变)
    void* inputAddr;    // 用户 send_data device 地址 (UpdateHostArgs 刷新)
    void* outputAddr;   // 用户 recv_data device 地址 (UpdateHostArgs 刷新)
};

class HcomAllToAllVOp : public hccl::HcclCustomOpBase, public ge::ArgsUpdater {
public:
    explicit HcomAllToAllVOp(HcomAllToAllKind kind) : kind_(kind) {}
    ~HcomAllToAllVOp() override = default;
    const char* GetOpTypeName() const override;

    // IR input 索引 (ge 仓 all_ops.cpp EsHcomAllToAll/EsHcomAllToAllV/EsHcomAllToAllVC)
    static constexpr size_t INPUT_IDX_SEND_DATA = 0;         // 三算子共用: x / send_data
    static constexpr size_t INPUT_IDX_SEND_COUNTS = 1;       // V: send_counts
    static constexpr size_t INPUT_IDX_SEND_DISPLS = 2;       // V: send_displacements
    static constexpr size_t INPUT_IDX_RECV_COUNTS = 3;       // V: recv_counts
    static constexpr size_t INPUT_IDX_RECV_DISPLS = 4;       // V: recv_displacements
    static constexpr size_t INPUT_IDX_SEND_COUNT_MATRIX = 1; // VC: send_count_matrix
    static constexpr size_t OUTPUT_IDX_RECV_DATA = 0;        // 三算子共用: y / recv_data
    // IR attr 索引: AllToAll/AllToAllV 仅 group; AllToAllVC 为 rank(0)/group(1)/fusion(2)/fusion_id(3)
    static constexpr size_t ATTR_IDX_GROUP = 0;
    static constexpr size_t ATTR_IDX_GROUP_VC = 1;
    static constexpr size_t ATTR_IDX_RANK_VC = 0;

protected:
    ge::graphStatus ExtractParamsStatic(hccl::HcclOpState& st) override;
    ge::graphStatus CalcResourcesStatic(hccl::HcclOpState& st) override;
    ge::graphStatus LaunchHcclOpStatic(hccl::HcclOpState& st) override;
    ge::graphStatus InferShape(gert::InferShapeContext* ctx) override;
    ge::graphStatus InferDataType(gert::InferDataTypeContext* ctx) override;

    ge::graphStatus LaunchHcclOpDynamic(hccl::HcclOpState& st) override;

    // 地址刷新回调 (仅刷新路径分配过 args 的算子被 GE 回调): 更新 send_data/recv_data 地址
    ge::graphStatus UpdateHostArgs(gert::UpdateArgsContext* ctx) override;

private:
    // 静态路径: offline 资源计算 (OpParamGraphMode 创建/设置/计算/selector/销毁)
    ge::graphStatus CalcOnlineResources(hccl::HcclOpState& st, uint32_t rankSize, uint64_t cclBuffSize);
    // 共享: 生成单 tag — AllToAll 系列不在 GetOpKernelLoopTime 覆盖列表, 无 "_looptime_" 后缀
    ge::graphStatus GenerateTag(hccl::HcclOpState& st);
    // 共享: 按 kind 分发到三个 GraphMode 入口, 直传用户地址 (数组地址无法切片, 无 batch)
    ge::graphStatus LaunchGraphMode(hccl::HcclOpState& st);
    // V/VC 刷新路径 (单次整块, 不分批): args struct + ② D2D ADDR 搬入 + ③ CCL buf 上通信 + ⑤ D2D ADDR 回写
    ge::graphStatus LaunchRefresh(hccl::HcclOpState& st);

    const char* GetOpTypeStr() const; // kind → HCCL_KERNEL_OP_TYPE_* (hcom.h)
    HcclCMDType GetCmdType() const;   // kind → HCCL_CMD_* (hccl_types.h)
    size_t GetExpectInputNum() const; // kind → IR input 数 (1 / 5 / 2)
    // AllToAll per-rank count: st.totalElemCount / rankSize (lowering GetCountByShape 后 /rankSize 语义)
    ge::graphStatus CalcAllToAllPerRankCount(const hccl::HcclOpState& st, uint64_t& perRankCount) const;
    // V/VC 形态 shape 推导实现 (移植 proto 三段 InferShapeV2 中的两段, AllToAll 镜像分支内联在
    // InferShape)
    ge::graphStatus InferVRecvShape(gert::InferShapeContext* ctx, gert::Shape* outputShape);
    ge::graphStatus InferVCRecvShape(gert::InferShapeContext* ctx, gert::Shape* outputShape);

    HcomAllToAllKind kind_;
    // 注: per-execution 状态 (totalElemCount / sendCountMatrix / V 四元组) 全部落
    // HcclOpState st — st 贯穿同一次 Execute 的全部 hook, 无需算子实例成员; 算子实例是
    // 模型级单例, 实例成员在并发执行同类型节点时互相覆盖
};

#endif // HCCL_CUSTOM_OP_GE_HCOM_ALLTOALLV_OP_H
