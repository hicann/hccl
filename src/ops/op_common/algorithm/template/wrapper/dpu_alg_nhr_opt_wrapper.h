/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DPU_ALG_NHR_OPT_WRAPPER
#define DPU_ALG_NHR_OPT_WRAPPER

#include "alg_param.h"
#include "template_utils.h"

namespace ops_hccl {

// 用于 step 同步的专用 notify 索引（不与 ACK(0)/DATA(1)/FIN_ACK(2) 冲突）
constexpr u32 NOTIFY_IDX_STEP_SYNC = 0;

/**
 * @brief 统一的 NHR BatchTransfer 接口，覆盖所有收发场景：
 *        - 只发（tx 非空, rx 空）
 *        - 只收（tx 空, rx 非空）
 *        - 同对端收发（tx/rx 都非空, toRank == fromRank）
 *        - 不同对端收发（tx/rx 都非空, toRank != fromRank）
 *
 *        内部自动完成 step 同步 + 确定性排序，避免串行场景下的空转和死锁。
 *
 * @param stepInfo     当前 step 的收发信息（txSliceIdxs / rxSliceIdxs / toRank / fromRank）
 * @param channels     所有 rank 的 channel 映射
 * @param tempAlgParam 算法参数
 * @param repeat       当前 repeat 索引
 * @param myRank       当前 rank 的 user rank
 * @param templateRankSize 模板 rank 总数
 * @param thread       执行搬运的DPU线程句柄，共享内存请求老路径不传（默认0）
 * @param isDrainSupported hcomm是否支持Host Channel Drain，仅AICPU侧需要（host/DPU侧直接查版本）
 */
HcclResult BatchTransferNHR(
    const AicpuNHRStepInfo& stepInfo, const std::map<u32, std::vector<ChannelInfo>>& channels,
    const TemplateDataParams& tempAlgParam, u32 repeat, u32 myRank, u32 templateRankSize, ThreadHandle thread = 0,
    bool isDrainSupported = false);

// 校验NHR write-reduce所需的数据类型、规约操作及slice与buffer范围。
HcclResult ValidateReduceScatterNhrParams(const TemplateDataParams& tempAlgParams, u32 rankSize);

/**
 * @brief 构造单个ReduceScatter NHR step的通道和slice，并委托DpuWriteReduceTransfer执行。
 *        调用前须通过ValidateReduceScatterNhrParams校验参数，step中的对端须转换为user rank，
 *        slice索引保留算法rank，入口校验索引未超出rank及元数据范围；repeat须小于repeatNum。
 *        零长度slice无需通道。
 *
 * @param stepInfo 当前step的收发信息
 * @param channels 所有user rank的channel映射
 * @param tempAlgParam 已校验的算法参数
 * @param repeat 当前repeat索引
 * @param templateRankSize 模板rank总数，须与参数校验时的rankSize一致
 */
HcclResult DpuBatchWriteReduceTransferNHR(
    const AicpuNHRStepInfo& stepInfo, const std::map<u32, std::vector<ChannelInfo>>& channels,
    const TemplateDataParams& tempAlgParam, u32 repeat, u32 templateRankSize);

// ========== 三阶段批量传输 ==========

struct DpuTransferCtx {
    const ChannelInfo* txCh;            // 发送方向通道（非空仅表示链路存在）
    const ChannelInfo* rxCh;            // 接收方向通道（非空仅表示链路存在，samePeer 时 == txCh）
    std::vector<DataSlice> txSrcSlices; // 发送源切片
    std::vector<DataSlice> txDstSlices; // 发送目标切片
    std::vector<DataSlice> rxSrcSlices; // 接收源切片
    std::vector<DataSlice> rxDstSlices; // 接收目标切片

    bool hasTxChannel() const { return txCh != nullptr; }
    bool hasRxChannel() const { return rxCh != nullptr; }
    bool hasSend() const { return hasTxChannel() && !txSrcSlices.empty(); }
    bool hasRecv() const { return hasRxChannel() && !rxSrcSlices.empty(); }
};

// 当前step的slice元数据、channel及发送目标范围校验。
HcclResult CheckWriteReduceTransfer(const DpuTransferCtx& transfer, HcclDataType dataType);

/**
 * @brief 三阶段批量传输：前同步 → 写数据 → 后同步
 *        调用方负责填充 DpuTransferCtx 列表（含正确的 txDst/txSrc 偏移），
 *        本函数批量执行 Phase 1（step sync）/ Phase 2（HcommWriteNbi）/ Phase 3（data sync + Drain）。
 */
HcclResult DpuBatchTransfer(std::vector<DpuTransferCtx>& pairs, ThreadHandle thread = 0, bool isDrainSupported = false);

/**
 * @brief 单个NHR step的DPU write-reduce传输：前同步 → write-reduce → Drain/Fence → 后同步。
 *        调用前须完成slice元数据、数据类型、规约操作和channel校验，本函数仅执行通信。
 *        空slice不下发通信原语，返回成功时当前step的接收数据已可供下一step使用。
 *
 * @param transfer 当前step的发送和接收通道及slice
 * @param dataType 规约数据类型
 * @param reduceOp 规约操作
 */
HcclResult DpuWriteReduceTransfer(const DpuTransferCtx& transfer, HcclDataType dataType, HcclReduceOp reduceOp);

} // namespace ops_hccl

#endif // DPU_ALG_NHR_OPT_WRAPPER
