/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RHD_SCATTER_STEP_H
#define RHD_SCATTER_STEP_H

#include <set>
#include <vector>
#include "template_utils.h"

namespace ops_hccl {

// Scatter RHD 步骤表产出物（主体 = root 持有者链单向折半，后处理 = 非 2^N 代表回传份额段）。
// 前处理对 Scatter 恒为数据面空操作（只有 root 有输入），不入表。
// 区段字段一律为 σ 布局的 unit 空间（物理单元），字节区间经 RhdSegToBytes* 换算。
struct RhdScatterStep {
    enum class Phase : u8 {
        MAIN, // 主体步（折半序）
        POST, // 非 2^N 后处理（代表 → 被合并方，份额段）
    };
    Phase phase{Phase::MAIN}; // 步骤相位
    u32 step{0};              // 主体步序（POST 段为 0）
    bool isSend{false};       // true=本步发送；false=接收；空转步不入表
    u32 peerAlgRank{0};       // 对端物理 algRank（XOR 伙伴或后处理配对对端）
    u32 segStartUnit{0};      // 区段起（unit 空间，含）
    u32 segEndUnit{0};        // 区段止（unit 空间，不含）
};

// 预计算全部步骤（主体 + 非 2^N 后处理），空转步过滤；part1 划分/代表跟随 root/σ 映射/后处理
// 区段全部消费 template_utils 公共函数族，模板不做私有步骤生成。
// 防御校验：rankSize == 0 / rootAlgRank >= rankSize / myAlgRank >= rankSize → HCCL_E_PARA。
HcclResult GenRhdScatterSteps(
    u32 rankSize, u32 rootAlgRank, u32 myAlgRank, std::vector<RhdScatterStep>& steps, bool skipPost = false);

// 防御性校验（返回 false 时模板报 HCCL_E_NOT_SUPPORT）：rankSize > 0、rootAlgRank < rankSize
bool RhdScatterParamValid(u32 rankSize, u32 rootAlgRank);

// 尾块契约校验：tailSize == 0 视为无尾；否则须为类型整数倍（可大于 sliceSize，末位吸收余数）
bool RhdTailSizeValid(u64 tailSize, u32 dataTypeSize);

// RHD 建链连接集（algRank 下标）：全部 root 步骤表对端的并集。对端集合依赖 root 的两个
// 因子——代表侧（pair 槽的物理映射）与 vRoot（步活跃性），全 root 并集按构造覆盖运行时
// 全部可能对端（每表 O(log p)，共 O(p·log p)，建链期一次性）。NHR 的 ±2^k 连接集不含
// 非 2^N 经 v→phys 平移产生的对端（如 p=9、root=0 时 rank0↔rank3 距离 3 缺链），
// RHD 建链覆盖面必须以本集合为准。
HcclResult RhdCalcConnectRanks(u32 rankSize, u32 myAlgRank, std::set<u32>& connectRanks);

// —— Scatter→AG 融合直通（fusedWithNextAgPhase）——【预留未接线】为后续 Scatter→AG 融合
// PR 预埋：当前无生产置位点（localCopyFlag 仅有 =0/=1 写点），skipPost 形参与模板侧跳过
// 分支在生产不可达，仅 UT 覆盖读取口；编排层置位（相邻段确为 AG 时 |= bit1）随融合 PR
// 一并接线。载体：localCopyFlag bit1（bit0 为 omnipipe localCopy 模式值 0/1，无交集），
// 随参数序列化跨 DPU 边界。置位 = 跳过 POST 步生成与 PostCopy/PostLocalCopy，代表在
// ccl 中的 σ 双份布局留给相邻 AG（对侧对称跳过前置置换）从同偏移续跑。仅最外层相邻的
// Scatter/AG 可置位，且只应由编排层在确认相邻段确为 AG 后置位——否则被合并方将读到
// 未写入的 ccl（静默错数据）。
constexpr u64 RHD_FLAG_FUSED_WITH_NEXT_AG = 0x2; // bit1（bit0 为 omnipipe localCopy 模式值 0/1）

// 模板侧读取融合直通标志的唯一入口（避免裸位运算散落各模板）。
inline bool RhdFusedWithNextAg(const TemplateDataParams& tempAlgParams)
{
    return (tempAlgParams.localCopyFlag & RHD_FLAG_FUSED_WITH_NEXT_AG) != 0;
}

} // namespace ops_hccl

#endif // RHD_SCATTER_STEP_H
