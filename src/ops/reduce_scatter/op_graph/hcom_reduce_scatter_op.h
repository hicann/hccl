/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CUSTOM_OP_GE_HCOM_REDUCE_SCATTER_OP_H
#define HCCL_CUSTOM_OP_GE_HCOM_REDUCE_SCATTER_OP_H

#include <cstdint>

#include "acl/acl_rt.h"
#include "hccl_custom_op.h"
#include "hccl_op_utils.h"

// 地址刷新 args struct: MallocReadOnlyDevArgs 分配 host+device 双份, GE 统一 H2D 同步
// D2D ADDR task 用字段 device 地址做二级指针间接寻址
// RS 的膨胀 (input = output × rankSize) 体现在搬运 size 与 loopMaxTime, 不体现在字段数 (仍 4 字段)
struct HcomReduceScatterArgs {
    void* cclInputPtr;  // CCL input buf device 地址 (Execute 写入, 不变)
    void* cclOutputPtr; // CCL output buf device 地址 (Execute 写入, 不变)
    void* inputAddr;    // 用户 input device 地址 (UpdateHostArgs 刷新)
    void* outputAddr;   // 用户 output device 地址 (UpdateHostArgs 刷新)
};

// LaunchRefresh 的 device 侧准备结果: args 分配句柄 (结尾 DFX 日志用) + 各字段的
// device 地址 (二级指针间接寻址的 ADDR 拷贝任务消费), 见 SetupRefreshDevArgs
struct RsRefreshDevArgs {
    const gert::KernelArgs* devArgs = nullptr;
    void* devCclInputPtr = nullptr;
    void* devCclOutputPtr = nullptr;
    void* devInputAddr = nullptr;
    void* devOutputAddr = nullptr;
};

// LaunchRefresh 的循环配置: maxCountPerLoop 与 GenerateTagVec 的 loopMaxTime 同源
// (st.cclBuffSize + 通信域 rankSize 单一来源); outputMaxSize = 输出全量 count*unitSize;
// singleIteration = 膨胀总量不超 CCL buf (对照 InfoStore secAddrCopyWithoutOffset 尺寸判据)
struct RsRefreshLoopConfig {
    uint64_t maxCountPerLoop = 0;
    uint64_t outputMaxSize = 0;
    bool singleIteration = false;
};

class HcomReduceScatterOp : public hccl::HcclCustomOpBase, public ge::ArgsUpdater {
public:
    const char* GetOpTypeName() const override;
    static constexpr size_t INPUT_INDEX = 0;
    static constexpr size_t OUTPUT_INDEX = 0;
    // attrs 索引对照本仓 ops_proto_hccl.h 的 HcomReduceScatter IR 定义顺序:
    // REQUIRED_ATTR(reduction, String) / ATTR(fusion) / ATTR(fusion_id) / REQUIRED_ATTR(group, String) /
    // REQUIRED_ATTR(rank_size, Int) — 注意 group 在 index 3, 与 AllReduce/Reduce 不同
    static constexpr size_t ATTR_REDUCTION = 0;
    static constexpr size_t ATTR_FUSION = 1;
    static constexpr size_t ATTR_FUSION_ID = 2;
    static constexpr size_t ATTR_GROUP = 3;
    static constexpr size_t ATTR_RANK_SIZE = 4;

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
    // SplitV 融合形态 (GE fusion pass 重写: N×rankSize 个 per-rank 切片输入/N 输出) 的参数提取,
    // 由 ExtractParamsStatic 判定后分流; 契约与校验见 .cc 实现注释
    ge::graphStatus
    ExtractParamsStaticFused(hccl::HcclOpState& st, size_t inputNum, uint64_t rankSize, const char* group);
    // 静态路径 (CalcOnlineResources/BuildTagVec 的公共实现见 hccl_op_utils)
    // Direct/Refresh 双路径保留: Direct 直传用户 buffer, Refresh 走 CCL buffer 中转 + args struct 地址刷新
    ge::graphStatus LaunchDirect(hccl::HcclOpState& st);
    ge::graphStatus LaunchRefresh(hccl::HcclOpState& st);
    ge::graphStatus GenerateTagVec(hccl::HcclOpState& st);

    // SplitV 融合形态判定: GE RS fusion pass (fusion=2, 按 fusion_id 分组) 将 N 个原算子重写为
    // N×rankSize 输入/N 输出的融合节点; 返回 true 时 inputNum 为预扫描输入数
    bool IsFusedSplitvForm(hccl::HcclOpState& st, uint64_t attrRankSizeU, size_t& inputNum);
    // 静态/动态路径公共的 attr 提取: input/attrs 判空、reduction/group/rank_size attr 读取与校验、
    // dataType/reduceOp 转换校验; rankSizeI 出参 (>0 已校验), group 出参 (指向 attr 字符串, 生命周期随 ctx)
    ge::graphStatus ExtractCommonAttrs(hccl::HcclOpState& st, const char*& group, int64_t& rankSizeI);
    // 单输入的 RS 输出 shape 变换 + 分配: dim0/rankSize 且 storage/origin 双 shape 同步改写
    // (MallocOutputTensor 按 storage shape 分配; 只改 origin 会漏改 storage),
    // UNKNOWN_DIM 直通、0 元素走 SetEmptyOutputTensor; 返回输出 Tensor (失败 nullptr, 调用方报错)
    gert::Tensor* AllocRsOutput(hccl::HcclOpState& st, size_t index, const gert::Tensor* inTensor, int64_t rankSize);
    // ExtractParamsStatic 循环内: 512B 对齐 crack 缝隙累计 (对齐 InfoStore GetCrackParamsInfo,
    // 与 AllReduce 同款口径; ALIGNED_SIZE 对齐粒度与 crack 计算同源)
    void AccumulateCrackInfo(uint64_t inputSize, uint64_t& crackTotalSize, hccl::HcclOpState& st);

    // ---- SplitV 融合形态的逐项布局校验/推导 (ExtractParamsStaticFused 拆分) ----
    // 校验 1: 同一 op 的各槽位切片 shape 一致 (input[r*N+op] ≡ input[op])
    ge::graphStatus ValidateFusedSliceSizes(const std::vector<uint64_t>& inBytes, size_t fusedOps, uint64_t rankSize);
    // 校验 2: 输入地址连续升序不重叠
    ge::graphStatus ValidateFusedContiguity(const std::vector<void*>& inAddrs, const std::vector<uint64_t>& inBytes);
    // 校验 3: 槽位均匀 (slotBytes = input[N]-input[0], 各槽位步长相等) + 元素对齐检查
    ge::graphStatus CalcUniformSlotBytes(
        const std::vector<void*>& inAddrs, size_t fusedOps, uint64_t rankSize, uint64_t unitSize, uint64_t& slotBytes);
    // 分配 N 个融合输出 (shape = 切片 shape 直通, dim0/rankSize 已由 GE SplitV 完成)
    ge::graphStatus AllocFusedOutputs(hccl::HcclOpState& st, size_t fusedOps, std::vector<void*>& outAddrs);
    // 校验 4: 输出与输入槽位布局镜像 (各 op 输出偏移 == 槽内输入偏移); 镜像通过即输出块容量
    // = slotBytes (GE eager 512 对齐分配保证, 见 .cc 注释), 无额外容量判定
    ge::graphStatus
    ValidateFusedOutputMirror(const std::vector<void*>& inAddrs, const std::vector<void*>& outAddrs, size_t fusedOps);
    // crack 由实际地址间隙推导: 相邻输入间隙 + 尾隙 (末输入末尾到总覆盖末尾)
    void DeriveFusedCracks(
        const std::vector<void*>& inAddrs, const std::vector<uint64_t>& inBytes, size_t inputNum, uint64_t slotBytes,
        uint64_t rankSize, hccl::HcclOpState& st);

    // ---- LaunchRefresh 循环体拆分 (单迭代/多迭代输入搬运、crack 清零、输出回写) ----
    // Refresh 路径的 CCL buf 获取 (get-or-create 语义, 对照 InfoStore GetHcomInCCLbufferSize 查→建→重查)
    ge::graphStatus GetRefreshCclBufs(
        const char* group, void*& commInputPtr, uint64_t& commInputSize, void*& commOutputPtr,
        uint64_t& commOutputSize);
    // 循环配置推导: maxCountPerLoop/outputMaxSize/singleIteration; maxCountPerLoop==0 (CCL buf 装不下
    // 单迭代最小数据) 时守卫报错
    ge::graphStatus ResolveRefreshLoopConfig(const hccl::HcclOpState& st, uint64_t unitSize, RsRefreshLoopConfig& cfg);
    // args struct 组装: MallocReadOnlyDevArgs 分配 host+device 双份 (GE 统一 H2D), 推导 device 侧
    // 字段地址 (offsetof, 二级指针间接寻址用)
    ge::graphStatus
    SetupRefreshDevArgs(const hccl::HcclOpState& st, void* commInputPtr, void* commOutputPtr, RsRefreshDevArgs& out);
    // 每迭代 kernel launch 前的 tag 越界守卫, 对照 InfoStore HcomReduceScatterLoop
    ge::graphStatus CheckRefreshTagBounds(const hccl::HcclOpState& st, uint64_t loopTime);
    // 每迭代 kernel launch: HcclReduceScatterGraphMode (独立 tag _looptime_N; 无 root 参数)
    ge::graphStatus LaunchRefreshIterationKernel(
        hccl::HcclOpState& st, void* commInputPtr, void* commOutputPtr, uint64_t curCount, uint64_t loopTime);
    // 单迭代: 一次连续膨胀拷贝 curSize*rankSize (用户 input → CCL input buf)
    ge::graphStatus RefreshInputToCclOnce(
        const hccl::HcclOpState& st, void* devCclInputPtr, uint64_t commInputSize, void* devInputAddr, uint64_t curSize,
        uint64_t inputOffset, uint64_t loopTime);
    // 多迭代: per-rank 分段拷贝 — 用户 input 按 rank 分块, 每迭代从 rank r chunk 的
    // [inputOffset, inputOffset+curSize) 段放入 CCL buf 的 r*curSize 处
    ge::graphStatus RefreshInputToCclPerRank(
        const hccl::HcclOpState& st, void* devCclInputPtr, uint64_t commInputSize, void* devInputAddr, uint64_t curSize,
        uint64_t unitSize, uint64_t inputOffset, uint64_t loopTime);
    // 末迭代 crack 清零: gate = 末迭代 + 多输入 + 非空 crack, 窗口按 RS 膨胀布局
    // (单迭代全布局直译 / 多迭代 per-rank 映射)
    ge::graphStatus CleanFusedCracksForLoop(
        hccl::HcclOpState& st, void* commInputPtr, uint64_t curSize, uint64_t curCount, uint64_t countLeft,
        uint64_t unitSize, uint64_t inputOffset, bool singleIteration);
    // ⑤ CCL output buf → 用户 output 等长回写 (dstOffset = outputOffset)
    ge::graphStatus RefreshOutputFromCcl(
        const hccl::HcclOpState& st, void* devCclOutputPtr, void* devOutputAddr, uint64_t outputMaxSize,
        uint64_t curSize, uint64_t outputOffset, uint64_t loopTime);

    // InferShape 的 SplitV 融合分支: N×rankSize 输入切片/N 输出, 输出 shape = 切片 shape 直通
    ge::graphStatus InferShapeFused(gert::InferShapeContext* ctx, size_t outputNum, int64_t rankSize);
    // InferShape 的 rank_size attr 提取与校验 (index 4, 编译期无通信域)
    ge::graphStatus GetInferRankSize(gert::InferShapeContext* ctx, int64_t& rankSize);
    // ExtractParamsStatic 收尾日志: 多 input 告警 (HCCL 调用仍只用首地址) + 参数汇总 INFO
    void LogExtractParamsStaticSummary(
        const hccl::HcclOpState& st, size_t inputCount, uint64_t attrRankSizeU, const int64_t* fusion,
        const int64_t* fusionId);

    // crack 零缓冲清理器 (无状态, 缓冲经 MallocWorkSpace 按次申请; RS 的清零 gate 含 tensorNum > 1, 见 LaunchRefresh)
    hccl::CrackCleaner crackCleaner_;
};

#endif // HCCL_CUSTOM_OP_GE_HCOM_REDUCE_SCATTER_OP_H
