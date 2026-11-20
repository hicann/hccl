/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_OP_UTILS_H
#define HCCL_OP_UTILS_H

#include <cstdint>
#include <string>
#include <vector>

#include "acl/acl_rt.h"
#include "graph/types.h"
#include "hccl/hccl_types.h"
#include "hccl_custom_op.h"

namespace hccl {
HcclDataType GeDataTypeToHccl(ge::DataType dt);
HcclReduceOp StringToReduceOp(const char* reduction);
uint64_t GetHcclDataTypeSize(HcclDataType dt);

// loopMaxTime 与 LaunchRefresh 循环同源推导: maxCountPerLoop = cclBuffSize / unitSize (组 1) 或
// cclBuffSize / (rankSize * unitSize) (组 2), loopMaxTime = ceil(count / maxCountPerLoop),
// 保证 tags.size() 恒等于实际迭代数。与 InfoStore GetOpKernelLoopTime 的 ceil(count*u*R/C) 在
// cclBuffSize 被 unitSize (或 rankSize*unitSize) 整除时等价、非整除时后者偏小 (tag 不足)。公式表 (opType 对照
// hccl/hcom.h 的 HCCL_KERNEL_OP_TYPE_* 常量):
//   组 1 — AllReduce / Broadcast / Reduce / Send / Receive: maxCountPerLoop = cclBuffSize / unitSize
//   组 2 — ReduceScatter / AllGather:                        maxCountPerLoop = cclBuffSize / (rankSize*unitSize)
//   不 loop — AllGatherV / ReduceScatterV / AlltoAll 系等:    返回 0 (不走 CCL buf 分迭代)
// 组 1 公式不消费 rankSize (传任意值均可, 惯例传 1)。
// 返回 GRAPH_SUCCESS + loopMaxTime=0 表示不 loop (非 loop 算子 / count==0);
// 返回 GRAPH_FAILED 表示错误 (unitSize==0 / 组 2 rankSize==0 / buf 装不下单迭代 / 迭代数超 u32 值域),
// 均带 ERROR 日志。
// 出参类型与 InfoStore (u32& loopMaxTime) 及 BuildTagVec/tags 索引同宽; 内部 u64 商到 u32 的窄化
// 显式受检 — 实际值域受 CCL buffer 尺寸约束远小于 u32 上限, 超限按错误显式拦截而非静默截断。
ge::graphStatus CalcLoopMaxTime(
    const std::string& opType, uint64_t count, uint64_t unitSize, uint64_t cclBuffSize, uint32_t rankSize,
    uint32_t& loopMaxTime);

// tag 构造公共化 (对应 InfoStore GenerateOpTagFromTaskInfo + GetTagVectorInfo):
// HcomGenerateCclOpTag 生成 base tag; loopMaxTime==0 → {tag}; loopMaxTime>0 → {tag + "_looptime_" + i}
// 首参 tagSeed 为喂给 HcomGenerateCclOpTag 的种子串: 集合通信类传 opType 常量
// (HCCL_KERNEL_OP_TYPE_*), P2P (Send/Recv) 传 BuildSendRecvBaseTag 预构的 baseTag 串
ge::graphStatus
BuildTagVec(const std::string& tagSeed, const char* group, uint32_t loopMaxTime, std::vector<std::string>& tags);

// 环境推导的 CCL buf 可用尺寸 — 对照 InfoStore GetCCLBufferAvailableSize (hcom_base_funs.cc):
// HCCL_BUFFSIZE 环境变量(MB, 缺省 200) × 1MB - 1MB 保留值, 与通信域状态无关。
// kernel_builder/graph_optimizer 的 opParam 组装 (SetOpParamGraphModeHCCLBufferSize) 输入,
// 与实际 in-CCL buffer 尺寸 (GetOrCreateInCCLbuf) 是两个来源。
ge::graphStatus GetCCLBufferAvailableSize(uint64_t& size);

// 在线资源计算 (抽自 HcomAllReduceOp::CalcOnlineResources, AllReduce 特有依赖仅 opType/cmdType 两处,
// 参数化后各集合通信算子共用; cclBufferAvailSize 为 SetOpParamGraphModeHCCLBufferSize 的输入——
// 环境推导 available size (GetCCLBufferAvailableSize), 结果写入 st 的
// scratchMemSize/streamNum/taskNum/aivCoreNum/ifAiv; st.cclBuffSize 由调用方从实际
// in-CCL buffer 取得 (GetOrCreateInCCLbuf), 本函数不写它)
ge::graphStatus CalcOnlineResources(
    HcclOpState& st, uint32_t rankSize, uint64_t cclBufferAvailSize, const std::string& opType, HcclCMDType cmdType);

// crack 零缓冲清理器 (抽自 HcomAllReduceOp::CleanCracks): 无状态 — 零缓冲经 st.ctx->MallocWorkSpace
// 按次申请, 归属模型级 workspace 分配器 (GE 模型卸载时 FreeDynamicWorkspaceMemory 统一释放)。
// 生命周期依据: D2D 清零任务加载期入队、随模型每步重放, 零缓冲作为拷贝源须跨重放存活 → 必须挂靠
// 模型而非算子实例 (算子实例被 registry 进程级缓存, 析构不在模型卸载期发生, 挂算子实例会泄漏且
// 跨模型复用裸指针不安全)。每 Execute 至多调用一次 (Direct 一次 / Refresh 仅末迭代),
// 按次申请无累积浪费 (crack 为 512B 对齐 padding, 缓冲 < 512B)。
class CrackCleaner {
public:
    CrackCleaner() = default;
    ~CrackCleaner() = default;
    CrackCleaner(const CrackCleaner&) = delete;
    CrackCleaner& operator=(const CrackCleaner&) = delete;

    ge::graphStatus
    Clean(HcclOpState& st, const std::vector<void*>& crackAddrs, const std::vector<uint64_t>& crackSizes);
};

// 仅从 host 内存数据读取 u64 向量 (InferShape 编译期 const 场景, 禁止 D2H)
// 读取长度以 shape 声明的元素数为准 (调用方 buffer 可能大于 shape, 如实测 shape=[2] 但 buffer 64 元素);
// tensor 为空 / addr 为空 / 非 host placement / dtype 非 INT32|INT64 / shape 非静态 / buffer 小于 shape → GRAPH_FAILED
// 注: 本接口面向输入侧 counts/displs 张量 (形状恒为 [rankSize] 等参数形状, 0 元素/空地址属喂参异常, 报错),
// 与 SetEmptyOutputTensor 的输出侧空定形 (0 元素输出合法, 不分配数据) 是两个角色, 契约不冲突
ge::graphStatus ReadHostU64Vector(const gert::Tensor* tensor, std::vector<uint64_t>& values);

// host 直读, device 数据 D2H 后读取 (Execute 运行期场景); host 分支转调 ReadHostU64Vector
// 读取长度同样以 shape 声明为准 (D2H 只拷 shape 声明的字节数); device 分支 D2H 前同步主流 —
// host 线程裸 D2H 不等产出数据的流上前序 kernel, 会抢跑读到旧值/分配器 fill (stream 为空则跳过)
ge::graphStatus ReadU64VectorWithD2H(const gert::Tensor* tensor, std::vector<uint64_t>& values, aclrtStream stream);

// D2H 前同步主流: stream 为空 (host 直读场景) 为 no-op; 失败返回 GRAPH_FAILED (fail-fast,
// 不再留痕继续)。批量读多个同流张量时先调用本入口同步一次, 再用 NoSync 版逐个读
ge::graphStatus SyncStreamBeforeD2H(aclrtStream stream);

// ReadU64VectorWithD2H 的不同步版本: 调用方须已保证流同步 (SyncStreamBeforeD2H 或前序
// 同流同步), 用于同流多张量批量读
ge::graphStatus ReadU64VectorWithD2HNoSync(const gert::Tensor* tensor, std::vector<uint64_t>& values);

// 调试辅助: 打印 counts 张量的 placement/shape/dtype 与前若干元素值 (上限 16 个),
// 用于契约校验失败时定位调用方实际喂入的数据 (如 padding 长度、全局索引或精确 rankSize)
std::string DumpU64TensorInfo(const gert::Tensor* tensor, const std::vector<uint64_t>& values);

// 空输出 (0 元素) 定形: GE eager MallocOutputTensor 不支持 0 字节申请
// (ge eager_op_execution_context.cc 断言 GetSize()>0, 报 "Malloc output tensor data failed, size: 0"),
// 而零计数在 HCCL 语义内合法 (RS: HcclReduceScatter recvCount==0 提前成功; RSV: recvCount==0 不校验
// recvBuf; AGV: recvCounts 全 0 为空操作、允许 recvBuf 为空)。故空输出改为只设置形状/格式/类型、
// 不分配数据 — 输出地址为 nullptr, 由通信入口按零计数语义跳过访问。
// 返回输出 Tensor (与 MallocOutputTensor 同款), 失败返回 nullptr。
gert::Tensor* SetEmptyOutputTensor(
    gert::EagerOpExecutionContext* ctx, size_t index, const gert::StorageShape& shape,
    const gert::StorageFormat& format, ge::DataType dtype);

// ---- V 变体 (AllGatherV / ReduceScatterV) 公共参数工具 (动静态提取共用) ----

// x shape dims[1:] 乘积 (对应 vinputvec pass GetOtherDims, custom op 侧基于 gert::Shape 的同构实现):
// dimNum==0 或 dims[1:] 存在零维/未知维 (UNKNOWN_DIM=-1/UNKNOWN_DIM_NUM=-2) 时显式报错 ——
// 乘 0 会静默清空 counts, 负值 cast 到 uint64 变垃圾大数, 集群一致性要求乘法前 fail fast
ge::graphStatus CalcOtherDimsFromShape(const gert::Shape& xShape, const char* opName, uint64_t& otherDims);

// 读 count/counts 两个 u64 计数张量并校验契约: count 非空、counts.size()==rankSize;
// 校验失败输出两个张量的 DumpU64TensorInfo 排查日志 (custom op 读不到 vInputVec,
// shape 超 rankSize 多为 padding/全局索引)。countRole/countsRole 为 "recv_count"/"send_counts"
// 这类参数角色名, 用于日志区分
ge::graphStatus ReadVCountPairWithCheck(
    const gert::Tensor* countTensor, const gert::Tensor* countsTensor, const char* opName, const char* countRole,
    const char* countsRole, uint32_t rankSize, std::vector<uint64_t>& rawCount, std::vector<uint64_t>& rawCounts,
    aclrtStream stream);

// 读取可选 displs 张量 (三态): 张量不存在 → hasDispls=false (回退前缀和);
// 存在但读取失败 → 显式报错 (D2H 拷贝等故障不静默吞); 读取成功但长度 != rankSize →
// hasDispls=false 回退前缀和 (对齐原 vinputvec pass 契约: displs 可选、长度不符忽略)
ge::graphStatus ReadOptionalVDispls(
    const gert::Tensor* displsTensor, const char* opName, const char* displsRole, uint32_t rankSize,
    std::vector<uint64_t>& rawDispls, bool& hasDispls, aclrtStream stream);

// displs 解析 (首维单位): hasDispls 时直取 rawDispls, 否则按 counts 前缀和
// (对应 vinputvec pass CalcDisplacements 的缺省分支)
ge::graphStatus ResolveVDispls(
    bool hasDispls, const std::vector<uint64_t>& rawDispls, const std::vector<uint64_t>& counts,
    std::vector<uint64_t>& displs);

// counts + displs 首维单位 → 元素单位 (×otherDims, 对应 lowering counts[i] *= multiplier):
// dimDispls 由调用方先经 ResolveVDispls 解析 (AGV 等还需用首维单位值算 outDim0)
// u64 溢出 / 长度不匹配 / otherDims==0 显式报错
ge::graphStatus ConvertVCountsToElementUnit(
    const std::vector<uint64_t>& rawCounts, const std::vector<uint64_t>& dimDispls, uint64_t otherDims,
    std::vector<uint64_t>& counts, std::vector<uint64_t>& displs);

// 运行期 rankSize 获取 (hcom.h HcomGetRankSize 的轻封装, 统一日志前缀):
// counts 读取长度与 displs 计算都依赖 rankSize; 基类 GetCommunicator 在 Execute 模板后续才执行,
// 参数提取阶段只能用 group 版接口。失败或 rankSize==0 报错
ge::graphStatus GetVRankSizeByGroup(const char* group, const char* opName, uint32_t& rankSize);

// 单输出 V 算子的输出分配: outShape 拷贝 x 的 storage/origin shape 且 dim0 置 outDim0,
// outDim0==0 走 SetEmptyOutputTensor 只定形不分配 (GE eager Malloc 不支持 0 字节申请;
// HCCL 零计数语义允许输出地址为空), 否则 MallocOutputTensor; 失败返回 nullptr (调用方报错)
gert::Tensor* AllocVOutputTensor(HcclOpState& st, const gert::Tensor* xTensor, int64_t outDim0);

// counts/displs 逐 rank INFO 摘要: rankSize 大时逐条打印会刷爆日志, 合并为单条
// "idx:counts/displs; ..." (前 MAX_LOG_RANK_NUM 个, 超出追加 total 提示)
std::string SummarizeVCounts(const std::vector<uint64_t>& counts, const std::vector<uint64_t>& displs);

// ---- Send/Receive (P2P) 专用工具 (InfoStore 公共版尚无对应物) ----

// Send/Receive 动态路径线协议常量: shape 元数据固定传输的 int64 个数。
// 必须与 GE lowering (hcom_launch_kernel.h 的 SHAPE_INFO_COUNT) 保持一致, 任一侧修改都会导致收发失配。
constexpr uint64_t SHAPE_INFO_COUNT = 100U;

// SendRecv baseTag: "SendRecv_{srTag}_{srcRank}_{destRank}"
// (InfoStore GenerateOpTagFromTaskInfo 的 SEND/RECEIVE 分支同格式; 作为种子串传给 BuildTagVec)
std::string BuildSendRecvBaseTag(uint32_t srTag, uint32_t srcRank, uint32_t destRank);

// 对齐 count: RoundUp(dataSize, alignSize) / unitSize。
// 动态路径 alignSize=32 (GE lowering GetSendCountByShapeAlign 同公式, 收发两侧严格对称)。
// 静态路径不走本函数 (count 取精确逻辑字节数, 与 Recv output 按 shape 逻辑尺寸分配对称)。
// unitSize 为 0 时防御返回 0 (调用方须先做 dataType 早校验)。
// alignSize==0 已防御 (与 unitSize 同口径); 当前两处调用 (send/receive) 均传编译期常量 32
uint64_t CalcAlignedCount(uint64_t dataSize, uint64_t unitSize, uint64_t alignSize);

// aivCoreLimit 三级回退链的纯解析层 (对照 InfoStore GetAivCoreLimit):
// ① strtol 安全解析 (null/空/非数字/超界/负数 → 0 保持缺省)
uint32_t ParseAivCoreLimit(const char* valueStr);
// ② ge.hardwareInfo 格式 "k:v;k:v" 中取 vector_core_cnt 字段值
uint32_t ParseVectorCoreCntFromHardwareInfo(const char* hardwareInfo);

} // namespace hccl

#endif // HCCL_OP_UTILS_H
