/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CUSTOM_OP_H
#define HCCL_CUSTOM_OP_H

#include "graph/custom_op.h"
#include "hccl/hccl_types.h"
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace hccl {

// aivCoreLimit 缺省值 (对照 InfoStore MAX_NUM_BLOCKS, op_common/inc/alg_param.h; 注意与
// AIV 模板侧 aiv_defines.h 的 48 是两处不同用途的定义, 勿混用 — 本处语义 = 算子核数上限缺省)
// 回退链 (InfoStore 与自定义算子路径同链): _op_vectorcore_num (OpDesc 节点属性, 经
// GetContextValue(kVectorCoreNum) 逐节点可达) → ge.hardwareInfo 的 vector_core_cnt → 缺省 56
constexpr uint32_t MAX_NUM_BLOCKS_DEFAULT = 56;

#define HCCL_GE_CHK_RET(stmt)            \
    do {                                 \
        ge::graphStatus _ret = (stmt);   \
        if (_ret != ge::GRAPH_SUCCESS) { \
            return _ret;                 \
        }                                \
    } while (0)

struct HcclOpState {
    gert::EagerOpExecutionContext* ctx = nullptr;
    HcclComm comm = nullptr;
    const char* group = nullptr;
    // 算子类型 (HCCL_KERNEL_OP_TYPE_*): Execute 入口从 GetOpTypeName() 取得, 基类/公共工具的
    // 失败日志据此带上算子身份
    const char* opType = nullptr;
    std::vector<void*> inputPtrs;
    std::vector<void*> outputPtrs;
    uint64_t count = 0;
    std::vector<uint64_t> inputCounts; // 动态路径: 每 input 独立 count (不对齐)
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    void* stream = nullptr;
    void* scratchMem = nullptr;
    uint64_t scratchMemSize = 0;
    HcclReduceOp reduceOp = HCCL_REDUCE_RESERVED;
    uint32_t root = 0; // Reduce: root rank (IR attr root_rank), 归约结果所在 rank
    uint32_t rankId
        = 0; // 本 rank 在通信域中的 rankId (CalcResourcesStatic 经 HcomGetRankId 取得, Reduce ⑤ root-only 判定)
    uint32_t rankSize = 0; // 通信域 rank size (AllReduce/RS 等 CalcResourcesStatic 经 HcclGetRankSize 取得;
                           // V 算子 ExtractParamsStatic 经 HcomGetRankSize(group) 取得)
    uint32_t destRank = 0; // P2P Send: 目标 rank (attr dest_rank) — 纯增量字段, 零行为变更
    uint32_t srcRank = 0;  // P2P Receive: 源 rank (attr src_rank) — 纯增量字段, 零行为变更
    uint32_t srTag = 0;    // P2P: Send/Recv 配对 tag (attr sr_tag) — 纯增量字段, 零行为变更
    uint32_t streamNum = 0;
    // 仅满足 HcclCalcOpResOnlineGraphMode 接口出参要求, 算子侧无功能消费者 —
    // 值经摘要日志 (hccl_op_utils.cc) 输出供现场定位, 不参与行为分支
    uint32_t taskNum = 0;
    uint32_t aivCoreNum = 0;
    bool ifAiv = false;
    uint64_t cclBuffSize = 0;
    // follow 流组: RequestFollowStreams 按 st.streamNum 申请 (key 编码主流 rtStreamId+序号),
    // 静态路径 GraphMode streams[] 参数交付; 空向量 = 不申请 (ifAiv / streamNum==0, 行为同改动前)
    std::vector<void*> followStreams;
    // attach 流两条 (对照旧 SetAttachedStreamInfoList 的 force_reuse 双流, 均跨主流复用):
    // attachGraphStream = 图粒度 (全模型一条, 所有通信域共享); attachGroupStream = 通信域粒度
    // (每 group 一条)。交付顺序 [graph, group] 与旧 AssembleAttachedRtStream 一致, HCCL 按位置
    // 消费; nullptr = 无 (ifAiv 跳过 — 旧机制 AIV 同样规避附属流申请)
    void* attachGraphStream = nullptr;
    void* attachGroupStream = nullptr;
    // Execute 内按三级回退解析: kVectorCoreNum (_op_vectorcore_num, 逐节点
    // 可达) → ge.hardwareInfo 的 vector_core_cnt (ctx->GetOption) → 缺省 56; 解析结果按
    // (graphId, nodeId) 缓存 (动态图免重复解析); 见 hccl_custom_op.cc Execute 内注释
    uint32_t aivCoreLimit = MAX_NUM_BLOCKS_DEFAULT; // 回退链: _op_vectorcore_num → hardwareInfo → 56
    bool isDynamicShape = false;                    // 基类 Execute 判定: GetContextValue(kStaticModel)
    bool needRefresh = false;                       // 基类 Execute 判定: GetContextValue(kAddressRefreshEnabled)
    // crack 清零门控 (仅静态路径消费): ResolveStaticSignals 读 ge.exec.atomicCleanPolicy=="1"
    // (对照旧 ATOMIC_CLEAN_POLICY/kCleanSeparately — builder/InfoStore 两处同款门控); 非 1 时
    // 静态分支清空 crackOffsets/Sizes, Clean no-op, 对齐旧路径"不清零"行为; 动态路径不涉及
    // crack (热路径裁剪后本信号不在动态分支解析)
    bool needCleanSeparately = false;
    std::vector<uint64_t> crackOffsets;
    std::vector<uint64_t> crackSizes;
    std::vector<std::string> tags;
    // ---- V 变体参数 (ReduceScatterV / AllGatherV 复用) ----
    // ReduceScatterV 方向: 每 rank 发送分片
    uint64_t recvCount = 0;           // 本 rank 接收元素总数 (= output tensor 元素数, lowering 语义)
    std::vector<uint64_t> sendCounts; // 每 rank 发送元素数 (已 ×otherDims)
    std::vector<uint64_t> sendDispls; // 每 rank 发送偏移, 元素单位 (const 直读 ×otherDims 或按 counts 累加)
    // AllGatherV 方向: 每 rank 接收分片
    uint64_t sendCount = 0;           // 本 rank 发送元素总数 (input[1] raw ×otherDims)
    std::vector<uint64_t> recvCounts; // 每 rank 接收元素数 (已 ×otherDims)
    std::vector<uint64_t> recvDispls; // 每 rank 接收偏移, 元素单位 (const 直读 ×otherDims 或按 counts 累加)
    // ---- AllToAll 族参数 (AllToAll/V/VC 复用) ----
    // per-execution 状态挂 st 而非算子实例成员: 算子实例是模型级单例, 实例成员在并发执行
    // 同类型节点时互相覆盖
    uint64_t totalElemCount = 0; // send_data 总元素数 — AllToAll per-rank count = totalElemCount / rankSize
    std::vector<uint64_t> sendCountMatrix; // VC: send_count_matrix 的 host 副本 (D2H 读出, launch host 消费)
};

// SoC 条件注册判定: 自定义算子仅在目标平台 (950/960) 接管 HCCL 算子, 其余 SoC 不注册 →
// 算子回退原 hccl_engine (InfoStore) 路径。注册发生在 .so dlopen 的静态初始化期, SoC 查询走
// HAL 层 (dlopen libascend_hal + halGetSocVersion, 不依赖 ACL 运行时与设备上下文, 加载期可用);
// 查询失败 (hal 库缺失等) 按 fail-closed 处理 — 不注册, 原路径接管。目标平台判定在实现内单点维护。
bool IsCustomOpTargetSoc();

// 运行时接口在位判定: 旧运行时 (9.2.0) 缺失本模块依赖的 GE/RTS 新接口。GE 头侧为
// 接口声明 weak 后, 经 GetContextValue 地址判空探测 GE 接口在位性 (同批接口含五参
// 构造器; 头未声明 weak 时编译期折叠为真, 此时 .so 仅能加载于接口齐备的运行时,
// 判定结果仍正确); 叠加 RTS 两接口弱引用判空, 堵双向版本错配。作为接管注册门控
// 之一, 判定通过即保证注册与 Execute 侧全部调用命中真实实现
bool HcclGeRuntimeInterfaceAvailable();

// 条件 5 元接管注册 (kBottom/kHccl): 认领 HCCL 引擎槽位, 与 GE 认领链
// (按优先级回退解析) 配套。全部算子使用 (12 个注册点)。注册条件 = 目标 SoC
// (IsCustomOpTargetSoc) + GE/RTS 新接口在位 (HcclGeRuntimeInterfaceAvailable),
// 任一不满足即不注册, 算子回退原 hccl_engine 路径。
ge::CustomOpCreatorRegister* CreateSocGatedRegister(const char* opType, ge::CustomOpCreateFunc creator);

// aivCoreLimit 配置值解析 (字符串→uint32, 对照 InfoStore GetAivCoreLimit 的 SalStrToInt + 校验);
// 非法值 (空/非数字/0/超范围) 返回 0, 调用方保持缺省

// 未找到/非法返回 0, 调用方保持缺省 (对照 InfoStore GetAivCoreLimit 的 hardwareInfo 回退分支)

class HcclCustomOpBase : public ge::EagerExecuteOp, public ge::ShapeInferOp {
public:
    ~HcclCustomOpBase() override = default;

    // 算子类型名 (HCCL_KERNEL_OP_TYPE_*): Execute 入口取回填入 st.opType, 基类与公共工具的
    // 失败日志带上算子身份
    virtual const char* GetOpTypeName() const = 0;

    ge::graphStatus Execute(gert::EagerOpExecutionContext* ctx) final;

protected:
    // 命名规则: 共享方法无后缀; 路径 hooks 一律带 Static/Dynamic 后缀
    // 路径判定 (isDynamicShape/needRefresh) 是基类职责, 在 Execute 模板内统一写入 st. 字段,
    // 各算子只消费不 override——动态/静态与刷新对所有算子是同一答案, 信号对接时基类一处修改

    // 静态路径 hooks
    virtual ge::graphStatus ExtractParamsStatic(HcclOpState& st) = 0;
    static ge::graphStatus GetCommunicator(HcclOpState& st);
    virtual ge::graphStatus CalcResourcesStatic(HcclOpState& st) { return ge::GRAPH_SUCCESS; }
    virtual ge::graphStatus LaunchHcclOpStatic(HcclOpState& st) = 0;

    // 动态路径 hooks (LaunchHcclOpDynamic 纯虚: 无算子的动态 launch 等于静态 launch,
    // 漏覆写时编译期报错优于运行期静默走带刷新语义的错误路径)
    virtual ge::graphStatus ExtractParamsDynamic(HcclOpState& st) { return ExtractParamsStatic(st); }
    virtual ge::graphStatus LaunchHcclOpDynamic(HcclOpState& st) = 0;

    // 公共工具函数 (无状态, 参数化)
    static ge::graphStatus GetOrCreateInCCLbuf(const char* group, void*& ptr, uint64_t& size);
    static ge::graphStatus GetOrCreateOutCCLbuf(const char* group, void*& ptr, uint64_t& size);
    // follow 流申请 (静态路径, Execute 下发期内调用一次): 按 st.streamNum 循环
    // ctx->RequestAttachedStream, key = "hccl_follow#<主流rtStreamId>#<序号>" — 同主流算子同 key
    // 复用 (复刻旧 main_stream_id 池化); ifAiv / streamNum==0 跳过; 流间同步/激活均不在
    // 算子侧, 本函数只创建与填充 st.followStreams
    static ge::graphStatus RequestFollowStreams(HcclOpState& st);
    // attach 流申请 + 交付 (静态路径): 两条流复刻旧 SetAttachedStreamInfoList 语义
    // (hcom_ops_kernel_builder.cc, 均按 force_reuse 跨主流复用) — [0] 图粒度 (key 沿用旧
    // reuse_key "hccl_attached_graph_stream", 全模型一条, 所有通信域共享) + [1] 通信域粒度
    // ("hccl_attached_group_stream_<group>", 每 group 一条); 交付顺序 [graph, group] 与旧
    // AssembleAttachedRtStream 一致 (HCCL 按位置消费)。经 HcomSetAttachedStream 注入通信域
    // (旧 InfoStore SetAttachedStream 每 op 一次的对应); ifAiv 跳过;
    // graphId 经 GetContextValue(kGraphId) 获取 (统一取根图, V1 静态下沉与 V2 动态路径);
    // 不可用时回退 0 并打 [STREAMDBG] 日志
    static ge::graphStatus RequestAttachStream(HcclOpState& st);

    // aicpu 提前展开流注册进模型 (静态路径, Execute 下发期内调用一次): 对照旧 InfoStore
    // GetHcclUnfoldStream (info_store.cc) — ctx 先查后建 (<group>_unfold,
    // COMM_ENGINE_CPU_TS, 存 ThreadHandle) → HcclDedicatedThreadAcquire (AICPU_LAUNCH,
    // 通信域内按 useType 缓存复用) → HcclThreadResGetInfo (thread→stream); 注册走 RTS 新
    // 接口 rtGetModelByStream(主流)→rtModel + rtModelAddHcclStream(unfoldStream, rtModel)
    // (直接调用, 链接期符号允许未定义, 由提供该接口的 RTS 运行时解析)。旧注册去向为 GE
    // 软件层 (HcomTopoInfo::SetGroupOrderedStream + OrderLaunch event 栅栏), 新去向为 RTS
    // 运行时层 — RTS 感知该 HCCL 流属于模型 (注册有去重语义, 多算子重复登记安全)。
    static ge::graphStatus RegisterUnfoldStreamToModel(HcclOpState& st);

private:
    // aivCoreLimit 三级回退解析 (kVectorCoreNum → ge.hardwareInfo → 缺省 56) + (graphId, nodeId)
    // 缓存 (动态图逐执行免重复解析) — 回退链与缓存语义详见 hccl_custom_op.cc 实现处注释
    uint32_t ResolveAivCoreLimit(gert::EagerOpExecutionContext* ctx);
    // aivCoreLimit 解析缓存 (key: {graphId, nodeId}): 动态图路径 (RT2 逐执行调 Execute) 反复
    // 走三级回退解析的性能规避 — 命中直接复用。本类实例按算子类型共享 (CreateOrGetCustomOp),
    // 同类型多节点/多线程并发执行, 读写均持锁; 容量上界 = 进程内各图 HCCL 节点总数, 不淘汰。
    std::map<std::pair<uint64_t, uint64_t>, uint32_t> aivCoreLimitCache_;
    std::mutex aivCoreLimitCacheMutex_;
};
} // namespace hccl

#endif // HCCL_CUSTOM_OP_H
