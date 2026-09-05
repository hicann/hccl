/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TUNER_SETUP_H
#define TUNER_SETUP_H

#include <cstdarg>
#include <cstddef>
#include "hccl_tuner_plugin.h"

/* HcclCMDType 定义在 hccl/hccl_types.h 中（经 hccl_tuner_plugin.h 引入），无需前向声明。 */

namespace ops_hccl {
struct TopoInfoWithNetLayerDetails;
}

#if defined(__cplusplus)
extern "C" {
#endif

/* ===== 内部 Host 函数包装（仅供 tuner_setup.cc 构造 hostFuncs 使用）===== */

HcclResult TunerCtxCreate(HcclComm comm, const char* ctxTag, uint64_t size, void** ctx);
HcclResult TunerCtxGet(HcclComm comm, const char* ctxTag, void** ctx, uint64_t* size);
HcclResult TunerCtxDestroy(HcclComm comm, const char* ctxTag);
void TunerLogFunction(int level, const char* file, int line, const char* fmt, ...);

#if defined(__cplusplus)
}
#endif

#if defined(__cplusplus)
/* ===== 对外接口 ===== */

/* 信任边界：插件 .so 在本进程内执行。
 * 慢调用保护：getCollInfo 连续 3 次超过 100ms 则自动禁用插件，回退 CostModel。 */

/* comm 创建时调用：dlopen 插件 + dlsym + 版本校验 + 引用计数 + 构造 hostFuncs + 调用插件 init。
 * topoInfo 用于填充 hcclTunerCommInfo_t。未配置 HCCL_TUNER_PLUGIN 或加载失败时为 no-op，返回 HCCL_SUCCESS。 */
HcclResult HcclTunerInit(HcclComm comm, const ops_hccl::TopoInfoWithNetLayerDetails* topoInfo);

/* 每次 op 时调用：构造 collInfo，调用插件 getCollInfo。
 * 插件未加载时为 no-op；不支持的算子类型（HCCL_CMD_INVALID）跳过，返回 HCCL_SUCCESS。
 * algoEntries 由调用方（Selector）提供，Enrich 已填好 3D 名，插件直接读。
 * modified 输出：true=插件命中规则并修改了 cost，false=未命中/未修改（per-call，线程安全）。 */
HcclResult HcclTunerCallGetCollInfo(
    HcclComm comm, HcclCMDType cmdType, size_t nBytes, HcclDataType dataType, hcclTunerAlgoEntry_t* algoEntries,
    int algoCount, bool* modified);

/* comm 销毁时调用：引用计数--。.so 不 dlclose（避免与在途 getCollInfo 竞争），随进程退出回收。 */
HcclResult HcclTunerDestroy(HcclComm comm);

/* 返回上次 getCollInfo 是否命中规则并修改了 cost（仅供 ST 测试在 comm destroy 后查询；
 * 生产路径应使用 HcclTunerCallGetCollInfo 的 modified 输出参数）。 */
bool HcclTunerDidModifyCost();

/* 重置 match 状态（测试 SetUp 调用，或 comm init 时调用）。 */
void HcclTunerResetMatchStatus();

/* 返回插件是否已成功加载（selector 可据此跳过 getCollInfo 调用）。 */
bool HcclTunerIsLoaded();
#endif

#endif /* TUNER_SETUP_H */
