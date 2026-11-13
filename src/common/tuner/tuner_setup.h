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

#include <cstddef>

#include "hccl_tuner_plugin.h"

/* HcclCMDType/HcclDataType/HcclReduceOp 定义在 hccl/hccl_types.h 中（经 hccl_tuner_plugin.h 引入）。 */

namespace ops_hccl {
struct TopoInfoWithNetLayerDetails;
}

#if defined(__cplusplus)
/* ===== 对外接口 ===== */

/* 信任边界：插件 .so 在本进程内执行。
 * 慢调用保护：UpdateAlgoCostTable 连续 3 次超过 100ms 则自动禁用插件，回退 CostModel。 */

/* comm 创建时调用：dlopen 插件 + dlsym HcclTunerPluginOpsGet + header 三连校验 + 调用插件 Create
 * + 拷贝进通信域 ctx（tag "__tuner_main"）。topoInfo 用于填充 HcclTunerTopoInfo。
 * 未配置 HCCL_ALG_TUNER_PLUGIN 或加载失败时为 no-op，返回 HCCL_SUCCESS。 */
HcclResult HcclTunerInit(HcclComm comm, const ops_hccl::TopoInfoWithNetLayerDetails* topoInfo);

/* 每次 op 时调用：取通信域 ctx，调用插件 UpdateAlgoCostTable。
 * 插件未加载 / ctx 不存在（Create 未成功或插件 inactive）时为 no-op；
 * 不支持的算子类型（HCCL_CMD_INVALID）跳过，返回 HCCL_SUCCESS。
 * algoCostEntries 由调用方（Selector）提供；costModel 物化时已填好 3D 名，
 * cost_table 生成条目时已初始化 ABI header，插件只读名字、可改 cost。
 * modified 输出：true=插件命中规则并修改了 cost，false=未命中/未修改（per-call，线程安全）。 */
HcclResult HcclTunerCallGetCollInfo(
    HcclComm comm, const HcclTunerCollOpInfo* collOpInfo, HcclTunerAlgoCostEntry* algoCostEntries, uint32_t algoCount,
    bool* modified);

/* 返回插件是否已成功加载（selector 可据此跳过 Update 调用）。 */
bool HcclTunerIsLoaded();
#endif

#endif /* TUNER_SETUP_H */
