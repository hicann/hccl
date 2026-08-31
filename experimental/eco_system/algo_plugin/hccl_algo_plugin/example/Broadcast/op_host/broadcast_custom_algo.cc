/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * 编译产物：libBroadcastCustomAlgoImpl.so
 * 部署路径：${HCCL_PLUGIN_ALG_DIR}/Broadcast/libBroadcastCustomAlgoImpl.so
 *
 * 【验证场景二：一个算法对应一个so】本so只导出一个执行函数 HcclAlgoPluginBroadcastTree，
 * 对应 selector/broadcast_selector.cc 中注册的唯一算法条目。
 *
 * 打桩说明：仅打印日志+做一次内存搬运代表"已处理"，不实现真实的树形广播语义，
 * 目的是验证插件框架"跨算子路由到正确的so并调用成功"的链路正确性。
 */

#include <cstdio>
#include <hccl/hccl_types.h>
#include <hccl/hccl_comm.h>
#include <acl/acl_rt.h>

extern "C" HcclResult HcclAlgoPluginBroadcastTree(
    void* buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
{
    (void)comm;
    (void)stream;
    (void)buf;
    std::fprintf(
        stderr,
        "[BroadcastCustomAlgoImpl] HcclAlgoPluginBroadcastTree invoked: "
        "count=%llu, dataType=%d, root=%u\n",
        static_cast<unsigned long long>(count), static_cast<int>(dataType), root);
    return HCCL_SUCCESS;
}
