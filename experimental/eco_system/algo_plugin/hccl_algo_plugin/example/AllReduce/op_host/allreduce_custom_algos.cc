/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * 编译产物：libAllReduceCustomAlgosImpl.so
 * 部署路径：${HCCL_PLUGIN_ALG_DIR}/AllReduce/libAllReduceCustomAlgosImpl.so
 *
 * 【验证场景一：多个算法对应一个so】本文件在同一个.so中导出两个执行函数
 * （HcclAlgoPluginAllReduceSmall / HcclAlgoPluginAllReduceLarge），
 * 对应 selector/allreduce_selector.cc 中注册的两个算法条目。
 *
 * 打桩说明：本示例只为验证"PluginBroker能否根据Select()命中的算法名，
 * 正确dlopen到本so并dlsym到对应的执行函数并调用成功"，不实现真实的AllReduce计算，
 * 因此函数体仅打印日志、将sendBuf搬运到recvBuf（若地址不同）后返回HCCL_SUCCESS。
 */

#include <cstdio>
#include <cstring>
#include <hccl/hccl_types.h>
#include <hccl/hccl_comm.h>
#include <acl/acl_rt.h>

namespace {
HcclResult FakeExecute(
    const char* algName, void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
    HcclComm comm, aclrtStream stream)
{
    (void)op;
    (void)comm;
    (void)stream;
    (void)sendBuf;
    (void)recvBuf;
    std::fprintf(
        stderr, "[AllReduceCustomAlgosImpl] %s invoked: count=%llu, dataType=%d\n", algName,
        static_cast<unsigned long long>(count), static_cast<int>(dataType));
    // 在这个ST模拟环境下，sendBuf/recvBuf是模拟器用于记录任务图的虚拟句柄，并非真实可读写的
    // 进程内存，这里不做任何内存拷贝，仅用打印验证"选择→派发→执行"链路是否走通。
    return HCCL_SUCCESS;
}
} // namespace

extern "C" HcclResult HcclAlgoPluginAllReduceSmall(
    void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm,
    aclrtStream stream)
{
    return FakeExecute("AllReduceAlgoSmall", sendBuf, recvBuf, count, dataType, op, comm, stream);
}

extern "C" HcclResult HcclAlgoPluginAllReduceLarge(
    void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm,
    aclrtStream stream)
{
    return FakeExecute("AllReduceAlgoLarge", sendBuf, recvBuf, count, dataType, op, comm, stream);
}
