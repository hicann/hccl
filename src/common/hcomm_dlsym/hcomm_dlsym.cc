/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcomm_dlsym.h"
#include "hccl_res_dl.h"
#include "ccu_res_dl.h"
#include "hccl_ccu_res_dl.h"
#include "ccu_launch_dl.h"
#include "ccu_primitives_impl_dl.h"
#include "hccl_rank_graph_dl.h"
#include "hcomm_primitives_dl.h"
#include "hccl_inner_dl.h"
#include "hcomm_host_profiling_dl.h"
#include "hccl_host_comm_dl.h"
#include "hccl_res_expt_dl.h"
#include "hccl_sym_win_dl.h"
#include <pthread.h>
#include <dlfcn.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <acl/acl.h>

static void* gLibHandle = nullptr;
static std::atomic<int> gHcommVersion{0};
static std::mutex gHcommDlInitMutex;

void* HcommGetLibHandle(void) { return gLibHandle; }

int GetHcommVersion(void)
{
    // gHcommVersion 为自包含缓存值，无伴随数据发布，relaxed 序足够；查询失败不缓存，下次调用重试
    int version = gHcommVersion.load(std::memory_order_relaxed);
    if (version != 0) {
        return version;
    }

    char hcommPkgName[] = "hcomm";
    int tmpVersion = 0;
    if (aclsysGetVersionNum(hcommPkgName, &tmpVersion) == ACL_SUCCESS) {
        gHcommVersion.store(tmpVersion, std::memory_order_relaxed);
        version = tmpVersion;
    }

    return version;
}

bool HcommIsProfilingSupported()
{
    if (GetHcommVersion() >= CANN_VERSION(9, 0, 0)) {
        return true;
    } else {
        return false;
    }
}

bool HcommIsExportThreadSupported()
{
    if (GetHcommVersion() >= CANN_VERSION(9, 0, 0) && HcommIsSupportHcclThreadExportToCommEngine()) {
        return true;
    } else {
        return false;
    }
}

// 实际初始化逻辑，由 HcommDlInit 持锁调用；加载失败时 gLibHandle 保持为空，后续可重试
static void DoHcommDlInit(void)
{
    gLibHandle = HcclDlopen("libhcomm.so", RTLD_NOW);
    if (!gLibHandle) {
        fprintf(stderr, "[HcclWrapper] Failed to open libhcomm: %s\n", dlerror());
        return;
    }

    dlerror();

    HcclResDlInit(gLibHandle);
    HcclRankGraphDlInit(gLibHandle);
    HcommPrimitivesDlInit(gLibHandle);
    HcclInnerDlInit(gLibHandle);
    HcommProfilingDlInit(gLibHandle);
    HcclCommDlInit(gLibHandle);
    HcclResExptDlInit(gLibHandle);
    CcuResDlInit(gLibHandle);
    HcclCcuResDlInit(gLibHandle);
    CcuLaunchDlInit(gLibHandle);
    CcuPrimitivesImplDlInit(gLibHandle);
    HcclSymWinHostDlInit(gLibHandle);
}

// 初始化，线程安全：通过互斥锁串行化；成功初始化后不重复执行，失败时允许后续重试
void HcommDlInit(void)
{
    std::lock_guard<std::mutex> lock(gHcommDlInitMutex);
    if (gLibHandle != nullptr) {
        return;
    }

    DoHcommDlInit();
}
