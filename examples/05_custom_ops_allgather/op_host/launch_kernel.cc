/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "launch_kernel.h"
#include "common.h"
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include "acl/acl.h"

namespace ops_hccl_allgather {

// Global variables for kernel registration
static bool g_init = false;
static std::mutex g_mut;
static aclrtBinHandle g_binHandle = nullptr;
static aclrtFuncHandle g_funcHandle = nullptr;

// Constants
const std::string AIV_BINARY_NAME = "hccl_custom_allgather_kernels.o";
const std::string KERNEL_NAME = "HcclAllGatherAivKernel";

static HcclResult LoadBinaryFromFile(const std::string& fileName, void*& buffer, size_t& length)
{
    std::ifstream file(fileName, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        HCCL_ERROR("[LoadBinaryFromFile] Failed to open file: %s", fileName.c_str());
        return HCCL_E_NOT_FOUND;
    }
    
    length = file.tellg();
    file.seekg(0, std::ios::beg);
    
    buffer = new(std::nothrow) char[length];
    if (buffer == nullptr) {
        HCCL_ERROR("[LoadBinaryFromFile] Failed to allocate memory for binary, size: %zu", length);
        return HCCL_E_INTERNAL;
    }
    
    if (!file.read(static_cast<char*>(buffer), length)) {
        HCCL_ERROR("[LoadBinaryFromFile] Failed to read file: %s", fileName.c_str());
        delete[] static_cast<char*>(buffer);
        buffer = nullptr;
        return HCCL_E_INTERNAL;
    }
    
    return HCCL_SUCCESS;
}

HcclResult RegisterKernel()
{
    std::lock_guard<std::mutex> guard(g_mut);
    if (g_init) {
        return HCCL_SUCCESS;
    }

    std::string binPath = AIV_BINARY_NAME;
    HCCL_INFO("[RegisterKernel] Binary path: %s", binPath.c_str());
    
    void* binBuffer = nullptr;
    size_t binSize = 0;
    CHK_RET(LoadBinaryFromFile(binPath, binBuffer, binSize));
    HCCL_INFO("[RegisterKernel] Binary loaded. Size: %zu", binSize);
    
    aclrtBinary binary = aclrtCreateBinary(binBuffer, binSize);
    if (binary == nullptr) {
        HCCL_ERROR("[RegisterKernel] aclrtCreateBinary failed");
        delete[] static_cast<char*>(binBuffer);
        return HCCL_E_INTERNAL;
    }
    
    aclError aclRet = aclrtBinaryLoad(binary, &g_binHandle);
    if (aclRet != ACL_SUCCESS) {
        HCCL_ERROR("[RegisterKernel] aclrtBinaryLoad failed, ret: %d", aclRet);
        aclrtDestroyBinary(binary);
        delete[] static_cast<char*>(binBuffer);
        return HCCL_E_INTERNAL;
    }
    
    aclrtDestroyBinary(binary);
    
    aclRet = aclrtBinaryGetFunction(g_binHandle, KERNEL_NAME.c_str(), &g_funcHandle);
    if (aclRet != ACL_SUCCESS) {
        HCCL_ERROR("[RegisterKernel] aclrtBinaryGetFunction failed for %s, ret: %d", KERNEL_NAME.c_str(), aclRet);
        return HCCL_E_INTERNAL;
    }

    g_init = true;
    HCCL_INFO("[RegisterKernel] Success. Function handle: %p", g_funcHandle);
    return HCCL_SUCCESS;
}

HcclResult ExecuteKernelLaunch(const OpParam &param, aclrtStream stream)
{
    if (!g_init) {
        CHK_RET(RegisterKernel());
    }

    aclrtLaunchKernelCfg cfg;
    aclrtLaunchKernelAttr attr[3]; 
    
    attr[0].id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
    attr[0].value.schemMode = 1;
    
    attr[1].id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT_US;
    attr[1].value.timeoutUs.timeoutLow = CUSTOM_TIMEOUT * 1000000;
    attr[1].value.timeoutUs.timeoutHigh = 0;
    
    attr[2].id = ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE;
    attr[2].value.engineType = ACL_RT_ENGINE_TYPE_AIV;
    
    cfg.numAttrs = 3;
    cfg.attrs = attr;

    HCCL_INFO("[ExecuteKernelLaunch] Invoking aclrtLaunchKernelWithHostArgs...");
    aclError aclRet = aclrtLaunchKernelWithHostArgs(g_funcHandle, 1, stream, &cfg, 
                                                    const_cast<OpParam*>(&param), sizeof(OpParam), 
                                                    nullptr, 0);
    
    if (aclRet != ACL_SUCCESS) {
        HCCL_ERROR("[ExecuteKernelLaunch] aclrtLaunchKernelWithHostArgs failed, ret: %d", aclRet);
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO("[ExecuteKernelLaunch] Launch command submitted successfully.");

    return HCCL_SUCCESS;
}

HcclResult LaunchKernel(OpParam &param, aclrtStream stream) {
    return ExecuteKernelLaunch(param, stream);
}

}

