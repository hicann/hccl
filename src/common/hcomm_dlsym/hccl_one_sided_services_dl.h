/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_ONE_SIDED_SERVICES_DL_H
#define HCCL_ONE_SIDED_SERVICES_DL_H

// #include "hccl_one_sided_services.h"   // 原始头文件，包含所有声明和类型定义
#include <hccl/hccl_types.h>
#include <hccl/base.h>

#ifdef __cplusplus
extern "C" {
#endif

// 需优化
typedef enum {
    HCCL_MEM_TYPE_DEVICE, ///< 设备侧内存（如NPU等）
    HCCL_MEM_TYPE_HOST,   ///< 主机侧内存
    HCCL_MEM_TYPE_NUM     ///< 内存类型数量
} HcclMemType;
/**
 * @struct HcclMem
 * @brief 内存段元数据描述结构体
 * @var type  - 内存物理位置类型，参见HcclMemType
 * @var addr  - 内存虚拟地址
 * @var size  - 内存区域字节数
 */
typedef struct {
    HcclMemType type;
    void *addr;
    uint64_t size;
} HcclMem;

const u32 HCCL_MEM_DESC_LENGTH = 511;

typedef struct {
    char desc[HCCL_MEM_DESC_LENGTH + 1]; // 具体内容对调用者不可见
} HcclMemDesc;

typedef struct {
    HcclMemDesc* array;
    u32 arrayLength;
} HcclMemDescs;

typedef struct {
    void* localAddr; // 本端VA
    void* remoteAddr; // 远端VA
    u64 count;
    HcclDataType dataType;
} HcclOneSideOpDesc;

typedef enum {
    HCCL_TOPO_FULLMESH = 0, // fullmesh连接
    HCCL_TOPO_NUM,
} HcclTopoType;

typedef struct {
    HcclTopoType topoType;
    u64 rsvd0;
    u64 rsvd1;
    u64 rsvd2;
} HcclPrepareConfig;

#ifndef HCCL_E_NOT_SUPPORTED
#define HCCL_E_NOT_SUPPORTED  ((HcclResult)(-2))
#endif

// 对外 API 的包装函数声明
HcclResult HcclRegisterMem(HcclComm comm, u32 remoteRank, int type, void* addr, u64 size, HcclMemDesc* desc);
HcclResult HcclDeregisterMem(HcclComm comm, HcclMemDesc* desc);
HcclResult HcclExchangeMemDesc(HcclComm comm, u32 remoteRank, HcclMemDescs* local, int timeout, HcclMemDescs* remote, u32* actualNum);
HcclResult HcclEnableMemAccess(HcclComm comm, HcclMemDesc* remoteMemDesc, HcclMem* remoteMem);
HcclResult HcclDisableMemAccess(HcclComm comm, HcclMemDesc* remoteMemDesc);
HcclResult HcclBatchPut(HcclComm comm, u32 remoteRank, HcclOneSideOpDesc* desc, u32 descNum, rtStream_t stream);
HcclResult HcclBatchGet(HcclComm comm, u32 remoteRank, HcclOneSideOpDesc* desc, u32 descNum, rtStream_t stream);
HcclResult HcclRemapRegistedMemory(HcclComm *comm, HcclMem *memInfoArray, u64 commSize, u64 arraySize);
HcclResult HcclRegisterGlobalMem(const HcclMem* mem, void** memHandle);
HcclResult HcclDeregisterGlobalMem(void* memHandle);
HcclResult HcclCommBindMem(HcclComm comm, void* memHandle);
HcclResult HcclCommUnbindMem(HcclComm comm, void* memHandle);
HcclResult HcclCommPrepare(HcclComm comm, const HcclPrepareConfig* prepareConfig, const int timeout);

// 查询函数声明
bool HcommIsSupportHcclRegisterMem(void);
bool HcommIsSupportHcclDeregisterMem(void);
bool HcommIsSupportHcclExchangeMemDesc(void);
bool HcommIsSupportHcclEnableMemAccess(void);
bool HcommIsSupportHcclDisableMemAccess(void);
bool HcommIsSupportHcclBatchPut(void);
bool HcommIsSupportHcclBatchGet(void);
bool HcommIsSupportHcclRemapRegistedMemory(void);
bool HcommIsSupportHcclRegisterGlobalMem(void);
bool HcommIsSupportHcclDeregisterGlobalMem(void);
bool HcommIsSupportHcclCommBindMem(void);
bool HcommIsSupportHcclCommUnbindMem(void);
bool HcommIsSupportHcclCommPrepare(void);

// 动态库管理接口
void HcclOneSidedServicesDlInit(void* libHcommHandle);
void HcclOneSidedServicesDlFini(void);

#ifdef __cplusplus
}
#endif

#endif // HCCL_ONE_SIDED_SERVICES_DL_H