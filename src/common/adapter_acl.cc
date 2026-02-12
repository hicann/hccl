/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "adapter_acl.h"
#include "acl_rt.h"
#include "workflow.h"
#include "mmpa_api.h"

namespace ops_hccl {
HcclResult haclrtGetDeviceIndexByPhyId(u32 devicePhyId, u32 &deviceLogicId)
{
#ifndef AICPU_COMPILE
    s32 phyDevId = static_cast<s32>(devicePhyId);
    s32 logicDevId;

    DevType deviceType;
    CHK_RET(hrtGetDeviceType(deviceType));
    if (deviceType == DevType::DEV_TYPE_NOSOC) {
        deviceLogicId = 0;
        return HCCL_SUCCESS;
    }
    aclError ret = aclrtGetLogicDevIdByPhyDevId(phyDevId, &logicDevId);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR("[Get][DeviceIndex]errNo[0x%016llx] rtGet device logicid by PhyId failed, return[%d], "\
            "para: phyId[%d], devIndex[%d]", HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, phyDevId, logicDevId);
        return HCCL_E_RUNTIME;
    }
    deviceLogicId = static_cast<u32>(logicDevId);
#endif
    return HCCL_SUCCESS;
};

HcclResult haclrtGetPairDeviceLinkType(s32 phyDevId, s32 otherPhyDevId, LinkTypeInServer &linkType)
{
#ifndef AICPU_COMPILE
    u32 logicIdLocal = 0;
    u32 logicIdDest = 0;
    CHK_RET(haclrtGetDeviceIndexByPhyId(phyDevId, logicIdLocal));
    CHK_RET(haclrtGetDeviceIndexByPhyId(otherPhyDevId, logicIdDest));
    HCCL_INFO("[haclrtGetPairDeviceLinkType]phyDevId[%u] otherPhyDevId[%u] logicIdLocal[%u] logicIdDest[%u]",
        phyDevId, otherPhyDevId, logicIdLocal, logicIdDest);

    u64 linkTypeRaw = 0;
    ACLCHECK(aclrtGetDevicesTopo(logicIdLocal, logicIdDest, reinterpret_cast<uint64_t*>(&linkTypeRaw)));
    HCCL_INFO("[haclrtGetPairDeviceLinkType]linkType[%u]", linkTypeRaw);

    // 若当前为标卡/虚拟机device间通过HCCS直接互联：HCCS_TYPE，device间通过HCCS交换芯片互联：TOPOLOGY_HCCS_SW
    // Ascend910_93 die间为SIO_TYPE，其他情况为PXI_TYPE
    switch (linkTypeRaw) {
        case ACL_RT_DEVS_TOPOLOGY_HCCS:
            linkType = LinkTypeInServer::HCCS_TYPE;
            break;
        case ACL_RT_DEVS_TOPOLOGY_HCCS_SW:
            linkType = LinkTypeInServer::HCCS_SW_TYPE;
            break;
        case ACL_RT_DEVS_TOPOLOGY_SIO:
            linkType = LinkTypeInServer::SIO_TYPE;
            break;
        default:
            linkType = LinkTypeInServer::PXI_TYPE;
    }
#endif
    return HCCL_SUCCESS;
}

HcclResult haclrtGetCaptureInfo(aclrtStream stream, aclmdlRICaptureStatus &captureStatus, u64 &modelId, bool &isCapture)
{
#ifndef AICPU_COMPILE
    isCapture = false;
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        HCCL_WARNING("[%s]Stream capture only support opbase mode!", __func__);
        return HCCL_SUCCESS;
    }
    aclmdlRI rtModel = nullptr;
    aclError ret = aclmdlRICaptureGetInfo(stream, &captureStatus, &rtModel);
    if (ret == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
        HCCL_WARNING("[%s]Stream capture does not support!", __func__);
        return HCCL_SUCCESS;
    } else {
        CHK_PRT_RET(ret != ACL_SUCCESS,
                    HCCL_ERROR("[%s]rtGet stream get capture status fail. return[%d]", __func__, ret), HCCL_E_RUNTIME);
    }
    if (captureStatus == ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE) {
        isCapture = true;
        modelId = reinterpret_cast<u64>(rtModel);  // 使用 rtModel 的地址作为 modelId
    }
    HCCL_DEBUG("[%s]captureStatus[%u] modelId[%llu] isCapture[%u]", __func__, captureStatus, modelId, isCapture);
#endif
    return HCCL_SUCCESS;
}

HcclResult hcalrtGetDeviceInfo(u32 deviceId, aclrtDevAttr devAttr, s64 &val)
{
#ifndef AICPU_COMPILE
    static const std::set<aclrtDevAttr> supportType = {
        {ACL_DEV_ATTR_PHY_CHIP_ID},
        {ACL_DEV_ATTR_SUPER_POD_DEVIDE_ID},
        {ACL_DEV_ATTR_SUPER_POD_SERVER_ID},
        {ACL_DEV_ATTR_SUPER_POD_ID},
        {ACL_DEV_ATTR_CUST_OP_PRIVILEGE}
    };

    auto it = supportType.find(devAttr);
    CHK_PRT_RET(it == supportType.end(),
                HCCL_ERROR("[hcalrtGetDeviceInfo]Unsupported aclrtDevAttr[%d].", devAttr),
                HCCL_E_NOT_SUPPORT);

    aclError ret = aclrtGetDeviceInfo(deviceId, devAttr, reinterpret_cast<int64_t *>(&val));
    CHK_PRT_RET(ret != ACL_SUCCESS,
        HCCL_ERROR("[hcalrtGetDeviceInfo]rt get device info failed. ret[%d], attr[%d], val[%ld]", ret, devAttr, val),
        HCCL_E_RUNTIME);
    HCCL_DEBUG("Call aclrtGetDeviceInfo, ret[%d], attr[%d], val[%ld]", ret, devAttr, val);
#endif
    return HCCL_SUCCESS;
}

HcclResult LoadBinaryFromFile(const char *binPath, aclrtBinaryLoadOptionType optionType, uint32_t cpuKernelMode,
    aclrtBinHandle &binHandle)
{
    CHK_PRT_RET(binPath == nullptr,
        HCCL_ERROR("[Load][Binary]binary path is nullptr"),
        HCCL_E_PTR);

    char realPath[PATH_MAX] = {0};
    CHK_PRT_RET(realpath(binPath, realPath) == nullptr,
        HCCL_ERROR("LoadBinaryFromFile: %s is not a valid real path, err[%d]", binPath, errno),
        HCCL_E_INTERNAL);
    HCCL_INFO("[LoadBinaryFromFile]realPath: %s", realPath);

    aclrtBinaryLoadOptions loadOptions = {0};
    aclrtBinaryLoadOption option;
    loadOptions.numOpt = 1;
    loadOptions.options = &option;
    option.type = optionType;
    option.value.cpuKernelMode = cpuKernelMode;
    aclError aclRet = aclrtBinaryLoadFromFile(realPath, &loadOptions, &binHandle); // ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE
    CHK_PRT_RET(aclRet != ACL_SUCCESS,
        HCCL_ERROR("[LoadBinaryFromFile]errNo[0x%016llx] load binary from file error.", aclRet),
        HCCL_E_OPEN_FILE_FAILURE);

    return HCCL_SUCCESS;
}

}
