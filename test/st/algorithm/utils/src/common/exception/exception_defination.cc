/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exception_defination.h"

namespace HcclSim {

HcclResult ExceptionInfo::GetErrorCode(const ExceptionType &type)
{
    static const std::map<ExceptionType, HcclResult> errorCodeMap{
        {ExceptionType::NULL_PTR_EXCEPTION, HcclResult::HCCL_E_PTR},
        {ExceptionType::INVALID_PARAMS_EXCEPTION, HcclResult::HCCL_E_PARA},
        {ExceptionType::TIMEOUT_EXCEPTION, HcclResult::HCCL_E_TIMEOUT},
        {ExceptionType::DRV_API_EXCEPTION, HcclResult::HCCL_E_DRV},
        {ExceptionType::NETWORK_API_EXCEPTION, HcclResult::HCCL_E_NETWORK},
        {ExceptionType::RUNTIME_API_EXCEPTION, HcclResult::HCCL_E_RUNTIME},
        {ExceptionType::RESOURCES_NOT_EXIST_EXCEPTION, HcclResult::HCCL_E_NOT_FOUND},
        {ExceptionType::NOT_SUPPORT_EXCEPTION, HcclResult::HCCL_E_NOT_SUPPORT},
        {ExceptionType::INTERNAL_EXCEPTION, HcclResult::HCCL_E_INTERNAL},
        {ExceptionType::RMA_CONN_EXCEPTION, HcclResult::HCCL_E_ROCE_CONNECT},
        {ExceptionType::SOCKET_EXCEPTION, HcclResult::HCCL_E_TCP_CONNECT},
        {ExceptionType::RES_UNPACKAGE_EXCEPTION, HcclResult::HCCL_E_INTERNAL},
        {ExceptionType::HOST_IP_NOT_FOUND_EXCEPTION, HcclResult::HCCL_E_NOT_FOUND},
        {ExceptionType::TRACE_API_EXCEPTION, HcclResult::HCCL_E_INTERNAL},
        {ExceptionType::CCU_API_EXCEPTION, HcclResult::HCCL_E_INTERNAL},
        {ExceptionType::SUSPENDING_EXCEPTION, HcclResult::HCCL_E_SUSPENDING}};
    return errorCodeMap.at(type);
}

std::string ExceptionInfo::GetErrorMsg(const ExceptionType &type)
{
    static const std::map<ExceptionType, std::string> errorMsgMap{
        {ExceptionType::NULL_PTR_EXCEPTION, "Null pointer exception: "},
        {ExceptionType::INVALID_PARAMS_EXCEPTION, "Invalid params exception: "},
        {ExceptionType::TIMEOUT_EXCEPTION, "Time out exception: "},
        {ExceptionType::DRV_API_EXCEPTION, "Drv api exception: "},
        {ExceptionType::NETWORK_API_EXCEPTION, "Network api exception: "},
        {ExceptionType::RUNTIME_API_EXCEPTION, "Runtime api exception: "},
        {ExceptionType::RESOURCES_NOT_EXIST_EXCEPTION, "Resources not exist exception: "},
        {ExceptionType::NOT_SUPPORT_EXCEPTION, "Not support exception: "},
        {ExceptionType::INTERNAL_EXCEPTION, "Internal exception: "},
        {ExceptionType::RMA_CONN_EXCEPTION, "RMA connection exception: "},
        {ExceptionType::SOCKET_EXCEPTION, "Socket exception: "},
        {ExceptionType::RES_UNPACKAGE_EXCEPTION, "Resource unpackage exception:"},
        {ExceptionType::HOST_IP_NOT_FOUND_EXCEPTION, "environment variable exception: "},
        {ExceptionType::TRACE_API_EXCEPTION, "Trace api exception: "},
        {ExceptionType::CCU_API_EXCEPTION, "Ccu api exception: "},
        {ExceptionType::SUSPENDING_EXCEPTION, "Suspending exception: "}};
    return errorMsgMap.at(type);
}
}
