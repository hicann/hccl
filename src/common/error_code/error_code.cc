/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "base/err_msg.h"
#include "log.h"
#include <string>
#include <stdio.h>
namespace {

const std::string hccl_g_msg = R"(
{
    "error_info_list": [
    {
      "errClass": "HCCL Errors",
      "errTitle": "Invalid_Argument_Collective_Communication_Operator",
      "ErrCode": "EI0003",
      "ErrMessage": "In [%s], value [%s] for parameter [%s] is invalid. Reason: The collective communication operator has an invalid argument. Reason[%s]",
      "Arglist": "ccl_op,value,parameter,value",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Try again with a valid argument."
      }
    }
  ]
}
)";
}

REG_FORMAT_ERROR_MSG(hccl_g_msg.c_str(), hccl_g_msg.size());