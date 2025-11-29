/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "adapter_error_manager_pub.h"
#include "base/err_mgr.h"

void RptInputErr(std::string error_code, std::vector<std::string> key,
    std::vector<std::string> value)
{
    // 将 std::vector<std::string> 转换为 std::vector<const char*>
    std::vector<const char*> key_cstr;
    for (const auto& k : key) {
        key_cstr.push_back(k.c_str());
    }
    
    std::vector<const char*> value_cstr;
    for (const auto& v : value) {
        value_cstr.push_back(v.c_str());
    }
    
    REPORT_PREDEFINED_ERR_MSG(error_code.c_str(), key_cstr, value_cstr);
    return;
}
 
void RptEnvErr(std::string error_code, std::vector<std::string> key,
    std::vector<std::string> value)
{
    // 将 std::vector<std::string> 转换为 std::vector<const char*>
    std::vector<const char*> key_cstr;
    for (const auto& k : key) {
        key_cstr.push_back(k.c_str());
    }
    
    std::vector<const char*> value_cstr;
    for (const auto& v : value) {
        value_cstr.push_back(v.c_str());
    }
    
    REPORT_PREDEFINED_ERR_MSG(error_code.c_str(), key_cstr, value_cstr);

    return;
}