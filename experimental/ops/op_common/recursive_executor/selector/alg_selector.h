/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALG_SELECTOR_H
#define ALG_SELECTOR_H

#include <map>
#include <mutex>
#include <string>

#include "algo_desc.h"

namespace ops_hccl {

class AlgSelector {
public:
    static AlgSelector& Instance();

    HcclResult Register(const std::string& algName, HcclAlgorithm algo);
    bool GetAlgorithm(const std::string& algName, HcclAlgorithm& algo) const;

private:
    AlgSelector() = default;
    std::map<std::string, HcclAlgorithm> algMap_;
    mutable std::mutex mu_;
};

} // namespace ops_hccl

#endif // ALG_SELECTOR_H
