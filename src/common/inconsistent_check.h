/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INCONSISTENT_CHECK_H
#define INCONSISTENT_CHECK_H

#include <string>

#include "alg_param.h"
#include "alg_type.h"

namespace ops_hccl {
bool NeedInconsistentCheck(const OpParam& param);

HcclResult CompareOpExchangeInfos(HcclComm comm, const OpParam& param, const AlgResourceRequest &resRequest,
    const OpExchangeInfo &exchangeInfo);

HcclResult InconsistentCheckParams(HcclComm comm, const OpExchangeInfo &exchangeInfo,
    const std::vector<HcclChannelDesc> &channels);

HcclResult InconsistentCheckOpType(const OpExchangeInfo &exchangeInfo, const HcclCMDType &rmtOpType);

HcclResult ReportOpExchangeInfoCheckFailed(const OpExchangeInfo &exchangeInfo, const std::string &paraName,
    uint32_t expectVal, uint32_t remotePara);

HcclResult ReportOpExchangeInfoCheckFailed(const OpExchangeInfo &exchangeInfo, const std::string &paraName,
    const std::string &expectVal, const std::string &remotePara);

HcclResult GetOpTypeName(const OpExchangeInfo &exchangeInfo, std::string &opInfo);

}
#endif // INCONSISTENT_CHECK_H