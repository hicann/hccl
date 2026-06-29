/* 
  * Copyright (c) 2025 Huawei Technologies Co., Ltd. 
  * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
  * CANN Open Software License Agreement Version 2.0 (the "License"). 
  * Please refer to the License for details. You may not use this file except in compliance with the License. 
  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
  * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
  * See LICENSE in the root of the software repository for the full text of the License. 
  */ 
 
 
 #include "omnipipe_template_utils.h" 
 #include <vector>
 
 namespace ops_hccl { 
 HcclResult FillOmniPipeTemplateAlgParams(TemplateDataParams &tempAlgParams, const StepSliceInfo &stepSliceInfo) 
 { 
    tempAlgParams.buffInfo.inBuffType = BufferType::HCCL_BUFFER;
    tempAlgParams.buffInfo.outBuffType = BufferType::HCCL_BUFFER;
    tempAlgParams.buffInfo.inBuffBaseOff = stepSliceInfo.buffInfo.inBuffBaseOff;
    tempAlgParams.buffInfo.outBuffBaseOff = stepSliceInfo.buffInfo.outBuffBaseOff;
    tempAlgParams.buffInfo.hcclBuffBaseOff = stepSliceInfo.buffInfo.hcclBuffBaseOff;
    tempAlgParams.stepSliceInfo = stepSliceInfo;
    return HcclResult::HCCL_SUCCESS;
 } 
 
 
HcclResult PrepareOmniPipeDataSplitForMultiChannel(CommonAlgTemplateBase* algTemplate, 
     const TemplateDataParams &tempAlgParams, HcclDataType dataType, TemplateResource &templateResource, 
     std::vector<std::vector<std::vector<u64>>> &dataSplitVec, 
     std::vector<std::vector<std::vector<u64>>> &dataOffsetVec) 
 { 
    dataSplitVec.clear();
    dataOffsetVec.clear();
    u32 dataTypeSize = DATATYPE_SIZE_TABLE[dataType];
    for (uint32_t idx = 0; idx < tempAlgParams.stepSliceInfo.stepSliceSize.size(); idx++) { 
        std::vector<std::vector<u64>> dataSplitVecByRepeat;
        std::vector<std::vector<u64>> dataOffsetVecByRepeat;
        for (uint32_t rpt = 0; rpt < tempAlgParams.stepSliceInfo.stepSliceSize[0].size(); rpt++) {
            u64 totalDataCount = tempAlgParams.stepSliceInfo.stepSliceSize[idx][rpt] / dataTypeSize;
            std::vector<u64> dataSplit;
            std::vector<u64> dataOffset;
            std::vector<u64> curElemCountOut;
            algTemplate->CalcDataSplitByPortGroup(totalDataCount, dataTypeSize,
                templateResource.channels.begin()->second, curElemCountOut, dataSplit, dataOffset);
            dataSplitVecByRepeat.push_back(dataSplit);
            dataOffsetVecByRepeat.push_back(dataOffset);
        } 
        dataSplitVec.push_back(dataSplitVecByRepeat); 
        dataOffsetVec.push_back(dataOffsetVecByRepeat); 
    } 
    return HcclResult::HCCL_SUCCESS;
 } 
 } // namespace ops_hccl 
