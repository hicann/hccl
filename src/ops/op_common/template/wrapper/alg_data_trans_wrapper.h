/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALG_DATA_TRANS_WRAPPER
#define ALG_DATA_TRANS_WRAPPER

#include "alg_v2_template_base.h"

namespace ops_hccl {

HcclResult SendWrite(const DataInfo &snedInfo, const ThreadHandle &thread);

HcclResult RecvWrite(const DataInfo &recvInfo, const ThreadHandle &thread);

HcclResult SendRecvWrite(const SendRecvInfo &sendRecvInfo, const ThreadHandle &thread);

HcclResult SendWriteReduce(const DataReduceInfo &snedInfo, const ThreadHandle &thread);

HcclResult RecvWriteReduce(const DataReduceInfo &recvInfo, const ThreadHandle &thread);

HcclResult SendRecvWriteReduce(const SendRecvReduceInfo &snedInfo, const ThreadHandle &thread);

HcclResult SendRead(const DataInfo &snedInfo, const ThreadHandle &thread);

HcclResult RecvRead(const DataInfo &recvInfo, const ThreadHandle &thread);

HcclResult SendRecvRead(const SendRecvInfo &snedInfo, const ThreadHandle &thread);

HcclResult SendReadReduce(const DataReduceInfo &snedInfo, const ThreadHandle &thread);

HcclResult RecvReadReduce(const DataReduceInfo &recvInfo, const ThreadHandle &thread);

HcclResult SendRecvReadReduce(const SendRecvReduceInfo &sendRecvInfo, const ThreadHandle &thread);

HcclResult LocalCopy(const ThreadHandle &thread, const DataSlice &srcSlice, const DataSlice &dstSlice);

HcclResult LocalReduce(const ThreadHandle &thread, const DataSlice &srcSlice, const DataSlice &dstSlice,
                       const HcclDataType dataType, const HcclReduceOp reduceOp);

HcclResult LocalCopySlices(const ThreadHandle &thread, const std::vector<DataSlice> &srcSlices,
                           const std::vector<DataSlice> &dstSlices);

bool IsContinuousSlice(const DataSlice &nxtSlice, const DataSlice &currSlice);

HcclResult PreSyncInterThreads(const ThreadHandle &mainThread, const std::vector<ThreadHandle> &subThreads,
                               const std::vector<u32> &notifyIdxMainToSub);

HcclResult PostSyncInterThreads(const ThreadHandle &mainThread, const std::vector<ThreadHandle> &subThreads,
                                const std::vector<u32> &notifyIdxSubToMain);

HcclResult AicpuReduce(const ThreadHandle &thread, const DataSlice &srcSlice, const DataSlice &dstSlice,
                       const HcclDataType dataType, const HcclReduceOp reduceOp);
 
template <typename T>
HcclResult AicpuReduceTemplate(T* dst, u64 dstSize, T* src, u64 srcSize, const HcclReduceOp reduceOp);
}

#endif // !ALG_DATA_TRANS_WRAPPER