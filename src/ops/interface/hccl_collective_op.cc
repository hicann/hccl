/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl.h"
#include "hccl_inner.h"

HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType,
                         HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    return HcclAllReduceInner(sendBuf, recvBuf, count, dataType, op, comm, stream);
}

HcclResult HcclBroadcast(void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm,
                         aclrtStream stream)
{
    return HcclBroadcastInner(buf, count, dataType, root, comm, stream);
}

HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
                             HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    return HcclReduceScatterInner(sendBuf, recvBuf, recvCount, dataType, op, comm, stream);
}

HcclResult HcclReduceScatterV(void *sendBuf, const void *sendCounts, const void *sendDispls,
    void *recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    return HcclReduceScatterVInner(sendBuf, sendCounts, sendDispls, recvBuf, recvCount, dataType, op, comm, stream);
}

HcclResult HcclAllGather(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType,
                         HcclComm comm, aclrtStream stream)
{
    return HcclAllGatherInner(sendBuf, recvBuf, sendCount, dataType, comm, stream);
}

HcclResult HcclAllGatherV(void *sendBuf, uint64_t sendCount, void *recvBuf,
    const void *recvCounts, const void *recvDispls, HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    return HcclAllGatherVInner(sendBuf, sendCount, recvBuf, recvCounts, recvDispls, dataType, comm, stream);
}

HcclResult HcclSend(void* sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank,
                    HcclComm comm, aclrtStream stream)
{
    return HcclSendInner(sendBuf, count, dataType, destRank, comm, stream);
}

HcclResult HcclRecv(void* recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank,
                    HcclComm comm, aclrtStream stream)
{
    return HcclRecvInner(recvBuf, count, dataType, srcRank, comm, stream);
}

HcclResult HcclAlltoAllVC(const void *sendBuf, const void *sendCountMatrix,
    HcclDataType sendType, const void *recvBuf, HcclDataType recvType,
    HcclComm comm, aclrtStream stream)
{
    return HcclAlltoAllVCInner(sendBuf, sendCountMatrix, sendType, recvBuf, recvType, comm, stream);
}

HcclResult HcclAlltoAllV(const void *sendBuf, const void *sendCounts, const void *sdispls, HcclDataType sendType,
                         const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType,
                         HcclComm comm, aclrtStream stream)
{
    return HcclAlltoAllVInner(sendBuf, sendCounts, sdispls, sendType, recvBuf, recvCounts, rdispls, recvType, comm, stream);
}

HcclResult HcclAlltoAll(const void *sendBuf, uint64_t sendCount, HcclDataType sendType, const void *recvBuf,
    uint64_t recvCount, HcclDataType recvType, HcclComm comm, aclrtStream stream)
{
    return HcclAlltoAllInner(sendBuf, sendCount, sendType, recvBuf, recvCount, recvType, comm, stream);
}

HcclResult HcclReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
                      uint32_t root, HcclComm comm, aclrtStream stream)
{
    return HcclReduceInner(sendBuf, recvBuf, count, dataType, op, root, comm, stream);
}

HcclResult HcclBatchSendRecvInner(HcclSendRecvItem* sendRecvInfo, uint32_t itemNum, HcclComm comm, aclrtStream stream)
{
    return HcclBatchSendRecvInner(sendRecvInfo, itemNum, comm, stream);
}
