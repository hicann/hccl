/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ops_proto_hccl.h
 * \brief huawei collective communication library ops.
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_OPS_PROTO_HCCL_H_
#define OPS_BUILT_IN_OP_PROTO_INC_OPS_PROTO_HCCL_H_

#ifndef OPS_BUILT_IN_OP_PROTO_INC_OPS_PROTO_HCCL_H_WARN_SHOWN
#define OPS_BUILT_IN_OP_PROTO_INC_OPS_PROTO_HCCL_H_WARN_SHOWN
#warning "ops_proto_hccl.h is scheduled to be deprecated in December 2026, and will be replaced by the \
ops_proto_math.h, ops_proto_cv.h, ops_proto_nn.h, ops_proto_transformer.h, ops_proto_legacy.h. \
We apologize for any inconvenience caused and appreciate your timely migration to the new interface."

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Outputs a tensor gathering all input tensors.
 * @par Inputs:
 * x: A tensor. Must be one of the following types: int8, int16, int32, int64, float16, bfloat16,
  float32, uint8, uint16, uint32, float64.
 * @par Attributes:
 * @li rank_size: A required integer identifying the number of ranks
  participating in the op.
 * @li group: A required string identifying the group name of ranks
  participating in the op.
 * @li fusion: An optional integer identifying the fusion flag of the op.
  0: no fusion; 2: fusion the ops by fusion id.
 * @li fusion_id: An optional integer identifying the fusion id of the op.
 * The HcomAllGather ops with the same fusion id will be fused.
 * @par Outputs:
 * y: A Tensor. Has the same type as "x".
 * @attention Constraints:
  "group" is limited to 128 characters. Use "hccl_world_group"
  as the name of a world group.
 */
REG_OP(HcomAllGather)
    .INPUT(
        x, TensorType(
               {DT_FLOAT, DT_INT32, DT_INT8, DT_INT16, DT_FLOAT16, DT_BFLOAT16, DT_INT64, DT_UINT64, DT_UINT8,
                DT_UINT16, DT_UINT32, DT_FLOAT64}))
    .OUTPUT(
        y, TensorType(
               {DT_FLOAT, DT_INT32, DT_INT8, DT_INT16, DT_FLOAT16, DT_BFLOAT16, DT_INT64, DT_UINT64, DT_UINT8,
                DT_UINT16, DT_UINT32, DT_FLOAT64}))
    .REQUIRED_ATTR(rank_size, Int)
    .REQUIRED_ATTR(group, String)
    .ATTR(fusion, Int, 0)
    .ATTR(fusion_id, Int, -1)
    .OP_END_FACTORY_REG(HcomAllGather)

    /**
     * @brief Outputs a tensor gathering all input tensors.
     * @par Inputs:
     * @li x: A tensor. Must be one of the following types: int8, int16, int32, int64, float16, bfloat16,
      float32, uint8, uint16, uint32, uint64, float64.
     * @li send_count: A data. specifies current rank the number of
      elements to receive to send, only support int64.
     * @li recv_counts: A list, where entry i specifies the first dimension of
      elements to receive from rank i, only support int64.
     * @li recv_displacements: A list, where entry i specifies the displacement
      (offset from recv_data) to which data from rank i should be written, only support int64.
     * @par Attributes:
     * group: A required string identifying the group name of ranks
      participating in the op.
     * @par Outputs:
     * y: A Tensor. Has the same type as "x".
     * @attention Constraints:
     * @li "group" is limited to 128 characters. Use "hccl_world_group"
      as the name of a world group.
     * @li Only the single-node system scenario of the Atlas A2 Training Series Product is supported.
     */
    REG_OP(HcomAllGatherV)
    .INPUT(
        x, TensorType(
               {DT_FLOAT, DT_INT32, DT_INT8, DT_INT16, DT_FLOAT16, DT_BFLOAT16, DT_INT64, DT_UINT64, DT_UINT8,
                DT_UINT16, DT_UINT32, DT_FLOAT64}))
    .INPUT(send_count, TensorType({DT_INT64}))
    .OUTPUT(
        y, TensorType(
               {DT_FLOAT, DT_INT32, DT_INT8, DT_INT16, DT_FLOAT16, DT_BFLOAT16, DT_INT64, DT_UINT64, DT_UINT8,
                DT_UINT16, DT_UINT32, DT_FLOAT64}))
    .INPUT(recv_counts, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(recv_displacements, TensorType({DT_INT64}))
    .REQUIRED_ATTR(group, String)
    .OP_END_FACTORY_REG(HcomAllGatherV)

    /**
     * @brief Outputs a tensor containing the reduction across all input tensors
      passed to op.
     * @par Inputs:
     * x: A tensor. Must be one of the following types: int8, int16, int32, int64, float16,
      float32.
     * @par Attributes:
     * @li reduction: A required string identifying the reduction operation to
      perform.The supported operation are: "sum", "max", "min", "prod".
     * @li group: A required string identifying the group name of ranks
      participating in the op.
     * @li fusion: An optional integer identifying the fusion flag of the op.
      0: no fusion; 1 (default): fusion the ops by gradient segmentation strategy; 2: fusion the ops by fusion id.
     * @li fusion_id: An optional integer identifying the fusion id of the op.
     * The HcomAllReduce ops with the same fusion id will be fused.
     * @par Outputs:
     * y: A Tensor. Has the same type as "x".
     * @attention Constraints:
     * @li "group" is limited to 128 characters. Use "hccl_world_group"
      as the name of a world group.
     * @li For Altas 300I Duo, "prod"/"max"/"min" does not support int16
     * @li For Altas A2, "prod" does not support int16/bfp16
     * @li For Altas A3, "prod" does not support int16/bfp16
     */
    REG_OP(HcomAllReduce)
    .INPUT(x, TensorType({DT_FLOAT, DT_INT32, DT_INT8, DT_INT16, DT_FLOAT16, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32, DT_INT8, DT_INT16, DT_FLOAT16, DT_INT64}))
    .REQUIRED_ATTR(reduction, String)
    .REQUIRED_ATTR(group, String)
    .ATTR(fusion, Int, 1)
    .ATTR(fusion_id, Int, -1)
    .OP_END_FACTORY_REG(HcomAllReduce)

    /**
     * @brief Broadcasts the input tensor in root rank to all ranks.
     * @par Inputs:
     * x: A list of dynamic input tensor. Must be one of the following types:
      int8, int16, int32, float16, float32. It's a dynamic input.
     * @par Attributes:
     * @li root_rank: A required integer identifying the root rank in the op
      input of this rank will be broadcast to other ranks.
     * @li fusion: A required integer identifying if the op need to fusion,
      0: no fusion; 2(default): fusion the ops by fusion id.
      * @li fusion_id: A required integer identifying the fusion id if para fusion
      is set.
     * @li group: A required string identifying the group name of ranks
      participating in the op.
     * @par Outputs:
     * y: A list of dynamic output tensor. Has the same type and length as "x".
     * It's a dynamic output.
     * @attention Constraints:
      "group" is limited to 128 characters. Use "hccl_world_group"
      as the name of a world group.
     */
    REG_OP(HcomBroadcast)
    .DYNAMIC_INPUT(
        x, TensorType(
               {DT_FLOAT, DT_INT32, DT_INT8, DT_INT16, DT_FLOAT16, DT_INT64, DT_UINT64, DT_UINT8, DT_UINT16, DT_UINT32,
                DT_FLOAT64}))
    .DYNAMIC_OUTPUT(
        y, TensorType(
               {DT_FLOAT, DT_INT32, DT_INT8, DT_INT16, DT_FLOAT16, DT_INT64, DT_UINT64, DT_UINT8, DT_UINT16, DT_UINT32,
                DT_FLOAT64}))
    .REQUIRED_ATTR(root_rank, Int)
    .REQUIRED_ATTR(group, String)
    .ATTR(fusion, Int, 0)
    .ATTR(fusion_id, Int, -1)
    .OP_END_FACTORY_REG(HcomBroadcast)

    /**
     * @brief preforms reduction from others rank to rootrank
     * @par Inputs:
    * @li root_rank: A required integer identifying the root rank in the op
      the reduction result will be on this root rank
     * x: A tensor. Must be one of the following types: int8, int16, int32, int64, float16,
      float32.
     * @par Attributes:
     * @li reduction: A required string identifying the reduction operation to
      perform.The supported operation are: "sum", "max", "min", "prod".
     * @li group: A required string identifying the group name of ranks
      participating in the op.
     * @li fusion: An optional integer identifying the fusion flag of the op.
      0: no fusion; 1 (default): fusion; 2: fusion the ops by fusion id.
     * @li fusion_id: An optional integer identifying the fusion id of the op.
     * The HcomReduce ops with the same fusion id will be fused.
     * @par Outputs:
     * y: A Tensor. Has the same type as "x".
     * @attention Constraints:
     *"group" is limited to 128 characters. Use "hccl_world_group"
      as the name of a world group.
     */
    REG_OP(HcomReduce)
    .INPUT(x, TensorType({DT_FLOAT, DT_INT32, DT_INT8, DT_INT16, DT_FLOAT16, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32, DT_INT8, DT_INT16, DT_FLOAT16, DT_INT64}))
    .REQUIRED_ATTR(root_rank, Int)
    .REQUIRED_ATTR(reduction, String)
    .REQUIRED_ATTR(group, String)
    .ATTR(fusion, Int, 0)
    .ATTR(fusion_id, Int, -1)
    .OP_END_FACTORY_REG(HcomReduce)
    /**
     * @brief Performs reduction across all input tensors, scattering in equal
      blocks among ranks, each rank getting a chunk of data based on its rank
      index.
     * @par Inputs:
     * x: A tensor. Must be one of the following types: int8, int16, int32, int64, float16,
      float32.
     * @par Attributes:
     * @li reduction: A required string identifying the reduction operation to
      perform. The supported operation are: "sum", "max", "min", "prod".
     * @li fusion: An optional integer identifying the fusion flag of the op.
      0: no fusion; 2: fusion the ops by fusion id.
     * @li fusion_id: An optional integer identifying the fusion id of the op.
     * The HcomReduceScatter ops with the same fusion id will be fused.
     * @li group: A required string identifying the group name of ranks
      participating in the op.
     * @li rank_size: A required integer identifying the number of ranks
      participating in the op.
     * @par Outputs:
     * y: A Tensor. Has the same type as "x".
     * @attention Constraints:
      "group" is limited to 128 characters. Use "hccl_world_group"
      as the name of a world group.
     */
    REG_OP(HcomReduceScatter)
    .INPUT(x, TensorType({DT_FLOAT, DT_INT32, DT_INT8, DT_INT16, DT_FLOAT16, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32, DT_INT8, DT_INT16, DT_FLOAT16, DT_INT64}))
    .REQUIRED_ATTR(reduction, String)
    .ATTR(fusion, Int, 0)
    .ATTR(fusion_id, Int, -1)
    .REQUIRED_ATTR(group, String)
    .REQUIRED_ATTR(rank_size, Int)
    .OP_END_FACTORY_REG(HcomReduceScatter)

    /**
     * @brief Performs reduction across all input tensors, scattering in vary size
      blocks among ranks, each rank getting a chunk of data based on its rank
      index.
     * @par Inputs:
     * @li x: A tensor. Must be one of the following types: int8, int16, int32, float16, float32.
     * @li send_counts: int64 array, where entry i specifies the first dimension number of elements to send to rank i.
     * @li send_displacesments: int64 array, optional, where entry i specifies the displacement from which to send data
     to rank i. If not provided, it is assumed to be contiguous memory by default.
     * @li recv_count: int64 array, only one entry which specifies the first dimension number of elements of the output
     data.
     * @par Attributes:
     * @li reduction: A required string identifying the reduction operation to
      perform. The supported operation are: "sum", "max", "min".
     * @li group: A required string identifying the group name of ranks
      participating in the op.
     * @par Outputs:
     * y: A Tensor. Has the same type as "x".
     * @attention Constraints:
     * @li "group" is limited to 128 characters. Use "hccl_world_group"
      as the name of a world group.
     * @li Only the single-node system scenario of the Atlas A2 Training Series Product is supported.
     */
    REG_OP(HcomReduceScatterV)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT16, DT_INT8}))
    .INPUT(recv_count, TensorType({DT_INT64}))
    .INPUT(send_counts, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(send_displacements, TensorType({DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT16, DT_INT8}))
    .REQUIRED_ATTR(reduction, String)
    .REQUIRED_ATTR(group, String)
    .OP_END_FACTORY_REG(HcomReduceScatterV)

    /**
     * @brief Sends the input tensor to destination rank.
     * @par Inputs:
     * x: A tensor. Must be one of the following types: int8, int16, int32, float16,
      float32.
     * @par Attributes:
     * @li sr_tag: A required integer identifying the send/recv message tag. The
       message will be received by the HcomReceive op with the same "sr_tag".
     * @li dest_rank: A required integer identifying the destination rank.
     * @li group: A string identifying the group name of ranks participating in
      the op.
     * @par Outputs:
     * None.
     * @attention Constraints:
      @li "group" is limited to 128 characters. Use
      "hccl_world_group" as the name of a world group.
     * @li Operators HcomSend and HcomReceive have the same "sr_tag".
     * @see HcomReceive
    */
    REG_OP(HcomSend)
    .INPUT(
        x, TensorType(
               {DT_FLOAT, DT_INT32, DT_INT8, DT_INT16, DT_FLOAT16, DT_INT64, DT_UINT64, DT_UINT8, DT_UINT16, DT_UINT32,
                DT_FLOAT64}))
    .REQUIRED_ATTR(group, String)
    .REQUIRED_ATTR(sr_tag, Int)
    .REQUIRED_ATTR(dest_rank, Int)
    .OP_END_FACTORY_REG(HcomSend)

    /**
     * @brief Receives the tensor from source rank.
     * @par Inputs:
     * None.
     * @par Attributes:
     * @li sr_tag: A required integer identifying the send/recv message tag. The
      message will be send by the HcomSend op with the same "sr_tag".
     * @li src_rank: A required integer identifying the source rank.
     * @li group: A required string identifying the group name of ranks
     * participating in the op.
     * @li shape: A required list identifying the shape of the tensor to be
      received.
     * @li dtype: A required integer identifying the type of the tensor to be
      received. The supported types are: int8, int16, int32, float16, float32.
     * @par Outputs:
     * y: A tensor with type identified in "dtype".
     * @attention Constraints:
      @li "group" is limited to 128 characters. Use
      "hccl_world_group" as the name of a world group.
     * @li Operators HcomSend and HcomReceive have the same "sr_tag".
     * @li "shape" should be same as the input tensor of HcomSend.
     * @li "dtype" should be same as the input tensor of HcomSend.
     * @see HcomSend
    */
    REG_OP(HcomReceive)
    .OUTPUT(
        y, TensorType(
               {DT_FLOAT, DT_INT32, DT_INT8, DT_INT16, DT_FLOAT16, DT_INT64, DT_UINT64, DT_UINT8, DT_UINT16, DT_UINT32,
                DT_FLOAT64}))
    .REQUIRED_ATTR(group, String)
    .REQUIRED_ATTR(sr_tag, Int)
    .REQUIRED_ATTR(src_rank, Int)
    .REQUIRED_ATTR(shape, ListInt)
    .REQUIRED_ATTR(dtype, Type)
    .OP_END_FACTORY_REG(HcomReceive)

    /**
     * @brief All ranks send different amount of data to, and receive different
      amount of data from, all ranks.
     * @par Inputs:
     * Five inputs, including:
     * @li send_data: A tensor. the memory to send.
     * @li send_counts: A list, where entry i specifies the number of elements in
      send_data to send to rank i.
     * @li send_displacements: A list, where entry i specifies the displacement
      (offset from sendbuf) from which to send data to rank i.
     * @li recv_counts: A list, where entry i specifies the number of
      elements to receive from rank i.
     * @li recv_displacements: A list, , where entry i specifies the displacement
      (offset from recv_data) to which data from rank i should be written.
     * @par Outputs:
     * recv_data: A Tensor  has same element type as send_data.
     * @par Attributes:
     * @li group: A string identifying the group name of ranks participating in
      the op.
    * @attention all ranks participating in the op should be full-mesh networking
      using the RDMA.
     */
    REG_OP(HcomAllToAllV)
    .INPUT(
        send_data, TensorType(
                       {DT_FLOAT, DT_INT32, DT_INT8, DT_INT16, DT_FLOAT16, DT_INT64, DT_UINT64, DT_UINT8, DT_UINT16,
                        DT_UINT32, DT_FLOAT64}))
    .INPUT(send_counts, TensorType({DT_INT64}))
    .INPUT(send_displacements, TensorType({DT_INT64}))
    .INPUT(recv_counts, TensorType({DT_INT64}))
    .INPUT(recv_displacements, TensorType({DT_INT64}))
    .OUTPUT(
        recv_data, TensorType(
                       {DT_FLOAT, DT_INT32, DT_INT8, DT_INT16, DT_FLOAT16, DT_INT64, DT_UINT64, DT_UINT8, DT_UINT16,
                        DT_UINT32, DT_FLOAT64}))
    .REQUIRED_ATTR(group, String)
    .OP_END_FACTORY_REG(HcomAllToAllV)

    /**
     * @brief All ranks send the same amount of data to each other, and receive the same amount of data from each other.
     * @par Inputs:
     * @li x: A tensor. Must be one of the following types: float32, int32, int8, int16, float16,
      int64, uint64, uint8, uint16, uint32, float64.
     * @par Outputs:
     * @li y: A Tensor. Has the same type as "x".
     * @par Attributes:
     * @li group: A string identifying the group name of ranks participating in
      the op.
     * @attention all ranks participating in the op should be full-mesh networking
      using the RDMA.
     */
    REG_OP(HcomAllToAll)
    .INPUT(
        x, TensorType(
               {DT_FLOAT, DT_INT32, DT_INT8, DT_INT16, DT_FLOAT16, DT_INT64, DT_UINT64, DT_UINT8, DT_UINT16, DT_UINT32,
                DT_FLOAT64}))
    .OUTPUT(
        y, TensorType(
               {DT_FLOAT, DT_INT32, DT_INT8, DT_INT16, DT_FLOAT16, DT_INT64, DT_UINT64, DT_UINT8, DT_UINT16, DT_UINT32,
                DT_FLOAT64}))
    .REQUIRED_ATTR(group, String)
    .OP_END_FACTORY_REG(HcomAllToAll)

    /**
     * @brief All ranks send different amount of data to, and receive different
      amount of data from, all ranks.
     * @par Inputs:
     * Two inputs, including:
     * @li send_data: A tensor. the memory to send.
     * @li send_count_matrix: A two dimensional matrix, where entry [i][j] specifies
     * the number of elements in the send_data that rank i to rank j.
     * @li fusion: An optional integer identifying the fusion flag of the op.
      0(default): no fusion; 2: fusion the ops by fusion id.
     * @li fusion_id: An optional integer identifying the fusion id of the op.
     * The HcomAllToAllVC ops with the same fusion id will be fused.
     * @par Outputs:
     * recv_data: A Tensor  has same element type as send_data.
     * @par Attributes:
     * @li rank: A required integer identifying the self rank.
     * @li group: A string identifying the group name of ranks participating in
      the op.
     */
    REG_OP(HcomAllToAllVC)
    .INPUT(
        send_data, TensorType(
                       {DT_FLOAT, DT_INT32, DT_INT8, DT_INT16, DT_FLOAT16, DT_INT64, DT_UINT64, DT_UINT8, DT_UINT16,
                        DT_UINT32, DT_FLOAT64}))
    .INPUT(send_count_matrix, TensorType({DT_INT64})) // [ranksize, ranksize]
    .OUTPUT(
        recv_data, TensorType(
                       {DT_FLOAT, DT_INT32, DT_INT8, DT_INT16, DT_FLOAT16, DT_INT64, DT_UINT64, DT_UINT8, DT_UINT16,
                        DT_UINT32, DT_FLOAT64}))
    .REQUIRED_ATTR(rank, Int)
    .REQUIRED_ATTR(group, String)
    .ATTR(fusion, Int, 0)
    .ATTR(fusion_id, Int, -1)
    .OP_END_FACTORY_REG(HcomAllToAllVC)
} // namespace ge
#endif // OPS_BUILT_IN_OP_PROTO_INC_OPS_PROTO_HCCL_H_WARN_SHOWN
#endif // OPS_BUILT_IN_OP_PROTO_INC_OPS_PROTO_HCCL_H_
