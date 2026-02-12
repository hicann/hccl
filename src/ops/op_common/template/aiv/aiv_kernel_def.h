/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_AIV_KERNEL_DEF_H
#define HCCL_AIV_KERNEL_DEF_H

#include "hccl_aiv_utils.h"

namespace ops_hccl {
static std::vector<AivKernelInfo> g_allreduceAivKernelInfoList = {
    {"aiv_allreduce_half", HcclDataType::HCCL_DATA_TYPE_FP16},
    {"aiv_allreduce_int16_t", HcclDataType::HCCL_DATA_TYPE_INT16},
    {"aiv_allreduce_float", HcclDataType::HCCL_DATA_TYPE_FP32},
    {"aiv_allreduce_int32_t", HcclDataType::HCCL_DATA_TYPE_INT32},
    {"aiv_allreduce_int8_t", HcclDataType::HCCL_DATA_TYPE_INT8},
    {"aiv_allreduce_bfloat16_t", HcclDataType::HCCL_DATA_TYPE_BFP16},
    {"aiv_allreduce_mesh1d_twoshot_half", HcclDataType::HCCL_DATA_TYPE_FP16, KernelArgsType::ARGS_TYPE_TWO_SHOT},
    {"aiv_allreduce_mesh1d_twoshot_int16_t", HcclDataType::HCCL_DATA_TYPE_INT16, KernelArgsType::ARGS_TYPE_TWO_SHOT},
    {"aiv_allreduce_mesh1d_twoshot_float", HcclDataType::HCCL_DATA_TYPE_FP32, KernelArgsType::ARGS_TYPE_TWO_SHOT},
    {"aiv_allreduce_mesh1d_twoshot_int32_t", HcclDataType::HCCL_DATA_TYPE_INT32, KernelArgsType::ARGS_TYPE_TWO_SHOT},
    {"aiv_allreduce_mesh1d_twoshot_int8_t", HcclDataType::HCCL_DATA_TYPE_INT8, KernelArgsType::ARGS_TYPE_TWO_SHOT},
    {"aiv_allreduce_mesh1d_twoshot_bfloat16_t", HcclDataType::HCCL_DATA_TYPE_BFP16, KernelArgsType::ARGS_TYPE_TWO_SHOT},
};
static std::string g_allreduceAivBinaryName = "hccl_aiv_all_reduce_op_910_95.o";

static std::vector<AivKernelInfo> g_reduceScatterAivKernelInfoList = {
    {"aiv_reduce_scatter_half", HcclDataType::HCCL_DATA_TYPE_FP16},
    {"aiv_reduce_scatter_int16_t", HcclDataType::HCCL_DATA_TYPE_INT16},
    {"aiv_reduce_scatter_float", HcclDataType::HCCL_DATA_TYPE_FP32},
    {"aiv_reduce_scatter_int32_t", HcclDataType::HCCL_DATA_TYPE_INT32},
    {"aiv_reduce_scatter_int8_t", HcclDataType::HCCL_DATA_TYPE_INT8},
    {"aiv_reduce_scatter_bfloat16_t", HcclDataType::HCCL_DATA_TYPE_BFP16},
};
static std::string g_reduceScatterAivBinaryName = "hccl_aiv_reduce_scatter_op_910_95.o";

static std::vector<AivKernelInfo> g_allgatherAivKernelInfoList = {
    {"aiv_all_gather_half", HcclDataType::HCCL_DATA_TYPE_FP16},
    {"aiv_all_gather_int16_t", HcclDataType::HCCL_DATA_TYPE_INT16},
    {"aiv_all_gather_uint16_t", HcclDataType::HCCL_DATA_TYPE_UINT16},
    {"aiv_all_gather_float", HcclDataType::HCCL_DATA_TYPE_FP32},
    {"aiv_all_gather_int32_t", HcclDataType::HCCL_DATA_TYPE_INT32},
    {"aiv_all_gather_uint32_t", HcclDataType::HCCL_DATA_TYPE_UINT32},
    {"aiv_all_gather_int8_t", HcclDataType::HCCL_DATA_TYPE_INT8},
    {"aiv_all_gather_uint8_t", HcclDataType::HCCL_DATA_TYPE_UINT8},
    {"aiv_all_gather_bfloat16_t", HcclDataType::HCCL_DATA_TYPE_BFP16},
    {"aiv_all_gather_uint64_t", HcclDataType::HCCL_DATA_TYPE_INT64},
    {"aiv_all_gather_int64_t", HcclDataType::HCCL_DATA_TYPE_UINT64},
};
static std::string g_allgatherAivBinaryName = "hccl_aiv_all_gather_op_910_95.o";

static std::vector<AivKernelInfo> g_broadcastAivKernelInfoList = {
    {"aiv_broadcast_half", HcclDataType::HCCL_DATA_TYPE_FP16},
    {"aiv_broadcast_int16_t", HcclDataType::HCCL_DATA_TYPE_INT16},
    {"aiv_broadcast_uint16_t", HcclDataType::HCCL_DATA_TYPE_UINT16},
    {"aiv_broadcast_float", HcclDataType::HCCL_DATA_TYPE_FP32},
    {"aiv_broadcast_int32_t", HcclDataType::HCCL_DATA_TYPE_INT32},
    {"aiv_broadcast_uint32_t", HcclDataType::HCCL_DATA_TYPE_UINT32},
    {"aiv_broadcast_int8_t", HcclDataType::HCCL_DATA_TYPE_INT8},
    {"aiv_broadcast_uint8_t", HcclDataType::HCCL_DATA_TYPE_UINT8},
    {"aiv_broadcast_bfloat16_t", HcclDataType::HCCL_DATA_TYPE_BFP16},
    {"aiv_broadcast_uint64_t", HcclDataType::HCCL_DATA_TYPE_INT64},
    {"aiv_broadcast_int64_t", HcclDataType::HCCL_DATA_TYPE_UINT64},
};
static std::string g_broadcastAivBinaryName = "hccl_aiv_broadcast_op_910_95.o";

static std::vector<AivKernelInfo> g_alltoallAivKernelInfoList = {
    {"aiv_alltoall_half", HcclDataType::HCCL_DATA_TYPE_FP16},
    {"aiv_alltoall_int16_t", HcclDataType::HCCL_DATA_TYPE_INT16},
    {"aiv_alltoall_uint16_t", HcclDataType::HCCL_DATA_TYPE_UINT16},
    {"aiv_alltoall_float", HcclDataType::HCCL_DATA_TYPE_FP32},
    {"aiv_alltoall_int32_t", HcclDataType::HCCL_DATA_TYPE_INT32},
    {"aiv_alltoall_uint32_t", HcclDataType::HCCL_DATA_TYPE_UINT32},
    {"aiv_alltoall_int8_t", HcclDataType::HCCL_DATA_TYPE_INT8},
    {"aiv_alltoall_uint8_t", HcclDataType::HCCL_DATA_TYPE_UINT8},
    {"aiv_alltoall_bfloat16_t", HcclDataType::HCCL_DATA_TYPE_BFP16},
    {"aiv_alltoall_uint64_t", HcclDataType::HCCL_DATA_TYPE_INT64},
    {"aiv_alltoall_int64_t", HcclDataType::HCCL_DATA_TYPE_UINT64},
};
static std::string g_alltoallAivBinaryName = "hccl_aiv_all_to_all_op_910_95.o";

static std::vector<AivKernelInfo> g_alltoallvAivKernelInfoList = {
    {"aiv_alltoallv_half", HcclDataType::HCCL_DATA_TYPE_FP16},
    {"aiv_alltoallv_int16_t", HcclDataType::HCCL_DATA_TYPE_INT16},
    {"aiv_alltoallv_uint16_t", HcclDataType::HCCL_DATA_TYPE_UINT16},
    {"aiv_alltoallv_float", HcclDataType::HCCL_DATA_TYPE_FP32},
    {"aiv_alltoallv_int32_t", HcclDataType::HCCL_DATA_TYPE_INT32},
    {"aiv_alltoallv_uint32_t", HcclDataType::HCCL_DATA_TYPE_UINT32},
    {"aiv_alltoallv_int8_t", HcclDataType::HCCL_DATA_TYPE_INT8},
    {"aiv_alltoallv_uint8_t", HcclDataType::HCCL_DATA_TYPE_UINT8},
    {"aiv_alltoallv_bfloat16_t", HcclDataType::HCCL_DATA_TYPE_BFP16},
    {"aiv_alltoallv_uint64_t", HcclDataType::HCCL_DATA_TYPE_INT64},
    {"aiv_alltoallv_int64_t", HcclDataType::HCCL_DATA_TYPE_UINT64},
};
static std::string g_alltoallvAivBinaryName = "hccl_aiv_all_to_all_v_op_910_95.o";

static std::vector<AivKernelInfo> g_scatterAivKernelInfoList = {
    {"aiv_scatter_half", HcclDataType::HCCL_DATA_TYPE_FP16},
    {"aiv_scatter_int16_t", HcclDataType::HCCL_DATA_TYPE_INT16},
    {"aiv_scatter_uint16_t", HcclDataType::HCCL_DATA_TYPE_UINT16},
    {"aiv_scatter_float", HcclDataType::HCCL_DATA_TYPE_FP32},
    {"aiv_scatter_int32_t", HcclDataType::HCCL_DATA_TYPE_INT32},
    {"aiv_scatter_uint32_t", HcclDataType::HCCL_DATA_TYPE_UINT32},
    {"aiv_scatter_int8_t", HcclDataType::HCCL_DATA_TYPE_INT8},
    {"aiv_scatter_uint8_t", HcclDataType::HCCL_DATA_TYPE_UINT8},
    {"aiv_scatter_bfloat16_t", HcclDataType::HCCL_DATA_TYPE_BFP16},
    {"aiv_scatter_uint64_t", HcclDataType::HCCL_DATA_TYPE_INT64},
    {"aiv_scatter_int64_t", HcclDataType::HCCL_DATA_TYPE_UINT64},
};
static std::string g_scatterAivBinaryName = "hccl_aiv_scatter_op_910_95.o";

static std::vector<AivKernelInfo> g_reduceAivKernelInfoList = {
    {"aiv_reduce_half", HcclDataType::HCCL_DATA_TYPE_FP16},
    {"aiv_reduce_int16_t", HcclDataType::HCCL_DATA_TYPE_INT16},
    {"aiv_reduce_float", HcclDataType::HCCL_DATA_TYPE_FP32},
    {"aiv_reduce_int32_t", HcclDataType::HCCL_DATA_TYPE_INT32},
    {"aiv_reduce_int8_t", HcclDataType::HCCL_DATA_TYPE_INT8},
    {"aiv_reduce_bfloat16_t", HcclDataType::HCCL_DATA_TYPE_BFP16},
};
static std::string g_reduceAivBinaryName = "hccl_aiv_reduce_op_910_95.o";

static std::map<HcclCMDType, std::pair<std::string, std::vector<AivKernelInfo>>> g_aivKernelInfoMap = {
    {HcclCMDType::HCCL_CMD_ALLREDUCE, {g_allreduceAivBinaryName, g_allreduceAivKernelInfoList}},
    {HcclCMDType::HCCL_CMD_REDUCE_SCATTER, {g_reduceScatterAivBinaryName, g_reduceScatterAivKernelInfoList}},
    {HcclCMDType::HCCL_CMD_ALLGATHER, {g_allgatherAivBinaryName, g_allgatherAivKernelInfoList}},
    {HcclCMDType::HCCL_CMD_BROADCAST, {g_broadcastAivBinaryName, g_broadcastAivKernelInfoList}},
    {HcclCMDType::HCCL_CMD_ALLTOALL, {g_alltoallAivBinaryName, g_alltoallAivKernelInfoList}},
    {HcclCMDType::HCCL_CMD_ALLTOALLV, {g_alltoallvAivBinaryName, g_alltoallvAivKernelInfoList}},
    {HcclCMDType::HCCL_CMD_SCATTER, {g_scatterAivBinaryName, g_scatterAivKernelInfoList}},
    {HcclCMDType::HCCL_CMD_REDUCE, {g_reduceAivBinaryName, g_reduceAivKernelInfoList}},
};
}

#endif // HCCL_AIV_UTILS_H