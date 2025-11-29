#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp


def run_hccl(rank: int, world_size: int, master_ip: str, master_port: int):
    # 指定当前进程使用的 NPU 设备
    torch_npu.npu.set_device(rank)

    # 初始化进程组，后端使用 HCCL
    init_method = f"tcp://{master_ip}:{master_port}"
    dist.init_process_group(
        backend="hccl", rank=rank, world_size=world_size, init_method=init_method
    )

    # 构造输入数据，1行8列，值为0~7
    torch_tensor = torch.arange(world_size, dtype=torch.float32, device="npu")
    print("[Rank %d] Input: %s" % (rank, torch_tensor))

    try:
        # 调用 HCCL 接口，下发 AllReduce 集合通信算子
        dist.all_reduce(torch_tensor, op=dist.ReduceOp.SUM)
    except Exception as e:
        print("[Rank %d] Error occurred: %s" % (rank, e))
    else:
        print("[Rank %d] Output: %s" % (rank, torch_tensor))


def main():
    print("Executing AllReduce collective operation via HCCL backend")
    ip = "127.0.0.1"
    port = 50001
    print("Listening on %s:%d" % (ip, port))

    rank_size = torch_npu.npu.device_count()
    print("Available NPU count: %d" % rank_size)

    # 启动多进程
    mp.spawn(run_hccl, args=(rank_size, ip, port), nprocs=rank_size, join=True)


if __name__ == "__main__":
    main()
