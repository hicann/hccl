# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# Makefile for testcase

CXX := mpic++
CXXFLAGS := -std=c++17 -O2 -g -Wall
ASCEND_HOME ?= /usr/local/Ascend/ascend-toolkit/latest
INCLUDES := -I../inc -I$(ASCEND_HOME)/include -I$(ASCEND_HOME)/include/hccl
LIB_DIR := ../build/op_host
LIBS := -L$(ASCEND_HOME)/lib64 -lhccl -lacl_rt -L$(LIB_DIR) -lhccl_custom_allgather -pthread

TARGET := main

all: $(TARGET)

$(TARGET): main.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LIBS)

clean:
	rm -f $(TARGET)

# 通过 make run NP=<卡数> 控制进程数，默认 2
NP ?= 2

run: $(TARGET)
	export LD_LIBRARY_PATH=$(LIB_DIR):$(ASCEND_HOME)/lib64:$$LD_LIBRARY_PATH && mpirun -n $(NP) ./$(TARGET)
