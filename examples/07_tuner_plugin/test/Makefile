# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# HCCL Tuner Plugin 单元测试 Makefile
#
# 使用方式：
#   source /usr/local/Ascend/cann/set_env.sh
#   make
#   ./test_plugin
#
# 首次构建自动触发上层 Makefile 下载 nlohmann/json。

ASCEND_HOME_PATH ?= /usr/local/Ascend/ascend-toolkit/latest
PLUGIN_INCLUDE ?= ../../../src/ops/op_common/selector/pub_inc

CXX ?= g++
CXXFLAGS := -O0 -g -Wall -Wextra -std=c++14 -D_GNU_SOURCE -DHOST_COMPILE
# 测试固定 INFO 级（诊断工具：看 compiled/rule hit 等 DFX 行）；生产 .so 默认 ERROR 级（上层 Makefile）
CXXFLAGS += -DLOG_LEVEL=LOG_LEVEL_INFO
CXXFLAGS += -I.. -I../inc -I../third_party -I$(ASCEND_HOME_PATH)/include -I$(ASCEND_HOME_PATH)/x86_64-linux/include -I$(PLUGIN_INCLUDE)
LDFLAGS := -lm -lpthread

TARGET := test_plugin
SRC := test_plugin.cpp

JSON_HEADER := ../third_party/nlohmann/json.hpp

.PHONY: all clean run fetch

all: $(TARGET)

fetch: $(JSON_HEADER)

$(JSON_HEADER):
	@$(MAKE) -C .. fetch

# ../plugin.cpp 与 ../inc/log.h 被 #include 进单测，必须显式作依赖（make 不追踪 include——
# 改插件源码不重编=旧二进制假结果，实测坑）；BUILD_FLAGS 同上层：编译选项变化触发重编
BUILD_FLAGS := .build-flags

$(TARGET): $(SRC) $(JSON_HEADER) ../plugin.cpp ../inc/log.h $(BUILD_FLAGS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

.PHONY: FORCE
FORCE:

$(BUILD_FLAGS): FORCE
	@echo '$(CXXFLAGS)' | cmp -s - $@ || echo '$(CXXFLAGS)' > $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) $(BUILD_FLAGS)
