# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# HCCL Tuner Plugin 示例 Makefile
#
# 使用方式：
#   source /usr/local/Ascend/cann/set_env.sh
#   make
#
# 产物：hccl_tuner_example.so
#
# 首次构建会自动下载 nlohmann/json（header-only）到 third_party/。
# 离线场景可手动预放 third_party/nlohmann/json.hpp，make 检测到即跳过下载。

ASCEND_HOME_PATH ?= /usr/local/Ascend/ascend-toolkit/latest
PLUGIN_INCLUDE ?= ../../src/ops/op_common/selector/inc

CXX ?= g++
CXXFLAGS := -fPIC -shared -O2 -Wall -Wextra -std=c++14
CXXFLAGS += -I$(ASCEND_HOME_PATH)/include -I$(ASCEND_HOME_PATH)/x86_64-linux/include -I$(PLUGIN_INCLUDE)
CXXFLAGS += -I$(ASCEND_HOME_PATH)/aarch64-linux/include
CXXFLAGS += -I./third_party

TARGET := hccl_tuner_example.so
SRC := plugin.cpp

JSON_VERSION := 3.11.3
JSON_URL     := https://gitcode.com/cann-src-third-party/json/releases/download/v$(JSON_VERSION)/include.zip
JSON_HEADER  := third_party/nlohmann/json.hpp

.PHONY: all clean test fetch

all: $(TARGET)

fetch: $(JSON_HEADER)

$(JSON_HEADER):
	@echo "[fetch] nlohmann/json v$(JSON_VERSION) -> $@"
	@mkdir -p $(dir $@)
	@curl -sL "$(JSON_URL)" -o /tmp/hccl_json_$$.zip
	@unzip -p /tmp/hccl_json_$$.zip single_include/nlohmann/json.hpp > $@
	@rm -f /tmp/hccl_json_$$.zip
	@test -s $@ || { echo "Error: fetch failed, $@ is empty"; rm -f $@; exit 1; }

$(TARGET): $(SRC) $(JSON_HEADER)
	$(CXX) $(CXXFLAGS) -o $@ $<

test:
	$(MAKE) -C test

clean:
	rm -f $(TARGET)
	$(MAKE) -C test clean
