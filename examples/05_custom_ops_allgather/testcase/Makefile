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
