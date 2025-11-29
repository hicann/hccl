#!/bin/sh
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

server_num=1
single_npus=8
train_file="hccl_tensorflow_allreduce_test.py"

export RANK_TABLE_FILE=ranktable.json

for i in $(seq 0 $((single_npus-1)))
do
		export DEVICE_ID=$i
		export RANK_ID=$i
		export RANK_SIZE=$[${server_num} * ${single_npus}]
		echo "------------------ train_$i.log ------------------" > train_$i.log
		python3 $train_file >> train_$i.log 2>&1 &
done

for i in $(seq 0 10)
do
		echo "wait 10s, Running......"
		sleep 10
		ps=`ps -ef | grep $train_file | grep -v grep | grep -v bash | grep -v vi`
		if [ ! "$ps" ]; then
				break
		fi
done

rm -rf tf_train.log
for i in $(seq 0 7); do cat train_$i.log >> tf_train.log; done
cat tf_train.log
rm -rf train_*.log
rm -rf *result*.json