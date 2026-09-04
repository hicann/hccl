#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tuner Plugin 最优配置生成脚本。

读取采集 CSV，按 op_type + data_type + reduce_type + ranks 4 维分组，
组内跨 engine / executor / template 按 alg_bandwidth 选优，生成 Tuner Plugin JSON。

用法：
    python3 optimize_config.py --input hccl_prof.csv --output hccl_tuner_config.json
"""

import argparse
import csv
import json
import logging
import sys
from collections import OrderedDict

from tuner_common import (
    CSV_FIELDS, DEVICE_ENGINES, PLUGIN_ENGINES, PLUGIN_EXECUTORS, PLUGIN_OP_TYPES, PLUGIN_TEMPLATES,
    templates_to_plugin_name,
)

# ===== 本脚本特有常量 =====

DPU_ENGINE = "dpu"

# plugin 支持的 dtype（含 float16/float32 等别名）
PLUGIN_DATA_TYPES = frozenset((
    "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    "fp16", "float16", "fp32", "float32", "fp64", "float64", "bfp16", "bfloat16",
))

# CSV ranks.* → conf match min/max_*
TOPO_FIELDS = (
    ("ranks.ranks", "ranks"),
    ("ranks.npus_per_server", "npus_per_server"),
    ("ranks.servers", "servers"),
    ("ranks.pods", "pods"),
    ("ranks.super_pods", "super_pods"),
)

# 白名单校验失败类别
INVALID_REASONS = ("op_type", "data_type", "engine", "executor", "template")

# 采集口径与插件 nBytes（运行时 param.inputSize）不一致的算子及换算方向：
# - allgather 系：-b/-e 是每 rank 收到的完整数据量，插件看到的是每 rank 发出的
#   量，写规则前需 ÷ ranks（向上取整）
# - scatter：-b/-e 是每 rank 收到的量，插件看到的是 root 总发出量，写规则前需
#   × ranks（前提：hccl_test scatter_test 的 -b 定义为每 rank 收到量，与
#   allgather 系同口径；若上机验证为 root 总量口径，删除本条即可）
NBYTES_NORMALIZE_OPS = {
    "allgather": "div",
    "allgatherv": "div",
    "scatter": "mul",
}


class OptimizeConfigError(Exception):
    """优化脚本输入 / 处理错误。"""


logger = logging.getLogger("optimize_config")


# ===== 读 CSV 与行清洗 =====

def read_rows(path):
    """读采集 CSV 为行字典列表（缺列按空处理）。"""
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row = {field: (raw.get(field) or "").strip() for field in CSV_FIELDS}
            rows.append(row)
    return rows


def _to_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def dedup_latest(rows):
    """同 key 行取 timestamp 最大者，其余丢弃。"""
    best = OrderedDict()
    dropped = 0
    for row in rows:
        key = dedup_key(row)
        if key not in best:
            best[key] = row
            continue
        if _to_int(row.get("timestamp(ms)")) > _to_int(best[key].get("timestamp(ms)")):
            best[key] = row
        dropped += 1
    if dropped:
        logger.info("[optimize_config] dedup: dropped {} stale row(s) (keep latest by timestamp)".format(dropped))
    return list(best.values()), dropped


def dedup_key(row):
    """去重 key：算法 + 拓扑 + 采集维度全字段。"""
    return (
        row.get("op_type"), row.get("size_bytes"), row.get("data_type"), row.get("reduce_type"),
        row.get("engine"), row.get("algorithm.executor_type"), row.get("algorithm.template_type"),
    ) + tuple(row.get(csv_field) for csv_field, _ in TOPO_FIELDS)


def filter_failed(rows):
    """过滤 check_result 为 failed/noresult 的不可信行。"""
    kept, dropped = [], 0
    for row in rows:
        if row.get("check_result", "").lower() in ("failed", "noresult"):
            dropped += 1
            continue
        kept.append(row)
    if dropped:
        logger.info("[optimize_config] filter: dropped {} failed/noresult row(s)".format(dropped))
    return kept, dropped


def normalize_bytes(rows):
    """采集口径换算成插件 nBytes 口径（NBYTES_NORMALIZE_OPS），其余算子透传。

    必须放在去重之前：否则同一点的新旧行因口径不同对不上去重 key。
    """
    normalized = []
    adjusted = 0
    for row in rows:
        direction = NBYTES_NORMALIZE_OPS.get(row.get("op_type"))
        if direction is not None:
            size = _to_int(row.get("size_bytes"))
            ranks = _to_int(row.get("ranks.ranks"))
            if size > 0 and ranks > 1:
                if direction == "div":
                    per_rank = -(-size // ranks)  # 向上取整
                else:
                    per_rank = size * ranks
                if per_rank != size:
                    adjusted += 1
                row = dict(row, size_bytes=str(per_rank))
                logger.info("[optimize_config] normalize: op={} size={}B {} ranks={} -> {}B (plugin nBytes)".format(
                    row.get("op_type"), size, direction, ranks, per_rank))
        normalized.append(row)
    if adjusted:
        logger.info("[optimize_config] normalize: adjusted {} row(s)".format(adjusted))
    return normalized


# ===== 分组与选优 =====

def group_key(row):
    """分组键：op_type + data_type + reduce_type + ranks 5 字段。"""
    return (
        row.get("op_type"), row.get("data_type"), row.get("reduce_type"),
    ) + tuple(row.get(csv_field) for csv_field, _ in TOPO_FIELDS)


def group_rows(rows):
    """按 4 维分组（engine 是组内比较维度，不参与分组）。"""
    groups = OrderedDict()
    for row in rows:
        groups.setdefault(group_key(row), []).append(row)
    return groups


def select_best_per_size(rows):
    """同一 size 点跨 engine/executor/template 选优，返回 {size: 胜者行}。

    带宽最大胜；相同比 latency 小者胜；再同保持序稳定。
    """
    ordered = sorted(rows, key=lambda r: (
        _to_int(r.get("size_bytes")), r.get("engine", ""),
        r.get("algorithm.executor_type", ""), r.get("algorithm.template_type", "")))
    best = {}
    for row in ordered:
        size = _to_int(row.get("size_bytes"))
        cur = best.get(size)
        if cur is None or _better(row, cur):
            best[size] = row
    return best


def _better(cand, cur):
    """cand 是否优于 cur。"""
    bw_c, bw_u = _to_float(cand.get("alg_bandwidth(GB/s)")), _to_float(cur.get("alg_bandwidth(GB/s)"))
    if bw_c != bw_u:
        return bw_c > bw_u
    return _to_float(cand.get("alg_latency(us)")) < _to_float(cur.get("alg_latency(us)"))


# ===== 白名单硬校验（产 rule 前） =====

def validate_winner(row):
    """校验胜者算法可写入 conf（plugin 白名单）；合法返回 None，否则返回失败类别。

    白名单外的算法不产出 rule——plugin 侧一条非法 rule 会导致整份 conf 失效。
    """
    if row.get("op_type") not in PLUGIN_OP_TYPES:
        return "op_type"
    if row.get("data_type") not in PLUGIN_DATA_TYPES:
        return "data_type"
    if row.get("engine") not in PLUGIN_ENGINES:
        return "engine"
    if row.get("algorithm.executor_type") not in PLUGIN_EXECUTORS:
        return "executor"
    for token in row.get("algorithm.template_type", "").split(","):
        if token not in PLUGIN_TEMPLATES:
            return "template"
    return None


# ===== 区间合并与补缝 =====

def _algo_id(row):
    """算法标识（engine + executor + template），用于相邻点同算法判断。"""
    return (row.get("engine"), row.get("algorithm.executor_type"), row.get("algorithm.template_type"))


def merge_intervals(points):
    """相邻数据点同算法则合并为 [lo, hi] 区间，否则各为单点区间。

    points：{size: 胜者行}；返回 [(lo, hi, row), ...]（按 lo 升序）。
    """
    intervals = []
    for size in sorted(points):
        row = points[size]
        if intervals and _algo_id(intervals[-1][2]) == _algo_id(row):
            intervals[-1] = (intervals[-1][0], size, row)
        else:
            intervals.append((size, size, row))
    return intervals


def fill_gaps(intervals):
    """补缝（归右）：区间间空隙归属右端区间，右侧向左扩展。"""
    filled = [list(item) for item in intervals]
    for i in range(1, len(filled)):
        prev_hi = filled[i - 1][1]
        if filled[i][0] > prev_hi + 1:
            filled[i][0] = prev_hi + 1
    return [tuple(item) for item in filled]


# ===== rule / conf 生成 =====

def build_rule(lo, hi, row):
    """单区间 → plugin rule：match 写拓扑字段（min=max=采集值）+ bytes + data_type。

    拓扑字段值为 0（未指定）时跳过写入——match 缺失即不约束该维度，规避
    单机采集值 1 与插件运行时 nPods=0 的错位。
    dpu 强制 min_servers>=2：dpu 算法注册 isHostDpuOnly=true，与 device 算法
    运行时拓扑互斥，单机环境 dpu rule 即使命中其目标算法也必被 topo 过滤，
    不加约束就是死配置（first-match-wins 下恒先命中）。
    """
    match = {}
    for csv_field, conf_name in TOPO_FIELDS:
        value = _to_int(row.get(csv_field), 0)
        if value <= 0:
            continue
        match["min_" + conf_name] = value
        match["max_" + conf_name] = value
    if row.get("engine") == DPU_ENGINE:
        lo_servers = _to_int(match.get("min_servers"), 0)
        if lo_servers < 2:
            match.pop("min_servers", None)
            match.pop("max_servers", None)
            match["min_servers"] = 2
    match["min_bytes"] = lo
    match["max_bytes"] = hi
    match["data_type"] = row.get("data_type")
    return {
        "match": match,
        "engine": row.get("engine"),
        "executor": row.get("algorithm.executor_type"),
        "template": templates_to_plugin_name(row.get("algorithm.template_type", "")),
        "cost": 0.0,
    }


def process_group(rows, counters):
    """处理单个分组：device 引擎与 dpu 各自独立选优、独立产 rule。

    两组运行时拓扑互斥（hostdpu 环境只剩 dpu 可选，普通环境 dpu 全被过滤），
    不做同台比较；排序 device 在前，保证普通环境优先命中 device 最优。
    """
    partitions = (
        [r for r in rows if r.get("engine") in DEVICE_ENGINES],
        [r for r in rows if r.get("engine") == DPU_ENGINE],
    )
    rules = []
    for partition in partitions:
        if not partition:
            continue
        points = select_best_per_size(partition)
        valid_points = {}
        for size, row in points.items():
            reason = validate_winner(row)
            if reason is not None:
                counters[reason] += 1
                logger.info("[optimize_config] skip: op={} size={} algo=({},{},{}) dtype={} reason={}".format(
                    row.get("op_type"), size, row.get("engine"), row.get("algorithm.executor_type"),
                    row.get("algorithm.template_type"), row.get("data_type"), reason))
                continue
            valid_points[size] = row
        if not valid_points:
            continue
        intervals = fill_gaps(merge_intervals(valid_points))
        for lo, hi, row in intervals:
            rules.append(build_rule(lo, hi, row))
    return rules


def build_conf(rows):
    """主处理链：清洗 → 分组 → 选优 → 校验 → 合并/补缝 → conf dict。"""
    counters = {reason: 0 for reason in INVALID_REASONS}
    rows = normalize_bytes(rows)
    rows, _ = dedup_latest(rows)
    rows, _ = filter_failed(rows)
    groups = group_rows(rows)
    op_rules = {}
    total_rules = 0
    for key, group in groups.items():
        op_type = key[0]
        rules = process_group(group, counters)
        if not rules:
            continue
        rules.sort(key=lambda r: (r["match"]["min_bytes"],
                                  0 if r["engine"] in DEVICE_ENGINES else 1))
        op_rules.setdefault(op_type, []).extend(rules)
        total_rules += len(rules)
    for reason, count in counters.items():
        if count:
            logger.info("[optimize_config] whitelist: dropped {} winner(s) by reason '{}'".format(
                count, reason))
    conf = {"version": 1, "op_types": OrderedDict()}
    for op_type in sorted(op_rules):
        conf["op_types"][op_type] = {"rules": op_rules[op_type]}
    return conf, total_rules


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    parser = argparse.ArgumentParser(
        description="处理采集 CSV，生成 Tuner Plugin 最优 JSON 配置（cost 表改写规则）")
    parser.add_argument("--input", required=True, help="采集 CSV 路径（prof_test.py 产物）")
    parser.add_argument("--output", required=True, help="输出 conf JSON 路径")
    args = parser.parse_args(argv)

    try:
        rows = read_rows(args.input)
    except OSError as err:
        logger.error("[optimize_config] read input failed: {}".format(err))
        return 1
    if not rows:
        logger.error("[optimize_config] input CSV is empty: {}".format(args.input))
        return 1

    conf, total_rules = build_conf(rows)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(conf, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    logger.info("[optimize_config] done: rows={} rules={} ops={} output={}".format(
        len(rows), total_rules, len(conf["op_types"]), args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
