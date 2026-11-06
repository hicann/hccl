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

"""采集与配置生成公共常量表。

perf_test.py（采集）与 optimize_config.py（生成配置）共用的枚举词表、算法名串语法、
注册名反分解与 probe 名单 JSON 数据模型，只在此处维护。算法名单的真相源是 probe
阶段运行环境的 costTable 日志（见 perf_test.py），本文件不维护静态注册白名单。
"""

import json
import re
import time

# ===== 算法名全量词表（真相源：alg_parse.cc；全量表管"解析"，采集白名单管"采集"）=====

# opType Pascal 中缀剥离表（长词优先，与 C++ GetOpTypePatternEntries 同序）；
# barrier/send 等非采集算子在 _PASCAL_TO_OP 无映射，由上层过滤
_OP_TYPE_PASCALS = (
    "ReduceScatterV", "ReduceScatter", "AllGatherV", "AllGather", "AllReduce",
    "AllToAllVC", "AllToAllV", "AllToAll", "BatchSendRecv", "Broadcast",
    "Barrier", "Reduce", "Scatter", "Send", "Recv",
)
_PASCAL_TO_OP = {
    "AllReduce": "allreduce", "AllGather": "allgather", "Broadcast": "broadcast", "Reduce": "reduce",
    "ReduceScatter": "reduce_scatter", "Scatter": "scatter", "AllToAll": "alltoall", "AllToAllV": "alltoallv",
}

# executor Pascal 词表（alg_parse.cc EXECUTOR_TYPES 全量 7 个）
_EXECUTOR_PASCALS = ("Sole", "Sequence", "Parallel", "PipeLine", "Concur", "StrictOrdered", "OmniPipe")

# 模板 token 全量 17 个（alg_parse.cc ALGO_TYPES）
_TEMPLATE_TOKENS = (
    ("Mesh", "mesh"), ("Mesh2Die", "mesh2die"), ("MeshOneShot", "meshoneshot"),
    ("MeshTwoShot", "meshtwoshot"), ("MeshConcur", "meshconcur"), ("MeshMultiLink", "meshmultilink"),
    ("MeshChunk", "meshchunk"), ("MeshChunkTwoShot", "meshchunktwoshot"), ("NHR", "nhr"),
    ("NHRMultiLink", "nhrmultilink"), ("NHRAicpuReduce", "nhraicpureduce"),
    ("MeshSingleChannel", "meshsinglechannel"), ("NHRMultiJetty", "nhrmultijetty"),
    ("MeshMultiJetty", "meshmultijetty"), ("MeshConcurrent", "meshconcurrent"),
    ("GroupMesh", "groupmesh"), ("Pairwise", "pairwise"),
)
# 贪婪匹配按 Pascal 词长降序（同 C++ std::sort size 降序）
_TEMPLATE_TOKENS_SORTED = tuple(sorted(_TEMPLATE_TOKENS, key=lambda item: -len(item[0])))

# op_type 采集白名单（对齐 plugin.cpp op_type_from_name；v 系列不采不产）——由映射表派生
OP_TYPES = tuple(_PASCAL_TO_OP.values())

# CSV 17 字段；性能数值字段名内嵌单位（latency=us、bandwidth=GB/s），timestamp 为 17 位紧凑时间戳。
# algorithm.template_type 落盘为带逗号语义串（"meshconcur,nhr"），写 conf.json 时经 templates_to_plugin_name 转拼接串。
CSV_FIELDS = (
    "op_type", "size_bytes", "data_type", "reduce_type", "engine",
    "algorithm.executor_type", "algorithm.template_type", "HCCL_BUFFSIZE",
    "ranks.ranks", "ranks.npus_per_server", "ranks.servers", "ranks.pods", "ranks.super_pods",
    "check_result", "alg_bandwidth(GB/s)", "alg_latency(us)", "timestamp",
)

# timestamp 列：17 位定长（年月日时分秒+毫秒），int 精确比较（超出 float64 精度，禁走 float）
TIMESTAMP_RE = re.compile(r"20\d{15}")


def now_timestamp():
    """当前时刻的紧凑时间戳（17 位：YYYYMMDDHHMMSSmmm）。"""
    now = time.time()
    return time.strftime("%Y%m%d%H%M%S", time.localtime(now)) + "{:03d}".format(int(now * 1000) % 1000)


def parse_timestamp(value):
    """timestamp 列转 int；非法（缺列/旧格式/被改）返回 -1，去重时视作最旧。"""
    return int(value) if isinstance(value, str) and TIMESTAMP_RE.fullmatch(value) else -1


# 拓扑 5 字段（CSV ranks.* 列名）
RANKS_FIELDS = ("ranks.ranks", "ranks.npus_per_server", "ranks.servers", "ranks.pods", "ranks.super_pods")

# engine 规范名：device 引擎 4 种 + dpu
DEVICE_ENGINES = ("aicpu", "aiv", "ccums", "ccusched")
ENGINE_ALL = DEVICE_ENGINES + ("dpu",)

# executor 采集白名单（全量词表减 omnipipe；新词元默认不采）
EXECUTOR_TYPES = ("sole", "sequence", "parallel", "pipeline", "concur", "strictordered")
# import 期一致性校验：白名单名字必须都在全量词表内（拼写漂移此处立即报错）
_UNKNOWN = [n for n in EXECUTOR_TYPES if n not in {p.lower() for p in _EXECUTOR_PASCALS}]
if _UNKNOWN:
    raise ValueError("EXECUTOR_TYPES 存在全量词表外的名字: {}".format(_UNKNOWN))

# template 采集白名单（multijetty/groupmesh/pairwise 解析得动但不采，记 unresolved；新词元默认不采）
TEMPLATE_TYPES = (
    "mesh", "mesh2die", "meshoneshot", "meshtwoshot", "meshconcur", "meshmultilink",
    "meshchunk", "meshchunktwoshot", "nhr", "nhrmultilink", "nhraicpureduce",
    "meshsinglechannel", "meshconcurrent",
)
_UNKNOWN = [n for n in TEMPLATE_TYPES if n not in {lower for _, lower in _TEMPLATE_TOKENS}]
if _UNKNOWN:
    raise ValueError("TEMPLATE_TYPES 存在全量词表外的名字: {}".format(_UNKNOWN))

# template 两种口径：
# - CSV 语义串（带逗号，"meshconcur,nhr,nhr"）——与 HCCL_ALGO 语法一致
# - plugin 拼接串（不带逗号，"meshconcurnhrnhr"）——plugin 只认这一种


def templates_to_plugin_name(template_str):
    """语义串转拼接串（去逗号）；单级原样透传。"""
    return (template_str or "").replace(",", "")


# ===== plugin 白名单 =====

PLUGIN_OP_TYPES = frozenset(OP_TYPES)
PLUGIN_ENGINES = frozenset(ENGINE_ALL)

# plugin.cpp 已不对 executor 做枚举校验（合法集合随版本演进）；仍保守排除 strictordered（不在常规采集集合）
PLUGIN_EXECUTORS = frozenset(EXECUTOR_TYPES) - frozenset(("strictordered",))

PLUGIN_TEMPLATES = frozenset(TEMPLATE_TYPES)


# ===== 注册名反分解（规则对齐 alg_attrs_registry.cc ParseAlgName） =====
#
# costTable 日志打印的是注册算法名（如 AicpuAllReduceSoleMeshOneShot），采集计划需要的
# (engine, executor, templates) 三元组口径由此反分解得到。

# engine Pascal 前缀 → 规范名（顺序与 C++ GetEnginePrefixEntries 一致）
_ENGINE_PREFIX_TO_NAME = (
    ("CcuSched", "ccusched"), ("CcuMS", "ccums"), ("Aicpu", "aicpu"), ("Aiv", "aiv"), ("Dpu", "dpu"),
)

# 采集域合法集（加速查表）
_TEMPLATE_VALID = frozenset(TEMPLATE_TYPES)
_EXECUTOR_VALID = frozenset(EXECUTOR_TYPES)


def split_algo_name(reg_name):
    """注册算法名反分解为 (op, engine, executor, templates)；解析失败返回 None。

    四步与 C++ ParseAlgName 一致：
    1. 剥 engine 前缀；
    2. 剥 opType 中缀（长词优先）→ 采集域 op 或 None；
    3. 剥 executor 词（≤2 个，首个记 executor）；
    4. 剩余串按模板 token 贪婪匹配。
    任一环节无法消化（UBX/Pcie 等非标后缀）或词不在采集域 → None。
    """
    remaining = reg_name

    engine = None
    for prefix, name in _ENGINE_PREFIX_TO_NAME:
        if remaining.startswith(prefix) and len(remaining) > len(prefix):
            engine = name
            remaining = remaining[len(prefix):]
            break
    if engine is None:
        return None

    op = None
    for pascal in _OP_TYPE_PASCALS:
        if remaining.startswith(pascal):
            remaining = remaining[len(pascal):]
            op = _PASCAL_TO_OP.get(pascal)
            break
    else:
        return None
    if op is None:
        return None  # barrier/send 等非采集算子

    executor = None
    for _ in range(2):  # C++ 双 pass 剥离
        for pascal in _EXECUTOR_PASCALS:
            if remaining.startswith(pascal):
                if executor is None:
                    executor = pascal.lower()
                remaining = remaining[len(pascal):]
                break
        else:
            break
    if executor is None or executor not in _EXECUTOR_VALID:
        return None

    templates = []
    pos = 0
    while pos < len(remaining):
        for pascal, lower in _TEMPLATE_TOKENS_SORTED:
            if remaining.startswith(pascal, pos):
                templates.append(lower)
                pos += len(pascal)
                break
        else:
            return None
    if not templates or any(tpl not in _TEMPLATE_VALID for tpl in templates):
        return None
    return (op, engine, executor, tuple(templates))


# ===== 算法名串语法（HCCL_ALGO 串 ↔ 名单 algos 项串） =====


def parse_algos(algos_str):
    """解析 HCCL_ALGO 语义算法串（";" 分隔多 token），返回 [(executor, [template, ...]), ...]。

    - 完整写法 "executor{tpl1,tpl2}"；简写 "mesh" 等价 "sole{mesh}"（executor 默认 sole）。
    - 多级模板记逗号序列（如 parallel{mesh,nhr} 的 templates 为 ["mesh","nhr"]）。
    - 空段跳过；token 中 executor / template 逐一校验权威枚举（6 executor / 13 template）。
    - 显式 level 语法（如 sequence{level1=nhr}）不表达，报错提示。
    """
    result = []
    for token in algos_str.split(";"):
        token = token.strip()
        if not token:
            continue  # 空段跳过
        brace_pos = token.find("{")
        if brace_pos >= 0:
            if not token.endswith("}"):
                raise ValueError("invalid algo token (missing '}}'): {!r}".format(token))
            executor = token[:brace_pos].strip()
            inner = token[brace_pos + 1:-1].strip()
            if not inner:
                raise ValueError("invalid algo token (empty template list): {!r}".format(token))
            templates = [t.strip() for t in inner.split(",") if t.strip()]
        else:
            executor = "sole"  # 简写默认 sole
            templates = [token]
        if executor not in EXECUTOR_TYPES:
            raise ValueError("invalid executor {!r} (valid: {})".format(executor, ",".join(EXECUTOR_TYPES)))
        for tpl in templates:
            if "=" in tpl:
                raise ValueError(
                    "explicit level syntax {!r} not supported (e.g. sequence{{level1=nhr}})".format(token))
            if tpl not in TEMPLATE_TYPES:
                raise ValueError("invalid template {!r} (valid: {})".format(tpl, ",".join(TEMPLATE_TYPES)))
        result.append((executor, templates))
    return result


def algo_to_env(executor, templates):
    """(executor, templates) → HCCL_ALGO 环境变量串（与 parse_algos 往返一致）。"""
    return "{}{{{}}}".format(executor, ",".join(templates))


def make_algo_string(engine, executor, templates):
    """(engine, executor, templates) → 名单 algos 项串，如 "aicpu_sole{mesh,nhr}"。"""
    return "{}_{}".format(engine, algo_to_env(executor, templates))


def parse_algo_string(algo_str):
    """名单 algos 项串 → (engine, executor, [tpl, ...])，与 make_algo_string 往返一致。

    引擎前缀必填（名单项由注册名反分解合成，均带 5 引擎前缀之一），
    executor{tpl} 段复用 parse_algos 校验（单 token）。
    """
    prefix, sep, rest = algo_str.partition("_")
    engine = prefix.lower()
    if not sep or engine not in ENGINE_ALL:
        raise ValueError("invalid algo string (engine prefix required): {!r}".format(algo_str))
    parsed = parse_algos(rest)
    if len(parsed) != 1:
        raise ValueError("invalid algo string (single token expected): {!r}".format(algo_str))
    executor, templates = parsed[0]
    return engine, executor, templates


# ===== probe 算法名单（JSON） =====
#
# probe 阶段从运行环境 costTable 日志收集 cost>=0 的候选算法并反分解落盘，
# 是采集计划剪枝的唯一真相源（分引擎轮次机制见 perf_test.py）。

PROBE_LIST_SCHEMA_VERSION = 4

# 分组条目字段（groups[]，每 (op × reduce_op × dtype) 一组；algo_count 为冗余派生字段，
# 便于速览）。算法池 algos 全组唯一一份，sizes 为 [数据量, [算法池下标...]] 升序列表，
# 档位间只重复编号不重复内容
PROBE_GROUP_FIELDS = ("op", "reduce_op", "data_type", "algo_count", "algos", "sizes")
# 算法池项字段（groups[].algos[]）：index 为池内下标（须等于数组位置，与 sizes 引用对齐），
# algo 为 HCCL_ALGO 写法（如 "aicpu_sole{mesh}"，不存 cost）；reg_name 为原始注册名，对照日志排查用
PROBE_ALGO_FIELDS = ("index", "algo", "reg_name")


def topo_fingerprint(topo):
    """拓扑 5 字段指纹；probe 名单与采集环境一致性校验用。"""
    return tuple(topo.get(field) for field in RANKS_FIELDS)


def save_probe_list(path, payload):
    """名单 JSON 落盘（覆盖写）；sizes 每档位压成单行，肉眼扫读。"""
    slots = []  # 压平的 sizes 条目，落盘前按占位符回填

    def _inline(item):
        slots.append(json.dumps(item, separators=(", ", ": ")))
        return "@@PROBE_SIZE_{}@@".format(len(slots) - 1)

    compact = json.loads(json.dumps(payload))  # 深拷贝，避免改写入参
    for group in compact.get("groups", []):
        group["sizes"] = [_inline(item) for item in group["sizes"]]
    text = json.dumps(compact, ensure_ascii=False, indent=1)
    for i, snippet in enumerate(slots):
        text = text.replace('"@@PROBE_SIZE_{}@@"'.format(i), snippet)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def load_probe_list(path):
    """读名单 JSON 并校验结构；版本不符或缺关键字段抛 ValueError。"""
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("probe list is not a JSON object: {}".format(path))
    if payload.get("schema_version") != PROBE_LIST_SCHEMA_VERSION:
        raise ValueError(
            "probe list schema version mismatch: file={} expected={}. Delete the file to re-probe.".format(
                payload.get("schema_version"), PROBE_LIST_SCHEMA_VERSION))
    if not isinstance(payload.get("topo"), dict) or not isinstance(payload.get("groups"), list):
        raise ValueError("probe list missing 'topo' or 'groups': {}".format(path))
    if not payload["groups"]:
        raise ValueError(
            "probe list has empty groups (last probe found nothing): {}".format(path))
    for group in payload["groups"]:
        missing = [field for field in PROBE_GROUP_FIELDS if field not in group]
        if missing:
            raise ValueError("probe group missing fields {}: {}".format(missing, group))
        algos, sizes = group["algos"], group["sizes"]
        if not isinstance(algos, list) or not algos:
            raise ValueError("probe group 'algos' must be a non-empty list: {}".format(group))
        if group["algo_count"] != len(algos):
            raise ValueError("probe group 'algo_count' != len(algos): {}".format(group))
        for pos, algo in enumerate(algos):
            missing = [field for field in PROBE_ALGO_FIELDS if field not in algo]
            if missing:
                raise ValueError("probe algo missing fields {}: {}".format(missing, algo))
            if algo["index"] != pos:
                raise ValueError("probe algo index must equal array position {}: {}".format(pos, algo))
        if not isinstance(sizes, list) or not sizes:
            raise ValueError("probe group 'sizes' must be a non-empty list: {}".format(group))
        for item in sizes:
            if (not isinstance(item, list) or len(item) != 2 or not isinstance(item[0], int)
                    or not isinstance(item[1], list)
                    or not all(isinstance(idx, int) and 0 <= idx < len(algos) for idx in item[1])):
                raise ValueError(
                    "probe group 'sizes' item must be [size_bytes, [algo index...]]: {}".format(item))
    return payload


def probe_list_covers(payload, op_types, reduce_ops, data_types, engines, size_params):
    """复用校验：名单探测参数须覆盖本次采集需求。

    op/reduce_op/dtype/engine 按子集判定（任一需求超出探测范围即不覆盖）；
    档位参数按相等判定（区间或步进不同则档位集不同，名单档位必然错位）。
    """
    params = payload.get("probe_params", {})
    return (set(op_types) <= set(params.get("op_types") or [])
            and set(reduce_ops) <= set(params.get("reduce_ops") or [])
            and set(data_types) <= set(params.get("data_types") or [])
            and set(engines) <= set(params.get("engines") or [])
            and size_params == (params.get("size_params") or {}))
