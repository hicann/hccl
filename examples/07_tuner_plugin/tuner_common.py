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

prof_test.py（采集）与 optimize_config.py（生成配置）共用的枚举与 CSV 数据模型，
只在此处维护。
"""

# op_type 白名单（对齐 plugin.cpp op_type_from_name；v 系列不采不产）
OP_TYPES = ("allreduce", "allgather", "broadcast", "reduce", "reduce_scatter", "scatter", "alltoall", "alltoallv")

# CSV 17 字段。性能数值字段名内嵌单位：latency=us、bandwidth=GB/s、timestamp=ms。
# algorithm.template_type 落 CSV 为带逗号语义串（"meshconcur,nhr"），
# 写 conf.json 时须转成 plugin 拼接串（见 templates_to_plugin_name）。
CSV_FIELDS = (
    "op_type", "size_bytes", "data_type", "reduce_type", "engine",
    "algorithm.executor_type", "algorithm.template_type", "HCCL_BUFFSIZE",
    "ranks.ranks", "ranks.npus_per_server", "ranks.servers", "ranks.pods", "ranks.super_pods",
    "check_result", "alg_bandwidth(GB/s)", "alg_latency(us)", "timestamp(ms)",
)

# 拓扑 5 字段（CSV ranks.* 列名）
RANKS_FIELDS = ("ranks.ranks", "ranks.npus_per_server", "ranks.servers", "ranks.pods", "ranks.super_pods")

# engine 规范名：device 引擎 4 种 + dpu
DEVICE_ENGINES = ("aicpu", "aiv", "ccums", "ccusched")
ENGINE_ALL = DEVICE_ENGINES + ("dpu",)

# executor 权威枚举（对齐 alg_parse.cc EXECUTOR_TYPES）
EXECUTOR_TYPES = ("sole", "sequence", "parallel", "pipeline", "concur", "strictordered")

# template 权威枚举（对齐 alg_parse.cc ALGO_TYPES）
TEMPLATE_TYPES = (
    "mesh", "mesh2die", "meshoneshot", "meshtwoshot", "meshconcur", "meshmultilink",
    "meshchunk", "meshchunktwoshot", "nhr", "nhrmultilink", "nhraicpureduce",
    "nhrsinglechannel", "meshconcurrent",
)

# template 两种口径：
# - CSV 语义串（带逗号，"meshconcur,nhr,nhr"）——与 HCCL_ALGO 语法一致
# - plugin 拼接串（不带逗号，"meshconcurnhrnhr"）——plugin 只认这一种


def templates_to_plugin_name(template_str):
    """语义串转拼接串（去逗号）；单级原样透传。"""
    return (template_str or "").replace(",", "")


# ===== plugin 白名单 =====

PLUGIN_OP_TYPES = frozenset(OP_TYPES)
PLUGIN_ENGINES = frozenset(ENGINE_ALL)

# plugin.cpp g_validExecutors 无 strictordered：写入会触发 SchemaError
# 整份 conf 失效，必须在此排除
PLUGIN_EXECUTORS = frozenset(EXECUTOR_TYPES) - frozenset(("strictordered",))

# plugin 对 template 不做枚举校验，全量放行
PLUGIN_TEMPLATES = frozenset(TEMPLATE_TYPES)

# ===== 每算子合法 (engine, executor, template) 组合表 =====
#
# 来源：src/ops/<op>/executor/*.cc 的 REGISTER_EXEC* 注册宏。算子并未注册
# 全笛卡尔积（如 allreduce 无 aicpu:sole{mesh}），采集时按本表过滤未注册组合。
# 仅收录能被 13 种 template token 完全分解的注册名；UBX/Pcie/MultiJetty 等
# 非 token 后缀的注册不入表。
# 写法 "engine:executor{tpl1,tpl2}"，多模板即多级算法。

_RAW_VALID_ALGOS = {
    "allreduce": (
        "aicpu:sole{meshoneshot}", "aicpu:sole{meshtwoshot}", "aicpu:sole{nhr}",
        "aicpu:sole{meshchunktwoshot}", "aicpu:sole{nhraicpureduce}", "aicpu:sole{meshconcur}",
        "aicpu:strictordered{mesh}",
        "aicpu:sequence{meshconcur,nhr}", "aicpu:sequence{meshconcur,nhr,nhr}",
        "aicpu:parallel{mesh,nhr}", "aicpu:concur{meshtwoshot,nhr}",
        "aiv:sole{meshoneshot}", "aiv:sole{meshtwoshot}",
        "ccums:sole{mesh}", "ccums:sole{mesh2die}", "ccums:sole{meshoneshot}", "ccums:sole{meshconcur}",
        "ccums:sequence{mesh2die}", "ccums:pipeline{mesh,nhr}", "ccums:concur{mesh,nhrmultilink}",
        "ccusched:sole{mesh}", "ccusched:sole{mesh2die}", "ccusched:sole{nhr}",
        "ccusched:sole{nhrmultilink}",
        "ccusched:sequence{mesh2die}", "ccusched:sequence{mesh,mesh}",
        "ccusched:parallel{mesh,nhr}", "ccusched:pipeline{mesh,nhr}",
        "ccusched:concur{mesh,nhrmultilink}",
        "dpu:sequence{mesh,nhr}", "dpu:pipeline{mesh,nhr,mesh}",
    ),
    "allgather": (
        "aicpu:sole{mesh}", "aicpu:sole{meshconcur}", "aicpu:sole{nhr}", "aicpu:sole{nhrmultilink}",
        "aicpu:sequence{meshconcur,nhr}", "aicpu:sequence{meshconcur,nhr,nhr}",
        "aicpu:parallel{mesh,nhr}", "aicpu:parallel{nhr,nhr}", "aicpu:concur{mesh,nhr}",
        "aiv:sole{mesh}",
        "ccums:sole{mesh}", "ccums:sole{mesh2die}", "ccums:concur{mesh,nhrmultilink}",
        "ccusched:sole{mesh}", "ccusched:sole{mesh2die}", "ccusched:sole{nhr}",
        "ccusched:sole{nhrmultilink}", "ccusched:sole{meshconcur}",
        "ccusched:sequence{mesh,mesh}", "ccusched:parallel{mesh,nhr}",
        "ccusched:parallel{mesh,nhrmultilink}", "ccusched:concur{mesh,nhrmultilink}",
        "ccusched:pipeline{mesh,nhr}",
        "dpu:sole{nhr}", "dpu:sequence{mesh,nhr}", "dpu:pipeline{mesh,nhr,nhr}",
    ),
    "broadcast": (
        "aicpu:sole{meshtwoshot}", "aicpu:sole{nhr}",
        "aicpu:sequence{meshconcur,nhr}", "aicpu:sequence{meshconcur,nhr,nhr}",
        "aicpu:parallel{mesh,nhr}",
        "aiv:sole{mesh}",
        "ccums:sole{mesh}",
        "ccusched:sole{mesh}", "ccusched:sole{nhr}", "ccusched:pipeline{mesh,nhr}",
        "ccusched:parallel{mesh,nhr}",
        "dpu:sequence{mesh,nhr}",
    ),
    "reduce": (
        "aicpu:sole{mesh}", "aicpu:sole{meshtwoshot}", "aicpu:sole{nhr}", "aicpu:sole{nhraicpureduce}",
        "aicpu:sequence{meshconcur,nhr}", "aicpu:sequence{meshconcur,nhr,nhr}",
        "aicpu:parallel{mesh,nhr}",
        "aiv:sole{mesh}",
        "ccums:sole{mesh}", "ccums:pipeline{mesh,nhr}",
        "ccusched:sole{mesh}", "ccusched:sole{meshtwoshot}", "ccusched:sole{nhr}",
        "ccusched:parallel{mesh,nhr}", "ccusched:pipeline{mesh,nhr}",
        "dpu:sequence{mesh,nhr}",
    ),
    "reduce_scatter": (
        "aicpu:sole{mesh}", "aicpu:sole{meshchunk}", "aicpu:sole{nhr}", "aicpu:sole{nhraicpureduce}",
        "aicpu:sole{nhrmultilink}", "aicpu:sole{meshconcur}", "aicpu:strictordered{mesh}",
        "aicpu:concur{mesh,nhr}", "aicpu:parallel{mesh,nhr}",
        "aicpu:sequence{meshconcur,nhr}", "aicpu:sequence{meshconcur,nhr,nhr}",
        "aiv:sole{mesh}",
        "ccums:sole{mesh}", "ccums:sole{mesh2die}", "ccums:sole{meshconcur}",
        "ccums:pipeline{mesh,nhr}", "ccums:concur{mesh,nhrmultilink}",
        "ccusched:sole{mesh}", "ccusched:sole{mesh2die}", "ccusched:sole{nhr}",
        "ccusched:sole{nhrmultilink}",
        "ccusched:sequence{mesh,mesh}", "ccusched:parallel{mesh,nhr}",
        "ccusched:parallel{mesh,nhrmultilink}", "ccusched:pipeline{mesh,nhr}",
        "ccusched:concur{mesh,nhrmultilink}",
        "dpu:sequence{mesh,mesh}", "dpu:pipeline{mesh,nhr,mesh}",
    ),
    "scatter": (
        "aicpu:sole{mesh}", "aicpu:sole{nhr}",
        "aicpu:sequence{meshconcur,nhr}", "aicpu:sequence{meshconcur,nhr,nhr}",
        "aicpu:parallel{mesh,nhr}",
        "aiv:sole{mesh}",
        "ccusched:sole{mesh}", "ccusched:sole{nhr}", "ccusched:pipeline{mesh,nhr}",
        "ccusched:parallel{mesh,nhr}",
        "dpu:sequence{mesh,nhr}",
    ),
    "alltoall": (
        "aicpu:sole{mesh}", "aicpu:sole{nhrsinglechannel}", "aicpu:sole{meshconcurrent}",
        "aiv:sole{mesh}",
        "ccusched:sole{mesh}", "ccusched:sole{mesh2die}", "ccusched:sole{meshmultilink}",
        "ccusched:sole{meshconcur}", "ccusched:sole{meshconcurrent}",
        "dpu:sole{mesh}",
    ),
    "alltoallv": (
        "aicpu:sole{mesh}", "aicpu:sole{meshconcurrent}",
        "aiv:sole{mesh}",
        "ccusched:sole{mesh}", "ccusched:sole{mesh2die}", "ccusched:sole{meshmultilink}",
        "ccusched:sole{meshconcurrent}",
        "dpu:sole{mesh}",
    ),
}


def _parse_valid_algo(spec):
    """'engine:executor{tpl1,tpl2}' → (engine, executor, (tpl1, tpl2))。"""
    engine, rest = spec.split(":", 1)
    executor, inner = rest.split("{", 1)
    templates = tuple(inner.rstrip("}").split(","))
    return (engine, executor, templates)


# op → frozenset{(engine, executor, templates_tuple)}
OP_VALID_ALGOS = {op: frozenset(_parse_valid_algo(s) for s in specs)
                  for op, specs in _RAW_VALID_ALGOS.items()}

# 同上但保序，供采集侧默认全量展开（含多级组合）
OP_VALID_ALGO_LIST = {op: [_parse_valid_algo(s) for s in specs]
                      for op, specs in _RAW_VALID_ALGOS.items()}

# ===== 拓扑层数约束（采集计划剪枝用） =====
#
# 依据各算子 REGISTER_ALG_ATTRS 声明的 min/maxTopoLevelNum 归纳。环境层数
# 不满足时 HCCL_ALGO 只会白跑出 failed/noresult，脚本据此提前剪枝。
# 运行时层数拿不到，只能按 CLI 参数估算（见 prof_test.estimate_topo_levels）。
# 表键 (engine, executor, templates) → (min, max)；表外组合回退默认值 (1, 3)；
# sequence 且首级 meshconcur 的多级算法要求恰为模板级数层。
# aiv 的 sole{mesh} 家族（allreduce 的 oneshot/twoshot 与其余算子的 mesh）统一
# 注册 maxTopoLevelNum=2，须列 (1,2)；aiv 的 alltoall/v sole{mesh} 未声明层数，
# 回退默认。mesh 家族并非全是单层：scatter 的 aicpu sole{mesh}/{nhr} 注册 max=3
# （恰好等于默认值，故不列）。ccums/ccusched 采集未放开，条目保留备用。
# 注意：部分 aiv 注册还带 topoCustomCheck（rank 数上限）等非层数约束，脚本
# 无法建模，层数估算只是必要条件而非充分条件。
_TOPO_LEVEL_CONSTRAINTS = {
    "allreduce": {
        ("aicpu", "parallel", ("mesh", "nhr")): (2, 2),
        ("aicpu", "sole", ("meshchunktwoshot",)): (1, 1),
        ("aicpu", "sole", ("meshconcur",)): (1, 1),
        ("aicpu", "sole", ("meshoneshot",)): (1, 1),
        ("aicpu", "sole", ("meshtwoshot",)): (1, 1),
        ("aiv", "sole", ("meshoneshot",)): (1, 2),
        ("aiv", "sole", ("meshtwoshot",)): (1, 2),
        ("ccums", "sole", ("mesh2die",)): (1, 1),
        ("ccums", "sole", ("meshconcur",)): (1, 1),
        ("ccums", "sequence", ("mesh2die",)): (1, 1),
        ("ccusched", "parallel", ("mesh", "nhr")): (1, 2),
        ("ccusched", "sequence", ("mesh", "mesh")): (1, 2),
        ("ccusched", "sequence", ("mesh2die",)): (1, 1),
        ("ccusched", "sole", ("mesh2die",)): (1, 1),
        ("ccusched", "sole", ("nhr",)): (1, 2),
        ("ccusched", "sole", ("nhrmultilink",)): (1, 1),
        ("dpu", "sequence", ("mesh", "nhr")): (2, 3),
    },
    "allgather": {
        ("aicpu", "parallel", ("mesh", "nhr")): (2, 2),
        ("aicpu", "parallel", ("nhr", "nhr")): (3, 3),
        ("aicpu", "sole", ("mesh",)): (1, 1),
        ("aicpu", "sole", ("meshconcur",)): (1, 1),
        ("aiv", "sole", ("mesh",)): (1, 2),
        ("ccums", "sole", ("mesh2die",)): (1, 1),
        ("ccusched", "parallel", ("mesh", "nhr")): (2, 2),
        ("ccusched", "sequence", ("mesh", "mesh")): (2, 2),
        ("ccusched", "sole", ("meshconcur",)): (1, 1),
        ("dpu", "sequence", ("mesh", "nhr")): (2, 3),
    },
    "broadcast": {
        ("aicpu", "sole", ("meshtwoshot",)): (1, 1),
        ("aiv", "sole", ("mesh",)): (1, 2),
    },
    "reduce": {
        ("aicpu", "parallel", ("mesh", "nhr")): (2, 2),
        ("aicpu", "sole", ("mesh",)): (1, 1),
        ("aicpu", "sole", ("meshtwoshot",)): (1, 1),
        ("aiv", "sole", ("mesh",)): (1, 2),
    },
    "reduce_scatter": {
        ("aicpu", "parallel", ("mesh", "nhr")): (2, 2),
        ("aicpu", "sole", ("mesh",)): (1, 1),
        ("aicpu", "sole", ("meshchunk",)): (1, 1),
        ("aicpu", "sole", ("meshconcur",)): (1, 1),
        ("aiv", "sole", ("mesh",)): (1, 2),
        ("ccums", "sole", ("mesh2die",)): (1, 1),
        ("ccums", "sole", ("meshconcur",)): (1, 1),
        ("ccusched", "parallel", ("mesh", "nhr")): (2, 2),
        ("ccusched", "sequence", ("mesh", "mesh")): (2, 2),
        ("dpu", "sequence", ("mesh", "mesh")): (2, 3),
    },
    "scatter": {
        ("aiv", "sole", ("mesh",)): (1, 2),
    },
}

# 表外组合的默认约束
_DEFAULT_LEVEL_CONSTRAINT = (1, 3)

# 采集黑名单：concur 执行器与含 multilink/meshconcurrent 的模板暂不采集。
# 老选择逻辑下这些算法在普通机型/组网上大概率静默回退选其他算法（HCCL_ALGO
# 匹配落空不报错），测出的耗时张冠李戴，故整组排除。meshconcurrent 注册为
# TopoMatchUBX（meshclos 方阵组网专属），executor 仍是 sole 故 concur 拦不住；
# meshconcur 模板（sequence{meshconcur,nhr} 等）不在此列，由层数门控负责。
ALGO_BLACKLIST_KEYWORDS = ("multilink", "meshconcurrent")


def is_blacklisted(executor, templates):
    """executor=concur 或任一模板含 multilink/meshconcurrent 时为 True。"""
    if executor == "concur":
        return True
    return any(kw in tpl for tpl in templates for kw in ALGO_BLACKLIST_KEYWORDS)


def _lookup_level_constraint(op_type, engine, executor, templates):
    """查 (min, max)：先查表，再回退 meshconcur 多级规则，最后默认值。"""
    key = (engine, executor, tuple(templates))
    constraint = _TOPO_LEVEL_CONSTRAINTS.get(op_type, {}).get(key)
    if constraint is not None:
        return constraint
    if executor == "sequence" and templates and templates[0] == "meshconcur":
        return (len(templates), len(templates))
    return _DEFAULT_LEVEL_CONSTRAINT


def topo_min_level(op_type, engine, executor, templates):
    """该注册组合要求的最低拓扑层数。"""
    return _lookup_level_constraint(op_type, engine, executor, templates)[0]


def topo_max_level(op_type, engine, executor, templates):
    """该注册组合允许的最高拓扑层数。"""
    return _lookup_level_constraint(op_type, engine, executor, templates)[1]
