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

"""Tuner Plugin 算法数据采集脚本。

构造 mpirun 命令拉起 hccl_test 可执行文件（./bin/<op>_test），解析其 stdout 表格，
将各算法在不同拓扑 / 数据量 / 数据类型下的性能数据落为 CSV，供 optimize_config.py 生成最优配置。

用法示例：
    python3 prof_test.py --op-types allreduce --engines aicpu,aiv \\
        --minbytes 1K --maxbytes 1M --stepfactor 2 --data-types fp32 \\
        --np-total 16 --npus 8 --hostfile hostfile --output hccl_prof.csv
"""

import argparse
import csv
import itertools
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from collections import namedtuple

from tuner_common import (
    CSV_FIELDS, DEVICE_ENGINES, ENGINE_ALL, EXECUTOR_TYPES, OP_TYPES, OP_VALID_ALGOS,
    RANKS_FIELDS, TEMPLATE_TYPES, OP_VALID_ALGO_LIST, is_blacklisted, topo_max_level,
    topo_min_level,
)

# ===== 本脚本特有常量 =====

# op_type → hccl_test 可执行文件映射（./bin/ 下）
OP_TO_EXE = {
    "allreduce": "all_reduce_test",
    "allgather": "all_gather_test",
    "broadcast": "broadcast_test",
    "reduce": "reduce_test",
    "reduce_scatter": "reduce_scatter_test",
    "scatter": "scatter_test",
    "alltoall": "alltoall_test",
    "alltoallv": "alltoallv_test",
}

# data_type 采集域（hccl_test -d 全部 16 种）
DATA_TYPES = (
    "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64",
    "fp16", "fp32", "fp64", "bfp16", "fp8e5m2", "fp8e4m3", "fp8e8m0", "hif8",
)

# reduce 类算子（hccl_test -o 仅对这些算子有效）
REDUCE_OP_TYPES = ("allreduce", "reduce", "reduce_scatter")
REDUCE_OPS = ("sum", "prod", "max", "min")

# engine → hccl_test -a 下发值（常量表集中维护；dpu 不走 -a，拓扑绑定）
# 注：ccu_ms_only / ccu_sched_only 为预留命名，当前 hccl_test -a 无 _only 变体，
#     待补齐后仅需改本表一行即可切换。
ENGINE_TO_ACCEL = {
    "aicpu": "aicpu_ts",
    "aiv": "aiv_only",
    "ccums": "ccu_ms",     # 预留 ccu_ms_only
    "ccusched": "ccu_sched",  # 预留 ccu_sched_only
}

# 采集侧当前支持的引擎白名单：aicpu / aiv / dpu。ccums / ccusched 暂不放开：
# 当前 hccl 不支持 ccu_ms_only / ccu_sched_only 模式，配了 ccu 算法也可能回退到
# aicpu 算法执行，测出的耗时不准，会污染选优结果，后续放开时改本表即可
# （dpu 不走 -a，由 build_test_args 的 accel=None 分支处理）。
PROF_ENGINES = ("aicpu", "aiv", "dpu")

# hccl_test stdout 表头列名（去掉单位后缀）→ CSV 字段映射
# 注：实际二进制打印 "aveg_time(us):"（hccl_opbase_rootinfo_base.h 源码拼写），
#     文档写 "avg_time(us):"，两者都收。
STDOUT_COL_MAP = {
    "data_size": "size_bytes",
    "avg_time": "alg_latency(us)",
    "aveg_time": "alg_latency(us)",
    "alg_bandwidth": "alg_bandwidth(GB/s)",
    "check_result": "check_result",
}

MIN_PYTHON = (3, 8)

# hccl_test 相对 $ASCEND_HOME_PATH 的默认子目录
DEFAULT_BIN_SUBDIR = "tools/hccl_test/bin"

MPIRUN_MISSING_HINT = (
    "mpirun not found in PATH. Please install MPICH 4.1.3 "
    "(https://www.mpich.org/downloads/) or Open MPI 4.1.5 "
    "(https://www.open-mpi.org/software/ompi/v4.1/).")
HCCL_TEST_MISSING_HINT = (
    "hccl_test executable not found: {path}. Please compile hccl_test from CANN "
    "source (oam-tools src/hccl_test) and place binaries under "
    "$ASCEND_HOME_PATH/tools/hccl_test/bin/."
)
ASCEND_HOME_MISSING_HINT = (
    "ASCEND_HOME_PATH not set. Please source the CANN environment script first, "
    "e.g. 'source ${INSTALL_DIR}/bin/setenv.bash', so hccl_test can be located "
    "at $ASCEND_HOME_PATH/tools/hccl_test/bin/."
)

# 模块级 logger：main() 内 basicConfig 统一配置
logger = logging.getLogger("prof_test")


class ProfTestError(Exception):
    """采集脚本参数 / 环境错误。"""


class PreflightError(ProfTestError):
    """启动自检（preflight）失败。"""


# ===== 纯函数（L0 单测入口）=====

def parse_bytes(spec):
    """解析数据量字符串，支持 K/M/G 后缀（大小写不敏感）与小数。

    "1K" -> 1024，"4M" -> 4194304，"2G" -> 2147483648，"512" -> 512，"1.5K" -> 1536。
    """
    spec = str(spec).strip()
    match = re.fullmatch(r"([0-9]*\.?[0-9]+)\s*([kKmMgG]?)", spec)
    if match is None:
        raise ValueError("invalid bytes spec: {!r}".format(spec))
    num = float(match.group(1))
    suffix = match.group(2).lower()
    multiplier = {"": 1, "k": 1024, "m": 1024 ** 2, "g": 1024 ** 3}[suffix]
    return int(round(num * multiplier))


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
                raise ValueError("invalid algo token (missing '}'): {!r}".format(token))
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


def expand_list(value, valid, name):
    """展开逗号分隔 CLI 列表；支持 all；逐项校验白名单。"""
    value = (value or "").strip()
    if not value or value == "all":
        return list(valid)
    items = [v.strip() for v in value.split(",") if v.strip()]
    for item in items:
        if item not in valid:
            raise ValueError("invalid {} {!r} (valid: {})".format(name, item, ",".join(valid)))
    return items


def build_algos(algos_arg, executors_arg, templates_arg):
    """算法组合双入口：--algos 优先透传解析；否则 --executors × --templates 笛卡尔积。

    三参数均未指定时返回 None：由 make_collect_plan 直接展开注册表全量组合
    （含多级），避免单模板笛卡尔积静默剪掉多级算法。
    """
    if algos_arg:
        return parse_algos(algos_arg)
    if executors_arg is None and templates_arg is None:
        return None
    executors = expand_list(executors_arg, EXECUTOR_TYPES, "executor")
    templates = expand_list(templates_arg, TEMPLATE_TYPES, "template")
    return [(executor, [tpl]) for executor in executors for tpl in templates]


def _topo_level_ok(op_type, engine, executor, templates, topo_levels):
    """组合层数约束是否被估算层数满足（min 用下界判断，max 用上界判断）。"""
    if topo_min_level(op_type, engine, executor, templates) > topo_levels:
        return False
    max_level = topo_max_level(op_type, engine, executor, templates)
    return max_level is None or topo_levels <= max_level


def filter_valid_algos(op_type, engine, algos, topo_levels=1):
    """按 (op, engine)、黑名单与拓扑层数过滤出可采的算法组合，返回 (有效, 被剪)。

    algos 为 None 时展开该 (op, engine) 的全部注册组合（含多级）；否则按
    (engine, executor, tuple(templates)) 匹配注册表——算子并非注册全笛卡尔积
    （如 allreduce 无 aicpu:sole{mesh}），未注册组合采了也是空跑。
    注册表未覆盖的 op 不剪裁。
    """
    valid_set = OP_VALID_ALGOS.get(op_type)
    if valid_set is None:
        return list(algos) if algos is not None else [], []
    if algos is None:
        valid = [(executor, list(templates)) for reg_engine, executor, templates
                 in OP_VALID_ALGO_LIST[op_type]
                 if reg_engine == engine and not is_blacklisted(executor, templates)
                 and _topo_level_ok(op_type, engine, executor, templates, topo_levels)]
        return valid, []
    valid, skipped = [], []
    for executor, templates in algos:
        key = (engine, executor, tuple(templates))
        if key not in valid_set or is_blacklisted(executor, templates):
            skipped.append((executor, templates))
        elif _topo_level_ok(op_type, engine, executor, tuple(templates), topo_levels):
            valid.append((executor, templates))
        else:
            skipped.append((executor, templates))
    return valid, skipped


def engine_to_accelerator(engine):
    """engine 规范名 → hccl_test -a 下发值；dpu 返回 None（不走 -a）。"""
    if engine not in ENGINE_ALL:
        raise ValueError("invalid engine {!r} (valid: {})".format(engine, ",".join(ENGINE_ALL)))
    return ENGINE_TO_ACCEL.get(engine)


def reduce_type_for_op(op_type, reduce_op):
    """reduce 类算子返回 reduce_op，其余固定 NA（CSV reduce_type 字段口径）。"""
    return reduce_op if op_type in REDUCE_OP_TYPES else "NA"


def detect_mpi_env():
    """探测当前进程 MPI 环境，返回 (rank, size)；无 MPI 环境返回 (0, 1)。

    Open MPI: OMPI_COMM_WORLD_RANK/SIZE；MPICH/PMI: PMI_RANK/PMI_SIZE。
    """
    for rank_key, size_key in (("OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"),
                               ("PMI_RANK", "PMI_SIZE")):
        if rank_key in os.environ and size_key in os.environ:
            return int(os.environ[rank_key]), int(os.environ[size_key])
    return 0, 1


def detect_mpi_flavor():
    """MPI 类别探测：OMPI_COMM_WORLD_RANK 存在 → openmpi，否则按 mpich 命令格式。"""
    return "openmpi" if "OMPI_COMM_WORLD_RANK" in os.environ else "mpich"


def check_mpi_env():
    """MPI 环境监测：返回 flavor 供命令构造，并拦截嵌套运行。

    脚本自身已在 MPI 会话内（size>1）时抛错——此时再拉 mpirun 会按 rank 数
    放大采集进程数，采集无法收敛。必须在会话外（登录节点）运行本脚本。
    """
    rank, size = detect_mpi_env()
    if size > 1:
        raise PreflightError(
            "prof_test.py is running inside an MPI session (rank {} of {}); run it "
            "outside mpirun, otherwise each round launches {} nested MPI ranks".format(
                rank, size, size))
    return detect_mpi_flavor()


def count_hostfile_lines(hostfile):
    """hostfile 非空行数（派生 servers；注释 # 与空行忽略）。"""
    servers = 0
    with open(hostfile, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                servers += 1
    return servers


def derive_topo(np_total, npus, hostfile, pods, super_pods):
    """派生 ranks.* 拓扑 5 字段：ranks / npus_per_server / servers / pods / super_pods。"""
    servers = count_hostfile_lines(hostfile) if hostfile else 1
    values = (np_total, npus, servers, pods, super_pods)
    return dict(zip(RANKS_FIELDS, values))


def estimate_topo_levels(np_total, npus, hostfile, pods, super_pods):
    """按 CLI 参数估算拓扑层数**下界**：1=单 POD/server；+server/POD 间一层；+superpod 间一层。

    HCCL 运行时从 rank 图精确算 topoLevelNums，脚本侧只能从参数估：
    - server >1 或 pods >1 → ≥2 层；super_pods >1 → 3 层；
    - pods/super_pods 未指定按 1 处理，属**下界**：多 superpod 环境务必显式传
      --super-pods，否则 3 层专属算法（sequence{meshconcur,nhr,nhr}）会被误剪。
    """
    servers = count_hostfile_lines(hostfile) if hostfile else max(1, np_total // max(npus, 1))
    if super_pods > 1:
        return 3
    if servers > 1 or pods > 1:
        return 2
    return 1


# 采集轮固定参数（整个采集过程不变，来自 CLI；与每轮变化的 op/dtype/engine 分离）
TestProfile = namedtuple("TestProfile", [
    "npus", "minbytes", "maxbytes", "stepbytes", "stepfactor",
    "reduce_op", "warmup_iters", "iters",
])

# 采集计划：维度输入 + 剪枝结果（make_collect_plan 的返回契约）
CollectPlan = namedtuple("CollectPlan", [
    "op_types", "data_types", "engines", "algos",
    "valid_by_op_engine", "tested_algos", "valid_combos", "raw_combos", "total_runs",
    "topo_levels",
])

# 采集执行环境：CLI args + 固定参数 + 路径/拓扑/MPI 环境（每轮共享）
RunContext = namedtuple("RunContext", [
    "args", "profile", "exe_paths", "hccl_buffsize", "env_keys", "topo", "total_runs",
    "mpi_flavor",
])


def make_test_profile(args):
    """CLI args → TestProfile（固定参数打包）。"""
    return TestProfile(npus=args.npus, minbytes=args.minbytes, maxbytes=args.maxbytes,
                       stepbytes=args.stepbytes, stepfactor=args.stepfactor,
                       reduce_op=args.reduce_op, warmup_iters=args.warmup_iters,
                       iters=args.iters)


def build_test_args(profile, op_type, data_type, engine):
    """构造 hccl_test 可执行文件参数（exe 之后部分）。

    profile 为采集轮固定参数（TestProfile）；op_type/data_type/engine 为本轮维度。
    -b/-e 传原始字符串（hccl_test 接受 K/M/G 后缀）；-i/-f 二选一；
    -o 仅 reduce 类算子下发；-a 仅 device 引擎下发（dpu 不走 -a）。
    """
    args = []
    if profile.npus:
        args += ["-p", str(profile.npus)]
    if profile.minbytes is not None:
        args += ["-b", profile.minbytes]
    if profile.maxbytes is not None:
        args += ["-e", profile.maxbytes]
    if profile.stepbytes is not None:
        args += ["-i", str(profile.stepbytes)]
    elif profile.stepfactor is not None:
        args += ["-f", str(profile.stepfactor)]
    args += ["-d", data_type]
    if op_type in REDUCE_OP_TYPES:
        args += ["-o", profile.reduce_op]
    args += ["-w", str(profile.warmup_iters), "-n", str(profile.iters)]
    accel = engine_to_accelerator(engine)
    if accel is not None:
        args += ["-a", accel]
    return args


def build_mpirun_cmd(np_total, hostfile, exe_argv, env_keys=None, flavor=None):
    """构造 mpirun 命令（MPICH / Open MPI 双支持）。

    exe_argv 为被拉起进程的完整 argv（exe 路径 + 参数）。
    MPICH:   mpirun [-f hostfile] -n N <exe> ...
    OpenMPI: mpirun [-hostfile hostfile] -n N [-x ENV ...] <exe> ...
    env_keys：需要 -x 显式传递到远端节点的环境变量名（仅 OpenMPI）。
    """
    flavor = flavor or detect_mpi_flavor()
    cmd = ["mpirun"]
    if hostfile:
        cmd += (["-hostfile", hostfile] if flavor == "openmpi" else ["-f", hostfile])
    cmd += ["-n", str(np_total)]
    if flavor == "openmpi" and env_keys:
        for key in env_keys:
            cmd += ["-x", key]
    cmd += exe_argv
    return cmd


def _norm_col(name):
    """表头列名归一化：去空白 / 冒号，截掉 "(单位)" 后缀，转小写。"""
    name = name.strip().rstrip(":").strip()
    paren = name.find("(")
    if paren > 0:
        name = name[:paren]
    return name.strip().lower()


def _parse_header_row(cells):
    """尝试把 '|' 分隔的 cells 解析为表头；是表头返回列映射，否则 None。

    须至少包含 data_size 列才认作表头行。
    """
    mapped = [(STDOUT_COL_MAP.get(_norm_col(cell)), cell) for cell in cells]
    return mapped if any(field == "size_bytes" for field, _ in mapped) else None


def _parse_data_row(columns, cells):
    """按表头列映射把 cells 解析为数据行 dict；解析失败返回 None。

    size_bytes 取 int，alg_latency / alg_bandwidth 取 float（非法值整行丢弃）。
    """
    row = {}
    for (field, _), cell in zip(columns, cells):
        if field is None:
            continue
        if field == "size_bytes":
            try:
                row[field] = int(cell)
            except ValueError:
                return None
        elif field in ("alg_latency(us)", "alg_bandwidth(GB/s)"):
            try:
                row[field] = float(cell)
            except ValueError:
                return None
        else:
            row[field] = cell
    return row if "size_bytes" in row else None


def parse_stdout(text):
    """解析 hccl_test stdout 表格，返回行字典列表。

    按表头列名匹配（非固定索引，兼容列序差异）；data_size/avg_time/alg_bandwidth/
    check_result 四项落 dict；首行汇总信息、表头行、空行、列数不符行忽略。
    """
    rows = []
    columns = None  # [(csv_field or None, 原始列名), ...]
    for line in text.splitlines():
        if "|" not in line:
            continue  # 汇总行 / 空行 / 提示信息忽略
        cells = [cell.strip() for cell in line.split("|")]
        if columns is None:
            columns = _parse_header_row(cells)
            continue
        if len(cells) != len(columns):
            continue  # 列数不匹配的行（如日志前缀）忽略
        row = _parse_data_row(columns, cells)
        if row is not None:
            rows.append(row)
    return rows


_ITER_RE = re.compile(r"iters\s+is\s+(\d+).*warmup_iters\s+is\s+(\d+)")


def parse_header_info(text):
    """解析 stdout 首行回显 '... iters is 20, warmup_iters is 5' → (iters, warmup_iters)。

    用于校验 -w/-n 是否真正下发生效（hccl_test 回显实际生效值）。
    找不到回显行返回 None。
    """
    for line in text.splitlines():
        match = _ITER_RE.search(line)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


# ===== preflight 启动自检 =====

def check_python_version(version_info=None):
    """Python 版本检查：>= 3.8，不满足报错退出。"""
    version_info = version_info or sys.version_info
    if version_info < MIN_PYTHON:
        raise PreflightError(
            "Python >= {}.{} required, got {}.{}. Please upgrade Python.".format(
                MIN_PYTHON[0], MIN_PYTHON[1], version_info[0], version_info[1]))
    return True


def resolve_bin_dir(bin_dir=None, env=None):
    """hccl_test bin 目录 = $ASCEND_HOME_PATH + 相对路径拼接。

    bin_dir 为相对子目录（默认 tools/hccl_test/bin）；绝对路径时直接使用。
    ASCEND_HOME_PATH 未设置抛 PreflightError。
    """
    sub_dir = bin_dir or DEFAULT_BIN_SUBDIR
    if os.path.isabs(sub_dir):
        return sub_dir
    home = (env or os.environ).get("ASCEND_HOME_PATH", "").strip()
    if not home:
        raise PreflightError(ASCEND_HOME_MISSING_HINT)
    return os.path.join(home, sub_dir)


def exe_path_of(op_type, bin_dir=None, env=None):
    """op_type → hccl_test 可执行文件完整路径（$ASCEND_HOME_PATH 拼接）。"""
    return os.path.join(resolve_bin_dir(bin_dir, env), OP_TO_EXE[op_type])


def preflight(op_types, bin_dir=None, env=None):
    """启动自检：Python 版本 / mpirun 存在 / MPI 环境 / hccl_test 可执行文件存在。

    任一关键项不满足即抛 PreflightError（调用方报错退出）；不触发任何下载。
    通过时返回 MPI flavor（"openmpi"/"mpich"）。脚本仅用 Python 标准库。
    """
    check_python_version()
    if shutil.which("mpirun") is None:
        raise PreflightError(MPIRUN_MISSING_HINT)
    flavor = check_mpi_env()
    for op_type in op_types:
        exe_path = exe_path_of(op_type, bin_dir, env)
        if not os.path.isfile(exe_path):
            raise PreflightError(HCCL_TEST_MISSING_HINT.format(path=exe_path))
    return flavor


# ===== CSV 写 =====

class CsvWriter:
    """CSV 追加写：新文件写表头，已有文件续写（表头校验防混排）。

    增量采集多轮 run 追加同一 CSV，optimize_config 侧按 timestamp 去重。
    """

    def __init__(self, path):
        need_header = not os.path.exists(path) or os.path.getsize(path) == 0
        self._handle = open(path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._handle)
        if need_header:
            self._writer.writerow(CSV_FIELDS)
        else:
            self._ensure_trailing_newline(path)
            self._check_existing_header(path)

    def _ensure_trailing_newline(self, path):
        """追加前检查末字节：非换行则补一个，避免续写行与旧行粘连成脏行。

        常见于上一轮进程被强杀（kill -9 / 超时清理）导致最后一行未完整落盘。
        """
        with open(path, "rb") as handle:
            handle.seek(-1, os.SEEK_END)
            last_byte = handle.read(1)
        if last_byte != b"\n":
            self._handle.write("\n")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def write_row(self, row):
        self._writer.writerow([row.get(field, "") for field in CSV_FIELDS])

    def close(self):
        self._handle.close()

    def _check_existing_header(self, path):
        """已有文件表头必须与当前 17 字段完全一致，否则拒绝追加（防异构混排）。"""
        with open(path, newline="", encoding="utf-8") as handle:
            existing = next(csv.reader(handle), None)
        if existing != list(CSV_FIELDS):
            self._handle.close()
            raise ValueError(
                "existing CSV header mismatch: {}. Expected {} fields {}. Remove/rename the "
                "file or use a new --output path.".format(
                    path, len(CSV_FIELDS), ",".join(CSV_FIELDS)))


# ===== CLI 与主流程 =====

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Tuner Plugin 算法数据采集：构造 mpirun + hccl_test 命令，解析 stdout 落 CSV")
    parser.add_argument("--op-types", required=True,
                        help="算子白名单 8 种，逗号分隔多值，支持 all")
    parser.add_argument("--minbytes", default=None, help="数据量区间起始（K/M/G 后缀），对应 -b")
    parser.add_argument("--maxbytes", default=None, help="数据量区间结束（K/M/G 后缀），对应 -e")
    parser.add_argument("--stepbytes", type=int, default=None, help="线性步进（Bytes），对应 -i，与 --stepfactor 互斥")
    parser.add_argument("--stepfactor", type=float, default=None, help="等比因子，对应 -f，与 --stepbytes 互斥")
    parser.add_argument("--data-types", default="fp32", help="dtype 16 种，逗号分隔多值，默认 fp32，对应 -d")
    parser.add_argument("--reduce-op", default="sum", choices=REDUCE_OPS,
                        help="reduce 类算子操作类型（CSV reduce_type），默认 sum，对应 -o")
    parser.add_argument("--engines", default="all",
                        help="采集引擎，当前仅支持 aicpu（其余拦截），all=aicpu")
    parser.add_argument("--algos", default=None,
                        help="算法串（HCCL_ALGO 语义，';' 分隔，如 sole{mesh};parallel{mesh,nhr}）")
    parser.add_argument("--executors", default=None,
                        help="快捷全组合入口，默认权威 6 executor，支持 all（--algos 存在时忽略）")
    parser.add_argument("--templates", default=None,
                        help="快捷全组合入口，默认权威 13 template，支持 all（--algos 存在时忽略）")
    parser.add_argument("--warmup-iters", type=int, default=10, help="预热次数，默认 10，对应 -w")
    parser.add_argument("--iters", type=int, default=20, help="迭代次数，默认 20，对应 -n")
    parser.add_argument("--run-timeout", type=int, default=600, metavar="SECONDS",
                        help="单轮 mpirun 超时秒数，默认 600；超时杀进程树，该轮落 failed 并继续采集")
    parser.add_argument("--pods", type=int, default=0,
                        help="pod 数（拓扑层级，用户指定）；0=未指定（默认），conf 规则不约束该维度")
    parser.add_argument("--super-pods", type=int, default=0,
                        help="super pod 数（拓扑层级，用户指定）；0=未指定（默认），conf 规则不约束该维度")
    parser.add_argument("--hostfile", default=None, help="多机 hostfile 文件（派生 servers），单机省略")
    parser.add_argument("--npus", type=int, default=None, help="每 server NPU 数（派生 npus_per_server），对应 -p")
    parser.add_argument("--np-total", type=int, required=True, help="总 rank 数（派生 ranks），对应 mpirun -n")
    parser.add_argument("--output", default="hccl_prof.csv", help="CSV 输出路径（覆盖模式重建）")
    parser.add_argument("--bin-dir", default=DEFAULT_BIN_SUBDIR,
                        help="hccl_test 可执行文件目录：相对 $ASCEND_HOME_PATH 的子目录"
                             "（默认 tools/hccl_test/bin），也可给绝对路径")
    return parser.parse_args(argv)


def validate_args(args):
    """CLI 参数白名单校验：op-type / engine / dtype / 区间 / 步进互斥。"""
    op_types = expand_list(args.op_types, OP_TYPES, "op_type")
    # v 系列（allgatherv / reducescatterv / alltoallvc）不在白名单，expand_list 直接报错
    if args.engines.strip() == "all":
        engines = list(PROF_ENGINES)
    else:
        engines = expand_list(args.engines, PROF_ENGINES, "engine")
    data_types = expand_list(args.data_types, DATA_TYPES, "data_type")
    if args.stepbytes is not None and args.stepfactor is not None:
        raise ValueError("--stepbytes and --stepfactor are mutually exclusive")
    if (args.minbytes is None) != (args.maxbytes is None):
        raise ValueError("--minbytes and --maxbytes must be given together")
    if args.minbytes is not None:
        if parse_bytes(args.minbytes) > parse_bytes(args.maxbytes):
            raise ValueError("--minbytes > --maxbytes")
    if args.hostfile and not os.path.isfile(args.hostfile):
        raise ValueError("hostfile not found: {}".format(args.hostfile))
    return op_types, engines, data_types


def format_duration(seconds):
    """秒数 → 人读时长："1h 02m 03s" / "02m 03s" / "12.3s"（<60s 保留 1 位小数）。"""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return "{:.1f}s".format(seconds)
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return "{}h {:02d}m {:02d}s".format(h, m, s)
    return "{:02d}m {:02d}s".format(m, s)


def run_one(cmd, env, timeout=600):
    """执行单条 mpirun 命令，返回 (stdout 文本, stderr 文本, 是否成功)。

    超时（默认 600s）时杀掉整个进程组（POSIX 下 mpirun 起独立进程组，远端
    rank 子进程随之终止；Windows 退回 taskkill 按映像名清理），该轮按失败处理
    并继续后续采集，避免单轮挂死阻塞整个采集流程。
    """
    kwargs = {"env": env, "stdout": subprocess.PIPE, "stderr": subprocess.PIPE,
              "universal_newlines": True}
    if os.name == "posix":
        kwargs["start_new_session"] = True  # 独立进程组，超时可整组清理
    proc = subprocess.Popen(cmd, **kwargs)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return stdout, stderr, proc.returncode == 0
    except subprocess.TimeoutExpired:
        _kill_process_tree(proc)
        logger.warning("[prof_test] WARN run timeout ({}s), killed: {}".format(timeout, " ".join(cmd)))
        return "", "[prof_test] run timeout after {}s: {}".format(timeout, " ".join(cmd)), False


def _kill_process_tree(proc):
    """超时清理：POSIX 杀 mpirun 所在进程组（含远端 rank），Windows 退回 taskkill。"""
    try:
        if os.name == "posix":
            import signal
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        proc.kill()


def tail_lines(text, limit=15):
    """取文本末尾若干非空行（失败轮 WARN 直接携带报错原因，如缺 so / dlopen 失败）。"""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-limit:]


def parse_run_output(stdout, stderr):
    """从 mpirun 输出解析数据行：stdout 优先，无行时回退 stderr。

    部分 mpirun 实现 / 重定向场景下 hccl_test 表格会落到 stderr，
    两个流用同一解析器（列数不匹配的噪声行天然被忽略）。
    """
    rows = parse_stdout(stdout)
    if not rows:
        rows = parse_stdout(stderr)
    return rows


def dump_raw_output(output_path, tag, stdout, stderr):
    """每轮原始输出取证 dump：stdout/stderr 全文追加写 <output>.raw。

    成功轮为 CSV 数值留证据链（被质疑时可对照二进制原文），
    失败/无结果轮供定位报错原因与格式差异。
    """
    raw_path = output_path + ".raw"
    with open(raw_path, "a", encoding="utf-8") as raw:
        raw.write("===== {} =====\n".format(tag))
        raw.write("----- stdout -----\n{}\n".format(stdout))
        raw.write("----- stderr -----\n{}\n".format(stderr))
    return raw_path


def log_collect_plan(ctx, plan):
    """打印采集计划（测试范围总览）：维度、算法、拓扑、迭代数与输出模式。"""
    args = ctx.args
    logger.info("[prof_test] ================= collect plan =================")
    logger.info("[prof_test] op_types   : {}".format(",".join(plan.op_types)))
    logger.info("[prof_test] data_types : {}".format(",".join(plan.data_types)))
    logger.info("[prof_test] engines    : {}".format(",".join(plan.engines)))
    tested_note = "; ".join(algo_to_env(executor, templates) for executor, templates in plan.tested_algos)
    logger.info("[prof_test] algos      : {} ({})".format(len(plan.tested_algos), tested_note))
    # 每算子 × 引擎的有效算法数（未注册组合已剪，值为 0 表示该引擎该算子整组跳过）
    for op_type in plan.op_types:
        counts = ", ".join("{}={}".format(engine, len(plan.valid_by_op_engine[(op_type, engine)]))
                           for engine in plan.engines)
        logger.info("[prof_test] valid algos : {:<14} {}".format(op_type, counts))
    size_desc = "{}..{}".format(args.minbytes, args.maxbytes) if args.minbytes else "default"
    step_desc = ("stepbytes {}".format(args.stepbytes) if args.stepbytes is not None
                 else "stepfactor {}".format(args.stepfactor) if args.stepfactor is not None
                 else "hccl_test default step")
    logger.info("[prof_test] size range : {} ({})".format(size_desc, step_desc))
    logger.info("[prof_test] reduce_op  : {} (reduce ops only: {})".format(
        args.reduce_op, ",".join(op for op in plan.op_types if op in REDUCE_OP_TYPES) or "none"))
    logger.info("[prof_test] topo       : ranks={} npus/server={} servers={} pods={} super_pods={}".format(
        ctx.topo["ranks.ranks"], ctx.topo["ranks.npus_per_server"], ctx.topo["ranks.servers"],
        ctx.topo["ranks.pods"], ctx.topo["ranks.super_pods"]))
    logger.info("[prof_test] topo levels: {} (estimated, lower bound; use --pods/--super-pods on "
                "superpod to keep 2/3-level algos)".format(plan.topo_levels))
    logger.info("[prof_test] mpi        : {} (mpirun command flavor)".format(ctx.mpi_flavor))
    logger.info("[prof_test] iters      : warmup={} iters={}".format(args.warmup_iters, args.iters))
    if ctx.hccl_buffsize:
        logger.info("[prof_test] env        : HCCL_BUFFSIZE={}".format(ctx.hccl_buffsize))
    logger.info("[prof_test] total runs : {} = {} dtypes x {} valid combos "
                "({} unregistered combos pruned)".format(
                    ctx.total_runs, len(plan.data_types), plan.valid_combos,
                    plan.raw_combos - plan.valid_combos))
    out_mode = ("append" if os.path.exists(args.output) and os.path.getsize(args.output) > 0 else "new")
    logger.info("[prof_test] output     : {} ({})".format(args.output, out_mode))
    logger.info("[prof_test] =================================================")


def build_run_combos(op_types, data_types, engines, valid_by_op_engine):
    """展开实际执行的采集组合列表：[(op_type, data_type, engine, executor, templates), ...]。"""
    combos = []
    for op_type, data_type, engine in itertools.product(op_types, data_types, engines):
        for executor, templates in valid_by_op_engine[(op_type, engine)]:
            combos.append((op_type, data_type, engine, executor, templates))
    return combos


def _base_csv_row(ctx, combo):
    """构造单轮 CSV 行的公共字段（成功行覆写性能值，失败行覆写 check_result）。"""
    op_type, data_type, engine, executor, templates = combo
    return {
        "op_type": op_type,
        "size_bytes": "",
        "data_type": data_type,
        "reduce_type": reduce_type_for_op(op_type, ctx.args.reduce_op),
        "engine": engine,
        "algorithm.executor_type": executor,
        "algorithm.template_type": ",".join(templates),
        "HCCL_BUFFSIZE": ctx.hccl_buffsize,
        **ctx.topo,
    }


def _exec_round(ctx, combo, run_idx, elapsed_total, first_run_cost):
    """拉起单轮 mpirun 采集并解析，返回 (rows, stdout, stderr, ok, duration)。

    含本轮配置/cmd/进度日志、-w/-n 回显校验与 ETA 预估。
    first_run_cost：首轮实际耗时（run_idx==1 时为本轮耗时，否则沿用入参）。
    """
    op_type, data_type, engine, executor, templates = combo
    args = ctx.args
    algo_str = algo_to_env(executor, templates)
    # 先打本轮配置再拉起 mpirun（长时间运行期间日志可见）
    logger.info("[prof_test] [{}/{}] op={} dtype={} engine={} algo={}".format(
        run_idx, ctx.total_runs, op_type, data_type, engine, algo_str))
    test_args = build_test_args(ctx.profile, op_type, data_type, engine)
    cmd = build_mpirun_cmd(args.np_total, args.hostfile,
                           [ctx.exe_paths[op_type]] + test_args, env_keys=ctx.env_keys,
                           flavor=ctx.mpi_flavor)
    env = os.environ.copy()
    env["HCCL_ALGO"] = algo_str
    start = time.monotonic()
    stdout, stderr, ok = run_one(cmd, env, timeout=args.run_timeout)
    duration = time.monotonic() - start
    # 完整命令打印（排查 -w/-n 等参数是否真的下发）
    logger.info("[prof_test]     cmd: HCCL_ALGO={} {}".format(algo_str, " ".join(cmd)))
    rows = parse_run_output(stdout, stderr) if ok else []
    # 校验 -w/-n 生效：hccl_test 首行回显实际生效值，与 CLI 配置不符即 WARN
    echo_iters = parse_header_info(stdout) or parse_header_info(stderr)
    if echo_iters is not None and echo_iters != (args.iters, args.warmup_iters):
        logger.warning("[prof_test] WARN warmup/iters mismatch: cli -w {} -n {} but "
                       "binary echoes warmup_iters={} iters={}".format(
                           args.warmup_iters, args.iters, echo_iters[1], echo_iters[0]))
    # ETA 用稳态均耗时外推（剔除首轮建链/so 加载等冷启动开销）；首轮无稳态基准不打
    eta = "--"
    if run_idx > 1:
        steady_avg = (elapsed_total + duration - first_run_cost) / (run_idx - 1)
        eta = format_duration(steady_avg * (ctx.total_runs - run_idx))
    logger.info("[prof_test]     -> {} rows ({}, ETA {})".format(
        len(rows), format_duration(duration), eta))
    return rows, stdout, stderr, ok, duration


def _write_failed_row(ctx, writer, combo, ok, stderr):
    """失败/无结果轮审计：落 check_result=failed/noresult 的 CSV 行并回显原因。

    性能字段留空；ok=False 时 stderr 尾部进日志（缺 so、dlopen 失败等）。
    """
    op_type, data_type, engine, executor, templates = combo
    algo_str = algo_to_env(executor, templates)
    row = _base_csv_row(ctx, combo)
    row["check_result"] = "failed" if not ok else "noresult"
    row["timestamp(ms)"] = int(time.time() * 1000)
    writer.write_row(row)
    logger.warning("[prof_test] WARN {}: op={} dtype={} engine={} algo={} "
                   "(raw output saved to {})".format(
                       "run failed" if not ok else "no parsed rows",
                       op_type, data_type, engine, algo_str, ctx.args.output + ".raw"))
    if not ok:
        for line in tail_lines(stderr):
            logger.warning("[prof_test]     stderr | {}".format(line))


def _run_single(ctx, writer, combo, run_idx, elapsed_total, first_run_cost):
    """执行单轮采集：拉起 mpirun → 解析输出 → 结果落 CSV。

    返回 (解析出的数据行数, 本轮耗时秒)。失败/无结果轮落一行审计记录
    （check_result=failed/noresult）。原始输出 dump 到 <output>.raw 取证。
    """
    op_type, data_type, engine, executor, templates = combo
    algo_str = algo_to_env(executor, templates)
    rows, stdout, stderr, ok, duration = _exec_round(ctx, combo, run_idx, elapsed_total,
                                                     first_run_cost)
    dump_raw_output(
        ctx.args.output, "op={} dtype={} engine={} algo={}".format(
            op_type, data_type, engine, algo_str), stdout, stderr)
    if not rows:
        _write_failed_row(ctx, writer, combo, ok, stderr)
        return 0, duration
    base_row = _base_csv_row(ctx, combo)
    timestamp = int(time.time() * 1000)
    for row in rows:
        writer.write_row({
            **base_row,
            "size_bytes": row.get("size_bytes"),
            "check_result": row.get("check_result", ""),
            "alg_bandwidth(GB/s)": row.get("alg_bandwidth(GB/s)"),
            "alg_latency(us)": row.get("alg_latency(us)"),
            "timestamp(ms)": timestamp,
        })
    return len(rows), duration


def make_collect_plan(op_types, engines, data_types, algos, topo_levels=1):
    """按 (op, engine) 剪掉未注册与层数不满足的算法组合，生成采集计划。

    algos 为 None（默认全量）时直接展开注册表（含多级组合）再按层数剪枝。
    - valid_by_op_engine：{(op, engine): [(executor, templates), ...]}，仅含可采组合
    - tested_algos：各 (op, engine) 有效集的并集（保持原始顺序）
    - total_runs：实际执行轮数（dtype 数 × 有效组合数），全被剪时为 0
    """
    valid_by_op_engine = {}
    for op_type in op_types:
        for engine in engines:
            valid_by_op_engine[(op_type, engine)] = filter_valid_algos(
                op_type, engine, algos, topo_levels)[0]
    valid_combos = sum(len(valid) for valid in valid_by_op_engine.values())
    raw_combos = valid_combos if algos is None else len(op_types) * len(engines) * len(algos)
    # templates 是 list（不可 hash），统一转 tuple 作 key
    valid_algo_keys = set()
    for valid in valid_by_op_engine.values():
        valid_algo_keys.update((executor, tuple(templates)) for executor, templates in valid)
    if algos is None:
        tested_algos, seen = [], set()
        for valid in valid_by_op_engine.values():
            for executor, templates in valid:
                key = (executor, tuple(templates))
                if key not in seen:
                    seen.add(key)
                    tested_algos.append((executor, templates))
    else:
        tested_algos = [algo for algo in algos if (algo[0], tuple(algo[1])) in valid_algo_keys]
    total_runs = len(data_types) * valid_combos
    return CollectPlan(op_types=op_types, data_types=data_types, engines=engines, algos=tested_algos,
                       valid_by_op_engine=valid_by_op_engine, tested_algos=tested_algos,
                       valid_combos=valid_combos, raw_combos=raw_combos, total_runs=total_runs,
                       topo_levels=topo_levels)


def _collect(ctx, writer, combos):
    """循环执行全部采集轮，返回 (runs_ok, runs_fail, total_rows)。

    失败/无结果轮也写了 1 行审计记录，计入 total_rows。
    """
    runs_ok, runs_fail, total_rows = 0, 0, 0
    elapsed_total = 0.0
    first_run_cost = 0.0
    for run_idx, combo in enumerate(combos, 1):
        n_rows, duration = _run_single(ctx, writer, combo, run_idx, elapsed_total,
                                       first_run_cost)
        first_run_cost = duration if run_idx == 1 else first_run_cost
        elapsed_total += duration
        if n_rows:
            runs_ok += 1
        else:
            runs_fail += 1
        total_rows += n_rows or 1
    return runs_ok, runs_fail, total_rows


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    args = parse_args(argv)
    try:
        op_types, engines, data_types = validate_args(args)
        algos = build_algos(args.algos, args.executors, args.templates)
    except ValueError as err:
        logger.error("[prof_test] argument error: {}".format(err))
        return 1
    if algos is not None and not algos:
        logger.error("[prof_test] argument error: empty algo set")
        return 1

    # 估算拓扑层数并剪枝未注册 / 层数不满足的组合
    topo_levels = estimate_topo_levels(args.np_total, args.npus, args.hostfile, args.pods, args.super_pods)
    plan = make_collect_plan(op_types, engines, data_types, algos, topo_levels)
    if plan.total_runs == 0:
        logger.error("[prof_test] argument error: no registered algo combo for op-types [{}] "
                     "under engines [{}], all {} combos pruned".format(
                         ",".join(op_types), ",".join(engines), plan.raw_combos))
        return 1

    try:
        mpi_flavor = preflight(op_types, args.bin_dir)
    except PreflightError as err:
        logger.error("[prof_test] preflight failed: {}".format(err))
        return 1

    topo = derive_topo(args.np_total, args.npus, args.hostfile, args.pods, args.super_pods)
    hccl_buffsize = os.environ.get("HCCL_BUFFSIZE", "")
    # OpenMPI 需 -x 显式传递到远端节点：HCCL_ALGO 每轮必设；HCCL_BUFFSIZE 存在才传
    env_keys = ["HCCL_ALGO"] + (["HCCL_BUFFSIZE"] if hccl_buffsize else [])
    exe_paths = {op_type: exe_path_of(op_type, args.bin_dir).replace("\\", "/")
                 for op_type in op_types}
    ctx = RunContext(args=args, profile=make_test_profile(args), exe_paths=exe_paths,
                     hccl_buffsize=hccl_buffsize, env_keys=env_keys, topo=topo,
                     total_runs=plan.total_runs, mpi_flavor=mpi_flavor)

    # 采集计划（测试范围总览）：先打全量维度再开跑
    log_collect_plan(ctx, plan)
    try:
        writer = CsvWriter(args.output)
    except ValueError as err:
        logger.error("[prof_test] error: {}".format(err))
        return 1
    combos = build_run_combos(op_types, data_types, engines, plan.valid_by_op_engine)
    with writer:
        runs_ok, runs_fail, total_rows = _collect(ctx, writer, combos)

    logger.info("[prof_test] done: runs_ok={} runs_fail={} rows={} output={}".format(
        runs_ok, runs_fail, total_rows, args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
