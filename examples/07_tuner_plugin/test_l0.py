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

"""L0 纯函数单测（本地 Python3，无 NPU）。

覆盖 prof_test.py / optimize_config.py / tuner_common.py 的核心纯函数：
算法串解析、命令构造、MPI 环境探测、stdout 解析、CSV 结构、参数白名单、
preflight 自检、去重 / 分组 / 选优 / 区间合并 / 补缝 / 白名单校验 / conf 结构。

运行：python3 test_l0.py  （或 python3 -m unittest test_l0 -v）
"""

import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import prof_test  # noqa: E402
import optimize_config  # noqa: E402
import tuner_common  # noqa: E402


class TunerCommonTest(unittest.TestCase):
    """C 一致性：公共表单一来源，两脚本引用同一对象；plugin 白名单 ⊆ 采集权威域。"""

    def test_two_scripts_share_common_tables(self):
        # 同一来源（非副本）
        self.assertIs(prof_test.CSV_FIELDS, tuner_common.CSV_FIELDS)
        self.assertIs(prof_test.OP_TYPES, tuner_common.OP_TYPES)
        self.assertIs(prof_test.DEVICE_ENGINES, tuner_common.DEVICE_ENGINES)
        self.assertIs(prof_test.ENGINE_ALL, tuner_common.ENGINE_ALL)
        self.assertIs(prof_test.EXECUTOR_TYPES, tuner_common.EXECUTOR_TYPES)
        self.assertIs(prof_test.TEMPLATE_TYPES, tuner_common.TEMPLATE_TYPES)
        self.assertIs(optimize_config.CSV_FIELDS, tuner_common.CSV_FIELDS)
        self.assertIs(optimize_config.PLUGIN_ENGINES, tuner_common.PLUGIN_ENGINES)
        self.assertIs(optimize_config.PLUGIN_EXECUTORS, tuner_common.PLUGIN_EXECUTORS)
        self.assertIs(optimize_config.PLUGIN_OP_TYPES, tuner_common.PLUGIN_OP_TYPES)
        self.assertIs(optimize_config.PLUGIN_TEMPLATES, tuner_common.PLUGIN_TEMPLATES)
        self.assertIs(optimize_config.DEVICE_ENGINES, tuner_common.DEVICE_ENGINES)

    def test_csv_fields_structure(self):
        self.assertEqual(len(tuner_common.CSV_FIELDS), 17)
        # 拓扑 5 字段 = CSV_FIELDS 的 ranks.* 后段
        self.assertEqual(tuner_common.CSV_FIELDS[-9:-4], tuner_common.RANKS_FIELDS)

    def test_plugin_whitelist_subset_of_collect_domain(self):
        # plugin 白名单是采集权威域的子集
        self.assertTrue(tuner_common.PLUGIN_EXECUTORS <= set(tuner_common.EXECUTOR_TYPES))
        self.assertTrue(tuner_common.PLUGIN_TEMPLATES <= set(tuner_common.TEMPLATE_TYPES))
        self.assertEqual(tuner_common.PLUGIN_OP_TYPES, set(tuner_common.OP_TYPES))
        self.assertEqual(tuner_common.PLUGIN_ENGINES, set(tuner_common.ENGINE_ALL))
        # 具体差集：executor 差 strictordered；template 白名单 = 权威域全量
        # （plugin.cpp 不做 template 枚举校验，13 种均对应已注册算法）
        self.assertEqual(set(tuner_common.EXECUTOR_TYPES) - tuner_common.PLUGIN_EXECUTORS,
                         {"strictordered"})
        self.assertEqual(tuner_common.PLUGIN_TEMPLATES, set(tuner_common.TEMPLATE_TYPES))
        # 权威枚举规模
        self.assertEqual(len(tuner_common.EXECUTOR_TYPES), 6)
        self.assertEqual(len(tuner_common.TEMPLATE_TYPES), 13)
        self.assertEqual(len(tuner_common.PLUGIN_EXECUTORS), 5)
        self.assertEqual(len(tuner_common.PLUGIN_TEMPLATES), 13)
        self.assertEqual(len(tuner_common.ENGINE_ALL), 5)


# kwargs 关键字名不能带括号/斜杠（如 alg_bandwidth(GB/s)），调用点用旧简名，
# 由 make_row 归一成 CSV 真实列名（带单位）。
ROW_KEY_ALIAS = {
    "alg_bandwidth": "alg_bandwidth(GB/s)",
    "alg_latency": "alg_latency(us)",
    "timestamp": "timestamp(ms)",
}


def make_row(**overrides):
    """构造一条标准采集行（默认 allreduce / fp32 / aicpu / sole{mesh} / 8 rank 单机拓扑）。"""
    row = {
        "op_type": "allreduce", "size_bytes": "8192", "data_type": "fp32", "reduce_type": "sum",
        "engine": "aicpu", "algorithm.executor_type": "sole", "algorithm.template_type": "mesh",
        "HCCL_BUFFSIZE": "2048", "ranks.ranks": "8", "ranks.npus_per_server": "8",
        "ranks.servers": "1", "ranks.pods": "1", "ranks.super_pods": "1",
        "check_result": "success", "alg_bandwidth(GB/s)": "10.0", "alg_latency(us)": "100.0",
        "timestamp(ms)": "1000000000000",
    }
    row.update((ROW_KEY_ALIAS.get(key, key), value) for key, value in overrides.items())
    return row


class ProfParseBytesTest(unittest.TestCase):
    """P01 parse_bytes 后缀解析。"""

    def test_parse_bytes(self):
        self.assertEqual(prof_test.parse_bytes("1K"), 1024)
        self.assertEqual(prof_test.parse_bytes("4M"), 4194304)
        self.assertEqual(prof_test.parse_bytes("2G"), 2147483648)
        self.assertEqual(prof_test.parse_bytes("512"), 512)
        self.assertEqual(prof_test.parse_bytes("1.5K"), 1536)
        self.assertEqual(prof_test.parse_bytes("1k"), 1024)
        with self.assertRaises(ValueError):
            prof_test.parse_bytes("abc")
        with self.assertRaises(ValueError):
            prof_test.parse_bytes("1T")


class ProfParseAlgosTest(unittest.TestCase):
    """parse_algos 算法串解析。"""

    def test_full_form(self):
        self.assertEqual(prof_test.parse_algos("sole{mesh}"), [("sole", ["mesh"])])

    def test_shorthand_defaults_sole(self):
        self.assertEqual(prof_test.parse_algos("mesh"), [("sole", ["mesh"])])

    def test_multi_level_comma_sequence(self):
        self.assertEqual(prof_test.parse_algos("parallel{mesh,nhr}"), [("parallel", ["mesh", "nhr"])])

    def test_multi_token_and_empty_segment(self):
        algos = prof_test.parse_algos("sequence{mesh,nhr,nhr};sole{mesh2die};;")
        self.assertEqual(algos, [("sequence", ["mesh", "nhr", "nhr"]), ("sole", ["mesh2die"])])

    def test_empty_string(self):
        self.assertEqual(prof_test.parse_algos(""), [])

    def test_invalid_executor_or_template(self):
        with self.assertRaises(ValueError):
            prof_test.parse_algos("badexec{mesh}")
        with self.assertRaises(ValueError):
            prof_test.parse_algos("sole{badtpl}")

    def test_level_syntax_rejected(self):
        with self.assertRaises(ValueError):
            prof_test.parse_algos("sequence{level1=nhr,level0=mesh}")


class ProfAlgoToEnvTest(unittest.TestCase):
    """P03 algo_to_env 往返。"""

    def test_roundtrip(self):
        self.assertEqual(prof_test.algo_to_env("sole", ["mesh"]), "sole{mesh}")
        self.assertEqual(prof_test.algo_to_env("parallel", ["mesh", "nhr"]), "parallel{mesh,nhr}")
        # 往返：parse_algos(algo_to_env(x)) == x
        for algos in [("sole", ["mesh"]), ("parallel", ["mesh", "nhr"])]:
            parsed = prof_test.parse_algos(prof_test.algo_to_env(*algos))
            self.assertEqual(parsed, [algos])


class ProfBuildAlgosTest(unittest.TestCase):
    """P04 build_algos 双入口。"""

    def test_algos_arg_passthrough(self):
        self.assertEqual(prof_test.build_algos("sole{mesh};parallel{mesh,nhr}", None, None),
                         [("sole", ["mesh"]), ("parallel", ["mesh", "nhr"])])

    def test_executors_templates_cartesian(self):
        algos = prof_test.build_algos(None, "sole,parallel", "mesh,nhr")
        self.assertEqual(algos, [("sole", ["mesh"]), ("sole", ["nhr"]),
                                 ("parallel", ["mesh"]), ("parallel", ["nhr"])])

    def test_default_full_enum(self):
        # 三参数均未指定 → None 哨兵（采集计划直接展开注册表全量组合，含多级）
        self.assertIsNone(prof_test.build_algos(None, None, None))


class ProfEstimateTopoLevelsTest(unittest.TestCase):
    """estimate_topo_levels：按 CLI 参数估算拓扑层数下界。"""

    def test_single_server_is_one_level(self):
        self.assertEqual(prof_test.estimate_topo_levels(8, 8, None, 0, 0), 1)

    def test_pods_is_two_level(self):
        self.assertEqual(prof_test.estimate_topo_levels(64, 8, None, 4, 0), 2)

    def test_multi_superpods_is_three_level(self):
        self.assertEqual(prof_test.estimate_topo_levels(512, 8, None, 8, 2), 3)

    def test_np_division_without_hostfile(self):
        # 无 hostfile 时按 np_total/npus 折算 server 数（16 卡/8 卡 = 2 server → 2 层）
        self.assertEqual(prof_test.estimate_topo_levels(16, 8, None, 0, 0), 2)


class ProfFilterValidAlgosTest(unittest.TestCase):
    """P04+ filter_valid_algos：按 (op, engine) 剪掉未注册组合（OP_VALID_ALGOS）。"""

    def test_allreduce_prunes_unregistered_aicpu_sole_mesh(self):
        algos = [("sole", ["mesh"]), ("sole", ["meshoneshot"])]
        valid, skipped = prof_test.filter_valid_algos("allreduce", "aicpu", algos)
        self.assertEqual(valid, [("sole", ["meshoneshot"])])
        self.assertEqual(skipped, [("sole", ["mesh"])])
        # 同一组合换 ccusched 引擎即已注册
        valid, skipped = prof_test.filter_valid_algos("allreduce", "ccusched", algos)
        self.assertEqual(valid, [("sole", ["mesh"])])
        self.assertEqual(skipped, [("sole", ["meshoneshot"])])

    def test_alltoall_ccums_has_no_registered_algo(self):
        algos = [("sole", ["mesh"]), ("sequence", ["mesh"])]
        valid, skipped = prof_test.filter_valid_algos("alltoall", "ccums", algos)
        self.assertEqual(valid, [])
        self.assertEqual(len(skipped), 2)

    def test_multi_template_combo_keyed_by_full_tuple(self):
        valid, _ = prof_test.filter_valid_algos(
            "allreduce", "ccusched", [("parallel", ["mesh", "nhr"]), ("parallel", ["mesh"])])
        # 模板序列参与匹配：parallel{mesh,nhr} 已注册，parallel{mesh} 未注册
        self.assertEqual(valid, [("parallel", ["mesh", "nhr"])])

    def test_unknown_op_not_pruned(self):
        algos = [("sole", ["mesh"])]
        valid, skipped = prof_test.filter_valid_algos("send", "aicpu", algos)
        self.assertEqual(valid, algos)
        self.assertEqual(skipped, [])

    def test_none_algos_expands_registered_full_single_level(self):
        # 默认全量（algos=None）+ 单层估算：9 个注册组合先剪黑名单
        # （sole{nhrmultilink}/concur{mesh,nhr}），再剪 2/3 层专属
        # sequence{meshconcur,*}/parallel{mesh,nhr}/parallel{nhr,nhr}；
        # 单层专属 sole{meshconcur}（1,1）保留，剩 3
        valid, skipped = prof_test.filter_valid_algos("allgather", "aicpu", None)
        self.assertEqual(skipped, [])
        self.assertEqual(len(valid), 3)
        self.assertIn(("sole", ["mesh"]), valid)
        self.assertIn(("sole", ["meshconcur"]), valid)
        self.assertIn(("sole", ["nhr"]), valid)
        self.assertNotIn(("sole", ["nhrmultilink"]), valid)
        self.assertNotIn(("sequence", ["meshconcur", "nhr"]), valid)
        self.assertNotIn(("parallel", ["mesh", "nhr"]), valid)

    def test_none_algos_two_level_keeps_two_level_algos(self):
        # 2 层估算：sequence{meshconcur,nhr} 与 parallel{mesh,nhr} 保留；
        # 黑名单（sole{nhrmultilink}/concur{mesh,nhr}）无论层数均被剪；
        # 3 层专属（sequence{meshconcur,nhr,nhr}、parallel{nhr,nhr}）与
        # 单层专属 sole{meshconcur}/sole{mesh}（max=1）剪掉
        valid, _ = prof_test.filter_valid_algos("allgather", "aicpu", None, topo_levels=2)
        self.assertIn(("sequence", ["meshconcur", "nhr"]), valid)
        self.assertIn(("parallel", ["mesh", "nhr"]), valid)
        self.assertIn(("sole", ["nhr"]), valid)
        self.assertNotIn(("sequence", ["meshconcur", "nhr", "nhr"]), valid)
        self.assertNotIn(("parallel", ["nhr", "nhr"]), valid)
        self.assertNotIn(("sole", ["meshconcur"]), valid)
        self.assertNotIn(("sole", ["mesh"]), valid)
        self.assertNotIn(("sole", ["nhrmultilink"]), valid)
        self.assertNotIn(("concur", ["mesh", "nhr"]), valid)
        self.assertEqual(len(valid), 3)

    def test_none_algos_three_level_keeps_three_level_only(self):
        # 3 层估算：仅 3 层专属（sequence{meshconcur,nhr,nhr}、parallel{nhr,nhr}）
        # 与默认 (1,3) 的 sole{nhr} 保留；2 层专属（max=2）与单层专属
        # sole{meshconcur}/sole{mesh}（max=1）、黑名单均剪掉
        valid, _ = prof_test.filter_valid_algos("allgather", "aicpu", None, topo_levels=3)
        self.assertIn(("sequence", ["meshconcur", "nhr", "nhr"]), valid)
        self.assertIn(("parallel", ["nhr", "nhr"]), valid)
        self.assertIn(("sole", ["nhr"]), valid)
        self.assertNotIn(("sequence", ["meshconcur", "nhr"]), valid)
        self.assertNotIn(("parallel", ["mesh", "nhr"]), valid)
        self.assertNotIn(("sole", ["meshconcur"]), valid)
        self.assertNotIn(("sole", ["mesh"]), valid)
        self.assertNotIn(("sole", ["nhrmultilink"]), valid)
        self.assertNotIn(("concur", ["mesh", "nhr"]), valid)
        self.assertEqual(len(valid), 3)

    def test_none_algos_plan_topo_aware(self):
        # 采集计划：2 层估算下 allgather+aicpu 3 个有效组合（黑名单+多级剪枝后）
        plan = prof_test.make_collect_plan(["allgather"], ["aicpu"], ["int32"], None, 2)
        self.assertEqual(plan.raw_combos, plan.valid_combos)
        self.assertEqual(plan.valid_combos, 3)
        self.assertEqual(plan.total_runs, 3)
        self.assertEqual(plan.topo_levels, 2)


class ProfBlacklistTest(unittest.TestCase):
    """P04+ 采集黑名单：concur 执行器 / multilink、meshconcurrent 模板不采（tuner_common.is_blacklisted）。"""

    def test_is_blacklisted_keywords(self):
        # executor 命中
        self.assertTrue(prof_test.is_blacklisted("concur", ["mesh", "nhr"]))
        # 任一模板含 multilink 关键字（含多级组合中任一级）
        self.assertTrue(prof_test.is_blacklisted("sole", ["nhrmultilink"]))
        self.assertTrue(prof_test.is_blacklisted("sole", ["meshmultilink"]))
        self.assertTrue(prof_test.is_blacklisted("parallel", ["mesh", "nhrmultilink"]))
        # meshconcurrent 注册为 TopoMatchUBX（meshclos 方阵组网专属），executor 仍是 sole，
        # 普通组网必回退，进黑名单
        self.assertTrue(prof_test.is_blacklisted("sole", ["meshconcurrent"]))
        # meshconcur 模板不在黑名单（由层数门控负责），concur 关键字不误伤
        self.assertFalse(prof_test.is_blacklisted("sole", ["meshconcur"]))
        self.assertFalse(prof_test.is_blacklisted("sequence", ["meshconcur", "nhr"]))
        # 正常组合不命中
        self.assertFalse(prof_test.is_blacklisted("sole", ["mesh"]))
        self.assertFalse(prof_test.is_blacklisted("sequence", ["mesh", "nhr"]))

    def test_explicit_algos_blacklist_skipped(self):
        # 显式指定黑名单组合：已注册但被拉黑 → 落 skipped 不采集
        algos = [("sole", ["mesh"]), ("sole", ["nhrmultilink"]), ("concur", ["mesh", "nhr"])]
        valid, skipped = prof_test.filter_valid_algos("allgather", "aicpu", algos)
        self.assertEqual(valid, [("sole", ["mesh"])])
        self.assertEqual(skipped, [("sole", ["nhrmultilink"]), ("concur", ["mesh", "nhr"])])

    def test_per_op_constraints_differ_for_same_signature(self):
        # 同签名组合不同算子约束不同：ccusched parallel{mesh,nhr} 在
        # allgather/reduce_scatter 是 min=max=2（1 层剪掉），allreduce 仅 max=2（1 层保留）
        self.assertNotIn(("parallel", ["mesh", "nhr"]),
                         prof_test.filter_valid_algos("allgather", "ccusched", None)[0])
        self.assertNotIn(("parallel", ["mesh", "nhr"]),
                         prof_test.filter_valid_algos("reduce_scatter", "ccusched", None)[0])
        valid = prof_test.filter_valid_algos("allreduce", "ccusched", None)[0]
        self.assertIn(("parallel", ["mesh", "nhr"]), valid)

    def test_aiv_sole_mesh_capped_at_two_levels(self):
        # aiv 的 sole{mesh} 家族注册 maxTopoLevelNum=2：2 层保留、3 层剪掉
        # （allreduce 为 oneshot/twoshot 模板，alltoall/v 未声明层数走默认不剪）
        for op_type, tpl in (("allgather", "mesh"), ("broadcast", "mesh"), ("reduce", "mesh"),
                             ("reduce_scatter", "mesh"), ("scatter", "mesh"),
                             ("allreduce", "meshoneshot"), ("allreduce", "meshtwoshot")):
            valid, _ = prof_test.filter_valid_algos(op_type, "aiv", None, topo_levels=2)
            self.assertIn(("sole", [tpl]), valid, op_type)
            valid, _ = prof_test.filter_valid_algos(op_type, "aiv", None, topo_levels=3)
            self.assertNotIn(("sole", [tpl]), valid, op_type)
        valid, _ = prof_test.filter_valid_algos("alltoall", "aiv", None, topo_levels=3)
        self.assertIn(("sole", ["mesh"]), valid)

    def test_dpu_sequence_min_level_two(self):
        # dpu sequence{mesh,nhr} 注册 minTopoLevelNum=2：allreduce/allgather 单层剪掉；
        # reduce/broadcast/scatter 的同名注册未声明层数走默认（单层保留）
        for op_type in ("allreduce", "allgather"):
            valid, _ = prof_test.filter_valid_algos(op_type, "dpu", None)
            self.assertNotIn(("sequence", ["mesh", "nhr"]), valid, op_type)
            valid, _ = prof_test.filter_valid_algos(op_type, "dpu", None, topo_levels=2)
            self.assertIn(("sequence", ["mesh", "nhr"]), valid, op_type)
        for op_type in ("reduce", "broadcast", "scatter"):
            valid, _ = prof_test.filter_valid_algos(op_type, "dpu", None)
            self.assertIn(("sequence", ["mesh", "nhr"]), valid, op_type)

    def test_explicit_algo_respects_level_gate(self):
        # 显式指定 2 层专属算法在单层估算下被层数门控剪掉（而非注册表剪掉）
        valid, skipped = prof_test.filter_valid_algos(
            "allgather", "aicpu", [("sequence", ["meshconcur", "nhr"])])
        self.assertEqual((valid, skipped), ([], [("sequence", ["meshconcur", "nhr"])]))
        valid, skipped = prof_test.filter_valid_algos(
            "allgather", "aicpu", [("sequence", ["meshconcur", "nhr"])], topo_levels=2)
        self.assertEqual((valid, skipped), ([("sequence", ["meshconcur", "nhr"])], []))

    def test_sole_mesh_pruned_on_multi_level(self):
        # 回归：sole{mesh} 注册 max=1，多级拓扑下显式指定也要剪掉，
        # 否则 HCCL_ALGO 匹配落空静默回退，测出的数据张冠李戴
        for op_type in ("allgather", "reduce", "reduce_scatter"):
            valid, skipped = prof_test.filter_valid_algos(
                op_type, "aicpu", [("sole", ["mesh"])], topo_levels=2)
            self.assertEqual((valid, skipped), ([], [("sole", ["mesh"])]), op_type)
        # scatter 的 sole{mesh} 注册 max=3，2 层保留
        valid, skipped = prof_test.filter_valid_algos(
            "scatter", "aicpu", [("sole", ["mesh"])], topo_levels=2)
        self.assertEqual((valid, skipped), ([("sole", ["mesh"])], []))

    def test_table_entries_within_enum_domain(self):
        # 表内容必须落在权威枚举域内，防止手抄错拼
        for op_type, valid_set in tuner_common.OP_VALID_ALGOS.items():
            self.assertIn(op_type, tuner_common.OP_TYPES)
            for engine, executor, templates in valid_set:
                self.assertIn(engine, tuner_common.ENGINE_ALL)
                self.assertIn(executor, tuner_common.EXECUTOR_TYPES)
                for template in templates:
                    self.assertIn(template, tuner_common.TEMPLATE_TYPES)


class ProfBuildMpirunCmdTest(unittest.TestCase):
    """P05 build_mpirun_cmd 命令构造与拓扑派生。"""

    def test_cmd_and_topo(self):
        with tempfile.TemporaryDirectory() as tmp:
            hostfile = os.path.join(tmp, "hosts")
            with open(hostfile, "w", encoding="utf-8") as handle:
                handle.write("10.10.130.22:8\n10.10.130.21:8\n")
            cmd = prof_test.build_mpirun_cmd(16, hostfile, ["./bin/all_reduce_test", "-p", "8"],
                                             env_keys=["HCCL_ALGO"], flavor="mpich")
            self.assertIn("mpirun", cmd[0])
            self.assertIn("-n", cmd)
            self.assertEqual(cmd[cmd.index("-n") + 1], "16")
            self.assertIn("-f", cmd)  # MPICH hostfile 旗标
            self.assertEqual(cmd[cmd.index("-f") + 1], hostfile)
            self.assertIn("./bin/all_reduce_test", cmd)
            topo = prof_test.derive_topo(16, 8, hostfile, 1, 1)
            self.assertEqual(topo["ranks.ranks"], 16)
            self.assertEqual(topo["ranks.npus_per_server"], 8)
            self.assertEqual(topo["ranks.servers"], 2)

    def test_openmpi_hostfile_and_env(self):
        cmd = prof_test.build_mpirun_cmd(8, "hostfile", ["./bin/reduce_test"],
                                         env_keys=["HCCL_ALGO"], flavor="openmpi")
        self.assertIn("-hostfile", cmd)
        self.assertIn("-x", cmd)
        self.assertEqual(cmd[cmd.index("-x") + 1], "HCCL_ALGO")

    def test_single_machine_no_hostfile(self):
        cmd = prof_test.build_mpirun_cmd(8, None, ["./bin/all_reduce_test"], flavor="mpich")
        self.assertNotIn("-f", cmd)
        self.assertEqual(prof_test.derive_topo(8, 8, None, 1, 1)["ranks.servers"], 1)


class ProfDetectMpiEnvTest(unittest.TestCase):
    """detect_mpi_env / check_mpi_env MPI 环境探测。"""

    def test_no_mpi_env(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(prof_test.detect_mpi_env(), (0, 1))

    def test_ompi_env(self):
        env = {"OMPI_COMM_WORLD_RANK": "2", "OMPI_COMM_WORLD_SIZE": "8"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(prof_test.detect_mpi_env(), (2, 8))
            self.assertEqual(prof_test.detect_mpi_flavor(), "openmpi")

    def test_default_flavor_mpich(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(prof_test.detect_mpi_flavor(), "mpich")

    def test_check_mpi_env_outside_session(self):
        # 会话外：返回 flavor，不报错
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(prof_test.check_mpi_env(), "mpich")
        env = {"OMPI_COMM_WORLD_RANK": "0", "OMPI_COMM_WORLD_SIZE": "1"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(prof_test.check_mpi_env(), "openmpi")

    def test_check_mpi_env_nested_rejected(self):
        # 已在 MPI 会话内（size>1）：拦截嵌套运行
        env = {"OMPI_COMM_WORLD_RANK": "2", "OMPI_COMM_WORLD_SIZE": "8"}
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(prof_test.PreflightError):
                prof_test.check_mpi_env()
        env = {"PMI_RANK": "3", "PMI_SIZE": "16"}
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(prof_test.PreflightError):
                prof_test.check_mpi_env()


class ProfParseStdoutTest(unittest.TestCase):
    """P07 parse_stdout 解析 hccl_test 表格。"""

    STDOUT = (
        "the minbytes is 8192, maxbytes is 67108864, iters is 20, warmup_iters is 5\n"
        "data_size(Bytes): |   avg_time(us): | alg_bandwidth(GB/s): | check_result:\n"
        "8192              |     764.55    |       0.00998        | success\n"
        "16384             |     858.80    |       0.01777        | success\n"
        "\n"
        "32768             |     901.10    |       0.03387        | failed\n"
        "some warning line without pipes\n"
        "65536             |     900.00    |       0.06782        | NULL\n"
    )

    def test_parse_rows(self):
        rows = prof_test.parse_stdout(self.STDOUT)
        self.assertEqual(len(rows), 4)
        first = rows[0]
        self.assertEqual(first["size_bytes"], 8192)
        self.assertEqual(first["alg_latency(us)"], 764.55)
        self.assertEqual(first["alg_bandwidth(GB/s)"], 0.00998)
        self.assertEqual(first["check_result"], "success")
        self.assertEqual(rows[2]["check_result"], "failed")
        self.assertEqual(rows[3]["check_result"], "NULL")
        self.assertEqual(rows[3]["size_bytes"], 65536)

    def test_column_order_tolerant(self):
        # 列序变化（check_result 提前）仍按列名匹配
        stdout = (
            "check_result: | data_size(Bytes): | avg_time(us): | alg_bandwidth(GB/s):\n"
            "success      | 1024             | 10.5          | 0.5\n"
        )
        rows = prof_test.parse_stdout(stdout)
        self.assertEqual(rows[0], {"size_bytes": 1024, "alg_latency(us)": 10.5,
                                   "alg_bandwidth(GB/s)": 0.5, "check_result": "success"})

    def test_real_binary_header_aveg_time(self):
        # 真实二进制表头是 "aveg_time(us):"（源码拼写笔误，非文档的 avg_time）
        stdout = (
            "the minbytes is 8192, maxbytes is 67108864, iters is 20, warmup_iters is 5\n"
            "data_size(Bytes): |   aveg_time(us): | alg_bandwidth(GB/s): | check_result:\n"
            "8192              |     764.55    |       0.00998        | success\n"
        )
        rows = prof_test.parse_stdout(stdout)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["alg_latency(us)"], 764.55)


class ProfParseHeaderInfoTest(unittest.TestCase):
    """parse_header_info：校验 -w/-n 是否下发生效（解析 hccl_test 首行回显）。"""

    def test_first_line_echo(self):
        stdout = ("the minbytes is 8192, maxbytes is 67108864, iters is 20, warmup_iters is 10\n"
                  "data_size(Bytes): | avg_time(us): | alg_bandwidth(GB/s): | check_result:\n")
        self.assertEqual(prof_test.parse_header_info(stdout), (20, 10))

    def test_real_binary_aveg_echo(self):
        # 真实回显顺序：iters 在前，warmup_iters 在后
        stdout = "the minbytes is 8, maxbytes is 67108864, iters is 20, warmup_iters is 5\n"
        self.assertEqual(prof_test.parse_header_info(stdout), (20, 5))

    def test_no_echo_returns_none(self):
        self.assertIsNone(prof_test.parse_header_info("no summary line here"))
        self.assertIsNone(prof_test.parse_header_info(""))

    def test_searches_all_lines(self):
        # 回显不在首行（前面有 mpirun banner）也能找到
        stdout = "mpirun banner...\n\nthe minbytes is 8, iters is 7, warmup_iters is 3\ntable..."
        self.assertEqual(prof_test.parse_header_info(stdout), (7, 3))


class ProfParseRunOutputTest(unittest.TestCase):
    """parse_run_output：stdout 优先、空时回退 stderr（表格可能落到 stderr）。"""
    def test_stdout_preferred(self):
        stdout = ("data_size(Bytes): | avg_time(us): | alg_bandwidth(GB/s): | check_result:\n"
                  "8192              | 10.5          | 0.5                | success\n")
        stderr = "some mpirun noise"
        rows = prof_test.parse_run_output(stdout, stderr)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["alg_latency(us)"], 10.5)

    def test_fallback_to_stderr_when_stdout_empty(self):
        stdout = "warmup and no table here"
        stderr = ("data_size(Bytes): | avg_time(us): | alg_bandwidth(GB/s): | check_result:\n"
                  "8192              | 10.5          | 0.5                | success\n")
        rows = prof_test.parse_run_output(stdout, stderr)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["size_bytes"], 8192)

    def test_both_empty_no_rows(self):
        self.assertEqual(prof_test.parse_run_output("", "noise only"), [])


class ProfDumpRawOutputTest(unittest.TestCase):
    """dump_raw_output：解析失败轮取证 dump。"""

    def test_dump_appends_with_tag_and_streams(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "out.csv")
            path1 = prof_test.dump_raw_output(csv_path, "op=allreduce engine=aicpu", "raw-out", "raw-err")
            path2 = prof_test.dump_raw_output(csv_path, "op=broadcast engine=aicpu", "out2", "err2")
            self.assertEqual(path1, path2)
            self.assertEqual(path1, csv_path + ".raw")
            with open(path1, encoding="utf-8") as raw:
                content = raw.read()
            self.assertIn("===== op=allreduce engine=aicpu =====", content)
            self.assertIn("----- stdout -----\nraw-out", content)
            self.assertIn("----- stderr -----\nraw-err", content)
            self.assertIn("===== op=broadcast engine=aicpu =====", content)
            # 追加写：两轮内容共存
            self.assertEqual(content.count("====="), 4)


class ProfCsvStructureTest(unittest.TestCase):
    """P08 CSV 结构与追加写。"""

    def test_header_and_append(self):
        import csv as csv_mod
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.csv")
            with prof_test.CsvWriter(path) as writer:
                writer.write_row(make_row())
                writer.write_row(make_row(size_bytes="16384"))
            with open(path, newline="", encoding="utf-8") as handle:
                reader = csv_mod.reader(handle)
                rows = list(reader)
            self.assertEqual(rows[0], list(prof_test.CSV_FIELDS))
            self.assertEqual(len(rows[0]), 17)
            self.assertIn("timestamp(ms)", rows[0])
            self.assertIn("check_result", rows[0])
            self.assertIn("reduce_type", rows[0])
            self.assertIn("HCCL_BUFFSIZE", rows[0])
            self.assertEqual(len(rows), 3)  # header + 2 行，字段数与 header 对齐
            self.assertTrue(all(len(r) == 17 for r in rows))
            # 二次 run 追加同一文件：表头不重写，旧行保留、新行续写
            with prof_test.CsvWriter(path) as writer:
                writer.write_row(make_row(size_bytes="32768"))
            with open(path, newline="", encoding="utf-8") as handle:
                rows2 = list(csv_mod.reader(handle))
            self.assertEqual(len(rows2), 4)
            self.assertEqual(rows2[0], list(prof_test.CSV_FIELDS))  # 表头仍只有一份
            self.assertEqual(rows2[1][1], "8192")   # 旧行保留
            self.assertEqual(rows2[3][1], "32768")  # 新行在尾部

    def test_append_rejects_mismatched_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.csv")
            with open(path, "w", newline="", encoding="utf-8") as handle:
                handle.write("op_type,size_bytes\n")  # 异构表头
            with self.assertRaises(ValueError):
                prof_test.CsvWriter(path)
            # 构造失败后不得追加任何数据
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "op_type,size_bytes\n")

    def test_append_completes_missing_trailing_newline(self):
        # 末字节非换行（上一轮被强杀等）→ 追加前补 \n，新行不与残行粘连
        import csv as csv_mod
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.csv")
            with prof_test.CsvWriter(path) as writer:
                writer.write_row(make_row())
            with open(path, "r+b") as handle:  # 砍掉末尾换行，模拟半行落盘
                handle.truncate(os.path.getsize(path) - 1)
            with prof_test.CsvWriter(path) as writer:
                writer.write_row(make_row(size_bytes="32768"))
            with open(path, newline="", encoding="utf-8") as handle:
                rows = list(csv_mod.reader(handle))
            # header + 残缺旧行 + 新行；新行独立成行不粘连（残行少一列，被 reader 原样收）
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0], list(prof_test.CSV_FIELDS))
            self.assertTrue(rows[2][1] == "32768" and len(rows[2]) == 17)


class ProfValidateArgsTest(unittest.TestCase):
    """P09 参数白名单校验。"""

    def test_v_series_op_rejected(self):
        with self.assertRaises(ValueError):
            prof_test.validate_args(self._args(op_types="allreduce,allgatherv"))

    def test_fp8_collectable(self):
        op_types, engines, data_types = prof_test.validate_args(
            self._args(data_types="fp32,fp8e5m2"))
        self.assertEqual(data_types, ["fp32", "fp8e5m2"])

    def test_engines_all_expands_supported(self):
        # all 展开为全部支持引擎（aicpu/aiv/dpu）
        _, engines, _ = prof_test.validate_args(self._args(engines="all"))
        self.assertEqual(engines, ["aicpu", "aiv", "dpu"])

    def test_other_engines_rejected(self):
        # ccums / ccusched 暂不放开（hccl 无 ccu_*_only 模式，ccu 算法可能回退
        # aicpu 导致耗时不准），及混合含 ccums 均被 CLI 拦截
        for bad in ("ccums", "ccusched", "aicpu,ccums"):
            with self.assertRaises(ValueError):
                prof_test.validate_args(self._args(engines=bad))

    def test_aiv_dpu_engines_accepted(self):
        # aiv / dpu 已放开：单传或混合均通过
        _, engines, _ = prof_test.validate_args(self._args(engines="aiv,dpu"))
        self.assertEqual(engines, ["aiv", "dpu"])

    def _args(self, **overrides):
        base = {"op_types": "allreduce", "engines": "aicpu", "data_types": "fp32",
                "minbytes": None, "maxbytes": None, "stepbytes": None, "stepfactor": None,
                "hostfile": None}
        base.update(overrides)
        return mock.Mock(**base)


class ProfEngineMapTest(unittest.TestCase):
    """P10 engine→-a 映射。"""

    def test_mapping(self):
        self.assertEqual(prof_test.engine_to_accelerator("aicpu"), "aicpu_ts")
        self.assertEqual(prof_test.engine_to_accelerator("aiv"), "aiv_only")
        self.assertEqual(prof_test.engine_to_accelerator("ccums"), "ccu_ms")
        self.assertEqual(prof_test.engine_to_accelerator("ccusched"), "ccu_sched")
        self.assertIsNone(prof_test.engine_to_accelerator("dpu"))  # dpu 不走 -a
        with self.assertRaises(ValueError):
            prof_test.engine_to_accelerator("default")

    def test_test_args_accel_flag(self):
        profile = prof_test.TestProfile(npus=8, minbytes="1K", maxbytes="1M", stepbytes=None,
                                        stepfactor=2, reduce_op="sum", warmup_iters=10, iters=20)
        args = prof_test.build_test_args(profile, "allreduce", "fp32", "aiv")
        self.assertEqual(args[args.index("-a") + 1], "aiv_only")
        dpu_args = prof_test.build_test_args(profile, "allreduce", "fp32", "dpu")
        self.assertNotIn("-a", dpu_args)


class ProfSizeAndReduceTest(unittest.TestCase):
    """size 区间参数与 reduce_type。"""

    def test_interval_maps_to_b_e_f(self):
        profile = prof_test.TestProfile(npus=8, minbytes="1K", maxbytes="1M", stepbytes=None,
                                        stepfactor=2, reduce_op="sum", warmup_iters=10, iters=20)
        args = prof_test.build_test_args(profile, "allreduce", "fp32", "aicpu")
        self.assertEqual(args[args.index("-b") + 1], "1K")
        self.assertEqual(args[args.index("-e") + 1], "1M")
        self.assertEqual(args[args.index("-f") + 1], "2")
        self.assertNotIn("-i", args)

    def test_stepbytes_maps_to_i(self):
        profile = prof_test.TestProfile(npus=8, minbytes="1K", maxbytes="1M", stepbytes=1024,
                                        stepfactor=None, reduce_op="sum", warmup_iters=10, iters=20)
        args = prof_test.build_test_args(profile, "allreduce", "fp32", "aicpu")
        self.assertEqual(args[args.index("-i") + 1], "1024")
        self.assertNotIn("-f", args)

    def test_step_args_mutually_exclusive(self):
        argv = ["--op-types", "allreduce", "--np-total", "8",
                "--minbytes", "1K", "--maxbytes", "1M",
                "--stepbytes", "1024", "--stepfactor", "2"]
        args = prof_test.parse_args(argv)
        with self.assertRaises(ValueError):
            prof_test.validate_args(args)

    def test_reduce_type_field(self):
        # reduce 类算子：传 -o 且 CSV reduce_type=reduce_op
        profile = prof_test.TestProfile(npus=None, minbytes=None, maxbytes=None, stepbytes=None,
                                        stepfactor=None, reduce_op="max", warmup_iters=10, iters=20)
        args = prof_test.build_test_args(profile, "reduce", "fp32", "aicpu")
        self.assertEqual(args[args.index("-o") + 1], "max")
        self.assertEqual(prof_test.reduce_type_for_op("reduce", "max"), "max")
        self.assertEqual(prof_test.reduce_type_for_op("allreduce", "sum"), "sum")
        # 非 reduce 类：不下发 -o，CSV reduce_type=NA
        args = prof_test.build_test_args(profile, "broadcast", "fp32", "aicpu")
        self.assertNotIn("-o", args)
        self.assertEqual(prof_test.reduce_type_for_op("broadcast", "sum"), "NA")


class ProfFormatDurationTest(unittest.TestCase):
    """P13 耗时格式化与 ETA 口径。"""

    def test_format_duration(self):
        self.assertEqual(prof_test.format_duration(12.34), "12.3s")
        self.assertEqual(prof_test.format_duration(59.9), "59.9s")
        self.assertEqual(prof_test.format_duration(63), "01m 03s")
        self.assertEqual(prof_test.format_duration(3661), "1h 01m 01s")
        self.assertEqual(prof_test.format_duration(0), "0.0s")
        self.assertEqual(prof_test.format_duration(-5), "0.0s")  # 负数钳 0


class ProfPreflightTest(unittest.TestCase):
    """preflight 环境自检。"""

    def test_python_version_too_old(self):
        with self.assertRaises(prof_test.PreflightError) as ctx:
            prof_test.check_python_version((3, 7, 5))
        self.assertIn("3.8", str(ctx.exception))

    def test_python_version_ok(self):
        self.assertTrue(prof_test.check_python_version((3, 8, 0)))
        self.assertTrue(prof_test.check_python_version((3, 12, 1)))

    def test_mpirun_missing(self):
        with mock.patch.object(prof_test.shutil, "which", return_value=None), \
                mock.patch.object(prof_test.sys, "version_info", (3, 9, 0)):
            with self.assertRaises(prof_test.PreflightError) as ctx:
                prof_test.preflight(["allreduce"], env={"ASCEND_HOME_PATH": "/usr/local/Ascend"})
        self.assertIn("MPICH", str(ctx.exception))
        self.assertIn("Open MPI", str(ctx.exception))

    def test_hccl_test_exe_missing(self):
        with mock.patch.object(prof_test.shutil, "which", return_value="/usr/bin/mpirun"), \
                mock.patch.object(prof_test.sys, "version_info", (3, 9, 0)), \
                mock.patch.object(prof_test.os.path, "isfile", return_value=False):
            with self.assertRaises(prof_test.PreflightError) as ctx:
                prof_test.preflight(["allreduce"], env={"ASCEND_HOME_PATH": "/usr/local/Ascend"})
        self.assertIn("all_reduce_test", str(ctx.exception))
        # 报错路径 = $ASCEND_HOME_PATH + 默认子目录 + exe
        self.assertIn(os.path.join("/usr/local/Ascend", "tools/hccl_test/bin", "all_reduce_test"),
                      str(ctx.exception))

    def test_ascend_home_path_not_set(self):
        with mock.patch.object(prof_test.shutil, "which", return_value="/usr/bin/mpirun"), \
                mock.patch.object(prof_test.sys, "version_info", (3, 9, 0)):
            with self.assertRaises(prof_test.PreflightError) as ctx:
                prof_test.preflight(["allreduce"], env={})
        self.assertIn("ASCEND_HOME_PATH", str(ctx.exception))
        self.assertIn("setenv.bash", str(ctx.exception))

    def test_all_satisfied(self):
        with mock.patch.object(prof_test.shutil, "which", return_value="/usr/bin/mpirun"), \
                mock.patch.object(prof_test.sys, "version_info", (3, 9, 0)), \
                mock.patch.object(prof_test.os.path, "isfile", return_value=True), \
                mock.patch.dict(os.environ, {}, clear=True):
            # 通过时返回 MPI flavor 供命令构造
            self.assertEqual(prof_test.preflight(
                ["allreduce", "alltoallv"], env={"ASCEND_HOME_PATH": "/usr/local/Ascend"}), "mpich")


class ProfExePathTest(unittest.TestCase):
    """P12b hccl_test 路径拼接（$ASCEND_HOME_PATH + 相对路径）。"""

    ENV = {"ASCEND_HOME_PATH": "/usr/local/Ascend"}

    def test_default_subdir_join(self):
        # 默认：$ASCEND_HOME_PATH/tools/hccl_test/bin/<exe>
        self.assertEqual(
            prof_test.exe_path_of("allreduce", None, env=self.ENV),
            os.path.join("/usr/local/Ascend", "tools/hccl_test/bin", "all_reduce_test"))

    def test_custom_subdir_join(self):
        self.assertEqual(
            prof_test.exe_path_of("reduce_scatter", "my/bin", env=self.ENV),
            os.path.join("/usr/local/Ascend", "my/bin", "reduce_scatter_test"))

    def test_absolute_bin_dir_passthrough(self):
        # 绝对路径直接使用，不拼 ASCEND_HOME_PATH
        path = prof_test.exe_path_of("allreduce", "/opt/custom/bin", env={})
        self.assertEqual(path, os.path.join("/opt/custom/bin", "all_reduce_test"))

    def test_missing_env_raises(self):
        with self.assertRaises(prof_test.PreflightError):
            prof_test.exe_path_of("allreduce", None, env={})
        # 空白串同未设置
        with self.assertRaises(prof_test.PreflightError):
            prof_test.exe_path_of("allreduce", None, env={"ASCEND_HOME_PATH": "  "})

    def test_env_whitespace_stripped(self):
        path = prof_test.exe_path_of("allgather", None, env={"ASCEND_HOME_PATH": " /usr/local/Ascend "})
        self.assertTrue(path.startswith(os.path.join("/usr/local/Ascend", "")))


def write_csv(path, rows):
    import csv as csv_mod
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv_mod.DictWriter(handle, fieldnames=list(prof_test.CSV_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class OptDedupTest(unittest.TestCase):
    """O01 最新优先去重。"""

    def test_keep_latest(self):
        rows = [
            make_row(timestamp="1000", alg_bandwidth="1.0"),
            make_row(timestamp="2000", alg_bandwidth="2.0"),  # 最新，保留
            make_row(timestamp="1500", alg_bandwidth="3.0"),  # 旧，丢弃
        ]
        kept, dropped = optimize_config.dedup_latest(rows)
        self.assertEqual(len(kept), 1)
        self.assertEqual(dropped, 2)
        self.assertEqual(kept[0]["timestamp(ms)"], "2000")
        self.assertEqual(kept[0]["alg_bandwidth(GB/s)"], "2.0")

    def test_different_key_not_deduped(self):
        rows = [make_row(), make_row(engine="aiv")]
        kept, dropped = optimize_config.dedup_latest(rows)
        self.assertEqual(len(kept), 2)
        self.assertEqual(dropped, 0)


class OptGroupTest(unittest.TestCase):
    """O02 分组键正确性。"""

    def test_engine_not_group_key(self):
        rows = [make_row(engine="aicpu"), make_row(engine="aiv"), make_row(engine="dpu")]
        groups = optimize_config.group_rows(rows)
        self.assertEqual(len(groups), 1)  # engine 不同仍同组（比较维度）
        rows2 = [make_row(), make_row(data_type="fp16")]
        self.assertEqual(len(optimize_config.group_rows(rows2)), 2)
        rows3 = [make_row(), make_row(reduce_type="NA")]
        self.assertEqual(len(optimize_config.group_rows(rows3)), 2)
        rows4 = [make_row(), make_row(**{"ranks.ranks": "16"})]
        self.assertEqual(len(optimize_config.group_rows(rows4)), 2)


class OptSelectBestTest(unittest.TestCase):
    """O03 带宽选优 + latency tie-break。"""

    def test_bandwidth_max_wins(self):
        rows = [
            make_row(alg_bandwidth="1.0", alg_latency="50"),
            make_row(alg_bandwidth="3.0", alg_latency="200", engine="aiv"),
            make_row(alg_bandwidth="2.0", alg_latency="80", engine="ccums"),
        ]
        best = optimize_config.select_best_per_size(rows)
        self.assertEqual(best[8192]["engine"], "aiv")

    def test_latency_tie_break(self):
        rows = [
            make_row(alg_bandwidth="2.0", alg_latency="100"),
            make_row(alg_bandwidth="2.0", alg_latency="60", engine="aiv"),
        ]
        best = optimize_config.select_best_per_size(rows)
        self.assertEqual(best[8192]["engine"], "aiv")

    def test_per_size_independent(self):
        rows = [
            make_row(size_bytes="8192", alg_bandwidth="1.0"),
            make_row(size_bytes="16384", alg_bandwidth="2.0", engine="aiv"),
        ]
        best = optimize_config.select_best_per_size(rows)
        self.assertEqual(len(best), 2)
        self.assertEqual(best[8192]["engine"], "aicpu")
        self.assertEqual(best[16384]["engine"], "aiv")

    def test_per_size_winners_differ_by_size(self):
        # 每个 size 都有 3 个候选，且不同 size 的最优算法不同：
        # 选优必须逐 size 独立比较，而非全局取一个最优
        rows = [
            # size=8192：aiv 最优（5.0 > 3.0 > 1.0）
            make_row(size_bytes="8192", alg_bandwidth="1.0"),
            make_row(size_bytes="8192", alg_bandwidth="5.0", engine="aiv"),
            make_row(size_bytes="8192", alg_bandwidth="3.0", engine="ccums"),
            # size=16384：ccums 最优（4.0 > 2.0 > 1.5）
            make_row(size_bytes="16384", alg_bandwidth="2.0"),
            make_row(size_bytes="16384", alg_bandwidth="1.5", engine="aiv"),
            make_row(size_bytes="16384", alg_bandwidth="4.0", engine="ccums"),
        ]
        best = optimize_config.select_best_per_size(rows)
        self.assertEqual(best[8192]["engine"], "aiv")
        self.assertEqual(best[16384]["engine"], "ccums")


class OptMergeIntervalsTest(unittest.TestCase):
    """O04 区间合并。"""

    def test_merge_same_algo_adjacent(self):
        points = {
            8192: make_row(size_bytes="8192"),
            16384: make_row(size_bytes="16384"),
            32768: make_row(size_bytes="32768", engine="aiv"),
        }
        intervals = optimize_config.merge_intervals(points)
        self.assertEqual(len(intervals), 2)
        self.assertEqual((intervals[0][0], intervals[0][1]), (8192, 16384))
        self.assertEqual((intervals[1][0], intervals[1][1]), (32768, 32768))

    def test_diff_algo_single_points(self):
        points = {
            8192: make_row(size_bytes="8192"),
            16384: make_row(size_bytes="16384", engine="aiv"),
        }
        intervals = optimize_config.merge_intervals(points)
        self.assertEqual([(lo, hi) for lo, hi, _ in intervals], [(8192, 8192), (16384, 16384)])


class OptFillGapsTest(unittest.TestCase):
    """O05 补缝（归右）。"""

    def test_gap_assigned_to_right(self):
        intervals = [
            (8192, 8192, make_row(size_bytes="8192")),                       # A: mesh
            (32768, 32768, make_row(size_bytes="32768", engine="aiv")),      # B: aiv
        ]
        filled = optimize_config.fill_gaps(intervals)
        self.assertEqual([(lo, hi) for lo, hi, _ in filled], [(8192, 8192), (8193, 32768)])
        # 右侧区间算法（B=aiv）覆盖整条缝
        self.assertEqual(filled[1][2]["engine"], "aiv")

    def test_no_gap_no_change(self):
        intervals = [(8192, 8192, make_row()), (8193, 9000, make_row(engine="aiv"))]
        filled = optimize_config.fill_gaps(intervals)
        self.assertEqual([(lo, hi) for lo, hi, _ in filled], [(8192, 8192), (8193, 9000)])

    def test_first_interval_not_extended_left(self):
        intervals = [(4096, 4096, make_row(size_bytes="4096"))]
        filled = optimize_config.fill_gaps(intervals)
        self.assertEqual(filled[0][0], 4096)  # 首区间不向 0 扩展


class OptWhitelistTest(unittest.TestCase):
    """O06 白名单硬校验。"""

    def test_strictordered_dropped_with_count(self):
        row = make_row(**{"algorithm.executor_type": "strictordered"})
        self.assertEqual(optimize_config.validate_winner(row), "executor")
        conf, _ = optimize_config.build_conf([row])
        self.assertEqual(conf["op_types"], {})  # 无 rule 产出

    def test_invalid_template_dtype_engine_op(self):
        # meshconcurrent / nhraicpureduce 已回白名单（plugin 不做 template 枚举校验），
        # 拼写错误的 template 仍须拒绝
        self.assertEqual(optimize_config.validate_winner(
            make_row(**{"algorithm.template_type": "meshconcurrent"})), None)
        self.assertEqual(optimize_config.validate_winner(
            make_row(**{"algorithm.template_type": "nhraicpureduce"})), None)
        self.assertEqual(optimize_config.validate_winner(
            make_row(**{"algorithm.template_type": "badtemplate"})), "template")
        self.assertEqual(optimize_config.validate_winner(
            make_row(engine="badengine")), "engine")
        self.assertEqual(optimize_config.validate_winner(
            make_row(data_type="fp8e5m2")), "data_type")
        self.assertEqual(optimize_config.validate_winner(
            make_row(op_type="allgatherv")), "op_type")
        self.assertIsNone(optimize_config.validate_winner(make_row()))

    def test_dropped_counted_in_build(self):
        # strictordered 带宽最高但白名单外 → 该点丢弃，不产 rule
        rows = [
            make_row(alg_bandwidth="5.0", **{"algorithm.executor_type": "strictordered"}),
            make_row(alg_bandwidth="1.0"),
        ]
        conf, total = optimize_config.build_conf(rows)
        # strictordered 胜者被丢弃后该 size 点无 rule；sole{mesh} 是唯一合法候选，
        # 但选优在先：strictordered 胜出 → 点丢弃 → 无 rule
        self.assertEqual(total, 0)


class OptMultiTemplateTest(unittest.TestCase):
    """多级模板放行 + conf 拼接串口径转换。"""

    def test_multi_template_tokens_validated_individually(self):
        # 多级逐 token 过白名单：全在白名单内放行，任一 token 白名单外拒绝
        self.assertIsNone(optimize_config.validate_winner(
            make_row(**{"algorithm.template_type": "meshconcur,nhr,nhr"})))
        self.assertEqual(optimize_config.validate_winner(
            make_row(**{"algorithm.template_type": "mesh,badtemplate"})), "template")

    def test_multi_template_concatenated_in_rule(self):
        # rule.template 写 plugin 拼接串口径（去逗号），CSV 行内保持语义串
        rows = [make_row(alg_bandwidth="9.0", **{"algorithm.executor_type": "sequence",
                                                "algorithm.template_type": "meshconcur,nhr,nhr"})]
        conf, total = optimize_config.build_conf(rows)
        self.assertEqual(total, 1)
        rule = conf["op_types"]["allreduce"]["rules"][0]
        self.assertEqual(rule["template"], "meshconcurnhrnhr")
        self.assertEqual(rule["executor"], "sequence")

    def test_single_template_passthrough(self):
        # 单级语义串与拼接串同形，转换透传
        rows = [make_row(alg_bandwidth="9.0")]
        conf, total = optimize_config.build_conf(rows)
        self.assertEqual(total, 1)
        rule = conf["op_types"]["allreduce"]["rules"][0]
        self.assertEqual(rule["template"], "mesh")


class OptEnginePassthroughTest(unittest.TestCase):
    """O08 engine 透传。"""

    def test_engine_written_as_is(self):
        for engine in ("aicpu", "aiv", "ccums", "ccusched"):
            conf, total = optimize_config.build_conf([make_row(engine=engine)])
            self.assertEqual(total, 1)
            self.assertEqual(conf["op_types"]["allreduce"]["rules"][0]["engine"], engine)

    def test_dpu_independent_rules(self):
        # dpu 与 device 并存：各成一组独立产 rule；device 排序在前
        rows = [
            make_row(engine="aicpu", alg_bandwidth="1.0"),
            make_row(engine="dpu", alg_bandwidth="5.0"),
        ]
        conf, total = optimize_config.build_conf(rows)
        self.assertEqual(total, 2)
        rules = conf["op_types"]["allreduce"]["rules"]
        self.assertEqual([r["engine"] for r in rules], ["aicpu", "dpu"])

    def test_dpu_rule_forces_min_servers_2(self):
        # dpu rule 强制 min_servers>=2（hostdpu 拓扑必要条件）：单机采集的 dpu
        # 数据不会产死 rule；servers=1 时丢弃 max_servers=1 避免 min>max SchemaError
        rows = [make_row(engine="dpu", **{"ranks.servers": "1"})]
        conf, total = optimize_config.build_conf(rows)
        self.assertEqual(total, 1)
        match = conf["op_types"]["allreduce"]["rules"][0]["match"]
        self.assertEqual(match["min_servers"], 2)
        self.assertNotIn("max_servers", match)

    def test_dpu_rule_keeps_multi_server_bounds(self):
        # 多 server 采集的 dpu 数据：min=max=servers 原样保留
        rows = [make_row(engine="dpu", **{"ranks.servers": "2"})]
        conf, total = optimize_config.build_conf(rows)
        self.assertEqual(total, 1)
        match = conf["op_types"]["allreduce"]["rules"][0]["match"]
        self.assertEqual(match["min_servers"], 2)
        self.assertEqual(match["max_servers"], 2)


class OptFailedFilterTest(unittest.TestCase):
    """O09 failed 行过滤。"""

    def test_failed_dropped_null_kept(self):
        rows = [
            make_row(check_result="failed", alg_bandwidth="99.0"),
            make_row(check_result="NULL", alg_bandwidth="1.0", engine="aiv"),
            make_row(check_result="success", alg_bandwidth="2.0", engine="ccums"),
        ]
        kept, dropped = optimize_config.filter_failed(rows)
        self.assertEqual(dropped, 1)
        self.assertEqual(len(kept), 2)
        # NULL 保留参与选优：胜者为 ccums（带宽 2.0）
        best = optimize_config.select_best_per_size(kept)
        self.assertEqual(best[8192]["engine"], "ccums")

    def test_noresult_dropped(self):
        rows = [
            make_row(check_result="noresult", alg_bandwidth=""),
            make_row(check_result="success", alg_bandwidth="2.0", engine="ccums"),
        ]
        kept, dropped = optimize_config.filter_failed(rows)
        self.assertEqual(dropped, 1)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["check_result"], "success")


class OptBytesNormalizeTest(unittest.TestCase):
    """字节口径归一化（allgather 系 ÷ ranks、scatter × ranks，其余透传）。"""

    def test_allgather_divided_by_ranks(self):
        # allgather：采集 size 是输出总量，归一化为每 rank 发送量（向上取整）
        rows = [make_row(op_type="allgather", size_bytes="81920", **{"ranks.ranks": "8"})]
        normalized = optimize_config.normalize_bytes(rows)
        self.assertEqual(normalized[0]["size_bytes"], "10240")

    def test_allgatherv_ceil_non_divisible(self):
        # allgatherv：非整除向上取整（ceil）
        rows = [make_row(op_type="allgatherv", size_bytes="8193", **{"ranks.ranks": "8"})]
        normalized = optimize_config.normalize_bytes(rows)
        self.assertEqual(normalized[0]["size_bytes"], "1025")

    def test_scatter_multiplied_by_ranks(self):
        # scatter：采集 size 是每 rank 收到量，插件 nBytes 是 root 总发出量，需 × ranks
        rows = [make_row(op_type="scatter", size_bytes="10240", **{"ranks.ranks": "8"})]
        normalized = optimize_config.normalize_bytes(rows)
        self.assertEqual(normalized[0]["size_bytes"], "81920")

    def test_aligned_ops_passthrough(self):
        # 其余 op（allreduce 等）采集口径与插件一致，原值透传
        rows = [make_row(op_type="allreduce", size_bytes="8192"),
                make_row(op_type="alltoall", size_bytes="81920")]
        normalized = optimize_config.normalize_bytes(rows)
        self.assertEqual(normalized[0]["size_bytes"], "8192")
        self.assertEqual(normalized[1]["size_bytes"], "81920")

    def test_degenerate_cases_passthrough(self):
        # ranks<=1 / size<=0 / 空值：不换算，原样透传（含字符串原样保留）
        rows = [
            make_row(op_type="allgather", size_bytes="8192", **{"ranks.ranks": "1"}),
            make_row(op_type="allgather", size_bytes="0"),
            make_row(op_type="allgather", size_bytes=""),
        ]
        normalized = optimize_config.normalize_bytes(rows)
        self.assertEqual(normalized[0]["size_bytes"], "8192")
        self.assertEqual(normalized[1]["size_bytes"], "0")
        self.assertEqual(normalized[2]["size_bytes"], "")

    def test_build_conf_end_to_end_allgather(self):
        # 端到端：conf 规则 min/max_bytes 落每 rank 口径（81920/8ranks → 10240）
        rows = [make_row(op_type="allgather", size_bytes="81920", **{"ranks.ranks": "8"})]
        conf, total = optimize_config.build_conf(rows)
        self.assertEqual(total, 1)
        match = conf["op_types"]["allgather"]["rules"][0]["match"]
        self.assertEqual(match["min_bytes"], 10240)
        self.assertEqual(match["max_bytes"], 10240)


class OptConfStructureTest(unittest.TestCase):
    """O10 conf 结构合规。"""

    def test_conf_structure(self):
        rows = [
            make_row(size_bytes="8192", alg_bandwidth="1.0"),
            make_row(size_bytes="8192", alg_bandwidth="3.0", engine="aiv"),
            make_row(size_bytes="16384", alg_bandwidth="1.5", engine="aiv"),
            make_row(size_bytes="32768", alg_bandwidth="2.0", engine="ccums"),
        ]
        conf, total = optimize_config.build_conf(rows)
        self.assertEqual(conf["version"], 1)
        self.assertIn("allreduce", conf["op_types"])
        rules = conf["op_types"]["allreduce"]["rules"]
        self.assertEqual(total, len(rules))
        # 8192/16384 均 aiv 胜 → 合并；32768 ccums → 单点 + 补缝归右
        self.assertEqual(len(rules), 2)
        match = rules[0]["match"]
        for field in ("min_ranks", "max_ranks", "min_npus_per_server", "max_npus_per_server",
                      "min_servers", "max_servers", "min_pods", "max_pods",
                      "min_super_pods", "max_super_pods", "min_bytes", "max_bytes", "data_type"):
            self.assertIn(field, match)
        self.assertEqual(match["min_ranks"], 8)
        self.assertEqual(match["max_npus_per_server"], 8)
        self.assertEqual(match["data_type"], "fp32")
        self.assertEqual((match["min_bytes"], match["max_bytes"]), (8192, 16384))
        self.assertEqual(rules[0]["engine"], "aiv")
        self.assertEqual(rules[0]["executor"], "sole")
        self.assertEqual(rules[0]["template"], "mesh")
        self.assertEqual(rules[0]["cost"], 0.0)
        # 第二条：ccums，补缝后向左扩展
        self.assertEqual(rules[1]["match"]["min_bytes"], 16385)
        self.assertEqual(rules[1]["match"]["max_bytes"], 32768)
        self.assertEqual(rules[1]["engine"], "ccums")

    def test_unspecified_pods_sentinel_skipped(self):
        # pods/super_pods=0（采集侧未指定哨兵）时 match 不写该维度（单机错位规避）
        rows = [make_row(size_bytes="8192", **{"ranks.pods": "0", "ranks.super_pods": "0"})]
        conf, total = optimize_config.build_conf(rows)
        self.assertEqual(total, 1)
        match = conf["op_types"]["allreduce"]["rules"][0]["match"]
        for absent in ("min_pods", "max_pods", "min_super_pods", "max_super_pods"):
            self.assertNotIn(absent, match)
        # 显式指定的维度仍写入
        self.assertEqual(match["min_ranks"], 8)
        self.assertEqual(match["max_servers"], 1)

    def test_zero_ranks_sentinel_skipped(self):
        # 哨兵跳过逻辑对全部 5 个拓扑字段一致生效（空值同样按 0 处理）
        rows = [make_row(size_bytes="8192", **{"ranks.ranks": "0", "ranks.servers": ""})]
        conf, _ = optimize_config.build_conf(rows)
        match = conf["op_types"]["allreduce"]["rules"][0]["match"]
        for absent in ("min_ranks", "max_ranks", "min_servers", "max_servers"):
            self.assertNotIn(absent, match)
        self.assertEqual(match["min_npus_per_server"], 8)

    def test_interval_algo_is_per_size_winner(self):
        # 端到端选优不变量：每个 [lo,hi] 区间的算法 = 落在该区间内各采集 size 点
        # 上的逐 size 最优合法算法（逐点比较 → 合并 → 补缝归右后的最终归属）
        rows = [
            # 8192：aiv 胜（5.0）
            make_row(size_bytes="8192", alg_bandwidth="1.0"),
            make_row(size_bytes="8192", alg_bandwidth="5.0", engine="aiv"),
            make_row(size_bytes="8192", alg_bandwidth="3.0", engine="ccums"),
            # 16384：aiv 胜（2.5）→ 与 8192 同算法，合并
            make_row(size_bytes="16384", alg_bandwidth="2.5", engine="aiv"),
            # 32768：ccums 胜（4.0）→ 单点区间 + 补缝归右
            make_row(size_bytes="32768", alg_bandwidth="4.0", engine="ccums"),
            make_row(size_bytes="32768", alg_bandwidth="0.5", engine="aiv"),
        ]
        conf, total = optimize_config.build_conf(rows)
        rules = conf["op_types"]["allreduce"]["rules"]
        self.assertEqual(total, 2)
        self.assertEqual(rules[0]["match"]["min_bytes"], 8192)
        self.assertEqual(rules[0]["match"]["max_bytes"], 16384)
        self.assertEqual(rules[0]["engine"], "aiv")
        self.assertEqual(rules[1]["match"]["min_bytes"], 16385)
        self.assertEqual(rules[1]["match"]["max_bytes"], 32768)
        self.assertEqual(rules[1]["engine"], "ccums")

    def test_end_to_end_files(self):
        # 端到端：CSV 文件 → conf JSON 文件
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "in.csv")
            dst = os.path.join(tmp, "out.json")
            write_csv(src, [make_row(alg_bandwidth="1.0"),
                            make_row(alg_bandwidth="3.0", engine="aiv")])
            argv = ["--input", src, "--output", dst]
            with mock.patch.object(sys, "argv", ["optimize_config.py"] + argv):
                self.assertEqual(optimize_config.main(argv), 0)
            import json
            with open(dst, encoding="utf-8") as handle:
                conf = json.load(handle)
            self.assertEqual(conf["version"], 1)
            rules = conf["op_types"]["allreduce"]["rules"]
            self.assertEqual(rules[0]["engine"], "aiv")


class ProfRunOneTimeoutTest(unittest.TestCase):
    """P04+ run_one 超时：挂死轮被杀、落失败、不阻塞采集。"""

    def test_run_one_timeout_returns_failure(self):
        # 模拟挂死进程：sleep 超过 timeout → TimeoutExpired → (输出, stderr, False)
        cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
        import time
        start = time.monotonic()
        stdout, stderr, ok = prof_test.run_one(cmd, env=None, timeout=2)
        elapsed = time.monotonic() - start
        self.assertFalse(ok)
        self.assertLess(elapsed, 15)  # 远小于 sleep(30)，证明被超时杀掉
        self.assertIn("run timeout after 2s", stderr)

    def test_run_one_normal_no_timeout(self):
        # 正常进程不受 timeout 影响，返回码透传
        cmd = [sys.executable, "-c", "print('hello')"]
        stdout, _, ok = prof_test.run_one(cmd, env=None, timeout=30)
        self.assertTrue(ok)
        self.assertIn("hello", stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
