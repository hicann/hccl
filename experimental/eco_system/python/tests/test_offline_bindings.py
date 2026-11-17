#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Device-free checks for CommConfig overrides and ops._as_comm."""

from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
_HCOMM_PY = os.path.abspath(
    os.path.join(HERE, "..", "..", "..", "..", "hcomm", "experimental", "eco_system", "python")
)
if os.path.isdir(_HCOMM_PY) and _HCOMM_PY not in sys.path:
    sys.path.insert(0, _HCOMM_PY)

from hccl4py import ops  # noqa: E402
from hcomm4py.comm import (  # noqa: E402
    Comm,
    comm_init_cluster_info_config,
    comm_init_root_info_config,
)
import hcomm4py.comm as hcomm_comm  # noqa: E402
from hccl4py.ops import _as_comm  # noqa: E402
from hccl4py.types import (  # noqa: E402
    HCCL_COMM_CONFIG_BYTES,
    HCCL_COMM_CONFIG_SIZE_V12,
    HCCL_COMM_DEFAULT_OP_EXPANSION_MODE,
    HCCL_OP_EXPANSION_MODE_CCU_SCHED,
    HcclComm,
    HcclCommConfig,
    HcclDataType,
    HcclReduceOp,
    HcclRootInfo,
    HcclSendRecvType,
    _OFF,
    fill_comm_config,
)


def _u32(cfg: HcclCommConfig, off: int) -> int:
    import ctypes

    return ctypes.c_uint32.from_buffer(cfg._raw, off).value


class TestOfflineBindings(unittest.TestCase):
    def test_field_overrides_and_version_gates(self):
        cfg = fill_comm_config(
            HcclCommConfig(),
            buffer_size=200,
            deterministic=1,
            exec_timeout=30,
            qos=3,
            op_expansion_mode=HCCL_OP_EXPANSION_MODE_CCU_SCHED,
        )
        self.assertEqual(_u32(cfg, _OFF["hcclBufferSize"]), 200)
        self.assertEqual(_u32(cfg, _OFF["hcclDeterministic"]), 1)
        self.assertEqual(_u32(cfg, _OFF["hcclExecTimeOut"]), 30)
        self.assertEqual(_u32(cfg, _OFF["hcclQos"]), 3)
        self.assertEqual(
            _u32(cfg, _OFF["hcclOpExpansionMode"]), HCCL_OP_EXPANSION_MODE_CCU_SCHED
        )
        self.assertEqual(HCCL_OP_EXPANSION_MODE_CCU_SCHED, 6)
        with self.assertRaises(TypeError):
            fill_comm_config(HcclCommConfig(), op_expansion_mode=6.5)
        with self.assertRaises(TypeError):
            fill_comm_config(HcclCommConfig(), op_expansion_mode=True)
        with self.assertRaises(ValueError):
            fill_comm_config(HcclCommConfig(), version=11, logical_size=2240)
        with self.assertRaises(ValueError):
            fill_comm_config(HcclCommConfig(), version=12, logical_size=2244)
        with self.assertRaises(ValueError):
            fill_comm_config(HcclCommConfig(), logical_size=0)
        with self.assertRaises(ValueError):
            fill_comm_config(HcclCommConfig(), logical_size=HCCL_COMM_CONFIG_BYTES)
        with self.assertRaises(ValueError):
            fill_comm_config(
                HcclCommConfig(), logical_size=HCCL_COMM_CONFIG_SIZE_V12 + 1
            )
        cfg_default = fill_comm_config(HcclCommConfig())
        self.assertEqual(
            _u32(cfg_default, _OFF["hcclOpExpansionMode"]),
            HCCL_COMM_DEFAULT_OP_EXPANSION_MODE,
        )

    def test_offsets_gold_full(self):
        gold = {
            "size": 0,
            "magic": 8,
            "version": 12,
            "reserved": 16,
            "hcclBufferSize": 24,
            "hcclDeterministic": 28,
            "hcclCommName": 32,
            "hcclUdi": 160,
            "hcclOpExpansionMode": 288,
            "hcclRdmaTrafficClass": 292,
            "hcclRdmaServiceLevel": 296,
            "hcclWorldRankID": 300,
            "hcclJobID": 304,
            "aclGraphZeroCopyEnable": 312,
            "hcclExecTimeOut": 316,
            "hcclAlgo": 320,
            "hcclRetryEnable": 1920,
            "hcclRetryParams": 1970,
            "hcclBufferName": 2098,
            "hcclQos": 2228,
            "hcclSymWinMaxMemSizePerRank": 2232,
            "hcclChannelSqDepth": 2240,
            "hcclChannelScqDepth": 2244,
        }
        self.assertEqual(set(_OFF), set(gold))
        for name, expect in gold.items():
            self.assertEqual(_OFF[name], expect, name)

    def test_handle_value_rejects_bool_and_opaque(self):
        from hccl4py._validate import handle_value, is_null_handle

        self.assertTrue(is_null_handle(None))
        self.assertTrue(is_null_handle(0))
        self.assertFalse(is_null_handle(1))
        with self.assertRaises(TypeError):
            is_null_handle(True)
        with self.assertRaises(TypeError):
            handle_value(0.0)
        with self.assertRaises(TypeError):
            handle_value(1.5)
        with self.assertRaises(TypeError):
            handle_value(object())

    def test_as_comm_rejects_destroyed_comm(self):
        class _FakeDestroyed:
            handle = None

        with self.assertRaises(ValueError):
            _as_comm(_FakeDestroyed())
        with self.assertRaises(ValueError):
            _as_comm(None)
        with self.assertRaises(ValueError):
            _as_comm(0)
        with self.assertRaises(ValueError):
            _as_comm(HcclComm(0))
        with self.assertRaises(ValueError):
            Comm(HcclComm(0))
        with self.assertRaises(ValueError):
            Comm(-1)
        c = Comm(0xDEAD)
        self.assertEqual(int(_as_comm(c).value), 0xDEAD)
        c._handle = None
        with self.assertRaises(ValueError):
            _as_comm(c)

    def test_comm_destroy_idempotent(self):
        lib = mock.Mock()
        lib.HcclCommDestroy.return_value = 0
        with mock.patch.object(hcomm_comm, "_bind", return_value=lib):
            c = Comm(0xABC)
            c.destroy()
            c.destroy()
            self.assertEqual(lib.HcclCommDestroy.call_count, 1)

    def test_ops_arg_validation(self):
        with self.assertRaises(TypeError):
            ops.reduce(1, 0, 0, count=1.5)
        with self.assertRaises(ValueError):
            ops.reduce(1, 0, 0, count=-1)
        with self.assertRaises(TypeError):
            ops.reduce(1, 0, 0, count=1, data_type=1.5)
        with self.assertRaises(TypeError):
            ops.reduce(1, 0, 0, count=1, data_type=True)
        with self.assertRaises(ValueError):
            ops.reduce(1, 0, 0, count=1, data_type=999)
        with self.assertRaises(ValueError):
            ops.reduce(1, 0, 0, count=1, reduce_op=999)
        with self.assertRaises(ValueError):
            ops.reduce(1, 0, 0, count=1, root=-1)
        with self.assertRaises(ValueError):
            ops.reduce(1, 0, 0, count=1, root=(1 << 32) + 3)
        with self.assertRaises(ValueError):
            ops.reduce(1, 0, 0, count=(1 << 64))
        with self.assertRaises(ValueError):
            ops.all_reduce(0, 0, 0, count=1)
        with self.assertRaises(ValueError):
            ops.broadcast(1, 0, count=1, data_type=13)
        with self.assertRaises(ValueError):
            ops.send(1, 0, count=1, data_type=HcclDataType.FP32, dest_rank=-2)
        with self.assertRaises(TypeError):
            ops.batch_send_recv(1, None)
        with self.assertRaises(ValueError):
            ops.make_send_recv_item(HcclSendRecvType.SEND, 0, -1, HcclDataType.FP32, 0)
        with self.assertRaises(ValueError):
            ops.all_gather_v(1, 0, 1, 0, None, None)
        item = ops.make_send_recv_item(
            HcclSendRecvType.SEND, 0, 4, HcclDataType.FP32, 1
        )
        self.assertEqual(item.count, 4)
        self.assertEqual(int(item.dataType), int(HcclDataType.FP32))
        self.assertEqual(int(HcclReduceOp.SUM), 0)

    def test_comm_arg_validation(self):
        with self.assertRaises(ValueError):
            comm_init_root_info_config(0, HcclRootInfo(), 0)
        with self.assertRaises(ValueError):
            comm_init_root_info_config(2, HcclRootInfo(), 2)
        with self.assertRaises(TypeError):
            comm_init_root_info_config(2, object(), 0)
        with self.assertRaises(TypeError):
            comm_init_root_info_config(2, HcclRootInfo(), 0, config=object())
        with self.assertRaises(ValueError):
            comm_init_root_info_config(2, HcclRootInfo(), 0, config=HcclCommConfig())
        with self.assertRaises(ValueError):
            comm_init_cluster_info_config("", 0)
        with self.assertRaises(TypeError):
            comm_init_cluster_info_config(123, 0)
        with self.assertRaises(ValueError):
            comm_init_cluster_info_config("/tmp/x", -1)
        with self.assertRaises(ValueError):
            comm_init_cluster_info_config("/tmp/x", 0, config=HcclCommConfig())

    def test_bind_missing_domain_symbol_message(self):
        class _OldLib:
            pass

        with hcomm_comm._BIND_LOCK:
            was_bound = hcomm_comm._BOUND
            hcomm_comm._BOUND = False
        try:
            with mock.patch.object(hcomm_comm, "get_hcomm", return_value=_OldLib()):
                with self.assertRaises(RuntimeError) as ctx:
                    hcomm_comm._bind()
            msg = str(ctx.exception)
            self.assertIn("libhcomm.so", msg)
            self.assertIn("HcclGetRootInfo", msg)
            self.assertIn("non-V2", msg)
            self.assertIn("op_base_v2", msg)
            self.assertNotIn("up/9.1.0", msg)
            self.assertNotIn("AttributeError", msg)
        finally:
            with hcomm_comm._BIND_LOCK:
                hcomm_comm._BOUND = was_bound

    def test_depends_on_hcomm4py(self):
        import hccl4py.types as t
        import hcomm4py.types as ht

        self.assertIs(t.HcclCommConfig, ht.HcclCommConfig)
        self.assertIs(t.fill_comm_config, ht.fill_comm_config)


    def test_lib_candidates_explicit_override(self):
        import os
        from hccl4py import _lib as libmod

        key = "HCCL4PY_HCCL_LIB"
        old = os.environ.get(key)
        old_ascend = os.environ.get("ASCEND_LIBRARY_PATH")
        try:
            os.environ[key] = "/tmp/hccl4py_explicit_libhccl.so"
            with self.assertRaises(OSError) as cm:
                libmod._load("libhccl.so")
            self.assertIn("explicit override", str(cm.exception))
            os.environ.pop(key, None)
            roots = libmod._env_lib_roots()
            self.assertIsInstance(roots, list)
            os.environ["ASCEND_LIBRARY_PATH"] = ""
            roots2 = libmod._env_lib_roots()
            self.assertIsInstance(roots2, list)
        finally:
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
            if old_ascend is None:
                os.environ.pop("ASCEND_LIBRARY_PATH", None)
            else:
                os.environ["ASCEND_LIBRARY_PATH"] = old_ascend
            libmod._hccl = None
            libmod._hcomm = None
            libmod._ascendcl = None

    def test_as_stream_as_ptr_reject_bool(self):
        from hccl4py._ops_util import _as_ptr, _as_stream

        with self.assertRaises(TypeError):
            _as_stream(True)
        with self.assertRaises(TypeError):
            _as_stream(False)
        with self.assertRaises(TypeError):
            _as_ptr(True)
        with self.assertRaises(TypeError):
            _as_ptr(False)
        self.assertEqual(_as_ptr(None).value, None)
        self.assertEqual(int(_as_ptr(0xABC).value), 0xABC)


if __name__ == "__main__":
    unittest.main()
