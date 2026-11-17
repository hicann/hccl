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

"""Chip / ranktable helpers shared by test suites (A2 / A3 / A5).

Override with env:
  HCCL4PY_CHIP=a2|a3|a5|auto   (default auto; aliases 950/atlas350 → a5)
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
from typing import Dict, List, Optional, Tuple


def detect_chip() -> str:
    """Return 'a2' | 'a3' | 'a5' from HCCL4PY_CHIP or npu-smi / product name."""
    forced = (os.environ.get("HCCL4PY_CHIP") or "auto").strip().lower()
    if forced in ("a2", "a3", "a5"):
        return forced

    text = ""
    try:
        text = subprocess.check_output(
            ["npu-smi", "info"], text=True, stderr=subprocess.STDOUT, timeout=20
        )
    except Exception:
        try:
            text = subprocess.check_output(
                ["npu-smi", "info", "-l"], text=True, stderr=subprocess.STDOUT, timeout=20
            )
        except Exception:
            text = ""

    upper = text.upper()
    # Order matters: 950/A5 before 910. Avoid bare "A3"/"A2" (too easy to false-match).
    if "950" in upper or "ASCEND950" in upper or "ASCEND 950" in upper:
        return "a5"
    if "910_93" in upper or "910C" in upper or "ATLAS A3" in upper:
        return "a3"
    if "910B" in upper or "ATLAS A2" in upper:
        return "a2"
    # Last-resort product tokens (only if unambiguous product name present).
    if re.search(r"\bA5\b", upper) or "ASCEND950PR" in upper or "ASCEND950DT" in upper:
        return "a5"
    if re.search(r"\bA3\b", upper):
        return "a3"
    if re.search(r"\bA2\b", upper):
        return "a2"
    assume = (os.environ.get("HCCL4PY_ASSUME_CHIP") or "").strip().lower()
    if assume in ("a2", "a3", "a5"):
        return assume
    raise RuntimeError(
        "chip detect failed (npu-smi gave no A2/A3/A5 product tag). "
        "Set HCCL4PY_CHIP=a2|a3|a5 or HCCL4PY_ASSUME_CHIP explicitly."
    )


def host_ipv4() -> str:
    try:
        out = subprocess.check_output(
            ["hostname", "-I"], text=True, stderr=subprocess.DEVNULL
        )
        for tok in out.split():
            if "." in tok and not tok.startswith("127."):
                return tok
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return socket.gethostname() or "127.0.0.1"


def parse_hccn_addresses(path: str = "/etc/hccn.conf") -> Dict[int, str]:
    """Map device_id -> address_N from hccn.conf (A2/A3 device_ip)."""
    out: Dict[int, str] = {}
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = re.match(r"address_(\d+)\s*=\s*(\S+)", line.strip())
                if m:
                    out[int(m.group(1))] = m.group(2)
    except OSError:
        pass
    return out


def logic_to_phy(logic_id: int, *, strict: bool = False) -> int:
    """Map ACL logic device index to physical id for ranktable ``device_id``.

    On query failure, warn and fall back to ``logic_id`` unless ``strict=True``.
    """
    err = None
    try:
        from hccl4py._lib import get_ascendcl
        import ctypes

        acl = get_ascendcl()
        for name in ("rtGetDevicePhyIdByIndex", "aclrtGetDevicePhyIdByIndex"):
            try:
                fn = getattr(acl, name)
            except AttributeError:
                continue
            fn.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
            fn.restype = ctypes.c_int
            phy = ctypes.c_uint32()
            rc = fn(ctypes.c_uint32(logic_id), ctypes.byref(phy))
            if rc == 0:
                return int(phy.value)
            err = "symbol %s returned rc=%s" % (name, rc)
        if err is None:
            err = "no rtGetDevicePhyIdByIndex / aclrtGetDevicePhyIdByIndex export"
    except Exception as exc:
        err = repr(exc)
    msg = (
        "logic_to_phy(%s) failed (%s); falling back to logic_id — "
        "ranktable device_id may be wrong when ASCEND_RT_VISIBLE_DEVICES remaps"
        % (logic_id, err)
    )
    if strict:
        raise RuntimeError(msg)
    import sys

    print("WARN: %s" % msg, file=sys.stderr, flush=True)
    return logic_id


def normalize_chip(chip: Optional[str] = None) -> str:
    """Normalize chip to ``a2``|``a3``|``a5``; raise ValueError on unknown values."""
    raw = (chip if chip is not None else (os.environ.get("HCCL4PY_CHIP") or "")).strip().lower()
    if not raw or raw == "auto":
        return detect_chip()
    aliases = {"950": "a5", "atlas350": "a5", "ascend950": "a5"}
    chip_n = aliases.get(raw, raw)
    if chip_n not in ("a2", "a3", "a5"):
        raise ValueError(
            "chip must be a2|a3|a5 (or aliases 950/atlas350→a5), got %r" % chip
        )
    return chip_n


def _split_env_paths(value: Optional[str]) -> List[str]:
    """Split PATH-like values and drop empty segments (empty ⇒ CWD, CWE-426)."""
    if not value:
        return []
    return [p for p in value.split(os.pathsep) if p]


def _query_port_cna(dev_id: int, die: int, port: int) -> Optional[int]:
    """Query port_cna via hccn_tool. ``dev_id`` is hccn/NPU index (often 0..N-1)."""
    env = os.environ.copy()
    drv = "/usr/local/Ascend/driver/lib64"
    # Filter empty parts: an empty PATH/LD_LIBRARY_PATH component means CWD (CWE-426).
    ld_parts = [f"{drv}/driver", f"{drv}/common", drv]
    ld_parts.extend(_split_env_paths(env.get("LD_LIBRARY_PATH")))
    env["LD_LIBRARY_PATH"] = os.pathsep.join(ld_parts)
    path_parts = ["/usr/local/Ascend/driver/tools"]
    path_parts.extend(_split_env_paths(env.get("PATH")))
    env["PATH"] = os.pathsep.join(path_parts)
    try:
        out = subprocess.check_output(
            [
                "hccn_tool",
                "-g",
                "-port_info",
                "-i",
                str(dev_id),
                "-u",
                str(die),
                "-p",
                str(port),
            ],
            text=True,
            stderr=subprocess.STDOUT,
            env=env,
            timeout=15,
        )
    except Exception:
        return None
    if "invalid" in out.lower() and "port_cna" not in out:
        return None
    m = re.search(r"port_cna:\s*(0x[0-9a-fA-F]+|\d+)", out)
    if not m:
        return None
    return int(m.group(1), 0)


def _eid_from_cna(cna: int) -> str:
    return "000000000000000000100000dfdf%04x" % (cna & 0xFFFF)


def _hccn_dev_id(logic_id: int, phy_id: int) -> int:
    """hccn_tool -i wants NPU id (0..N-1), which may differ from davinci phy id.

    Try logic first (common when ASCEND maps visible devices 0..N-1), then phy.
    """
    for cand in (logic_id, phy_id):
        if _query_port_cna(cand, 0, 4) is not None:
            return cand
    return logic_id


def _device_eid_addrs(hccn_id: int) -> List[dict]:
    addrs = []
    missing = []
    for port in (4, 5, 6):
        cna = _query_port_cna(hccn_id, 0, port)
        if cna is None:
            missing.append(port)
            continue
        addrs.append(
            {
                "addr_type": "EID",
                "addr": _eid_from_cna(cna),
                "ports": ["0/%d" % port],
                "plane_id": "plane0",
            }
        )
    if missing:
        raise RuntimeError(
            "hccn_tool failed to query CNA for hccn_id=%s ports=%s; "
            "refusing to fabricate EID (set working hccn_tool / driver)"
            % (hccn_id, missing)
        )
    return addrs


def write_ranktable_a5(path: str, logic_ids) -> str:
    """A5 / Ascend 950PR: version 2.0 + EID on net_layer 0 (no host RDMA).

    Notes (cannlab-14 / davinci4-7):
    - ``local_id`` must start from 0 within the server (not equal to davinci phy).
    - ``device_id`` is ACL physical id (davinci N).
    - EID comes from hccn_tool ``-i <npu_id>`` where npu_id is 0..N-1.
    """
    host_ip = host_ipv4()
    rank_list = []
    for rank, logic in enumerate(logic_ids):
        phy = logic_to_phy(logic)
        hccn_id = _hccn_dev_id(logic, phy)
        eid_addrs = _device_eid_addrs(hccn_id)
        rank_list.append(
            {
                "rank_id": rank,
                "device_id": phy,
                "local_id": rank,  # server-local unique id, from 0
                "device_port": 16666 + rank,
                "level_list": [
                    {
                        "net_layer": 0,
                        "net_instance_id": "superpod_0_0",
                        "net_type": "TOPO_FILE_DESC",
                        "net_attr": "",
                        "rank_addr_list": eid_addrs,
                    },
                ],
            }
        )
        print(
            "rank",
            rank,
            "logic",
            logic,
            "phy",
            phy,
            "hccn_id",
            hccn_id,
            "local_id",
            rank,
            "eids",
            [a["addr"] for a in eid_addrs],
            "host",
            host_ip,
            flush=True,
        )
    table = {
        "status": "completed",
        "version": "2.0",
        "rank_count": len(logic_ids),
        "rank_list": rank_list,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)
    return path


def write_ranktable_a3_server_mode(path: str, logic_ids) -> str:
    """A3 typical AI-Server mode (version 1.0), single-server local test.

    Docs: docs/zh/user_guide/cluster_info_config/rank_table_config_a3.md
    Single-host: device_ip optional; multi-host: must fill from /etc/hccn.conf.
    Note: each A3 NPU exposes 2 Die (= 2 Device / 2 ranks).
    """
    host_ip = host_ipv4()
    addrs = parse_hccn_addresses()
    devices = []
    for rank, logic in enumerate(logic_ids):
        phy = logic_to_phy(logic)
        entry = {
            "device_id": str(phy),
            "rank_id": str(rank),
            "device_port": str(16666 + rank),
            "host_port": str(16665 + rank),
        }
        if phy in addrs:
            entry["device_ip"] = addrs[phy]
        devices.append(entry)
        print(
            "rank",
            rank,
            "logic",
            logic,
            "phy",
            phy,
            "device_ip",
            entry.get("device_ip", "(omit)"),
            "host",
            host_ip,
            flush=True,
        )
    table = {
        "status": "completed",
        "version": "1.0",
        "server_count": "1",
        "server_list": [
            {
                "server_id": "node_0",
                "host_ip": host_ip,
                "device": devices,
            }
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)
    return path


def write_ranktable_a2(path: str, logic_ids) -> str:
    """A2 ranktable version 1.0 (same shape as A3 server mode, no host_port required)."""
    host_ip = host_ipv4()
    addrs = parse_hccn_addresses()
    devices = []
    for rank, logic in enumerate(logic_ids):
        phy = logic_to_phy(logic)
        entry = {
            "device_id": str(phy),
            "rank_id": str(rank),
            "device_port": str(16666 + rank),
        }
        if phy in addrs:
            entry["device_ip"] = addrs[phy]
        devices.append(entry)
        print(
            "rank",
            rank,
            "logic",
            logic,
            "phy",
            phy,
            "device_ip",
            entry.get("device_ip", "(omit)"),
            "host",
            host_ip,
            flush=True,
        )
    table = {
        "status": "completed",
        "version": "1.0",
        "server_count": "1",
        "server_list": [{"server_id": "node_0", "device": devices}],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)
    return path


def write_ranktable(path: str, logic_ids, chip: Optional[str] = None) -> Tuple[str, str]:
    """Write chip-appropriate ranktable. Returns (path, chip)."""
    chip = normalize_chip(chip)
    if chip == "a5":
        write_ranktable_a5(path, logic_ids)
    elif chip == "a3":
        write_ranktable_a3_server_mode(path, logic_ids)
    else:
        write_ranktable_a2(path, logic_ids)
    return path, chip


# --- Official dtype / reduce-op skip matrices (from hccl api_ref) ----------------

_REDUCE_OPS = ("all_reduce", "reduce", "reduce_scatter", "reduce_scatter_v")

# A5 AllReduce: int8/16/32/64, uint64, fp16/32/64, bfp16 — no UINT8/16/32
_A5_REDUCE_NO = ("UINT8", "UINT16", "UINT32")
_A5_RSV_NO = ("UINT8", "UINT16", "UINT32", "UINT64", "FP64")

# A3/A2 AllReduce/Reduce/ReduceScatter: int8/16/32/64, fp16/32, bfp16 — no UINT*, no FP64
_A3_REDUCE_NO = ("UINT8", "UINT16", "UINT32", "UINT64", "FP64")
# A3 ReduceScatterV: same dtypes; ops = sum/max/min only (no prod)


def dtype_skip_cases(chip: Optional[str] = None) -> Dict[Tuple[str, str], str]:
    """Unsupported dtype pairs for the detected chip."""
    chip = normalize_chip(chip)
    skips: Dict[Tuple[str, str], str] = {}
    if chip == "a5":
        for op in _REDUCE_OPS:
            for dt in _A5_REDUCE_NO:
                skips[(op, dt)] = "%s %s no %s" % (chip.upper(), op, dt)
        for dt in _A5_RSV_NO:
            skips[("reduce_scatter_v", dt)] = "%s ReduceScatterV no %s" % (chip.upper(), dt)
    else:
        # a2 / a3
        for op in _REDUCE_OPS:
            for dt in _A3_REDUCE_NO:
                skips[(op, dt)] = "%s %s no %s" % (chip.upper(), op, dt)
        if chip == "a2":
            # Official A2 ReduceScatterV: no INT64 (A3 does support INT64).
            skips[("reduce_scatter_v", "INT64")] = "A2 ReduceScatterV no INT64"
    return skips


def reduce_op_skip_cases(chip: Optional[str] = None) -> Dict[Tuple[str, str], str]:
    """Skip (op, reduce_op) pairs not supported on this chip."""
    chip = normalize_chip(chip)
    skips: Dict[Tuple[str, str], str] = {}
    if chip in ("a2", "a3"):
        # A3 ReduceScatterV: sum/max/min only (no prod)
        skips[("reduce_scatter_v", "PROD")] = (
            "%s ReduceScatterV no PROD (official: sum/max/min only)" % chip.upper()
        )
    return skips


def ops_skip_cases() -> Dict[str, str]:
    """Whole-op skip table (currently empty on A5)."""
    return {}


def check_acl(rc, what: str) -> None:
    """Raise RuntimeError on non-zero ACL return."""
    if int(rc) != 0:
        raise RuntimeError("%s failed rc=%s" % (what, rc))


def is_not_supported(exc) -> bool:
    """True if ``exc`` looks like HCCL ``E_NOT_SUPPORT`` / NOT_SUPPORT."""
    msg = str(exc)
    if "NOT_SUPPORT" in msg or "E_NOT_SUPPORT" in msg:
        return True
    try:
        from hccl4py.types import HcclResult

        code = int(HcclResult.E_NOT_SUPPORT)
    except Exception:
        code = 5
    return ("code=%s" % code) in msg or ("HcclResult=%s" % code) in msg


_ACL_READY = False


def bind_acl(acl=None):
    """Bind common ACL ctypes prototypes and return the CDLL."""
    import ctypes

    if acl is None:
        from hccl4py._lib import get_ascendcl

        acl = get_ascendcl()
    acl.aclInit.argtypes = [ctypes.c_char_p]
    acl.aclInit.restype = ctypes.c_int
    acl.aclrtSetDevice.argtypes = [ctypes.c_int32]
    acl.aclrtSetDevice.restype = ctypes.c_int
    acl.aclrtGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
    acl.aclrtGetDeviceCount.restype = ctypes.c_int
    acl.aclrtCreateStream.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    acl.aclrtCreateStream.restype = ctypes.c_int
    acl.aclrtDestroyStream.argtypes = [ctypes.c_void_p]
    acl.aclrtDestroyStream.restype = ctypes.c_int
    acl.aclrtMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int]
    acl.aclrtMalloc.restype = ctypes.c_int
    acl.aclrtFree.argtypes = [ctypes.c_void_p]
    acl.aclrtFree.restype = ctypes.c_int
    acl.aclrtMemcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    acl.aclrtMemcpy.restype = ctypes.c_int
    acl.aclrtSynchronizeStream.argtypes = [ctypes.c_void_p]
    acl.aclrtSynchronizeStream.restype = ctypes.c_int
    acl.aclrtMemset.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int32, ctypes.c_size_t]
    acl.aclrtMemset.restype = ctypes.c_int
    acl.aclrtResetDevice.argtypes = [ctypes.c_int32]
    acl.aclrtResetDevice.restype = ctypes.c_int
    if hasattr(acl, "aclFinalize"):
        acl.aclFinalize.argtypes = []
        acl.aclFinalize.restype = ctypes.c_int
    return acl


def ensure_acl(device: int = 0):
    """Bind ACL, call aclInit once per process, then SetDevice.

    Fail hard if aclInit returns non-zero — do not treat a later SetDevice
    success as proof that initialization succeeded.
    """
    global _ACL_READY
    acl = bind_acl()
    if not _ACL_READY:
        rc = acl.aclInit(None)
        if int(rc) != 0:
            raise RuntimeError(
                "aclInit failed rc=%s; source CANN set_env.sh or set HCCL4PY_ASCENDCL_LIB"
                % rc
            )
        _ACL_READY = True
    check_acl(acl.aclrtSetDevice(device), "aclrtSetDevice(%s)" % device)
    return acl


def safe_ops_skips():
    return ops_skip_cases()


def start_processes(procs):
    """Start processes; on failure kill already-started ones (no false green / hang)."""
    started = []
    try:
        for p in procs:
            p.start()
            started.append(p)
    except Exception:
        for p in started:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
                if p.is_alive() and hasattr(p, "kill"):
                    p.kill()
        raise
    return started


def platform_summary() -> dict:
    chip = detect_chip()
    addrs = parse_hccn_addresses()
    return {
        "chip": chip,
        "host_ip": host_ipv4(),
        "hccn_addresses": addrs,
        "cann_home": os.environ.get("ASCEND_HOME_PATH", ""),
        "hccl4py_chip_env": os.environ.get("HCCL4PY_CHIP", "auto"),
    }


# Ascend 950PR CommConfig encoding: ccu_sched == 6 (lab-verified; header
# enum ordinal 5 is E_NOT_SUPPORT on CANN 9.2.0-beta.2). Same as hccl_test
# accelerator table index for -a ccu_sched.
HCCL_OP_EXPANSION_MODE_CCU_SCHED = 6


def make_test_comm_config(h, **overrides):
    """Build HcclCommConfig; A5 defaults op_expansion_mode to CCU_SCHED(=6).

    ``HCCL4PY_CHIP`` accepts ``a2|a3|a5|auto`` plus aliases ``950``/``atlas350``→a5
    (see ``normalize_chip``).
    """
    kw = dict(overrides)
    mode = getattr(
        h, "HCCL_OP_EXPANSION_MODE_CCU_SCHED", HCCL_OP_EXPANSION_MODE_CCU_SCHED
    )
    try:
        if normalize_chip(os.environ.get("HCCL4PY_CHIP") or "auto") == "a5":
            kw.setdefault("op_expansion_mode", mode)
    except (RuntimeError, ValueError):
        pass
    return h.comm_config_init(**kw)


def mp_collect(q, n: int, timeout: float, procs, join_timeout: float = 5.0):
    """Drain ``n`` queue items, then join / terminate / kill workers."""
    results = []
    try:
        for _ in range(n):
            results.append(q.get(timeout=timeout))
    finally:
        for p in procs:
            p.join(timeout=join_timeout)
            if p.is_alive():
                p.terminate()
                p.join(timeout=join_timeout)
                if p.is_alive() and hasattr(p, "kill"):
                    p.kill()
    return results
