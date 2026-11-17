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

"""Day-1 probe for A2/A3/A5 boxes — print everything needed before running suites.

Usage (on the NPU host, after source set_env.sh):
  python3 tests/probe_platform.py
  HCCL4PY_CHIP=a3 python3 tests/probe_platform.py --write-ranktable /tmp/rt.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.environ.get(
    "HCCL4PY_PYTHON",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _run(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=30)
    except Exception as e:
        return "ERR: %s" % e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-ranktable", metavar="PATH", help="emit chip-matched ranktable JSON")
    ap.add_argument("--nranks", type=int, default=2)
    args = ap.parse_args()

    from _platform import (
        dtype_skip_cases,
        platform_summary,
        reduce_op_skip_cases,
        write_ranktable,
    )

    print("=== env ===")
    for k in (
        "ASCEND_HOME_PATH",
        "ASCEND_AICPU_PATH",
        "LD_LIBRARY_PATH",
        "HCCL4PY_CHIP",
        "HCCL4PY_PYTHON",
    ):
        v = os.environ.get(k, "")
        if k == "LD_LIBRARY_PATH" and len(v) > 200:
            v = v[:200] + "..."
        print("%s=%s" % (k, v))

    print("\n=== npu-smi info -l ===")
    print(_run(["npu-smi", "info", "-l"]))
    print("=== npu-smi info (head) ===")
    full = _run(["npu-smi", "info"])
    print("\n".join(full.splitlines()[:40]))

    print("\n=== /etc/hccn.conf (address_*) ===")
    try:
        with open("/etc/hccn.conf", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip().startswith("address_"):
                    print(line.rstrip())
    except OSError as e:
        print("missing:", e)

    print("\n=== cann toolkit version ===")
    home = os.environ.get("ASCEND_HOME_PATH", "")
    import glob as _glob
    import platform as _platform

    arch = _platform.machine()
    arch_dir = "aarch64-linux" if arch in ("aarch64", "arm64") else "x86_64-linux"
    candidates = []
    if home:
        candidates.append(os.path.join(home, arch_dir, "ascend_toolkit_install.info"))
        candidates.extend(
            _glob.glob(os.path.join(home, "*-linux", "ascend_toolkit_install.info"))
        )
    info = next((p for p in candidates if os.path.isfile(p)), "")
    if info:
        print("path", info)
        print(_run(["grep", "-E", "^(version|package_version)=", info]))
    else:
        print("no install.info under", home or "(ASCEND_HOME_PATH unset)")

    # ACL device count (needs set_env)
    print("\n=== ACL device count ===")
    n = -1
    try:
        import ctypes
        from _platform import bind_acl, check_acl, logic_to_phy

        acl = bind_acl()
        rc = acl.aclInit(None)
        print("aclInit", rc)
        check_acl(rc, "aclInit")
        check_acl(acl.aclrtSetDevice(0), "aclrtSetDevice(0)")
        cnt = ctypes.c_uint32()
        check_acl(acl.aclrtGetDeviceCount(ctypes.byref(cnt)), "aclrtGetDeviceCount")
        n = int(cnt.value)
        print("device_count=%d" % n)

        for i in range(min(n, 8)):
            print("  logic=%d phy=%d" % (i, logic_to_phy(i)))
    except Exception as e:
        print("ACL probe failed:", e)

    chip = None
    try:
        summary = platform_summary()
        chip = summary.get("chip")
    except RuntimeError as e:
        print("WARN: detect_chip failed:", e)
        from _platform import host_ipv4, parse_hccn_addresses

        summary = {
            "chip": None,
            "host_ip": host_ipv4(),
            "hccn_addresses": parse_hccn_addresses(),
            "cann_home": os.environ.get("ASCEND_HOME_PATH", ""),
            "hccl4py_chip_env": os.environ.get("HCCL4PY_CHIP", "auto"),
            "detect_error": str(e),
        }
    print("\n=== detected ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if chip:
        print("dtype_pre_skips=%d" % len(dtype_skip_cases(chip)))
        print("reduce_op_pre_skips=%d" % len(reduce_op_skip_cases(chip)))
        for (op, dt), reason in sorted(dtype_skip_cases(chip).items())[:12]:
            print("  SKIP dtype", op, dt, "->", reason)
        for (op, rop), reason in sorted(reduce_op_skip_cases(chip).items()):
            print("  SKIP reduce", op, rop, "->", reason)
    else:
        print("dtype/reduce matrices skipped (chip unknown)")

    print("\n=== A3 day-1 checklist ===")
    checks = [
        ("CANN sourced", bool(home)),
        ("device_count >= 2", n >= 2),
        ("chip forced or detected", chip in ("a2", "a3", "a5")),
        ("hccn address_* present (multi-host A3 needs them)", bool(summary.get("hccn_addresses"))),
    ]
    for name, ok in checks:
        print("[%s] %s" % ("OK" if ok else "WARN", name))

    print("\nNext:")
    if chip:
        print("  export HCCL4PY_CHIP=%s   # pin if auto wrong" % chip)
    else:
        print("  export HCCL4PY_CHIP=a2|a3|a5   # required when auto-detect fails")
    print("  python3 tests/run_ops_suite.py")
    print("  python3 tests/run_cluster_suite.py")
    print("  python3 tests/run_dtype_suite.py")
    print("  python3 tests/run_reduce_op_suite.py")
    print("  python3 tests/run_perf_bench.py")

    if args.write_ranktable:
        if chip is None:
            print("cannot write ranktable: chip unknown (set HCCL4PY_CHIP)")
            return 2
        if n < args.nranks:
            print("cannot write ranktable: need %d devices, got %d" % (args.nranks, n))
            return 2
        write_ranktable(args.write_ranktable, list(range(args.nranks)), chip=chip)
        print("wrote", args.write_ranktable)
        with open(args.write_ranktable, encoding="utf-8") as f:
            print(f.read())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
