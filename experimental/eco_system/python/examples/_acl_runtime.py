# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""ACL CDLL helper for official examples (avoids internal ``hccl4py._lib``).

Search order matches PUBLIC_API / ``_lib`` docs:
  HCCL4PY_ASCENDCL_LIB → HCCL4PY_LIB_DIR / ASCEND_* → LD_LIBRARY_PATH → soname.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from typing import Iterable, Optional

_ACL = None
_ACL_READY = False


def _split_paths(value: Optional[str]) -> Iterable[str]:
    if not value:
        return
    for part in value.split(os.pathsep):
        part = part.strip()
        if part:
            yield part


def _load_ascendcl() -> ctypes.CDLL:
    explicit = (os.environ.get("HCCL4PY_ASCENDCL_LIB") or "").strip()
    if explicit:
        return ctypes.CDLL(explicit, mode=ctypes.RTLD_GLOBAL)
    roots = []
    for key in ("HCCL4PY_LIB_DIR", "ASCEND_LIBRARY_PATH"):
        roots.extend(_split_paths(os.environ.get(key)))
    for key in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        base = os.environ.get(key)
        if not base:
            continue
        roots.append(os.path.join(base, "lib64"))
        roots.append(os.path.join(base, "lib"))
    roots.extend(_split_paths(os.environ.get("LD_LIBRARY_PATH")))
    tried = []
    last = None
    for root in roots:
        path = os.path.join(root, "libascendcl.so")
        tried.append(path)
        try:
            return ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        except OSError as e:
            last = e
    found = ctypes.util.find_library("ascendcl")
    if found:
        tried.append(found)
        try:
            return ctypes.CDLL(found, mode=ctypes.RTLD_GLOBAL)
        except OSError as e:
            last = e
    tried.append("libascendcl.so")
    try:
        return ctypes.CDLL("libascendcl.so", mode=ctypes.RTLD_GLOBAL)
    except OSError as e:
        last = e
    raise OSError(
        "cannot load libascendcl.so: %s (tried=%s). "
        "Source CANN set_env.sh or set HCCL4PY_ASCENDCL_LIB." % (last, tried)
    )


def bind_acl(acl=None):
    """Bind common ACL prototypes used by examples."""
    global _ACL
    if acl is None:
        if _ACL is None:
            _ACL = _load_ascendcl()
        acl = _ACL
    acl.aclInit.argtypes = [ctypes.c_char_p]
    acl.aclInit.restype = ctypes.c_int
    acl.aclrtSetDevice.argtypes = [ctypes.c_int32]
    acl.aclrtSetDevice.restype = ctypes.c_int
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
    return acl


def ensure_acl(device: int = 0):
    """aclInit once per process (fail hard on non-zero), then SetDevice."""
    global _ACL_READY
    acl = bind_acl()
    if not _ACL_READY:
        rc = int(acl.aclInit(None))
        if rc != 0:
            raise RuntimeError(
                "aclInit failed rc=%s; source CANN set_env.sh or set HCCL4PY_ASCENDCL_LIB"
                % rc
            )
        _ACL_READY = True
    rc = int(acl.aclrtSetDevice(device))
    if rc != 0:
        raise RuntimeError("aclrtSetDevice(%s) failed rc=%s" % (device, rc))
    return acl
