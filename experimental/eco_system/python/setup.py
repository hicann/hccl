# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Wheel packaging entry for hccl4py."""

import re
from pathlib import Path

from setuptools import setup


def _version() -> str:
    init = Path(__file__).resolve().parent / "hccl4py" / "__init__.py"
    m = re.search(r'^__version__\s*=\s*[\'"]([^\'"]+)[\'"]', init.read_text(encoding="utf-8"), re.M)
    if not m:
        raise RuntimeError(f"__version__ not found in {init}")
    return m.group(1)


setup(
    name="hccl4py",
    version=_version(),
    description="HCCL collective/P2P Python bindings (ctypes); requires hcomm4py",
    packages=["hccl4py"],
    python_requires=">=3.8",
    # hcomm4py is delivered from cann/hcomm (same experimental tree). Install that
    # wheel first, or put hcomm/.../python on PYTHONPATH — not published on PyPI.
    install_requires=[],
)
