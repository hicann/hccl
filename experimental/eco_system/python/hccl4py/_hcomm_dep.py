# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Make ``hcomm4py`` importable when checked out beside ``hccl``."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional


def ensure_hcomm4py() -> None:
    last_exc: Optional[BaseException] = None
    try:
        import hcomm4py  # noqa: F401

        return
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", None) not in (None, "hcomm4py"):
            raise
        last_exc = exc
    except ImportError as exc:
        # Installed but broken internal import — surface the real cause.
        raise ImportError(
            "hccl4py requires hcomm4py, but importing it failed (package present, "
            "inner import error). See chained exception."
        ) from exc

    here = Path(__file__).resolve()
    candidates = [
        here.parents[5] / "hcomm" / "experimental" / "eco_system" / "python",
    ]
    env_root = (os.environ.get("HCOMM4PY_PYTHON_PATH") or "").strip()
    if env_root:
        candidates.append(Path(env_root))
    for root in candidates:
        if not root.is_dir():
            continue
        root_s = str(root)
        if root_s not in sys.path:
            sys.path.insert(0, root_s)
        try:
            import hcomm4py  # noqa: F401

            return
        except ModuleNotFoundError as exc:
            if getattr(exc, "name", None) not in (None, "hcomm4py"):
                raise
            last_exc = exc
            continue
        except ImportError as exc:
            raise ImportError(
                "hccl4py found a hcomm4py candidate at %s but import failed; "
                "see chained exception." % root_s
            ) from exc

    raise ImportError(
        "hccl4py requires hcomm4py (cann/hcomm communication-domain package). "
        "Install hcomm4py or put hcomm/experimental/eco_system/python on PYTHONPATH "
        "(or set HCOMM4PY_PYTHON_PATH)."
    ) from last_exc
