#!/bin/bash

sourcedir="${INSTALL_PATH}"
whl_source_dir="${sourcedir}/ops_hccl/es_packages/whl"

mkdir -p "${sourcedir}/python/site-packages"
chmod 755 "${sourcedir}/python"
chmod 755 "${sourcedir}/python/site-packages"

mkdir -p "${sourcedir}/opp/built-in/op_impl/aicpu/kernel"
chmod 555 "${sourcedir}/opp/built-in/op_impl/aicpu/kernel"

unset PYTHONPATH
export PIP_BREAK_SYSTEM_PACKAGES=1

run_pip()
{
    if python3 -m pip --version >/dev/null 2>&1; then
        python3 -m pip "$@"
    elif command -v pip3 >/dev/null 2>&1; then
        pip3 "$@"
    else
        return 127
    fi
}

whl_file=$(find "${whl_source_dir}" -maxdepth 1 \
    -name "es_hccl-*.whl" -type f 2>/dev/null | head -n 1)

if [ -n "${whl_file}" ]; then

    if ! run_pip install \
        --disable-pip-version-check \
        --upgrade \
        --no-deps \
        --force-reinstall \
        "${whl_file}"; then
        echo "[hccl] failed to install ${whl_file}"
        exit 1
    fi

    rm -rf "${sourcedir}/ops_hccl"
else
    echo "[hccl] no es_hccl wheel found in ${whl_source_dir}, skip installation"
fi