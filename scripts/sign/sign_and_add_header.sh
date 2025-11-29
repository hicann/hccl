#!/bin/bash
set -e

function usage() {
    echo "Usage: $0 <file_to_sign> <sign_config_file>"
    echo "Example: $0 /path/to/file.bin /path/to/sign_config.xml"
}

function check_file_absolute_and_exist() {
    local file_path="$1"
    local file_desc="${2:-file}"

    # Check if it's an absolute path (starts with /)
    if [[ ! "$file_path" =~ ^/ ]]; then
        echo "Error: $file_desc must use an absolute path: $file_path"
        exit 1
    fi

    # Check if the file exists and is a regular file
    if [ ! -f "$file_path" ]; then
        echo "Error: $file_desc does not exist or is not a regular file: $file_path"
        exit 1
    fi
}

# Execute signing operation
function execute_sign() {
    local file_to_sign="$1"
    local sign_config_file="$2"
    local sign_flag="$3"
    local script="$4"
    local version="$5"

    local delivery_dir=${ROOT_DIR}/delivery/lib/device
    # local delivery_dir=${ROOT_DIR}/vendor/hisi/build/delivery/onetrack/lib/device
    local filename=$(basename ${file_to_sign})
    local temp_dir=${delivery_dir}/temp_sign_${filename}

    if [ -d "${temp_dir}" ];then
        rm -rf ${temp_dir}
    fi

    mkdir -p ${temp_dir}

    cp -rf ${file_to_sign} ${temp_dir}

    python3 ${script} ${temp_dir} ${ROOT_DIR} onetrack ascend ${sign_flag} --sign_certificate=atlas --bios_check_cfg=${sign_config_file} --version=${version}

    local sign_exit_code=$?
    if [ "$sign_exit_code" -ne 0 ]; then
        echo "Error: Signing process failed (exit code: $sign_exit_code)"
        exit 1
    fi

    cp -rf ${temp_dir}/${filename} ${file_to_sign}
}

# Main workflow

# Check number of arguments
if [ $# -lt 2 ]; then
    echo "Error: Incorrect number of arguments"
    usage
    exit 1
fi

# Parse input arguments
FILE_TO_SIGN="$1"
SIGN_CONFIG_FILE="$2"
SIGN_FLAG=${3:-false}
SCRIPT="$4"
VERSION="$5"

CURRENT_DIR=$(dirname $(readlink -f ${BASH_SOURCE[0]}))
# ROOT_DIR=$(cd ${CURRENT_DIR}/../../../; pwd)
ROOT_DIR=$(cd ${CURRENT_DIR}/../../; pwd)
echo "${ROOT_DIR}"
echo "SCRIPT:${SCRIPT}"
echo "VERSION:${VERSION}"

# Validate input files
check_file_absolute_and_exist "$FILE_TO_SIGN" "file to sign"
check_file_absolute_and_exist "$SIGN_CONFIG_FILE" "signing configuration file"

# Execute signing
execute_sign "$FILE_TO_SIGN" "$SIGN_CONFIG_FILE" "${SIGN_FLAG}" "${SCRIPT}" "${VERSION}"

exit 0
