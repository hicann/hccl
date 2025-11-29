CURRENT_DIR=$(dirname $(readlink -f ${BASH_SOURCE[0]}))
BUILD_DIR=${CURRENT_DIR}/../build/

# 设置使用的device
if [ -f "/root/set_davinci.txt" ]; then
    # 读取文件内容并去除换行符
    DAVINCI_VALUE=$(cat /root/set_davinci.txt | tr -d '\n')
    export ASCEND_RT_VISIBLE_DEVICES="$DAVINCI_VALUE"
    echo "ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"
fi

for dir in ${CURRENT_DIR}/*/*/;do
    # 检查是否是需要跳过的目录
    if [ "$dir" = "${CURRENT_DIR}/02_collectives/09_scatter/" ]; then
        echo "Skipping directory: $dir" | tee -a ${BUILD_DIR}/build.log
        continue
    fi

    # 进入子目录
    if ! cd "$dir"; then
        echo "Failed to enter directory: $dir" | tee -a ${BUILD_DIR}/build.log
        continue
    fi

    # 检查是否存在Makefile
    if [ -f Makefile ]; then
        echo "Processing directory: $dir" | tee -a ${BUILD_DIR}/build.log
        # 执行make和make test，并将输出记录到build.log
        make ${JOB_NUM} && echo "Make Success" || echo "Make Failure" | tee -a ${BUILD_DIR}/build.log
        if grep -q "Make Failure" ${BUILD_DIR}/build.log; then
            echo "Processing directory: $dir .. make failed" | tee -a ${BUILD_DIR}/build.log
            break
        fi

        make test && echo "Make test Success" || echo "Make test Failure" | tee -a ${BUILD_DIR}/build.log
        if grep -q "Make test Failure" ${BUILD_DIR}/build.log; then
            echo "Processing directory: $dir .. make test failed" | tee -a ${BUILD_DIR}/build.log
            break
        fi
    else
        echo "No Makefile found in directory: $dir" | tee -a ${BUILD_DIR}/build.log
    fi
    
    # 返回上级目录
    cd ..
done