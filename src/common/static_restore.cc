/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* 确保 fileno() 可用 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <securec.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "hccl_common.h"

/* 配置参数 */
#define DEFAULT_BASE_PATH         "/usr/local/Ascend/ascend-toolkit/latest"
#define AICPU_TAR_RELATIVE_PATH   "opp/built_in/op_impl/aicpu/kernel/aicpu_hccl.tar.gz"
#define FILE_PERMISSIONS          0644
#define PATH_BUFFER_SIZE          PATH_MAX
#define MAX_BUFFER_SIZE           (50 * 1024 * 1024)  /* 50MB */

/* 声明嵌入的二进制数据符号 */
extern char _binary_aicpu_hccl_tar_gz_start[];
extern char _binary_aicpu_hccl_tar_gz_end[];

/* CRC32 查找表 */
static uint32_t crc32_table[256];
static int crc32_table_initialized = 0;

/**
 * @brief 初始化 CRC32 查找表
 */
static void init_crc32_table(void) {
    if (crc32_table_initialized) {
        return;
    }

    for (uint32_t i = 0; i < 256; i++) {
        uint32_t crc = i;
        for (int j = 0; j < 8; j++) {
            crc = (crc >> 1) ^ ((crc & 1) ? 0xEDB88320 : 0);
        }
        crc32_table[i] = crc;
    }
    crc32_table_initialized = 1;
}

/**
 * @brief 计算内存数据的 CRC32 校验值
 *
 * @param data 数据指针
 * @param length 数据长度
 * @return uint32_t CRC32 校验值
 */
static uint32_t calc_crc32(const void* data, size_t length) {
    init_crc32_table();

    const uint8_t* bytes = (const uint8_t*)data;
    uint32_t crc = 0xFFFFFFFF;

    for (size_t i = 0; i < length; i++) {
        crc = (crc >> 8) ^ crc32_table[(crc ^ bytes[i]) & 0xFF];
    }

    return crc ^ 0xFFFFFFFF;
}

/**
 * @brief 检查路径是否包含无效字符
 */
static int has_valid_chars(const char* path) {
    const char* whitelist = "abcdefghijklmnopqrstuvwxyz"
                           "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                           "0123456789/_-.";
    for (const char* p = path; *p != '\0'; p++) {
        if (strchr(whitelist, *p) == NULL) {
            HCCL_ERROR("Path contains invalid character at position %ld: '%c'",
                    (long)(p - path), *p);
            return 0;
        }
    }
    return 1;
}

/**
 * @brief 检查路径是否包含穿越序列
 */
static int has_no_traversal(const char* path, size_t len) {
    if (strstr(path, "/../") != NULL) {
        HCCL_ERROR("Path contains traversal sequence '../': '%s'", path);
        return 0;
    }
    if (len >= 3 && strcmp(path + len - 3, "/..") == 0) {
        HCCL_ERROR("Path ends with traversal sequence '/..': '%s'", path);
        return 0;
    }
    if (strstr(path, "/./") != NULL) {
        HCCL_ERROR("Path contains current directory sequence './': '%s'", path);
        return 0;
    }
    if (len >= 3 && strcmp(path + len - 3, "/./") == 0) {
        HCCL_ERROR("Path ends with current directory sequence '/./': '%s'", path);
        return 0;
    }
    return 1;
}

/**
 * @brief 检查路径格式是否合法
 */
static int has_valid_format(const char* path, size_t len) {
    if (strstr(path, "//") != NULL) {
        HCCL_ERROR("Path contains double slash: '%s'", path);
        return 0;
    }
    if (strstr(path, "...") != NULL) {
        HCCL_ERROR("Path contains invalid dot sequence: '%s'", path);
        return 0;
    }
    (void)len;
    return 1;
}

/**
 * @brief 检查路径是否为安全的绝对路径
 *
 * 安全路径定义：
 * 1. 非空
 * 2. 绝对路径（以/开头）
 * 3. 仅包含白名单字符（字母、数字、/_-.）
 * 4. 不包含路径穿越序列（..）
 *
 * @param path 待检查的路径
 * @return int 1 表示安全，0 表示不安全
 */
static int is_safe_path(const char* path) {
    size_t len;
    if (path == NULL || *path == '\0') {
        HCCL_ERROR("Path is empty");
        return 0;
    }
    if (path[0] != '/') {
        HCCL_ERROR("Path must be absolute, got '%s'", path);
        return 0;
    }
    if (!has_valid_chars(path)) {
        return 0;
    }
    len = strlen(path);
    if (!has_no_traversal(path, len)) {
        return 0;
    }
    if (!has_valid_format(path, len)) {
        return 0;
    }
    return 1;
}

/**
 * @brief 安全地复制字符串，确保目标缓冲区以 null 结尾
 *
 * @param dest 目标缓冲区
 * @param dest_size 目标缓冲区大小
 * @param src 源字符串
 * @return int 0 成功，-1 失败（源字符串过长被截断）
 */
static int safe_strcpy(char* dest, size_t dest_size, const char* src) {
    if (dest == NULL || src == NULL || dest_size == 0) {
        return -1;
    }

    size_t src_len = strlen(src);
    if (src_len >= dest_size) {
        /* 源字符串过长，进行截断 */
        if (memcpy_s(dest, dest_size, src, dest_size - 1) != EOK) {
            return -1;
        }
        dest[dest_size - 1] = '\0';
        return -1;
    }

    if (memcpy_s(dest, dest_size, src, src_len + 1) != EOK) {
        return -1;
    }
    return 0;
}

/**
 * @brief 逐级安全打开（按需创建）目录，返回最末一级目录的 fd
 *
 * 从根目录起，按 '/' 切分输入路径的每一个 component，每段执行：
 *   1. mkdirat(dirfd, comp, 0755)，EEXIST 视为已存在
 *   2. fstatat(dirfd, comp, AT_SYMLINK_NOFOLLOW) 校验是普通目录
 *   3. openat(dirfd, comp, O_DIRECTORY|O_NOFOLLOW|O_RDONLY|O_CLOEXEC)
 *      O_NOFOLLOW 在 race 极端情况下也会拒绝跟随 symlink
 *   4. 关闭旧 dirfd，用新 fd 推进到下一级
 *
 * 整条链都基于 dirfd 上的 *at 系列 syscall，不再用路径字符串"check-then-use"，
 * 从根本上消除攻击者通过中间目录 symlink 替换实现的 TOCTOU 跳转。
 *
 * @param path 必须为已通过 is_safe_path 的绝对路径
 * @return int 成功返回最末级目录的 fd（调用者负责 close），失败返回 -1
 */
static int safe_open_dir_chain(const char* path) {
    char buf[PATH_BUFFER_SIZE];
    char* saveptr = NULL;
    int dirfd;

    if (path == NULL || path[0] != '/' ||
        safe_strcpy(buf, sizeof(buf), path) != 0) {
        return -1;
    }
    dirfd = open("/", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (dirfd < 0) {
        HCCL_ERROR("open root failed: %s", strerror(errno));
        return -1;
    }
    for (char* comp = strtok_r(buf, "/", &saveptr); comp != NULL;
         comp = strtok_r(NULL, "/", &saveptr)) {
        struct stat st;
        int next_fd;

        if (mkdirat(dirfd, comp, 0755) == -1 && errno != EEXIST) {
            HCCL_ERROR("mkdirat '%s' failed: %s", comp, strerror(errno));
            close(dirfd); return -1;
        }
        if (fstatat(dirfd, comp, &st, AT_SYMLINK_NOFOLLOW) != 0 ||
            S_ISLNK(st.st_mode) || !S_ISDIR(st.st_mode)) {
            HCCL_ERROR("Component '%s' invalid (symlink or non-dir)", comp);
            close(dirfd); return -1;
        }
        next_fd = openat(dirfd, comp,
                         O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        if (next_fd < 0) {
            HCCL_ERROR("openat '%s' failed: %s", comp, strerror(errno));
            close(dirfd); return -1;
        }
        close(dirfd);
        dirfd = next_fd;
    }
    return dirfd;
}

/**
 * @brief 获取并校验基础路径
 *
 * 优先使用环境变量 ASCEND_HOME_PATH，如果未设置或不安全则使用默认值
 *
 * @return const char* 安全的基础路径（静态缓冲区）
 */
static const char* get_safe_base_path(void) {
    static char safe_path[PATH_BUFFER_SIZE];
    const char* env_path;

    env_path = getenv("ASCEND_HOME_PATH");
    if (env_path != NULL && is_safe_path(env_path)) {
        if (safe_strcpy(safe_path, sizeof(safe_path), env_path) == 0) {
            return safe_path;
        }
    }

    /* 使用默认路径 */
    if (safe_strcpy(safe_path, sizeof(safe_path), DEFAULT_BASE_PATH) != 0) {
        return NULL;
    }

    return safe_path;
}

/**
 * @brief 构建完整的目标文件路径
 *
 * @param base_path 基础路径
 * @param relative_path 相对路径
 * @param output 输出缓冲区
 * @param output_size 输出缓冲区大小
 * @return int 0 成功，-1 失败
 */
static int build_safe_path(const char* base_path, const char* relative_path,
                          char* output, size_t output_size) {
    size_t base_len, rel_len;

    if (base_path == NULL || relative_path == NULL || output == NULL) {
        return -1;
    }

    base_len = strlen(base_path);
    rel_len = strlen(relative_path);
    /* 检查总长度是否溢出 */
    if (base_len + rel_len + 2 > output_size) {
        HCCL_ERROR("Combined path too long");
        return -1;
    }

    /* 安全拼接路径 */
    int ret = snprintf_s(output, output_size, output_size - 1, "%s/%s", base_path, relative_path);
    if (ret < 0) {
        HCCL_ERROR("snprintf_s failed");
        return -1;
    }

    /* 校验生成的路径 */
    if (!is_safe_path(output)) {
        return -1;
    }

    return 0;
}

/**
 * @brief 获取文件描述符并加锁
 */
static FILE* lock_file(FILE* fp, const char* target_path) {
    int fd = fileno(fp);
    if (fd < 0) {
        HCCL_ERROR("Cannot get file descriptor for '%s'", target_path);
        fclose(fp);
        return NULL;
    }

    /* 5 秒超时，每 50ms 重试一次 */
    const int timeout_ms = 5000;
    const int retry_interval_ms = 50;
    const int max_retries = timeout_ms / retry_interval_ms;

    for (int i = 0; i < max_retries; i++) {
        if (flock(fd, LOCK_EX | LOCK_NB) == 0) {
            return fp;  /* 成功获取锁 */
        }
        if (errno != EWOULDBLOCK && errno != EAGAIN) {
            HCCL_ERROR("flock failed on '%s': %s",
                    target_path, strerror(errno));
            fclose(fp);
            return NULL;
        }
        usleep(retry_interval_ms * 1000);
    }

    HCCL_ERROR("Timeout acquiring lock on '%s' after %dms",
            target_path, timeout_ms);
    fclose(fp);
    return NULL;
}

/**
 * @brief 通过父目录 fd 安全地打开并加锁目标文件
 *
 * 关键设计：
 *   1. 把 target_path 拆为 dir + basename，先用 safe_open_dir_chain(dir)
 *      逐级 openat 拿到锚定 inode 的父目录 fd，从而保证
 *      "目录链中任一级被 symlink 替换都不会被跟随"。
 *   2. 再用 openat(parent_fd, basename, O_NOFOLLOW|...) 打开/创建目标文件，
 *      O_NOFOLLOW 让最后一段是 symlink 时直接 ELOOP；
 *      创建分支用 O_CREAT|O_EXCL 防止并发 race 中被 symlink 抢占。
 *   3. 多进程并发时，若另一进程已先创建，本次拿到 EEXIST，
 *      回退用 O_RDWR|O_NOFOLLOW 重开，标记 file_exists=1。
 *
 * @param target_path 已通过 is_safe_path 的绝对路径
 * @param file_exists 输出：1 表示已存在并被打开，0 表示由本调用创建
 * @return FILE* 已加 flock 的文件指针；NULL 表示失败
 */
static FILE* open_target_secure(const char* target_path, int* file_exists) {
    char dir_buf[PATH_BUFFER_SIZE];
    int parent_fd, fd = -1;
    char* slash;
    const char* basename;
    FILE* fp;

    if (file_exists) *file_exists = 0;
    if (target_path == NULL ||
        safe_strcpy(dir_buf, sizeof(dir_buf), target_path) != 0) {
        return NULL;
    }
    slash = strrchr(dir_buf, '/');
    if (slash == NULL) return NULL;
    /* 目标位于根目录下时父目录退化为 "/"，其它情况截断到最后一级目录 */
    if (slash == dir_buf) slash[1] = '\0'; else *slash = '\0';
    basename = strrchr(target_path, '/') + 1;
    if (*basename == '\0') {
        HCCL_ERROR("Target '%s' missing basename", target_path);
        return NULL;
    }
    parent_fd = safe_open_dir_chain(dir_buf);
    if (parent_fd < 0) return NULL;

    fd = openat(parent_fd, basename, O_RDWR | O_NOFOLLOW | O_CLOEXEC);
    if (fd >= 0) {
        if (file_exists) *file_exists = 1;
    } else if (errno == ENOENT) {
        /* O_EXCL 让多进程并发创建只赢一个；EEXIST 时回退按已存在打开 */
        fd = openat(parent_fd, basename,
                    O_CREAT | O_EXCL | O_RDWR | O_NOFOLLOW | O_CLOEXEC, 0644);
        if (fd < 0 && errno == EEXIST) {
            fd = openat(parent_fd, basename, O_RDWR | O_NOFOLLOW | O_CLOEXEC);
            if (fd >= 0 && file_exists) *file_exists = 1;
        }
    }
    close(parent_fd);
    if (fd < 0) {
        HCCL_ERROR("Cannot open target '%s': %s", target_path, strerror(errno));
        return NULL;
    }
    fp = fdopen(fd, "r+b");
    if (fp == NULL) {
        HCCL_ERROR("fdopen failed for '%s': %s", target_path, strerror(errno));
        close(fd);
        return NULL;
    }
    return lock_file(fp, target_path);
}

/**
 * @brief 释放文件锁并关闭文件
 *
 * @param fp 文件指针
 */
static void unlock_and_close(FILE* fp) {
    if (fp != NULL) {
        int fd = fileno(fp);
        if (fd >= 0) {
            flock(fd, LOCK_UN);
        }
        fclose(fp);
    }
}

/**
 * @brief 比较文件内容的 CRC
 *
 * @param buffer 文件内容缓冲区
 * @param expected_data 期望的数据
 * @param expected_size 数据大小
 * @param expected_crc 期望的 CRC 值（0 表示在函数内计算）
 * @return int 1 表示 CRC 匹配，0 表示不匹配
 */
static int compare_crc(const uint8_t* buffer, const void* expected_data,
                       size_t expected_size, uint32_t expected_crc) {
    uint32_t existing_crc = calc_crc32(buffer, expected_size);
    uint32_t computed_crc = (expected_crc != 0) ? expected_crc : calc_crc32(expected_data, expected_size);
    if (existing_crc != computed_crc) {
        HCCL_WARNING("Existing file CRC (0x%08X) differs from embedded (0x%08X), will overwrite",
                existing_crc, computed_crc);
        return 0;
    }
    return 1;
}

/**
 * @brief 读取文件内容到缓冲区
 *
 * @param fp 已打开并加锁的文件指针
 * @param buffer 输出缓冲区
 * @param expected_size 期望读取的大小
 * @param original_pos 输入/输出参数，原始文件位置/恢复后的文件位置
 * @return int 1 表示成功，0 表示失败
 */
static int read_file_content(FILE* fp, uint8_t* buffer, size_t expected_size, long* original_pos) {
    /* 记录当前文件位置 */
    *original_pos = ftell(fp);
    if (*original_pos < 0) {
        return 0;
    }

    /* 回到文件开头读取内容 */
    if (fseek(fp, 0, SEEK_SET) != 0) {
        return 0;
    }

    size_t read_size = fread(buffer, 1, expected_size, fp);
    if (read_size != expected_size) {
        HCCL_ERROR("Failed to read file for comparison: read %zu, expected %zu",
                read_size, expected_size);
        return 0;
    }

    /* 恢复文件位置 */
    if (fseek(fp, *original_pos, SEEK_SET) != 0) {
        return 0;
    }

    return 1;
}

/**
 * @brief 检查并比较文件 CRC
 *
 * @param fp 已打开并加锁的文件指针
 * @param file_path 文件路径（用于错误信息）
 * @param expected_data 期望的数据
 * @param expected_size 期望的数据大小
 * @param expected_crc 期望的 CRC 值（0 表示不传入，函数内计算）
 * @return int 1 表示文件存在且 CRC 匹配，0 表示不匹配或文件不存在
 */
static int check_file_integrity(FILE* fp, const char* file_path, const void* expected_data,
                                size_t expected_size, uint32_t expected_crc) {
    struct stat st;
    uint8_t* buffer = NULL;
    int fd;
    long original_pos;
    int result;

    /* 使用已加锁的文件指针获取文件状态 */
    fd = fileno(fp);
    if (fd < 0 || fstat(fd, &st) != 0) {
        return 0;
    }

    /* 检查文件大小 */
    if ((size_t)st.st_size != expected_size) {
        HCCL_WARNING("Existing file size (%ld) differs from embedded (%zu), will overwrite",
                (long)st.st_size, expected_size);
        return 0;
    }

    /* 分配缓冲区 */
    if (expected_size > MAX_BUFFER_SIZE) {
        HCCL_ERROR("Buffer size %zu exceeds maximum allowed size (50MB)", expected_size);
        return 0;
    }
    buffer = (uint8_t*)malloc(expected_size);
    if (buffer == NULL) {
        HCCL_ERROR("Memory allocation failed for integrity check");
        return 0;
    }

    /* 读取文件内容 */
    if (!read_file_content(fp, buffer, expected_size, &original_pos)) {
        free(buffer);
        return 0;
    }

    /* 比较 CRC */
    result = compare_crc(buffer, expected_data, expected_size, expected_crc);
    free(buffer);

    return result;
}

/**
 * @brief 删除写入失败的不完整文件
 */
static void remove_incomplete_file(const char* path) {
    if (unlink(path) != 0) {
        HCCL_WARNING("Failed to remove incomplete file '%s': %s",
                path, strerror(errno));
    }
}

/**
 * @brief 解析目标路径
 *
 * 目录创建已转移到 open_target_secure() 内通过 safe_open_dir_chain
 * 用 fd 链方式安全完成，本函数只做字符串拼接 + 校验。
 *
 * @param target_path 输出缓冲区
 * @param path_size 缓冲区大小
 * @return int 0 成功，-1 失败
 */
static int resolve_target_path(char* target_path, size_t path_size) {
    const char* base_path = get_safe_base_path();
    if (base_path == NULL) {
        HCCL_WARNING("Failed to get safe base path, using default");
        base_path = DEFAULT_BASE_PATH;
    }

    if (build_safe_path(base_path, AICPU_TAR_RELATIVE_PATH, target_path, path_size) != 0) {
        HCCL_ERROR("Failed to build safe target path");
        return -1;
    }

    return 0;
}

/**
 * @brief 将嵌入数据写入已锁定的文件并验证
 *
 * @param fp 已打开并加锁的文件指针
 * @param target_path 目标文件路径（用于错误信息和清理）
 * @param data 待写入的数据
 * @param size 数据大小
 * @param expected_crc 期望的 CRC32 校验值
 * @param file_exists 文件是否已存在（决定是否需要 truncate）
 * @return int 0 成功，-1 失败（失败时文件已被清理）
 */
static int write_and_verify_tar(FILE* fp, const char* target_path,
                                const char* data, size_t size,
                                uint32_t expected_crc, int file_exists) {
    if (fseek(fp, 0, SEEK_SET) != 0) {
        HCCL_ERROR("Cannot seek to beginning of '%s'", target_path);
        return -1;
    }

    if (file_exists && ftruncate(fileno(fp), 0) != 0) {
        HCCL_ERROR("Cannot truncate '%s': %s", target_path, strerror(errno));
        return -1;
    }

    size_t written = fwrite(data, 1, size, fp);
    if (written != size) {
        HCCL_ERROR("Failed to write tar file: expected %zu bytes, wrote %zu",
                size, written);
        fclose(fp);
        remove_incomplete_file(target_path);
        return -1;
    }

    if (fflush(fp) != 0) {
        HCCL_ERROR("Failed to flush file '%s': %s", target_path, strerror(errno));
        fclose(fp);
        remove_incomplete_file(target_path);
        return -1;
    }

    uint32_t written_crc = calc_crc32(data, size);
    if (written_crc != expected_crc) {
        HCCL_ERROR("Post-write CRC check failed: 0x%08X vs expected 0x%08X",
                written_crc, expected_crc);
        fclose(fp);
        remove_incomplete_file(target_path);
        return -1;
    }

    return 0;
}

/**
 * @brief 恢复 AICPU tar 包的构造函数
 *
 * 在程序启动时自动执行，将嵌入的 AICPU tar 包恢复到文件系统。
 * 使用文件锁确保多进程场景下的安全性，并在恢复前进行完整性校验。
 */
__attribute__((constructor))
static void restore_aicpu_tar(void) {
    char target_path[PATH_MAX];
    int file_exists = 0;

    size_t tar_size = (size_t)(_binary_aicpu_hccl_tar_gz_end - _binary_aicpu_hccl_tar_gz_start);
    if (tar_size == 0) {
        HCCL_WARNING("No embedded AICPU tar package found, skipping restore.");
        return;
    }

    uint32_t embedded_crc = calc_crc32(_binary_aicpu_hccl_tar_gz_start, tar_size);

    if (resolve_target_path(target_path, sizeof(target_path)) != 0) {
        return;
    }

    FILE* fp = open_target_secure(target_path, &file_exists);
    if (fp == NULL) {
        HCCL_ERROR("Failed to acquire file lock on '%s'. Aborting restore.",
                target_path);
        return;
    }

    if (file_exists && check_file_integrity(fp, target_path, _binary_aicpu_hccl_tar_gz_start,
                                            tar_size, embedded_crc)) {
        unlock_and_close(fp);
        return;
    }

    if (write_and_verify_tar(fp, target_path, _binary_aicpu_hccl_tar_gz_start,
                             tar_size, embedded_crc, file_exists) != 0) {
        return;
    }

    /* 用 fchmod 直接对 fd 操作，绕过路径解析，race-free */
    if (fchmod(fileno(fp), FILE_PERMISSIONS) != 0) {
        HCCL_WARNING("Failed to set permissions on %s: %s",
                target_path, strerror(errno));
    }

    unlock_and_close(fp);

    HCCL_INFO("AICPU tar package restored to: %s (%zu bytes, CRC: 0x%08X)",
            target_path, tar_size, embedded_crc);
}
