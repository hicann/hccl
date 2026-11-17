/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * Native C++ AllReduce microbench (2 ranks, two processes via re-exec).
 *
 * Build (headers need C++):
 *   source /usr/local/Ascend/ascend-toolkit/set_env.sh
 *   g++ -O2 -std=c++14 -o c_allreduce_bench c_allreduce_bench.c \
 *     -I$ASCEND_HOME_PATH/include -L$ASCEND_HOME_PATH/lib64 \
 *     -lhccl -lhcomm -lascendcl -Wl,-rpath,$ASCEND_HOME_PATH/lib64
 *
 * Run: ./c_allreduce_bench [iters=200] [count=8] [warmup=20]
 *
 * Avoids fork-after-aclInit (hangs on some CANN builds). Parent writes
 * rootinfo to a temp file and re-execs itself as rank worker.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cerrno>
#include <string>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

static double now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

static int run_rank(int rank, int nranks, const char* root_path, int warmup, int iters, int count)
{
    FILE* fp = fopen(root_path, "rb");
    if (!fp) {
        perror("fopen root");
        return 1;
    }
    HcclRootInfo root;
    if (fread(&root, 1, sizeof(root), fp) != sizeof(root)) {
        fprintf(stderr, "read rootinfo fail\n");
        fclose(fp);
        return 1;
    }
    fclose(fp);

    if (aclInit(NULL) != ACL_SUCCESS) {
        fprintf(stderr, "aclInit fail rank=%d\n", rank);
        return 1;
    }
    if (aclrtSetDevice(rank) != ACL_SUCCESS) {
        fprintf(stderr, "setDevice fail rank=%d\n", rank);
        return 1;
    }
    aclrtStream stream = NULL;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
        fprintf(stderr, "createStream fail\n");
        return 1;
    }

    HcclCommConfig cfg;
    HcclCommConfigInit(&cfg);
    HcclComm comm = NULL;
    HcclResult hr = HcclCommInitRootInfoConfig(nranks, &root, rank, &cfg, &comm);
    if (hr != HCCL_SUCCESS) {
        fprintf(stderr, "commInit rank=%d hr=%d\n", rank, (int)hr);
        return 1;
    }

    size_t nbytes = (size_t)count * sizeof(float);
    float* host = (float*)malloc(nbytes);
    if (!host) {
        fprintf(stderr, "malloc host %zu bytes fail\n", nbytes);
        HcclCommDestroy(comm);
        return 1;
    }
    for (int i = 0; i < count; ++i)
        host[i] = (float)(rank + 1);
    void *send_buf = NULL, *recv_buf = NULL;
    aclError ae = aclrtMalloc(&send_buf, nbytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ae != ACL_SUCCESS) {
        fprintf(stderr, "aclrtMalloc send_buf fail %d\n", (int)ae);
        free(host);
        HcclCommDestroy(comm);
        return 1;
    }
    ae = aclrtMalloc(&recv_buf, nbytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ae != ACL_SUCCESS) {
        fprintf(stderr, "aclrtMalloc recv_buf fail %d\n", (int)ae);
        aclrtFree(send_buf);
        free(host);
        HcclCommDestroy(comm);
        return 1;
    }
    ae = aclrtMemcpy(send_buf, nbytes, host, nbytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ae != ACL_SUCCESS) {
        fprintf(stderr, "aclrtMemcpy H2D fail %d\n", (int)ae);
        aclrtFree(send_buf);
        aclrtFree(recv_buf);
        free(host);
        HcclCommDestroy(comm);
        return 1;
    }

    for (int i = 0; i < warmup; ++i) {
        hr = HcclAllReduce(send_buf, recv_buf, (uint64_t)count, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, comm, stream);
        if (hr != HCCL_SUCCESS) {
            fprintf(stderr, "warmup ar %d\n", (int)hr);
            return 1;
        }
    }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS)
        return 1;

    double t0 = now_us();
    for (int i = 0; i < iters; ++i) {
        hr = HcclAllReduce(send_buf, recv_buf, (uint64_t)count, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, comm, stream);
        if (hr != HCCL_SUCCESS) {
            fprintf(stderr, "ar %d\n", (int)hr);
            return 1;
        }
    }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS)
        return 1;
    double t1 = now_us();
    double per = (t1 - t0) / (double)iters;
    printf("[C_NATIVE] rank=%d mean_us=%.3f iters=%d count=%d\n", rank, per, iters, count);
    fflush(stdout);

    HcclCommDestroy(comm);
    free(host);
    return 0;
}

int main(int argc, char** argv)
{
    // Worker mode: ./c_allreduce_bench --rank R NRANKS ROOTPATH iters count warmup
    if (argc >= 2 && strcmp(argv[1], "--rank") == 0) {
        if (argc < 8) {
            fprintf(stderr, "bad --rank args\n");
            return 1;
        }
        int rank = atoi(argv[2]);
        int nranks = atoi(argv[3]);
        const char* root_path = argv[4];
        int iters = atoi(argv[5]);
        int count = atoi(argv[6]);
        int warmup = atoi(argv[7]);
        return run_rank(rank, nranks, root_path, warmup, iters, count);
    }

    int iters = argc > 1 ? atoi(argv[1]) : 200;
    int count = argc > 2 ? atoi(argv[2]) : 8;
    int warmup = argc > 3 ? atoi(argv[3]) : 20;

    if (aclInit(NULL) != ACL_SUCCESS) {
        fprintf(stderr, "aclInit fail\n");
        return 1;
    }
    uint32_t ndev = 0;
    if (aclrtGetDeviceCount(&ndev) != ACL_SUCCESS || ndev < 2) {
        fprintf(stderr, "need >=2 devices, got %u\n", ndev);
        return 1;
    }
    // Always 2-rank microbench (only two workers are spawned below).
    // Using full device_count>2 would hang waiting for missing ranks.
    int nranks = 2;
    if (aclrtSetDevice(0) != ACL_SUCCESS)
        return 1;

    HcclRootInfo root;
    if (HcclGetRootInfo(&root) != HCCL_SUCCESS) {
        fprintf(stderr, "getRootInfo fail\n");
        return 1;
    }

    char root_path[] = "/tmp/hccl4py_rootinfo_XXXXXX";
    int fd = mkstemp(root_path);
    if (fd < 0) {
        perror("mkstemp");
        return 1;
    }
    FILE* fp = fdopen(fd, "wb");
    if (!fp || fwrite(&root, 1, sizeof(root), fp) != sizeof(root)) {
        fprintf(stderr, "write rootinfo fail\n");
        return 1;
    }
    fclose(fp);

    char self[512];
    ssize_t n = readlink("/proc/self/exe", self, sizeof(self) - 1);
    if (n < 0) {
        perror("readlink");
        return 1;
    }
    self[n] = '\0';

    char rank0[16], rank1[16], nr_s[16], it_s[16], ct_s[16], wu_s[16];
    snprintf(rank0, sizeof(rank0), "%d", 0);
    snprintf(rank1, sizeof(rank1), "%d", 1);
    snprintf(nr_s, sizeof(nr_s), "%d", nranks);
    snprintf(it_s, sizeof(it_s), "%d", iters);
    snprintf(ct_s, sizeof(ct_s), "%d", count);
    snprintf(wu_s, sizeof(wu_s), "%d", warmup);

    pid_t p0 = fork();
    if (p0 < 0) {
        perror("fork r0");
        unlink(root_path);
        return 1;
    }
    if (p0 == 0) {
        execl(self, self, "--rank", rank0, nr_s, root_path, it_s, ct_s, wu_s, (char*)NULL);
        perror("execl r0");
        _exit(127);
    }
    pid_t p1 = fork();
    if (p1 < 0) {
        perror("fork r1");
        kill(p0, SIGKILL); /* avoid orphan worker hanging in HCCL init */
        while (waitpid(p0, NULL, 0) < 0 && errno == EINTR) {
        }
        unlink(root_path);
        return 1;
    }
    if (p1 == 0) {
        execl(self, self, "--rank", rank1, nr_s, root_path, it_s, ct_s, wu_s, (char*)NULL);
        perror("execl r1");
        _exit(127);
    }

    int st0 = 0, st1 = 0;
    while (waitpid(p0, &st0, 0) < 0) {
        if (errno == EINTR) {
            continue;
        }
        perror("waitpid r0");
        kill(p1, SIGKILL);
        while (waitpid(p1, NULL, 0) < 0 && errno == EINTR) {
        }
        unlink(root_path);
        return 1;
    }
    while (waitpid(p1, &st1, 0) < 0) {
        if (errno == EINTR) {
            continue;
        }
        perror("waitpid r1");
        unlink(root_path);
        return 1;
    }
    unlink(root_path);
    int rc0 = WIFEXITED(st0) ? WEXITSTATUS(st0) : 1;
    int rc1 = WIFEXITED(st1) ? WEXITSTATUS(st1) : 1;
    if (rc0 | rc1) {
        fprintf(stderr, "workers failed rc0=%d rc1=%d\n", rc0, rc1);
    }
    return rc0 | rc1;
}
