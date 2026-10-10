# HCCL Header Files and Library Files

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-12T08:36:28.619Z pushedAt=2026-09-18T08:03:58.150Z -->

Based on Ascend AI Processors, Huawei Collective Communication Library (HCCL) provides high-performance collective communication and point-to-point communication capabilities in single-node and multi-node environments. It is one of the core components of CANN. HCCL is decoupled from the underlying communication library HCOMM through dlsym dynamic loading. HCCL and HCOMM are compiled independently and evolve with independent versions. For HCOMM external header files and library files, see [HCOMM External Header Files and Library Files](https://gitcode.com/cann/hcomm/blob/master/docs/en/api_ref/hcomm_header_and_lib.md).

This section describes the header files and library files of HCCL external APIs.

## API Categories

HCCL external APIs are classified into the following categories by function:

**Table 1** API categories

| API Category | Description |
| --- | --- |
| Collective communication operator | A total of 11 collective communication APIs: AllReduce, Broadcast, AllGather, AllGatherV, ReduceScatter, ReduceScatterV, Scatter, Reduce, AlltoAll, AlltoAllV, and AlltoAllVC. |
| Point-to-point communication operator | A total of 3 point-to-point communication APIs: Send, Recv, and BatchSendRecv. |
| MC2 custom operator | MC2 (Kernel Fusion Custom) custom operator framework APIs, such as HcclKfc\* parameter objects and HcclCreateOpResCtx. |

## Header Files and Library Files Required for Calling APIs

After installing the firmware, driver, and CANN software package, you can reference the HCCL API header files and library files when building and running your app.

The HCCL API header files are in the `${INSTALL_DIR}/include/hccl/` directory, and the library files are in the `${INSTALL_DIR}/lib64/` directory. Replace `${INSTALL_DIR}` with the actual CANN installation path. For example, if the software is installed as user `root`, the default installation path is `/usr/local/Ascend/cann`.

> [!CAUTION]
> When compiling an HCCL API program, link the library files according to the dependencies of the included header files. Linking unnecessary `.so` files may cause version functionality exceptions or compatibility issues during subsequent version upgrades.

Include the required files based on the HCCL APIs you actually use. The purpose of each header file is described in the following table.

**Table 2** Header files

| API-Defining Header File | Description | Matching Library File |
| --- | --- | --- |
| hccl/hccl.h | Defines the collective communication and point-to-point communication operator APIs. | libhccl.so |
| hccl/hccl_mc2.h | Defines the MC2 custom operator framework APIs, including HcclKfc\* parameter object allocation/setting and HcclCreateOpResCtx communication resource context creation. | libhccl.so |

HCCL is released as the `cann-hccl_<version>_linux-<arch>.run` installation package, which contains `libhccl.so`, external header files, and `aicpu_hccl.tar.gz` (the HCCL AI CPU operator package). For static builds, the package additionally includes `libhccl_static.a`.

- For details about the source code build and installation procedure, see [Build Guide](../build/build.md).
- For details about the prototype, parameters, data types, and constraints of communication operator APIs, see [Communication Operator APIs](./comm_op_interface/README.md).
