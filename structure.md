+ inc                        对外头文件
  + hccl.h                   所有算子的入口
+ src
  + common                    公共方法、工具函数
    + adapter_acl.cc        ACLRT接口封装   ---基本不会变的放外层，其他放自验目录内部
    + adapter_acl.h
    + register.cc          scratch内存注册封装 零拷贝内存注册判断等
    + coll_alg_utils.cc
    + common.h
  + ops
    + kernel_base.cc
    + alg_template_base.cc
    + inc                       算子内部公共头文件
      + coll_alg_param.h
      + ctx.h
    + registry               kernel/template注册机制
      + coll_alg_exec_registry.cc
    + channel                  channel资源申请 建链对端计算  只给自定义算法用
      + channel_request.cc
      + channel.cc
    + topo                    拓扑信息解析   -> ranktable带版本号
      + topo.cc
    + scatter   自定义算子/算法和自研算子/算法区分  商发隔离
      + scatter_op.cc      scatter算子入口处理
      + contribute_algo        用户贡献算法
        + scatter_aaa_executor.cc
      + algo
        + template            scatter host/aicpu/aiv 编排实现
          + scatter_nhr.cc
          + scatter_ring.cc
          + scatter_aiv.cc
        + scatter_executor_base.cc
        + scatter_mesh_executor.cc  -> channel / memory / thread
        + scatter_ring_executor.cc
        + scatter_comm_executor.cc
  + contri_ops           用户贡献算子，只依赖基础包接口
    + channel                  是否需要
      + channel_request.cc
      + channel.cc
    + custom_alltoallmt      // owner
      + custom_a2a_op.cc
      + channel
+ docs                       自定义开发文档
+ cmake                      工程编译
+ examples                  最简易demo 参考接口评审最简易流程
+ test
  + llt                     checker算法验证，只看基础包接口



.
├── inc                           对外头文件
│   └── hccl.h                    所有算子的入口
├── src
│   ├── common                    公共方法、工具函数
│   │   ├── adapter_acl.cc        ACLRT接口封装   ---基本不会变的放外层，其他放自验目录内部
│   │   ├── adapter_acl.h
│   │   ├── register.cc           scratch内存注册封装 零拷贝内存注册判断等
│   │   ├── coll_alg_utils.cc
│   │   └── common.h
│   ├── ops
│   │   ├── kernel_base.cc
│   │   ├── alg_template_base.cc
│   │   ├── inc                   算子内部公共头文件
│   │   │   ├── coll_alg_param.h
│   │   │   └── ctx.h
│   │   ├── registry               kernel/template注册机制
│   │   │   └── coll_alg_exec_registry.cc
│   │   ├── channel                  channel资源申请 建链对端计算  只给自定义算法用
│   │   │   ├── channel_request.cc
│   │   │   └── channel.cc
│   │   ├── topo                    拓扑信息解析   -> ranktable带版本号
│   │   │   └── topo.cc
│   │   └── scatter   自定义算子/算法和自研算子/算法区分  商发隔离
│   │       ├── scatter_op.cc      scatter算子入口处理
│   │       ├── contribute_algo        用户贡献算法
│   │       │   └── scatter_aaa_executor.cc
│   │       └── algo
│   │           ├── template            scatter host/aicpu/aiv 编排实现
│   │           │   ├── scatter_nhr.cc
│   │           │   ├── scatter_ring.cc
│   │           │   └── scatter_aiv.cc
│   │           ├── scatter_executor_base.cc
│   │           ├── scatter_mesh_executor.cc  -> channel / memory / thread
│   │           ├── scatter_ring_executor.cc
│   │           └── scatter_comm_executor.cc
│   └── contri_ops           用户贡献算子，只依赖基础包接口
│       ├── channel                  是否需要
│       │   ├── channel_request.cc
│       │   └── channel.cc
│       └── custom_alltoallmt      // owner
│           ├── custom_a2a_op.cc
│           └── channel
├── docs                       自定义开发文档
├── cmake                      工程编译
├── examples                  最简易demo 参考接口评审最简易流程
└── test
    └── llt                     checker算法验证，只看基础包接口




ops-hccl
1. 目录结构是否合理
2. 是否能支持每个算子(scatter)独立编译so 独立打包      算子so+公共so? --> 公共.o  -> 单独so
3. rankGraph获取接口待讨论
4. plugin目录，对profiling等DFX工具的调用封装 其他三方组件，  暂时不要
