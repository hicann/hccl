# CI 失败诊断手册

CI（openlibing 平台）失败时的定位与修复模式。触发方式为 PR 评论 `/compile`，任务含 Compile_Ascend_X86/ARM(_ubuntu24)、codecheck(+codestyle)、staticcheck(markdownlint 等)、UT、ST、API_Check、precommit(OAT)、PreSmoke。

## 诊断路径

1. `--ci-logs` 取失败信息：pre-commit/markdownlint 日志从 OBS 直链免登录下载；其他任务看 cann-robot 评论里的流水线链接，浏览器打开任务详情页看日志。
2. 日志里 `grep -E "error|Error|ERROR|FAILED|exit 1"` 定位根因行。
3. 对照下表修复；修不了的（平台问题）记录并重触发。

## C++ 变更常见失败模式（仓主体语言）

| 失败模式 | 特征 | 修复 |
|----------|------|------|
| 编译错误（Compile_X86/ARM） | 日志 `error:` | 定位文件行号本地复现：Linux 环境 `bash build.sh --pkg`（命令以仓 AGENTS.md 第 4 节为准）；注意 CMake 缓存会掩盖错误，目录结构变更后须清 build 目录重编 |
| clang-format 风格（precommit） | precommit 失败，clang-format hook 报 diff | `clang-format -i <文件>`（版本须与 `.pre-commit-config.yaml` 的 rev 一致）；只对本次改动的文件跑，勿全仓格式化 |
| OAT 许可头（precommit） | 日志 `License Header Invalid` | 新增源文件头加 CANN-2.0 许可头，与仓内已有 C++ 文件逐字节一致（对照 `src/` 下任一 `.cc`） |
| OAT 二进制误判（precommit） | `Invalid File Type — Content: binary` | 文件注释改纯英文 ASCII（中文多字节字符被 chardet 误判） |
| UT/ST 用例失败 | UT_Test/ST_Test 任务失败 | 先看是否环境抖动（见"已知非阻塞"）；真实失败按日志定位用例，本地 `bash build.sh -u`（`-s`）复现 |
| 链接错误 | `undefined reference to` | 检查新增符号是否漏加进 CMakeLists.txt 的目标源文件列表；acl* 符号未定义通常是本地 CANN 版本差异，CI 不报则不阻塞 |
| add_subdirectory 被注释 | 特定模块 .o 缺失、chmod 报错 | 恢复被注释的 `add_subdirectory`（BUILD_OPEN_PROJECT 依赖完整目录树） |
| 目录重命名遗漏 | `fatal error: xxx.h: No such file` | 全仓 grep 旧路径（含 experimental/）：CMakeLists、`#include` 相对路径、cmake/、build.sh、classify_rule.yaml、blacklist.txt |
| CMake 缓存掩盖 | 本地增量通过 CI 失败 | `rm -rf build*` 后干净重编验证 |
| codecheck 静态告警 | codecheck 任务失败，详情页 `G.*` 规则 | 浏览器打开 cann-robot 评论里的 entryCheckDashCode 链接看告警清单，按规则修复 |

## codecheck 规则与修复模式（新增脚本文件常遇）

codecheck 对 `.agents/` 下 Python 亦全量检查；C++ 告警在 codecheck 任务详情页看规则与行号。

| 规则 | 含义 | 修复模式 |
|------|------|---------|
| G.LOG.02 | 禁 print | 用 `logging`（basicConfig + LOG.info） |
| G.FMT.02 | 行宽超 120 | 拆行（按字符数算，中文 1 字符） |
| G.FMT.03 | 嵌套 def 前缺空行 | 函数体内定义函数前补空行 |
| G.FMT.04 | 标点后多余空格 | 删多余空格 |
| G.FMT.05/07 | import 位置/顺序 | import 全部放顶部 |
| G.FNM.03 | 函数参数过多（>5） | 用类（如 NamedTuple）封装参数 |
| G.CTL.03 | if 布尔表达式过多（>3） | 提取中间变量或辅助函数 |
| G.EDV.05 | 外部命令无绝对路径 | `shutil.which("git")` 解析绝对路径 |
| G.VAR.03 | 覆盖外部标识符 | 改名避免覆盖顶部 import |
| G.EXP.04 | 推导式子句过多（>2） | 改普通 for 循环 |
| G.CLS.06 | 类的方法排列（helper 应在测试方法后） | helper 方法移到类定义末尾，或提升为模块级函数 |
| G.NAM.02 | 禁单字符变量名（l/I/o） | 改有含义名（item/entry 等） |
| G.ERR.09 | 同一 except 捕父子类异常（如 HTTPError+URLError） | 只捕父类 |

## markdownlint（staticcheck_md_check）

按行号修 Markdown 格式：列表前缺空行（MD032）、有序列表编号风格（MD029）、标题层级跳跃（MD001）。

## 环境坑（本地跑 UT/ST 前先排查，全部实测踩过）

| 症状 | 根因 | 处置 |
|------|------|------|
| 编译报 `acl* 符号 was not declared` | master 用了新版 CANN 才有的符号，本机 CANN 落后 | `grep <符号> $ASCEND_HOME_PATH/include/acl/acl_rt.h` 确认后，按 build.md 镜像站（`https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/` 最新时间戳目录）下载 toolkit 更新；勿改代码迁就旧 CANN |
| UT 的 aicpu 套件报 `ccl_kernel.json is not a valid real path` | 未安装 device kernel：须 `build.sh --pkg --full` 并安装到 CANN（`chmod -R u+w $CANN && bash build_out/cann-hccl_*.run --full --install-path=$CANN`） | 装完重跑；执行测试的 shell 须已 source set_env.sh |
| WSL `source set_env.sh` 后 `$ASCEND_HOME_PATH` 仍为空 | set_env.sh 内 `read -r` 需要 stdin，`bash -c "source ..."` 内联方式静默失败 | 用 heredoc（`wsl << EOF ... EOF`）方式执行并回显校验变量 |
| ARM 环境 UT 大面积 `SIGILL`/`Illegal instruction`（37 个测试 dumped core）或 mockcpp `Virtual method address should be odd` 失败 | mockcpp 2.7 的自由函数打桩（`MOCKER(<libc函数>)` 的 trampoline）在 aarch64 + gcc 10 系组合下生成非法指令（gdb 可见被桩函数首指令被 `udf #0` 覆盖）；仓内 CI 的 ARM 通道用 gcc-14 镜像无此问题，master 代码本身支持 ARM | 工具链限制而非代码问题：用 master 干净 worktree 对照确认后可判定环境性失败；在 gcc-14 环境（CI 或 x86）同用例通过即非阻塞 |

## 已知非阻塞判定

- **UT_Test FAILED ≠ 测试失败**：日志里 `[  PASSED  ]`/`[  FAILED  ]` 只看测试本身；增量覆盖率脚本 `get_ai_inc_cov.py` 报错导致的 FAILED 不影响 `ci_state_passed`。先重触发一轮再判断。
- **codecheck DEV-CODECI-35002**：CI 平台级错误（"构建任务执行失败"），与代码无关，重触发即可。
- **`api-check-failed` 与 `ci-pipeline-passed` 并存**：后者是 stale 残留标签（聚合流水线成功已含 API_Check），不需要重触发。
- **流水线"过期"提示**：GitCode 门禁校验流水线 commitID == PR 当前 head；push 新 commit 后旧 passed 失效属正常，重新 `/compile` 即可。

## 修复闭环

修复 → 本地验证（C++ 按仓 AGENTS.md 构建命令；skill 脚本跑单测）→ commit → push → 评论 `/compile`（单次，勿重复）→ `--ci-status --wait` 轮询 → 直至 `passed`。
