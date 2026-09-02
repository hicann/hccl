#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""贡献流程辅助工具。

为 AI Agent / 贡献者固化「从代码获取到 PR 合入」流程中必须精确可靠的机械步骤：
仓库同步（clone/fetch/worktree 隔离）、Issue 查重与创建、fork 推送与 PR 创建
（自动触发 CI）、CI 状态轮询与失败日志获取、检视意见列表、临时环境清理。
本地构建与测试不属于本脚本职责，按仓内 AGENTS.md 的构建命令直接执行。

用法示例:
    # 首次使用前校验环境（token、git、仓库路径、remote 配置）
    python3 contribute.py --check-env

    # 同步仓库（缺则 clone，有则 fetch；工作区脏时自动创建隔离 worktree）
    python3 contribute.py --sync-repo

    # Issue 查重与创建（标题关键词匹配，已有则复用不重复建）
    python3 contribute.py --issue-ensure --title "标题" --body-file body.md

    # 提交 PR（push fork + 创建/复用 PR + 自动评论 /compile 触发 CI）
    python3 contribute.py --submit-pr --title "[fix]xxx" --body-file pr_body.md --issue 123

    # 查询 CI 状态（saw_running 状态机防旧标签误判）
    python3 contribute.py --ci-status --pr 123
    python3 contribute.py --ci-status --pr 123 --wait --timeout 1800

    # 下载 CI 失败日志（OBS 直链，无需登录）
    python3 contribute.py --ci-logs --pr 123 --output-dir ./ci_logs

    # 列出 PR 未处理的检视意见
    python3 contribute.py --list-review-comments --pr 123

环境变量:
    GITCODE_TOKEN    GitCode 个人访问令牌（也可由 git credential fill 自动获取）

依赖: Python 3.7+，仅标准库。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Windows 终端默认 GBK，强制 UTF-8 避免中文/emoji 输出崩溃
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
LOG = logging.getLogger("contribute")

GITCODE_API_V5 = "https://gitcode.com/api/v5"
OBS_LOG_BASE = "https://ascend-ci.obs.cn-north-4.myhuaweicloud.com"
UPSTREAM_OWNER = "cann"
BASE_BRANCH_CANDIDATES = ("master", "main")
CI_LABEL_RUNNING = "ci-pipeline-running"
CI_LABEL_PASSED = "ci-pipeline-passed"
CI_LABEL_FAILED = "ci-pipeline-failed"


# ---------------------------------------------------------------------------
# 基础设施：git / HTTP / token
# ---------------------------------------------------------------------------

GIT_EXECUTABLE = shutil.which("git") or "git"


def run_git(args, cwd, check=True):
    """执行 git 命令，check 失败时抛出 RuntimeError。"""
    result = subprocess.run(
        [GIT_EXECUTABLE] + args,
        cwd=str(cwd), capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    if check and result.returncode != 0:
        raise RuntimeError("git {} failed: {}".format(" ".join(args), result.stderr.strip()))
    return result


def get_token():
    """获取 GitCode token：优先环境变量 GITCODE_TOKEN，否则 git credential fill。

    credential fill 加 timeout：交互式 helper（如 Windows GCM 弹 GUI）无缓存时
    会挂起，超时视同取不到 token。
    """
    token = os.environ.get("GITCODE_TOKEN", "").strip()
    if token:
        return token
    try:
        proc = subprocess.run(
            [GIT_EXECUTABLE, "credential", "fill"],
            input="protocol=https\nhost=gitcode.com\n\n",
            capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=15,
        )
    except subprocess.TimeoutExpired:
        return ""
    for line in proc.stdout.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1].strip()
    return ""


def api_request(method, path, token, payload=None):
    """调用 GitCode API 并返回 (status_code, json_or_text)。

    v5 用 access_token 查询参数（写操作用 Bearer 头更稳），v4 用 PRIVATE-TOKEN 头。
    429 限速时等待 60s 后重试一次（与 references/gitcode-api.md 错误码表一致）。
    """
    result = api_request_once(method, path, token, payload)
    if result[0] == 429:
        time.sleep(60)
        result = api_request_once(method, path, token, payload)
    return result


def api_request_once(method, path, token, payload=None):
    """单次 API 调用（429 重试逻辑在 api_request）。"""
    url = "{}{}".format(GITCODE_API_V5, path)
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    headers["Authorization"] = "Bearer {}".format(token)
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            code = resp.status
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8", errors="replace")
        code = err.code
    except urllib.error.URLError as err:
        return 0, str(err)
    if not body:
        return code, {}
    try:
        return code, json.loads(body)
    except ValueError:
        return code, body


def api_get_all(path, token, max_pages=50, page_size=100):
    """分页拉全量列表。防两类死循环：返回数 < page_size 终止、首条 id 重复终止。"""
    items = []
    first_id_prev = None
    # 剥离调用方可能已带的分页参数，避免 URL 出现重复参数
    clean_path = re.sub(r"[?&](per_page|page)=\d+", "", path)
    clean_path = clean_path.rstrip("?")
    for page in range(1, max_pages + 1):
        sep = "&" if "?" in clean_path else "?"
        code, data = api_request(
            "GET", "{}{}per_page={}&page={}".format(clean_path, sep, page_size, page),
            token,
        )
        if code != 200 or not isinstance(data, (list, dict)):
            break
        if isinstance(data, dict):
            # v4 端点返回 {data:[...], end_id} 包装，需解包
            data = data.get("data") or []
            if not isinstance(data, list):
                break
        if not data:
            break
        first_id = data[0].get("id") if isinstance(data[0], dict) else None
        if first_id is not None and first_id == first_id_prev:
            break  # 翻页重复返回第一页（GitCode 已知行为）
        items.extend(data)
        first_id_prev = first_id
        if len(data) < page_size:
            break
    return items


# ---------------------------------------------------------------------------
# remote / 仓库探测
# ---------------------------------------------------------------------------

def normalize_remote_url(url):
    """归一化 remote URL：去协议、SCP 形式、.git 后缀，返回 owner/repo。"""
    url = url.strip()
    url = re.sub(r"^https?://", "", url)
    url = re.sub(r"^git@", "", url)
    if ":" in url.split("/")[0]:
        url = url.replace(":", "/", 1)
    url = re.sub(r"\.git$", "", url)
    parts = [p for p in url.split("/") if p]
    if len(parts) >= 2:
        return "{}/{}".format(parts[-2], parts[-1])
    return ""


def find_remote(repo_root, owner, repo):
    """按 owner/repo 精确匹配 remote 名；未命中返回 None。"""
    result = run_git(["remote", "-v"], cwd=repo_root)
    for line in result.stdout.splitlines():
        if "(fetch)" not in line:
            continue
        name, url = line.split("\t")[0], line.split("\t")[1].replace(" (fetch)", "")
        if normalize_remote_url(url).lower() == "{}/{}".format(owner, repo).lower():
            return name
    return None


def detect_repo_layout(repo_root, repo_name):
    """探测仓库 remote 布局：upstream remote 名与 fork remote 名。

    优先按 owner/repo 精确匹配；无精确匹配时回退到惯例命名
    （名为 upstream 的 remote 视为上游）。fork 探测与 token 账号绑定：
    只认 owner == 当前 token 账号（GET /user）的同名仓 remote，
    找不到时回退惯例名 origin，避免把他人 fork 误当自己的 fork。
    返回 dict: upstream_remote, fork_remote, fork_owner, base_branch。
    """
    upstream_remote = find_remote(repo_root, UPSTREAM_OWNER, repo_name)
    remotes = []
    result = run_git(["remote", "-v"], cwd=repo_root)
    for line in result.stdout.splitlines():
        if "(fetch)" not in line or "\t" not in line:
            continue
        name = line.split("\t")[0]
        url = line.split("\t")[1].replace(" (fetch)", "")
        remotes.append((name, url))
    if upstream_remote is None:
        for name, _url in remotes:
            if name == "upstream":
                upstream_remote = name
                break
    if upstream_remote is None:
        return {}
    fork_owner = ""
    fork_remote = ""
    # fork 必须与 token 账号绑定，不做"第一个非 cann"的猜测
    token = get_token()
    login = ""
    if token:
        code, me = api_request("GET", "/user", token)
        if code == 200 and isinstance(me, dict):
            login = (me.get("login") or "").lower()
    for name, url in remotes:
        if name == upstream_remote:
            continue
        owner_repo = normalize_remote_url(url)
        if not owner_repo:
            continue
        parts = owner_repo.split("/")
        if len(parts) != 2 or parts[-1] != repo_name:
            continue
        if login and parts[0].lower() == login:
            fork_owner, fork_remote = parts[0], name
            break
        if not login and parts[0] != UPSTREAM_OWNER:
            fork_owner, fork_remote = parts[0], name
            break
    if not fork_remote:
        # 回退：token 账号匹配失败时用惯例名 origin
        for name, url in remotes:
            if name == "origin":
                owner_repo = normalize_remote_url(url)
                parts = owner_repo.split("/")
                if len(parts) == 2 and parts[0] != UPSTREAM_OWNER and parts[-1] == repo_name:
                    fork_owner, fork_remote = parts[0], name
                break
    base_branch = ""
    for branch in BASE_BRANCH_CANDIDATES:
        probe = run_git(
            ["rev-parse", "--verify", "--quiet", "refs/remotes/{}/{}".format(upstream_remote, branch)],
            cwd=repo_root, check=False,
        )
        if probe.returncode == 0:
            base_branch = branch
            break
    return {
        "upstream_remote": upstream_remote,
        "fork_remote": fork_remote,
        "fork_owner": fork_owner,
        "base_branch": base_branch,
    }


def infer_repo_name(repo_root):
    """从 remote URL 推断仓名：优先 cann 官方 remote。"""
    result = run_git(["remote", "-v"], cwd=repo_root)
    for line in result.stdout.splitlines():
        if "(fetch)" not in line or "\t" not in line:
            continue
        owner_repo = normalize_remote_url(line.split("\t")[1].replace(" (fetch)", ""))
        parts = owner_repo.split("/")
        if len(parts) == 2 and parts[0] == UPSTREAM_OWNER and parts[1] == "hccl":
            return parts[1]
    return ""


# ---------------------------------------------------------------------------
# 子命令：--check-env
# ---------------------------------------------------------------------------

class EnvReport(object):
    """环境校验结果收集器。"""

    def __init__(self):
        self.ok_items = []
        self.err_items = []
        self.warn_items = []

    def ok(self, msg):
        self.ok_items.append(msg)

    def err(self, msg):
        self.err_items.append(msg)

    def warn(self, msg):
        self.warn_items.append(msg)

    def passed(self):
        return not self.err_items


def cmd_check_env(args):
    """校验 git / 仓库路径 / remote 配置；token 按需分级。

    纯 clone + 本地构建的用户不需要 token（issue/PR/CI 子命令才需要），
    未配置 token 记 WARN 而非 ERR，不阻断环境校验。
    """
    report = EnvReport()
    if shutil.which("git"):
        report.ok("git: {}".format(GIT_EXECUTABLE))
    else:
        report.err("git 不可用")
    token = get_token()
    if token:
        code, data = api_request("GET", "/user", token)
        if code == 200 and isinstance(data, dict):
            login = data.get("login", "?")
            report.ok("token: 已配置（账号 {}）".format(login))
        else:
            report.warn("token 已配置但 /user 校验失败（HTTP {}），涉 API 子命令可能不可用".format(code))
    else:
        report.warn("GitCode token 未配置——仅在提交 Issue/PR、查询 CI、列检视意见等"
                    "平台操作时需要；纯 clone/本地构建/跑测试无需配置"
                    "（配置方式：export GITCODE_TOKEN=<token>，或 git clone 凭据自动读取）")
    repo_root = Path(args.repo_root).resolve()
    if (repo_root / ".git").exists():
        report.ok("仓库路径: {}".format(repo_root))
    else:
        report.err("仓库路径不是 git 仓库: {}".format(repo_root))
        print_env_report(report)
        return report.passed()
    repo_name = infer_repo_name(repo_root)
    if repo_name:
        report.ok("仓名: {}".format(repo_name))
        layout = detect_repo_layout(repo_root, repo_name)
        if layout.get("upstream_remote"):
            report.ok("upstream remote: {} (cann/{})".format(
                layout["upstream_remote"], repo_name))
        else:
            report.warn("未找到指向 cann/{} 的 remote（clone 后请配置）".format(repo_name))
        if layout.get("fork_remote"):
            report.ok("fork remote: {} ({}/{})".format(
                layout["fork_remote"], layout["fork_owner"], repo_name))
        else:
            report.warn("未找到个人 fork remote——仅在提交 PR 时需要"
                        "（git remote add fork https://gitcode.com/<你的账号>/{}.git）".format(repo_name))
        if layout.get("base_branch"):
            report.ok("默认分支: {}".format(layout["base_branch"]))
    else:
        report.warn("无法从 remote 推断仓名（非本仓？）")
    name = run_git(["config", "user.name"], cwd=repo_root, check=False).stdout.strip()
    email = run_git(["config", "user.email"], cwd=repo_root, check=False).stdout.strip()
    if name and email:
        report.ok("git 身份: {} <{}>".format(name, email))
    else:
        report.err("git user.name / user.email 未配置（commit 身份必需，且 email 须与 CLA 签署邮箱一致）")
    print_env_report(report)
    return report.passed()


def print_env_report(report):
    """打印环境校验报告。"""
    for msg in report.ok_items:
        LOG.info("[OK] %s", msg)
    for msg in report.warn_items:
        LOG.info("[WARN] %s", msg)
    for msg in report.err_items:
        LOG.info("[ERR] %s", msg)
    LOG.info("---")
    LOG.info("环境校验: %s", "通过" if report.passed() else "未通过")


# ---------------------------------------------------------------------------
# 子命令：--sync-repo
# ---------------------------------------------------------------------------

def cmd_sync_repo(args):
    """仓库同步：clone（缺则）/ fetch / 脏工作区自动 worktree 隔离。

    输出 JSON: {repo_root, action, worktree, base_branch, head_sha}
    action: cloned | fetched_rebase | fetched_worktree | up_to_date | noop
    """
    repo_root = Path(args.repo_root).resolve()
    repo_name = args.repo or infer_repo_name(repo_root)
    if not repo_name:
        raise RuntimeError("无法推断仓名，请用 --repo hccl 指定")
    if not (repo_root / ".git").exists():
        if args.parent_dir is None:
            raise RuntimeError("仓库不存在，首次 clone 请用 --parent-dir 指定父目录")
        parent = Path(args.parent_dir).resolve()
        parent.mkdir(parents=True, exist_ok=True)
        run_git(
            ["clone", "--origin", "upstream",
             "https://gitcode.com/{}/{}.git".format(UPSTREAM_OWNER, repo_name),
             str(parent / repo_name)],
            cwd=".",
        )
        repo_root = parent / repo_name
        run_git(["remote", "set-url", "--push", "upstream", "DISABLE_PUSH"], cwd=repo_root)
        action = "cloned"
        head_sha = run_git(["rev-parse", "HEAD"], cwd=repo_root).stdout.strip()
        # 探测 clone 结果的默认分支（不写死 master）
        head_ref = run_git(["symbolic-ref", "--short", "refs/remotes/upstream/HEAD"],
                           cwd=repo_root, check=False).stdout.strip()
        base_branch = head_ref.split("/")[-1] if head_ref else BASE_BRANCH_CANDIDATES[0]
        emit_json({"repo_root": str(repo_root), "action": action,
                   "base_branch": base_branch, "head_sha": head_sha,
                   "next_hint": "提交 PR 前请配置个人 fork remote："
                                "git remote add fork https://gitcode.com/<你的账号>/{}.git"
                                .format(repo_name)})
        return
    layout = detect_repo_layout(repo_root, repo_name)
    upstream_remote = layout.get("upstream_remote")
    if not upstream_remote:
        raise RuntimeError("未找到指向 cann/{} 的 upstream remote".format(repo_name))
    base_branch = layout.get("base_branch") or BASE_BRANCH_CANDIDATES[0]
    run_git(["fetch", upstream_remote, base_branch], cwd=repo_root)
    upstream_sha = run_git(
        ["rev-parse", "{}/{}".format(upstream_remote, base_branch)], cwd=repo_root,
    ).stdout.strip()
    status = run_git(["status", "--porcelain"], cwd=repo_root).stdout.strip()
    current_branch = run_git(
        ["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root,
    ).stdout.strip()
    on_base = current_branch == base_branch
    if status:
        # 工作区脏：不动现有改动，开隔离 worktree 跟进最新代码
        branch_name = "contribute/{}-{}".format(base_branch, upstream_sha[:8])
        exists = run_git(
            ["rev-parse", "--verify", "--quiet", "refs/heads/" + branch_name],
            cwd=repo_root, check=False,
        ).returncode == 0
        worktree_path = str(repo_root.parent / (".wt-" + repo_name + "-sync"))
        if exists:
            # 分支已存在：复用分支重建 worktree（不带 -b，避免 already exists 报错）
            run_git(["worktree", "remove", "--force", worktree_path], cwd=repo_root, check=False)
            run_git(["worktree", "prune"], cwd=repo_root)
            run_git(["worktree", "add", worktree_path, branch_name], cwd=repo_root)
        else:
            run_git(["worktree", "add", "-b", branch_name, worktree_path,
                     "{}/{}".format(upstream_remote, base_branch)], cwd=repo_root)
        action = "fetched_worktree"
        repo_root = Path(worktree_path)
    elif on_base:
        local_sha = run_git(["rev-parse", "HEAD"], cwd=repo_root).stdout.strip()
        if local_sha == upstream_sha:
            action = "up_to_date"
        else:
            # 本地 HEAD 须为上游祖先才允许快进；有本地自有提交时隔离到 worktree，绝不静默丢弃
            is_ancestor = run_git(
                ["merge-base", "--is-ancestor", "HEAD",
                 "{}/{}".format(upstream_remote, base_branch)],
                cwd=repo_root, check=False,
            ).returncode == 0
            if is_ancestor:
                run_git(["reset", "--hard", "{}/{}".format(upstream_remote, base_branch)],
                        cwd=repo_root)
                run_git(["clean", "-fd"], cwd=repo_root, check=False)
                action = "fetched_rebase"
            else:
                local_branch = "contribute/{}-local-{}".format(base_branch, local_sha[:8])
                worktree_path = str(repo_root.parent / (".wt-" + repo_name + "-sync"))
                run_git(["branch", local_branch, local_sha], cwd=repo_root, check=False)
                run_git(["worktree", "remove", "--force", worktree_path], cwd=repo_root, check=False)
                run_git(["worktree", "prune"], cwd=repo_root)
                run_git(["worktree", "add", worktree_path, local_branch], cwd=repo_root)
                action = "fetched_worktree"
                repo_root = Path(worktree_path)
    else:
        # 干净但在 feature 分支：只 fetch 不切分支
        action = "noop"
    head_sha = run_git(["rev-parse", "HEAD"], cwd=repo_root).stdout.strip()
    emit_json({"repo_root": str(repo_root), "action": action,
               "base_branch": base_branch, "head_sha": head_sha,
               "upstream_sha": upstream_sha})


def emit_json(obj):
    """输出一行 JSON 供 agent 消费。"""
    LOG.info(json.dumps(obj, ensure_ascii=False))


# ---------------------------------------------------------------------------
# 子命令：--issue-ensure
# ---------------------------------------------------------------------------

ISSUE_TITLE_PREFIXES = {
    "hccl": "[Requirement|需求建议]: ",
    "hcomm": "【需求】",
}


def issue_keywords(title):
    """从标题提取查重关键词（去前缀/标点；中文串取 2 字滑窗片段 + 英文整词）。"""
    cleaned = re.sub(r"^\[.*?\]:\s*", "", title)
    cleaned = re.sub(r"^【.*?】\s*", "", cleaned)
    keywords = []
    for seg in re.split(r"[（）()\[\]【】:：,，。.\s]+", cleaned):
        if not seg:
            continue
        latin = re.findall(r"[A-Za-z0-9_-]{2,}", seg)
        keywords.extend(latin)
        cjk = "".join(re.findall(r"[一-鿿]", seg))
        for i in range(len(cjk) - 1):
            keywords.append(cjk[i:i + 2])
    return keywords or [cleaned]


def search_existing_issue(repo_name, title, token, state):
    """按标题关键词在 open/all issues 中找已有 Issue，返回匹配列表。

    命中标准：过半关键词出现在已有 Issue 标题中。
    """
    path = "/repos/{}/{}/issues?state={}&sort=created&direction=desc".format(
        UPSTREAM_OWNER, repo_name, state)
    issues = api_get_all(path, token, max_pages=10)
    keywords = issue_keywords(title)
    matches = []
    threshold = max(2, len(keywords) // 2)
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        issue_title = (issue.get("title") or "").lower()
        hits = sum(1 for kw in keywords if kw.lower() in issue_title)
        if keywords and hits >= threshold:
            matches.append({"number": issue.get("number"), "title": issue.get("title"),
                            "state": issue.get("state"), "web_url": issue.get("html_url")})
    return matches


def cmd_issue_ensure(args):
    """Issue 查重与创建：已有则复用，无则按仓模板前缀创建。输出 JSON。"""
    token = require_token()
    repo_name = args.repo
    if not repo_name:
        raise RuntimeError("请用 --repo hccl 指定仓")
    matches = search_existing_issue(repo_name, args.title, token, "open")
    if matches:
        emit_json({"action": "reused", "existing": matches[:3],
                   "note": "已有同主题 open Issue，未重复创建"})
        return
    closed_matches = search_existing_issue(repo_name, args.title, token, "closed")
    if closed_matches:
        emit_json({"action": "reused_closed", "existing": closed_matches[:3],
                   "note": "同主题 Issue 已关闭。请确认是否应 reopen 该 Issue 或另建新 Issue；"
                           "未自动创建"})
        return
    prefix = ISSUE_TITLE_PREFIXES.get(repo_name, "")
    full_title = "{}{}".format(prefix, args.title) if prefix else args.title
    body = read_text_arg(args.body, args.body_file)
    code, data = api_request(
        "POST", "/repos/{}/{}/issues".format(UPSTREAM_OWNER, repo_name), token,
        payload={"title": full_title, "body": body},
    )
    if code not in (200, 201) or not isinstance(data, dict):
        raise RuntimeError("Issue 创建失败 (HTTP {}): {}".format(code, data))
    emit_json({"action": "created", "number": data.get("number"),
               "title": full_title, "web_url": data.get("html_url")})


def read_text_arg(text, text_file):
    """读入文本：--body 直接取值，--body-file 读文件（二选一）。"""
    if text_file:
        path = Path(text_file)
        if not path.is_file():
            raise RuntimeError("文件不存在: {}".format(path))
        return path.read_text(encoding="utf-8")
    if text:
        return text
    raise RuntimeError("内容为空，请用 --body 或 --body-file 提供")


# ---------------------------------------------------------------------------
# 子命令：--submit-pr
# ---------------------------------------------------------------------------

def cmd_submit_pr(args):
    """push fork + 创建/复用 PR + 自动评论 /compile。输出 JSON。"""
    token = require_token()
    repo_root = Path(args.repo_root).resolve()
    repo_name = args.repo or infer_repo_name(repo_root)
    layout = detect_repo_layout(repo_root, repo_name)
    fork_remote = layout.get("fork_remote")
    fork_owner = layout.get("fork_owner")
    if not fork_remote or not fork_owner:
        raise RuntimeError("未找到个人 fork remote（需要指向 <你的账号>/{} 的 remote）".format(repo_name))
    base_branch = args.base or layout.get("base_branch") or "master"
    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root).stdout.strip()
    if branch in BASE_BRANCH_CANDIDATES:
        raise RuntimeError("当前在 {} 分支，请先切到 feature 分支再提交 PR".format(branch))
    status = run_git(["status", "--porcelain"], cwd=repo_root).stdout.strip()
    if status:
        raise RuntimeError("工作树有未提交改动，请先 commit:\n{}".format(status))
    # git 身份校验（硬阻断：缺身份的 commit 无法关联 CLA）
    name = run_git(["config", "user.name"], cwd=repo_root, check=False).stdout.strip()
    email = run_git(["config", "user.email"], cwd=repo_root, check=False).stdout.strip()
    if not name or not email:
        raise RuntimeError("git user.name/user.email 未配置，无法安全提交")
    # push fork：先 fetch 建立 remote-tracking ref，--force-with-lease 才真正设防
    run_git(["fetch", fork_remote, branch], cwd=repo_root, check=False)
    lease = run_git(["ls-remote", fork_remote, "refs/heads/{}".format(branch)],
                    cwd=repo_root, check=False).stdout.strip()
    push_args = ["push", fork_remote, "HEAD:refs/heads/{}".format(branch)]
    if lease:
        remote_sha = lease.split()[0]
        push_args.append("--force-with-lease=refs/heads/{}:{}".format(branch, remote_sha))
    else:
        push_args.append("--force-with-lease")
    run_git(push_args, cwd=repo_root)
    body = read_text_arg(args.body, args.body_file)
    # --issue：body 未引用 Issue 时自动追加关联（所有 PR 必须关联 Issue）
    issue_number = getattr(args, "issue_number", None)
    issue_appended = False
    # 精确匹配 #N 引用（子串匹配会把 #12 误判为已引用 #1234）
    cited = set(re.findall(r"#(\d+)", body))
    if issue_number and str(issue_number) not in cited:
        body = "{}\n\n## 关联的Issue\n#{}".format(body.rstrip(), issue_number)
        issue_appended = True
    payload = {"title": args.title, "body": body,
               "head": "{}:{}".format(fork_owner, branch), "base": base_branch}
    path = "/repos/{}/{}/pulls".format(UPSTREAM_OWNER, repo_name)
    code, data = api_request("POST", path, token, payload=payload)
    action = "created"
    if code in (200, 201) and isinstance(data, dict) and data.get("number"):
        pr_number = data.get("number")
    elif code == 422 or (isinstance(data, (str, dict)) and "exist" in str(data).lower()):
        # 已有同源分支 PR：翻页查 open 列表复用（仓 open PR 可达数百个，单页 50 条不够）
        open_prs = api_get_all(
            "/repos/{}/{}/pulls?state=open".format(UPSTREAM_OWNER, repo_name),
            token, max_pages=20)
        pr_number = None
        for pr in open_prs:
            head_info = pr.get("head") or {}
            if head_info.get("ref") == branch and (head_info.get("user") or {}).get("login") == fork_owner:
                pr_number = pr.get("number")
                action = "reused"
                break
        if pr_number is None:
            raise RuntimeError("PR 创建失败（疑似已存在但未能定位）: {}".format(data))
    else:
        raise RuntimeError("PR 创建失败 (HTTP {}): {}".format(code, data))
    compile_ok = True
    if not args.no_compile:
        ccode, _ = api_request(
            "POST", "/repos/{}/{}/pulls/{}/comments".format(UPSTREAM_OWNER, repo_name, pr_number),
            token, payload={"body": "/compile"},
        )
        compile_ok = ccode in (200, 201)
    # GET 回查（HTTP 成功不等于写入成功）
    gcode, gdata = api_request(
        "GET", "/repos/{}/{}/pulls/{}".format(UPSTREAM_OWNER, repo_name, pr_number), token)
    verified = gcode == 200 and isinstance(gdata, dict) and gdata.get("state") in ("open", "opened")
    emit_json({"action": action, "pr_number": pr_number,
               "web_url": "https://gitcode.com/{}/{}/pulls/{}".format(UPSTREAM_OWNER, repo_name, pr_number),
               "branch": branch, "fork_owner": fork_owner, "base": base_branch,
               "compile_triggered": compile_ok, "verified": verified,
               "issue_number": issue_number, "issue_appended": issue_appended,
               "head_sha": run_git(["rev-parse", "HEAD"], cwd=repo_root).stdout.strip()})
    if not verified:
        raise RuntimeError("PR GET 回查未通过，请人工核验")


# ---------------------------------------------------------------------------
# 子命令：--ci-status
# ---------------------------------------------------------------------------

def get_pr_labels(repo_name, pr_number, token):
    """取 PR 当前标签名集合。"""
    code, data = api_request(
        "GET", "/repos/{}/{}/pulls/{}".format(UPSTREAM_OWNER, repo_name, pr_number), token)
    if code != 200 or not isinstance(data, dict):
        raise RuntimeError("查询 PR 失败 (HTTP {}): {}".format(code, data))
    labels = [item.get("name", "") for item in (data.get("labels") or []) if isinstance(item, dict)]
    return labels, data


def judge_ci(labels_history, labels_now):
    """saw_running 状态机判定 CI 终态。

    必须出现过 running 且 running 已消失后，才认 passed/failed，
    防旧 run 残留的 stale failed 标签误判。
    """
    saw_running = any(CI_LABEL_RUNNING in labels for labels in labels_history)
    running_now = CI_LABEL_RUNNING in labels_now
    passed_now = CI_LABEL_PASSED in labels_now
    failed_now = CI_LABEL_FAILED in labels_now
    if running_now:
        return "running"
    if saw_running and passed_now:
        return "passed"
    if saw_running and failed_now:
        return "failed"
    if saw_running:
        return "finishing"  # running 消失但终态标签未上（出标签间隙）
    if passed_now or failed_now:
        return "stale"  # 无 running 历史的残留标签，不能当本轮结论
    return "not_triggered"


def cmd_ci_status(args):
    """查询/轮询 CI 状态。单次输出 JSON；--wait 轮询至终态。

    wait 模式下 stale/not_triggered 前 3 个 interval 内不退出——/compile 评论
    到 robot 打 running 标签之间有窗口期，立即退出会诱导重复触发。
    """
    token = require_token()
    repo_name = args.repo
    if not repo_name:
        raise RuntimeError("请用 --repo hccl 指定仓")
    labels_history = []
    deadline = time.time() + args.timeout
    stale_rounds = 0
    while True:
        labels, pr_data = get_pr_labels(repo_name, args.pr_number, token)
        labels_history.append(labels)
        state = judge_ci(labels_history, labels)
        if not args.wait or state in ("passed", "failed"):
            emit_json({"pr": args.pr_number, "state": state, "labels": labels,
                       "mergeable": pr_data.get("mergeable"),
                       "pr_state": pr_data.get("state")})
            return
        if state in ("stale", "not_triggered"):
            stale_rounds += 1
            if stale_rounds > 3:
                # 连续多轮无 running：旧标签残留且新轮未触发，提示确认 /compile
                emit_json({"pr": args.pr_number, "state": state, "labels": labels,
                           "stale_rounds": stale_rounds,
                           "note": "连续多轮未见 ci-pipeline-running；若本进程未触发过"
                                   "/compile，请先评论 /compile 再轮询（若已触发请"
                                   "确认 PR 是否被更新）",
                           "mergeable": pr_data.get("mergeable"),
                           "pr_state": pr_data.get("state")})
                return
        if time.time() >= deadline:
            emit_json({"pr": args.pr_number, "state": "timeout", "labels": labels,
                       "mergeable": pr_data.get("mergeable"),
                       "pr_state": pr_data.get("state")})
            return
        time.sleep(args.interval)


# ---------------------------------------------------------------------------
# 子命令：--ci-logs
# ---------------------------------------------------------------------------

def parse_pipeline_links(comments):
    """从 PR 评论中提取流水线链接与失败任务信息（cann-robot 评论）。"""
    pipelines = []
    for comment in comments:
        if not isinstance(comment, dict):
            continue
        user = (comment.get("user") or {}).get("login", "")
        body = comment.get("body") or ""
        if user != "cann-robot":
            continue
        for url in re.findall(r"https?://[^\s'\"<>]+pipelineDetail[^\s'\"<>]*", body):
            pipelines.append({"url": url, "comment_at": comment.get("created_at")})
    return pipelines


def cmd_ci_logs(args):
    """收集 CI 失败信息：流水线链接（PR 评论）+ OBS 直链日志下载。"""
    token = require_token()
    repo_name = args.repo
    if not repo_name:
        raise RuntimeError("请用 --repo hccl 指定仓")
    comments = api_get_all(
        "/repos/{}/{}/pulls/{}/comments?per_page=100".format(
            UPSTREAM_OWNER, repo_name, args.pr_number), token, max_pages=5)
    pipelines = parse_pipeline_links(comments)
    out_dir = Path(args.output_dir or ".").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    known_logs = ("pre-commit.txt", "markdownlint.csv")
    for log_name in known_logs:
        url = "{}/{}/package/{}/{}".format(OBS_LOG_BASE, repo_name, args.pr_number, log_name)
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=60) as resp:
                if resp.status == 200:
                    target = out_dir / log_name
                    target.write_bytes(resp.read())
                    downloaded.append(str(target))
        except urllib.error.URLError:
            # URLError 是 HTTPError 的父类，只捕父类即同时覆盖 404 与网络错误
            continue
    emit_json({"pr": args.pr_number, "pipelines": pipelines[-3:],
               "logs_downloaded": downloaded, "output_dir": str(out_dir),
               "note": "流水线详情页日志需在浏览器打开 pipeline 链接获取；"
                       "pre-commit/markdownlint 日志已从 OBS 直链下载"})


# ---------------------------------------------------------------------------
# 子命令：--list-review-comments
# ---------------------------------------------------------------------------

def cmd_list_review_comments(args):
    """列出 PR 未 resolved 的检视意见。

    权威数据源是 v5 comments（resolved 字段真实有效；v4 discussions 的
    note.resolved 实测恒为 None、翻页返回重复内容且数据不全，不可用）。
    文件路径用 v5 单条接口 GET /pulls/comments/{id} 补齐——其
    position.new_path 字段可靠返回（v5 列表接口的 path 常为 None）。
    """
    token = require_token()
    repo_name = args.repo
    if not repo_name:
        raise RuntimeError("请用 --repo hccl 指定仓")
    base = "/repos/{}/{}/pulls/{}".format(UPSTREAM_OWNER, repo_name, args.pr_number)
    comments = api_get_all(base + "/comments?per_page=100", token, max_pages=10)
    findings = []
    for comment in comments:
        if not isinstance(comment, dict):
            continue
        if comment.get("resolved") is True:
            continue
        body = (comment.get("body") or "").strip()
        if not body or body.startswith("/"):
            continue
        user = (comment.get("user") or {}).get("login", "?")
        if user in ("cann-robot", "atomgit-bot"):
            continue
        position = comment.get("diff_position") or {}
        file_path = position.get("path") or ""
        new_line = position.get("start_new_line") or position.get("new_line")
        old_line = position.get("start_old_line") or position.get("old_line")
        line = new_line or old_line
        if not file_path and not line:
            continue  # 无位置信息的普通评论（检视报告等）不是行级检视意见
        if not file_path and comment.get("id"):
            # v5 单条接口补文件路径（列表接口 path 常为 None；
            # 端点不带 PR 号：/repos/{owner}/{repo}/pulls/comments/{id}）
            scode, sdata = api_request(
                "GET", "/repos/{}/{}/pulls/comments/{}".format(
                    UPSTREAM_OWNER, repo_name, comment.get("id")), token)
            if scode == 200 and isinstance(sdata, dict):
                spos = sdata.get("position") or {}
                file_path = spos.get("new_path") or spos.get("old_path") or file_path
        findings.append({"id": comment.get("id"),
                         "discussion_id": comment.get("discussion_id"),
                         "author": user, "file": file_path, "line": line,
                         "side": "old" if old_line and not new_line else "new",
                         "body": body[:400]})
    emit_json({"pr": args.pr_number, "open_findings": findings,
               "count": len(findings)})


# ---------------------------------------------------------------------------
# 子命令：--cleanup
# ---------------------------------------------------------------------------

def cmd_cleanup(args):
    """清理 worktree 与临时文件。"""
    repo_root = Path(args.repo_root).resolve()
    removed = []
    if args.worktree:
        wt = Path(args.worktree)
        if wt.exists():
            run_git(["worktree", "remove", "--force", str(wt)], cwd=repo_root, check=False)
            removed.append(str(wt))
    run_git(["worktree", "prune"], cwd=repo_root, check=False)
    for pattern in ("ci_logs",):
        target = repo_root / pattern
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
            removed.append(str(target))
    emit_json({"removed": removed, "repo_root": str(repo_root)})


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def require_token():
    """取 token，取不到直接报错（附配置方法指引）。"""
    token = get_token()
    if not token:
        raise RuntimeError(
            "该操作需要 GitCode token。配置方式（二选一）：\n"
            "  1) export GITCODE_TOKEN=<个人访问令牌>（Windows: $env:GITCODE_TOKEN 或 set）\n"
            "  2) 已用 git clone 拉取过本仓凭据时，脚本会经 git credential fill 自动读取")
    return token


def main():
    parser = argparse.ArgumentParser(description="贡献流程辅助工具（本仓 contribute skill）")
    parser.add_argument("--repo-root", default=".", help="本仓本地 clone 路径（默认当前目录）")
    parser.add_argument("--repo", help="仓库名（本仓 hcomm；默认自动推断）")
    parser.add_argument("--check-env", action="store_true", help="校验运行环境")
    parser.add_argument("--sync-repo", action="store_true", help="同步仓库（clone/fetch/脏区自动 worktree）")
    parser.add_argument("--parent-dir", help="--sync-repo 首次 clone 时的父目录")
    parser.add_argument("--issue-ensure", action="store_true", help="Issue 查重与创建")
    parser.add_argument("--title", help="Issue/PR 标题")
    parser.add_argument("--body", help="Issue/PR 描述内容")
    parser.add_argument("--body-file", help="Issue/PR 描述文件路径（优先于 --body）")
    parser.add_argument("--submit-pr", action="store_true", help="push fork + 创建 PR + 触发 CI")
    parser.add_argument("--issue", dest="issue_number", type=int, help="PR 关联的 Issue 编号（记录用）")
    parser.add_argument("--base", help="PR 目标分支（默认仓默认分支）")
    parser.add_argument("--no-compile", action="store_true", help="创建 PR 后不评论 /compile")
    parser.add_argument("--ci-status", action="store_true", help="查询 CI 状态")
    parser.add_argument("--pr", dest="pr_number", type=int, help="PR 编号")
    parser.add_argument("--wait", action="store_true", help="--ci-status 轮询至终态")
    parser.add_argument("--interval", type=int, default=60, help="轮询间隔秒（默认 60）")
    parser.add_argument("--timeout", type=int, default=1800, help="轮询超时秒（默认 1800）")
    parser.add_argument("--ci-logs", action="store_true", help="收集 CI 失败日志")
    parser.add_argument("--output-dir", help="日志下载目录（默认当前目录）")
    parser.add_argument("--list-review-comments", action="store_true", help="列出未处理检视意见")
    parser.add_argument("--cleanup", action="store_true", help="清理 worktree 与临时文件")
    parser.add_argument("--worktree", help="--cleanup 要移除的 worktree 路径")
    args = parser.parse_args()

    if args.check_env:
        ok = cmd_check_env(args)
        sys.exit(0 if ok else 1)
    if args.sync_repo:
        cmd_sync_repo(args)
        return
    if args.issue_ensure:
        cmd_issue_ensure(args)
        return
    if args.submit_pr:
        cmd_submit_pr(args)
        return
    if args.ci_status:
        cmd_ci_status(args)
        return
    if args.ci_logs:
        cmd_ci_logs(args)
        return
    if args.list_review_comments:
        cmd_list_review_comments(args)
        return
    if args.cleanup:
        cmd_cleanup(args)
        return
    parser.print_help()


if __name__ == "__main__":
    main()
