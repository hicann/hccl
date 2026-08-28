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

"""PR 检视辅助工具。

为 AI Agent / 人工检视者提供 PR 检视流程中必须精确可靠的机械步骤：
取 PR 元数据、验证检视意见行号、计算 diff 行位置、按行提交检视意见、
提交检视汇总报告、清理检视 worktree。其余检视判断步骤由 SKILL.md 编排。

用法示例:
    # 首次使用前校验环境（token、git、仓库路径）
    python3 pr_review.py --check-env

    # 取 PR 元数据（状态、目标分支、base/head sha、变更文件数）
    python3 pr_review.py --pr 123 --meta

    # 校验 findings.json 中每条意见的行号
    python3 pr_review.py --pr 123 --verify-only --findings findings.json

    # 计算 diff 行位置但不提交（预演）
    python3 pr_review.py --pr 123 --dry-run --findings findings.json

    # 按行提交检视意见（自动去重，行不在 diff 时回退为 PR 评论）
    python3 pr_review.py --pr 123 --findings findings.json

    # 提交检视汇总报告评论
    python3 pr_review.py --pr 123 --report findings.json

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

# CLI 输出走 logging（stdout），便于重定向与分级
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
LOG = logging.getLogger("pr_review")

GITCODE_API_V5 = "https://gitcode.com/api/v5"
GITCODE_API_V4 = "https://api.gitcode.com/api/v4"
SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}


# ---------------------------------------------------------------------------
# 基础设施：git / HTTP / token
# ---------------------------------------------------------------------------

GIT_EXECUTABLE = shutil.which("git") or "git"


def run_git(args, cwd):
    """执行 git 命令，失败时抛出 RuntimeError。"""
    result = subprocess.run(
        [GIT_EXECUTABLE] + args,
        cwd=str(cwd), capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError("git {} failed: {}".format(" ".join(args), result.stderr.strip()))
    return result.stdout


def get_token():
    """获取 GitCode token：优先环境变量 GITCODE_TOKEN，否则 git credential fill。"""
    token = os.environ.get("GITCODE_TOKEN", "").strip()
    if token:
        return token
    try:
        out = subprocess.run(
            [GIT_EXECUTABLE, "credential", "fill"],
            input="protocol=https\nhost=gitcode.com\n\n",
            capture_output=True, text=True, encoding="utf-8", timeout=10,
        )
        for line in out.stdout.splitlines():
            if line.startswith("password="):
                return line.split("=", 1)[1].strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return ""


def api_request(url, method="GET", data=None, token=""):
    """调用 GitCode API。v5 用 access_token 查询参数，v4 用 PRIVATE-TOKEN 请求头。"""
    headers = {"Content-Type": "application/json; charset=utf-8", "User-Agent": "agents-skill-review"}
    if "api/v4" in url:
        if token:
            headers["PRIVATE-TOKEN"] = token
    else:
        if token and "access_token" not in url:
            url = "{}{}access_token={}".format(url, "&" if "?" in url else "?", token)
    body = None
    if data is not None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            if not raw.strip():
                return {"status": "ok", "http_code": resp.status}
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError("HTTP {}: {}".format(exc.code, error_body[:500])) from exc


def api_write(url, method="POST", data=None, token=""):
    """写操作（POST/PUT/DELETE）：v5 端点用 Authorization: Bearer 头认证。

    实测 v5 的 DELETE/PUT（resolve）与 discussions 线程回复必须用 Bearer 头，
    access_token 查询参数会返回 405/403。
    """
    headers = {"Content-Type": "application/json; charset=utf-8",
               "User-Agent": "agents-skill-review",
               "Authorization": "Bearer {}".format(token)}
    body = json.dumps(data, ensure_ascii=False).encode("utf-8") if data is not None else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            if not raw.strip():
                return {"status": "ok", "http_code": resp.status}
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError("HTTP {}: {}".format(exc.code, error_body[:500])) from exc


def get_current_user(token):
    """取当前认证用户登录名（v5 /user），失败返回空串。"""
    try:
        user = api_request("{}/user".format(GITCODE_API_V5), token=token)
        return (user or {}).get("login") or ""
    except RuntimeError:
        return ""


def get_findings_discussions(owner, repo, pr_number, token=""):
    """取当前认证账号发出的行内检视意见及其 discussion_id（供回复/resolve）。

    通过 v5 /user 取当前账号，只保留该账号的 diff_comment，他人意见不列入。

    返回 [{discussion_id, note_id, file, line, body, resolved, replies}]。
    v5 GET /pulls/{n}/comments 不带 discussion_id 与路径，从 v4 discussions 补齐。
    """
    comments = get_pr_comments(owner, repo, pr_number, token)
    current_user = get_current_user(token)
    mine_ids = set()
    for c in comments:
        is_mine = (c.get("user") or {}).get("login") == current_user
        if is_mine and c.get("comment_type") == "diff_comment":
            mine_ids.add(c.get("id"))
    discussions = api_get_all(
        "{}/projects/{}%2F{}/merge_requests/{}/discussions".format(GITCODE_API_V4, owner, repo, pr_number),
        token=token)
    results = []
    for disc in discussions or []:
        notes = disc.get("notes") or []
        if not notes:
            continue
        head = notes[0]
        if head.get("id") not in mine_ids:
            continue
        pos = head.get("position") or {}
        results.append({
            "discussion_id": str(disc.get("id")),
            "note_id": head.get("id"),
            "file": head.get("diff_file") or pos.get("new_path") or "",
            "line": head.get("new_line") or pos.get("new_line"),
            "body": head.get("body") or "",
            "resolved": head.get("resolved"),
            "replies": [n.get("body") for n in notes[1:] if n.get("body")],
        })
    return results


def reply_and_resolve(target, discussion_id, body, token, resolve=True):
    """对检视意见线程回复处置说明；resolve=True 时同时关闭该意见。

    线程回复走 v5 POST /pulls/{n}/discussions/{did}/comments（Bearer 头），
    回复真实挂到原检视意见线程（v4 GET discussions 核验 is_reply）；
    resolve 走 v5 PUT /pulls/{n}/comments/{did} {"resolved": true}（hex discussion_id）。
    自己提的意见修复后用本函数一站式闭环，不要发独立的顶层评论。
    target 为 PrTarget（见 make_target）。
    """
    reply = api_write(
        "{}/repos/{}/{}/pulls/{}/discussions/{}/comments".format(
            GITCODE_API_V5, target.owner, target.repo, target.pr_number, discussion_id),
        method="POST", data={"body": body}, token=token)
    result = {"reply_note_id": reply.get("note_id")}
    if resolve:
        api_write(
            "{}/repos/{}/{}/pulls/{}/comments/{}".format(
                GITCODE_API_V5, target.owner, target.repo, target.pr_number, discussion_id),
            method="PUT", data={"resolved": True}, token=token)
        result["resolved"] = True
    return result


def api_get_all(url, token=""):
    """分页拉全量列表（per_page=100 翻页）。

    兼容两种响应形态：普通 JSON 数组（v5）与 {data: [...]} 包装（v4 discussions）。
    防死循环：超出总页数后 GitCode 可能重复返回第一页内容，用首条 id
    重复检测终止翻页；另设 50 页上限兜底。
    """
    items = []
    seen_first_ids = set()
    page = 1
    max_pages = 50  # 翻页上限兜底（50 页 * 100 条），防服务端异常时无限翻页
    while page <= max_pages:
        paged = "{}{}per_page=100&page={}".format(url, "&" if "?" in url else "?", page)
        try:
            data = api_request(paged, token=token)
        except RuntimeError:
            break
        if isinstance(data, dict):
            data = data.get("data") or []
        if not isinstance(data, list) or not data:
            break
        first_id = (data[0] or {}).get("id")
        if first_id is not None:
            if first_id in seen_first_ids:
                break  # 服务端重复返回首页内容，已到末尾
            seen_first_ids.add(first_id)
        items.extend(data)
        if len(data) < 100:
            break
        page += 1
    return items


# ---------------------------------------------------------------------------
# PR 元数据
# ---------------------------------------------------------------------------

def get_pr_meta(owner, repo, pr_number, token=""):
    """取 PR 元数据，合并 v5（state/base 分支）与 v4（diff_refs 权威基线）。

    返回字段: state / target_branch / source_branch / title / body /
    head_sha / base_sha / changed_files / additions / deletions / author /
    html_url。state=="merged" 时检视意见应改走 Issue 汇总（见 SKILL.md）。
    """
    v5_url = "{}/repos/{}/{}/pulls/{}".format(GITCODE_API_V5, owner, repo, pr_number)
    pr = api_request(v5_url, token=token)
    meta = {
        "state": pr.get("state", ""),
        "target_branch": pr.get("target_branch") or (pr.get("base") or {}).get("ref", ""),
        "source_branch": pr.get("source_branch") or (pr.get("head") or {}).get("ref", ""),
        "title": pr.get("title", ""),
        "body": pr.get("body", "") or "",
        "head_sha": (pr.get("head") or {}).get("sha", ""),
        "changed_files": pr.get("changed_files", 0),
        "additions": pr.get("additions", 0),
        "deletions": pr.get("deletions", 0),
        "author": (pr.get("user") or {}).get("login", ""),
        "html_url": "https://gitcode.com/{}/{}/pull/{}".format(owner, repo, pr_number),
    }
    # v4 diff_refs 是 diff 真值；v5 base.sha 在部分场景与实际 diff 基线不一致
    v4_url = "{}/projects/{}%2F{}/merge_requests/{}".format(GITCODE_API_V4, owner, repo, pr_number)
    try:
        mr = api_request(v4_url, token=token)
        refs = mr.get("diff_refs") or {}
        if refs.get("base_sha"):
            meta["base_sha"] = refs["base_sha"]
        if refs.get("head_sha"):
            meta["head_sha"] = refs["head_sha"]
        if mr.get("state"):
            meta["state_v4"] = mr["state"]
    except RuntimeError:
        # v4 不可用时回退 v5 base
        meta["base_sha"] = (pr.get("base") or {}).get("sha", "")
    meta.setdefault("base_sha", "")
    return meta


def get_pr_files(owner, repo, pr_number, token=""):
    """取 PR 变更文件权威列表（GitCode API，不依赖本地 git 状态）。"""
    url = "{}/repos/{}/{}/pulls/{}/files".format(GITCODE_API_V5, owner, repo, pr_number)
    return api_get_all(url, token=token)


def get_pr_comments(owner, repo, pr_number, token=""):
    """取 PR 全部评论（供去重）。

    v5 comments 的 path 字段常为 None，行内评论的文件路径靠 v4 discussions
    的 note.diff_file / position.new_path 补齐；v4 不可用时仅按标题与回退格式判重。
    """
    url = "{}/repos/{}/{}/pulls/{}/comments".format(GITCODE_API_V5, owner, repo, pr_number)
    comments = api_get_all(url, token=token)
    # v4 discussions enrich（best-effort）
    try:
        v4_url = "{}/projects/{}%2F{}/merge_requests/{}/discussions".format(
            GITCODE_API_V4, owner, repo, pr_number)
        discussions = api_get_all(v4_url, token=token)
    except RuntimeError:
        discussions = []
    v4_index = {}
    for disc in discussions or []:
        for note in disc.get("notes") or []:
            if not note.get("id"):
                continue
            pos = note.get("position") or {}
            path = note.get("diff_file") or pos.get("new_path")
            if path:
                v4_index[note["id"]] = path
    for comment in comments:
        if not comment.get("path"):
            dp = comment.get("diff_position") or {}
            path = dp.get("new_path") or dp.get("path")
            if not path:
                path = v4_index.get(comment.get("id"))
            if path:
                comment["path"] = path
    return comments


def find_remote(repo_root, owner="", repo=""):
    """自动探测指向目标仓（owner/repo）的 remote 名；无精确匹配时退回第一个 fetch remote。"""
    out = run_git(["remote", "-v"], repo_root)
    if owner and repo:
        # 归一化 URL：去协议前缀/SCP 风格 host:/、去 .git 后缀，再按 owner/repo 尾段匹配
        for line in out.splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            path = parts[1].rstrip("/")
            if path.endswith(".git"):
                path = path[:-len(".git")]
            if "://" in path:
                path = path.split("://", 1)[1].split("/", 1)[-1]
            elif ":" in path:
                path = path.rsplit(":", 1)[-1]
            segs = [s for s in path.split("/") if s]
            if len(segs) >= 2 and segs[-2] == owner and segs[-1] == repo:
                return parts[0]
    # 无精确匹配：优先指向官方组织（cann）的 remote，再退回第一个 fetch remote
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and "(fetch)" in line and "/cann/" in parts[1]:
            return parts[0]
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and "(fetch)" in line:
            return parts[0]
    raise RuntimeError("未找到可用的 git remote，请在仓库内运行或用 --repo-root 指定路径")


def ensure_pr_refs(repo_root, remote, pr_number, meta):
    """fetch PR head ref 与目标分支，返回 (head_sha, base_sha)。

    检视基线一律以 GitCode diff_refs 为准（meta 里的 base_sha），
    本地只负责把 head_sha fetch 下来供 git show / git diff 使用。
    """
    head_ref = "refs/remotes/{}/mr/{}-head".format(remote, pr_number)
    try:
        run_git(["rev-parse", "--verify", head_ref], repo_root)
    except RuntimeError:
        run_git([
            "fetch", remote,
            "refs/merge-requests/{}/head:{}".format(pr_number, head_ref),
        ], repo_root)
    local_head = run_git(["rev-parse", head_ref], repo_root).strip()
    if meta.get("head_sha") and local_head != meta["head_sha"]:
        # 作者 force-push 过，强制更新本地 ref
        run_git([
            "fetch", remote,
            "refs/merge-requests/{}/head:{}".format(pr_number, head_ref),
            "--force",
        ], repo_root)
        local_head = run_git(["rev-parse", head_ref], repo_root).strip()
    base_sha = meta.get("base_sha") or ""
    if not base_sha:
        base_sha = run_git(["merge-base", head_ref, "{}/master".format(remote)], repo_root).strip()
    return local_head, base_sha


# ---------------------------------------------------------------------------
# 行号验证与 diff 位置计算
# ---------------------------------------------------------------------------

def verify_line_number(repo_root, head_sha, filepath, line, snippet=""):
    """验证 head_sha 版本文件的第 line 行存在，且内容含 snippet 片段（若提供）。"""
    try:
        content = run_git(["show", "{}:{}".format(head_sha, filepath)], repo_root)
    except RuntimeError:
        return {"verified": False, "reason": "file not found at HEAD", "actual": ""}
    lines = content.splitlines()
    if not (1 <= line <= len(lines)):
        return {"verified": False, "reason": "line {} out of range (total {})".format(line, len(lines)), "actual": ""}
    actual = lines[line - 1].strip()
    if snippet and snippet.strip() and snippet.strip() not in actual:
        return {"verified": False, "reason": "snippet mismatch", "actual": actual}
    return {"verified": True, "reason": "", "actual": actual}


def find_diff_position(repo_root, base_sha, head_sha, filepath, target_line):
    """验证目标行在 diff 的新增/修改行中，返回文件行号作为 GitCode position。

    GitCode v5 的 position 参数是文件行数（代码所在行号），不是 diff hunk 内偏移。
    只有 diff 中的新增(+)行才能挂行内评论；已有未变更行返回 None（回退 PR 评论）。
    """
    diff = run_git(["diff", "{}..{}".format(base_sha, head_sha), "--", filepath], repo_root)
    diff_lines = diff.splitlines()
    current_new_line = 0
    in_hunk = False
    for line in diff_lines:
        if line.startswith("@@"):
            match = re.search(r"\+(\d+)", line)
            if match:
                current_new_line = int(match.group(1))
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if line.startswith("+++") or line.startswith("---"):
            # diff 头标记行，不是内容行（跳过，不推进行号）
            continue
        if line.startswith("+"):
            if current_new_line == target_line:
                return target_line
            current_new_line += 1
        elif line.startswith("-"):
            pass
        else:
            if current_new_line == target_line:
                return target_line
            current_new_line += 1
    return None


def check_baseline_drift(repo_root, base_sha, head_sha, meta):
    """核验本地 diff 文件数与 PR changed_files 是否一致，防基线漂移误检视。"""
    out = run_git(["diff", "--name-only", "{}..{}".format(base_sha, head_sha)], repo_root)
    local_files = [f for f in out.splitlines() if f.strip()]
    expected = meta.get("changed_files", 0)
    # 文件数远超预期（>2 倍且多 5 个以上）视为基线漂移
    if expected and len(local_files) > max(expected * 2, expected + 5):
        return {
            "drift": True,
            "local_files": len(local_files),
            "expected": expected,
            "hint": "本地 diff 文件数({})远超 PR changed_files({})，"
                    "疑似基线漂移。请勿据此检视，"
                    "改用 --pr N --files 取权威文件列表".format(len(local_files), expected),
        }
    return {"drift": False, "local_files": len(local_files), "expected": expected}


# ---------------------------------------------------------------------------
# 检视意见提交
# ---------------------------------------------------------------------------

# PR 目标封装（owner/repo/pr_number 三元组，供评论提交函数使用）
class PrTarget:
    """GitCode PR 定位信息：owner/repo/pr_number。"""

    def __init__(self, owner, repo, pr_number):
        self.owner = owner
        self.repo = repo
        self.pr_number = pr_number


def make_target(owner, repo, pr_number):
    """构造 PrTarget。"""
    return PrTarget(owner, repo, pr_number)


def build_comment_body(finding):
    """构造检视意见评论正文（含严重级别与建议修复）。"""
    severity = finding.get("severity", "MEDIUM")
    title = finding.get("title", "")
    body = finding.get("body", "")
    return "**[{}] {}**\n\n{}".format(severity, title, body)


def dedup_findings(findings, existing_comments):
    """内容指纹去重：已有评论命中的 file+line 或正文标题相同则跳过。

    返回 (deduped, skipped)：deduped 为待提交列表，skipped 为重复项（含命中原因）。
    已有评论的文件路径从 path / diff_position.new_path 取；两者都缺时
    用回退格式正文里的 "> file:line" 引用行解析。
    """
    seen = {}
    for comment in existing_comments:
        body = comment.get("body", "") or ""
        position = comment.get("diff_position") or {}
        line = position.get("start_new_line")
        path = comment.get("path") or position.get("new_path") or position.get("path")
        # 已有评论正文里解析文件位置（pr_comment 回退格式 "> file:line"）
        quote = re.search(r"^>\s*([\w./+-]+):(\d+)", body, re.M)
        if line is None and quote:
            path, line = quote.group(1), int(quote.group(2))
        if path and line:
            seen.setdefault((path, int(line)), comment)
        title_match = re.match(r"\*\*\[(\w+)\]\s*(.+?)\*\*", body)
        if title_match:
            seen.setdefault(("title", title_match.group(2).strip()), comment)
    deduped, skipped = [], []
    for finding in findings:
        key_line = (finding.get("file", ""), int(finding.get("line", 0)))
        key_title = ("title", finding.get("title", ""))
        if key_line in seen:
            skipped.append({"finding": finding, "reason": "file+line 已有评论"})
        elif key_title in seen:
            skipped.append({"finding": finding, "reason": "标题与已有评论重复"})
        else:
            deduped.append(finding)
    return deduped, skipped


def post_finding(target, head_sha, finding, position, token):
    """提交单条检视意见。position 有效走行内评论，否则回退 PR 评论。

    target 为 PrTarget（owner/repo/pr_number 封装，见 make_target）。
    """
    url = "{}/repos/{}/{}/pulls/{}/comments".format(
        GITCODE_API_V5, target.owner, target.repo, target.pr_number)
    body = build_comment_body(finding)
    if position:
        payload = {
            "body": body,
            "commit_id": head_sha,
            "path": finding["file"],
            "position": position,
        }
    else:
        payload = {"body": "> {}:{}\n\n{}".format(finding["file"], finding["line"], body)}
    return api_request(url, method="POST", data=payload, token=token)


def post_report(target, findings, meta, token):
    """生成 Markdown 检视汇总报告并提交为 PR 评论。"""
    stats = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for f in findings:
        stats[f.get("severity", "LOW")] = stats.get(f.get("severity", "LOW"), 0) + 1
    lines = [
        "## 代码检视汇总报告",
        "",
        "- PR: [#{}]({})".format(target.pr_number, meta.get("html_url", "")),
        "- 标题: {}".format(meta.get("title", "")),
        "- 检视基准: head `{}` / base `{}`".format(meta.get("head_sha", "")[:12], meta.get("base_sha", "")[:12]),
        "- 检视意见: {} 条（CRITICAL {} / HIGH {} / MEDIUM {} / LOW {}）".format(
            len(findings), stats["CRITICAL"], stats["HIGH"], stats["MEDIUM"], stats["LOW"]),
        "",
        "| # | 级别 | 位置 | 维度 | 摘要 |",
        "|---|------|------|------|------|",
    ]
    ordered = sorted(findings, key=lambda f: SEVERITY_ORDER.get(f.get("severity", "LOW"), 9))
    for i, f in enumerate(ordered, 1):
        lines.append("| {} | {} | `{}:{}` | {} | {} |".format(
            i, f.get("severity", ""), f.get("file", ""), f.get("line", ""),
            f.get("dimension", ""), f.get("title", "").replace("|", "\\|")))
    lines.append("")
    lines.append("详细意见已按行提交（见上方行内评论）。每条意见均含建议修复方式。")
    url = "{}/repos/{}/{}/pulls/{}/comments".format(
        GITCODE_API_V5, target.owner, target.repo, target.pr_number)
    return api_request(url, method="POST", data={"body": "\n".join(lines)}, token=token)


# ---------------------------------------------------------------------------
# 子命令实现
# ---------------------------------------------------------------------------

def cmd_check_env(args):
    """校验运行环境：token、git、仓库路径、remote。"""
    LOG.info("=" * 60)
    LOG.info("环境检查")
    LOG.info("=" * 60)
    ok = True

    token = get_token()
    if token:
        LOG.info("  [OK] GITCODE_TOKEN: 已配置")
        # 探活：查一次公开端点验证 token 有效性
        try:
            api_request("{}/repos/{}/{}/pulls/1".format(GITCODE_API_V5, args.owner, args.repo), token=token)
            LOG.info("  [OK] GitCode API 可达")
        except RuntimeError as exc:
            LOG.info("  [OK] GitCode API 探测: {}（token 无权限读该端点不影响检视公开 PR）"
                     .format(str(exc)[:80]))
    else:
        ok = False
        LOG.info("  [FAIL] GITCODE_TOKEN 未配置。配置方式二选一：")
        LOG.info("         1) export GITCODE_TOKEN=<你的token>（Windows: set 或 $env:GITCODE_TOKEN）")
        LOG.info("         2) git credential fill 自动读取（需已 git clone 凭据）")

    repo_root = Path(args.repo_root).resolve()
    if (repo_root / ".git").exists():
        LOG.info("  [OK] 仓库路径: {}".format(repo_root))
        try:
            remote = find_remote(repo_root, args.owner, args.repo)
            url = run_git(["remote", "get-url", remote], repo_root).strip()
            LOG.info("  [OK] remote '{}' -> {}".format(remote, url))
        except RuntimeError as exc:
            ok = False
            LOG.info("  [FAIL] remote 探测失败: {}".format(exc))
    else:
        ok = False
        LOG.info("  [FAIL] 仓库路径无效: {}（--repo-root 指向本仓 clone 的本地路径）"
                 .format(repo_root))

    LOG.info("环境检查 {}".format("通过" if ok else "未通过，请按上述提示配置后重试"))
    return 0 if ok else 1


def cmd_meta(args, token):
    """输出 PR 元数据 JSON。"""
    meta = get_pr_meta(args.owner, args.repo, args.pr_number, token)
    LOG.info(json.dumps(meta, ensure_ascii=False, indent=2))
    if meta.get("state") == "merged":
        LOG.info("\n[提示] 该 PR 已合入（state=merged），无法按行提交检视意见，"
              "请将检视意见汇总后通过 Issue 跟踪（见 SKILL.md「已合入 PR」一节）。")
    return 0


def cmd_files(args, token):
    """输出 PR 变更文件权威列表。"""
    files = get_pr_files(args.owner, args.repo, args.pr_number, token)
    for f in files:
        LOG.info("{}\t+{}\t-{}".format(f.get("filename", ""), f.get("additions", 0), f.get("deletions", 0)))
    LOG.info("\n共 {} 个文件".format(len(files)))
    return 0


def load_findings(path):
    """加载 findings.json 并校验 schema。"""
    p = Path(path)
    if not p.exists():
        raise RuntimeError("findings 文件不存在: {}".format(path))
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]
    for i, f in enumerate(data):
        for field in ("severity", "file", "line", "title", "body"):
            if field not in f:
                raise RuntimeError("findings[{}] 缺少必填字段 '{}'（schema 见 SKILL.md）".format(i, field))
    return data


def cmd_review(args, token):
    """verify-only / dry-run / POST 三档检视意见处理。"""
    repo_root = Path(args.repo_root).resolve()
    meta = get_pr_meta(args.owner, args.repo, args.pr_number, token)

    LOG.info("=" * 60)
    LOG.info("PR #{} 检视意见处理 | state={} | 目标分支={}".format(
        args.pr_number, meta.get("state"), meta.get("target_branch")))
    LOG.info("=" * 60)

    if meta.get("state") == "merged" and not args.dry_run and not args.verify_only:
        LOG.info("[终止] PR 已合入（state=merged），无法按行提交检视意见。"
              "请改走 Issue 汇总流程（见 SKILL.md）。")
        return 1

    findings = load_findings(args.findings)
    LOG.info("加载 {} 条检视意见".format(len(findings)))

    remote = find_remote(repo_root, args.owner, args.repo)
    head_sha, base_sha = ensure_pr_refs(repo_root, remote, args.pr_number, meta)
    if args.head_sha:
        head_sha = args.head_sha
    if args.base_sha:
        base_sha = args.base_sha
    LOG.info("head_sha={} base_sha={}".format(head_sha[:12], base_sha[:12]))

    # 基线漂移核验
    drift = check_baseline_drift(repo_root, base_sha, head_sha, meta)
    if drift["drift"]:
        LOG.info("[警告] {}".format(drift["hint"]))
        if not args.dry_run and not args.verify_only:
            LOG.info("[终止] 基线漂移，拒绝提交，避免把意见发到非本 PR 的代码上。")
            return 1
    else:
        LOG.info("[OK] 本地 diff 文件数 {} 与 changed_files {} 一致性通过".format(
            drift["local_files"], drift["expected"]))

    # Step 1: 行号验证
    LOG.info("\n--- Step 1: 行号验证 ---")
    verified = []
    for i, f in enumerate(findings, 1):
        result = verify_line_number(repo_root, head_sha, f["file"], f["line"], f.get("code_snippet", ""))
        status = "OK " if result["verified"] else "FAIL"
        LOG.info("  [{}] {}/{} {} {}:{}".format(status, i, len(findings), f["severity"], f["file"], f["line"]))
        if not result["verified"]:
            LOG.info("        原因: {} | 实际内容: {}".format(result["reason"], result["actual"][:60]))
        f["_verified"] = result["verified"]
        verified.append(f)
    unverified = [f for f in verified if not f["_verified"]]
    if unverified:
        LOG.info("\n[警告] {} 条意见行号未验证通过，请用 git show HEAD:path | grep -n 修正后重跑".format(len(unverified)))
        if not args.dry_run and not args.verify_only:
            LOG.info("[终止] 存在未验证行号，拒绝提交。")
            return 1
    if args.verify_only:
        LOG.info("\nverify-only 完成。")
        return 0

    # Step 2: position 计算
    LOG.info("\n--- Step 2: diff 位置计算 ---")
    for f in verified:
        pos = find_diff_position(repo_root, base_sha, head_sha, f["file"], f["line"])
        f["_position"] = pos
        LOG.info("  [{}] {}:{} -> position={}".format("OK " if pos else "FALLBACK", f["file"], f["line"], pos))
    if args.dry_run:
        postable = sum(1 for f in verified if f["_position"])
        LOG.info("\ndry-run 完成：{} 条可发行内评论，{} 条将回退为 PR 评论。".format(
            postable, len(verified) - postable))
        return 0

    # Step 3: 去重
    LOG.info("\n--- Step 3: 已有评论去重 ---")
    existing = get_pr_comments(args.owner, args.repo, args.pr_number, token)
    deduped, skipped = dedup_findings(verified, existing)
    LOG.info("  已有评论 {} 条，命中重复跳过 {} 条".format(len(existing), len(skipped)))
    for s in skipped:
        LOG.info("    [跳过] {}:{} ({})".format(s["finding"]["file"], s["finding"]["line"], s["reason"]))

    # Step 4: 提交
    LOG.info("\n--- Step 4: 提交检视意见 ---")
    success, failed = 0, 0
    for i, f in enumerate(deduped, 1):
        try:
            target = make_target(args.owner, args.repo, args.pr_number)
            result = post_finding(target, head_sha, f, f["_position"], token)
            ctype = result.get("comment_type", "N/A")
            LOG.info("  [{}/{}] {} {}:{} -> id={} type={}".format(
                i, len(deduped), f["severity"], f["file"], f["line"], result.get("id", "?"), ctype))
            success += 1
        except RuntimeError as exc:
            LOG.info("  [{}/{}] FAIL {}:{}: {}".format(i, len(deduped), f["file"], f["line"], str(exc)[:120]))
            failed += 1
        time.sleep(args.delay)

    LOG.info("\n结果: 提交 {} 条 / 失败 {} 条 / 去重跳过 {} 条 / 行号未验证 0 条".format(
        success, failed, len(skipped)))
    return 0 if failed == 0 else 1


def cmd_report(args, token):
    """提交检视汇总报告评论。"""
    meta = get_pr_meta(args.owner, args.repo, args.pr_number, token)
    findings = load_findings(args.findings)
    result = post_report(make_target(args.owner, args.repo, args.pr_number), findings, meta, token)
    LOG.info("汇总报告已提交: id={}".format(result.get("id", "?")))
    return 0


def cmd_list_findings(args, token):
    """列出本账号在该 PR 发出的行内检视意见（含 resolved 状态与 discussion_id）。"""
    findings = get_findings_discussions(args.owner, args.repo, args.pr_number, token)
    for f in findings:
        status = "resolved" if f["resolved"] else "open"
        title = (f["body"] or "").splitlines()[0][:50]
        LOG.info("[{}] {}:{} | did={} | {}".format(
            status, f["file"], f["line"], str(f["discussion_id"])[:12], title))
    resolved_cnt = sum(1 for f in findings if f["resolved"])
    LOG.info("共 {} 条（resolved {} / open {}）".format(len(findings), resolved_cnt, len(findings) - resolved_cnt))
    return 0


def _locate_finding(findings, args):
    """按 note_id 或 line(+file) 定位目标检视意见，找不到返回 None。"""
    for f in findings:
        if args.note_id:
            if f["note_id"] == args.note_id:
                return f
            continue
        line_match = f["line"] == args.line
        file_match = not args.file or f["file"] == args.file
        if line_match and file_match:
            return f
    return None


def cmd_dispose(args, token):
    """对检视意见处置：线程回复 + resolve（自提意见修复后的标准闭环）。

    --line 定位意见（同文件多意见时用 --note-id 精确定位）；
    --body 回复内容（如"已修复（commit xxx）：..."）；
    默认 resolve 关闭，--no-resolve 只回复不关闭。
    """
    findings = get_findings_discussions(args.owner, args.repo, args.pr_number, token)
    target = _locate_finding(findings, args)
    if not target:
        LOG.error("未找到目标检视意见（line={} note_id={}），用 --list-findings 查看全部".format(
            args.line, args.note_id))
        return 1
    if target["resolved"] and not args.no_resolve:
        LOG.info("意见 {}:{} 已是 resolved 状态，只回复不重复关闭".format(target["file"], target["line"]))
        pr_target = make_target(args.owner, args.repo, args.pr_number)
        result = reply_and_resolve(pr_target, target["discussion_id"], args.body, token, resolve=False)
    else:
        pr_target = make_target(args.owner, args.repo, args.pr_number)
        result = reply_and_resolve(pr_target, target["discussion_id"], args.body, token,
                                   resolve=not args.no_resolve)
    LOG.info("已回复线程（note_id={}）{}".format(
        result.get("reply_note_id", "?"),
        " 并 resolve 关闭" if result.get("resolved") else ""))
    return 0


def cmd_cleanup(args, token):
    """清理检视 worktree 与临时文件。"""
    repo_root = Path(args.repo_root).resolve()
    if args.worktree:
        wt = Path(args.worktree)
        if wt.exists():
            subprocess.run([GIT_EXECUTABLE, "worktree", "remove", "--force", str(wt)],
                           cwd=str(repo_root), capture_output=True, text=True)
            LOG.info("已移除 worktree: {}".format(wt))
        else:
            LOG.info("worktree 不存在: {}".format(wt))
        subprocess.run([GIT_EXECUTABLE, "worktree", "prune"], cwd=str(repo_root), capture_output=True, text=True)
        LOG.info("已执行 git worktree prune")
    # 清理临时 findings 文件：优先 --findings 指定文件，另在 repo-root 与 cwd 两处 glob
    # （脚本可能从任意 cwd 启动，findings 落盘位置与 repo-root 可能不同）
    candidates = []
    if args.findings:
        candidates.append(Path(args.findings))
    for base in (repo_root, Path(".")):
        for pattern in ("findings*.json",):
            candidates.extend(base.glob(pattern))
    removed = set()
    for path in candidates:
        if path.is_file() and str(path) not in removed:
            path.unlink()
            removed.add(str(path))
            LOG.info("已删除临时文件: {}".format(path))
    return 0


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", default=".", help="本仓本地 clone 路径（默认当前目录）")
    parser.add_argument("--owner", default="cann", help="PR 所在仓库 owner（默认 cann）")
    parser.add_argument("--repo", help="仓库名（hcomm/hccl，默认从 remote URL 推断）")
    parser.add_argument("--pr", dest="pr_number", type=int, help="PR 编号")
    parser.add_argument("--findings", help="findings.json 路径（--verify-only/--dry-run/POST 必填）")
    parser.add_argument("--check-env", action="store_true", help="校验运行环境")
    parser.add_argument("--meta", action="store_true", help="输出 PR 元数据 JSON")
    parser.add_argument("--files", action="store_true", help="输出 PR 变更文件权威列表")
    parser.add_argument("--verify-only", action="store_true", help="只验证行号，不计算 position 不提交")
    parser.add_argument("--dry-run", action="store_true", help="验证行号+计算 position，不提交")
    parser.add_argument("--report", action="store_true", help="提交检视汇总报告评论")
    parser.add_argument("--cleanup", action="store_true", help="清理 worktree 与临时文件")
    parser.add_argument("--list-findings", action="store_true", help="列出本账号在该 PR 的检视意见与状态")
    parser.add_argument("--dispose", action="store_true", help="对检视意见处置：线程回复+resolve")
    parser.add_argument("--line", type=int, help="--dispose 时按行号定位意见")
    parser.add_argument("--note-id", type=int, help="--dispose 时按 note_id 精确定位意见")
    parser.add_argument("--no-resolve", action="store_true", help="--dispose 时只回复不关闭")
    parser.add_argument("--body", help="--dispose 时的回复内容")
    parser.add_argument("--file", help="--dispose 时可选：按文件路径精确定位意见")
    parser.add_argument("--worktree", help="--cleanup 时要移除的 worktree 路径")
    parser.add_argument("--head-sha", help="覆盖 head sha（默认取 PR 元数据）")
    parser.add_argument("--base-sha", help="覆盖 base sha（默认取 v4 diff_refs）")
    parser.add_argument("--delay", type=float, default=0.5, help="提交间隔秒数（默认 0.5）")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.check_env:
        return cmd_check_env(args)

    # 推断 repo 名
    if not args.repo:
        try:
            remote_url = run_git(["remote", "get-url", find_remote(Path(args.repo_root).resolve())],
                                 Path(args.repo_root).resolve()).strip()
            # 取 URL 路径最后一段（去掉 .git 后缀；兼容 SSH 格式 git@host:owner/repo.git）
            path = remote_url.rstrip("/")
            if path.endswith(".git"):
                path = path[:-len(".git")]
            if ":" in path and not path.startswith(("http://", "https://")):
                path = path.rsplit(":", 1)[-1]
            args.repo = path.rsplit("/", 1)[-1] if "/" in path else ""
        except (RuntimeError, OSError):
            args.repo = ""
    needs_repo = args.meta or args.files or args.findings or args.report
    if not args.repo and needs_repo:
        LOG.error("错误: 无法推断仓库名，请用 --repo hcomm|hccl 指定")
        return 1

    if args.meta:
        if not args.pr_number:
            LOG.error("错误: --meta 需要 --pr 编号")
            return 1
        return cmd_meta(args, get_token())

    if args.files:
        if not args.pr_number:
            LOG.error("错误: --files 需要 --pr 编号")
            return 1
        return cmd_files(args, get_token())

    if args.report:
        if not args.pr_number or not args.findings:
            LOG.error("错误: --report 需要 --pr 与 --findings")
            return 1
        return cmd_report(args, get_token())

    if args.list_findings:
        if not args.pr_number:
            LOG.error("错误: --list-findings 需要 --pr 编号")
            return 1
        return cmd_list_findings(args, get_token())

    if args.dispose:
        dispose_ready = args.pr_number and args.body and (args.line or args.note_id)
        if not dispose_ready:
            LOG.error("错误: --dispose 需要 --pr、--body、(--line 或 --note-id)")
            return 1
        return cmd_dispose(args, get_token())

    if args.cleanup:
        return cmd_cleanup(args, get_token())

    if args.findings:
        if not args.pr_number:
            LOG.error("错误: 检视意见处理需要 --pr 编号")
            return 1
        return cmd_review(args, get_token())

    LOG.info("未指定操作。用法见 --help")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # pylint: disable=broad-except
        LOG.info("ERROR: {}".format(exc))
        sys.exit(1)
