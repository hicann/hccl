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

"""contribute.py 单元测试（mock git 仓 + mock API，无网络依赖）。

运行: python3 -m unittest test_contribute
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contribute  # noqa: E402


class GitRepoFixture(unittest.TestCase):
    """构造真实临时 git 仓（hccl fork 布局）供测试。

    upstream remote 默认指向假 URL（探测类测试用）；需要真实 fetch 的测试
    调 attach_local_upstream() 换成本地 bare 仓。
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.repo = self.root / "hccl"
        self.repo.mkdir()
        self.git("init", "-q", "-b", "master")
        self.git("config", "user.name", "tester")
        self.git("config", "user.email", "tester@example.com")
        self.git("config", "commit.gpgsign", "false")
        self.write("README.md", "# hccl\n")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "init")
        self.git("remote", "add", "upstream", "https://gitcode.com/cann/hccl.git")
        self.git("remote", "add", "origin", "https://gitcode.com/tester/hccl.git")

    def attach_local_upstream(self):
        """把 upstream 换成本地 bare 仓并推送初始提交（fetch 类测试用）。

        bare 放在 <root>/cann/hccl.git 路径下，使 URL 归一化后仍是 cann/hccl，
        find_remote/detect_repo_layout 逻辑与真实环境一致。
        """
        bare = self.root / "cann" / "hccl.git"
        bare.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([contribute.GIT_EXECUTABLE, "init", "-q", "--bare", str(bare)],
                       check=True, capture_output=True)
        self.git("remote", "set-url", "upstream", str(bare))
        self.git("push", "-q", "upstream", "master:master")
        self.git("fetch", "-q", "upstream")
        return bare

    def tearDown(self):
        self.tmp.cleanup()

    def git(self, *args, cwd=None):
        result = subprocess.run(
            [contribute.GIT_EXECUTABLE] + list(args),
            cwd=str(cwd or self.repo), capture_output=True, text=True, encoding="utf-8",
        )
        if result.returncode != 0:
            raise RuntimeError("git {} failed: {}".format(args, result.stderr))
        return result.stdout

    def write(self, rel, content):
        target = self.repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def make_feature_branch(self, name="feat/test-branch"):
        self.git("checkout", "-q", "-b", name)
        self.write("docs/test.md", "# test\n")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "test commit")


class TestNormalizeRemoteUrl(unittest.TestCase):
    """remote URL 归一化（HTTPS/SSH/.git 后缀）。"""

    def test_https_with_git_suffix(self):
        self.assertEqual(contribute.normalize_remote_url(
            "https://gitcode.com/cann/hccl.git"), "cann/hccl")

    def test_https_without_suffix(self):
        self.assertEqual(contribute.normalize_remote_url(
            "https://gitcode.com/tester/hccl"), "tester/hccl")

    def test_ssh_scp_form(self):
        self.assertEqual(contribute.normalize_remote_url(
            "git@gitcode.com:cann/hccl.git"), "cann/hccl")

    def test_local_path(self):
        self.assertEqual(contribute.normalize_remote_url(
            "/mnt/d/work/hccl"), "work/hccl")


class TestFindRemote(GitRepoFixture):
    """remote 精确探测。"""

    def test_find_upstream(self):
        remote = contribute.find_remote(self.repo, "cann", "hccl")
        self.assertEqual(remote, "upstream")

    def test_find_fork(self):
        remote = contribute.find_remote(self.repo, "tester", "hccl")
        self.assertEqual(remote, "origin")

    def test_not_found(self):
        remote = contribute.find_remote(self.repo, "someone", "hccl")
        self.assertIsNone(remote)


class TestDetectLayout(GitRepoFixture):
    """仓库布局探测（upstream/fork/base_branch）。"""

    def test_layout(self):
        self.attach_local_upstream()
        layout = contribute.detect_repo_layout(self.repo, "hccl")
        self.assertEqual(layout["upstream_remote"], "upstream")
        self.assertEqual(layout["fork_remote"], "origin")
        self.assertEqual(layout["fork_owner"], "tester")
        self.assertEqual(layout["base_branch"], "master")


class TestSyncRepo(GitRepoFixture):
    """--sync-repo 三态：up_to_date / fetched_worktree / clone。"""

    def test_up_to_date_when_clean_on_base(self):
        self.attach_local_upstream()
        args = mock.Mock(repo_root=str(self.repo), repo="hccl", parent_dir=None)
        with mock.patch.object(contribute, "emit_json") as emit:
            contribute.cmd_sync_repo(args)
        payload = emit.call_args[0][0]
        self.assertEqual(payload["action"], "up_to_date")

    def test_dirty_creates_worktree(self):
        self.attach_local_upstream()
        self.write("dirty.txt", "uncommitted")
        args = mock.Mock(repo_root=str(self.repo), repo="hccl", parent_dir=None)
        with mock.patch.object(contribute, "emit_json") as emit:
            contribute.cmd_sync_repo(args)
        payload = emit.call_args[0][0]
        self.assertEqual(payload["action"], "fetched_worktree")
        self.assertTrue(Path(payload["repo_root"]).exists())
        # 清理
        self.git("worktree", "remove", "--force", payload["repo_root"])

    def test_dirty_worktree_rerun_reuses_branch(self):
        # 同一 upstream sha 二次运行：分支已存在时复用分支而非 -b 重建（曾 fatal）
        self.attach_local_upstream()
        self.write("dirty.txt", "uncommitted")
        args = mock.Mock(repo_root=str(self.repo), repo="hccl", parent_dir=None)
        with mock.patch.object(contribute, "emit_json") as emit:
            contribute.cmd_sync_repo(args)
        first_wt = emit.call_args[0][0]["repo_root"]
        self.git("worktree", "remove", "--force", first_wt)
        # 二次运行：worktree 已移除但分支仍在
        with mock.patch.object(contribute, "emit_json") as emit2:
            contribute.cmd_sync_repo(args)
        payload = emit2.call_args[0][0]
        self.assertEqual(payload["action"], "fetched_worktree")  # 不再 fatal
        self.git("worktree", "remove", "--force", payload["repo_root"])

    def test_local_commits_on_master_not_discarded(self):
        # 本地 master 有自有提交（非上游祖先）时不得 reset --hard 丢弃，改走 worktree 隔离
        self.attach_local_upstream()
        self.git("checkout", "-q", "-b", "local-work")
        self.write("mywork.md", "local work")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "local work")
        self.git("checkout", "-q", "master")
        self.git("merge", "-q", "--ff-only", "local-work")
        args = mock.Mock(repo_root=str(self.repo), repo="hccl", parent_dir=None)
        with mock.patch.object(contribute, "emit_json") as emit:
            contribute.cmd_sync_repo(args)
        payload = emit.call_args[0][0]
        # 本地提交被隔离保留，不被静默丢弃
        self.assertEqual(payload["action"], "fetched_worktree")
        mywork = Path(payload["repo_root"]) / "mywork.md"
        self.assertTrue(mywork.exists(), "本地自有提交的文件必须在隔离 worktree 中存活")
        self.git("worktree", "remove", "--force", payload["repo_root"])

    def test_fetch_moves_base_forward(self):
        bare = self.attach_local_upstream()
        # 在 upstream bare 上造新提交（通过临时 clone）
        work = self.root / "upwork"
        subprocess.run([contribute.GIT_EXECUTABLE, "clone", "-q",
                        str(bare), str(work)], check=True, capture_output=True)
        subprocess.run([contribute.GIT_EXECUTABLE, "-C", str(work), "config",
                        "user.name", "upstreamer"], check=True, capture_output=True)
        subprocess.run([contribute.GIT_EXECUTABLE, "-C", str(work), "config",
                        "user.email", "up@example.com"], check=True, capture_output=True)
        subprocess.run([contribute.GIT_EXECUTABLE, "-C", str(work), "commit", "-q",
                        "--allow-empty", "-m", "upstream new"],
                       check=True, capture_output=True)
        subprocess.run([contribute.GIT_EXECUTABLE, "-C", str(work), "push", "-q",
                        "origin", "master"], check=True, capture_output=True)
        args = mock.Mock(repo_root=str(self.repo), repo="hccl", parent_dir=None)
        with mock.patch.object(contribute, "emit_json") as emit:
            contribute.cmd_sync_repo(args)
        payload = emit.call_args[0][0]
        self.assertEqual(payload["action"], "fetched_rebase")
        # head 已跟进 upstream
        self.assertEqual(payload["head_sha"], payload["upstream_sha"])


class TestIssueKeywords(unittest.TestCase):
    """Issue 标题关键词提取（中文 2 字滑窗 + 英文整词）。"""

    def test_strips_requirement_prefix(self):
        kws = contribute.issue_keywords("[Requirement|需求建议]: 仓内新增贡献流程skill")
        self.assertIn("skill", kws)
        self.assertTrue(any(k in kws for k in ("贡献", "流程")))
        self.assertFalse(any("Requirement" in k for k in kws))

    def test_strips_cn_prefix(self):
        kws = contribute.issue_keywords("【需求】仓内新增贡献流程skill")
        self.assertIn("skill", kws)
        self.assertFalse(any(k.startswith("需求】") for k in kws))

    def test_english_only(self):
        kws = contribute.issue_keywords("[Bug-Report|缺陷反馈]: hccl build failure")
        self.assertIn("hccl", kws)
        self.assertIn("build", kws)
        self.assertIn("failure", kws)


def make_issue_args(title, body="desc"):
    """构造 --issue-ensure 的 args 替身。"""
    return mock.Mock(repo="hccl", title=title, body=body, body_file=None)


class TestIssueEnsure(unittest.TestCase):
    """--issue-ensure：查重命中复用 / 未命中创建。"""

    def test_reuse_when_open_match(self):
        issues = [{"number": 42, "title": "[Requirement|需求建议]: 仓内新增贡献流程skill", "state": "open"}]
        with mock.patch.object(contribute, "api_get_all", return_value=issues), \
             mock.patch.object(contribute, "emit_json") as emit, \
             mock.patch.object(contribute, "get_token", return_value="t"):
            contribute.cmd_issue_ensure(make_issue_args("仓内新增贡献流程skill"))
        payload = emit.call_args[0][0]
        self.assertEqual(payload["action"], "reused")
        self.assertEqual(payload["existing"][0]["number"], 42)

    def test_create_when_no_match(self):
        created = {"number": 100, "html_url": "https://gitcode.com/cann/hccl/issues/100"}
        responses = iter([
            (201, created),  # POST 创建
        ])

        def fake_request(method, path, token, payload=None):
            return next(responses)

        with mock.patch.object(contribute, "api_get_all", return_value=[]), \
             mock.patch.object(contribute, "api_request", side_effect=fake_request), \
             mock.patch.object(contribute, "emit_json") as emit, \
             mock.patch.object(contribute, "get_token", return_value="t"):
            contribute.cmd_issue_ensure(make_issue_args("仓内新增贡献流程skill"))
        payload = emit.call_args[0][0]
        self.assertEqual(payload["action"], "created")
        self.assertEqual(payload["number"], 100)
        self.assertTrue(payload["title"].startswith("[Requirement|需求建议]: "))

    def test_search_closed_then_reuse(self):
        calls = {"n": 0}

        def fake_get_all(path, token, max_pages=None, page_size=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return []  # open 无匹配
            return [{"number": 7, "title": "[Requirement|需求建议]: 仓内新增贡献流程skill", "state": "closed"}]

        with mock.patch.object(contribute, "api_get_all", side_effect=fake_get_all), \
             mock.patch.object(contribute, "emit_json") as emit, \
             mock.patch.object(contribute, "get_token", return_value="t"):
            contribute.cmd_issue_ensure(make_issue_args("仓内新增贡献流程skill"))
        payload = emit.call_args[0][0]
        # closed 匹配单独输出 reused_closed，把 reopen/新建的决定权交给调用方
        self.assertEqual(payload["action"], "reused_closed")
        self.assertEqual(payload["existing"][0]["number"], 7)


REAL_RUN_GIT_REF = contribute.run_git  # 模块级保存真实函数引用（mock 前捕获）


def make_submit_args(repo_root, **kw):
    """构造 --submit-pr 的 args 替身。"""
    defaults = dict(repo_root=str(repo_root), repo="hccl", title="[fix]test",
                    body="PR body", body_file=None, issue_number=1,
                    base=None, no_compile=False)
    defaults.update(kw)
    return mock.Mock(**defaults)


def fake_git_no_push(args, cwd, check=True):
    """run_git 替身：push 命令返回成功，其余真实执行。"""
    if args and args[0] == "push":
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout="push ok\n", stderr="")
    return REAL_RUN_GIT_REF(args, cwd, check=check)


def plain_labels(names):
    """judge_ci 用的标签名列表。"""
    return list(names)


def api_labels(names):
    """API 返回格式的标签列表。"""
    return [{"name": n} for n in names]


def make_pr_api(pr_created):
    """构造 submit-pr 测试的按 path 分流 API mock。

    detect_repo_layout 会先调 GET /user（fork 探测绑定账号），
    之后按序：POST /pulls 创建、POST comments(/compile)、GET /pulls/{n} 回查。
    """
    state = {"created": False, "pr": pr_created}

    def fake_request(method, path, token, payload=None):
        if path == "/user":
            return 200, {"login": "tester"}
        if method == "POST" and path.endswith("/pulls"):
            state["created"] = True
            return 201, state["pr"]
        if method == "POST" and "/comments" in path:
            return 201, {}
        if method == "GET" and "/pulls/" in path and path.rstrip("/").split("/")[-1].isdigit():
            return 200, {"state": "open"}
        return 200, {}

    return fake_request


class TestSubmitPr(GitRepoFixture):
    """--submit-pr：push + 创建 + /compile + GET 回查。"""

    def test_submit_creates_pr_and_compiles(self):
        self.make_feature_branch()
        pr_created = {"number": 500, "state": "open"}

        with mock.patch.object(contribute, "api_request",
                               side_effect=make_pr_api(pr_created)), \
             mock.patch.object(contribute, "emit_json") as emit, \
             mock.patch.object(contribute, "get_token", return_value="t"), \
             mock.patch.object(contribute, "run_git", side_effect=fake_git_no_push):
            contribute.cmd_submit_pr(make_submit_args(self.repo))
        payload = emit.call_args[0][0]
        self.assertEqual(payload["action"], "created")
        self.assertEqual(payload["pr_number"], 500)
        self.assertTrue(payload["compile_triggered"])
        self.assertTrue(payload["verified"])
        self.assertEqual(payload["branch"], "feat/test-branch")
        self.assertEqual(payload["fork_owner"], "tester")

    def test_submit_reuses_on_422(self):
        self.make_feature_branch()
        existing = [{"number": 321, "head": {"ref": "feat/test-branch",
                                              "user": {"login": "tester"}}}]
        calls = {"pr_post": False}

        def fake_request(method, path, token, payload=None):
            if path == "/user":
                return 200, {"login": "tester"}
            if method == "POST" and path.endswith("/pulls"):
                calls["pr_post"] = True
                return 422, {"message": "Another open merge request already exists"}
            if method == "GET" and "/pulls?" in path:
                return 200, existing  # 查开列表复用
            if method == "POST" and "/comments" in path:
                return 201, {}
            if method == "GET" and "/pulls/" in path and path.rstrip("/").split("/")[-1].isdigit():
                return 200, {"state": "open"}
            return 200, {}

        with mock.patch.object(contribute, "api_request", side_effect=fake_request), \
             mock.patch.object(contribute, "emit_json") as emit, \
             mock.patch.object(contribute, "get_token", return_value="t"), \
             mock.patch.object(contribute, "run_git", side_effect=fake_git_no_push):
            contribute.cmd_submit_pr(make_submit_args(self.repo))
        payload = emit.call_args[0][0]
        self.assertEqual(payload["action"], "reused")
        self.assertEqual(payload["pr_number"], 321)

    def test_submit_rejects_base_branch(self):
        args = make_submit_args(self.repo)  # 当前在 master
        with self.assertRaises(RuntimeError):
            contribute.cmd_submit_pr(args)

    def test_submit_rejects_dirty_tree(self):
        self.make_feature_branch()
        self.write("uncommitted.txt", "x")
        with self.assertRaises(RuntimeError):
            contribute.cmd_submit_pr(make_submit_args(self.repo))

    def test_submit_rejects_missing_identity(self):
        self.make_feature_branch()
        self.git("config", "--unset", "user.email")
        with self.assertRaises(RuntimeError), \
             mock.patch.object(contribute, "get_token", return_value="t"), \
             mock.patch.object(contribute, "run_git", side_effect=fake_git_no_push):
            contribute.cmd_submit_pr(make_submit_args(self.repo))

    def test_submit_appends_issue_when_body_lacks_reference(self):
        # body 未引用 Issue 时自动追加关联（所有 PR 必须关联 Issue）
        self.make_feature_branch()
        captured = {}

        def fake_request(method, path, token, payload=None):
            if path == "/user":
                return 200, {"login": "tester"}
            if method == "POST" and path.endswith("/pulls"):
                captured["body"] = (payload or {}).get("body", "")
                return 201, {"number": 600, "state": "open"}
            if method == "POST" and "/comments" in path:
                return 201, {}
            if method == "GET" and "/pulls/" in path and path.rstrip("/").split("/")[-1].isdigit():
                return 200, {"state": "open"}
            return 200, {}

        with mock.patch.object(contribute, "api_request", side_effect=fake_request), \
             mock.patch.object(contribute, "emit_json") as emit, \
             mock.patch.object(contribute, "get_token", return_value="t"), \
             mock.patch.object(contribute, "run_git", side_effect=fake_git_no_push):
            contribute.cmd_submit_pr(make_submit_args(self.repo, issue_number=807))
        payload = emit.call_args[0][0]
        self.assertTrue(payload["issue_appended"])
        self.assertEqual(payload["issue_number"], 807)
        self.assertIn("#807", captured["body"])


class TestCiStatus(unittest.TestCase):
    """--ci-status：saw_running 状态机。"""

    def test_judge_passed_after_running(self):
        history = [plain_labels(["ci-pipeline-running"]), plain_labels([])]
        state = contribute.judge_ci(history, ["ci-pipeline-passed"])
        self.assertEqual(state, "passed")

    def test_judge_failed_requires_saw_running(self):
        # 未见过 running、只有 stale failed → 不判失败
        state = contribute.judge_ci([plain_labels([])], ["ci-pipeline-failed"])
        self.assertEqual(state, "stale")
        # 见过 running 且 failed 仍在 → 判失败
        history = [plain_labels(["ci-pipeline-running"]), plain_labels(["ci-pipeline-failed"])]
        state = contribute.judge_ci(history, ["ci-pipeline-failed"])
        self.assertEqual(state, "failed")

    def test_judge_not_triggered(self):
        state = contribute.judge_ci([plain_labels([])], [])
        self.assertEqual(state, "not_triggered")

    def test_judge_running(self):
        state = contribute.judge_ci([plain_labels(["ci-pipeline-running"])],
                                    ["ci-pipeline-running"])
        self.assertEqual(state, "running")

    def test_judge_finishing_gap(self):
        # running 出现过、已消失、终态标签未上 → finishing（出标签间隙）
        history = [plain_labels(["ci-pipeline-running"])]
        state = contribute.judge_ci(history, [])
        self.assertEqual(state, "finishing")

    def test_judge_stale_passed_without_running(self):
        state = contribute.judge_ci([plain_labels([])], ["ci-pipeline-passed"])
        self.assertEqual(state, "stale")

    def test_cmd_single_shot_running(self):
        pr_data = {"labels": api_labels(["ci-pipeline-running"]), "state": "open",
                   "mergeable": None}
        with mock.patch.object(contribute, "api_request", return_value=(200, pr_data)), \
             mock.patch.object(contribute, "emit_json") as emit, \
             mock.patch.object(contribute, "get_token", return_value="t"):
            contribute.cmd_ci_status(mock.Mock(repo="hccl", pr_number=1, wait=False,
                                               interval=1, timeout=10))
        payload = emit.call_args[0][0]
        self.assertEqual(payload["state"], "running")

    def test_cmd_single_shot_stale_never_fails(self):
        # 单次查询只有残留 passed/failed（无 running 历史）：不误判，输出 stale
        pr_data = {"labels": api_labels(["ci-pipeline-failed"]), "state": "open"}
        with mock.patch.object(contribute, "api_request", return_value=(200, pr_data)), \
             mock.patch.object(contribute, "emit_json") as emit, \
             mock.patch.object(contribute, "get_token", return_value="t"):
            contribute.cmd_ci_status(mock.Mock(repo="hccl", pr_number=1, wait=False,
                                               interval=1, timeout=10))
        payload = emit.call_args[0][0]
        self.assertEqual(payload["state"], "stale")

    def test_cmd_wait_stale_tolerates_then_exits_with_hint(self):
        # --wait 在 stale 态前 3 轮容忍（/compile 到打标签的窗口期），之后退出并提示
        pr_data = {"labels": api_labels(["ci-pipeline-failed"]), "state": "open"}
        with mock.patch.object(contribute, "api_request", return_value=(200, pr_data)), \
             mock.patch.object(contribute, "emit_json") as emit, \
             mock.patch.object(contribute, "get_token", return_value="t"), \
             mock.patch.object(contribute.time, "sleep"):
            contribute.cmd_ci_status(mock.Mock(repo="hccl", pr_number=1, wait=True,
                                               interval=1, timeout=10))
        payload = emit.call_args[0][0]
        self.assertEqual(payload["state"], "stale")
        self.assertEqual(payload["stale_rounds"], 4)
        self.assertIn("/compile", payload.get("note", ""))

    def test_cmd_wait_stale_recovers_when_running_appears(self):
        # stale 窗口期内 running 出现：继续轮询至终态，不误退出
        pr_stale = {"labels": api_labels(["ci-pipeline-failed"]), "state": "open"}
        pr_running = {"labels": api_labels(["ci-pipeline-running"]), "state": "open"}
        pr_passed = {"labels": api_labels(["ci-pipeline-passed"]), "state": "open"}
        responses = iter([(200, pr_stale), (200, pr_stale), (200, pr_running),
                          (200, pr_passed)])
        with mock.patch.object(contribute, "api_request",
                               side_effect=lambda *a, **kw: next(responses)), \
             mock.patch.object(contribute, "emit_json") as emit, \
             mock.patch.object(contribute, "get_token", return_value="t"), \
             mock.patch.object(contribute.time, "sleep"):
            contribute.cmd_ci_status(mock.Mock(repo="hccl", pr_number=1, wait=True,
                                               interval=1, timeout=10))
        payload = emit.call_args[0][0]
        self.assertEqual(payload["state"], "passed")

    def test_cmd_wait_until_passed(self):
        pr_running = {"labels": api_labels(["ci-pipeline-running"]), "state": "open"}
        pr_passed = {"labels": api_labels(["ci-pipeline-passed"]), "state": "open"}
        responses = iter([(200, pr_running), (200, pr_passed)])
        with mock.patch.object(contribute, "api_request", side_effect=lambda *a, **kw: next(responses)), \
             mock.patch.object(contribute, "emit_json") as emit, \
             mock.patch.object(contribute, "get_token", return_value="t"), \
             mock.patch.object(contribute.time, "sleep"):
            contribute.cmd_ci_status(mock.Mock(repo="hccl", pr_number=1, wait=True,
                                               interval=1, timeout=10))
        payload = emit.call_args[0][0]
        self.assertEqual(payload["state"], "passed")


class TestParsePipelineLinks(unittest.TestCase):
    """流水线链接解析。"""

    def test_extracts_cann_robot_pipeline_urls(self):
        comments = [
            {"user": {"login": "cann-robot"},
             "body": "流水线任务触发成功 任务链接 https://www.openlibing.com/apps/pipelineDetail?pipelineId=1171",
             "created_at": "2026-08-29T10:00:00Z"},
            {"user": {"login": "someone"}, "body": "https://www.openlibing.com/x"},
        ]
        links = contribute.parse_pipeline_links(comments)
        self.assertEqual(len(links), 1)
        self.assertIn("pipelineDetail", links[0]["url"])


class TestCiLogs(unittest.TestCase):
    """--ci-logs：OBS 直链下载（404 容忍）。"""

    def test_downloads_available_logs(self):
        comments = [{"user": {"login": "cann-robot"},
                     "body": "link https://www.openlibing.com/apps/pipelineDetail?pipelineId=1",
                     "created_at": "t"}]

        class FakeResp(object):
            def __init__(self, body):
                self.status = 200
                self._body = body

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return self._body

        import urllib.request as ur
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(contribute, "api_get_all", return_value=comments), \
                 mock.patch.object(contribute, "emit_json") as emit, \
                 mock.patch.object(contribute, "get_token", return_value="t"), \
                 mock.patch.object(ur, "urlopen",
                                   side_effect=lambda req, timeout=None: FakeResp(b"log content")):
                contribute.cmd_ci_logs(mock.Mock(
                    repo="hccl", pr_number=1, output_dir=tmp))
            payload = emit.call_args[0][0]
            self.assertEqual(len(payload["logs_downloaded"]), 2)
            self.assertEqual(len(payload["pipelines"]), 1)

    def test_tolerates_404(self):
        import urllib.error as ue

        with tempfile.TemporaryDirectory() as tmp:

            def raise_404(req, timeout=None):
                raise ue.HTTPError(req.full_url, 404, "Not Found", {}, None)

            with mock.patch.object(contribute, "api_get_all", return_value=[]), \
                 mock.patch.object(contribute, "emit_json") as emit, \
                 mock.patch.object(contribute, "get_token", return_value="t"), \
                 mock.patch("urllib.request.urlopen", side_effect=raise_404):
                contribute.cmd_ci_logs(mock.Mock(
                    repo="hccl", pr_number=1, output_dir=tmp))
            payload = emit.call_args[0][0]
            self.assertEqual(payload["logs_downloaded"], [])


class TestListReviewComments(unittest.TestCase):
    """--list-review-comments：v5 comments（resolved 权威）+ v4 补路径。"""

    def test_lists_unresolved_findings(self):
        comments = [
            # 未 resolve 的 diff_comment（主体；列表接口 path 为 None）
            {"id": 11, "user": {"login": "reviewer"}, "resolved": False,
             "comment_type": "diff_comment", "discussion_id": "hex11",
             "diff_position": {"path": None, "start_new_line": 42},
             "body": "🔴 严重问题"},
            # 已 resolve → 跳过
            {"id": 12, "user": {"login": "reviewer"}, "resolved": True,
             "comment_type": "diff_comment", "diff_position": {}, "body": "已处理"},
            # bot 评论 → 跳过
            {"id": 13, "user": {"login": "cann-robot"}, "resolved": False,
             "comment_type": "pr_comment", "body": "流水线任务触发成功"},
            # 斜杠命令 → 跳过
            {"id": 14, "user": {"login": "me"}, "resolved": False,
             "comment_type": "pr_comment", "body": "/compile"},
            # 无位置信息的普通评论（检视报告）→ 跳过
            {"id": 15, "user": {"login": "wlwy"}, "resolved": None,
             "comment_type": "pr_comment", "diff_position": None,
             "body": "# PR #1 代码检视报告"},
        ]
        # v5 单条接口返回 position.new_path（列表接口 path 为 None 时的补路径来源）
        single_detail = {"id": 11, "position": {"new_path": "src/a.cc", "new_line": 42}}

        def fake_get_all(path, token, max_pages=None, page_size=None):
            return comments

        def fake_request(method, path, token, payload=None):
            if path.endswith("/pulls/comments/11"):
                return 200, single_detail
            return 200, {}

        with mock.patch.object(contribute, "api_get_all", side_effect=fake_get_all), \
             mock.patch.object(contribute, "api_request", side_effect=fake_request), \
             mock.patch.object(contribute, "emit_json") as emit, \
             mock.patch.object(contribute, "get_token", return_value="t"):
            contribute.cmd_list_review_comments(mock.Mock(repo="hccl", pr_number=1))
        payload = emit.call_args[0][0]
        self.assertEqual(payload["count"], 1)
        finding = payload["open_findings"][0]
        self.assertEqual(finding["file"], "src/a.cc")  # v5 单条接口补的路径
        self.assertEqual(finding["line"], 42)
        self.assertEqual(finding["discussion_id"], "hex11")

    def test_all_resolved_returns_empty(self):
        comments = [
            {"id": 11, "user": {"login": "reviewer"}, "resolved": True,
             "comment_type": "diff_comment", "diff_position": {}, "body": "已处理"},
        ]

        def fake_get_all(path, token, max_pages=None, page_size=None):
            if "discussions" in path:
                return []
            return comments

        with mock.patch.object(contribute, "api_get_all", side_effect=fake_get_all), \
             mock.patch.object(contribute, "emit_json") as emit, \
             mock.patch.object(contribute, "get_token", return_value="t"):
            contribute.cmd_list_review_comments(mock.Mock(repo="hccl", pr_number=1))
        payload = emit.call_args[0][0]
        self.assertEqual(payload["count"], 0)


class TestCleanup(GitRepoFixture):
    """--cleanup：worktree 移除与临时目录清理。"""

    def test_cleanup_removes_worktree(self):
        wt_path = self.root / "wt-test"
        self.git("worktree", "add", "--detach", str(wt_path), "HEAD")
        self.assertTrue(wt_path.exists())
        args = mock.Mock(repo_root=str(self.repo), worktree=str(wt_path))
        with mock.patch.object(contribute, "emit_json") as emit:
            contribute.cmd_cleanup(args)
        payload = emit.call_args[0][0]
        self.assertIn(str(wt_path), payload["removed"])
        self.assertFalse(wt_path.exists())


class TestCheckEnvNoToken(GitRepoFixture):
    """--check-env：纯 clone 用户（无 token）不阻断，记 WARN。"""

    def test_no_token_warns_but_passes(self):
        args = mock.Mock(repo_root=str(self.repo))
        with mock.patch.object(contribute, "get_token", return_value=""), \
             mock.patch.object(contribute, "emit_json"):
            ok = contribute.cmd_check_env(args)
        self.assertTrue(ok, "无 token 时环境校验应通过（WARN 不阻断）")


class TestToken(unittest.TestCase):
    """token 获取优先级。"""

    def test_env_var_first(self):
        with mock.patch.dict(os.environ, {"GITCODE_TOKEN": "envtoken"}):
            self.assertEqual(contribute.get_token(), "envtoken")

    def test_credential_fill_fallback(self):
        env = {k: v for k, v in os.environ.items() if k != "GITCODE_TOKEN"}
        proc = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout="protocol=https\nhost=gitcode.com\nusername=u\npassword=credtoken\n",
            stderr="")
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch.object(subprocess, "run", return_value=proc):
            self.assertEqual(contribute.get_token(), "credtoken")


class TestApiGetAll(unittest.TestCase):
    """翻页防护：首条 id 重复终止、返回数不足终止、max_pages 兜底。"""

    def test_stops_on_short_page(self):
        # 第一页 50 个 < page_size 100 → 应在第一页后终止
        responses = [(200, [{"id": i} for i in range(50)])]
        with mock.patch.object(contribute, "api_request", side_effect=responses):
            items = contribute.api_get_all("/x", "t", max_pages=10, page_size=100)
        self.assertEqual(len(items), 50)

    def test_full_page_continues_then_stops(self):
        # 第一页满 100 → 翻第二页 1 个 → 终止；共 101
        responses = [
            (200, [{"id": i} for i in range(100)]),
            (200, [{"id": 100}]),
        ]
        with mock.patch.object(contribute, "api_request", side_effect=responses):
            items = contribute.api_get_all("/x", "t", max_pages=10, page_size=100)
        self.assertEqual(len(items), 101)

    def test_stops_on_repeated_first_id(self):
        same = [{"id": 1}, {"id": 2}]
        responses = [(200, same), (200, same), (200, same)]
        with mock.patch.object(contribute, "api_request", side_effect=responses):
            items = contribute.api_get_all("/x", "t", max_pages=10, page_size=2)
        self.assertEqual(len(items), 2)

    def test_max_pages_bound(self):
        # 每页满 100 且 first_id 各页不同（防重复检测误触发），应停在 max_pages=50
        pages = []
        for page_no in range(60):
            pages.append((200, [{"id": page_no * 1000 + i} for i in range(100)]))
        with mock.patch.object(contribute, "api_request", side_effect=pages):
            items = contribute.api_get_all("/x", "t", max_pages=50, page_size=100)
        self.assertEqual(len(items), 50 * 100)

    def test_stops_on_error(self):
        with mock.patch.object(contribute, "api_request", return_value=(404, {"msg": "x"})):
            items = contribute.api_get_all("/x", "t", max_pages=10)
        self.assertEqual(items, [])

    def test_unwraps_v4_dict_envelope(self):
        # v4 端点返回 {data:[...], end_id} 包装，须解包后再翻页
        responses = [
            (200, {"end_id": "e1", "data": [{"id": 1}, {"id": 2}]}),
            (200, {"end_id": "e2", "data": []}),
        ]
        with mock.patch.object(contribute, "api_request", side_effect=responses):
            items = contribute.api_get_all("/x", "t", max_pages=10, page_size=2)
        self.assertEqual(len(items), 2)

    def test_strips_existing_pagination_params(self):
        # path 已带 per_page 时不再重复拼接（URL 不得出现重复参数）
        captured = []

        def fake_request(method, path, token, payload=None):
            captured.append(path)
            return 200, [{"id": 1}]

        with mock.patch.object(contribute, "api_request", side_effect=fake_request):
            contribute.api_get_all("/x?per_page=100", "t", max_pages=5, page_size=100)
        self.assertIn("per_page=100&page=1", captured[0])
        self.assertEqual(captured[0].count("per_page="), 1)


class TestApiRequestRetry(unittest.TestCase):
    """api_request 的 429 重试（等待 60s 后重试一次）。"""

    def test_retry_once_on_429(self):
        calls = {"n": 0}

        def fake_once(method, path, token, payload=None, api_base=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return 429, {"message": "rate limited"}
            return 200, {"ok": True}

        with mock.patch.object(contribute, "api_request_once", side_effect=fake_once), \
             mock.patch.object(contribute.time, "sleep") as sleep_mock:
            code, data = contribute.api_request("GET", "/x", "t")
        self.assertEqual(code, 200)
        self.assertEqual(calls["n"], 2)
        sleep_mock.assert_called_once_with(60)

    def test_no_retry_on_500(self):
        calls = {"n": 0}

        def fake_once(method, path, token, payload=None, api_base=None):
            calls["n"] += 1
            return 500, {"message": "server error"}

        with mock.patch.object(contribute, "api_request_once", side_effect=fake_once):
            code, _ = contribute.api_request("GET", "/x", "t")
        self.assertEqual(code, 500)
        self.assertEqual(calls["n"], 1)


class TestGetTokenTimeout(unittest.TestCase):
    """git credential fill 超时不挂起。"""

    def test_timeout_returns_empty(self):
        import subprocess as sp

        def raise_timeout(*a, **kw):
            raise sp.TimeoutExpired(cmd="git credential fill", timeout=15)

        env = {k: v for k, v in os.environ.items() if k != "GITCODE_TOKEN"}
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch.object(subprocess, "run", side_effect=raise_timeout):
            self.assertEqual(contribute.get_token(), "")


if __name__ == "__main__":
    unittest.main()
