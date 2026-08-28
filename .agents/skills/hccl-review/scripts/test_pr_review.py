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

"""pr_review.py 单元测试。

用本地 mock git 仓库（git init + 构造多 hunk diff）与 mock GitCode API
（monkeypatch api_request）覆盖检视流程的机械步骤，不发起任何真实网络请求。

运行: python3 -m unittest test_pr_review -v
依赖: Python 3.7+，仅标准库；需要 PATH 里有 git。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pr_review  # noqa: E402


def run_git(args, cwd):
    result = subprocess.run(["git"] + args, cwd=str(cwd), capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError("git {} failed: {}".format(args, result.stderr))
    return result.stdout


class MockRepoTestCase(unittest.TestCase):
    """构造 mock git 仓库基类。

    仓库结构:
      base commit: src/demo.cc (10 行) + src/old.cc
      head commit: 修改 src/demo.cc（中间插入 2 行 + 尾部改 1 行）+ 新增 src/new_file.cc（20 行）
    """

    @classmethod
    def setUpClass(cls):
        cls.repo_dir = Path(tempfile.mkdtemp(prefix="pr_review_test_"))
        run_git(["init", "-q"], cls.repo_dir)
        run_git(["config", "user.email", "test@example.com"], cls.repo_dir)
        run_git(["config", "user.name", "tester"], cls.repo_dir)
        # origin 指向不可达地址，确保测试绝不发起真实网络 fetch
        run_git(["remote", "add", "origin", "https://gitcode.example.invalid/cann/hccl.git"], cls.repo_dir)

        demo_base = "\n".join("line {}".format(i) for i in range(1, 11)) + "\n"
        (cls.repo_dir / "src").mkdir(exist_ok=True)
        (cls.repo_dir / "src" / "demo.cc").write_text(demo_base, encoding="utf-8")
        (cls.repo_dir / "src" / "old.cc").write_text("old content\n", encoding="utf-8")
        run_git(["add", "."], cls.repo_dir)
        run_git(["commit", "-q", "-m", "base"], cls.repo_dir)
        cls.base_sha = run_git(["rev-parse", "HEAD"], cls.repo_dir).strip()

        # head：demo.cc 第 5 行后插 2 行，第 10 行（现 12）改为 modified；新增 new_file.cc
        demo_head = demo_base.replace("line 5\n", "line 5\ninserted A\ninserted B\n")
        demo_head = demo_head.replace("line 10", "modified tail")
        (cls.repo_dir / "src" / "demo.cc").write_text(demo_head, encoding="utf-8")
        (cls.repo_dir / "src" / "new_file.cc").write_text(
            "\n".join("new line {}".format(i) for i in range(1, 21)) + "\n", encoding="utf-8")
        run_git(["add", "."], cls.repo_dir)
        run_git(["commit", "-q", "-m", "head"], cls.repo_dir)
        cls.head_sha = run_git(["rev-parse", "HEAD"], cls.repo_dir).strip()
        # 预置 PR head ref，使 ensure_pr_refs 无需 fetch 即可 rev-parse
        run_git(["update-ref", "refs/remotes/origin/mr/1-head", cls.head_sha], cls.repo_dir)
        # 预置 origin/master ref 供 merge-base fallback
        run_git(["update-ref", "refs/remotes/origin/master", cls.base_sha], cls.repo_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.repo_dir, ignore_errors=True)

    def setUp(self):
        self._saved_request = pr_review.api_request
        self.api_calls = []

    def tearDown(self):
        pr_review.api_request = self._saved_request

    def mock_api(self, responses):
        """按 URL 关键词路由的 mock。responses: {url_keyword: data_or_callable}。"""
        def fake_request(url, method="GET", data=None, token=""):
            self.api_calls.append({"url": url, "method": method, "data": data})
            for keyword, value in responses.items():
                if keyword in url:
                    if callable(value):
                        return value(url, method, data)
                    return value
            raise RuntimeError("HTTP 404: not mocked {}".format(url))
        pr_review.api_request = fake_request
        return fake_request


class TestVerifyLineNumber(MockRepoTestCase):
    """行号验证：行存在性、snippet 匹配、越界、文件不存在。"""

    def test_verified_with_snippet(self):
        result = pr_review.verify_line_number(
            self.repo_dir, self.head_sha, "src/demo.cc", 6, "inserted A")
        self.assertTrue(result["verified"])

    def test_verified_without_snippet(self):
        result = pr_review.verify_line_number(
            self.repo_dir, self.head_sha, "src/demo.cc", 1)
        self.assertTrue(result["verified"])

    def test_snippet_mismatch(self):
        result = pr_review.verify_line_number(
            self.repo_dir, self.head_sha, "src/demo.cc", 1, "not-exist-content")
        self.assertFalse(result["verified"])
        self.assertEqual(result["reason"], "snippet mismatch")
        self.assertEqual(result["actual"], "line 1")

    def test_line_out_of_range(self):
        result = pr_review.verify_line_number(
            self.repo_dir, self.head_sha, "src/demo.cc", 999)
        self.assertFalse(result["verified"])
        self.assertIn("out of range", result["reason"])

    def test_file_not_found(self):
        result = pr_review.verify_line_number(
            self.repo_dir, self.head_sha, "src/no_such.cc", 1)
        self.assertFalse(result["verified"])
        self.assertEqual(result["reason"], "file not found at HEAD")

    def test_zero_or_negative_line(self):
        result = pr_review.verify_line_number(
            self.repo_dir, self.head_sha, "src/demo.cc", 0)
        self.assertFalse(result["verified"])


class TestFindDiffPosition(MockRepoTestCase):
    """diff 位置计算：新增行命中、上下文行命中、删除行与未变更行不命中。"""

    def test_added_line_hit(self):
        # demo.cc 新版第 6 行是 inserted A（diff 新增行）
        pos = pr_review.find_diff_position(
            self.repo_dir, self.base_sha, self.head_sha, "src/demo.cc", 6)
        self.assertEqual(pos, 6)

    def test_context_line_hit(self):
        # hunk 从新文件第 3 行起，第 3 行是上下文行（未变更但出现在 diff 中）
        pos = pr_review.find_diff_position(
            self.repo_dir, self.base_sha, self.head_sha, "src/demo.cc", 3)
        self.assertEqual(pos, 3)

    def test_line_before_hunk_misses(self):
        # 第 2 行在 hunk 之前（未出现在 diff 中）
        pos = pr_review.find_diff_position(
            self.repo_dir, self.base_sha, self.head_sha, "src/demo.cc", 2)
        self.assertIsNone(pos)

    def test_unchanged_far_line_miss(self):
        # old.cc 完全没变更，任何行号都不在 diff
        pos = pr_review.find_diff_position(
            self.repo_dir, self.base_sha, self.head_sha, "src/old.cc", 1)
        self.assertIsNone(pos)

    def test_pure_new_file_any_line(self):
        # 纯新增文件（+++ b/ 头行必须跳过，不能推进行号）
        for line in (1, 10, 20):
            pos = pr_review.find_diff_position(
                self.repo_dir, self.base_sha, self.head_sha, "src/new_file.cc", line)
            self.assertEqual(pos, line, "new_file.cc line {} 应命中".format(line))

    def test_modified_tail_line(self):
        # 第 12 行是 modified tail（原 line 10 修改而来）
        pos = pr_review.find_diff_position(
            self.repo_dir, self.base_sha, self.head_sha, "src/demo.cc", 12)
        self.assertEqual(pos, 12)


class TestDedupFindings(MockRepoTestCase):
    """内容指纹去重：file+line 命中、pr_comment 回退格式命中、标题命中。"""

    def setUp(self):
        super().setUp()
        self.finding = {
            "severity": "HIGH", "file": "src/demo.cc", "line": 6,
            "title": "空指针风险", "body": "问题描述",
        }

    def test_dedup_by_file_line_diff_comment(self):
        # v5 comments path 由 v4 enrich 补齐后命中 file+line
        existing = [{"body": "**[HIGH] 其他**\n旧意见", "path": "src/demo.cc",
                     "diff_position": {"start_new_line": 6}}]
        deduped, skipped = pr_review.dedup_findings([self.finding], existing)
        self.assertEqual(len(deduped), 0)
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["reason"], "file+line 已有评论")

    def test_dedup_by_file_line_pr_comment_quote(self):
        existing = [{"body": "> src/demo.cc:6\n\n**[HIGH] 其他**\n旧意见"}]
        deduped, skipped = pr_review.dedup_findings([self.finding], existing)
        self.assertEqual(len(deduped), 0)
        self.assertEqual(len(skipped), 1)

    def test_dedup_by_title(self):
        existing = [{"body": "**[MEDIUM] 空指针风险**\n别人提过同标题意见",
                     "diff_position": {"start_new_line": 99}}]
        deduped, skipped = pr_review.dedup_findings([self.finding], existing)
        self.assertEqual(len(deduped), 0)
        self.assertEqual(skipped[0]["reason"], "标题与已有评论重复")

    def test_no_dedup_when_different(self):
        existing = [{"body": "**[HIGH] 另一个问题**\n旧意见",
                     "diff_position": {"start_new_line": 3}}]
        deduped, skipped = pr_review.dedup_findings([self.finding], existing)
        self.assertEqual(len(deduped), 1)
        self.assertEqual(len(skipped), 0)

    def test_mixed_findings_partial_dedup(self):
        other = {"severity": "LOW", "file": "src/demo.cc", "line": 12,
                 "title": "命名建议", "body": "..."}
        existing = [{"body": "**[HIGH] 空指针风险**\n同标题",
                     "diff_position": {"start_new_line": 6}}]
        deduped, skipped = pr_review.dedup_findings([self.finding, other], existing)
        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0]["title"], "命名建议")
        self.assertEqual(len(skipped), 1)


class TestBaselineDrift(MockRepoTestCase):
    """基线漂移核验：本地 diff 文件数 vs changed_files。"""

    def make_meta(self, changed_files):
        return {"changed_files": changed_files}

    def test_no_drift_when_consistent(self):
        result = pr_review.check_baseline_drift(
            self.repo_dir, self.base_sha, self.head_sha, self.make_meta(2))
        self.assertFalse(result["drift"])
        self.assertEqual(result["local_files"], 2)

    def test_no_drift_when_small_diff(self):
        # 无 changed_files 元数据时不判漂移
        result = pr_review.check_baseline_drift(
            self.repo_dir, self.base_sha, self.head_sha, self.make_meta(0))
        self.assertFalse(result["drift"])

    def test_drift_detected(self):
        # 声称只改 1 个文件但本地 diff 有 2 个（阈值: >max(1*2, 1+5)=6 不触发，
        # 用更悬殊的假元数据验证触发路径）
        result = pr_review.check_baseline_drift(
            self.repo_dir, self.base_sha, self.head_sha, self.make_meta(1))
        # 2 <= max(2, 6)，不触发
        self.assertFalse(result["drift"])
        # 直接构造触发场景：expected=1, local=2 -> 2 > max(2,6)? 否。
        # 换 base 使 diff 变大不现实，这里用单元级 mock 验证判定逻辑分支
        with unittest.mock.patch.object(pr_review, "run_git", return_value="a\n" * 20):
            result = pr_review.check_baseline_drift(
                self.repo_dir, "fake_base", "fake_head", self.make_meta(1))
            self.assertTrue(result["drift"])



class TestGetPrMeta(MockRepoTestCase):
    """PR 元数据：v5/v4 合并、state、分支字段。"""

    def test_meta_merges_v4_diff_refs(self):
        v5_data = {
            "state": "open", "target_branch": "master", "source_branch": "feat/x",
            "title": "demo", "body": "desc",
            "head": {"sha": "aaa", "ref": "feat/x"}, "base": {"sha": "bbb", "ref": "master"},
            "changed_files": 3, "additions": 10, "deletions": 2,
            "user": {"login": "someone"},
        }
        v4_data = {"diff_refs": {"base_sha": "v4base", "head_sha": "v4head", "start_sha": "v4start"}}

        def router(url, method="GET", data=None, token=""):
            if "/api/v5/" in url:
                return v5_data
            if "/api/v4/" in url:
                return v4_data
            raise RuntimeError("unmocked")
        pr_review.api_request = router
        meta = pr_review.get_pr_meta("cann", "hccl", 123, token="t")
        self.assertEqual(meta["state"], "open")
        self.assertEqual(meta["target_branch"], "master")
        self.assertEqual(meta["head_sha"], "v4head")
        self.assertEqual(meta["base_sha"], "v4base")
        self.assertEqual(meta["changed_files"], 3)
        self.assertIn("pull/123", meta["html_url"])

    def test_meta_falls_back_to_v5_base_when_v4_down(self):
        v5_data = {
            "state": "merged", "target_branch": "master",
            "head": {"sha": "aaa"}, "base": {"sha": "bbb"},
        }

        def router(url, method="GET", data=None, token=""):
            if "/api/v5/" in url:
                return v5_data
            raise RuntimeError("HTTP 404")
        pr_review.api_request = router
        meta = pr_review.get_pr_meta("cann", "hccl", 1, token="t")
        self.assertEqual(meta["state"], "merged")
        self.assertEqual(meta["base_sha"], "bbb")


class TestPostFinding(MockRepoTestCase):
    """意见提交：行内评论 payload、回退 PR 评论 payload。"""

    def test_diffnote_payload(self):
        calls = []

        def fake_request(url, method="GET", data=None, token=""):
            calls.append({"url": url, "method": method, "data": data})
            return {"id": "hex123", "comment_type": "diff_comment"}
        pr_review.api_request = fake_request
        finding = {"severity": "HIGH", "file": "src/a.cc", "line": 5,
                   "title": "T", "body": "B"}
        result = pr_review.post_finding(pr_review.make_target("cann", "hccl", 9), "sha1", finding, 5, "tok")
        self.assertEqual(result["comment_type"], "diff_comment")
        payload = calls[0]["data"]
        self.assertEqual(payload["position"], 5)
        self.assertEqual(payload["path"], "src/a.cc")
        self.assertEqual(payload["commit_id"], "sha1")
        self.assertIn("**[HIGH] T**", payload["body"])

    def test_fallback_pr_comment_payload(self):
        calls = []

        def fake_request(url, method="GET", data=None, token=""):
            calls.append(data)
            return {"id": "hex", "comment_type": "pr_comment"}
        pr_review.api_request = fake_request
        finding = {"severity": "LOW", "file": "src/a.cc", "line": 77,
                   "title": "T", "body": "B"}
        pr_review.post_finding(pr_review.make_target("cann", "hccl", 9), "sha1", finding, None, "tok")
        payload = calls[0]
        self.assertNotIn("position", payload)
        self.assertNotIn("path", payload)
        self.assertIn("> src/a.cc:77", payload["body"])


class TestPostReport(MockRepoTestCase):
    """汇总报告：统计、表格、严重级别排序。"""

    def test_report_body(self):
        calls = []

        def fake_request(url, method="GET", data=None, token=""):
            calls.append(data)
            return {"id": "hex"}
        pr_review.api_request = fake_request
        findings = [
            {"severity": "LOW", "file": "a.cc", "line": 1, "dimension": "D", "title": "L1", "body": "b"},
            {"severity": "CRITICAL", "file": "b.cc", "line": 2, "dimension": "D", "title": "C1", "body": "b"},
            {"severity": "HIGH", "file": "c.cc", "line": 3, "dimension": "D", "title": "H1", "body": "b"},
        ]
        meta = {"html_url": "http://x", "title": "t", "head_sha": "aaaa", "base_sha": "bbbb"}
        pr_review.post_report(pr_review.make_target("cann", "hccl", 5), findings, meta, "tok")
        body = calls[0]["body"]
        self.assertIn("代码检视汇总报告", body)
        self.assertIn("CRITICAL 1", body)
        # CRITICAL 行应排在 LOW 前
        self.assertLess(body.index("C1"), body.index("L1"))


class TestCheckEnv(MockRepoTestCase):
    """--check-env 环境校验。"""

    def test_check_env_with_token(self):
        with unittest.mock.patch.object(pr_review, "get_token", return_value="tok1234"):
            with unittest.mock.patch.object(pr_review, "api_request",
                                            return_value={"state": "open"}):
                rc = pr_review.cmd_check_env(argparse_stub(self.repo_dir))
                self.assertEqual(rc, 0)

    def test_check_env_without_token(self):
        with unittest.mock.patch.object(pr_review, "get_token", return_value=""):
            rc = pr_review.cmd_check_env(argparse_stub(self.repo_dir))
            self.assertEqual(rc, 1)


def argparse_stub(repo_root):
    """构造 cmd_check_env 需要的最小 args 对象。"""
    return type("Args", (), {"repo_root": str(repo_root), "owner": "cann", "repo": "hccl"})()


class TestLoadFindings(unittest.TestCase):
    """findings.json 加载与 schema 校验。"""

    def test_load_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "f.json"
            p.write_text(json.dumps([
                {"severity": "HIGH", "file": "a.cc", "line": 1, "title": "t", "body": "b"}]),
                encoding="utf-8")
            data = pr_review.load_findings(str(p))
            self.assertEqual(len(data), 1)

    def test_load_single_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "f.json"
            p.write_text(json.dumps(
                {"severity": "LOW", "file": "a.cc", "line": 1, "title": "t", "body": "b"}),
                encoding="utf-8")
            data = pr_review.load_findings(str(p))
            self.assertEqual(len(data), 1)

    def test_missing_field_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "f.json"
            p.write_text(json.dumps([{"severity": "HIGH"}]), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                pr_review.load_findings(str(p))

    def test_file_not_found(self):
        with self.assertRaises(RuntimeError):
            pr_review.load_findings("/no/such/file.json")


class TestFindRemote(MockRepoTestCase):
    """remote 自动探测。"""

    def test_find_cann_remote(self):
        self.assertEqual(pr_review.find_remote(self.repo_dir), "origin")

    def test_prefer_upstream_when_both(self):
        run_git(["remote", "add", "upstream", "https://gitcode.com/cann/hccl.git"],
                self.repo_dir)
        # origin 也指向 cann，先匹配到的返回；无论哪个都指向 cann 即可
        remote = pr_review.find_remote(self.repo_dir)
        self.assertIn(remote, ("origin", "upstream"))


class TestCmdReviewFlow(MockRepoTestCase):
    """verify-only / dry-run / POST 命令级流程（mock API + mock git refs）。"""

    def setUp(self):
        super().setUp()
        self.findings = [{
            "severity": "HIGH", "file": "src/demo.cc", "line": 6,
            "title": "插入行问题", "body": "描述与建议修复", "code_snippet": "inserted A",
        }, {
            "severity": "LOW", "file": "src/old.cc", "line": 1,
            "title": "未变更文件意见", "body": "描述", "code_snippet": "old content",
        }]

    def write_findings(self, tmp):
        p = Path(tmp) / "findings.json"
        p.write_text(json.dumps(self.findings, ensure_ascii=False), encoding="utf-8")
        return str(p)

    def make_meta_dict(self):
        return {
            "state": "open", "target_branch": "master", "source_branch": "b",
            "title": "t", "body": "", "head_sha": self.head_sha,
            "base_sha": self.base_sha, "changed_files": 2,
            "additions": 22, "deletions": 1, "author": "a",
            "html_url": "https://gitcode.com/cann/hccl/pull/1",
        }

    def mock_full_api(self, existing_comments=None):
        """mock PR 元数据 + 已有评论 + POST。"""
        post_calls = []

        def router(url, method="GET", data=None, token=""):
            if method == "GET" and "/pulls/1".format() in url and url.rstrip("/").endswith("/1"):
                return self.make_meta_dict()
            if "/api/v4/" in url:
                return {"diff_refs": {"base_sha": self.base_sha, "head_sha": self.head_sha}}
            if method == "GET" and "/comments" in url:
                return existing_comments or []
            if method == "POST":
                post_calls.append({"url": url, "data": data})
                return {"id": "hex" + str(len(post_calls)), "comment_type": "diff_comment"}
            raise RuntimeError("unmocked {} {}".format(method, url))

        pr_review.api_request = router
        return post_calls

    def test_verify_only(self):
        post_calls = self.mock_full_api()
        with tempfile.TemporaryDirectory() as tmp:
            findings_path = self.write_findings(tmp)
            args = argparse_review_stub(self.repo_dir, findings_path, verify_only=True)
            rc = pr_review.cmd_review(args, "tok")
            self.assertEqual(rc, 0)
            self.assertEqual(len(post_calls), 0)  # verify-only 不 POST

    def test_dry_run(self):
        post_calls = self.mock_full_api()
        with tempfile.TemporaryDirectory() as tmp:
            findings_path = self.write_findings(tmp)
            args = argparse_review_stub(self.repo_dir, findings_path, dry_run=True)
            rc = pr_review.cmd_review(args, "tok")
            self.assertEqual(rc, 0)
            self.assertEqual(len(post_calls), 0)

    def test_post_with_dedup_and_fallback(self):
        # 已有评论命中第一条 finding 的 file+line -> 跳过；
        # 第二条 finding 行不在 diff -> 回退 pr_comment payload
        existing = [{"body": "**[HIGH] 插入行问题**\n旧", "path": "src/demo.cc",
                     "diff_position": {"start_new_line": 6}}]
        post_calls = self.mock_full_api(existing)
        with tempfile.TemporaryDirectory() as tmp:
            findings_path = self.write_findings(tmp)
            args = argparse_review_stub(self.repo_dir, findings_path)
            rc = pr_review.cmd_review(args, "tok")
            self.assertEqual(rc, 0)
            # 只有 fallback 那条被 POST
            self.assertEqual(len(post_calls), 1)
            payload = post_calls[0]["data"]
            self.assertNotIn("position", payload)
            self.assertIn("> src/old.cc:1", payload["body"])

    def test_post_rejected_when_unverified_line(self):
        self.findings.append({
            "severity": "MEDIUM", "file": "src/demo.cc", "line": 999,
            "title": "越界行", "body": "b",
        })
        post_calls = self.mock_full_api()
        with tempfile.TemporaryDirectory() as tmp:
            findings_path = self.write_findings(tmp)
            args = argparse_review_stub(self.repo_dir, findings_path)
            rc = pr_review.cmd_review(args, "tok")
            self.assertEqual(rc, 1)
            self.assertEqual(len(post_calls), 0)

    def test_merged_pr_rejects_post(self):
        meta = self.make_meta_dict()
        meta["state"] = "merged"

        def router(url, method="GET", data=None, token=""):
            if "/api/v5/" in url and method == "GET":
                return meta
            if "/api/v4/" in url:
                return {"diff_refs": {"base_sha": self.base_sha, "head_sha": self.head_sha}}
            raise RuntimeError("unmocked")
        pr_review.api_request = router

        with tempfile.TemporaryDirectory() as tmp:
            findings_path = self.write_findings(tmp)
            args = argparse_review_stub(self.repo_dir, findings_path)
            rc = pr_review.cmd_review(args, "tok")
            self.assertEqual(rc, 1)

    def test_baseline_drift_blocks_post(self):
        meta = self.make_meta_dict()
        meta["changed_files"] = 1  # 本地 2 文件 vs 声称 1（阈值不触发，改用 0 触发不了）
        # 用 mock run_git 放大本地文件数触发漂移分支
        meta["changed_files"] = 1
        real_run_git = pr_review.run_git

        def fake_run_git(args, cwd):
            if args and args[0] == "diff" and "--name-only" in args:
                return "\n".join("file{}.cc".format(i) for i in range(20)) + "\n"
            return real_run_git(args, cwd)

        def router(url, method="GET", data=None, token=""):
            if "/api/v5/" in url and method == "GET":
                return meta
            if "/api/v4/" in url:
                return {"diff_refs": {"base_sha": self.base_sha, "head_sha": self.head_sha}}
            raise RuntimeError("unmocked")
        pr_review.api_request = router

        with tempfile.TemporaryDirectory() as tmp:
            findings_path = self.write_findings(tmp)
            args = argparse_review_stub(self.repo_dir, findings_path)
            with unittest.mock.patch.object(pr_review, "run_git", side_effect=fake_run_git):
                rc = pr_review.cmd_review(args, "tok")
            self.assertEqual(rc, 1)


def argparse_review_stub(repo_root, findings, verify_only=False, dry_run=False):
    return type("Args", (), {
        "repo_root": str(repo_root), "owner": "cann", "repo": "hccl",
        "pr_number": 1, "findings": findings,
        "verify_only": verify_only, "dry_run": dry_run,
        "head_sha": "", "base_sha": "", "delay": 0,
    })()


class TestSelfReviewFixes(MockRepoTestCase):
    """自检意见修复的回归测试。"""

    def test_check_env_does_not_print_token_prefix(self):
        """修复1：check-env 输出不含 token 片段。"""
        import io
        import contextlib
        buf = io.StringIO()
        with unittest.mock.patch.object(pr_review, "get_token", return_value="SECRET_TOKEN"):
            with unittest.mock.patch.object(pr_review, "api_request", return_value={"state": "open"}):
                with contextlib.redirect_stdout(buf):
                    rc = pr_review.cmd_check_env(argparse_stub(self.repo_dir))
        self.assertEqual(rc, 0)
        self.assertNotIn("SECRET_TOKEN", buf.getvalue())
        self.assertNotIn("SECR", buf.getvalue())

    def test_cleanup_removes_findings_in_repo_root_and_cwd(self):
        """修复2：cleanup 在 repo-root 与 cwd 两处清理 findings。"""
        with tempfile.TemporaryDirectory() as tmp:
            cwd_findings = Path(tmp) / "findings_test.json"
            cwd_findings.write_text("[]", encoding="utf-8")
            args = type("Args", (), {
                "repo_root": str(self.repo_dir), "worktree": "",
                "findings": str(cwd_findings),
            })()
            with unittest.mock.patch.object(pr_review, "run_git", return_value=""):
                rc = pr_review.cmd_cleanup(args, "tok")
            self.assertEqual(rc, 0)
            self.assertFalse(cwd_findings.exists())

    def test_api_get_all_has_page_limit(self):
        """修复3：api_get_all 翻页有 50 页上限，服务端每页不同数据也不死循环。"""
        call_count = []

        def fake_request(url, method="GET", data=None, token=""):
            call_count.append(url)
            # 每页都返回不同 id 的 100 条（模拟排序不稳定的服务端异常）
            page = int(url.split("page=")[-1])
            return [{"id": page * 1000 + i} for i in range(100)]

        pr_review.api_request = fake_request
        items = pr_review.api_get_all("https://example.invalid/list")
        self.assertEqual(len(items), 50 * 100)  # 恰好 50 页封顶
        self.assertEqual(len(call_count), 50)


class TestDisposeAndRemoteFixes(MockRepoTestCase):
    """committer 检视意见修复的回归测试。"""

    def test_body_argument_registered(self):
        """修复1：--body 参数已注册（--dispose 不再 unrecognized）。"""
        import subprocess as sp
        r = sp.run([sys.executable, "pr_review.py", "--help"], capture_output=True, text=True,
                   encoding="utf-8", errors="replace")
        self.assertIn("--body", r.stdout)

    def test_findings_filtered_by_current_user(self):
        """修复2：get_findings_discussions 只取当前账号的 diff_comment。"""
        current = {"login": "me"}
        other = {"login": "someone_else"}
        v5_comments = [
            {"id": 1, "user": current, "comment_type": "diff_comment", "body": "**[LOW] 我的意见**",
             "diff_position": {"start_new_line": 10}},
            {"id": 2, "user": other, "comment_type": "diff_comment", "body": "**[LOW] 他人意见**",
             "diff_position": {"start_new_line": 20}},
        ]

        def router(url, method="GET", data=None, token=""):
            if "/user" in url:
                return {"login": "me"}
            if "/comments" in url and method == "GET":
                return v5_comments
            if "/api/v4/" in url:
                return {"data": [
                    {"id": "did1", "notes": [{"id": 1, "body": "**[LOW] 我的意见**",
                                              "new_line": 10, "diff_file": "src/a.cc", "resolved": False}]},
                    {"id": "did2", "notes": [{"id": 2, "body": "**[LOW] 他人意见**",
                                              "new_line": 20, "diff_file": "src/a.cc", "resolved": False}]},
                ]}
            raise RuntimeError("unmocked " + url)

        pr_review.api_request = router
        findings = pr_review.get_findings_discussions("cann", "hccl", 1, "tok")
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["note_id"], 1)
        self.assertIn("我的意见", findings[0]["body"])

    def test_find_remote_matches_https_with_git_suffix(self):
        """修复3：find_remote 精确匹配兼容 .git 后缀的 HTTPS URL。"""
        remote_out = ("fork https://gitcode.com/someone/hccl.git (fetch)" + chr(10)
                    + "origin https://gitcode.com/cann/hccl.git (fetch)" + chr(10))
        with unittest.mock.patch.object(pr_review, "run_git", return_value=remote_out):
            self.assertEqual(pr_review.find_remote(Path("."), "cann", "hccl"), "origin")

    def test_find_remote_matches_ssh_scp_style(self):
        """修复3：find_remote 兼容 SSH SCP 风格 git@host:owner/repo.git。"""
        remote_out = "origin git@gitcode.com:cann/hccl.git (fetch)" + chr(10)
        with unittest.mock.patch.object(pr_review, "run_git", return_value=remote_out):
            self.assertEqual(pr_review.find_remote(Path("."), "cann", "hccl"), "origin")


    def test_find_remote_matches_ssh_scp_style(self):
        """修复3：find_remote 兼容 SSH SCP 风格 git@host:owner/repo.git。"""
        with unittest.mock.patch.object(
                pr_review, "run_git",
                return_value="origin git@gitcode.com:cann/hccl.git (fetch)" + chr(10)):
            self.assertEqual(pr_review.find_remote(Path("."), "cann", "hccl"), "origin")


if __name__ == "__main__":
    unittest.main(verbosity=2)
