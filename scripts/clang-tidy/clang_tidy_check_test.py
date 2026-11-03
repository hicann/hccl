#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Unit tests for the pure helpers of clang_tidy_check.py.

Run: python3 scripts/clang-tidy/clang_tidy_check_test.py

Covers changed-file discovery (cwd independence), -U0 added-line hunk
parsing, warning parsing (file/line/check) and compile database loading.
Shallow-clone healing: a CI-style --depth=1 checkout of a merge ref must
deepen itself until the merge base is reachable, and diff failures must
surface as None (blocking in main) instead of empty results. main() fail
paths (unreachable merge base, failed diff, unresolvable base ref in CI)
block with exit 1.
Requires git on PATH; all scenarios run inside throwaway repositories.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import unittest.mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import clang_tidy_check as ctc


def _git(args, cwd):
    return subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True)


class GitRepoTestCase(unittest.TestCase):
    """Base class providing a throwaway git repo with one committed file."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="clang_tidy_check_test_")
        os.makedirs(os.path.join(self.tmp, "src"))
        self._write("src/foo.cc", "int a;\nint b;\nint c;\nint d;\nint e;\n")
        _git(["init", "-q"], self.tmp)
        # pin the default branch name: ensure_merge_base's deepen refspec
        # targets refs/heads/master and must not depend on init.defaultBranch
        _git(["symbolic-ref", "HEAD", "refs/heads/master"], self.tmp)
        _git(["config", "user.email", "ut@example.com"], self.tmp)
        _git(["config", "user.name", "ut"], self.tmp)
        _git(["add", "."], self.tmp)
        _git(["commit", "-q", "-m", "base"], self.tmp)
        self.base_ref = _git(["rev-parse", "HEAD"], self.tmp).stdout.strip()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, rel, content):
        path = os.path.join(self.tmp, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fp:
            fp.write(content)

    def _commit(self, rel):
        _git(["add", rel], self.tmp)
        _git(["commit", "-q", "-m", "change"], self.tmp)


class TestGetAddedLineRanges(GitRepoTestCase):
    def test_inserted_lines(self):
        self._write("src/foo.cc", "int a;\nint added1;\nint added2;\nint b;\nint c;\nint d;\nint e;\n")
        self._commit("src/foo.cc")
        added = ctc.get_added_line_ranges(self.base_ref, "src/foo.cc", self.tmp)
        self.assertEqual(added, {2, 3})

    def test_lines_added_at_top(self):
        self._write("src/foo.cc", "int z1;\nint z2;\nint a;\nint b;\nint c;\nint d;\nint e;\n")
        self._commit("src/foo.cc")
        added = ctc.get_added_line_ranges(self.base_ref, "src/foo.cc", self.tmp)
        self.assertEqual(added, {1, 2})

    def test_no_added_lines_on_pure_deletion(self):
        self._write("src/foo.cc", "int a;\nint b;\nint c;\nint e;\n")
        self._commit("src/foo.cc")
        added = ctc.get_added_line_ranges(self.base_ref, "src/foo.cc", self.tmp)
        self.assertEqual(added, set())

    def test_diff_failure_returns_none_not_empty(self):
        # a failed diff must never be mistaken for "no added lines"
        self._write("src/foo.cc", "int a;\nint added;\n")
        self._commit("src/foo.cc")
        broken = subprocess.CompletedProcess(args=[], returncode=128, stdout="", stderr="fatal: no merge base")
        with unittest.mock.patch.object(ctc, "run", return_value=broken):
            self.assertIsNone(ctc.get_added_line_ranges(self.base_ref, "src/foo.cc", self.tmp))


class TestGetChangedFiles(GitRepoTestCase):
    def test_cpp_only_and_cwd_independent(self):
        self._write("src/bar.cc", "int f;\n")
        self._write("doc/readme.md", "text\n")
        self._commit("src/bar.cc")
        self._commit("doc/readme.md")
        sub = os.path.join(self.tmp, "build")
        os.makedirs(sub)
        old = os.getcwd()
        os.chdir(sub)  # simulate build.sh invoking the check from the build dir
        try:
            files = ctc.get_changed_files(self.base_ref, self.tmp)
        finally:
            os.chdir(old)
        self.assertEqual(files, ["src/bar.cc"])

    def test_renamed_and_modified_file_included(self):
        # rename + modify: --diff-filter=ACMR must still list the new path
        _git(["mv", "src/foo.cc", "src/renamed.cc"], self.tmp)
        self._write("src/renamed.cc", "int a;\nint b;\nint c;\nint d;\nint e;\nint newline;\n")
        self._commit("src/renamed.cc")
        files = ctc.get_changed_files(self.base_ref, self.tmp)
        self.assertEqual(files, ["src/renamed.cc"])

    def test_diff_failure_returns_none_not_empty(self):
        # a failed diff must never be mistaken for "no changed files" (the
        # shallow-clone silent-skip regression)
        self._write("src/bar.cc", "int f;\n")
        self._commit("src/bar.cc")
        broken = subprocess.CompletedProcess(args=[], returncode=128, stdout="", stderr="fatal: no merge base")
        with unittest.mock.patch.object(ctc, "run", return_value=broken):
            self.assertIsNone(ctc.get_changed_files(self.base_ref, self.tmp))

    def test_non_ascii_path_is_not_quoted_away(self):
        # with core.quotepath=true (git default) the name-only diff emits
        # quoted/escaped paths that never match a real file: the change would
        # silently vanish from the interception
        self._write("src/算子.cc", "int a;\nint b;\n")
        _git(["add", "src/算子.cc"], self.tmp)
        _git(["commit", "-q", "-m", "cjk path"], self.tmp)
        files = ctc.get_changed_files(self.base_ref, self.tmp)
        self.assertEqual(files, ["src/算子.cc"])


class TestShallowHealing(GitRepoTestCase):
    """CI checkouts are --depth=1 fetches of the PR merge ref: the merge base
    is unreachable until the history is deepened."""

    def _make_merge_ref(self):
        """Create a merge commit (base + PR head) like the CI merge ref and return its sha."""
        _git(["checkout", "-q", "-b", "pr"], self.tmp)
        self._write("src/foo.cc", "int a;\nint prline;\nint b;\nint c;\nint d;\nint e;\n")
        self._commit("src/foo.cc")
        pr_head = _git(["rev-parse", "HEAD"], self.tmp).stdout.strip()
        tree = _git(["rev-parse", pr_head + "^{tree}"], self.tmp).stdout.strip()
        m = _git(["commit-tree", "-p", self.base_ref, "-p", pr_head, "-m", "merge ref", tree], self.tmp)
        merge_sha = m.stdout.strip()
        # the CI fetches the merge ref by sha; a local file:// transport only
        # advertises refs, so publish the same commit on a branch for the test
        _git(["branch", "merge-ref", merge_sha], self.tmp)
        _git(["checkout", "-q", "--detach", self.base_ref], self.tmp)
        return merge_sha

    def _shallow_checkout(self, sha):
        """Replicate the CI checkout: init + remote + --depth=1 fetch of the merge ref."""
        dst = tempfile.mkdtemp(prefix="clang_tidy_shallow_")
        self.addCleanup(shutil.rmtree, dst, ignore_errors=True)
        _git(["init", "-q"], dst)
        _git(["remote", "add", "origin", "file://" + self.tmp], dst)
        _git(["fetch", "--depth=1", "origin", "merge-ref"], dst)
        _git(["checkout", "--force", sha], dst)
        return dst

    def test_ensure_merge_base_heals_shallow_checkout(self):
        merge_sha = self._make_merge_ref()
        dst = self._shallow_checkout(merge_sha)
        self.assertTrue(ctc.is_shallow_repo(dst))
        # the pre-fix failure mode: three-dot diff has no merge base
        self.assertNotEqual(_git(["diff", self.base_ref + "...HEAD"], dst).returncode, 0)
        # pin TARGET_BRANCH so a host-level export cannot skew the refspec
        with unittest.mock.patch.dict(os.environ, {"TARGET_BRANCH": "master"}):
            mb = ctc.ensure_merge_base(self.base_ref, dst)
        self.assertTrue(mb)  # healed: deepened until the merge base is reachable
        self.assertEqual(_git(["diff", "--name-only", "--diff-filter=ACMR", self.base_ref + "...HEAD", "--", "*.cc"], dst).returncode, 0)
        files = ctc.get_changed_files(self.base_ref, dst)
        self.assertEqual(files, ["src/foo.cc"])

    def test_ensure_merge_base_unshallow_recovers_unadvertised_merge_ref(self):
        # "F1" shape: the server keeps only master (pr + merge-ref branches
        # deleted). The deepen ladder cannot deliver the pr-side parent; the
        # restricted (refspec-carrying) unshallow must still recover it.
        merge_sha = self._make_merge_ref()
        dst = self._shallow_checkout(merge_sha)
        # pre-fetch master like resolve_base_ref does in CI (the probe below
        # still fails: the shallow boundary hides HEAD's parents)
        _git(["fetch", "--quiet", "--no-tags", "origin", "+refs/heads/master:refs/remotes/origin/master"], dst)
        _git(["branch", "-D", "pr", "merge-ref"], self.tmp)
        # trace the git fetches so the test proves WHICH healing path ran:
        # the ladder must fail (3 deepen fetches, probe still failing) and
        # the unshallow must be the one that recovers
        fetch_calls = []
        real_run = ctc.run
        probe_failures = {"count": 0}

        def traced_run(cmd, cwd=None, timeout=None):
            if cmd[1] == "fetch":
                fetch_calls.append(" ".join(cmd))
            r = real_run(cmd, cwd=cwd, timeout=timeout)
            if cmd[1] == "merge-base" and r.returncode != 0:
                probe_failures["count"] += 1
            return r

        with unittest.mock.patch.dict(os.environ, {"TARGET_BRANCH": "master"}), \
             unittest.mock.patch.object(ctc, "run", side_effect=traced_run):
            mb = ctc.ensure_merge_base(self.base_ref, dst)
        self.assertTrue(mb)
        # the ladder genuinely ran and failed before the unshallow recovered
        self.assertEqual(len([c for c in fetch_calls if "--deepen=" in c]), 3)
        self.assertTrue(any("--unshallow" in c for c in fetch_calls))
        self.assertGreater(probe_failures["count"], 0)
        files = ctc.get_changed_files(self.base_ref, dst)
        self.assertEqual(files, ["src/foo.cc"])

    def test_deepen_fetch_failure_falls_back_to_refspecless_unshallow(self):
        # a failing deepen fetch ("couldn't find remote ref": the derived
        # target branch does not exist) must NOT be read as an unreachable
        # remote: the refspec-less unshallow expands from the shallow boundary
        # and heals anyway - otherwise a wrong branch name would hard-block
        # every PR in CI
        calls = []

        def fake_run(cmd, cwd=None, timeout=None):
            calls.append(list(cmd))
            if cmd[1] == "merge-base":
                unshallowed = any("--unshallow" in " ".join(c) for c in calls[:-1])
                rc, out = (0, "abc123\n") if unshallowed else (1, "")
                return subprocess.CompletedProcess(args=[], returncode=rc, stdout=out, stderr="")
            if cmd[1] == "rev-parse" and "--is-shallow-repository" in cmd:
                return subprocess.CompletedProcess(args=[], returncode=0, stdout="true\n", stderr="")
            if cmd[1] == "remote":
                return subprocess.CompletedProcess(args=[], returncode=0, stdout="https://example.com/x.git\n", stderr="")
            if cmd[1] == "fetch" and "--deepen=" in " ".join(cmd):
                return subprocess.CompletedProcess(args=[], returncode=1, stdout="",
                                                    stderr="fatal: couldn't find remote ref refs/heads/master")
            return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

        with unittest.mock.patch.object(ctc, "run", side_effect=fake_run):
            self.assertEqual(ctc.ensure_merge_base("origin/master", self.tmp), "abc123")
        unshallow_calls = [" ".join(c) for c in calls if "--unshallow" in " ".join(c)]
        self.assertEqual(len(unshallow_calls), 1)
        # the fallback unshallow must be refspec-less: a dangling refspec
        # would make it fail the same way the deepen fetches did
        self.assertNotIn("refs/heads/", unshallow_calls[0])

    def test_deepen_and_unshallow_both_failing_returns_empty(self):
        # both heal fetches failing (truly unreachable remote): no sha
        calls = []

        def fake_run(cmd, cwd=None, timeout=None):
            calls.append(list(cmd))
            if cmd[1] == "merge-base":
                return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="")
            if cmd[1] == "rev-parse" and "--is-shallow-repository" in cmd:
                return subprocess.CompletedProcess(args=[], returncode=0, stdout="true\n", stderr="")
            if cmd[1] == "remote":
                return subprocess.CompletedProcess(args=[], returncode=0, stdout="https://example.com/x.git\n", stderr="")
            if cmd[1] == "fetch":
                return subprocess.CompletedProcess(args=[], returncode=1, stdout="",
                                                    stderr="fatal: unable to access: connection refused")
            return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

        with unittest.mock.patch.object(ctc, "run", side_effect=fake_run):
            self.assertEqual(ctc.ensure_merge_base("origin/master", self.tmp), "")
        self.assertTrue(any("--unshallow" in " ".join(c) for c in calls))

    def test_ensure_merge_base_ok_in_full_repo(self):
        self._write("src/foo.cc", "int a;\nint added;\n")
        self._commit("src/foo.cc")
        mb = ctc.ensure_merge_base(self.base_ref, self.tmp)
        self.assertEqual(mb, self.base_ref)

    def test_ensure_merge_base_unreachable_remote_returns_empty(self):
        merge_sha = self._make_merge_ref()
        dst = self._shallow_checkout(merge_sha)
        _git(["remote", "set-url", "origin", "file:///nonexistent/repo"], dst)
        self.assertEqual(ctc.ensure_merge_base(self.base_ref, dst), "")


class TestTargetBranchOf(unittest.TestCase):
    """_target_branch_of must strip only real prefixes, never the branch's
    own directory segments (release/v1.0 stays release/v1.0)."""

    def _tb(self, base_ref, remote, target_branch):
        with unittest.mock.patch.dict(os.environ, {"TARGET_BRANCH": target_branch}):
            return ctc._target_branch_of(base_ref, remote)

    def test_plain_branch_name_with_slash_kept(self):
        self.assertEqual(self._tb("origin/master", "origin", "release/v1.0"), "release/v1.0")

    def test_remote_qualified_value_stripped(self):
        self.assertEqual(self._tb("origin/master", "origin", "origin/release/v1.0"), "release/v1.0")

    def test_refs_heads_prefix_stripped(self):
        self.assertEqual(self._tb("origin/master", "origin", "refs/heads/master"), "master")

    def test_base_ref_remote_qualified_derivation_keeps_slashes(self):
        self.assertEqual(self._tb("upstream/release/v1.0", "upstream", ""), "release/v1.0")

    def test_base_ref_gitcode_upstream_fallback_derivation(self):
        # resolve_base_ref's canonical-URL fallback returns
        # gitcode-upstream/<tb>; the branch name must still be derived
        self.assertEqual(self._tb("gitcode-upstream/master", "origin", ""), "master")
        self.assertEqual(self._tb("gitcode-upstream/release/v1.0", "origin", ""), "release/v1.0")

    def test_base_ref_local_branch_used_verbatim(self):
        self.assertEqual(self._tb("feature/x", "origin", ""), "feature/x")

    def test_bare_sha_falls_back_to_master(self):
        self.assertEqual(self._tb("0123abcd", "origin", ""), "master")


class TestMainFailClosed(GitRepoTestCase):
    """main() must block (exit 1) when the incremental diff cannot be computed,
    and block on an unresolvable base ref in CI (TARGET_BRANCH set) while still
    skipping it on an offline developer box."""

    def setUp(self):
        super().setUp()
        os.makedirs(os.path.join(self.tmp, "build"), exist_ok=True)
        with open(os.path.join(self.tmp, "build", "compile_commands.json"), "w", encoding="utf-8") as fp:
            json.dump([], fp)
        self._write("src/foo.cc", "int a;\nint b;\n")
        self._commit("src/foo.cc")

    def _run_main(self, *extra):
        old = os.getcwd()
        os.chdir(self.tmp)
        try:
            argv = ["clang_tidy_check.py", "--build-dir", "build"] + list(extra)
            with unittest.mock.patch.object(sys, "argv", argv):
                return ctc.main()
        finally:
            os.chdir(old)

    def test_merge_base_unreachable_blocks_in_ci_skips_locally(self):
        # same layering as the unresolvable base ref: CI blocks, a local box
        # that cannot heal its shallow clone (offline) only warns and skips
        with unittest.mock.patch.object(ctc.shutil, "which", return_value="/usr/bin/clang-tidy"), \
             unittest.mock.patch.object(ctc, "resolve_base_ref", return_value="origin/master"), \
             unittest.mock.patch.object(ctc, "ensure_merge_base", return_value=""):
            with unittest.mock.patch.dict(os.environ, {"TARGET_BRANCH": "master"}):
                self.assertEqual(self._run_main(), 1)
            with unittest.mock.patch.dict(os.environ, {"TARGET_BRANCH": ""}):
                self.assertEqual(self._run_main(), 0)

    def test_changed_files_diff_failure_blocks(self):
        with unittest.mock.patch.object(ctc.shutil, "which", return_value="/usr/bin/clang-tidy"), \
             unittest.mock.patch.object(ctc, "resolve_base_ref", return_value=self.base_ref), \
             unittest.mock.patch.object(ctc, "ensure_merge_base", return_value=self.base_ref), \
             unittest.mock.patch.object(ctc, "get_changed_files", return_value=None):
            self.assertEqual(self._run_main(), 1)

    def test_unresolvable_base_ref_blocks_in_ci_skips_locally(self):
        with unittest.mock.patch.object(ctc.shutil, "which", return_value="/usr/bin/clang-tidy"), \
             unittest.mock.patch.object(ctc, "resolve_base_ref", return_value=""):
            with unittest.mock.patch.dict(os.environ, {"TARGET_BRANCH": "master"}):
                self.assertEqual(self._run_main(), 1)
            # empty (not deleted) TARGET_BRANCH selects the local-box layer
            with unittest.mock.patch.dict(os.environ, {"TARGET_BRANCH": ""}):
                self.assertEqual(self._run_main(), 0)

    def test_unreadable_compile_db_blocks(self):
        # the db file exists (checked before) but cannot be parsed: the
        # interception would be silently empty, so main must block
        with unittest.mock.patch.object(ctc.shutil, "which", return_value="/usr/bin/clang-tidy"), \
             unittest.mock.patch.object(ctc, "resolve_base_ref", return_value=self.base_ref), \
             unittest.mock.patch.object(ctc, "ensure_merge_base", return_value=self.base_ref), \
             unittest.mock.patch.object(ctc, "get_changed_files", return_value=["src/foo.cc"]), \
             unittest.mock.patch.object(ctc, "load_compile_db", return_value=None):
            self.assertEqual(self._run_main(), 1)

    def test_per_file_added_diff_failure_blocks(self):
        # a per-file added-line diff failure must reach the diff_failed
        # blocking branch (never be mistaken for "no added lines")
        with unittest.mock.patch.object(ctc.shutil, "which", return_value="/usr/bin/clang-tidy"), \
             unittest.mock.patch.object(ctc, "resolve_base_ref", return_value=self.base_ref), \
             unittest.mock.patch.object(ctc, "ensure_merge_base", return_value=self.base_ref), \
             unittest.mock.patch.object(ctc, "get_changed_files", return_value=["src/foo.cc"]), \
             unittest.mock.patch.object(ctc, "load_compile_db", return_value={ctc.norm_path(os.path.join(self.tmp, "src/foo.cc"))}), \
             unittest.mock.patch.object(ctc, "get_added_line_ranges", return_value=None):
            self.assertEqual(self._run_main(), 1)

    def test_ensure_merge_base_unshallow_fallback(self):
        # deepen walks first-parent chains and cannot reach merge-ref parents
        # that are not advertised as refs; the unshallow last resort must be
        # attempted and must win
        calls = []

        def fake_run(cmd, cwd=None, timeout=None):
            calls.append(list(cmd))
            if cmd[1] == "merge-base":
                unshallowed = any("--unshallow" in cc for cc in calls[:-1])
                rc, out = (0, "abc123\n") if unshallowed else (1, "")
                return subprocess.CompletedProcess(args=[], returncode=rc, stdout=out, stderr="")
            if cmd[1] == "rev-parse" and "--is-shallow-repository" in cmd:
                return subprocess.CompletedProcess(args=[], returncode=0, stdout="true\n", stderr="")
            if cmd[1] == "remote":
                return subprocess.CompletedProcess(args=[], returncode=0, stdout="https://example.com/x.git\n", stderr="")
            return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

        with unittest.mock.patch.object(ctc, "run", side_effect=fake_run):
            self.assertEqual(ctc.ensure_merge_base("origin/master", self.tmp), "abc123")
        self.assertTrue(any("--unshallow" in cc for cc in calls))


class TestParseWarnings(unittest.TestCase):
    WARN_OWN = "src/foo.cc:42:7: warning: function has 7 parameters (threshold 6) [readability-function-size]"
    WARN_HEADER = "include/proj/bar.h:10:5: warning: function is too complex [readability-function-cognitive-complexity]"
    WARN_OTHER_CHECK = "src/foo.cc:99:1: warning: use a trailing return type [modernize-use-trailing-return-type]"
    WARN_ERROR_LINE = "error: unable to handle compilation, expected exactly one compiler job in ''"

    def test_yields_file_line_check_for_target_checks_only(self):
        out = list(ctc.parse_warnings(
            "\n".join([self.WARN_OWN, self.WARN_HEADER, self.WARN_OTHER_CHECK, self.WARN_ERROR_LINE])))
        self.assertEqual(out, [
            ("src/foo.cc", 42, "readability-function-size"),
            ("include/proj/bar.h", 10, "readability-function-cognitive-complexity"),
        ])

    def test_ignores_non_warning_lines(self):
        out = list(ctc.parse_warnings("1 warning generated.\nSuppressed 1 warnings.\n" + self.WARN_OTHER_CHECK))
        self.assertEqual(out, [])

    def test_windows_drive_letter_path_is_parsed(self):
        # clang-tidy on Windows reports absolute paths like D:\repo\src\foo.cc
        # or D:/repo/src/foo.cc; the file field must survive the drive colon
        for path in ("D:\\repo\\src\\foo.cc:42:7: warning: function has 7 parameters [readability-function-size]",
                     "D:/repo/src/foo.cc:42:7: warning: function has 7 parameters [readability-function-size]"):
            out = list(ctc.parse_warnings(path))
            self.assertEqual(out, [("D:\\repo\\src\\foo.cc" if "\\" in path else "D:/repo/src/foo.cc", 42, "readability-function-size")])


class TestDefaultBaseRef(unittest.TestCase):
    def test_defaults_to_upstream_master(self):
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(ctc.default_base_ref(), "upstream/master")

    def test_target_branch_selects_upstream_remote(self):
        with unittest.mock.patch.dict(os.environ, {"TARGET_BRANCH": "dev"}, clear=True):
            self.assertEqual(ctc.default_base_ref(), "upstream/dev")


class TestResolveBaseRefExplicitLocalBranch(GitRepoTestCase):
    def test_local_branch_with_slash_is_used_as_is(self):
        # an explicitly passed local branch containing '/' must not be
        # silently replaced by the upstream/origin fallback
        _git(["branch", "feature/x"], self.tmp)
        old = os.getcwd()
        os.chdir(self.tmp)
        try:
            self.assertEqual(ctc.resolve_base_ref("feature/x"), "feature/x")
        finally:
            os.chdir(old)


class TestDetectRepoBasename(GitRepoTestCase):
    def _in_repo(self, fn):
        old = os.getcwd()
        os.chdir(self.tmp)
        try:
            return fn()
        finally:
            os.chdir(old)

    def test_canonical_https_url(self):
        _git(["remote", "add", "origin", "https://gitcode.com/cann/hccl.git"], self.tmp)
        self.assertEqual(self._in_repo(ctc.detect_repo_basename), "hccl")

    def test_fork_url_with_token(self):
        _git(["remote", "add", "origin", "https://user:token@gitcode.com/yanzhenghang/hccl.git"], self.tmp)
        self.assertEqual(self._in_repo(ctc.detect_repo_basename), "hccl")

    def test_ssh_url(self):
        _git(["remote", "add", "origin", "git@gitcode.com:cann/hccl.git"], self.tmp)
        self.assertEqual(self._in_repo(ctc.detect_repo_basename), "hccl")

    def test_no_gitcode_remote_returns_empty(self):
        _git(["remote", "add", "origin", "https://mirror.example.com/team/hccl.git"], self.tmp)
        self.assertEqual(self._in_repo(ctc.detect_repo_basename), "")


class TestRunWithTimeout(unittest.TestCase):
    def test_timeout_becomes_failed_command(self):
        # a hung remote must surface as a non-zero return code, not as an
        # unhandled exception that crashes the gate
        def fake_run(args, **kwargs):
            raise subprocess.TimeoutExpired(cmd=args, timeout=kwargs.get("timeout"))
        with unittest.mock.patch.object(ctc.subprocess, "run", side_effect=fake_run):
            r = ctc.run(["git", "fetch", "origin"], timeout=1)
        self.assertEqual(r.returncode, 124)
        self.assertEqual(r.stderr, "timeout after 1s")


class TestLoadCompileDb(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="clang_tidy_check_cdb_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_missing_file_returns_none(self):
        self.assertIsNone(ctc.load_compile_db(self.tmp))

    def test_relative_entries_resolved_against_build_dir(self):
        # platform-neutral absolute path (Linux: /abs/..., Windows: C:\abs\...)
        abs_file = os.path.join(os.path.abspath(os.sep), "abs", "src", "bar.cc")
        with open(os.path.join(self.tmp, "compile_commands.json"), "w", encoding="utf-8") as fp:
            json.dump([{"file": "src/foo.cc"}, {"file": abs_file}], fp)
        db = ctc.load_compile_db(self.tmp)
        self.assertEqual(db, {
            ctc.norm_path(os.path.join(self.tmp, "src/foo.cc")),
            ctc.norm_path(abs_file),
        })


if __name__ == "__main__":
    unittest.main()
