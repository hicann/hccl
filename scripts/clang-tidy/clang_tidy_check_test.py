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
from pathlib import Path
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
        _git(["remote", "add", "origin", Path(self.tmp).as_uri()], dst)
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
        test_file = ctc.norm_path(os.path.join(self.tmp, "src/foo.cc"))
        compile_db = {test_file: [ctc.norm_path(self.tmp)]}
        with unittest.mock.patch.object(ctc.shutil, "which", return_value="/usr/bin/clang-tidy"), \
             unittest.mock.patch.object(ctc, "resolve_base_ref", return_value=self.base_ref), \
             unittest.mock.patch.object(ctc, "ensure_merge_base", return_value=self.base_ref), \
             unittest.mock.patch.object(ctc, "get_changed_files", return_value=["src/foo.cc"]), \
             unittest.mock.patch.object(ctc, "load_compile_db", return_value=compile_db), \
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

    def test_relative_warning_path_matches_compilation_directory(self):
        with tempfile.TemporaryDirectory() as root:
            source = os.path.join(root, "src", "a.cpp")
            compile_dir = os.path.join(root, "nested")
            os.makedirs(compile_dir)
            self.assertTrue(ctc.warning_matches_source("../src/a.cpp", source, [compile_dir]))
            self.assertFalse(ctc.warning_matches_source("../src/other.cpp", source, [compile_dir]))


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


class TestRunClangTidy(unittest.TestCase):
    def test_skip_and_actual_failure_have_distinct_diagnostics(self):
        with self.assertLogs(ctc.log, level="WARNING") as logs:
            ctc.log_source_analysis_failure("/src/a.cc", 1, "Skipping /src/a.cc. Compile command not found.\n")
        self.assertIn("skipped /src/a.cc: no matching compile command", logs.output[0])
        self.assertNotIn("exited", logs.output[0])
        with self.assertLogs(ctc.log, level="WARNING") as logs:
            ctc.log_source_analysis_failure("/src/a.cc", 2, "fatal error")
        self.assertIn("exited 2", logs.output[0])

    def test_missing_compile_command_is_failure_even_when_tool_exits_zero(self):
        result = subprocess.CompletedProcess([], 0, "", "Skipping /src/a.cc. Compile command not found.\n")
        with unittest.mock.patch.object(ctc, "run_tool", return_value=result):
            output, code = ctc.run_clang_tidy("/src/a.cc", "/build", "/config")
        self.assertEqual(code, 1)
        self.assertIn("Compile command not found", output)

    def test_successful_analysis_is_not_changed(self):
        result = subprocess.CompletedProcess([], 0, "", "")
        with unittest.mock.patch.object(ctc, "run_tool", return_value=result):
            self.assertEqual(ctc.run_clang_tidy("/src/a.cc", "/build", "/config"), ("", 0))

    def test_missing_command_retries_relative_directory_using_temporary_database(self):
        with tempfile.TemporaryDirectory() as root:
            build = os.path.join(root, "build")
            os.mkdir(build)
            compile_dir = os.path.join(root, "relative-build")
            os.mkdir(compile_dir)
            os.mkdir(os.path.join(root, "src"))
            source = os.path.join(root, "src", "a.cc")
            with open(source, "w", encoding="utf-8") as fp:
                fp.write("int value;\n")
            path = os.path.join(build, "compile_commands.json")
            entries = [{"directory": "relative-build", "file": "../src/a.cc", "arguments": ["clang++", "-c", "../src/a.cc"]}]
            with open(path, "w", encoding="utf-8") as fp:
                json.dump(entries, fp)
            calls = []

            def fake_run(cmd, cwd=None):
                calls.append(cmd)
                if len(calls) == 1:
                    return subprocess.CompletedProcess(cmd, 0, "", f"Skipping {source}. Compile command not found.\n")
                retry_dir = cmd[cmd.index("-p") + 1]
                with open(os.path.join(retry_dir, "compile_commands.json"), encoding="utf-8") as fp:
                    retry = json.load(fp)
                self.assertEqual(retry[0]["directory"], compile_dir)
                self.assertEqual(retry[0]["file"], "../src/a.cc")
                self.assertTrue(os.path.isfile(os.path.join(retry[0]["directory"], retry[0]["file"])))
                self.assertEqual(retry[0]["arguments"], entries[0]["arguments"])
                return subprocess.CompletedProcess(cmd, 0, "warning output", "")

            original_cwd = os.getcwd()
            try:
                os.chdir(root)
                with unittest.mock.patch.object(ctc, "run_tool", side_effect=fake_run):
                    self.assertEqual(ctc.run_clang_tidy(source, build, "/config"), ("warning output", 0))
            finally:
                os.chdir(original_cwd)
            self.assertEqual(len(calls), 2)
            self.assertFalse(os.path.exists(calls[1][calls[1].index("-p") + 1]))
            with open(path, encoding="utf-8") as fp:
                self.assertEqual(json.load(fp), entries)

    def test_database_load_error_retries_missing_or_invalid_directories(self):
        errors = (
            (0, 'Error while trying to load a compilation database:\nRunning without flags.\n'),
            (250, 'LLVM ERROR: Cannot chdir into ""!\n'),
            (0, 'Skipping file.cc. Compile command not found.\n'),
            (0, 'json-compilation-database: Missing key: \"directory\"\n'),
            (0, 'Running without flags.\n'),
        )
        with tempfile.TemporaryDirectory() as root:
            for value in (None, "", 42, "missing"):
                entry = {"file": "file.cc", "arguments": ["clang++", "-c", "file.cc"]}
                if value != "missing":
                    entry["directory"] = value
                with open(os.path.join(root, "compile_commands.json"), "w", encoding="utf-8") as fp:
                    json.dump([entry], fp)
                for first_code, first_error in errors:
                    calls = []

                    def fake_run(cmd, cwd=None):
                        calls.append(cmd)
                        if len(calls) == 1:
                            return subprocess.CompletedProcess(cmd, first_code, "", first_error)
                        retry_dir = cmd[cmd.index("-p") + 1]
                        with open(os.path.join(retry_dir, "compile_commands.json"), encoding="utf-8") as fp:
                            repaired = json.load(fp)
                        self.assertEqual(repaired[0]["directory"], ctc.norm_path(root))
                        return subprocess.CompletedProcess(cmd, 0, "analyzed", "")

                    with self.subTest(directory=value, first_code=first_code):
                        with unittest.mock.patch.object(ctc, "run_tool", side_effect=fake_run):
                            self.assertEqual(ctc.run_clang_tidy("file.cc", root, "/config"), ("analyzed", 0))
                        self.assertEqual(len(calls), 2)

    def test_database_load_error_without_safe_retry_is_failure(self):
        with tempfile.TemporaryDirectory() as root:
            with open(os.path.join(root, "compile_commands.json"), "w", encoding="utf-8") as fp:
                json.dump([{"directory": root, "file": "file.cc"}], fp)
            errors = (
                subprocess.CompletedProcess([], 0, "", "Error while trying to load a compilation database:\n"),
                subprocess.CompletedProcess([], 250, "", 'LLVM ERROR: Cannot chdir into "bad"!\n'),
            )
            for result in errors:
                with self.subTest(returncode=result.returncode):
                    with unittest.mock.patch.object(ctc, "run_tool", return_value=result) as tool:
                        output, code = ctc.run_clang_tidy("file.cc", root, "/config")
                    self.assertNotEqual(code, 0)
                    self.assertEqual(output, result.stderr)
                    self.assertEqual(tool.call_count, 1)
                    with self.assertLogs(ctc.log, level="WARNING") as logs:
                        ctc.log_source_analysis_failure("file.cc", code, output)
                    self.assertIn("compilation database unavailable for file.cc", logs.output[0])
                    self.assertNotIn("exited", logs.output[0])


    def test_retry_does_not_hide_missing_command_or_tool_failure(self):
        with tempfile.TemporaryDirectory() as root:
            with open(os.path.join(root, "compile_commands.json"), "w", encoding="utf-8") as fp:
                json.dump([{"directory": "relative", "file": "file.cc"}], fp)
            skipped = subprocess.CompletedProcess([], 0, "", "Skipping file.cc. Compile command not found.\n")
            for second, expected in ((skipped, 1), (subprocess.CompletedProcess([], 2, "", "fatal error"), 2)):
                with self.subTest(second=second):
                    with unittest.mock.patch.object(ctc, "run_tool", side_effect=[skipped, second]):
                        _, code = ctc.run_clang_tidy("file.cc", root, "/config")
                    self.assertEqual(code, expected)


    def test_fallback_words_in_source_diagnostics_do_not_block(self):
        message = 'a.cc:1:1: note: Running without flags.\n  puts("Running without flags.");\n'
        result = subprocess.CompletedProcess([], 0, message, "")
        with unittest.mock.patch.object(ctc, "run_tool", return_value=result):
            self.assertEqual(ctc.run_clang_tidy("/src/a.cc", "/build", "/config"), (message, 0))

    def test_retry_cannot_report_success_without_database_flags(self):
        with tempfile.TemporaryDirectory() as root:
            with open(os.path.join(root, "compile_commands.json"), "w", encoding="utf-8") as fp:
                json.dump([{"directory": "relative", "file": "file.cc"}], fp)
            skipped = subprocess.CompletedProcess([], 0, "", "Skipping file.cc. Compile command not found.\n")
            for message in ('json-compilation-database: Missing key: "directory".\n', "Running without flags.\n"):
                with self.subTest(message=message):
                    fallback = subprocess.CompletedProcess([], 0, "", message)
                    with unittest.mock.patch.object(ctc, "run_tool", side_effect=[skipped, fallback]):
                        self.assertEqual(ctc.run_clang_tidy("file.cc", root, "/config"), (message, 1))




class TestLoadCompileDb(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="clang_tidy_check_cdb_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_missing_file_returns_none(self):
        self.assertIsNone(ctc.load_compile_db(self.tmp))

    def test_main_blocks_empty_database_for_changed_source(self):
        (Path(self.tmp) / "compile_commands.json").write_text("[]", encoding="utf-8")
        old = os.getcwd()
        try:
            with unittest.mock.patch.object(sys, "argv", ["check", "--build-dir", self.tmp]), \
                 unittest.mock.patch.object(ctc, "get_repo_root", return_value=self.tmp), \
                 unittest.mock.patch.object(ctc.shutil, "which", return_value="clang-tidy"), \
                 unittest.mock.patch.object(ctc, "resolve_base_ref", return_value="base"), \
                 unittest.mock.patch.object(ctc, "ensure_merge_base", return_value="base"), \
                 unittest.mock.patch.object(ctc, "get_changed_files", return_value=["changed.cc"]), \
                 unittest.mock.patch.object(ctc, "get_added_line_ranges", return_value={1}), \
                 unittest.mock.patch.object(ctc, "run_clang_tidy") as tool:
                self.assertEqual(ctc.main(), 1)
            tool.assert_not_called()
        finally:
            os.chdir(old)

    def test_empty_database_preserves_header_only_analysis(self):
        (Path(self.tmp) / "compile_commands.json").write_text("[]", encoding="utf-8")
        old = os.getcwd()
        try:
            with unittest.mock.patch.object(sys, "argv", ["check", "--build-dir", self.tmp]), \
                 unittest.mock.patch.object(ctc, "get_repo_root", return_value=self.tmp), \
                 unittest.mock.patch.object(ctc.shutil, "which", return_value="clang-tidy"), \
                 unittest.mock.patch.object(ctc, "resolve_base_ref", return_value="base"), \
                 unittest.mock.patch.object(ctc, "ensure_merge_base", return_value="base"), \
                 unittest.mock.patch.object(ctc, "get_changed_files", return_value=["changed.h"]), \
                 unittest.mock.patch.object(ctc, "get_added_line_ranges", return_value={1}), \
                 unittest.mock.patch.object(ctc, "run_clang_tidy_standalone", return_value=("", 0)) as tool:
                self.assertEqual(ctc.main(), 0)
            tool.assert_called_once()
        finally:
            os.chdir(old)

    def test_main_rejects_invalid_database_after_source_changes(self):
        path = os.path.join(self.tmp, "compile_commands.json")
        for invalid in ("{", '{"entries": []}', '[{"file": 42}]'):
            with self.subTest(invalid=invalid):
                with open(path, "w", encoding="utf-8") as fp:
                    fp.write(invalid)
                old = os.getcwd()
                try:
                    with unittest.mock.patch.object(sys, "argv", ["check", "--build-dir", self.tmp]):
                        with unittest.mock.patch.object(ctc, "get_repo_root", return_value=self.tmp):
                            with unittest.mock.patch.object(ctc.shutil, "which", return_value="clang-tidy"):
                                with unittest.mock.patch.object(ctc, "resolve_base_ref", return_value="base"):
                                    with unittest.mock.patch.object(ctc, "get_changed_files", return_value=["changed.cc"]):
                                        with unittest.mock.patch.object(ctc, "ensure_merge_base", return_value="base"):
                                            self.assertEqual(ctc.main(), 1)
                finally:
                    os.chdir(old)

    def test_main_skips_if_database_not_generated(self):
        old = os.getcwd()
        try:
            with unittest.mock.patch.object(sys, "argv", ["check", "--build-dir", self.tmp]):
                with unittest.mock.patch.object(ctc, "get_repo_root", return_value=self.tmp):
                    with unittest.mock.patch.object(ctc.shutil, "which", return_value="clang-tidy"):
                        self.assertEqual(ctc.main(), 0)
        finally:
            os.chdir(old)

    def test_main_detects_warning_relative_to_compile_directory(self):
        compile_dir = os.path.join(self.tmp, "nested")
        os.makedirs(compile_dir)
        with open(os.path.join(self.tmp, "compile_commands.json"), "w", encoding="utf-8") as fp:
            json.dump([{"directory": compile_dir, "file": "../src/a.cpp"}], fp)
        output = "../src/a.cpp:1:5: warning: too many parameters [readability-function-size]\n"
        old = os.getcwd()
        try:
            with unittest.mock.patch.object(sys, "argv", ["check", "--build-dir", self.tmp]):
                with unittest.mock.patch.object(ctc, "get_repo_root", return_value=self.tmp):
                    with unittest.mock.patch.object(ctc.shutil, "which", return_value="clang-tidy"):
                        with unittest.mock.patch.object(ctc, "resolve_base_ref", return_value="base"):
                            with unittest.mock.patch.object(ctc, "get_changed_files", return_value=["src/a.cpp"]):
                                with unittest.mock.patch.object(ctc, "get_added_line_ranges", return_value={1}):
                                    with unittest.mock.patch.object(ctc, "run_clang_tidy", return_value=(output, 0)):
                                        with unittest.mock.patch.object(ctc, "ensure_merge_base", return_value="base"):
                                            self.assertEqual(ctc.main(), 1)
        finally:
            os.chdir(old)

    def test_malformed_entries_are_rejected(self):
        with open(os.path.join(self.tmp, "compile_commands.json"), "w", encoding="utf-8") as fp:
            json.dump([{"file": 42}, "not-a-dict", {"file": "valid.cc"}], fp)
        with self.assertRaises(ValueError):
            ctc.load_compile_db(self.tmp)

    def test_non_array_database_is_reported(self):
        with open(os.path.join(self.tmp, "compile_commands.json"), "w", encoding="utf-8") as fp:
            json.dump({"entries": []}, fp)
        with self.assertRaises(ValueError):
            ctc.load_compile_db(self.tmp)

    def test_relative_entries_resolved_against_build_dir(self):
        # platform-neutral absolute path (Linux: /abs/..., Windows: C:\abs\...)
        abs_file = os.path.join(os.path.abspath(os.sep), "abs", "src", "bar.cc")
        with open(os.path.join(self.tmp, "compile_commands.json"), "w", encoding="utf-8") as fp:
            json.dump([{"file": "src/foo.cc"}, {"file": abs_file}], fp)
        db = ctc.load_compile_db(self.tmp)
        self.assertEqual(set(db), {
            ctc.norm_path(os.path.join(self.tmp, "src/foo.cc")),
            ctc.norm_path(abs_file),
        })

    def test_empty_directory_falls_back_to_build_dir(self):
        for directory in (None, "", 42):
            with self.subTest(directory=directory):
                with open(os.path.join(self.tmp, "compile_commands.json"), "w", encoding="utf-8") as fp:
                    json.dump([{"directory": directory, "file": "foo.cc"}], fp)
                self.assertEqual(set(ctc.load_compile_db(self.tmp)), {
                    ctc.norm_path(os.path.join(self.tmp, "foo.cc")),
                })

    def test_relative_file_resolved_against_entry_directory(self):
        source_dir = os.path.join(self.tmp, "source")
        with open(os.path.join(self.tmp, "compile_commands.json"), "w", encoding="utf-8") as fp:
            json.dump([{"directory": source_dir, "file": "foo.cc"}], fp)
        db = ctc.load_compile_db(self.tmp)
        self.assertEqual(set(db), {ctc.norm_path(os.path.join(source_dir, "foo.cc"))})
        self.assertEqual(db[ctc.norm_path(os.path.join(source_dir, "foo.cc"))], [ctc.norm_path(source_dir)])


    def test_header_with_database_entry_analysis_failure_blocks(self):
        header = os.path.join(self.tmp, "header.h")
        args = ctc.argparse.Namespace(config_file="config", build_dir=self.tmp)
        database = {ctc.norm_path(header): [ctc.norm_path(self.tmp)]}
        with unittest.mock.patch.object(ctc, "get_added_line_ranges", return_value={1}):
            with unittest.mock.patch.object(ctc, "run_clang_tidy", return_value=("error", 1)):
                self.assertEqual(ctc.analyze_files([header], "base", self.tmp, database, args), 1)

    def test_source_without_database_entry_is_exempt_without_invoking_tool(self):
        source = os.path.join(self.tmp, "generated.cc")
        args = ctc.argparse.Namespace(config_file="config", build_dir=self.tmp)
        with unittest.mock.patch.object(ctc, "get_added_line_ranges", return_value={1}):
            with unittest.mock.patch.object(ctc, "run_clang_tidy") as tool:
                self.assertEqual(ctc.analyze_files([source], "base", self.tmp, {}, args), 0)
        tool.assert_not_called()

    def test_header_without_database_entry_analysis_failure_is_warn_only(self):
        args = ctc.argparse.Namespace(config_file="config", build_dir=self.tmp)
        with unittest.mock.patch.object(ctc, "get_added_line_ranges", return_value={1}):
            with unittest.mock.patch.object(ctc, "run_clang_tidy_standalone", return_value=("error", 1)):
                self.assertEqual(ctc.analyze_files(["header.h"], "base", self.tmp, {}, args), 0)




class TestRetryLifetime(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.old_cwd = os.getcwd()
        os.chdir(self.root)
        (self.root / "relative-build").mkdir()
        (self.root / "src").mkdir()
        self.sources = []
        entries = []
        for name in ("a.cc", "b.cc"):
            source = self.root / "src" / name
            source.write_text("int value;\n", encoding="utf-8")
            self.sources.append(str(source))
            entries.append({"directory": "relative-build", "file": "../src/" + name,
                            "arguments": ["clang++", "-c", "../src/" + name]})
        self.path = self.root / "compile_commands.json"
        self.path.write_text(json.dumps(entries), encoding="utf-8")
        self.args = ctc.argparse.Namespace(build_dir=str(self.root), config_file="/config")

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.temp.cleanup()

    def run_analysis(self, tool):
        with unittest.mock.patch.object(ctc, "get_added_line_ranges", return_value={1}), \
             unittest.mock.patch.object(ctc, "run_tool", side_effect=tool):
            return ctc.analyze_files(self.sources, "base", str(self.root),
                                     ctc.load_compile_db(str(self.root)), self.args)

    def test_retry_reuses_one_database_and_cleans_it(self):
        retry_dirs = []

        def tool(cmd, cwd=None):
            directory = cmd[cmd.index("-p") + 1]
            if directory == str(self.root):
                return subprocess.CompletedProcess(cmd, 0, "Skipping file.cc. Compile command not found.\n", "")
            retry_dirs.append(directory)
            self.assertTrue(Path(directory, "compile_commands.json").is_file())
            return subprocess.CompletedProcess(cmd, 0, "", "")

        self.assertEqual(self.run_analysis(tool), 0)
        self.assertEqual(len(retry_dirs), 2)
        self.assertEqual(len(set(retry_dirs)), 1)
        self.assertFalse(Path(retry_dirs[0]).exists())

    def test_changed_database_is_not_retried_with_stale_snapshot(self):
        calls = []

        def tool(cmd, cwd=None):
            calls.append(cmd)
            if cmd[cmd.index("-p") + 1] == str(self.root):
                return subprocess.CompletedProcess(cmd, 0, "Skipping file.cc. Compile command not found.\n", "")
            self.path.write_text(self.path.read_text(encoding="utf-8") + " ", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with self.assertLogs(ctc.log, level="WARNING") as logs:
            self.assertEqual(self.run_analysis(tool), 1)
        self.assertEqual(len(calls), 3)
        self.assertIn("changed during analysis", " ".join(logs.output))

    def test_retry_warning_reaches_incremental_violation_gate(self):
        self.sources = self.sources[:1]
        calls = []

        def tool(cmd, cwd=None):
            calls.append(cmd)
            if len(calls) == 1:
                return subprocess.CompletedProcess(cmd, 0, "Skipping file.cc. Compile command not found.\n", "")
            warning = "../src/a.cc:1:5: warning: too many parameters [readability-function-size]\n"
            return subprocess.CompletedProcess(cmd, 0, warning, "")

        with self.assertLogs(ctc.log, level="WARNING") as logs:
            self.assertEqual(self.run_analysis(tool), 1)
        self.assertEqual(len(calls), 2)
        self.assertIn("1 new-code complexity violation", " ".join(logs.output))

    def test_failure_logs_last_three_nonempty_diagnostics(self):
        with self.assertLogs(ctc.log, level="WARNING") as logs:
            ctc.log_source_analysis_failure("a.cc", 2, "old\n\n detail one \ndetail two\ndetail three\n")
        self.assertIn("detail one | detail two | detail three", " ".join(logs.output))
        self.assertNotIn("old", " ".join(logs.output))


if __name__ == "__main__":
    unittest.main()
