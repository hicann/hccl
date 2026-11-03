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
"""Clang-Tidy incremental check: block only complexity warnings on newly added lines.

Usage: python3 scripts/clang-tidy/clang_tidy_check.py [--build-dir build] [--base-ref upstream/master]
Exit 0: no new-code violation; Exit 1: new-code violation found (block).

Incremental base: the diff base defaults to the target branch of the gitcode
upstream remote (upstream/master, or upstream/<TARGET_BRANCH> in CI) and is
refreshed from the remote right before use, so "newly added" always means
"added by this branch relative to the latest upstream target branch". The
three-dot diff (base...HEAD) only counts lines added on this branch; lines
added upstream after the fork point are never attributed to this branch.

Shallow-clone healing (CI): CI checkouts are --depth=1 fetches of the PR merge
ref, whose parents sit behind the shallow boundary - `git diff base...HEAD`
then fails with "no merge base" and empty output. ensure_merge_base deepens the
history (1/10/100 levels) from the base ref's remote (falling back to
upstream/origin) until the merge base is reachable.

Blocking vs skipping (layering): a gate that cannot even start (clang-tidy not
installed, no compile_commands.json, no resolvable base ref, or a local
shallow clone whose merge base cannot be healed - typically an offline
developer box) skips with a warning and exit 0; a gate that has started but
cannot compute its incremental diff BLOCKS (exit 1) - a silently skipped diff
is an empty interception. In CI (TARGET_BRANCH set) an unresolvable base ref
or an unhealable merge base also blocks, because there it means a broken
pipeline rather than an offline box. The escape hatch for offline builds is
ENABLE_CLANG_TIDY_CHECK=off.

Exemptions (special cases):
  - // NOLINT(readability-function-size) / // NOLINTNEXTLINE(...) comments suppress
    warnings at the clang-tidy level, so suppressed warnings never reach this script.
  - Changed C/C++ sources without a compile_commands.json entry (e.g. generated or
    third-party sources) are skipped with a WARNING instead of being analyzed.

Limitations:
  - Warnings are anchored at the function definition line: growing the BODY of an
    existing function past a threshold (without touching its head line) is not
    flagged, matching the incremental (new-lines-only) interception model.
  - Standalone header analysis runs without include paths; a non-self-contained
    header that fails to parse is reported as a warning, never as a block.
"""
import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

WARN_RE = re.compile(r"^(?P<file>(?:[A-Za-z]:)?[^:]+):(?P<line>\d+):\d+:\s+warning:\s+.*\[(?P<check>[^\]]+)\]")
TARGET_CHECKS = ("readability-function-size", "readability-function-cognitive-complexity")
CPP_EXT = (".cc", ".cpp", ".c", ".cxx", ".h", ".hpp", ".hh", ".hxx")
HEADER_EXT = (".h", ".hpp", ".hh", ".hxx")
DEEPEN_LEVELS = (1, 10, 100)
FETCH_TIMEOUT = 300  # seconds: a hung remote must not stall the gate until CI-level timeout


def run(cmd_list, cwd=None, timeout=None):
    """Run a git command and return CompletedProcess (UTF-8, replace, rc-124 timeout).

    Git always writes UTF-8; decoding with the platform locale would crash on
    non-ASCII paths (core.quotepath=false makes them reach this code) and on
    non-UTF-8 source content in diff output, so decode as UTF-8 and replace
    undecodable bytes instead of raising. A timeout is converted into a
    non-zero return code (124) so callers treat it as a failed command.
    Known limitation: on Windows the wall clock can exceed the timeout when a
    grandchild process (git http helper) holds the output pipes; the command
    is still killed and rc=124 is still returned, just later.
    """
    return _run_decoded(cmd_list, cwd, timeout, "utf-8")


def run_tool(cmd_list, cwd=None, timeout=None):
    """Run a non-git tool (clang-tidy) and return CompletedProcess (locale, replace).

    clang-tidy writes its diagnostics in the console code page of the host
    (cp936 on zh-CN Windows, UTF-8 on Linux), not necessarily UTF-8: decoding
    it as UTF-8 would turn non-ASCII source paths into U+FFFD, the path
    comparison would mismatch and the file's violations would be silently
    dropped. Decode with the platform locale and replace undecodable bytes.
    """
    return _run_decoded(cmd_list, cwd, timeout, None)


def _run_decoded(cmd_list, cwd, timeout, encoding):
    try:
        return subprocess.run(cmd_list, cwd=cwd, capture_output=True, text=True,
                              encoding=encoding, errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd_list, 124, "", "timeout after %ss" % timeout)


def is_shallow_repo(root):
    """Return True when the repository has a shallow boundary (CI depth=1 checkout)."""
    r = run(["git", "rev-parse", "--is-shallow-repository"], cwd=root)
    return r.returncode == 0 and r.stdout.strip().lower() == "true"


def _first_available_remote(root, preferred):
    """Return the first existing remote among preferred/upstream/origin, or ''."""
    candidates = [preferred] + [r for r in ("upstream", "origin") if r != preferred]
    for remote in candidates:
        if remote and run(["git", "remote", "get-url", remote], cwd=root).returncode == 0:
            return remote
    return ""


def _target_branch_of(base_ref, remote):
    """Best-effort target branch name for fetch refspecs (master unless told).

    Accepts the plain branch name in TARGET_BRANCH (the CI form, which may
    itself contain slashes, e.g. release/v1.0), strips a refs/heads/ prefix
    or a known remote prefix (defensive: some pipelines qualify the value as
    origin/master), and derives from a remote-qualified base_ref by keeping
    everything after the remote segment. Branch names that merely contain
    slashes are never truncated.
    """
    tb = os.environ.get("TARGET_BRANCH", "")
    if not tb:
        if "/" in base_ref:
            first, _, rest = base_ref.partition("/")
            # remote-qualified ref (upstream/release/v1.0, or the
            # gitcode-upstream/<tb> fallback from resolve_base_ref): the
            # branch is the remainder; anything else (a local branch like
            # feature/x) is used verbatim - only a real remote prefix is
            # stripped
            tb = rest if first in (remote, "origin", "upstream", "gitcode-upstream") else base_ref
        else:
            # a bare sha or short ref carries no branch name: fall back
            tb = ""
    if tb.startswith("refs/heads/"):
        tb = tb[len("refs/heads/"):]
    for owner in (remote, "origin", "upstream", "gitcode-upstream"):
        if owner and tb.startswith(owner + "/"):
            tb = tb[len(owner) + 1:]
            break
    return tb or "master"


def ensure_merge_base(base_ref, root):
    """Return the merge-base sha of base_ref...HEAD, healing shallow checkouts.

    git diff A...B needs merge-base(A, B). In a --depth=1 CI checkout of the PR
    merge ref, HEAD's parents sit behind the shallow boundary, so the merge
    base is unreachable and every three-dot diff fails with empty output (which
    used to be mistaken for "no changed files"). Deepen the history from the
    same remote the base ref would use, 1/10/100 levels at a time, re-probing
    after each step; each deepen fetch carries an explicit refspec for the
    target branch (the shallow boundaries themselves travel as protocol
    shallow-lines, so the healing does not depend on the refspec covering the
    merge ref). Returns the sha, or '' when it stays unreachable.
    """
    probe = run(["git", "merge-base", base_ref, "HEAD"], cwd=root)
    if probe.returncode == 0 and probe.stdout.strip():
        return probe.stdout.strip()
    if not is_shallow_repo(root):
        return ""
    preferred = base_ref.split("/", 1)[0] if "/" in base_ref else "origin"
    remote = _first_available_remote(root, preferred)
    if not remote:
        return ""
    tb = _target_branch_of(base_ref, remote)
    deepen_refspec = "+refs/heads/%s:refs/remotes/%s/%s" % (tb, remote, tb)
    for level in DEEPEN_LEVELS:
        fr = run(["git", "fetch", "--quiet", "--no-tags", "--deepen=%d" % level, remote, deepen_refspec],
                 cwd=root, timeout=FETCH_TIMEOUT)
        probe = run(["git", "merge-base", base_ref, "HEAD"], cwd=root)
        if probe.returncode == 0 and probe.stdout.strip():
            log.info("[clang-tidy] shallow checkout: deepened %d level(s) from '%s' to restore the merge base", level, remote)
            return probe.stdout.strip()
        if fr.returncode != 0:
            log.warning("[clang-tidy] deepen fetch from '%s' failed (exit %d): %s", remote, fr.returncode, (fr.stderr or "").strip()[:200])
            # A failing deepen fetch is NOT proof of an unreachable remote: a
            # wrong target-branch name makes the refspec dangle ("couldn't
            # find remote ref") while the remote is fine. Fall through to the
            # refspec-less unshallow below - its expansion follows the client's
            # shallow boundary and needs no refspec, so it heals exactly that
            # case; a truly unreachable remote fails it fast (connection
            # refused) or within the fetch timeout.
            break
    # Last resort when the deepen ladder ran out of depth (boundary chains
    # longer than 1+10+100 levels, a merge ref whose pr-side ancestors the
    # ladder cannot deliver, or a dangling target-branch refspec): a full
    # unshallow WITHOUT a refspec, on purpose - the server-side expansion
    # follows the client's shallow boundary rather than any refspec wants,
    # so it recovers even when the derived target branch does not exist on
    # the remote (verified against the deleted-server-branch form, where it
    # still restores the pr-head parents). It only runs after the whole
    # ladder failed, keeping the full-history cost confined to the
    # pathological case.
    ur = run(["git", "fetch", "--quiet", "--no-tags", "--unshallow", remote], cwd=root, timeout=FETCH_TIMEOUT)
    if ur.returncode == 0:
        probe = run(["git", "merge-base", base_ref, "HEAD"], cwd=root)
        if probe.returncode == 0 and probe.stdout.strip():
            log.info("[clang-tidy] shallow checkout: unshallowed from '%s' to restore the merge base", remote)
            return probe.stdout.strip()
    else:
        log.warning("[clang-tidy] unshallow fetch from '%s' failed (exit %d): %s", remote, ur.returncode, (ur.stderr or "").strip()[:200])
    return ""


def get_repo_root():
    """Return the git repository root, or '' if not inside one."""
    r = run(["git", "rev-parse", "--show-toplevel"])
    return r.stdout.strip() if r.returncode == 0 else ""


def get_changed_files(base_ref, root):
    """Return changed C/C++ files (paths relative to the repo root) vs base_ref,
    or None when the diff itself fails (an empty list means "no changes", a
    failed diff means interception is incomplete and the caller must block).

    Both the git pathspec and the existence check are resolved against the
    repo root, so the result does not depend on the caller's cwd (build.sh
    invokes this script from the build directory). --diff-filter includes
    renames (R) so a renamed-and-modified file is still intercepted.
    """
    # -c core.quotepath=false: with the default true, git quotes non-ASCII
    # paths ("src/\347...\262.cc"), which the name-only parser cannot match
    # back to files - the change would silently vanish from the interception.
    cmd = ["git", "-c", "core.quotepath=false", "diff", "--name-only", "--diff-filter=ACMR", base_ref + "...HEAD", "--"]
    cmd += ["*" + e for e in CPP_EXT]
    r = run(cmd, cwd=root)
    if r.returncode != 0:
        log.warning("[clang-tidy] git diff --name-only failed (exit %d): %s", r.returncode, (r.stderr or "").strip()[:200])
        return None
    files = []
    for line in r.stdout.splitlines():
        rel = line.strip()
        if rel and os.path.isfile(os.path.join(root, rel)):
            files.append(rel)
    return files


def get_added_line_ranges(base_ref, f, root):
    """Return the set of added line numbers (1-indexed) via git diff -U0 hunk
    parsing, or None when the diff itself fails (callers treat that as an
    analysis failure, never as "no added lines")."""
    # -c core.quotepath=false: see get_changed_files
    r = run(["git", "-c", "core.quotepath=false", "diff", "-U0", base_ref + "...HEAD", "--", f], cwd=root)
    if r.returncode != 0:
        log.warning("[clang-tidy] git diff -U0 failed on %s (exit %d): %s", f, r.returncode, (r.stderr or "").strip()[:200])
        return None
    added = set()
    new_start = None
    for line in r.stdout.splitlines():
        m = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", line)
        if m:
            new_start = int(m.group(1))
            continue
        if line.startswith("+") and new_start is not None:
            added.add(new_start)
            new_start += 1
    return added


def norm_path(p):
    """Absolute normalized path, for reliable comparison of warning files."""
    return os.path.normpath(os.path.abspath(p))


def load_compile_db(build_dir):
    """Return the set of normalized absolute paths covered by compile_commands.json.

    Returns None if the file is missing or unreadable; the empty set if it
    exists but lists no files. main() blocks when the file exists but cannot
    be parsed (an unreadable database would silently empty the interception),
    and skips only when the file is missing (the gate cannot even start).
    """
    path = os.path.join(build_dir, "compile_commands.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fp:
            entries = json.load(fp)
    except (OSError, ValueError):
        log.warning("[clang-tidy] WARNING: %s unreadable", path)
        return None
    files = set()
    for entry in entries:
        f = entry.get("file", "")
        if f:
            if not os.path.isabs(f):
                f = os.path.join(build_dir, f)
            files.add(norm_path(f))
    return files


def run_clang_tidy(f, build_dir, config_file):
    """Run clang-tidy on f with the compile database; returns (output, returncode)."""
    r = run_tool(["clang-tidy", "--config-file=" + config_file, "-p", build_dir, f])
    return r.stdout + r.stderr, r.returncode


def run_clang_tidy_standalone(f, config_file):
    """Run clang-tidy on a header that has no compile database entry."""
    r = run_tool(["clang-tidy", "--config-file=" + config_file, f, "--", "-x", "c++"])
    return r.stdout + r.stderr, r.returncode


def parse_warnings(output):
    """Yield (file, line, check) for target-check warnings only.

    The file field is kept so callers can filter out warnings reported for
    headers included by the checked file.
    """
    for line in output.splitlines():
        m = WARN_RE.match(line)
        if m and m.group("check") in TARGET_CHECKS:
            yield m.group("file"), int(m.group("line")), m.group("check")


def default_base_ref():
    """Default diff base: the target branch of the gitcode upstream remote.

    TARGET_BRANCH (set by CI) only selects the branch name; the remote is
    always upstream first (the gitcode canonical repo, e.g. cann/hcomm), so a
    stale local branch can never become the base. resolve_base_ref refreshes
    the remote-tracking ref right before use.
    """
    tb = os.environ.get("TARGET_BRANCH", "")
    return ("upstream/" + tb) if tb else "upstream/master"


def detect_repo_basename():
    """Infer the repository basename from any gitcode remote URL.

    Works for the canonical repo (cann/hccl) and any fork (someone/hccl), for
    https and ssh URL forms. Returns '' if no gitcode remote is configured.
    """
    for remote in ("upstream", "origin"):
        r = run(["git", "remote", "get-url", remote])
        if r.returncode != 0:
            continue
        url = r.stdout.strip().rstrip("/")
        m = re.match(r".*(?:gitcode\.com[/:])[^/]+/([^/]+?)(?:\.git)?$", url)
        if m and m.group(1):
            return m.group(1)
    return ""


def fetch_canonical_target(tb):
    """Fetch the target branch straight from the canonical gitcode repository.

    Last-resort fallback for developers whose remotes are named differently
    or point elsewhere: derive the repo name from any gitcode remote and
    fetch cann/<repo> directly, so the diff base is always the true gitcode
    upstream target branch. Returns the ref name or ''.
    """
    basename = detect_repo_basename()
    if not basename:
        return ""
    url = "https://gitcode.com/cann/" + basename + ".git"
    ref = "refs/remotes/gitcode-upstream/" + tb
    if run(["git", "fetch", "--quiet", "--no-tags", url, "+refs/heads/" + tb + ":" + ref], timeout=FETCH_TIMEOUT).returncode == 0:
        candidate = "gitcode-upstream/" + tb
        if run(["git", "rev-parse", "--verify", candidate]).returncode == 0:
            return candidate
    return ""


def resolve_base_ref(base_ref):
    """Resolve the diff base: refresh the ref if it exists, else fetch the target branch.

    A locally-present remote-tracking ref is refreshed before use so a stale
    upstream/master cannot make the diff base outdated (which would count
    upstream's lines as new and false-block). If the refresh fails (offline
    development), the local ref is still used with a warning. The fallback
    tries the remote named by the default base ref (upstream) first, then
    origin (the only remote in CI clones of cann/<repo>), and finally the
    canonical gitcode repository fetched directly (so the base is correct
    even when the developer's remotes are named differently). The branch
    name is taken from TARGET_BRANCH or derived from the default base ref.
    """
    if run(["git", "rev-parse", "--verify", base_ref]).returncode == 0:
        if "/" in base_ref:
            remote, tb = base_ref.split("/", 1)
            if run(["git", "remote", "get-url", remote]).returncode == 0:
                fr = run(["git", "fetch", "--quiet", "--no-tags", remote, "+refs/heads/" + tb + ":refs/remotes/" + remote + "/" + tb], timeout=FETCH_TIMEOUT)
                if fr.returncode != 0:
                    log.warning("[clang-tidy] WARNING: failed to refresh %s, using local (possibly stale) ref", base_ref)
                if run(["git", "rev-parse", "--verify", base_ref]).returncode == 0:
                    return base_ref
            else:
                # first segment is not a known remote (e.g. a local branch
                # named feature/x): honor the explicitly passed ref as-is
                return base_ref
        else:
            return base_ref
    tb = _target_branch_of(base_ref, "upstream")
    for remote in ("upstream", "origin"):
        if run(["git", "remote", "get-url", remote]).returncode != 0:
            continue
        run(["git", "fetch", "--quiet", "--no-tags", remote, "+refs/heads/" + tb + ":refs/remotes/" + remote + "/" + tb], timeout=FETCH_TIMEOUT)
        candidate = remote + "/" + tb
        if run(["git", "rev-parse", "--verify", candidate]).returncode == 0:
            return candidate
    return fetch_canonical_target(tb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default=os.environ.get("CLANG_TIDY_BUILD_DIR", "build"))
    ap.add_argument("--base-ref", default=default_base_ref())
    ap.add_argument("--config-file", default=".clang-tidy")
    args = ap.parse_args()

    root = get_repo_root()
    if not root:
        log.warning("[clang-tidy] WARNING: not in a git repository, skip (not block)")
        return 0
    os.chdir(root)
    if not os.path.isabs(args.build_dir):
        args.build_dir = os.path.join(root, args.build_dir)
    if not os.path.isabs(args.config_file):
        args.config_file = os.path.join(root, args.config_file)

    if not shutil.which("clang-tidy"):
        log.warning("[clang-tidy] WARNING: clang-tidy not installed, skip (not block)")
        log.warning("[clang-tidy] hint: yum install -y clang-tools-extra")
        return 0
    if not os.path.isfile(os.path.join(args.build_dir, "compile_commands.json")):
        log.warning("[clang-tidy] WARNING: %s/compile_commands.json not found, skip (build first)", args.build_dir)
        return 0
    base = resolve_base_ref(args.base_ref)
    if not base:
        # Layering: a gate that cannot even start (no clang-tidy, no compile
        # database, no resolvable base ref - typically an offline developer
        # box) skips with a warning; a gate that starts but cannot compute its
        # diff blocks. CI always sets TARGET_BRANCH, so an unresolvable base
        # there is a broken pipeline, not an offline box: block it.
        if os.environ.get("TARGET_BRANCH"):
            log.warning("[clang-tidy] FAIL: cannot resolve base ref %s in CI (TARGET_BRANCH=%s), interception incomplete", args.base_ref, os.environ["TARGET_BRANCH"])
            return 1
        log.warning("[clang-tidy] WARNING: cannot resolve base ref %s, skip (not a gitcode CI run)", args.base_ref)
        return 0
    if not ensure_merge_base(base, root):
        hint_remote = _first_available_remote(root, base.split("/", 1)[0] if "/" in base else "origin") or "origin"
        if os.environ.get("TARGET_BRANCH"):
            # In CI an unreachable merge base means the pipeline itself is
            # broken (shallow healing failed): block loudly.
            log.warning("[clang-tidy] FAIL: merge base of %s...HEAD unreachable after shallow healing in CI, interception incomplete", base)
            log.warning("[clang-tidy] hint: git fetch --unshallow %s, or set ENABLE_CLANG_TIDY_CHECK=off to skip this gate", hint_remote)
            return 1
        # A local shallow clone that cannot heal (typically offline) is in the
        # same layer as "the gate cannot even start": warn and skip.
        log.warning("[clang-tidy] WARNING: merge base of %s...HEAD unreachable (shallow clone, healing failed), skip (not a gitcode CI run)", base)
        hint_tb = _target_branch_of(base, hint_remote)
        log.warning("[clang-tidy] hint: git fetch --deepen=1 %s +refs/heads/%s:refs/remotes/%s/%s, or git fetch --unshallow %s",
                    hint_remote, hint_tb, hint_remote, hint_tb, hint_remote)
        return 0

    files = get_changed_files(base, root)
    if files is None:
        log.warning("[clang-tidy] FAIL: incremental diff failed, interception incomplete")
        log.warning("[clang-tidy] hint: set ENABLE_CLANG_TIDY_CHECK=off to skip this gate")
        return 1
    if not files:
        log.info("[clang-tidy] incremental diff computed: no changed C++ files, skip")
        return 0
    compile_db = load_compile_db(args.build_dir)
    if compile_db is None:
        # The build directory was checked for the file earlier, so reaching
        # here means it exists but cannot be parsed: the interception would
        # be silently empty. Block.
        log.warning("[clang-tidy] FAIL: %s/compile_commands.json unreadable, interception incomplete", args.build_dir)
        return 1
    log.info("[clang-tidy] incremental check: base=%s, %d changed file(s)", base, len(files))

    violations = []
    diff_failed = []
    source_failed = []
    header_failed = 0
    for f in files:
        # Skip files with no added lines before invoking clang-tidy: only new
        # code can be intercepted, and pure-deletion files need no analysis.
        added = get_added_line_ranges(base, f, root)
        if added is None:
            # A per-file diff failure leaves the interception blind on this
            # file: block, never treat it as "no added lines".
            diff_failed.append(f)
            log.warning("[clang-tidy] WARNING: added-line diff failed on %s, diff failure", f)
            continue
        if not added:
            continue
        is_header = f.endswith(HEADER_EXT)
        if norm_path(f) not in compile_db:
            if is_header:
                # Headers have no compile command of their own: analyze standalone
                # so new complexity in headers is still intercepted.
                output, rc = run_clang_tidy_standalone(f, args.config_file)
                if rc != 0:
                    # Standalone headers frequently cannot parse without -I flags:
                    # warn only, never block.
                    header_failed += 1
                    log.warning("[clang-tidy] WARNING: standalone analysis failed on header %s (exit %d), warn-only", f, rc)
            else:
                # Generated/third-party sources are out of scope: exempt them
                # explicitly instead of failing their analysis.
                log.warning("[clang-tidy] WARNING: %s has no entry in compile_commands.json (generated/third-party?), skip", f)
                continue
        else:
            output, rc = run_clang_tidy(f, args.build_dir, args.config_file)
            if rc != 0:
                # A changed source that cannot be analyzed would be a silent gap in
                # interception: treat it as a blocking failure.
                source_failed.append(f)
                log.warning("[clang-tidy] WARNING: clang-tidy exited %d on %s, analysis failure", rc, f)
        for wfile, lno, check in parse_warnings(output):
            if norm_path(wfile) == norm_path(f) and lno in added:
                violations.append((f, lno, check))

    if violations:
        log.warning("[clang-tidy] FAIL: %d new-code complexity violation(s):", len(violations))
        for f, lno, check in violations:
            log.warning("  %s:%d [%s]", f, lno, check)
        return 1
    if diff_failed:
        log.warning("[clang-tidy] FAIL: added-line diff failed on %d file(s), interception incomplete:", len(diff_failed))
        for f in diff_failed:
            log.warning("  %s", f)
        return 1
    if source_failed:
        log.warning("[clang-tidy] FAIL: clang-tidy analysis failed on %d changed source file(s), interception incomplete:", len(source_failed))
        for f in source_failed:
            log.warning("  %s", f)
        return 1
    if header_failed:
        log.warning("[clang-tidy] WARNING: %d header file(s) failed standalone analysis (warn-only)", header_failed)
    log.info("[clang-tidy] PASS: no new-code complexity violation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
