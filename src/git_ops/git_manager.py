"""Git operations for self-improvement PRs and branch management."""

from __future__ import annotations

import subprocess
import uuid
from datetime import datetime, timezone, timezone
from pathlib import Path
from typing import Optional

from src.core.models import PRStatus, PullRequest


class GitOps:
    """Manages git operations for the self-improving agent.

    Handles branching, committing, PR simulation, merging, and rollback.
    Operates on a local git repository — can optionally push to remote.
    """

    def __init__(self, repo_path: Path, remote: Optional[str] = None):
        self.repo_path = repo_path
        self.remote = remote
        self._ensure_repo()

    def _run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        """Execute a git command in the repo directory."""
        result = subprocess.run(
            ["git", *args],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=False,
        )
        if check and result.returncode != 0:
            raise GitError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
        return result

    def _ensure_repo(self) -> None:
        """Initialize git repo if it doesn't exist."""
        git_dir = self.repo_path / ".git"
        if not git_dir.exists():
            self._run("init", "-b", "main")
            self._run("config", "user.email", "agent@self-improving.ai")
            self._run("config", "user.name", "Self-Improving Agent")
            # Create initial commit so we have a main branch
            readme = self.repo_path / "README.md"
            if not readme.exists():
                readme.write_text("# Self-Improving Agent\n\nThis repo evolves itself.\n")
            self._run("add", "-A")
            self._run("commit", "-m", "[agent] init: bootstrap repository")

    @property
    def current_branch(self) -> str:
        """Get current branch name."""
        result = self._run("branch", "--show-current")
        return result.stdout.strip()

    @property
    def current_sha(self) -> str:
        """Get current commit SHA."""
        result = self._run("rev-parse", "HEAD")
        return result.stdout.strip()[:12]

    def get_diff(self, base: str = "main") -> str:
        """Get diff between current branch and base."""
        result = self._run("diff", base, "--stat", check=False)
        stat = result.stdout.strip()
        result = self._run("diff", base, check=False)
        full_diff = result.stdout.strip()
        return f"## Diff Summary\n{stat}\n\n## Full Diff\n{full_diff}"

    def get_changed_files(self, base: str = "main") -> list[str]:
        """List files changed relative to base branch."""
        result = self._run("diff", "--name-only", base, check=False)
        return [f for f in result.stdout.strip().split("\n") if f]

    def create_improvement_branch(self, description: str) -> str:
        """Create a new branch for a self-improvement."""
        slug = description.lower().replace(" ", "-")[:40]
        branch = f"improve/{slug}-{uuid.uuid4().hex[:6]}"
        self._run("checkout", "-b", branch)
        return branch

    def commit_changes(self, message: str, files: Optional[list[str]] = None) -> str:
        """Stage and commit changes. Returns commit SHA."""
        if files:
            for f in files:
                self._run("add", f)
        else:
            self._run("add", "-A")
        self._run("commit", "-m", message, check=False)
        return self.current_sha

    def create_pr(self, title: str, description: str) -> PullRequest:
        """Create a simulated pull request (local-only)."""
        pr_id = f"PR-{uuid.uuid4().hex[:8]}"
        branch = self.current_branch
        files = self.get_changed_files()

        pr = PullRequest(
            pr_id=pr_id,
            branch_name=branch,
            title=title,
            description=description,
            status=PRStatus.OPEN,
            files_changed=files,
        )
        return pr

    def merge_pr(self, pr: PullRequest) -> None:
        """Merge a PR branch into main."""
        self._run("stash", "--include-untracked", check=False)
        self._run("checkout", "main")
        self._run("merge", "--no-ff", pr.branch_name,
                   "-m", f"[agent] merge: {pr.title} ({pr.pr_id})")
        pr.status = PRStatus.MERGED
        pr.merged_at = datetime.now(timezone.utc).isoformat()
        # Clean up branch
        self._run("branch", "-d", pr.branch_name, check=False)
        self._run("stash", "pop", check=False)

    def reject_pr(self, pr: PullRequest) -> None:
        """Reject a PR and return to main."""
        pr.status = PRStatus.REJECTED
        self._run("stash", "--include-untracked", check=False)
        self._run("checkout", "main")
        self._run("branch", "-D", pr.branch_name, check=False)
        self._run("stash", "drop", check=False)

    def rollback_to(self, sha: str) -> None:
        """Rollback main to a specific commit."""
        self._run("checkout", "main")
        self._run("reset", "--hard", sha)

    def switch_to_main(self) -> None:
        """Return to main branch."""
        self._run("stash", "--include-untracked", check=False)
        self._run("checkout", "main")
        self._run("stash", "pop", check=False)

    def get_log(self, count: int = 10) -> str:
        """Get recent commit log."""
        result = self._run(
            "log", f"--oneline", f"-{count}",
            "--format=%h %s (%ar)"
        )
        return result.stdout.strip()

    def tag_generation(self, generation: int, score: float) -> None:
        """Tag current commit with generation info."""
        tag = f"gen-{generation:04d}-score-{score:.2f}"
        self._run("tag", "-a", tag, "-m",
                   f"Generation {generation}, benchmark score: {score:.2f}",
                   check=False)


class GitError(Exception):
    """Raised when a git operation fails."""
    pass
