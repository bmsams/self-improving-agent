"""GitHub integration for remote PR management.

Provides real GitHub PR creation, review fetching, and merging
when the --github flag is enabled. Falls back to local simulation
when disabled.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class GitHubConfig:
    """Configuration for GitHub integration."""

    enabled: bool = False
    token: str = ""
    repo: str = ""  # format: owner/repo
    auto_mode: bool = False  # auto-merge without waiting for reviews
    review_timeout_hours: int = 24

    @classmethod
    def from_env(cls) -> GitHubConfig:
        """Create config from environment variables."""
        return cls(
            enabled=os.environ.get("GITHUB_ENABLED", "").lower() in ("true", "1"),
            token=os.environ.get("GITHUB_TOKEN", ""),
            repo=os.environ.get("GITHUB_REPO", ""),
            auto_mode=os.environ.get("GITHUB_AUTO", "").lower() in ("true", "1"),
        )


class GitHubIntegration:
    """Manages remote GitHub operations for the self-improving agent.

    Uses the `gh` CLI for GitHub API operations (PRs, reviews, merging).
    Requires GITHUB_TOKEN or gh auth to be configured.
    """

    def __init__(self, config: GitHubConfig, project_root: Path):
        self.config = config
        self.project_root = project_root

    def push_branch(self, branch_name: str) -> bool:
        """Push a branch to the remote repository.

        Returns True if push succeeded.
        """
        try:
            result = subprocess.run(
                ["git", "push", "-u", "origin", branch_name],
                cwd=self.project_root, capture_output=True, text=True,
                check=False, timeout=60,
            )
            if result.returncode != 0:
                logger.error(f"Push failed: {result.stderr}")
                return False
            logger.info(f"Pushed branch: {branch_name}")
            return True
        except subprocess.TimeoutExpired:
            logger.error(f"Push timed out for branch: {branch_name}")
            return False

    def create_pr(
        self, title: str, body: str, head: str, base: str = "main",
    ) -> Optional[int]:
        """Create a GitHub PR and return the PR number.

        Returns None if creation failed.
        """
        try:
            result = subprocess.run(
                ["gh", "pr", "create",
                 "--title", title,
                 "--body", body,
                 "--head", head,
                 "--base", base],
                cwd=self.project_root, capture_output=True, text=True,
                check=False, timeout=30,
            )
            if result.returncode != 0:
                logger.error(f"PR creation failed: {result.stderr}")
                return None
            # gh pr create outputs the URL, extract PR number
            url = result.stdout.strip()
            pr_number = int(url.rstrip("/").split("/")[-1])
            logger.info(f"Created PR #{pr_number}: {title}")
            return pr_number
        except (subprocess.TimeoutExpired, ValueError, FileNotFoundError) as e:
            logger.error(f"Failed to create PR: {e}")
            return None

    def get_reviews(self, pr_number: int) -> list[dict]:
        """Fetch reviews for a PR.

        Returns list of review dicts with 'state', 'user', 'body'.
        """
        try:
            result = subprocess.run(
                ["gh", "pr", "view", str(pr_number),
                 "--json", "reviews"],
                cwd=self.project_root, capture_output=True, text=True,
                check=False, timeout=30,
            )
            if result.returncode != 0:
                logger.error(f"Failed to fetch reviews: {result.stderr}")
                return []
            import json
            data = json.loads(result.stdout)
            return data.get("reviews", [])
        except Exception as e:
            logger.error(f"Failed to fetch reviews for PR #{pr_number}: {e}")
            return []

    def merge_pr(self, pr_number: int, method: str = "squash") -> bool:
        """Merge a PR via GitHub API.

        Returns True if merge succeeded.
        """
        try:
            result = subprocess.run(
                ["gh", "pr", "merge", str(pr_number),
                 f"--{method}", "--delete-branch"],
                cwd=self.project_root, capture_output=True, text=True,
                check=False, timeout=30,
            )
            if result.returncode != 0:
                logger.error(f"Merge failed: {result.stderr}")
                return False
            logger.info(f"Merged PR #{pr_number}")
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.error(f"Merge failed for PR #{pr_number}: {e}")
            return False

    def close_pr(self, pr_number: int) -> bool:
        """Close a PR without merging.

        Returns True if close succeeded.
        """
        try:
            result = subprocess.run(
                ["gh", "pr", "close", str(pr_number),
                 "--delete-branch"],
                cwd=self.project_root, capture_output=True, text=True,
                check=False, timeout=30,
            )
            if result.returncode != 0:
                logger.error(f"Close failed: {result.stderr}")
                return False
            logger.info(f"Closed PR #{pr_number}")
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.error(f"Close failed for PR #{pr_number}: {e}")
            return False

    def is_pr_approved(self, pr_number: int) -> bool:
        """Check if a PR has at least one approving review."""
        reviews = self.get_reviews(pr_number)
        return any(r.get("state") == "APPROVED" for r in reviews)
