import os
import subprocess
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def is_git_repo(path: str) -> bool:
    """Check if a directory is a git repository (accepts .git as file or directory)."""
    abs_path = os.path.abspath(path)
    git_dir = os.path.join(abs_path, '.git')
    is_repo = os.path.exists(git_dir)
    logger.debug(f"Checking if path {abs_path} is a git repo: {is_repo}")
    return is_repo

def get_git_hash(repo_path: str = None) -> Optional[str]:
    """Get the current git commit hash."""
    try:
        abs_repo_path = os.path.abspath(repo_path) if repo_path else os.path.dirname(__file__)
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=abs_repo_path,
            timeout=5,
        )
        logger.debug(f"Git hash for repo at {abs_repo_path}: {result.stdout.strip()}")
        return result.stdout.strip()
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ) as e:
        logger.warning(f"Could not determine git hash for repo at {repo_path}: {e}")
        return None
