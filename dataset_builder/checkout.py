"""Read-only pinned checkout helper shared by package and preparation CLI."""
import subprocess

def git(path, *args):
    return subprocess.check_output(
        ["git", "-C", str(path), *args], text=True, stderr=subprocess.PIPE,
        timeout=180).strip()


def checkout(row, directory):
    """Fetch source without installing dependencies or executing repository code."""
    path = directory / (row["repo"].replace("/", "--") + "-" + row["commit_id"])
    url = "https://github.com/" + row["repo"] + ".git"
    path.mkdir(parents=True, exist_ok=True)
    if not (path / ".git").exists():
        if any(path.iterdir()):
            raise ValueError(f"Checkout directory is not empty: {path}")
        git(path, "init")
        git(path, "remote", "add", "origin", url)
    if git(path, "remote", "get-url", "origin") != url:
        raise ValueError(f"Unexpected checkout remote: {path}")
    if git(path, "status", "--porcelain"):
        raise ValueError(f"Checkout has local changes: {path}")
    try:
        head = git(path, "rev-parse", "HEAD")
    except subprocess.CalledProcessError:
        head = None
    if head != row["commit_id"]:
        git(path, "fetch", "--depth=1", "origin", row["commit_id"])
        git(path, "-c", "core.hooksPath=/dev/null", "checkout", "--detach", row["commit_id"])
    if git(path, "rev-parse", "HEAD") != row["commit_id"]:
        raise ValueError("Checkout revision mismatch")
    files = git(path, "ls-files").splitlines()
    if not files:
        raise ValueError("Empty repository checkout")
    return {"repo": row["repo"], "commit_id": row["commit_id"],
            "path": str(path.resolve()), "tracked_files": len(files)}

