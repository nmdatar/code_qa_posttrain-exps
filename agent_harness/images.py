"""Build reusable, pinned Modal environments without executing repository code locally."""
from __future__ import annotations

import hashlib
import importlib
import io
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

WORKSPACE_PATH = "/repo"
_RECIPE_KEYS = {"base_image", "base_image_id", "mode", "apt_packages", "dependency_files", "dependency_commands", "build_commands", "readiness_commands", "setup_commands"}


def _hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def _relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or ".git" in path.parts or "\\" in value or "\x00" in value or str(path) != value:
        raise ValueError(f"Unsafe repository-relative path: {value!r}")
    return path


def normalize_recipe(recipe: dict) -> dict:
    if not isinstance(recipe, dict) or set(recipe) - _RECIPE_KEYS:
        raise ValueError("Unknown environment recipe fields")
    result = dict(recipe)
    if result.get("mode") not in {"source_reading", "executable"}:
        raise ValueError("mode must be source_reading or executable")
    if ("base_image" in result) == ("base_image_id" in result):
        raise ValueError("Specify exactly one of base_image or base_image_id")
    if "base_image" in result:
        base = result["base_image"]
        if not isinstance(base, str) or not re.fullmatch(r"[^\s]+@sha256:[a-f0-9]{64}", base):
            raise ValueError("base_image must use an immutable @sha256 registry digest")
    else:
        base_id = result["base_image_id"]
        if not isinstance(base_id, str) or not re.fullmatch(r"im-[A-Za-z0-9]+", base_id):
            raise ValueError("base_image_id must be a concrete Modal image ID (im-...)")
    for key in ("apt_packages", "dependency_files", "dependency_commands", "build_commands"):
        values = result.setdefault(key, [])
        if not isinstance(values, list) or any(not isinstance(v, str) or not v.strip() or "\x00" in v for v in values):
            raise ValueError(f"{key} must be a list of nonempty strings")
    for path in result["dependency_files"]:
        _relative(path)
    for key in ("setup_commands", "readiness_commands"):
        commands = result.setdefault(key, [])
        if not isinstance(commands, list) or any(not isinstance(cmd, list) or not cmd or any(not isinstance(arg, str) or "\x00" in arg for arg in cmd) or not cmd[0] for cmd in commands):
            raise ValueError(f"{key} must contain argument lists")
    if not result["readiness_commands"]:
        raise ValueError("At least one readiness command is required")
    return json.loads(json.dumps(result))


def _identity(manifest: dict) -> dict:
    return {key: manifest[key] for key in ("schema_version", "image_id", "commit", "source_sha256", "recipe", "workspace_path")}


def validate_manifest(manifest: dict) -> None:
    """Validate an environment identity; this is an integrity check, not a signature."""
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or manifest.get("status") != "ready":
        raise ValueError("Expected a ready schema_version=1 environment manifest")
    if manifest.get("workspace_path") not in {WORKSPACE_PATH, "/workspace"}:
        raise ValueError("Unexpected environment workspace path")
    for key, pattern in (("commit", r"[a-f0-9]{40}|[a-f0-9]{64}"), ("source_sha256", r"[a-f0-9]{64}")):
        if not isinstance(manifest.get(key), str) or not re.fullmatch(pattern, manifest[key]):
            raise ValueError(f"Invalid manifest {key}")
    if normalize_recipe(manifest.get("recipe")) != manifest["recipe"]:
        raise ValueError("Manifest recipe is not normalized")
    for key in ("image_id", "app_name"):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise ValueError(f"Missing manifest {key}")
    if manifest.get("environment_id") != _hash(_identity(manifest)):
        raise ValueError("Environment manifest identity mismatch")
    checks = manifest.get("readiness")
    if not isinstance(checks, list) or len(checks) != len(manifest["recipe"]["readiness_commands"]):
        raise ValueError("Missing readiness results")
    for check, command in zip(checks, manifest["recipe"]["readiness_commands"]):
        if not isinstance(check, dict) or check.get("command") != command or check.get("exit_code") != 0:
            raise ValueError("Environment readiness did not pass")


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def _snapshot(repo: Path, commit: str, destination: Path) -> str:
    if not re.fullmatch(r"[a-f0-9]{40}|[a-f0-9]{64}", commit):
        raise ValueError("commit must be a full lowercase Git commit hash")
    if _git(repo, "rev-parse", f"{commit}^{{commit}}").decode().strip() != commit:
        raise ValueError("commit must identify a commit, not a tag")
    entries = _git(repo, "ls-tree", "-r", "-z", commit).split(b"\0")
    if any(entry.startswith(b"160000 ") for entry in entries):
        raise ValueError("Submodules are unsupported; vendor their pinned contents first")
    archive = _git(repo, "archive", "--format=tar", commit)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        members = tar.getmembers()
        # Extract files before symlinks so no archive path can redirect a write.
        for member in members:
            path = destination / _relative(member.name.rstrip("/"))
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                path.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, path.open("wb") as target:
                    shutil.copyfileobj(source, target)
                path.chmod(member.mode & 0o777)
            elif not member.issym():
                raise ValueError(f"Unsupported archive member: {member.name}")
        for member in members:
            if member.issym():
                path = destination / member.name
                path.parent.mkdir(parents=True, exist_ok=True)
                if PurePosixPath(member.linkname).is_absolute():
                    raise ValueError(f"Absolute symlink: {member.name}")
                path.symlink_to(member.linkname)
        for member in members:
            if member.issym():
                try:
                    (destination / member.name).resolve().relative_to(destination.resolve())
                except (ValueError, RuntimeError) as exc:
                    raise ValueError(f"Escaping or cyclic symlink: {member.name}") from exc
    return hashlib.sha256(archive).hexdigest()


def prepare_environment(repo: Path, commit: str, recipe: dict, app_name: str, *, modal_module=None) -> dict:
    """Build once and verify remotely; return a manifest only after readiness succeeds.

    The explicitly selected repository's committed snapshot is the only local upload.
    Keep private grading artifacts in a separate checkout. Build commands may use the
    network to install locked dependencies; readiness and rollouts block the network.
    """
    from .process import run_process

    recipe = normalize_recipe(recipe)
    if not isinstance(app_name, str) or not app_name.strip():
        raise ValueError("app_name is required")
    with tempfile.TemporaryDirectory(prefix="environment-image-") as staging:
        snapshot = Path(staging) / "source"
        snapshot.mkdir()
        source_sha256 = _snapshot(Path(repo).resolve(), commit, snapshot)
        dependencies = Path(staging) / "dependencies"
        dependencies.mkdir()
        for value in recipe["dependency_files"]:
            source = snapshot / value
            relative_parts = PurePosixPath(value).parts
            if not source.is_file() or any((snapshot.joinpath(*relative_parts[:i])).is_symlink() for i in range(1, len(relative_parts) + 1)):
                raise ValueError(f"Dependency must be a regular archived file: {value}")
            target = dependencies / value
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        modal = modal_module if modal_module is not None else importlib.import_module("modal")
        app = modal.App.lookup(app_name, create_if_missing=True)
        image = (modal.Image.from_registry(recipe["base_image"]) if "base_image" in recipe
                 else modal.Image.from_id(recipe["base_image_id"]))
        if recipe["apt_packages"]:
            image = image.apt_install(*recipe["apt_packages"])
        image = image.workdir(WORKSPACE_PATH)
        if recipe["dependency_files"]:
            image = image.add_local_dir(dependencies, WORKSPACE_PATH, copy=True)
        if recipe["dependency_commands"]:
            image = image.run_commands(*recipe["dependency_commands"])
        image = image.add_local_dir(snapshot, WORKSPACE_PATH, copy=True)
        if recipe["build_commands"]:
            image = image.run_commands(*recipe["build_commands"])
        image.build(app)
        sandbox = modal.Sandbox.create(
            app=app, image=image, workdir=WORKSPACE_PATH, block_network=True,
            timeout=300, cpu=(0.5, 2), memory=(1024, 2048),
            include_oidc_identity_token=False,
        )
        checks = []
        try:
            for command in recipe["setup_commands"] + recipe["readiness_commands"]:
                result = run_process(sandbox, command, timeout_seconds=120, max_output_bytes=65536, workdir=WORKSPACE_PATH)
                if result.exit_code != 0:
                    raise RuntimeError(f"Environment setup/readiness failed: {command!r}: {result.stderr}")
                checks.append({"command": command, "exit_code": result.exit_code, "stdout": result.stdout, "stderr": result.stderr})
        finally:
            try:
                try:
                    sandbox.terminate(wait=True)
                finally:
                    sandbox.detach()
            except Exception as exc:
                raise RuntimeError(
                    f"Readiness sandbox cleanup failed for {getattr(sandbox, 'object_id', 'unknown')}; "
                    "check its status and terminate it manually if still running"
                ) from exc
        manifest = {
            "schema_version": 1, "status": "ready", "image_id": image.object_id,
            "commit": commit, "source_sha256": source_sha256, "recipe": recipe,
            "app_name": app_name, "workspace_path": WORKSPACE_PATH,
            "readiness": checks[len(recipe["setup_commands"]):],
            "modal_sdk_version": getattr(modal, "__version__", None),
        }
        manifest["environment_id"] = _hash(_identity(manifest))
        validate_manifest(manifest)
        return manifest
