"""Live integration test: tiny pinned fixture, prepared image, two clean attempts.

Allocates paid Modal CPU resources. Requires the modal extra and configured Modal
credentials. No model, grader, trainer, private repository, or GPU is involved.
"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_harness.cli import save_new, smoke
from agent_harness.images import prepare_environment
from agent_harness.modal_backend import ModalSandboxBackend, SandboxLimits


def main():
    import modal
    run_id = uuid.uuid4().hex
    output = Path("artifacts/modal-smoke") / run_id
    output.mkdir(parents=True)
    print("Building/resolving a pinned Modal Python base", flush=True)
    app = modal.App.lookup("action-interview-environments", create_if_missing=True)
    base = modal.Image.debian_slim(python_version="3.11").build(app)
    recipe = {"base_image_id": base.object_id, "mode": "executable",
              "build_commands": ["python3 -m compileall -q /repo"],
              "setup_commands": [["python3", "-c", "from pathlib import Path; Path('/tmp/fixture-ready').write_text('ready')"]],
              "readiness_commands": [["python3", "-c", "import example; from pathlib import Path; assert example.add(2,3)==5; assert Path('/tmp/fixture-ready').read_text()=='ready'"]]}
    save_new(output / "recipe.json", recipe)
    with tempfile.TemporaryDirectory(prefix="modal-fixture-") as directory:
        repo = Path(directory)
        (repo / "example.py").write_text("def add(a, b):\n    return a + b\n")
        def git(*args):
            return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()
        git("init", "-q")
        git("add", "example.py")
        git("-c", "user.name=Sandbox fixture", "-c", "user.email=fixture@example.invalid",
            "-c", "commit.gpgsign=false", "commit", "-qm", "Pinned sandbox fixture")
        print("Preparing fixture image and readiness sandbox", flush=True)
        manifest = prepare_environment(repo, git("rev-parse", "HEAD"), recipe, "action-interview-environments")
    save_new(output / "environment.json", manifest)
    backend = ModalSandboxBackend(manifest, output / "events", limits=SandboxLimits(lifetime_seconds=180))
    print("Checking two fresh rollout sandboxes", flush=True)
    report = smoke(backend, f"live-{run_id}")
    save_new(output / "report.json", report)
    print(json.dumps({"status": report["status"], "report": str(output / "report.json"),
                      "image_id": manifest["image_id"], "base_image_id": recipe["base_image_id"]}), flush=True)


if __name__ == "__main__":
    main()
