"""Run the same bounded smoke test from a temporary Modal controller Function.

Useful when the local network cannot reach regional sandbox command routers.
The controller receives the harness code only. It creates a synthetic repository,
runs scripts/smoke_modal_sandbox.py, and returns its artifacts to this host. No
persistent function deployment, model, GPU, or private repository is involved.
"""

from pathlib import Path
import sys

import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
app = modal.App("action-interview-sandbox-smoke")
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("git")
         .pip_install("modal==1.5.5")
         .add_local_dir(ROOT / "agent_harness", "/opt/harness/agent_harness", copy=True)
         .add_local_file(ROOT / "scripts/smoke_modal_sandbox.py", "/opt/harness/scripts/smoke_modal_sandbox.py", copy=True))


@app.function(image=image, timeout=600, cpu=0.25, memory=512)
def run_smoke():
    import runpy
    from pathlib import Path
    runpy.run_path("/opt/harness/scripts/smoke_modal_sandbox.py", run_name="__main__")
    root = Path("artifacts/modal-smoke")
    return {str(path.relative_to(root)): path.read_text() for path in root.rglob("*") if path.is_file()}


if __name__ == "__main__":
    with modal.enable_output(), app.run():
        artifacts = run_smoke.remote()
    destination = ROOT / "artifacts/modal-smoke-remote"
    for name, contents in artifacts.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Invalid returned artifact path")
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x") as stream:
            stream.write(contents)
    print(f"Remote smoke passed; artifacts: {destination}")
