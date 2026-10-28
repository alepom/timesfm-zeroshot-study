"""Environment and model provenance recorded with every run."""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import platform
import subprocess
from pathlib import Path

PACKAGES = ("tfmstudy", "timesfm", "chronos-forecasting", "tirex-2", "torch", "numpy", "pandas", "statsforecast")


def package_versions() -> dict[str, str | None]:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def gpu_info() -> dict:
    info: dict = {}
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info.update(name=props.name, total_memory_gb=round(props.total_memory / 2**30, 2),
                        capability=f"{props.major}.{props.minor}", cuda=torch.version.cuda)
    except ImportError:
        info["cuda_available"] = None
    try:
        info["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def hf_revision(repo: str | None) -> str | None:
    """Current commit sha of a Hugging Face model repo."""
    if not repo:
        return None
    try:
        from huggingface_hub import model_info

        return model_info(repo).sha
    except Exception as exc:  # network or auth errors are recorded, not fatal
        return f"error: {exc}"


def code_hash() -> str:
    """sha256 over the tfmstudy source files, so each result can be tied to the exact code."""
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def environment() -> dict:
    from . import __version__

    return {
        "tfmstudy": {"version": __version__, "code_sha256": code_hash()},
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "gpu": gpu_info(),
    }
