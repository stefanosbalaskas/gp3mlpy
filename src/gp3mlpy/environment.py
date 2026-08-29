from __future__ import annotations

from collections.abc import Sequence
from importlib import metadata
from pathlib import Path
from hashlib import sha256
import platform
import subprocess
import sys
import numpy as np
import pandas as pd

from ._utils import worst_status
from .exceptions import GP3MLError
from .objects import GP3MLEnvironment, GP3MLEnvironmentComparison


def _pkg_version(name: str) -> str | None:
    if name=="gp3ml": return "0.3.0-reference"
    if name=="gp3mlpy":
        try: return metadata.version("gp3mlpy")
        except metadata.PackageNotFoundError: return "0.1.0.dev0"
    try: return metadata.version(name)
    except metadata.PackageNotFoundError: return None


def _git_sha(root: str|Path) -> str | None:
    try:
        out=subprocess.run(["git","-C",str(Path(root).resolve()),"rev-parse","HEAD"],capture_output=True,text=True,check=True).stdout.strip()
        return out or None
    except Exception: return None


def capture_gazepoint_environment(packages: Sequence[str]|str|None=None,root: str|Path=".",include_renv: bool=False) -> GP3MLEnvironment:
    if packages is None: packages=["gp3mlpy","numpy","pandas","scipy","scikit-learn","statsmodels","matplotlib"]
    if isinstance(packages,str): packages=[packages]
    versions={name:v for name in dict.fromkeys(packages) if (v:=_pkg_version(name)) is not None}
    lock_hash=None
    if include_renv:
        lock=Path(root)/"renv.lock"
        if lock.exists(): lock_hash=sha256(lock.read_bytes()).hexdigest()
    return GP3MLEnvironment(
        R_version=f"Python {sys.version.split()[0]}", R_platform=platform.platform(),
        OS=f"{platform.system()} {platform.release()} {platform.version()}",
        BLAS=str(np.__config__.CONFIG.get("Build Dependencies",{}).get("blas",{}).get("name","")) if hasattr(np.__config__,"CONFIG") else "",
        LAPACK="", RNGkind=["numpy", "RandomState"], repositories={}, package_versions=versions,
        gp3ml_git_sha=_git_sha(root), renv_lock_sha256=lock_hash,
    )


def compare_gazepoint_environments(reference: GP3MLEnvironment,current: GP3MLEnvironment) -> GP3MLEnvironmentComparison:
    if not isinstance(reference,GP3MLEnvironment) or not isinstance(current,GP3MLEnvironment): raise GP3MLError("Both objects must be gp3ml environment records.")
    pkgs=sorted(set(reference.package_versions)|set(current.package_versions)); rows=[]
    for pkg in pkgs:
        r=reference.package_versions.get(pkg); c=current.package_versions.get(pkg); status="review" if r is None or c is None else ("pass" if r==c else "review")
        rows.append({"package":pkg,"reference":r,"current":c,"status":status})
    packages=pd.DataFrame(rows,columns=["package","reference","current","status"])
    core_rows=[]
    for comp in ["R_version","R_platform","gp3ml_git_sha","renv_lock_sha256"]:
        r=getattr(reference,comp); c=getattr(current,comp)
        status="pass" if (r is None and c is None) or r==c else "review"
        core_rows.append({"component":comp,"reference":r,"current":c,"status":status})
    core=pd.DataFrame(core_rows)
    return GP3MLEnvironmentComparison(status=worst_status([*packages.status,*core.status]),packages=packages,core=core)


def validate_gazepoint_environment(reference: GP3MLEnvironment,root: str|Path=".",include_renv: bool=False) -> GP3MLEnvironmentComparison:
    current=capture_gazepoint_environment(packages=list(reference.package_versions),root=root,include_renv=include_renv)
    return compare_gazepoint_environments(reference,current)
