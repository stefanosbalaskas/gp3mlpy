from __future__ import annotations
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
import pandas as pd
from ._utils import worst_status
from .exceptions import GP3MLError
from .objects import GP3MLReleaseChecksumManifest, GP3MLReleaseChecksumValidation


def _sha(path: Path)->str: return sha256(path.read_bytes()).hexdigest()

def write_gazepoint_release_checksums(files: Sequence[str|Path]|str|Path,path: str|Path="SHA256SUMS.csv") -> GP3MLReleaseChecksumManifest:
    if isinstance(files,(str,Path)): files=[files]
    values=[Path(x) for x in files]
    if any(not x.exists() for x in values): raise GP3MLError("Every release artifact must exist.")
    tab=pd.DataFrame([{"file":x.name,"sha256":_sha(x),"size":x.stat().st_size} for x in values])
    target=Path(path); target.parent.mkdir(parents=True,exist_ok=True); tab.to_csv(target,index=False)
    return GP3MLReleaseChecksumManifest(path=target.resolve().as_posix(),checksums=tab)

def validate_gazepoint_release_checksums(manifest,directory: str|Path=".") -> GP3MLReleaseChecksumValidation:
    if isinstance(manifest,GP3MLReleaseChecksumManifest): tab=manifest.checksums.copy()
    else: tab=pd.read_csv(manifest)
    root=Path(directory); exists=[]; actual=[]; status=[]
    for _,row in tab.iterrows():
        target=root/str(row.file); ok=target.exists(); digest=_sha(target) if ok else None
        exists.append(ok); actual.append(digest); status.append("pass" if ok and str(row.sha256)==digest else "fail")
    tab["exists"]=exists; tab["actual_sha256"]=actual; tab["status"]=status
    return GP3MLReleaseChecksumValidation(status=worst_status(status),files=tab)
