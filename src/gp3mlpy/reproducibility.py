from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
import os
import re
from typing import TypeVar
import pandas as pd

from ._utils import write_tables
from .exceptions import GP3MLError
from .objects import GP3MLReproducibilityAudit

T=TypeVar("T")
_PATTERNS={
    "r_temp_directory":re.compile(r"Rtmp[A-Za-z0-9]+"),
    "windows_temp_path":re.compile(r"[A-Z]:[/\\][^\r\n]*AppData[/\\]Local[/\\]Temp[^\r\n <>\'\"]*",re.I),
    "unix_temp_path":re.compile(r"/tmp/Rtmp[^\r\n <>\'\"]*"),
    "memory_address":re.compile(r"0x[0-9A-Fa-f]{6,}"),
    "generated_timestamp":re.compile(r"Generated:\s+[0-9]{4}-[0-9]{2}-[0-9]{2}[ T][0-9]{2}:[0-9]{2}:[0-9]{2}",re.I),
}
_REPL={"r_temp_directory":"<RTMP>","windows_temp_path":"<TEMP_PATH>","unix_temp_path":"<TEMP_PATH>","memory_address":"<ADDRESS>","generated_timestamp":"Generated: <timestamp>"}


def normalize_gazepoint_artifact_text(x: Sequence[str]|str,project_path: str|Path|None=None):
    scalar=isinstance(x,str); values=[x] if scalar else [str(v) for v in x]
    project=None if project_path is None else str(Path(project_path).resolve()).replace("\\","/")
    out=[]
    for line in values:
        z=str(line)
        if project:
            z=z.replace(project,"<PROJECT>").replace(project.replace("/","\\"),"<PROJECT>")
        for name,pattern in _PATTERNS.items(): z=pattern.sub(_REPL[name],z)
        out.append(z)
    return out[0] if scalar else out


def audit_gazepoint_reproducibility(paths: Sequence[str|Path]|str|Path,recursive: bool=True,extensions: Sequence[str]=("R","Rmd","Rd","md","txt","html","json","csv","yml","yaml")) -> GP3MLReproducibilityAudit:
    if isinstance(paths,(str,Path)): paths=[paths]
    if not paths: raise GP3MLError("Supply at least one file or directory.")
    files=[]
    for item in paths:
        p=Path(item)
        if p.is_dir(): files.extend(p.rglob("*") if recursive else p.glob("*"))
        elif p.is_file(): files.append(p)
    allowed={"."+x.lower().lstrip(".") for x in extensions}; files=sorted({p.resolve() for p in files if p.is_file() and p.suffix.lower() in allowed})
    rows=[]
    for path in files:
        try: lines=path.read_text(encoding="utf-8").splitlines()
        except Exception: continue
        for issue,pattern in _PATTERNS.items():
            for idx,line in enumerate(lines,1):
                if pattern.search(line): rows.append({"path":path.as_posix(),"line":idx,"issue":issue,"excerpt":str(normalize_gazepoint_artifact_text(line))[:180]})
    findings=pd.DataFrame(rows,columns=["path","line","issue","excerpt"])
    summary=(findings.issue.value_counts(sort=False).rename_axis("issue").reset_index(name="n") if len(findings) else pd.DataFrame(columns=["issue","n"]))
    return GP3MLReproducibilityAudit(status="review" if len(findings) else "pass",files_scanned=len(files),findings=findings,summary=summary)


def write_gazepoint_reproducibility_audit(audit: GP3MLReproducibilityAudit,directory: str|Path=".",prefix: str="gp3ml_reproducibility_audit",overwrite: bool=False):
    if not isinstance(audit,GP3MLReproducibilityAudit): raise GP3MLError("`audit` must be created by audit_gazepoint_reproducibility().")
    return write_tables({"summary":audit.summary,"findings":audit.findings},directory,prefix,overwrite)


def with_gazepoint_reproducible_output(code: Callable[[],T]|T) -> T:
    old=os.environ.get("GP3MLPY_REPRODUCIBLE_EXAMPLES"); os.environ["GP3MLPY_REPRODUCIBLE_EXAMPLES"]="1"
    try: return code() if callable(code) else code
    finally:
        if old is None: os.environ.pop("GP3MLPY_REPRODUCIBLE_EXAMPLES",None)
        else: os.environ["GP3MLPY_REPRODUCIBLE_EXAMPLES"]=old
