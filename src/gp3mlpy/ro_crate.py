from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import pandas as pd
from .exceptions import GP3MLError
from .objects import GP3MLROCrate, GP3MLROCrateValidation
from ._utils import worst_status


def _sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()


def write_gazepoint_ro_crate(path, files, name, description, creator_name, creator_orcid=None, license="MIT", doi=None, copy_files=True):
    out=Path(path).expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
    sources=[Path(x).expanduser().resolve() for x in ([files] if isinstance(files,(str,Path)) else files)]
    if any(not x.exists() for x in sources): raise GP3MLError("All RO-Crate source files must exist.")
    entities=[]; manifest=[]
    for src in sources:
        rel=src.name; dest=out/rel
        if copy_files: shutil.copy2(src,dest)
        target=dest if copy_files else src; digest=_sha256_file(target); size=target.stat().st_size
        entities.append({"@id":rel,"@type":"File","name":rel,"contentSize":size,"sha256":digest})
        manifest.append({"file":rel,"sha256":digest,"size":size})
    creator_id = (str(creator_orcid) if str(creator_orcid).startswith(("http://","https://")) else f"https://orcid.org/{creator_orcid}") if creator_orcid else "#creator"
    graph=[
        {"@id":"ro-crate-metadata.json","@type":"CreativeWork","about":{"@id":"./"},"conformsTo":{"@id":"https://w3id.org/ro/crate/1.2"}},
        {"@id":"./","@type":"Dataset","name":name,"description":description,"license":license,"identifier":doi,"creator":{"@id":creator_id},"hasPart":[{"@id":e["@id"]} for e in entities]},
        {"@id":creator_id,"@type":"Person","name":creator_name},
        *entities,
    ]
    metadata_obj={"@context":"https://w3id.org/ro/crate/1.2/context","@graph":graph}
    meta=out/"ro-crate-metadata.json"; meta.write_text(json.dumps(metadata_obj,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    man=pd.DataFrame(manifest); manifest_path=out/"sha256-manifest.csv"; man.to_csv(manifest_path,index=False)
    return GP3MLROCrate(path=str(out),metadata=str(meta),manifest=str(manifest_path),file_manifest=man,note="RO-Crate 1.2 metadata export with gp3ml SHA-256 manifest; independent conformance validation remains recommended.")


def validate_gazepoint_ro_crate(path):
    if isinstance(path,GP3MLROCrate): path=path.path
    root=Path(path); meta=root/"ro-crate-metadata.json"; manifest=root/"sha256-manifest.csv"
    statuses={"metadata":"pass","manifest":"pass","context":"pass","root_dataset":"pass","hashes":"pass"}; details={k:"" for k in statuses}
    if not meta.exists(): statuses["metadata"]="fail"
    if not manifest.exists(): statuses["manifest"]="fail"
    if meta.exists():
        try: obj=json.loads(meta.read_text(encoding="utf-8"))
        except Exception: obj={}; statuses["metadata"]="fail"
        context=obj.get("@context","");
        if "ro/crate/1.2" not in str(context): statuses["context"]="fail"
        ids=[z.get("@id","") for z in obj.get("@graph",[]) if isinstance(z,dict)]
        if "./" not in ids: statuses["root_dataset"]="fail"
    if manifest.exists():
        try:
            man=pd.read_csv(manifest)
            bad=[]
            for _,row in man.iterrows():
                target=root/str(row["file"]); bad.append(not target.exists() or _sha256_file(target)!=str(row["sha256"]))
            if any(bad): statuses["hashes"]="fail"
        except Exception: statuses["hashes"]="fail"
    checks=pd.DataFrame({"check":list(statuses),"status":list(statuses.values()),"detail":[details[k] for k in statuses]})
    return GP3MLROCrateValidation(status=worst_status(checks.status),checks=checks,note="This validates gp3mlpy's minimal RO-Crate structure and hashes; formal external RO-Crate conformance validation is still recommended.")
