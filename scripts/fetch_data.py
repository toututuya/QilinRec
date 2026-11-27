"""Download the seven pinned files and verify their exact byte hashes."""
import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen


def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def fetch(root):
    audit = json.loads((Path(__file__).resolve().parents[1]/"results/audit.json").read_text())
    revision = audit["revision"]
    root.mkdir(parents=True,exist_ok=True)
    marker = root/"revision.txt"
    if marker.exists() and marker.read_text(encoding="utf-8-sig").strip() != revision:
        raise ValueError("Destination contains another revision")
    for name, expected in audit["hashes"].items():
        path = root/name
        if path.exists():
            if sha256(path) != expected:
                raise ValueError(f"Existing file differs from pinned data: {path}")
            print("Verified",name,flush=True)
            continue
        part = ("notes/" + name.removeprefix("notes-") if name.startswith("notes-")
                else path.stem + "/train-00000-of-00001.parquet")
        url = f"https://huggingface.co/datasets/THUIR/Qilin/resolve/{revision}/{part}"
        with tempfile.NamedTemporaryFile(dir=root,delete=False) as dest:
            tmp = Path(dest.name)
            try:
                with urlopen(url,timeout=120) as source:
                    shutil.copyfileobj(source,dest)
            except BaseException:
                dest.close()
                tmp.unlink(missing_ok=True)
                raise
        if sha256(tmp) != expected:
            tmp.unlink()
            raise ValueError(f"Hash mismatch for {name}")
        tmp.replace(path)
        print("Downloaded and verified",name,flush=True)
    if not marker.exists():
        marker.write_text(revision+"\n")
    return revision


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest",type=Path,required=True)
    print(fetch(parser.parse_args().dest))
