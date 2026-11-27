"""Replay each saved test shortlist and verify its membership and hit denominators."""
import argparse
import json
from pathlib import Path

import numpy as np

from src.data import load_requests
from src.features import FEATURES
from src.metrics import retrieval_metrics
from src.runtime import write_json


def read(path):
    return json.loads(path.read_text())


def verify(run):
    base = run/"base"
    rows = load_requests(run)["test"]
    c = read(run/"config.json")["config"]
    catalog = np.load(run/"catalog.npy")
    available = set(catalog.tolist())
    report = {}
    with np.load(base/"test.candidates.npz") as logged:
        if list(logged["names"]) != list(FEATURES):
            raise AssertionError("Feature schema mismatch")
        vals = {n:logged[n] for n in ["items","x","offsets"]}
    last = None
    stages = [("rrf",vals,read(base/"recall.json")["test"]["rrf"],c["recall_k"]),
              ("lambdarank",base/"test.shortlist.npz",read(base/"prerank.json")["test"]["lambdarank"]["cascade"],c["prerank_k"]),
              ("mlp",base/"test.mlp.top.npz",read(base/"mlp.json")["test"]["cascade"],c["rank_k"]),
              ("din",base/"test.din.top.npz",read(base/"din.json")["test"]["cascade"],c["rank_k"])]
    for method,source,reference,k in stages:
        if isinstance(source,dict):
            f=source
        else:
            with np.load(source) as z:
                f={n:z[n] for n in z.files}
            np.testing.assert_array_equal(f["request_ids"],[r.rid for r in rows])
        offsets,items = f["offsets"],f["items"]
        assert len(offsets) == len(rows)+1 and offsets[0] == 0 and offsets[-1] == len(items)
        groups=[]
        for j,(r,a,b) in enumerate(zip(rows,offsets[:-1],offsets[1:],strict=True)):
            ids = items[a:b].tolist()
            members = set(ids)
            assert len(ids)<=k and len(members)==len(ids)
            assert available.issuperset(members) and not members.intersection(r.seen)
            if last is not None:
                assert last[j].issuperset(members)
            groups.append(ids)
        if method in ("rrf","lambdarank"):
            last = [set(g) for g in groups]
        if method=="rrf":
            ranks=f["x"][:,9:12]
            np.testing.assert_array_equal(ranks>0,f["x"][:,15:18].astype(bool))
            np.testing.assert_allclose(f["x"][:,-1],np.where(ranks>0,1/(c["rrf_offset"]+ranks),0.).sum(1),atol=1e-7)
        actual = retrieval_metrics(groups,rows,len(catalog),(k,))
        for name in ["positives","requests",f"recall@{k}",f"ndcg@{k}",f"hits@{k}",f"retained@{k}"]:
            np.testing.assert_allclose(reference[name],actual[name],rtol=1e-9)
        report[method]=dict(k=k,requests=len(rows),positives=actual["positives"],hits=actual[f"hits@{k}"],
                            min_returned=int(np.diff(offsets).min()),max_returned=int(np.diff(offsets).max()))
    write_json(run/"verification.json",report)
    print(report,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True)
    verify(p.parse_args().run)
