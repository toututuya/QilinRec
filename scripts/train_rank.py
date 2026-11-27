import json
import time

import numpy as np
import torch
from torch import nn

from src.data import load_requests
from src.features import rank_groups
from src.metrics import exposure_metrics, retrieval_metrics
from src.prerank import load_frame
from src.rank import CTRModel, RankingDataset
from src.runtime import configure, save_torch, write_json


def train(args,cfg):
    run,folder = args.run,args.run / "base"
    data = load_requests(run)
    logged = load_frame(folder,"rank_train","exposures")
    scale = np.stack([logged["x"].mean(axis=0,dtype=np.float64),logged["x"].std(axis=0,dtype=np.float64).clip(.001)]).astype(np.float32)
    np.save(folder / "rank_feature_scale.npy",scale)
    def dataset(stage,frame,labeled=True,short=None):
        return RankingDataset(data[stage],frame,cfg,scale,args.device,labeled,short)
    fit = dataset("rank_train",logged)
    valid = dataset("rank_val",load_frame(folder,"rank_val","exposures"))
    users = json.loads((run/"split.json").read_text())["users"]
    model = CTRModel(args.model,np.load(folder/"vectors.npy"),np.load(run/"codes.npy"),users,cfg,len(scale[0])).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=cfg["rank_lr"])
    criterion = nn.BCEWithLogitsLoss()
    last,best = [folder / f"{args.model}.{tag}.pt" for tag in ("last","best")]
    history,score,stale = [],-1.,0
    if last.exists():
        state = torch.load(last,map_location=args.device,weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history,score,stale = state["history"],state["best"],state["stale"]
    for epoch in range(len(history),cfg["rank_epochs"]):
        if stale >= cfg["patience"]:
            break
        start = time.perf_counter()
        torch.manual_seed(cfg["seed"]+epoch)
        order = torch.randperm(len(fit.items),device=args.device)
        loss_sum = 0.
        model.train()
        for ix in order.split(cfg["rank_batch"]):
            pred = fit.score(model,ix)
            loss = criterion(pred,fit.y[ix])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()*len(ix)
        pred = valid.predict(model,cfg["rank_batch"])
        vm = exposure_metrics(pred,data["rank_val"],valid.ids,valid.offsets,logits=True)
        metric = vm["auc"]
        entry = dict(epoch=epoch+1,train_bce=loss_sum/len(order),val_auc=metric,val_ndcg10=vm["ndcg@10"],
                     val_logloss=vm["logloss"],seconds=time.perf_counter()-start)
        history.append(entry)
        if metric > score:
            score,stale = metric,0
            save_torch(best,dict(model=model.state_dict(),epoch=epoch+1,val_auc=metric))
        else:
            stale += 1
        save_torch(last,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),history=history,best=score,stale=stale))
        print(args.model,entry,flush=True)
    state = torch.load(best,map_location=args.device,weights_only=True)
    model.load_state_dict(state["model"])
    n = len(np.load(run/"catalog.npy"))
    report = dict(history=history,best_epoch=state["epoch"])
    for stage in ("rank_val","test"):
        r = data[stage]
        exp = dataset(stage,load_frame(folder,stage,"exposures"))
        logits = exp.predict(model,cfg["rank_batch"])
        with np.load(folder / f"{stage}.shortlist.npz") as short:
            cand = dataset(stage,load_frame(folder,stage,"candidates"),False,short)
        pred = cand.predict(model,cfg["rank_batch"])
        groups,ix = rank_groups(pred,cand.ids,cand.offsets,cfg["rank_k"])
        np.savez(folder/f"{stage}.{args.model}.top.npz",request_ids=np.array([a.rid for a in r]),
                 offsets=np.r_[0,np.cumsum([len(g) for g in groups])],items=cand.ids[ix],logits=pred[ix])
        report[stage] = dict(exposure=exposure_metrics(logits,r,exp.ids,exp.offsets,logits=True),
                             cascade=retrieval_metrics(groups,r,n,(cfg["rank_k"],)))
    write_json(folder / f"{args.model}.json",report)


if __name__ == "__main__":
    args,cfg = configure("Train and evaluate the MLP or DIN exposure ranker",stage="rank")
    train(args,cfg)
