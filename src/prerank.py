import json

import lightgbm as lgb
import numpy as np

from .features import FEATURES
from .runtime import write_json


def load_frame(folder, stage, kind):
    with np.load(folder / f"{stage}.{kind}.npz", allow_pickle=False) as z:
        if z["names"].tolist() != list(FEATURES):
            raise ValueError("Unexpected feature order")
        return {k:z[k] for k in ("x","items","offsets")}


def fit_prerank(folder, train, valid, y, vy, cfg):
    checkpoint = folder / "lambdarank.last.json"
    state = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
    history = state.get("history", [])
    best_score = state.get("best_score", -1.)
    best_iter = state.get("best_iter", 0)
    start = len(history)
    patience = cfg["lgb_patience"]
    if start < cfg["lgb_rounds"] and (not start or start - best_iter < patience):
        init = lgb.Booster(model_str=state["model"]) if state else None
        model = lgb.LGBMRanker(objective="lambdarank", n_estimators=cfg["lgb_rounds"]-start, learning_rate=.04,
                               num_leaves=31, min_child_samples=60, random_state=cfg["seed"], deterministic=True,
                               force_col_wise=True, verbosity=-1, n_jobs=cfg["faiss_threads"])
        def monitor(env):
            nonlocal best_score, best_iter
            iteration = env.iteration + 1
            metric = float(env.evaluation_result_list[0][2])
            history.append(dict(iteration=iteration, val_ndcg10=metric))
            if metric > best_score:
                best_score, best_iter = metric, iteration
                tmp = folder / "lambdarank.best.tmp"
                env.model.save_model(str(tmp), num_iteration=iteration)
                tmp.replace(folder / "lambdarank.best.txt")
            stop = iteration-best_iter >= patience
            if iteration % 10 == 0 or stop or iteration == cfg["lgb_rounds"] or iteration == best_iter:
                write_json(checkpoint, dict(model=env.model.model_to_string(num_iteration=iteration), history=history,
                                            best_score=best_score, best_iter=best_iter))
            if stop:
                raise lgb.callback.EarlyStopException(best_iter-1, env.evaluation_result_list)
        model.fit(train["x"], y, group=np.diff(train["offsets"]), eval_set=[(valid["x"],vy)],
                  eval_group=[np.diff(valid["offsets"])], eval_metric="ndcg", eval_at=(10,),
                  callbacks=[monitor], feature_name=list(FEATURES), init_model=init)
    write_json(folder / "lgb_train.json",dict(best_iteration=best_iter, validation=history))
    return lgb.Booster(model_file=str(folder / "lambdarank.best.txt"))
