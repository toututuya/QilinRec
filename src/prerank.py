import json

import lightgbm as lgb
import numpy as np

from .features import FEATURES
from .runtime import write_json


def load_frame(folder, stage, kind):
    with np.load(folder / f"{stage}.{kind}.npz", allow_pickle=False) as z:
        if z["names"].tolist() != list(FEATURES):
            raise ValueError("Unexpected feature order")
        return {k: z[k] for k in ("x", "items", "offsets")}


def canonical_order(frame):
    groups = np.repeat(np.arange(len(frame["offsets"]) - 1), np.diff(frame["offsets"]))
    return np.lexsort((frame["items"], groups))


def request_ndcg_metric(items, offsets):
    sizes = np.diff(offsets)
    groups = np.repeat(np.arange(len(sizes)), sizes)
    position = np.arange(len(items)) - np.repeat(offsets[:-1], sizes)
    discount = 1 / np.log2(np.arange(2, 12))
    ideal = np.r_[0.0, discount.cumsum()]
    weights = np.where(position < 10, discount[position.clip(max=9)], 0.0)

    def score(labels, predictions):
        order = np.lexsort((items, -predictions, groups))
        positives = np.add.reduceat(labels, offsets[:-1]).astype(int)
        valid = positives > 0
        dcg = np.add.reduceat(labels[order] * weights, offsets[:-1])
        ndcg = np.mean(dcg[valid] / ideal[positives[valid].clip(max=10)])
        return "request_ndcg10", float(ndcg), True

    return score


def fit_prerank(folder, train, valid, y, vy, cfg):
    checkpoint = folder / "lambdarank.last.json"
    state = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
    history = state.get("history", [])
    best_score = state.get("best_score", -1.0)
    best_iter = state.get("best_iter", 0)
    start = len(history)
    patience = cfg["lgb_patience"]
    if start < cfg["lgb_rounds"] and (not start or start - best_iter < patience):
        init = lgb.Booster(model_str=state["model"]) if state else None
        model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="None",
            n_estimators=cfg["lgb_rounds"] - start,
            learning_rate=0.04,
            num_leaves=31,
            min_child_samples=60,
            random_state=cfg["seed"],
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
            n_jobs=cfg["faiss_threads"],
        )

        def monitor(env):
            nonlocal best_score, best_iter
            iteration = env.iteration + 1
            metric = next(
                float(value) for _, name, value, _ in env.evaluation_result_list if name == "request_ndcg10"
            )
            history.append({"iteration": iteration, "val_ndcg10": metric})
            if metric > best_score:
                best_score, best_iter = metric, iteration
                tmp = folder / "lambdarank.best.tmp"
                env.model.save_model(str(tmp), num_iteration=iteration)
                tmp.replace(folder / "lambdarank.best.txt")
            stop = iteration - best_iter >= patience
            if iteration % 10 == 0 or stop or iteration == cfg["lgb_rounds"] or iteration == best_iter:
                write_json(
                    checkpoint,
                    {
                        "model": env.model.model_to_string(num_iteration=iteration),
                        "history": history,
                        "best_score": best_score,
                        "best_iter": best_iter,
                    },
                )
            if stop:
                raise lgb.callback.EarlyStopException(best_iter - 1, env.evaluation_result_list)

        train_ix, val_ix = canonical_order(train), canonical_order(valid)
        metric = request_ndcg_metric(valid["items"][val_ix], valid["offsets"])
        model.fit(
            train["x"][train_ix],
            y[train_ix],
            group=np.diff(train["offsets"]),
            eval_X=valid["x"][val_ix],
            eval_y=vy[val_ix],
            eval_group=[np.diff(valid["offsets"])],
            eval_metric=metric,
            eval_at=(10,),
            callbacks=[monitor],
            feature_name=list(FEATURES),
            init_model=init,
        )
    write_json(folder / "lgb_train.json", {"best_iteration": best_iter, "validation": history})
    return lgb.Booster(model_file=str(folder / "lambdarank.best.txt"))
