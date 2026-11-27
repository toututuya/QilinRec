import itertools

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


def retrieval_metrics(ranked, rows, catalog_size, ks=(10, 50, 100)):
    out = {
        "requests": len(rows),
        "click_requests": sum(bool(r.y.any()) for r in rows),
        "positives": sum(int(r.y.sum()) for r in rows),
    }
    for k in ks:
        recall, ndcg, mrr, ap, hit, recovered = 0.0, 0.0, 0.0, 0.0, 0, 0
        shown = set()
        discounts = 1 / np.log2(np.arange(2, k + 2))
        ideals = discounts.cumsum()
        for predictions, r in zip(ranked, rows, strict=True):
            top = predictions[:k]
            shown.update(top)
            pos = set(r.positives.tolist())
            if not pos:
                continue
            at = np.array([j for j, i in enumerate(top) if i in pos], dtype=int)
            n = len(at)
            recall += n / len(pos)
            if n:
                hit += 1
                recovered += n
                ndcg += discounts[at].sum() / ideals[min(k, len(pos)) - 1]
                mrr += 1 / (at[0] + 1)
                ap += (np.arange(1, n + 1) / (at + 1)).sum() / min(k, len(pos))
        denom = max(1, out["click_requests"])
        out.update(
            {
                f"recall@{k}": recall / denom,
                f"hr@{k}": hit / denom,
                f"ndcg@{k}": ndcg / denom,
                f"mrr@{k}": mrr / denom,
                f"map@{k}": ap / denom,
                f"coverage@{k}": len(shown) / catalog_size,
                f"retained@{k}": recovered / max(1, out["positives"]),
                f"hits@{k}": recovered,
            }
        )
    return out


def exposure_metrics(scores, rows, items, offsets, logits=False):
    labels = np.concatenate([r.y for r in rows])
    if len(scores) != len(labels) or not np.array_equal(items, np.concatenate([r.items for r in rows])):
        raise ValueError("Scores must align with logged exposure pairs")
    groups = []
    for a, b in itertools.pairwise(offsets):
        # An item ID tiebreak avoids using original exposure order as a feature.
        order = np.lexsort((items[a:b], -scores[a:b]))
        groups.append(items[a:b][order].tolist())
    m = retrieval_metrics(groups, rows, max(1, len(np.unique(items))), (10,))
    out = {
        k: m[k]
        for k in ["requests", "click_requests", "positives", "ndcg@10", "mrr@10", "map@10", "retained@10"]
    }
    out["auc"] = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None
    if logits:
        out["logloss"] = float(np.mean(np.logaddexp(0.0, scores.astype(np.float64)) - labels * scores))
    users = np.repeat([r.uid for r in rows], np.diff(offsets))
    order = np.argsort(users, kind="stable")
    boundaries = np.r_[0, np.flatnonzero(np.diff(users[order])) + 1, len(order)]
    total, weight, count = 0.0, 0, 0
    for a, b in itertools.pairwise(boundaries):
        ix = order[a:b]
        y = labels[ix].astype(bool)
        n = int(y.sum())
        if 0 < n < len(y):
            auc = (rankdata(scores[ix])[y].sum() - n * (n + 1) / 2) / (n * (len(y) - n))
            total += len(y) * auc
            weight += len(y)
            count += 1
    out.update(gauc=float(total / weight) if weight else None, gauc_users=count, exposures=len(labels))
    return out
