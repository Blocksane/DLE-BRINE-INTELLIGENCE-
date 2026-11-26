import os, math, time, numpy as np, pandas as pd, warnings
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import decision_layer as D
warnings.filterwarnings("ignore")
OUT = D.OUT; NS = 5
d = D.load().dropna(subset=["K", "Br", "Sr"]).copy()          # the SAME 2,230 rows for both feature sets
assert len(d) == 2230 and np.isfinite(d[D.F2].values).all()
assert not (set(D.F1) | set(D.F2)) & D.FORBIDDEN                # T3

def cal_split(tr, gc, seed):
    cnt = tr[gc].value_counts(); cand = sorted(cnt[cnt >= 30].index)
    n_cal = max(3, int(round(0.25 * len(cand))))
    return list(np.random.default_rng(seed).choice(cand, n_cal, replace=False))

def fold(tr, te, feats, gc, i, held):
    o = D.fit_predict_fold(tr, te, feats, quantile=False)
    X, Xt = tr[feats].values, te[feats].values
    for q in (0.1, 0.5, 0.9):
        r = HistGradientBoostingRegressor(loss="quantile", quantile=q, **D.HGB).fit(X, tr["ylog"].values)
        o[f"q{int(q*100)}"] = r.predict(Xt)
    for s in range(NS):
        cg = cal_split(tr, gc, 1000 * s + i)
        assert held not in cg and set(cg) <= set(tr[gc])           # T2
        pr, ca = tr[~tr[gc].isin(cg)], tr[tr[gc].isin(cg)]
        lo = HistGradientBoostingRegressor(loss="quantile", quantile=0.1, **D.HGB).fit(pr[feats].values, pr["ylog"].values)
        hi = HistGradientBoostingRegressor(loss="quantile", quantile=0.9, **D.HGB).fit(pr[feats].values, pr["ylog"].values)
        E = np.maximum(lo.predict(ca[feats].values) - ca["ylog"].values, ca["ylog"].values - hi.predict(ca[feats].values))
        n = len(E); Q = np.quantile(E, min(1, math.ceil((n + 1) * 0.8) / n), method="higher")
        o[f"lo{s}"] = lo.predict(Xt) - Q; o[f"hi{s}"] = hi.predict(Xt) + Q
    return o

def run(gc, groups, scheme, checks):
    rows = []
    for i, g in enumerate(groups):
        te, tr = d[d[gc] == g], d[d[gc] != g]
        assert g not in set(tr[gc]) and not set(tr.IDUSGS) & set(te.IDUSGS)
        parts = []
        for fs, feats in [("F1", D.F1), ("F2", D.F2)]:
            o = fold(tr, te, feats, gc, i, g); o.columns = [f"{c}_{fs}" for c in o.columns]; parts.append(o)
            checks["T2"] += 1
        o = pd.concat(parts, axis=1); o["IDUSGS"] = te["IDUSGS"].values; o["group"] = g
        o["scheme"] = scheme; o["y_true"] = te["ylog"].values; rows.append(o)
        print(scheme, g, "done", flush=True)
    return pd.concat(rows, ignore_index=True)

t0 = time.time(); checks = {"T2": 0}
bc = d["BASIN"].value_counts(); g3 = sorted(bc[bc >= 30].index)
g4 = list(d["IDDB"].value_counts().head(10).index)
res = pd.concat([run("BASIN", g3, "V3", checks), run("IDDB", g4, "V4", checks)], ignore_index=True)
res.to_csv(f"{OUT}/identical_rows_oof_predictions.csv", index=False)
# T1: shuffled labels, classifiers only, both feature sets
rng = np.random.default_rng(D.SEED); ds = d.copy(); ds["Li"] = rng.permutation(ds["Li"].values); ds["ylog"] = np.log10(ds["Li"])
t1 = {}
sh = []
for i, g in enumerate(g3):
    te, tr = ds[ds.BASIN == g], ds[ds.BASIN != g]
    for fs, feats in [("F1", D.F1), ("F2", D.F2)]:
        o = D.fit_predict_fold(tr, te, feats, quantile=False); o["fs"] = fs; o["y"] = te["ylog"].values; sh.append(o)
sh = pd.concat(sh)
for fs in ("F1", "F2"):
    x = sh[sh.fs == fs]
    for tag, thr in [("70", D.T70), ("150", D.T150)]:
        for m in ("logit", "hgb"):
            t1[f"{fs}_AUC_p{tag}_{m}"] = roc_auc_score((x.y >= thr).astype(int), x[f"p{tag}_{m}"])
print("T1", {k: round(v, 3) for k, v in t1.items()}); print("T2 folds", checks["T2"]); print("DONE", round(time.time() - t0))
pd.Series(t1).to_csv(f"{OUT}/identical_rows_T1_shuffle.csv")
