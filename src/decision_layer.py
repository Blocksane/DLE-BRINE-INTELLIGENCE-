import os, sys, math, time, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
warnings.filterwarnings("ignore")

SEED = 42
OUT = "/content/reproduced"
IN = "/content/modelling_table.csv"
T70, T150, T20 = np.log10(70), np.log10(150), np.log10(20)
AW = {"Na": 22.990, "K": 39.098, "Ca": 40.078, "Mg": 24.305, "Sr": 87.62, "Cl": 35.45, "Br": 79.904}
HGB = dict(max_depth=3, learning_rate=0.05, max_iter=300, random_state=SEED, early_stopping=False)
F1 = ["lnTDS10", "lnNaCl", "lnCaCl", "lnMgCl", "lnCaMg"]
F2 = F1 + ["lnClBr", "lnNaK", "lnSrCa"]
FORBIDDEN = {"IDUSGS","IDDB","BASIN","FIELD","STATE","Li","TEMP","TEMP_R","LATITUDE","LONGITUDE",
             "CHARGEBAL","Mg_Li","Li_Cl","ylog"}
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True)
    with open(os.path.join(OUT, "decision_layer_log.txt"), "a") as f:
        f.write(s + "\n")

def load():
    d = pd.read_csv(IN, keep_default_na=False, na_values=[""],
                    dtype={"FIELD": str, "IDDB": str, "BASIN": str, "STATE": str})
    m = lambda e: d[e] / AW[e]
    d["lnTDS10"] = np.log10(d["TDS"])
    d["lnNaCl"] = np.log(m("Na") / m("Cl"))
    d["lnCaCl"] = np.log(m("Ca") / m("Cl"))
    d["lnMgCl"] = np.log(m("Mg") / m("Cl"))
    d["lnCaMg"] = np.log(m("Ca") / m("Mg"))
    d["lnClBr"] = np.log(d["Cl"] / d["Br"])          # mass ratio (constant offset vs molar; no effect after scaling)
    d["lnNaK"] = np.log(m("Na") / m("K"))
    d["lnSrCa"] = np.log(m("Sr") / m("Ca"))
    d["ylog"] = np.log10(d["Li"])
    return d

def fit_predict_fold(tr, te, feats, quantile=True, seed_i=0, group_col="BASIN"):
    X, Xt = tr[feats].values, te[feats].values
    out = pd.DataFrame(index=te.index)
    for thr, tag in [(T70, "70"), (T150, "150")]:
        y = (tr["ylog"] >= thr).astype(int).values
        lg = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, penalty="l2", max_iter=2000)).fit(X, y)
        out[f"p{tag}_logit"] = lg.predict_proba(Xt)[:, 1]
        hg = HistGradientBoostingClassifier(**HGB).fit(X, y)
        out[f"p{tag}_hgb"] = hg.predict_proba(Xt)[:, 1]
    if quantile:
        for q in (0.1, 0.5, 0.9):
            r = HistGradientBoostingRegressor(loss="quantile", quantile=q, **HGB).fit(X, tr["ylog"].values)
            out[f"q{int(q*100)}"] = r.predict(Xt)
        # conformalized quantile regression, calibration on whole held-back GROUPS
        cnt = tr[group_col].value_counts()
        cand = sorted(cnt[cnt >= 30].index)
        n_cal = max(3, int(round(0.25 * len(cand))))
        rng = np.random.default_rng(SEED + seed_i)
        cal_groups = list(rng.choice(cand, n_cal, replace=False))
        proper, cal = tr[~tr[group_col].isin(cal_groups)], tr[tr[group_col].isin(cal_groups)]
        lo = HistGradientBoostingRegressor(loss="quantile", quantile=0.1, **HGB).fit(proper[feats].values, proper["ylog"].values)
        hi = HistGradientBoostingRegressor(loss="quantile", quantile=0.9, **HGB).fit(proper[feats].values, proper["ylog"].values)
        E = np.maximum(lo.predict(cal[feats].values) - cal["ylog"].values, cal["ylog"].values - hi.predict(cal[feats].values))
        n = len(E); level = min(1.0, math.ceil((n + 1) * 0.8) / n)
        Q = np.quantile(E, level, method="higher")
        out["cqr_lo"] = lo.predict(Xt) - Q
        out["cqr_hi"] = hi.predict(Xt) + Q
        out["cqr_Q"] = Q
        out["n_cal"] = n
        out["_cal_groups"] = "|".join(cal_groups)
    return out

def run_scheme(d, feats, group_col, groups, scheme, fs, quantile=True, checks=None):
    res = []
    for i, g in enumerate(groups):
        te = d[d[group_col] == g]
        tr = d[d[group_col] != g]
        # T2 integrity
        assert g not in set(tr[group_col]), "held-out group in training"
        assert len(set(tr["IDUSGS"]) & set(te["IDUSGS"])) == 0, "row overlap"
        o = fit_predict_fold(tr, te, feats, quantile, seed_i=i, group_col=group_col)
        if quantile:
            cg = o["_cal_groups"].iloc[0].split("|")
            assert g not in cg and set(cg) <= set(tr[group_col]), "calibration group problem"
            o = o.drop(columns="_cal_groups")
        o["IDUSGS"] = te["IDUSGS"].values; o["scheme"] = scheme; o["feature_set"] = fs
        o["group"] = g; o["y_true"] = te["ylog"].values
        res.append(o)
        if checks is not None: checks["T2"] += 1
    return pd.concat(res, ignore_index=True)

if __name__ == "__main__":
    t0 = time.time()
    d = load()
    checks = {"T2": 0}
    # T3: features exactly as specified and nothing forbidden
    assert F1 == ["lnTDS10","lnNaCl","lnCaCl","lnMgCl","lnCaMg"]
    assert not (set(F1) | set(F2)) & FORBIDDEN
    d1 = d.copy()
    d2 = d.dropna(subset=["K", "Br", "Sr"]).copy()
    for nm, dd in [("F1", d1), ("F2", d2)]:
        assert np.isfinite(dd[(F1 if nm == "F1" else F2)].values).all()
        log(nm, "rows", len(dd), "pos70", int((dd.ylog >= T70).sum()), "pos150", int((dd.ylog >= T150).sum()))
    allres = []
    for fs, dd, feats in [("F1", d1, F1), ("F2", d2, F2)]:
        bc = dd["BASIN"].value_counts()
        g3 = sorted(bc[bc >= 30].index)
        log(fs, "V3 held-out basins:", len(g3), "rows tested:", int(bc[bc >= 30].sum()))
        allres.append(run_scheme(dd, feats, "BASIN", g3, "V3", fs, True, checks)); log(fs, "V3 done", round(time.time() - t0))
        ic = dd["IDDB"].value_counts()
        g4 = list(ic.head(10).index)
        allres.append(run_scheme(dd, feats, "IDDB", g4, "V4", fs, True, checks)); log(fs, "V4 done", round(time.time() - t0))
    oof = pd.concat(allres, ignore_index=True)
    oof.to_csv(os.path.join(OUT, "decision_layer_oof_predictions.csv"), index=False)
    # T1: shuffled labels, V3 F1, classifiers only
    rng = np.random.default_rng(SEED)
    ds = d1.copy(); ds["Li"] = rng.permutation(ds["Li"].values); ds["ylog"] = np.log10(ds["Li"])
    bc = ds["BASIN"].value_counts(); g3 = sorted(bc[bc >= 30].index)
    sh = run_scheme(ds, F1, "BASIN", g3, "V3", "F1", False, None)
    t1 = {}
    for tag, thr in [("70", T70), ("150", T150)]:
        y = (sh["y_true"] >= thr).astype(int)
        for m in ("logit", "hgb"):
            t1[f"AUC_p{tag}_{m}"] = roc_auc_score(y, sh[f"p{tag}_{m}"])
    log("T1 shuffled-label pooled AUCs:", {k: round(v, 3) for k, v in t1.items()})
    pd.Series(t1).to_csv(os.path.join(OUT, "decision_layer_T1_shuffle.csv"))
    log("T2 folds checked:", checks["T2"])
    log("DONE", round(time.time() - t0), "s")
