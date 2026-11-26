import numpy as np, pandas as pd, warnings
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
warnings.filterwarnings("ignore")
OUT="/content/drive/MyDrive/Lithium_Project/reproduced"
T70,T150,T20=np.log10(70),np.log10(150),np.log10(20)
o=pd.read_csv(f"{OUT}/decision_layer_oof_predictions.csv")
def auc(y,p):
    return roc_auc_score(y,p) if y.sum()>=10 and (len(y)-y.sum())>=10 else np.nan
rows=[]
for (sc,fs),g in o.groupby(["scheme","feature_set"]):
    for tag,thr in [("70",T70),("150",T150)]:
        y=(g.y_true>=thr).astype(int)
        for m in ("logit","hgb"):
            p=g[f"p{tag}_{m}"]
            # ECE with 10 quantile bins
            b=pd.qcut(p.rank(method="first"),10,labels=False)
            ece=sum(len(x)/len(g)*abs(x_y-x_p) for x_y,x_p,x in [(y[b==k].mean(),p[b==k].mean(),y[b==k]) for k in range(10)])
            rows.append(dict(scheme=sc,feature_set=fs,thr=tag,model=m,n=len(g),pos=int(y.sum()),
                             AUC=roc_auc_score(y,p),AP=average_precision_score(y,p),
                             Brier=brier_score_loss(y,p),Brier_null=brier_score_loss(y,np.full(len(y),y.mean())),ECE=ece))
pool=pd.DataFrame(rows).round(3)
print("POOLED CLASSIFICATION"); print(pool.to_string(index=False))
# per-group classification (hgb, y70 and y150)
pg=[]
for (sc,fs,grp),g in o.groupby(["scheme","feature_set","group"]):
    for tag,thr in [("70",T70),("150",T150)]:
        y=(g.y_true>=thr).astype(int)
        r=dict(scheme=sc,feature_set=fs,group=grp,thr=tag,n=len(g),pos=int(y.sum()))
        for m in ("logit","hgb"):
            r[f"AUC_{m}"]=auc(y,g[f"p{tag}_{m}"])
            r[f"Brier_{m}"]=brier_score_loss(y,g[f"p{tag}_{m}"])
        r["Brier_null_trainfree"]=brier_score_loss(y,np.full(len(y),y.mean()))
        pg.append(r)
pg=pd.DataFrame(pg)
# quantile / CQR
qr=[]
for (sc,fs,grp),g in o.groupby(["scheme","feature_set","group"]):
    r=dict(scheme=sc,feature_set=fs,group=grp,n=len(g))
    for q,c in [(0.1,"q10"),(0.5,"q50"),(0.9,"q90")]:
        d=g.y_true-g[c]; r[f"pinball_{c}"]=np.mean(np.maximum(q*d,(q-1)*d))
    r["raw80_cov"]=((g.y_true>=g.q10)&(g.y_true<=g.q90)).mean()
    r["cqr_cov"]=((g.y_true>=g.cqr_lo)&(g.y_true<=g.cqr_hi)).mean()
    r["cqr_width"]=(g.cqr_hi-g.cqr_lo).mean()
    r["raw_width"]=(g.q90-g.q10).mean()
    qr.append(r)
qr=pd.DataFrame(qr)
pq=[]
for (sc,fs),g in o.groupby(["scheme","feature_set"]):
    r=dict(scheme=sc,feature_set=fs,n=len(g))
    r["raw80_cov"]=((g.y_true>=g.q10)&(g.y_true<=g.q90)).mean()
    r["cqr_cov"]=((g.y_true>=g.cqr_lo)&(g.y_true<=g.cqr_hi)).mean()
    r["raw_width"]=(g.q90-g.q10).mean(); r["cqr_width"]=(g.cqr_hi-g.cqr_lo).mean()
    # group-level coverage spread
    sub=qr[(qr.scheme==sc)&(qr.feature_set==fs)]
    r["groups"]=len(sub); r["groups_cov_below_0.70"]=int((sub.cqr_cov<0.70).sum())
    r["min_group_cov"]=sub.cqr_cov.min(); r["max_group_cov"]=sub.cqr_cov.max()
    pq.append(r)
pq=pd.DataFrame(pq).round(3)
print("\nPOOLED CONFORMAL (80% nominal)"); print(pq.to_string(index=False))
# decision states
def states(g,thr):
    s=np.where(g.cqr_lo>=thr,"Go",np.where(g.cqr_hi<thr,"NoGo","Measure"))
    return pd.Series(s,index=g.index)
ds=[];dp=[]
for (sc,fs,grp),g in o.groupby(["scheme","feature_set","group"]):
    for tag,thr in [("70",T70),("150",T150)]:
        s=states(g,thr); y=g.y_true
        r=dict(scheme=sc,feature_set=fs,group=grp,thr=tag,n=len(g),
               go=int((s=="Go").sum()),nogo=int((s=="NoGo").sum()),measure=int((s=="Measure").sum()),
               pos=int((y>=thr).sum()),
               miss_n=int(((s=="NoGo")&(y>=thr)).sum()),
               go_true_pos=int(((s=="Go")&(y>=thr)).sum()),
               go_false_alarm_lt20=int(((s=="Go")&(y<T20)).sum()))
        ds.append(r)
ds=pd.DataFrame(ds)
agg=ds.groupby(["scheme","feature_set","thr"])[["n","go","nogo","measure","pos","miss_n","go_true_pos","go_false_alarm_lt20"]].sum().reset_index()
agg["go_share"]=agg.go/agg.n; agg["nogo_share"]=agg.nogo/agg.n; agg["measure_share"]=agg.measure/agg.n
agg["miss_rate_of_nogo"]=agg.miss_n/agg.nogo.replace(0,np.nan)
agg["go_precision"]=agg.go_true_pos/agg.go.replace(0,np.nan)
agg["go_false_alarm_rate"]=agg.go_false_alarm_lt20/agg.go.replace(0,np.nan)
agg["recall_of_pos_in_go"]=agg.go_true_pos/agg.pos
agg["pos_in_nogo_share"]=agg.miss_n/agg.pos
print("\nPOOLED DECISION STATES"); print(agg.round(3).to_string(index=False))
# reference basins
ref=[]
for (sc,fs,grp),g in o[(o.scheme=="V3")&(o.group.isin(["San Joaquin","Sacramento","Michigan","Gulf Coast"]))].groupby(["scheme","feature_set","group"]):
    s=states(g,T70)
    ref.append(dict(feature_set=fs,basin=grp,n=len(g),pos70=int((g.y_true>=T70).sum()),
        median_true_Li=10**g.y_true.median(),median_pred_q50=10**g.q50.median(),
        mean_p70_hgb=g.p70_hgb.mean(),go=int((s=="Go").sum()),nogo=int((s=="NoGo").sum()),measure=int((s=="Measure").sum()),
        cqr_cov=((g.y_true>=g.cqr_lo)&(g.y_true<=g.cqr_hi)).mean(),cqr_width=(g.cqr_hi-g.cqr_lo).mean()))
ref=pd.DataFrame(ref).round(3)
print("\nREFERENCE BASINS (V3, threshold 70)"); print(ref.to_string(index=False))
with pd.ExcelWriter(f"{OUT}/decision_layer_results.xlsx") as w:
    pool.to_excel(w,sheet_name="pooled_classification",index=False)
    pg.round(3).to_excel(w,sheet_name="per_group_classification",index=False)
    pq.to_excel(w,sheet_name="pooled_conformal",index=False)
    qr.round(3).to_excel(w,sheet_name="per_group_conformal",index=False)
    agg.round(3).to_excel(w,sheet_name="decision_pooled",index=False)
    ds.to_excel(w,sheet_name="decision_per_group",index=False)
    ref.to_excel(w,sheet_name="reference_basins",index=False)
pg.to_pickle("/home/claude/pg.pkl"); qr.to_pickle("/home/claude/qr.pkl"); ds.to_pickle("/home/claude/ds.pkl")
