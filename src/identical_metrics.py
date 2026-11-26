import numpy as np, pandas as pd, warnings
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
warnings.filterwarnings("ignore")
OUT="/content/drive/MyDrive/Lithium_Project/reproduced"; NS=5; B=1000
T70,T150,T20=np.log10(70),np.log10(150),np.log10(20)
res=pd.read_csv(f"{OUT}/identical_rows_oof_predictions.csv")
rng=np.random.default_rng(42)
def auc(y,p): return roc_auc_score(y,p) if 0<y.sum()<len(y) else np.nan
tables={}
for sc in ("V3","V4"):
    r=res[res.scheme==sc]; groups=list(r.group.unique()); G=len(groups)
    gd={g:r[r.group==g] for g in groups}
    print(f"\n===== {sc}: {len(r)} rows, {G} held-out groups =====")
    # ---- classification ----
    def cls(frames):
        x=pd.concat(frames); out={}
        for fs in ("F1","F2"):
            for tag,thr in (("70",T70),("150",T150)):
                y=(x.y_true>=thr).astype(int).values
                for m in ("logit","hgb"):
                    p=x[f"p{tag}_{m}_{fs}"].values
                    out[(fs,tag,m,"AUC")]=auc(y,p); out[(fs,tag,m,"Brier")]=brier_score_loss(y,p)
                    out[(fs,tag,m,"AP")]=average_precision_score(y,p)
        return out
    base=cls(list(gd.values()))
    boots=[cls([gd[groups[i]] for i in rng.integers(0,G,G)]) for _ in range(B)]
    rows=[]
    for tag in ("70","150"):
        y=(r.y_true>=(T70 if tag=="70" else T150)).astype(int)
        for m in ("logit","hgb"):
            for met in ("AUC","Brier","AP"):
                v1,v2=base[("F1",tag,m,met)],base[("F2",tag,m,met)]
                dif=[b[("F2",tag,m,met)]-b[("F1",tag,m,met)] for b in boots]
                rows.append(dict(thr=tag,model=m,metric=met,F1=v1,F2=v2,diff_F2_minus_F1=v2-v1,
                                 ci_lo=np.nanpercentile(dif,2.5),ci_hi=np.nanpercentile(dif,97.5),pos=int(y.sum())))
    ct=pd.DataFrame(rows).round(3); tables[f"{sc}_classification"]=ct
    print("CLASSIFICATION (identical rows)"); print(ct.to_string(index=False))
    # ---- conformal / decisions (mean over 5 calibration draws) ----
    def cnt(frames):
        x=pd.concat(frames); y=x.y_true.values; out={}
        for fs in ("F1","F2"):
            for s in range(NS):
                lo,hi=x[f"lo{s}_{fs}"].values,x[f"hi{s}_{fs}"].values
                for tag,thr in (("70",T70),("150",T150)):
                    go,ng=lo>=thr,hi<thr; pos=y>=thr
                    out[(fs,s,tag)]=np.array([len(y),go.sum(),ng.sum(),pos.sum(),(pos&ng).sum(),(go&pos).sum(),(go&(y<T20)).sum(),
                                              ((y>=lo)&(y<=hi)).sum(),(hi-lo).sum()])
        return out
    def summ(c,fs,tag):
        a=np.array([c[(fs,s,tag)] for s in range(NS)],float)
        n,go,ng,pos,pin,gtp,gfa,cov,wid=a.T
        return dict(go=go/n,nogo=ng/n,measure=1-(go+ng)/n,pos_in_nogo=pin/pos,cov=cov/n,width=wid/n,go_n=go,go_tp=gtp)
    cb=cnt(list(gd.values()))
    cboots=[cnt([gd[groups[i]] for i in rng.integers(0,G,G)]) for _ in range(B)]
    rows=[]
    for tag in ("70","150"):
        s1,s2=summ(cb,"F1",tag),summ(cb,"F2",tag)
        for k in ("cov","width","go","nogo","measure","pos_in_nogo"):
            dif=[np.nanmean(summ(b,"F2",tag)[k])-np.nanmean(summ(b,"F1",tag)[k]) for b in cboots]
            rows.append(dict(thr=tag,metric=k,F1=np.nanmean(s1[k]),F1_range=f"{np.nanmin(s1[k]):.3f}-{np.nanmax(s1[k]):.3f}",
                             F2=np.nanmean(s2[k]),F2_range=f"{np.nanmin(s2[k]):.3f}-{np.nanmax(s2[k]):.3f}",
                             diff=np.nanmean(s2[k])-np.nanmean(s1[k]),ci_lo=np.nanpercentile(dif,2.5),ci_hi=np.nanpercentile(dif,97.5)))
        rows.append(dict(thr=tag,metric="go_calls_mean",F1=s1["go_n"].mean(),F2=s2["go_n"].mean()))
    cq=pd.DataFrame(rows).round(3); tables[f"{sc}_conformal_decision"]=cq
    print("\nCONFORMAL + DECISION (means over 5 calibration draws; ranges across draws)"); print(cq.to_string(index=False))
    # ---- transitions at 70 ----
    x=r; y=x.y_true.values; tr=[]
    for s in range(NS):
        def st(fs):
            lo,hi=x[f"lo{s}_{fs}"].values,x[f"hi{s}_{fs}"].values
            return np.where(lo>=T70,"Go",np.where(hi<T70,"NoGo","Measure"))
        a,b=st("F1"),st("F2")
        t=pd.crosstab(pd.Series(a,name="F1"),pd.Series(b,name="F2")).reindex(index=["Go","Measure","NoGo"],columns=["Go","Measure","NoGo"],fill_value=0)
        pos=y>=T70
        tp=pd.crosstab(pd.Series(a[pos],name="F1"),pd.Series(b[pos],name="F2")).reindex(index=["Go","Measure","NoGo"],columns=["Go","Measure","NoGo"],fill_value=0)
        tr.append((t,tp))
    t=sum(a for a,_ in tr)/NS; tp=sum(b for _,b in tr)/NS
    m1=t.loc["Measure"].sum()
    print("\nTRANSITIONS @70 (rows, mean over draws): F1 state (rows) -> F2 state (cols)"); print(t.round(0).astype(int).to_string())
    print(f"F1 'Measure-first' rows resolved by F2: to NoGo {t.loc['Measure','NoGo']/m1:.1%}, to Go {t.loc['Measure','Go']/m1:.1%}, still Measure {t.loc['Measure','Measure']/m1:.1%}")
    print("True positives (>=70) by transition (rows):"); print(tp.round(1).to_string())
    tables[f"{sc}_transitions"]=t.reset_index()
    # ---- per-group AUC (>=10 pos and >=10 neg) ----
    pg=[]
    for g in groups:
        x=gd[g]; y=(x.y_true>=T70).astype(int).values
        if y.sum()>=10 and (len(y)-y.sum())>=10:
            pg.append(dict(group=g,n=len(x),pos=int(y.sum()),AUC_hgb_F1=auc(y,x.p70_hgb_F1),AUC_hgb_F2=auc(y,x.p70_hgb_F2),
                           AUC_logit_F1=auc(y,x.p70_logit_F1),AUC_logit_F2=auc(y,x.p70_logit_F2)))
    pg=pd.DataFrame(pg).round(3); tables[f"{sc}_auc_per_group"]=pg
    print("\nPER-GROUP AUC @70 (>=10 pos/neg)"); print(pg.to_string(index=False))
    # ---- median prediction error ----
    for fs in ("F1","F2"):
        e=(r[f"q50_{fs}"]-r.y_true)
        print(f"{fs} q50 log10 MAE {e.abs().mean():.3f} RMSE {np.sqrt((e**2).mean()):.3f} bias {e.mean():+.3f}")
with pd.ExcelWriter(f"{OUT}/identical_rows_results.xlsx") as w:
    for k,v in tables.items(): v.to_excel(w,sheet_name=k[:31],index=False)
