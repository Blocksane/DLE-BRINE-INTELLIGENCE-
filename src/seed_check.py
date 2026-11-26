import math, numpy as np, pandas as pd, warnings
from sklearn.ensemble import HistGradientBoostingRegressor
import decision_layer as D
warnings.filterwarnings("ignore")
d=D.load()
rows=[]
for fs,dd,feats in [("F1",d,D.F1),("F2",d.dropna(subset=["K","Br","Sr"]),D.F2)]:
    bc=dd["BASIN"].value_counts(); g3=sorted(bc[bc>=30].index)
    for seed in range(1,6):
        cov=[];st=[];grp_cov=[]
        for i,g in enumerate(g3):
            te=dd[dd.BASIN==g]; tr=dd[dd.BASIN!=g]
            cnt=tr.BASIN.value_counts(); cand=sorted(cnt[cnt>=30].index)
            n_cal=max(3,int(round(0.25*len(cand))))
            rng=np.random.default_rng(1000*seed+i)
            cg=list(rng.choice(cand,n_cal,replace=False))
            pr,ca=tr[~tr.BASIN.isin(cg)],tr[tr.BASIN.isin(cg)]
            lo=HistGradientBoostingRegressor(loss="quantile",quantile=0.1,**D.HGB).fit(pr[feats].values,pr.ylog.values)
            hi=HistGradientBoostingRegressor(loss="quantile",quantile=0.9,**D.HGB).fit(pr[feats].values,pr.ylog.values)
            E=np.maximum(lo.predict(ca[feats].values)-ca.ylog.values,ca.ylog.values-hi.predict(ca[feats].values))
            n=len(E); Q=np.quantile(E,min(1,math.ceil((n+1)*0.8)/n),method="higher")
            l,h=lo.predict(te[feats].values)-Q,hi.predict(te[feats].values)+Q
            y=te.ylog.values
            c=((y>=l)&(y<=h)); cov.append(c)
            go=l>=D.T70; ng=h<D.T70
            st.append((go.sum(),ng.sum(),(~go&~ng).sum(),((y>=D.T70)&ng).sum(),((y>=D.T70)).sum(),(go&(y>=D.T70)).sum(),(go&(y<D.T20)).sum(),len(y)))
            grp_cov.append(c.mean())
        cov=np.concatenate(cov); s=np.array(st).sum(axis=0)
        rows.append(dict(fs=fs,seed=seed,pooled_cov=cov.mean(),groups_below_0p70=int((np.array(grp_cov)<0.70).sum()),
            go=s[0],nogo=s[1],measure=s[2],pos_in_nogo=s[3],pos=s[4],go_tp=s[5],go_fa=s[6]))
        print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv("/content/reproduced/cqr_calibration_seed_check.csv",index=False)
print("DONE")
