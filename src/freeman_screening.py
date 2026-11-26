import math, numpy as np, pandas as pd, warnings
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import decision_layer as D
warnings.filterwarnings("ignore")
OUT=D.OUT
# Bolaji et al. 2021 (Heliyon 7:e06281) Table 1, Freeman field, offshore Niger Delta, mg/L
fw=pd.DataFrame({
 "sample":range(1,10),
 "Ca":[368.74,657.32,545.09,432.86,561.12,737.47,689.38,416.83,601.20],
 "Mg":[797.04,233.28,349.92,806.76,1117.80,330.48,408.24,330.48,121.50],
 "Na":[4450.53,7742.67,13328.00,7974.67,10772.00,7561.33,7686.67,3748.93,3576.33],
 "K":[53.47,55.00,61.13,46.67,135.67,112.27,97.40,38.93,43.47],
 "Cl":[9712.67,12956.49,12043.30,13313.82,18761.21,15672.24,13480.58,7693.72,7211.32],
 "SO4":[400,1100,600,1000,2000,1300,600,400,500],
 "HCO3":[1415.20,1268.80,1415.20,1122.40,927.20,2147.20,1512.80,1756.80,1464.00],
 "NO3":[7.20,18.10,17.50,6.10,19.10,6.70,16.80,12.00,12.70],
 "TDS":[17671.43,23984.00,22246.00,23568.00,31761.00,28430.00,24043.00,14617.00,14537.00]})
# charge balance (meq/L)
cat=fw.Ca/20.039+fw.Mg/12.153+fw.Na/22.990+fw.K/39.098
an=fw.Cl/35.45+fw.SO4/48.03+fw.HCO3/61.017+fw.NO3/62.005
fw["CB_pct"]=(cat-an)/(cat+an)*100
print(fw[["sample","TDS","CB_pct"]].round(1).to_string(index=False))
m=lambda e,w:fw[e]/w
fw["lnTDS10"]=np.log10(fw.TDS)
fw["lnNaCl"]=np.log(m("Na",22.990)/m("Cl",35.45)); fw["lnCaCl"]=np.log(m("Ca",40.078)/m("Cl",35.45))
fw["lnMgCl"]=np.log(m("Mg",24.305)/m("Cl",35.45)); fw["lnCaMg"]=np.log(m("Ca",40.078)/m("Mg",24.305))
d=D.load()
tr=d.copy(); X=tr[D.F1].values; Xn=fw[D.F1].values
# domain check: percentile of each Freeman feature within training, Mahalanobis distance
for f in D.F1:
    pr=[(tr[f]<v).mean() for v in fw[f]]
    print(f"{f}: training range {tr[f].min():.2f}..{tr[f].max():.2f}; Freeman percentiles {min(pr):.2f}-{max(pr):.2f}")
mu=X.mean(0); S=np.linalg.inv(np.cov(X.T))
md=lambda A: np.sqrt(np.einsum('ij,jk,ik->i',A-mu,S,A-mu))
dtr=md(X); dfw=md(Xn)
print("Mahalanobis: training median %.2f, 95th pct %.2f; Freeman %s"%(np.median(dtr),np.percentile(dtr,95),np.round(dfw,2)))
# models trained on all 4,639 rows
res=pd.DataFrame({"sample":fw["sample"]})
for thr,tag in [(D.T70,"70"),(D.T150,"150")]:
    y=(tr.ylog>=thr).astype(int).values
    lg=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=2000)).fit(X,y); res[f"p{tag}_logit"]=lg.predict_proba(Xn)[:,1]
    hg=HistGradientBoostingClassifier(**D.HGB).fit(X,y); res[f"p{tag}_hgb"]=hg.predict_proba(Xn)[:,1]
for q in (0.1,0.5,0.9):
    r=HistGradientBoostingRegressor(loss="quantile",quantile=q,**D.HGB).fit(X,tr.ylog.values); res[f"q{int(q*100)}"]=r.predict(Xn)
bc=tr.BASIN.value_counts(); cand=sorted(bc[bc>=30].index); n_cal=max(3,int(round(.25*len(cand))))
los=[];his=[]
for s in range(5):
    cg=list(np.random.default_rng(1000+s).choice(cand,n_cal,replace=False))
    pr_,ca=tr[~tr.BASIN.isin(cg)],tr[tr.BASIN.isin(cg)]
    lo=HistGradientBoostingRegressor(loss="quantile",quantile=0.1,**D.HGB).fit(pr_[D.F1].values,pr_.ylog.values)
    hi=HistGradientBoostingRegressor(loss="quantile",quantile=0.9,**D.HGB).fit(pr_[D.F1].values,pr_.ylog.values)
    E=np.maximum(lo.predict(ca[D.F1].values)-ca.ylog.values,ca.ylog.values-hi.predict(ca[D.F1].values))
    n=len(E);Q=np.quantile(E,min(1,math.ceil((n+1)*.8)/n),method="higher")
    los.append(lo.predict(Xn)-Q);his.append(hi.predict(Xn)+Q)
los=np.array(los);his=np.array(his)
res["cqr_lo_min"]=los.min(0);res["cqr_lo_max"]=los.max(0);res["cqr_hi_min"]=his.min(0);res["cqr_hi_max"]=his.max(0)
st=[]
for i in range(9):
    s=[("Go" if los[k,i]>=D.T70 else "NoGo" if his[k,i]<D.T70 else "Measure") for k in range(5)]
    st.append("/".join(sorted(set(s))) + f" ({s.count('NoGo')} NoGo, {s.count('Measure')} Measure, {s.count('Go')} Go of 5)")
res["state_at_70"]=st
res["Li_q50_mgL"]=10**res.q50; res["Li_lo_mgL"]=10**res.cqr_lo_min; res["Li_hi_mgL"]=10**res.cqr_hi_max
print(res[["sample","p70_logit","p70_hgb","p150_hgb","Li_q50_mgL","Li_lo_mgL","Li_hi_mgL","state_at_70"]].round(3).to_string(index=False))
fw.merge(res,on="sample").to_csv(f"{OUT}/freeman_field_screening.csv",index=False)
