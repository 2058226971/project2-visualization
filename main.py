"""Run the supplied data/model pipeline, then complete tasks.py to produce figures."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from tasks import plot_regions, plot_probabilities, plot_boundary_3d, plot_probability_3d


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,default=Path("outputs"));ap.add_argument("--completed",action="store_true");args=ap.parse_args();args.out.mkdir(exist_ok=True,parents=True)
    iris=load_iris();X,y=iris.data[:,2:4],iris.target
    train,val=train_test_split(np.arange(len(y)),test_size=.3,stratify=y,random_state=42)
    scaler=StandardScaler().fit(X[train])
    xx,yy=np.meshgrid(np.linspace(.6,7.3,160),np.linspace(-.1,2.8,120))
    grid=np.c_[xx.ravel(),yy.ravel()]
    todo=0
    jobs=[]
    for name,model in [("logistic",LogisticRegression(max_iter=500)),("forest",RandomForestClassifier(n_estimators=100,random_state=42))]:
        model.fit(scaler.transform(X[train]),y[train])
        p=model.predict_proba(scaler.transform(grid))
        region=model.classes_[p.argmax(1)].reshape(xx.shape)
        print(name,"validation accuracy",model.score(scaler.transform(X[val]),y[val]))
        jobs.extend([(name+"_regions",lambda ax,r=region:plot_regions(ax,xx,yy,r,X,y),"2d"),
                     (name+"_probabilities",lambda axes,p=p:plot_probabilities(axes,xx,yy,p.reshape(*xx.shape,-1),iris.target_names),"panels")])
    mask=y>0;X3=iris.data[mask][:,[1,2,3]];y3=y[mask]
    model3=LogisticRegression(max_iter=1000).fit(X3,y3)
    jobs.extend([("boundary_3d",lambda ax:plot_boundary_3d(ax,model3,X3,y3),"3d"),
                 ("probability_3d",lambda ax:plot_probability_3d(ax,model3,X3,y3),"3d")])
    for name,draw,kind in jobs:
        if kind=="panels":fig,ax=plt.subplots(1,3,figsize=(12,4),layout="constrained")
        elif kind=="3d":fig=plt.figure(figsize=(8,6));ax=fig.add_subplot(projection="3d")
        else:fig,ax=plt.subplots(figsize=(7,5),layout="constrained")
        try:draw(ax);fig.savefig(args.out/(name+".png"),dpi=160);print("SAVED",name)
        except NotImplementedError as exc:todo+=1;print("TODO",exc)
        finally:plt.close(fig)
    print("Unfinished plotting calls:",todo)
    raise SystemExit(1 if args.completed and todo else 0)

if __name__=="__main__":main()
