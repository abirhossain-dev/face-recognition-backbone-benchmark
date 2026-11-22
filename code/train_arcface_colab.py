"""Train 4 backbones with an ArcFace head on a CelebA subset (identity labels), then evaluate on LFW pairs
(10-fold, threshold chosen on 9 folds). No labels at inference. Logs everything to train_results.json."""
import json, math, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torchvision, torchvision.models as M
from PIL import Image
from sklearn.datasets import fetch_lfw_pairs
from sklearn.metrics import roc_auc_score, roc_curve
torch.manual_seed(0); np.random.seed(0)
DEV="cuda" if torch.cuda.is_available() else "cpu"; S=128
EP=int(__import__("os").environ.get("EP",8))
ds=torchvision.datasets.CelebA(root="data",split="train",target_type="identity",download=True)
ids=ds.identity.squeeze().numpy(); fn=ds.filename
import os, collections
cache="celeba_sub.npz"
if os.path.exists(cache):
    z=np.load(cache); X,Y=z["X"],z["Y"]
else:
    cnt=collections.Counter(ids.tolist()); keep=[i for i,c in cnt.items() if c>=15]
    rng=np.random.default_rng(0); keep=set(rng.choice(keep,size=min(2500,len(keep)),replace=False).tolist())
    sel=[i for i in range(len(ids)) if ids[i] in keep]
    X=np.zeros((len(sel),S,S,3),np.uint8); Y=np.zeros(len(sel),np.int64); remap={}
    for j,i in enumerate(sel):
        im=Image.open(os.path.join("data/celeba/img_align_celeba",fn[i])).convert("RGB").crop((9,29,169,189)).resize((S,S))
        X[j]=np.asarray(im); Y[j]=remap.setdefault(int(ids[i]),len(remap))
    np.savez(cache,X=X,Y=Y)
NC=int(Y.max())+1; print("train images",len(X),"identities",NC,"device",DEV,flush=True)
mean=torch.tensor([0.485,0.456,0.406],device=DEV).view(1,3,1,1); std=torch.tensor([0.229,0.224,0.225],device=DEV).view(1,3,1,1)
def prep(xu8):  # uint8 NHWC numpy -> normalised NCHW on device
    t=torch.from_numpy(xu8).to(DEV).float().permute(0,3,1,2)/255.0
    if t.shape[-1]!=S: t=F.interpolate(t,size=(S,S),mode="bilinear",align_corners=False)
    return (t-mean)/std
class Arc(nn.Module):
    def __init__(s,d,n,sc=30.,m=0.5):
        super().__init__(); s.W=nn.Parameter(torch.randn(n,d)*0.01); s.s=sc; s.m=m
    def forward(s,e,y):
        c=F.linear(F.normalize(e),F.normalize(s.W)).clamp(-1+1e-7,1-1e-7)
        th=torch.acos(c); tgt=torch.cos(th+s.m); oh=F.one_hot(y,c.shape[1]).bool()
        return s.s*torch.where(oh,tgt,c)
def backbone(name):
    if name=="resnet50": m=M.resnet50(weights="IMAGENET1K_V1"); d=2048; m.fc=nn.Identity(); return m,d
    if name=="densenet121": m=M.densenet121(weights="IMAGENET1K_V1"); d=1024; m.classifier=nn.Identity(); return m,d
    if name=="mobilenet_v2": m=M.mobilenet_v2(weights="IMAGENET1K_V1"); d=1280; m.classifier=nn.Identity(); return m,d
    if name=="efficientnet_b0": m=M.efficientnet_b0(weights="IMAGENET1K_V1"); d=1280; m.classifier=nn.Identity(); return m,d
data=fetch_lfw_pairs(subset="10_folds",color=True,resize=1.0); pairs,yp=data.pairs,data.target
@torch.no_grad()
def embed(net,imgs,bs=128):
    net.eval(); out=[]
    for i in range(0,len(imgs),bs):
        x=imgs[i:i+bs]; x=x if x.max()>1.5 else (x*255).astype(np.uint8)
        out.append(F.normalize(net(prep(x.astype(np.uint8))),dim=1).cpu())
    return torch.cat(out).numpy()
def cv_acc(sim,y,folds=10):
    n=len(y)//folds; accs=[]
    for k in range(folds):
        te=np.arange(k*n,(k+1)*n); tr=np.setdiff1d(np.arange(len(y)),te)
        ths=np.unique(sim[tr]); best=max(ths[::5],key=lambda t:((sim[tr]>=t)==y[tr]).mean())
        accs.append(((sim[te]>=best)==y[te]).mean())
    return float(np.mean(accs)),float(np.std(accs))
def evaluate(net):
    a=embed(net,pairs[:,0]); b=embed(net,pairs[:,1]); sim=(a*b).sum(1); acc,sd=cv_acc(sim,yp)
    fpr,tpr,_=roc_curve(yp,sim); rng=np.random.default_rng(0)
    au=[roc_auc_score(yp[i],sim[i]) for i in (rng.integers(0,len(yp),len(yp)) for _ in range(200))]
    return dict(acc_10fold=round(acc,4),acc_std=round(sd,4),auc=round(float(roc_auc_score(yp,sim)),4),auc_ci95=[round(float(np.percentile(au,2.5)),4),round(float(np.percentile(au,97.5)),4)],
        tar_far_1e1=round(float(np.interp(0.1,fpr,tpr)),4),tar_far_1e2=round(float(np.interp(0.01,fpr,tpr)),4))
res={}
for name in ["mobilenet_v2","efficientnet_b0","densenet121","resnet50"]:
    bb,d=backbone(name); net=nn.Sequential(bb,nn.Linear(d,512),nn.BatchNorm1d(512)).to(DEV); head=Arc(512,NC).to(DEV)
    opt=torch.optim.AdamW([{"params":net.parameters(),"lr":3e-4},{"params":head.parameters(),"lr":3e-3}],weight_decay=1e-4)
    steps=EP*math.ceil(len(X)/64); sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=[3e-4,3e-3],total_steps=steps)
    t0=time.time(); r=dict(params_M=round(sum(p.numel() for p in net.parameters())/1e6,2),epochs=EP)
    for ep in range(EP):
        net.train(); perm=np.random.permutation(len(X)); tl=0; n=0
        for i in range(0,len(X),64):
            idx=np.sort(perm[i:i+64]); x=prep(X[idx]); 
            if np.random.rand()<0.5: x=x.flip(3)
            y=torch.from_numpy(Y[idx]).to(DEV); loss=F.cross_entropy(head(net(x),y),y)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step(); tl+=loss.item(); n+=1
        print(name,"epoch",ep+1,"loss",round(tl/n,3),flush=True)
    r["train_minutes"]=round((time.time()-t0)/60,1); r.update(evaluate(net))
    t1=time.time(); embed(net,pairs[:500,0]); r["ms_per_img_gpu"]=round(1000*(time.time()-t1)/500,2)
    res[name]=r; print(name,r,flush=True); json.dump(res,open("train_results.json","w"),indent=2)
print("DONE")
