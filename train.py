import json,os,glob,argparse
import numpy as np

def load_data(data_dir):
    files=glob.glob(f"{data_dir}/session_*.json")
    if not files:raise FileNotFoundError(f"No sessions in {data_dir}")
    X,Y=[],[]
    for f in files:
        sess=json.load(open(f))
        for s in sess["samples"]:
            kp=s["keypoints"]
            if len(s["csi"])>0 and len(kp)==51:
                X.append(s["csi"]);Y.append(kp)
    print(f"[DATA] {len(X)} samples from {len(files)} sessions")
    return np.array(X,dtype=np.float32),np.array(Y,dtype=np.float32)

def relu(x):return np.maximum(0,x)
def relu_g(x):return(x>0).astype(np.float32)

class MLP:
    def __init__(self,ind,lr=0.001):
        self.lr=lr
        self.W1=np.random.randn(ind,256)*np.sqrt(2/ind)
        self.b1=np.zeros(256)
        self.W2=np.random.randn(256,128)*np.sqrt(2/256)
        self.b2=np.zeros(128)
        self.W3=np.random.randn(128,51)*np.sqrt(2/128)
        self.b3=np.zeros(51)
        self.m={k:np.zeros_like(v) for k,v in self._p()}
        self.v={k:np.zeros_like(v) for k,v in self._p()}
        self.t=0
    def _p(self):
        return[("W1",self.W1),("b1",self.b1),
               ("W2",self.W2),("b2",self.b2),
               ("W3",self.W3),("b3",self.b3)]
    def fwd(self,X):
        self.z1=X@self.W1+self.b1;self.a1=relu(self.z1)
        self.z2=self.a1@self.W2+self.b2;self.a2=relu(self.z2)
        self.out=self.a2@self.W3+self.b3;return self.out
    def bwd(self,X,Y):
        n=X.shape[0];dL=2*(self.out-Y)/n
        dW3=self.a2.T@dL;db3=dL.sum(0)
        da2=dL@self.W3.T*relu_g(self.z2)
        dW2=self.a1.T@da2;db2=da2.sum(0)
        da1=da2@self.W2.T*relu_g(self.z1)
        dW1=X.T@da1;db1=da1.sum(0)
        self._adam({"W1":dW1,"b1":db1,"W2":dW2,"b2":db2,"W3":dW3,"b3":db3})
    def _adam(self,g,b1=0.9,b2=0.999,eps=1e-8):
        self.t+=1
        for n,p in[("W1",self.W1),("b1",self.b1),
                   ("W2",self.W2),("b2",self.b2),
                   ("W3",self.W3),("b3",self.b3)]:
            self.m[n]=b1*self.m[n]+(1-b1)*g[n]
            self.v[n]=b2*self.v[n]+(1-b2)*g[n]**2
            mh=self.m[n]/(1-b1**self.t)
            vh=self.v[n]/(1-b2**self.t)
            p-=self.lr*mh/(np.sqrt(vh)+eps)
    def loss(self,yp,yt):return float(np.mean((yp-yt)**2))
    def save(self,path,Xm,Xs,Ym,Ys):
        np.savez(path,W1=self.W1,b1=self.b1,W2=self.W2,b2=self.b2,
                 W3=self.W3,b3=self.b3,
                 X_mean=Xm,X_std=Xs,Y_mean=Ym,Y_std=Ys)
        print(f"[SAVE] {path}.npz")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",default="training-data")
    ap.add_argument("--out",default="model")
    ap.add_argument("--epochs",type=int,default=200)
    ap.add_argument("--batch",type=int,default=32)
    ap.add_argument("--lr",type=float,default=0.001)
    args=ap.parse_args()
    np.random.seed(42)
    X,Y=load_data(args.data)
    Xm=X.mean(0);Xs=X.std(0)+1e-8
    Ym=Y.mean(0);Ys=Y.std(0)+1e-8
    X=(X-Xm)/Xs;Y=(Y-Ym)/Ys
    n_val=max(1,int(len(X)*0.15))
    idx=np.random.permutation(len(X))
    Xv,Yv=X[idx[:n_val]],Y[idx[:n_val]]
    Xt,Yt=X[idx[n_val:]],Y[idx[n_val:]]
    model=MLP(Xt.shape[1],args.lr)
    best=float("inf")
    print(f"[TRAIN] {len(Xt)} train  {len(Xv)} val  epochs={args.epochs}")
    for ep in range(1,args.epochs+1):
        p=np.random.permutation(len(Xt));Xt=Xt[p];Yt=Yt[p]
        losses=[]
        for i in range(0,len(Xt),args.batch):
            xb=Xt[i:i+args.batch];yb=Yt[i:i+args.batch]
            losses.append(model.loss(model.fwd(xb),yb))
            model.bwd(xb,yb)
        vl=model.loss(model.fwd(Xv),Yv)
        tl=float(np.mean(losses))
        if vl<best:best=vl;model.save(args.out,Xm,Xs,Ym,Ys)
        if ep%20==0 or ep==1:
            bar="█"*int(20*ep/args.epochs)+"░"*(20-int(20*ep/args.epochs))
            print(f"  [{bar}] {ep:4d}/{args.epochs}  loss={tl:.4f}  val={vl:.4f}  best={best:.4f}")
    print(f"[DONE] Best val={best:.4f}  saved to {args.out}.npz")

if __name__=="__main__":
    main()
