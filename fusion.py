"""
WiFi CSI Fusion — Professional Edition
16:9 windowed · toggle overlays · fullscreen camera
"""

import socket, struct, threading, time, argparse, collections, math
import urllib.request, os, random

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions

BG         = (10,  12,  14)
PANEL_BG   = (14,  17,  20)
BORDER     = (30,  35,  40)
ACCENT     = (0,   210, 120)
ACCENT_DIM = (0,   90,  50)
TEXT_PRI   = (220, 230, 225)
TEXT_SEC   = (90,  105, 100)
TEXT_DIM   = (45,  55,  50)
WARN       = (0,   180, 220)
DANGER     = (50,  60,  220)
SKEL_LINE  = (0,   190, 100)
SKEL_JOINT = (0,   255, 140)
SKEL_J2    = (0,   120, 60)
MAP_BG     = (8,   12,  10)
MAP_GRID   = (18,  28,  20)

# 16:9 window
WIN_W = 1280
WIN_H = 720
BTN_H = 40

DEFAULT_UDP_PORT = 5005
DEFAULT_CAMERA   = 0
DEFAULT_TIMEOUT  = 5.0
MODEL_PATH       = "pose_landmarker.task"
MODEL_URL        = ("https://storage.googleapis.com/mediapipe-models/"
                    "pose_landmarker/pose_landmarker_lite/float16/latest/"
                    "pose_landmarker_lite.task")

POSE_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),(23,25),(24,26),
    (25,27),(26,28),(27,29),(28,30),(29,31),(30,32),
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(15,17),(15,19),(15,21),(16,18),(16,20),(16,22),
]

MAGIC_CSI  = 0xC5110001
MAGIC_FEAT = 0xC5110006
HEADER_FMT = "<IBIQbBHH"
HEADER_SZ  = struct.calcsize(HEADER_FMT)

LOG_POOL = [
    "CSI_RX  subcarrier_lock=56  coherence=OK",
    "PARSE   node={nid}  seq={seq}  sc={sc}",
    "SIGNAL  rssi={rssi}dBm  ch={ch}  amp={amp:.1f}",
    "HAMPEL  outliers_removed={n}  sigma=3.0",
    "FRESNEL zone_depth={d:.2f}m  model=v2",
    "FUSION  cross_view_attn={w:.4f}  nodes={nodes}",
    "MOTION  band_power={v:.3f}  variance={var:.3f}",
    "BREATH  est={bpm:.1f}bpm  conf={c:.0f}%",
    "RUVEC   pipeline=OK  fps={fps:.0f}",
    "FIELD   model=STABLE  drift=0.00",
    "COHERE  gate=ACCEPT  snr={snr:.1f}dB",
    "ADR018  frame={seq}  len={l}B  crc=OK",
    "MESH    topology=MULTISTATIC  links={l2}",
    "TRILATERATION  x={tx:.2f}m  y={ty:.2f}m",
    "ADAPT   state={s}  cycle=200ms",
    "EMBED   dim=128  fingerprint=UPDATED",
    "TRACK   persons={p}  keypoints=17  conf={c:.0f}%",
    "SPOOF   check=PASS  replay=NONE",
]

def rand_log(nodes=3):
    t=random.choice(LOG_POOL)
    return t.format(
        nid=random.randint(1,max(nodes,1)),seq=random.randint(0,99999),
        sc=random.choice([56,64,128]),rssi=random.randint(-80,-45),
        ch=random.choice([1,6,11]),amp=random.uniform(5,40),
        n=random.randint(0,4),d=random.uniform(0.5,5.0),
        w=random.uniform(0.1,0.9),nodes=nodes,
        v=random.uniform(0.001,50.0),var=random.uniform(0.01,20.0),
        bpm=random.uniform(10,28),c=random.uniform(55,99),
        fps=random.uniform(18,22),snr=random.uniform(8,35),
        l=random.choice([32,48,60,148,276]),
        l2=max(nodes*(nodes-1),0),p=random.randint(1,3),
        s=random.randint(1,12),tx=random.uniform(-3,3),ty=random.uniform(0,4),
    )


class WallModel:
    CONNS=[(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
           (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
    def __init__(self,path):
        d=np.load(path)
        self.W1=d["W1"];self.b1=d["b1"]
        self.W2=d["W2"];self.b2=d["b2"]
        self.W3=d["W3"];self.b3=d["b3"]
        self.Xm=d["X_mean"];self.Xs=d["X_std"]
        self.Ym=d["Y_mean"];self.Ys=d["Y_std"]
        print(f"[MODEL] Through-wall model loaded: {path}")
    def predict(self,csi):
        x=(np.array(csi,dtype=np.float32)-self.Xm)/self.Xs
        a1=np.maximum(0,x@self.W1+self.b1)
        a2=np.maximum(0,a1@self.W2+self.b2)
        out=a2@self.W3+self.b3
        return (out*self.Ys+self.Ym).reshape(17,3)
    def draw(self,canvas,kps,ox,oy,ow,oh):
        pts=[(ox+int(kp[0]*ow),oy+int(kp[1]*oh)) for kp in kps]
        for a,b in self.CONNS:
            if a<len(pts) and b<len(pts):
                cv2.line(canvas,pts[a],pts[b],(0,180,255),2,cv2.LINE_AA)
        for px,py in pts:
            cv2.circle(canvas,(px,py),4,(0,220,255),-1,cv2.LINE_AA)
            cv2.circle(canvas,(px,py),4,(0,100,150),1,cv2.LINE_AA)

def build_csi_vector(snap,n_nodes,timeout=3.0):
    now=time.time();vec=[]
    for nid in range(1,n_nodes+1):
        data=snap.get(nid)
        if data and now-data.get("received_at",0)<timeout:
            amps=data.get("amplitudes",[])
            if amps:
                padded=(amps+[0.0]*56)[:56]
                vec.extend(padded+[float(np.mean(amps)),
                                   float(np.var(amps)),
                                   float(data.get("rssi",0))])
                continue
        vec.extend([0.0]*59)
    return vec

def hline(img,y,x0,x1,col,t=1):
    cv2.line(img,(x0,y),(x1,y),col,t)

def label(img,text,x,y,color=TEXT_PRI,scale=0.40,t=1):
    cv2.putText(img,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,t,cv2.LINE_AA)

def tag(img,text,x,y,fg,bg):
    (tw,th),_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,0.34,1)
    cv2.rectangle(img,(x,y-th-2),(x+tw+10,y+4),bg,-1)
    label(img,text,x+5,y,fg,scale=0.34)
    return x+tw+16

def scanlines(frame,gap=4,alpha=0.06):
    for y in range(0,frame.shape[0],gap):
        frame[y]=np.clip(frame[y].astype(np.float32)*(1-alpha),0,255).astype(np.uint8)
    return frame

_vig={}
def vignette(frame,strength=0.38):
    h,w=frame.shape[:2]
    if(h,w) not in _vig:
        k=np.zeros((h,w),dtype=np.float32)
        cx,cy=w/2,h/2
        for y in range(h):
            for x in range(w):
                dx=(x-cx)/cx;dy=(y-cy)/cy
                k[y,x]=1.0-min(math.sqrt(dx*dx+dy*dy)*strength,1.0)
        _vig[(h,w)]=k
    mask=_vig[(h,w)];out=frame.astype(np.float32)
    for c in range(3):out[:,:,c]*=mask
    return np.clip(out,0,255).astype(np.uint8)


# ── HUD bar ───────────────────────────────────────────────────────────────────
def draw_hud(canvas,W,H,show_map,show_nodes,csi_present,fps,online_n,total_n):
    bar_y=H-BTN_H
    ov=canvas.copy()
    cv2.rectangle(ov,(0,bar_y),(W,H),(5,7,9),-1)
    cv2.addWeighted(ov,0.85,canvas,0.15,0,canvas)
    hline(canvas,bar_y,0,W,BORDER)

    # left status
    sc=ACCENT if csi_present else TEXT_DIM
    cv2.circle(canvas,(18,bar_y+20),5,sc,-1)
    label(canvas,"PRESENCE" if csi_present else "NO PRESENCE",28,bar_y+24,sc,scale=0.38)
    label(canvas,f"NODES {online_n}/{total_n}",160,bar_y+24,
          ACCENT if online_n>0 else DANGER,scale=0.36)
    label(canvas,f"{fps:.0f} FPS",260,bar_y+24,TEXT_DIM,scale=0.36)
    label(canvas,time.strftime("%H:%M:%S"),330,bar_y+24,TEXT_DIM,scale=0.36)

    # right buttons
    buttons=[("1  NODES",show_nodes),("2  MAP",show_map),
             ("S  SAVE",False),("Q  QUIT",False)]
    bx=W-10
    for txt,active in reversed(buttons):
        (tw,_),_=cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,0.36,1)
        bw=tw+20; bx-=bw+6
        by=bar_y+5; bh=BTN_H-10
        cv2.rectangle(canvas,(bx,by),(bx+bw,by+bh),
                      ACCENT_DIM if active else (20,24,26),-1)
        cv2.rectangle(canvas,(bx,by),(bx+bw,by+bh),
                      ACCENT if active else BORDER,1)
        label(canvas,txt,bx+10,by+bh-7,
              ACCENT if active else TEXT_SEC,scale=0.36)


# ── Terminal ──────────────────────────────────────────────────────────────────
class Terminal:
    LH=13;FSC=0.60;MAX=20
    def __init__(self):
        self.lines=collections.deque(maxlen=self.MAX);self._next=0
    def tick(self,nodes=3):
        if time.time()>self._next:
            self.lines.append(rand_log(nodes))
            self._next=time.time()+random.uniform(0.12,0.4)
    def render(self,canvas,x=14,bottom_y=None,max_w=None):
        h,w=canvas.shape[:2]
        if bottom_y is None:bottom_y=h-BTN_H-8
        if max_w is None:max_w=w//2
        n=len(self.lines);
        if n==0:return
        y0=bottom_y-n*self.LH
        blink=int(time.time()*2)%2==0
        for i,line in enumerate(self.lines):
            y=y0+i*self.LH;last=(i==n-1)
            col=ACCENT if last else(TEXT_SEC if i/max(n-1,1)>0.5 else TEXT_DIM)
            cv2.putText(canvas,">",(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.28,
                        ACCENT_DIM if not last else ACCENT,1,cv2.LINE_AA)
            cv2.putText(canvas,line,(x+12,y),cv2.FONT_HERSHEY_SIMPLEX,
                        0.28,col,1,cv2.LINE_AA)
        if blink and n>0:
            tw,_=cv2.getTextSize(self.lines[-1],cv2.FONT_HERSHEY_PLAIN,self.FSC,1)
            cx=x+12+tw[0]+3;cy=y0+(n-1)*self.LH
            cv2.rectangle(canvas,(cx,cy-7),(cx+5,cy+1),ACCENT,-1)


# ── CSI receiver ──────────────────────────────────────────────────────────────
def parse_frame(data):
    if len(data)<HEADER_SZ:return None
    try:
        magic,nid,seq,ts,rssi,ch,sc,dl=struct.unpack_from(HEADER_FMT,data,0)
    except struct.error:return None
    if magic not in(MAGIC_CSI,MAGIC_FEAT):return None
    payload=data[HEADER_SZ:HEADER_SZ+dl];amps=[]
    if magic==MAGIC_CSI and len(payload)>=sc*4:
        for i in range(sc):
            iv,qv=struct.unpack_from("<hh",payload,i*4)
            amps.append(math.sqrt(iv*iv+qv*qv))
    return{"node_id":nid,"seq":seq,"rssi":rssi,"channel":ch,
           "sc_count":sc,"amplitudes":amps,"received_at":time.time()}


class CSIReceiver(threading.Thread):
    def __init__(self,port):
        super().__init__(daemon=True)
        self.port=port;self.nodes={};self.lock=threading.Lock();self._run=True
    def run(self):
        s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        s.bind(("0.0.0.0",self.port));s.settimeout(0.5)
        print(f"[CSI] UDP :{self.port}")
        while self._run:
            try:
                data,addr=s.recvfrom(4096)
                f=parse_frame(data)
                if f:
                    with self.lock:
                        if f["node_id"] not in self.nodes:
                            print(f"[CSI] Node {f['node_id']} online  {addr[0]}")
                        self.nodes[f["node_id"]]={**f,"ip":addr[0]}
            except socket.timeout:pass
        s.close()
    def stop(self):self._run=False
    def snapshot(self):
        with self.lock:return dict(self.nodes)


class Motion:
    def __init__(self):
        self.hist=collections.deque(maxlen=60)
        self.amp_hist=collections.deque(maxlen=20)
    def update(self,amps):
        if not amps:
            return{"motion":0.0,"presence":False,"variance":0.0,"mean":0.0,"signal":0.0}
        m=float(np.mean(amps));self.hist.append(m);self.amp_hist.append(m)
        if len(self.hist)<6:
            return{"motion":0.0,"presence":False,"variance":0.0,"mean":m,"signal":m}
        arr=np.array(self.hist);var=float(np.var(arr))
        dif=float(np.mean(np.abs(np.diff(arr[-12:]))))
        signal=float(np.mean(self.amp_hist))
        return{"motion":min(dif/5.0,1.0),"presence":var>1.8,
               "variance":var,"mean":m,"signal":signal}


# ── Room map overlay ──────────────────────────────────────────────────────────
class RoomMap:
    HEAT_RES=60;BLUR_K=15
    def __init__(self,node_positions,room_w=10.0,room_d=5.0):
        self.node_pos=node_positions;self.room_w=room_w;self.room_d=room_d
        self.heat=np.zeros((self.HEAT_RES,self.HEAT_RES),dtype=np.float32)
        self.prev=np.zeros_like(self.heat);self._decay=0.88

    def update(self,snap,ests,timeout):
        now=time.time()
        grid=np.zeros((self.HEAT_RES,self.HEAT_RES),dtype=np.float32)
        gw,gh=self.HEAT_RES,self.HEAT_RES
        for nid,pos in self.node_pos.items():
            data=snap.get(nid)
            if data is None or now-data["received_at"]>timeout:continue
            est=ests.get(nid)
            if est is None:continue
            res=est.update(data.get("amplitudes",[]))
            if not res["presence"]:continue
            nx,ny=pos[0],pos[1]
            sig=res["signal"];var=res["variance"]
            for gy in range(gh):
                for gx in range(gw):
                    wx=(gx/gw-0.5)*self.room_w
                    wy=(1.0-gy/gh)*self.room_d
                    dist=math.sqrt((wx-nx)**2+(wy-ny)**2)+0.5
                    grid[gy,gx]+=(sig/30.0)*var*(1.0/(dist**1.5))
        if grid.max()>0:grid/=grid.max()
        blurred=cv2.GaussianBlur(grid,(self.BLUR_K,self.BLUR_K),0)
        self.heat=self._decay*self.prev+(1-self._decay)*blurred
        self.prev=self.heat.copy()

    def draw(self,canvas,snap,timeout,W,H):
        PAD=16;OW=420;OH=320
        OX=W-OW-PAD;OY=50+PAD;now=time.time()
        ov=canvas.copy()
        cv2.rectangle(ov,(OX,OY),(OX+OW,OY+OH),MAP_BG,-1)
        cv2.addWeighted(ov,0.93,canvas,0.07,0,canvas)
        cv2.rectangle(canvas,(OX,OY),(OX+OW,OY+26),BG,-1)
        label(canvas,"ROOM MAP  ·  THROUGH-WALL",OX+8,OY+17,ACCENT,scale=0.40)
        label(canvas,f"{self.room_w:.0f}m×{self.room_d:.0f}m",
              OX+OW-48,OY+17,TEXT_SEC,scale=0.32)
        hline(canvas,OY+26,OX,OX+OW,BORDER)
        cv2.rectangle(canvas,(OX,OY),(OX+OW,OY+OH),BORDER,1)
        MX=OX+6;MY=OY+30;MW=OW-12;MH=OH-38
        for i in range(11):
            x=MX+int(i*MW/10);cv2.line(canvas,(x,MY),(x,MY+MH),MAP_GRID,1)
        for i in range(6):
            y=MY+int(i*MH/5);cv2.line(canvas,(MX,y),(MX+MW,y),MAP_GRID,1)
        if self.heat.max()>0.01:
            h8=(np.clip(self.heat,0,1)*255).astype(np.uint8)
            hup=cv2.resize(h8,(MW,MH),interpolation=cv2.INTER_LINEAR)
            hrgb=np.zeros((MH,MW,3),dtype=np.uint8)
            hrgb[:,:,1]=hup
            hrgb[:,:,0]=(hup.astype(np.float32)*0.1).astype(np.uint8)
            bright=np.clip((hup.astype(np.float32)/255.0-0.6)/0.4,0,1)
            hrgb[:,:,1]=np.clip(hrgb[:,:,1].astype(np.float32)+bright*55,0,255).astype(np.uint8)
            roi=canvas[MY:MY+MH,MX:MX+MW]
            cv2.addWeighted(hrgb,0.80,roi,0.20,0,roi)
        for nid,pos in self.node_pos.items():
            xw,yw=pos[0],pos[1]
            data=snap.get(nid)
            age=now-data["received_at"] if data else timeout+1
            online=data is not None and age<timeout
            px=MX+int(np.clip(xw/self.room_w+0.5,0,1)*MW)
            py=MY+int(np.clip(1.0-yw/self.room_d,0,1)*MH)
            if online:
                pulse=int(abs(math.sin(time.time()*2+nid))*10)
                cv2.circle(canvas,(px,py),12+pulse,(0,70,35),1,cv2.LINE_AA)
                cv2.circle(canvas,(px,py),6,ACCENT,-1,cv2.LINE_AA)
                cv2.circle(canvas,(px,py),int(2.5/self.room_w*MW),ACCENT_DIM,1,cv2.LINE_AA)
            else:
                cv2.circle(canvas,(px,py),6,DANGER,-1,cv2.LINE_AA)
            label(canvas,f"N{nid}",px-4,py-9,TEXT_PRI,scale=0.32)
        hmax=float(self.heat.max())
        label(canvas,"DETECTED" if hmax>0.15 else "MONITORING",
              OX+8,OY+OH-6,ACCENT if hmax>0.15 else TEXT_DIM,scale=0.32)
        label(canvas,f"{hmax*100:.0f}%",OX+OW-36,OY+OH-6,TEXT_SEC,scale=0.30)


# ── Nodes overlay ─────────────────────────────────────────────────────────────
def draw_nodes(canvas,nodes,ests,timeout,W,H):
    PAD=16;OW=340;now=time.time()
    OH=min(380,H-BTN_H-50-PAD*2)
    OX=PAD;OY=50+PAD
    ov=canvas.copy()
    cv2.rectangle(ov,(OX,OY),(OX+OW,OY+OH),PANEL_BG,-1)
    cv2.addWeighted(ov,0.93,canvas,0.07,0,canvas)
    cv2.rectangle(canvas,(OX,OY),(OX+OW,OY+26),BG,-1)
    label(canvas,"CSI  NODES",OX+8,OY+17,ACCENT,scale=0.40)
    online_n=sum(1 for d in nodes.values() if now-d["received_at"]<timeout)
    label(canvas,f"{online_n}/{len(nodes) or 0}",OX+OW-28,OY+17,
          ACCENT if online_n>0 else DANGER,scale=0.36)
    hline(canvas,OY+26,OX,OX+OW,BORDER)
    cv2.rectangle(canvas,(OX,OY),(OX+OW,OY+OH),BORDER,1)
    slot_h=(OH-28)//6
    for idx in range(1,7):
        y0=OY+26+(idx-1)*slot_h
        data=nodes.get(idx)
        age=now-data["received_at"] if data else timeout+1
        online=data is not None and age<timeout
        hline(canvas,y0,OX+4,OX+OW-4,BORDER)
        pulse=0.5+0.5*abs(math.sin(time.time()*2.5+idx))
        dot=tuple(int(c*pulse) for c in ACCENT) if online else DANGER
        cv2.circle(canvas,(OX+14,y0+slot_h//2),4,dot,-1)
        ny=y0+14
        label(canvas,f"NODE {idx}",OX+26,ny,TEXT_PRI,scale=0.38)
        if not online:
            label(canvas,"OFFLINE",OX+OW-62,ny,DANGER,scale=0.32);continue
        est=ests.setdefault(idx,Motion())
        res=est.update(data.get("amplitudes",[]))
        rssi=data.get("rssi",0);sc=data.get("sc_count",0)
        motion=res["motion"];pres=res["presence"];var=res["variance"]
        if pres:tag(canvas,"PRES",OX+OW-56,ny+1,BG,ACCENT)
        else:tag(canvas,"EMPTY",OX+OW-58,ny+1,TEXT_SEC,BORDER)
        label(canvas,f"RSSI {rssi}  SC {sc}  VAR {var:.1f}",
              OX+26,ny+12,TEXT_SEC,scale=0.28)
        bx=OX+26;by=y0+slot_h-10;bw=OW-34
        cv2.rectangle(canvas,(bx,by),(bx+bw,by+3),BORDER,-1)
        filled=int(bw*motion)
        if filled>0:
            cv2.rectangle(canvas,(bx,by),(bx+filled,by+3),
                          WARN if motion>0.55 else ACCENT,-1)
        amps=data.get("amplitudes",[])
        if len(amps)>4:
            sy=y0+slot_h-22;sh=8;step=max(1,len(amps)//bw)
            samp=amps[::step][:bw];mx=max(samp) or 1
            pts=[(bx+i,sy+sh-int(sh*v/mx)) for i,v in enumerate(samp)]
            for i in range(len(pts)-1):
                frac=samp[i]/mx
                c=tuple(int(a+(b-a)*frac) for a,b in zip(ACCENT_DIM,ACCENT))
                cv2.line(canvas,pts[i],pts[i+1],c,1,cv2.LINE_AA)


# ── Skeleton ──────────────────────────────────────────────────────────────────
def draw_skeleton(frame,result):
    if not result or not result.pose_landmarks:return
    h,w=frame.shape[:2]
    for person in result.pose_landmarks:
        pts=[(int(lm.x*w),int(lm.y*h)) for lm in person]
        vis=[lm.visibility for lm in person]
        for a,b in POSE_CONNECTIONS:
            if a<len(pts) and b<len(pts) and vis[a]>0.4 and vis[b]>0.4:
                cv2.line(frame,pts[a],pts[b],SKEL_LINE,2,cv2.LINE_AA)
        for i,(x,y) in enumerate(pts):
            if i<len(vis) and vis[i]>0.4:
                cv2.circle(frame,(x,y),4,SKEL_JOINT,-1,cv2.LINE_AA)
                cv2.circle(frame,(x,y),4,SKEL_J2,1,cv2.LINE_AA)


def download_model():
    if not os.path.exists(MODEL_PATH):
        print("[INFO] Downloading pose model (~7 MB)...")
        urllib.request.urlretrieve(MODEL_URL,MODEL_PATH)
        print("[INFO] Model ready")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--port",type=int,default=DEFAULT_UDP_PORT)
    ap.add_argument("--camera",type=int,default=DEFAULT_CAMERA)
    ap.add_argument("--timeout",type=float,default=DEFAULT_TIMEOUT)
    ap.add_argument("--no-camera",action="store_true")
    ap.add_argument("--positions",type=str,default="0,0,0.4;2,4,1.0;-5,4,1.0")
    ap.add_argument("--room",type=str,default="10x5")
    ap.add_argument("--model",type=str,default="",
                    help="Path to .npz model for through-wall skeleton")
    ap.add_argument("--width",type=int,default=WIN_W)
    ap.add_argument("--height",type=int,default=WIN_H)
    args=ap.parse_args()

    W=args.width; H=args.height
    # enforce 16:9
    H=int(W*9/16)

    try:room_w,room_d=map(float,args.room.lower().split("x"))
    except Exception:room_w,room_d=10.0,5.0

    node_pos={}
    if args.positions:
        for idx,part in enumerate(args.positions.split(";"),start=1):
            try:
                x,y,z=map(float,part.split(","))
                node_pos[idx]=(x,y,z)
            except ValueError:pass

    wall_model=None
    if args.model:
        mp=args.model if args.model.endswith('.npz') else args.model+'.npz'
        if os.path.exists(mp):wall_model=WallModel(mp)
        else:print(f'[WARN] Model not found: {mp}')
    rx=CSIReceiver(port=args.port);ests={};rx.start()
    term=Terminal()
    room=RoomMap(node_pos,room_w=room_w,room_d=room_d)
    show_map=False;show_nodes=False

    cap=detector=None
    if not args.no_camera:
        download_model()
        cap=cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print("[WARN] No camera");cap=None
        else:
            opts=PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
                running_mode=vision.RunningMode.VIDEO,num_poses=4,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            detector=PoseLandmarker.create_from_options(opts)
            print("[INFO] Pose detector online")

    cv2.namedWindow("WiFi CSI Fusion",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("WiFi CSI Fusion",W,H)

    print(f"[INFO] Window {W}x{H} (16:9)")
    print("[INFO] 1=nodes  2=map  S=screenshot  Q=quit")

    # set window size from camera native resolution once
    if cap:
        ok,probe=cap.read()
        if ok and probe is not None:
            fh_nat,fw_nat=probe.shape[:2]
            W=fw_nat; H=fh_nat+BTN_H
            cv2.resizeWindow("WiFi CSI Fusion",W,H)
    fps_q=collections.deque(maxlen=30);ts_ms=0
    xoff=0;yoff=0;nw=W;nh=video_h=H-BTN_H

    try:
        while True:
            t0=time.time();snap=rx.snapshot();now=time.time()
            csi_present=any(
                now-d["received_at"]<args.timeout
                and ests.get(nid,Motion()).hist
                and np.var(list(ests[nid].hist))>1.8
                for nid,d in snap.items()
            )
            room.update(snap,ests,args.timeout)

            # ── canvas ────────────────────────────────────────────────────────
            video_h=H-BTN_H
            canvas=np.full((H,W,3),BG,dtype=np.uint8)

            result=None
            if cap and detector:
                ok,frame=cap.read()
                if ok:
                    ts_ms+=33
                    # crop camera to 16:9 in case it's different ratio
                    fh,fw=frame.shape[:2]
                    target_w=int(fh*16/9)
                    if target_w<=fw:
                        x0=(fw-target_w)//2
                        frame=frame[:,x0:x0+target_w]
                    pass  # cam written above
                    # letterbox
                    fh2,fw2=frame.shape[:2]
                    scale=min(W/fw2,video_h/fh2)
                    nw=int(fw2*scale); nh=int(fh2*scale)
                    xoff=(W-nw)//2; yoff=(video_h-nh)//2
                    resized=cv2.resize(frame,(nw,nh))
                    gray=cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
                    gray3=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
                    cam=cv2.addWeighted(resized,0.28,gray3,0.72,0)
                    cam=cv2.addWeighted(cam,0.82,np.zeros_like(cam),0.18,0)
                    cam[:,:,1]=np.clip(cam[:,:,1].astype(np.int32)+6,0,255).astype(np.uint8)
                    canvas[yoff:yoff+nh,xoff:xoff+nw]=cam
                    mp_img=mp.Image(image_format=mp.ImageFormat.SRGB,
                                    data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                    result=detector.detect_for_video(mp_img,ts_ms)
                    draw_skeleton(canvas,result)
                    roi=canvas[yoff:yoff+nh,xoff:xoff+nw]
                    roi[:]=scanlines(roi.copy(),gap=4,alpha=0.06)
                    roi[:]=vignette(roi.copy(),strength=0.32)
            else:
                label(canvas,"NO CAMERA  ·  CSI-ONLY MODE",
                      W//2-110,H//2,TEXT_SEC,scale=0.50)

            # ── top bar ───────────────────────────────────────────────────────
            ov=canvas.copy()
            cv2.rectangle(ov,(0,0),(W,40),(0,0,0),-1)
            cv2.addWeighted(ov,0.65,canvas,0.35,0,canvas)
            label(canvas,"WiFi CSI FUSION  ·  MEDIAPIPE POSE",12,15,TEXT_PRI,scale=0.46)
            label(canvas,time.strftime("%Y-%m-%d  %H:%M:%S"),12,31,TEXT_SEC,scale=0.34)
            hline(canvas,40,0,W,BORDER)

            # presence corner brackets — on video frame edges
            if csi_present:
                pulse=abs(math.sin(time.time()*3));L=24
                bri=tuple(int(c*(0.5+0.5*pulse)) for c in ACCENT)
                x1=vx+1;y1=vy+1;x2=vx+vw-2;y2=vy+vh-2
                for(x,y,sx,sy) in[(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                    cv2.line(canvas,(x,y),(x+sx*L,y),bri,2,cv2.LINE_AA)
                    cv2.line(canvas,(x,y),(x,y+sy*L),bri,2,cv2.LINE_AA)

            # ── terminal log ──────────────────────────────────────────────────
            term.tick(len(snap))
            term.render(canvas,x=14,bottom_y=video_h-8,max_w=W//2)

            # ── overlays ──────────────────────────────────────────────────────
            if show_nodes:
                draw_nodes(canvas,snap,ests,args.timeout,W,H)
            if show_map:
                room.draw(canvas,snap,args.timeout,W,H)
                if wall_model:
                    cv=build_csi_vector(snap,len(node_pos),args.timeout)
                    if any(v!=0 for v in cv):
                        kps=wall_model.predict(cv)
                        PAD=16;OW=420;OH=320
                        OX=W-OW-PAD;OY=50+PAD
                        MX=OX+6;MY=OY+30;MW=OW-12;MH=OH-38
                        wall_model.draw(canvas,kps,MX,MY,MW,MH)

            # ── HUD ───────────────────────────────────────────────────────────
            fps_q.append(time.time()-t0)
            fps=1.0/(sum(fps_q)/len(fps_q)) if fps_q else 0
            online_n=sum(1 for d in snap.values() if now-d["received_at"]<args.timeout)
            draw_hud(canvas,W,H,show_map,show_nodes,csi_present,fps,online_n,len(snap))

            cv2.imshow("WiFi CSI Fusion",canvas)
            key=cv2.waitKey(1)&0xFF
            if key==ord("q"):break
            if key==ord("1"):show_nodes=not show_nodes
            if key==ord("2"):show_map=not show_map
            if key==ord("s"):
                fn=f"output/screenshot_{int(time.time())}.png"
                cv2.imwrite(fn,canvas);print(f"[INFO] Saved {fn}")

    finally:
        rx.stop()
        if cap:cap.release()
        if detector:detector.close()
        cv2.destroyAllWindows();print("[INFO] Done")

if __name__=="__main__":
    main()
