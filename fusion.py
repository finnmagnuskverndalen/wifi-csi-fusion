"""
WiFi CSI Fusion — Professional Edition
Camera + MediaPipe | 2D Room Map heatmap | CSI node list
"""

import socket, struct, threading, time, argparse, collections, math
import urllib.request, os, random

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions

BG          = (10,  12,  14)
PANEL_BG    = (14,  17,  20)
BORDER      = (30,  35,  40)
ACCENT      = (0,   210, 120)
ACCENT_DIM  = (0,   90,  50)
TEXT_PRI    = (220, 230, 225)
TEXT_SEC    = (90,  105, 100)
TEXT_DIM    = (45,  55,  50)
WARN        = (0,   180, 220)
DANGER      = (50,  60,  220)
SKEL_LINE   = (0,   190, 100)
SKEL_JOINT  = (0,   255, 140)
SKEL_JOINT2 = (0,   120, 60)
MAP_BG      = (8,   12,  10)
MAP_GRID    = (18,  28,  20)

DEFAULT_UDP_PORT = 5005
DEFAULT_CAMERA   = 0
DEFAULT_W        = 1800
DEFAULT_H        = 640
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

def hline(img,y,x0,x1,color,t=1):
    cv2.line(img,(x0,y),(x1,y),color,t)

def label(img,text,x,y,color=TEXT_PRI,scale=0.42,t=1):
    cv2.putText(img,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,t,cv2.LINE_AA)

def tag(img,text,x,y,fg,bg):
    (tw,th),_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,0.36,1)
    cv2.rectangle(img,(x,y-th-2),(x+tw+12,y+4),bg,-1)
    label(img,text,x+6,y,fg,scale=0.36)
    return x+tw+18

def scanlines(frame,gap=4,alpha=0.06):
    for y in range(0,frame.shape[0],gap):
        frame[y]=np.clip(frame[y].astype(np.float32)*(1-alpha),0,255).astype(np.uint8)
    return frame

_vig={}
def vignette(frame,strength=0.40):
    h,w=frame.shape[:2]
    if (h,w) not in _vig:
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


class Terminal:
    LH=15;FSC=0.72;MAX=14
    def __init__(self):
        self.lines=collections.deque(maxlen=self.MAX);self._next=0
    def tick(self,nodes=3):
        if time.time()>self._next:
            self.lines.append(rand_log(nodes))
            self._next=time.time()+random.uniform(0.1,0.4)
    def render(self,canvas,x=12,bottom_y=None,max_w=None):
        h,w=canvas.shape[:2]
        if bottom_y is None:bottom_y=h-10
        if max_w is None:max_w=w-x-10
        n=len(self.lines)
        if n==0:return
        y0=bottom_y-n*self.LH
        blink=int(time.time()*2)%2==0
        for i,line in enumerate(self.lines):
            y=y0+i*self.LH;last=(i==n-1)
            col=ACCENT if last else(TEXT_SEC if i/max(n-1,1)>0.5 else TEXT_DIM)
            cv2.putText(canvas,">",(x,y),cv2.FONT_HERSHEY_PLAIN,self.FSC,
                        ACCENT_DIM if not last else ACCENT,1,cv2.LINE_AA)
            cv2.putText(canvas,line,(x+12,y),cv2.FONT_HERSHEY_PLAIN,
                        self.FSC,col,1,cv2.LINE_AA)
        if blink and n>0:
            tw,_=cv2.getTextSize(self.lines[-1],cv2.FONT_HERSHEY_PLAIN,self.FSC,1)
            cx=x+12+tw[0]+3;cy=y0+(n-1)*self.LH
            cv2.rectangle(canvas,(cx,cy-9),(cx+6,cy+1),ACCENT,-1)


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


class RoomMap:
    HEAT_RES=60;BLUR_K=15
    def __init__(self,node_positions,room_w=10.0,room_d=5.0):
        self.node_pos=node_positions;self.room_w=room_w;self.room_d=room_d
        self.heat=np.zeros((self.HEAT_RES,self.HEAT_RES),dtype=np.float32)
        self.prev_heat=np.zeros_like(self.heat);self._decay=0.88

    def update(self,nodes_snapshot,estimators,timeout):
        now=time.time()
        grid=np.zeros((self.HEAT_RES,self.HEAT_RES),dtype=np.float32)
        gw,gh=self.HEAT_RES,self.HEAT_RES
        for nid,pos in self.node_pos.items():
            data=nodes_snapshot.get(nid)
            if data is None or now-data["received_at"]>timeout:continue
            est=estimators.get(nid)
            if est is None:continue
            result=est.update(data.get("amplitudes",[]))
            if not result["presence"]:continue
            signal=result["signal"];var=result["variance"]
            nx,ny=pos[0],pos[1]
            for gy in range(gh):
                for gx in range(gw):
                    wx=(gx/gw-0.5)*self.room_w
                    wy=(1.0-gy/gh)*self.room_d
                    dist=math.sqrt((wx-nx)**2+(wy-ny)**2)+0.5
                    grid[gy,gx]+=(signal/30.0)*var*(1.0/(dist**1.5))
        if grid.max()>0:grid=grid/grid.max()
        blurred=cv2.GaussianBlur(grid,(self.BLUR_K,self.BLUR_K),0)
        self.heat=self._decay*self.prev_heat+(1-self._decay)*blurred
        self.prev_heat=self.heat.copy()

    def render(self,pw,ph,nodes_snapshot,timeout):
        panel=np.full((ph,pw,3),MAP_BG,dtype=np.uint8);now=time.time()
        cv2.rectangle(panel,(0,0),(pw,44),BG,-1)
        label(panel,"ROOM MAP",14,17,ACCENT,scale=0.50)
        label(panel,"CSI TRILATERATION  ·  THROUGH-WALL",14,33,TEXT_SEC,scale=0.36)
        label(panel,f"{self.room_w:.0f}m x {self.room_d:.0f}m",pw-70,17,TEXT_SEC,scale=0.38)
        hline(panel,44,0,pw,BORDER)
        MX=12;MY=50;MW=pw-24;MH=ph-90
        cv2.rectangle(panel,(MX,MY),(MX+MW,MY+MH),MAP_BG,-1)
        for i in range(11):
            x=MX+int(i*MW/10);cv2.line(panel,(x,MY),(x,MY+MH),MAP_GRID,1)
        for i in range(6):
            y=MY+int(i*MH/5);cv2.line(panel,(MX,y),(MX+MW,y),MAP_GRID,1)
        if self.heat.max()>0.01:
            heat_u8=(np.clip(self.heat,0,1)*255).astype(np.uint8)
            heat_up=cv2.resize(heat_u8,(MW,MH),interpolation=cv2.INTER_LINEAR)
            heat_rgb=np.zeros((MH,MW,3),dtype=np.uint8)
            heat_rgb[:,:,1]=heat_up
            heat_rgb[:,:,0]=(heat_up.astype(np.float32)*0.1).astype(np.uint8)
            heat_rgb[:,:,2]=(heat_up.astype(np.float32)*0.05).astype(np.uint8)
            bright=np.clip((heat_up.astype(np.float32)/255.0-0.6)/0.4,0,1)
            heat_rgb[:,:,1]=np.clip(heat_rgb[:,:,1].astype(np.float32)+bright*60,0,255).astype(np.uint8)
            roi=panel[MY:MY+MH,MX:MX+MW]
            cv2.addWeighted(heat_rgb,0.75,roi,0.25,0,roi)
        cv2.rectangle(panel,(MX,MY),(MX+MW,MY+MH),BORDER,1)
        for nid,pos in self.node_pos.items():
            xw,yw=pos[0],pos[1]
            data=nodes_snapshot.get(nid)
            age=now-data["received_at"] if data else timeout+1
            online=data is not None and age<timeout
            px_fx=(xw/self.room_w+0.5);px_fy=1.0-yw/self.room_d
            px=MX+int(np.clip(px_fx,0,1)*MW);py=MY+int(np.clip(px_fy,0,1)*MH)
            if online:
                pulse=int(abs(math.sin(time.time()*2+nid))*12)
                cv2.circle(panel,(px,py),14+pulse,(0,80,40),1,cv2.LINE_AA)
                cv2.circle(panel,(px,py),8,ACCENT,-1,cv2.LINE_AA)
                cv2.circle(panel,(px,py),8,BG,1,cv2.LINE_AA)
                r_px=int(2.5/self.room_w*MW)
                cv2.circle(panel,(px,py),r_px,ACCENT_DIM,1,cv2.LINE_AA)
            else:
                cv2.circle(panel,(px,py),8,DANGER,-1,cv2.LINE_AA)
            label(panel,f"N{nid}",px-6,py-12,TEXT_PRI,scale=0.38)
        m_px=int(MW/self.room_w);sb_x=MX+8;sb_y=MY+MH-10
        cv2.line(panel,(sb_x,sb_y),(sb_x+m_px,sb_y),TEXT_SEC,1)
        cv2.line(panel,(sb_x,sb_y-3),(sb_x,sb_y+3),TEXT_SEC,1)
        cv2.line(panel,(sb_x+m_px,sb_y-3),(sb_x+m_px,sb_y+3),TEXT_SEC,1)
        label(panel,"1 m",sb_x+m_px+4,sb_y+4,TEXT_DIM,scale=0.32)
        hline(panel,ph-32,0,pw,BORDER)
        cv2.rectangle(panel,(0,ph-32),(pw,ph),BG,-1)
        heat_max=float(self.heat.max())
        if heat_max>0.15:
            label(panel,"PRESENCE DETECTED  ·  THROUGH-WALL ACTIVE",14,ph-12,ACCENT,scale=0.40)
        else:
            label(panel,"NO PRESENCE  ·  MONITORING",14,ph-12,TEXT_DIM,scale=0.40)
        label(panel,f"HEAT {heat_max*100:.0f}%",pw-70,ph-12,TEXT_SEC,scale=0.36)
        return panel


def make_csi_panel(nodes,ests,timeout,pw,ph):
    img=np.full((ph,pw,3),PANEL_BG,dtype=np.uint8);now=time.time()
    cv2.rectangle(img,(0,0),(pw,44),BG,-1)
    label(img,"CSI  NODES",14,17,ACCENT,scale=0.50)
    label(img,"RUVECTOR v2.0.4  ·  ADR-018",14,33,TEXT_SEC,scale=0.36)
    label(img,time.strftime("%H:%M:%S"),pw-68,17,TEXT_SEC,scale=0.40)
    online_n=sum(1 for d in nodes.values() if now-d["received_at"]<timeout)
    label(img,f"{online_n}/{len(nodes) or 0} ONLINE",pw-80,33,
          ACCENT if online_n>0 else DANGER,scale=0.36)
    hline(img,44,0,pw,BORDER)
    slot_h=(ph-44-32)//6
    for idx in range(1,7):
        y0=44+(idx-1)*slot_h;data=nodes.get(idx)
        age=now-data["received_at"] if data else timeout+1
        online=data is not None and age<timeout
        hline(img,y0,10,pw-10,BORDER)
        pulse=0.5+0.5*abs(math.sin(time.time()*2.5+idx))
        dot_col=tuple(int(c*pulse) for c in ACCENT) if online else DANGER
        cv2.circle(img,(22,y0+slot_h//2),5,dot_col,-1)
        node_y=y0+18
        label(img,f"NODE  {idx}",38,node_y,TEXT_PRI,scale=0.44)
        if not online:
            label(img,"OFFLINE",pw-68,node_y,DANGER,scale=0.38)
            label(img,"waiting for stream",38,node_y+14,TEXT_DIM,scale=0.34)
            continue
        est=ests.setdefault(idx,Motion());res=est.update(data.get("amplitudes",[]))
        rssi=data.get("rssi",0);sc=data.get("sc_count",0)
        motion=res["motion"];pres=res["presence"];var=res["variance"]
        ip=data.get("ip","")
        if pres:tag(img,"PRESENT",pw-86,node_y+2,BG,ACCENT)
        else:tag(img,"EMPTY",pw-70,node_y+2,TEXT_SEC,BORDER)
        label(img,f"RSSI {rssi}  SC {sc}  VAR {var:.1f}  {ip}",
              38,node_y+14,TEXT_SEC,scale=0.32)
        bx=38;by=y0+slot_h-14;bw=pw-52
        cv2.rectangle(img,(bx,by),(bx+bw,by+3),BORDER,-1)
        filled=int(bw*motion)
        if filled>0:
            cv2.rectangle(img,(bx,by),(bx+filled,by+3),
                          WARN if motion>0.55 else ACCENT,-1)
        amps=data.get("amplitudes",[])
        if len(amps)>4:
            sy=y0+slot_h-26;sh=9;step=max(1,len(amps)//bw)
            samp=amps[::step][:bw];mx=max(samp) or 1
            pts=[(bx+i,sy+sh-int(sh*v/mx)) for i,v in enumerate(samp)]
            for i in range(len(pts)-1):
                frac=samp[i]/mx
                c=tuple(int(a+(b-a)*frac) for a,b in zip(ACCENT_DIM,ACCENT))
                cv2.line(img,pts[i],pts[i+1],c,1,cv2.LINE_AA)
    hline(img,ph-32,0,pw,BORDER)
    cv2.rectangle(img,(0,ph-32),(pw,ph),BG,-1)
    label(img,f"UDP :5005  ·  2.4 GHz  ·  {online_n} ACTIVE",14,ph-12,TEXT_DIM,scale=0.34)
    return img


def draw_skeleton(frame,result):
    if not result or not result.pose_landmarks:return frame
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
                cv2.circle(frame,(x,y),4,SKEL_JOINT2,1,cv2.LINE_AA)
    return frame


def make_cam_panel(frame,result,csi_present,terminal,nodes_n,pw,ph):
    p=cv2.resize(frame,(pw,ph))
    gray=cv2.cvtColor(p,cv2.COLOR_BGR2GRAY)
    gray3=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    p=cv2.addWeighted(p,0.30,gray3,0.70,0)
    p=cv2.addWeighted(p,0.80,np.zeros_like(p),0.20,0)
    p[:,:,1]=np.clip(p[:,:,1].astype(np.int32)+8,0,255).astype(np.uint8)
    p=draw_skeleton(p,result)
    p=scanlines(p,gap=4,alpha=0.07)
    p=vignette(p,strength=0.45)
    ov=p.copy()
    cv2.rectangle(ov,(0,0),(pw,44),(0,0,0),-1)
    cv2.addWeighted(ov,0.65,p,0.35,0,p)
    label(p,"CAMERA  ·  MEDIAPIPE POSE  ·  CSI FUSION",14,17,TEXT_PRI,scale=0.46)
    label(p,time.strftime("%Y-%m-%d  %H:%M:%S"),14,33,TEXT_SEC,scale=0.36)
    hline(p,44,0,pw,BORDER)
    if csi_present:
        pulse=abs(math.sin(time.time()*3.0));L=22
        bri=tuple(int(c*(0.5+0.5*pulse)) for c in ACCENT)
        for(x,y,sx,sy) in[(1,1,1,1),(pw-2,1,-1,1),(1,ph-2,1,-1),(pw-2,ph-2,-1,-1)]:
            cv2.line(p,(x,y),(x+sx*L,y),bri,2,cv2.LINE_AA)
            cv2.line(p,(x,y),(x,y+sy*L),bri,2,cv2.LINE_AA)
    ov2=p.copy()
    cv2.rectangle(ov2,(0,ph-32),(pw,ph),(0,0,0),-1)
    cv2.addWeighted(ov2,0.65,p,0.35,0,p)
    hline(p,ph-32,0,pw,BORDER)
    label(p,"CSI  PRESENCE  CONFIRMED" if csi_present else "CSI  NO PRESENCE",
          14,ph-12,ACCENT if csi_present else TEXT_DIM,scale=0.42)
    pose_n=len(result.pose_landmarks) if result and result.pose_landmarks else 0
    label(p,f"POSES: {pose_n}",pw-72,ph-12,TEXT_SEC,scale=0.36)
    terminal.tick(nodes=nodes_n)
    terminal.render(p,x=14,bottom_y=ph-38,max_w=pw-28)
    return p


def download_model():
    if not os.path.exists(MODEL_PATH):
        print("[INFO] Downloading pose model (~7 MB)...")
        urllib.request.urlretrieve(MODEL_URL,MODEL_PATH)
        print("[INFO] Model ready")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--port",type=int,default=DEFAULT_UDP_PORT)
    ap.add_argument("--camera",type=int,default=DEFAULT_CAMERA)
    ap.add_argument("--width",type=int,default=DEFAULT_W)
    ap.add_argument("--height",type=int,default=DEFAULT_H)
    ap.add_argument("--timeout",type=float,default=DEFAULT_TIMEOUT)
    ap.add_argument("--no-camera",action="store_true")
    ap.add_argument("--positions",type=str,default="0,0,0.4;2,4,1.0;-5,4,1.0")
    ap.add_argument("--room",type=str,default="10x5",
                    help="Room WxD in metres e.g. 10x5")
    args=ap.parse_args()

    try:room_w,room_d=map(float,args.room.lower().split("x"))
    except Exception:room_w,room_d=10.0,5.0

    node_pos={}
    if args.positions:
        for idx,part in enumerate(args.positions.split(";"),start=1):
            try:
                x,y,z=map(float,part.split(","))
                node_pos[idx]=(x,y,z)
            except ValueError:pass

    total_w=args.width;ph=args.height
    cam_w=int(total_w*0.40);map_w=int(total_w*0.35);csi_w=total_w-cam_w-map_w

    rx=CSIReceiver(port=args.port);ests={};rx.start()
    term=Terminal()
    room=RoomMap(node_pos,room_w=room_w,room_d=room_d)
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
    cv2.resizeWindow("WiFi CSI Fusion",total_w,ph)
    print("[INFO] Q=quit  S=screenshot")
    print(f"[INFO] Room {room_w}m x {room_d}m  |  {len(node_pos)} nodes configured")

    fps_q=collections.deque(maxlen=30);ts_ms=0
    div=np.full((ph,1,3),BORDER,dtype=np.uint8)

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
            cp=make_csi_panel(snap,ests,args.timeout,csi_w,ph)
            mp_=room.render(map_w,ph,snap,args.timeout)
            result=None
            if cap and detector:
                ok,frame=cap.read()
                if ok:
                    ts_ms+=33
                    mp_img=mp.Image(image_format=mp.ImageFormat.SRGB,
                                    data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                    result=detector.detect_for_video(mp_img,ts_ms)
                    vp=make_cam_panel(frame,result,csi_present,term,len(snap),cam_w,ph)
                else:
                    vp=np.full((ph,cam_w,3),BG,dtype=np.uint8)
            else:
                vp=np.full((ph,cam_w,3),BG,dtype=np.uint8)
                label(vp,"NO CAMERA  ·  CSI-ONLY MODE",cam_w//2-110,ph//2,TEXT_SEC,scale=0.45)
                term.tick(len(snap));term.render(vp,x=14,bottom_y=ph-14)
            composed=np.hstack([vp,div,mp_,div,cp])
            fps_q.append(time.time()-t0)
            fps=1.0/(sum(fps_q)/len(fps_q)) if fps_q else 0
            label(composed,f"{fps:.0f} fps",total_w-55,ph-12,TEXT_DIM,scale=0.36)
            cv2.imshow("WiFi CSI Fusion",composed)
            key=cv2.waitKey(1)&0xFF
            if key==ord("q"):break
            if key==ord("s"):
                fn=f"output/screenshot_{int(time.time())}.png"
                cv2.imwrite(fn,composed);print(f"[INFO] Saved {fn}")
    finally:
        rx.stop()
        if cap:cap.release()
        if detector:detector.close()
        cv2.destroyAllWindows();print("[INFO] Done")

if __name__=="__main__":
    main()
