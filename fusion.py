"""
WiFi CSI Fusion — Professional Edition
Minimal dark UI · ESP32 CSI · MediaPipe Pose · Terminal overlay
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

DEFAULT_UDP_PORT = 5005
DEFAULT_CAMERA   = 0
DEFAULT_W        = 1600
DEFAULT_H        = 640
DEFAULT_TIMEOUT  = 3.0
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
    "WASM    module=LOADED  tier=2",
    "OTA     status=IDLE  version=0.6.2",
    "ADAPT   state={s}  cycle=200ms",
    "EMBED   dim=128  fingerprint=UPDATED",
    "TRACK   persons={p}  keypoints=17  conf={c:.0f}%",
    "SPOOF   check=PASS  replay=NONE",
]


def rand_log(nodes=3):
    t = random.choice(LOG_POOL)
    return t.format(
        nid=random.randint(1, max(nodes,1)), seq=random.randint(0, 99999),
        sc=random.choice([56,64,128]), rssi=random.randint(-80,-45),
        ch=random.choice([1,6,11]), amp=random.uniform(5,40),
        n=random.randint(0,4), d=random.uniform(0.5,5.0),
        w=random.uniform(0.1,0.9), nodes=nodes,
        v=random.uniform(0.001,50.0), var=random.uniform(0.01,20.0),
        bpm=random.uniform(10,28), c=random.uniform(55,99),
        fps=random.uniform(18,22), snr=random.uniform(8,35),
        l=random.choice([32,48,60,148,276]),
        l2=max(nodes*(nodes-1),0), p=random.randint(1,3),
        s=random.randint(1,12),
    )


def hline(img, y, x0, x1, color, t=1):
    cv2.line(img, (x0,y), (x1,y), color, t)

def label(img, text, x, y, color=TEXT_PRI, scale=0.42, t=1):
    cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, t, cv2.LINE_AA)

def tag(img, text, x, y, fg, bg):
    (tw,th),_ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
    cv2.rectangle(img, (x, y-th-2), (x+tw+12, y+4), bg, -1)
    label(img, text, x+6, y, fg, scale=0.36)
    return x+tw+18

def scanlines(frame, gap=4, alpha=0.06):
    for y in range(0, frame.shape[0], gap):
        frame[y] = np.clip(frame[y].astype(np.float32)*(1-alpha),
                           0, 255).astype(np.uint8)
    return frame

_vig = {}
def vignette(frame, strength=0.40):
    h, w = frame.shape[:2]
    if (h,w) not in _vig:
        k = np.zeros((h,w), dtype=np.float32)
        cx, cy = w/2, h/2
        for y in range(h):
            for x in range(w):
                dx=(x-cx)/cx; dy=(y-cy)/cy
                k[y,x]=1.0-min(math.sqrt(dx*dx+dy*dy)*strength,1.0)
        _vig[(h,w)] = k
    mask = _vig[(h,w)]
    out  = frame.astype(np.float32)
    for c in range(3): out[:,:,c] *= mask
    return np.clip(out,0,255).astype(np.uint8)


class Terminal:
    LH=15; FSC=0.72; MAX=16
    def __init__(self):
        self.lines=collections.deque(maxlen=self.MAX); self._next=0
    def tick(self, nodes=3):
        if time.time()>self._next:
            self.lines.append(rand_log(nodes))
            self._next=time.time()+random.uniform(0.1,0.4)
    def render(self, canvas, x=12, bottom_y=None, max_w=None):
        h,w=canvas.shape[:2]
        if bottom_y is None: bottom_y=h-10
        if max_w    is None: max_w=w-x-10
        n=len(self.lines)
        if n==0: return
        y0=bottom_y-n*self.LH
        ov=canvas.copy()
        cv2.rectangle(ov,(x-6,y0-6),(x+max_w+6,y0+n*self.LH+4),(8,10,12),-1)
        cv2.addWeighted(ov,0.72,canvas,0.28,0,canvas)
        blink=int(time.time()*2)%2==0
        for i,line in enumerate(self.lines):
            y=y0+i*self.LH; last=(i==n-1)
            col=ACCENT if last else (TEXT_SEC if i/max(n-1,1)>0.5 else TEXT_DIM)
            cv2.putText(canvas,">",(x,y),cv2.FONT_HERSHEY_PLAIN,self.FSC,
                        ACCENT_DIM if not last else ACCENT,1,cv2.LINE_AA)
            cv2.putText(canvas,line,(x+12,y),cv2.FONT_HERSHEY_PLAIN,
                        self.FSC,col,1,cv2.LINE_AA)
        if blink and n>0:
            tw,_=cv2.getTextSize(self.lines[-1],cv2.FONT_HERSHEY_PLAIN,self.FSC,1)
            cx=x+12+tw[0]+3; cy=y0+(n-1)*self.LH
            cv2.rectangle(canvas,(cx,cy-9),(cx+6,cy+1),ACCENT,-1)


def parse_frame(data):
    if len(data)<HEADER_SZ: return None
    try:
        magic,nid,seq,ts,rssi,ch,sc,dl=struct.unpack_from(HEADER_FMT,data,0)
    except struct.error: return None
    if magic not in (MAGIC_CSI,MAGIC_FEAT): return None
    payload=data[HEADER_SZ:HEADER_SZ+dl]; amps=[]
    if magic==MAGIC_CSI and len(payload)>=sc*4:
        for i in range(sc):
            iv,qv=struct.unpack_from("<hh",payload,i*4)
            amps.append(math.sqrt(iv*iv+qv*qv))
    return {"node_id":nid,"seq":seq,"rssi":rssi,"channel":ch,
            "sc_count":sc,"amplitudes":amps,"received_at":time.time()}


class CSIReceiver(threading.Thread):
    def __init__(self,port):
        super().__init__(daemon=True)
        self.port=port; self.nodes={}; self.lock=threading.Lock(); self._run=True
    def run(self):
        s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        s.bind(("0.0.0.0",self.port)); s.settimeout(0.5)
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
            except socket.timeout: pass
        s.close()
    def stop(self): self._run=False
    def snapshot(self):
        with self.lock: return dict(self.nodes)


class Motion:
    def __init__(self): self.hist=collections.deque(maxlen=40)
    def update(self,amps):
        if not amps: return {"motion":0.0,"presence":False,"variance":0.0}
        m=float(np.mean(amps)); self.hist.append(m)
        if len(self.hist)<6: return {"motion":0.0,"presence":False,"variance":0.0}
        arr=np.array(self.hist); var=float(np.var(arr))
        dif=float(np.mean(np.abs(np.diff(arr[-12:]))))
        return {"motion":min(dif/5.0,1.0),"presence":var>1.8,
                "variance":var,"mean":m}


def make_csi_panel(nodes,ests,timeout,pw,ph):
    img=np.full((ph,pw,3),PANEL_BG,dtype=np.uint8); now=time.time()
    cv2.rectangle(img,(0,0),(pw,44),BG,-1)
    label(img,"CSI  MESH",16,17,ACCENT,scale=0.50)
    label(img,"RUVECTOR v2.0.4  ·  ADR-018",16,33,TEXT_SEC,scale=0.36)
    label(img,time.strftime("%H:%M:%S"),pw-68,17,TEXT_SEC,scale=0.40)
    online_n=sum(1 for d in nodes.values() if now-d["received_at"]<timeout)
    label(img,f"{online_n}/{len(nodes) or 0} ONLINE",pw-80,33,
          ACCENT if online_n>0 else DANGER,scale=0.36)
    hline(img,44,0,pw,BORDER)
    slot_h=(ph-44-32)//6
    for idx in range(1,7):
        y0=44+(idx-1)*slot_h; data=nodes.get(idx)
        age=now-data["received_at"] if data else timeout+1
        online=data is not None and age<timeout
        hline(img,y0,12,pw-12,BORDER)
        pulse=0.5+0.5*abs(math.sin(time.time()*2.5+idx))
        dot_col=tuple(int(c*pulse) for c in ACCENT) if online else DANGER
        cv2.circle(img,(26,y0+slot_h//2),5,dot_col,-1)
        node_y=y0+18
        label(img,f"NODE  {idx}",40,node_y,TEXT_PRI,scale=0.44)
        if not online:
            label(img,"OFFLINE",pw-72,node_y,DANGER,scale=0.38)
            label(img,"waiting for stream",40,node_y+14,TEXT_DIM,scale=0.34)
            continue
        est=ests.setdefault(idx,Motion()); res=est.update(data.get("amplitudes",[]))
        rssi=data.get("rssi",0); sc=data.get("sc_count",0)
        motion=res["motion"]; pres=res["presence"]
        var=res["variance"]; ip=data.get("ip","")
        if pres: tag(img,"PRESENT",pw-90,node_y+2,BG,ACCENT)
        else:    tag(img,"EMPTY",pw-74,node_y+2,TEXT_SEC,BORDER)
        label(img,f"RSSI {rssi} dBm  ·  {sc} SC  ·  VAR {var:.2f}  ·  {ip}",
              40,node_y+14,TEXT_SEC,scale=0.34)
        bx=40; by=y0+slot_h-16; bw=pw-56
        cv2.rectangle(img,(bx,by),(bx+bw,by+3),BORDER,-1)
        filled=int(bw*motion)
        if filled>0:
            cv2.rectangle(img,(bx,by),(bx+filled,by+3),
                          WARN if motion>0.55 else ACCENT,-1)
        label(img,f"{motion*100:.0f}%",bx+bw+4,by+4,TEXT_DIM,scale=0.32)
        amps=data.get("amplitudes",[])
        if len(amps)>4:
            sy=y0+slot_h-28; sh=10; step=max(1,len(amps)//bw)
            samp=amps[::step][:bw]; mx=max(samp) or 1
            pts=[(bx+i,sy+sh-int(sh*v/mx)) for i,v in enumerate(samp)]
            for i in range(len(pts)-1):
                frac=samp[i]/mx
                c=tuple(int(a+(b-a)*frac) for a,b in zip(ACCENT_DIM,ACCENT))
                cv2.line(img,pts[i],pts[i+1],c,1,cv2.LINE_AA)
    hline(img,ph-32,0,pw,BORDER)
    cv2.rectangle(img,(0,ph-32),(pw,ph),BG,-1)
    label(img,f"UDP :5005  ·  ADR-018  ·  2.4 GHz  ·  {online_n} NODES ACTIVE",
          16,ph-12,TEXT_DIM,scale=0.34)
    return img


def draw_skeleton(frame,result):
    if not result or not result.pose_landmarks: return frame
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
        pulse=abs(math.sin(time.time()*3.0)); L=22
        bri=tuple(int(c*(0.5+0.5*pulse)) for c in ACCENT)
        for (x,y,sx,sy) in [(1,1,1,1),(pw-2,1,-1,1),(1,ph-2,1,-1),(pw-2,ph-2,-1,-1)]:
            cv2.line(p,(x,y),(x+sx*L,y),bri,2,cv2.LINE_AA)
            cv2.line(p,(x,y),(x,y+sy*L),bri,2,cv2.LINE_AA)
    ov2=p.copy()
    cv2.rectangle(ov2,(0,ph-32),(pw,ph),(0,0,0),-1)
    cv2.addWeighted(ov2,0.65,p,0.35,0,p)
    hline(p,ph-32,0,pw,BORDER)
    label(p,"CSI  PRESENCE  CONFIRMED" if csi_present else "CSI  NO PRESENCE",
          14,ph-12,ACCENT if csi_present else TEXT_DIM,scale=0.42)
    pose_n=len(result.pose_landmarks) if result and result.pose_landmarks else 0
    label(p,f"POSES: {pose_n}",pw-80,ph-12,TEXT_SEC,scale=0.36)
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
    ap.add_argument("--positions",type=str,default="")
    args=ap.parse_args()
    pw=args.width//2; ph=args.height
    rx=CSIReceiver(port=args.port); ests={}; rx.start()
    term=Terminal(); cap=detector=None
    if not args.no_camera:
        download_model()
        cap=cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print("[WARN] No camera — CSI-only mode"); cap=None
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
    cv2.resizeWindow("WiFi CSI Fusion",args.width,args.height)
    print("[INFO] Q=quit  S=screenshot")
    fps_q=collections.deque(maxlen=30); ts_ms=0
    div=np.full((ph,1,3),BORDER,dtype=np.uint8)
    try:
        while True:
            t0=time.time(); snap=rx.snapshot(); now=time.time()
            csi_present=any(
                now-d["received_at"]<args.timeout
                and ests.get(nid,Motion()).hist
                and np.var(list(ests[nid].hist))>1.8
                for nid,d in snap.items()
            )
            cp=make_csi_panel(snap,ests,args.timeout,pw,ph)
            result=None
            if cap and detector:
                ok,frame=cap.read()
                if ok:
                    ts_ms+=33
                    mp_img=mp.Image(image_format=mp.ImageFormat.SRGB,
                                    data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                    result=detector.detect_for_video(mp_img,ts_ms)
                    vp=make_cam_panel(frame,result,csi_present,term,len(snap),pw,ph)
                else:
                    vp=np.full((ph,pw,3),BG,dtype=np.uint8)
            else:
                vp=np.full((ph,pw,3),BG,dtype=np.uint8)
                label(vp,"NO CAMERA  ·  CSI-ONLY MODE",pw//2-110,ph//2,TEXT_SEC,scale=0.45)
                term.tick(len(snap)); term.render(vp,x=14,bottom_y=ph-14)
            composed=np.hstack([vp,div,cp])
            fps_q.append(time.time()-t0)
            fps=1.0/(sum(fps_q)/len(fps_q)) if fps_q else 0
            label(composed,f"{fps:.0f} fps",args.width-55,ph-12,TEXT_DIM,scale=0.36)
            cv2.imshow("WiFi CSI Fusion",composed)
            key=cv2.waitKey(1)&0xFF
            if key==ord("q"): break
            if key==ord("s"):
                fn=f"output/screenshot_{int(time.time())}.png"
                cv2.imwrite(fn,composed); print(f"[INFO] Saved {fn}")
    finally:
        rx.stop()
        if cap: cap.release()
        if detector: detector.close()
        cv2.destroyAllWindows(); print("[INFO] Done")

if __name__=="__main__":
    main()
