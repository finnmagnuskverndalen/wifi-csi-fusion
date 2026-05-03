import socket,struct,threading,time,math,os,json,argparse
import cv2,numpy as np,mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker,PoseLandmarkerOptions

MAGIC_CSI=0xC5110001;MAGIC_FEAT=0xC5110006
HEADER_FMT="<IBIQbBHH";HEADER_SZ=struct.calcsize(HEADER_FMT)
MODEL_PATH="pose_landmarker.task"
MODEL_URL=("https://storage.googleapis.com/mediapipe-models/"
           "pose_landmarker/pose_landmarker_lite/float16/latest/"
           "pose_landmarker_lite.task")
ACCENT=(0,210,120);RED=(50,60,220);WHITE=(220,230,225);DIM=(60,70,65)

def parse_frame(data):
    if len(data)<HEADER_SZ:return None
    try:
        magic,nid,seq,ts,rssi,ch,sc,dl=struct.unpack_from(HEADER_FMT,data,0)
    except:return None
    if magic not in(MAGIC_CSI,MAGIC_FEAT):return None
    payload=data[HEADER_SZ:HEADER_SZ+dl];amps=[]
    if magic==MAGIC_CSI and len(payload)>=sc*4:
        for i in range(sc):
            iv,qv=struct.unpack_from("<hh",payload,i*4)
            amps.append(math.sqrt(iv*iv+qv*qv))
    return{"node_id":nid,"rssi":rssi,"sc_count":sc,
           "amplitudes":amps,"ts":time.time()}

class CSIReceiver(threading.Thread):
    def __init__(self,port=5005):
        super().__init__(daemon=True)
        self.port=port;self.nodes={};self.lock=threading.Lock();self._run=True
    def run(self):
        s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        s.bind(("0.0.0.0",self.port));s.settimeout(0.5)
        while self._run:
            try:
                data,addr=s.recvfrom(4096)
                f=parse_frame(data)
                if f:
                    with self.lock:self.nodes[f["node_id"]]={**f,"ip":addr[0]}
            except socket.timeout:pass
        s.close()
    def stop(self):self._run=False
    def snapshot(self):
        with self.lock:return dict(self.nodes)

def download_model():
    if not os.path.exists(MODEL_PATH):
        import urllib.request
        print("[INFO] Downloading model...")
        urllib.request.urlretrieve(MODEL_URL,MODEL_PATH)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--port",type=int,default=5005)
    ap.add_argument("--camera",type=int,default=0)
    ap.add_argument("--out",type=str,default="training-data")
    ap.add_argument("--nodes",type=int,default=3)
    args=ap.parse_args()
    os.makedirs(args.out,exist_ok=True)
    download_model()
    rx=CSIReceiver(args.port);rx.start()
    opts=PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,num_poses=1,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    detector=PoseLandmarker.create_from_options(opts)
    cap=cv2.VideoCapture(args.camera)
    cv2.namedWindow("CSI Collector",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("CSI Collector",960,540)
    recording=False;samples=[];ts_ms=0;frame_no=0
    session_id=int(time.time())
    print("[INFO] R=record/stop  Q=quit")
    while True:
        ok,frame=cap.read()
        if not ok:continue
        ts_ms+=33
        fh,fw=frame.shape[:2]
        scale=min(960/fw,540/fh)
        nw,nh=int(fw*scale),int(fh*scale)
        display=cv2.resize(frame,(nw,nh))
        mp_img=mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        result=detector.detect_for_video(mp_img,ts_ms)
        snap=rx.snapshot();now=time.time()
        keypoints=None
        if result and result.pose_landmarks:
            lms=result.pose_landmarks[0]
            keypoints=[[lm.x,lm.y,lm.z,lm.visibility] for lm in lms]
            h2,w2=display.shape[:2]
            for lm in lms:
                x,y=int(lm.x*w2),int(lm.y*h2)
                if lm.visibility>0.5:
                    cv2.circle(display,(x,y),4,ACCENT,-1,cv2.LINE_AA)
        csi_vector=[]
        for nid in range(1,args.nodes+1):
            data=snap.get(nid)
            if data and now-data["ts"]<2.0:
                amps=data["amplitudes"]
                if amps:
                    padded=(amps+[0.0]*56)[:56]
                    csi_vector.extend(padded+[float(np.mean(amps)),
                                              float(np.var(amps)),
                                              float(data["rssi"])])
                    continue
            csi_vector.extend([0.0]*59)
        if recording and keypoints and any(v!=0 for v in csi_vector):
            flat_kp=[]
            for lm in keypoints[:17]:flat_kp.extend([lm[0],lm[1],lm[2]])
            if len(flat_kp)==51:
                samples.append({"csi":csi_vector,"keypoints":flat_kp,"ts":now})
                frame_no+=1
        canvas=np.full((540,960,3),(10,12,14),dtype=np.uint8)
        xo=(960-nw)//2
        canvas[:nh,xo:xo+nw]=display
        cv2.rectangle(canvas,(0,0),(960,36),(5,7,9),-1)
        online_n=sum(1 for d in snap.values() if now-d["ts"]<2.0)
        if recording:
            pulse=int(abs(math.sin(time.time()*3))*200)
            cv2.circle(canvas,(16,18),6,(0,0,150+pulse),-1)
            cv2.putText(canvas,f"REC {frame_no} frames",(28,22),
                        cv2.FONT_HERSHEY_SIMPLEX,0.45,(100,100,220),1,cv2.LINE_AA)
        else:
            cv2.circle(canvas,(16,18),6,DIM,-1)
            cv2.putText(canvas,"STANDBY — press R to record",(28,22),
                        cv2.FONT_HERSHEY_SIMPLEX,0.45,DIM,1,cv2.LINE_AA)
        pose_col=ACCENT if keypoints else (60,60,60)
        cv2.putText(canvas,f"POSE:{'OK' if keypoints else 'NO'}",(400,22),
                    cv2.FONT_HERSHEY_SIMPLEX,0.42,pose_col,1,cv2.LINE_AA)
        cv2.putText(canvas,f"CSI:{online_n}/{args.nodes}",(530,22),
                    cv2.FONT_HERSHEY_SIMPLEX,0.42,
                    ACCENT if online_n>0 else RED,1,cv2.LINE_AA)
        cv2.putText(canvas,"R=rec  Q=quit",(820,22),
                    cv2.FONT_HERSHEY_SIMPLEX,0.38,DIM,1,cv2.LINE_AA)
        cv2.rectangle(canvas,(0,504),(960,540),(5,7,9),-1)
        cv2.line(canvas,(0,504),(960,504),(30,35,40),1)
        tip=("Walk every part of the room slowly. Do different poses."
             if recording else
             f"Samples: {len(samples)}. R to record more.")
        cv2.putText(canvas,tip,(12,526),cv2.FONT_HERSHEY_SIMPLEX,
                    0.36,DIM,1,cv2.LINE_AA)
        cv2.imshow("CSI Collector",canvas)
        key=cv2.waitKey(1)&0xFF
        if key==ord("q"):break
        if key==ord("r"):
            if not recording:
                recording=True;frame_no=0
                print(f"[REC] Recording session {session_id}")
            else:
                recording=False
                out_path=f"{args.out}/session_{session_id}.json"
                with open(out_path,"w") as f:
                    json.dump({"samples":samples,"nodes":args.nodes,
                               "recorded":time.time()},f)
                print(f"[REC] Saved {len(samples)} samples to {out_path}")
                session_id=int(time.time());samples=[]
    if recording and samples:
        out_path=f"{args.out}/session_{session_id}.json"
        with open(out_path,"w") as f:
            json.dump({"samples":samples,"nodes":args.nodes,
                       "recorded":time.time()},f)
        print(f"[REC] Auto-saved {len(samples)} samples")
    rx.stop();cap.release();cv2.destroyAllWindows()

if __name__=="__main__":
    main()
