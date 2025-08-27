# test3.py — One-file, layered architecture with contracts, FSM, and full metrics/logging
# Run: streamlit run test3.py

from __future__ import annotations
import os, math, time, threading, queue, json, pathlib, typing as T
from dataclasses import dataclass
import numpy as np
import streamlit as st

# --- Optional deps
try: import cv2
except Exception: cv2 = None
try:
    import mediapipe as mp
except Exception:
    mp = None
try:
    import pyautogui
except Exception:
    pyautogui = None
try:
    import pandas as pd
except Exception:
    pd = None

from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
from av import VideoFrame

# ========================= Contracts (dataclasses) =========================
@dataclass
class Features:
    eye_cx_norm: float
    eye_cy_norm: float
    face_cx_norm: float
    face_cy_norm: float
    eye_open: bool
    quality: float

@dataclass
class GazePoint:
    x: float
    y: float

@dataclass
class Metrics:
    fps: float
    latency_ms: float
    model: str

@dataclass
class CalibrationReport:
    n_points: int
    rmse_px: float | None
    rmse_cv_px: float | None
    mae_px: float | None
    rmse_norm: float | None
    mae_norm: float | None
    uniformity: float
    width: int
    height: int
    def passed(self, min_points=9, umin=0.55, rmse_px_max=30, rmse_cv_max=35) -> bool:
        if self.n_points < min_points: return False
        if self.uniformity < umin: return False
        if self.rmse_px is None or self.rmse_cv_px is None: return False
        return (self.rmse_px <= rmse_px_max and self.rmse_cv_px <= rmse_cv_max)

# ========================= Utils =========================
def _safe_size() -> tuple[int, int]:
    if pyautogui:
        try: return pyautogui.size()
        except Exception: pass
    return 1920, 1080

def _now_iso() -> str:
    import datetime as dt
    return dt.datetime.now().isoformat(timespec="seconds")

# High-resolution timer
def _now_perf() -> float:
    return time.perf_counter()

# ========================= Filters / Controllers =========================
class OneEuro:
    def __init__(self, freq=120.0, mincutoff=1.0, beta=0.01, dcutoff=1.0):
        self.freq=freq; self.mincutoff=mincutoff; self.beta=beta; self.dcutoff=dcutoff
        self.x_prev=None; self.dx_prev=0.0; self.t_prev=None
    def _alpha(self, cutoff):
        te = 1.0/max(1e-6,self.freq); tau=1.0/(2*math.pi*cutoff)
        return 1.0/(1.0+tau/te)
    def filter(self, x, t=None):
        tnow=_now_perf() if t is None else t
        if self.t_prev is None: self.t_prev=tnow; self.x_prev=x; return x
        dt=max(1e-6,tnow-self.t_prev); self.freq=1.0/dt; self.t_prev=tnow
        dx=(x-self.x_prev)*self.freq
        a_d=self._alpha(self.dcutoff)
        dxh=a_d*dx + (1-a_d)*self.dx_prev
        cutoff=self.mincutoff + self.beta*abs(dxh)
        a=self._alpha(cutoff)
        xh=a*x+(1-a)*self.x_prev
        self.x_prev=xh; self.dx_prev=dxh
        return xh

class MouseController:
    def __init__(self, enable=False, dwell_ms=700, dwell_radius_px=40):
        self.sw,self.sh=_safe_size()
        self.enabled=enable
        self.dwell_ms=dwell_ms
        self.dwell_radius_px=dwell_radius_px
        self._last_in=None; self._last_tgt=None
        self._last_move=_now_perf()
    def set_enable(self, v: bool):
        self.enabled=v
        if not v: self._last_in=None; self._last_tgt=None
    def update(self, x_norm: float, y_norm: float, do_click=True):
        if not self.enabled: self._last_in=None; return
        x_px=int(x_norm*self.sw); y_px=int(y_norm*self.sh)
        now=_now_perf()
        if (now-self._last_move)*1000.0>=33:
            if pyautogui is not None:
                try: pyautogui.moveTo(x_px,y_px)
                except Exception: pass
            self._last_move=now
        tgt=(x_px,y_px)
        if (self._last_tgt is None) or (math.hypot(tgt[0]-(self._last_tgt[0] if self._last_tgt else 0), tgt[1]-(self._last_tgt[1] if self._last_tgt else 0))>self.dwell_radius_px):
            self._last_tgt=tgt; self._last_in=_now_perf(); return
        if do_click and self._last_in is not None:
            if (_now_perf()-self._last_in)*1000.0>=self.dwell_ms:
                if pyautogui is not None:
                    try: pyautogui.click()
                    except Exception: pass
                self._last_in=None

# ========================= Trackers / Extractors =========================
class Kalman2D:
    def __init__(self, x0, y0, vx0=0.0, vy0=0.0):
        self.x=np.array([[x0],[y0],[vx0],[vy0]],dtype=np.float32)
        self.P=np.eye(4,dtype=np.float32)*100.0
        self.Q=np.diag([1e-2,1e-2,5e-1,5e-1]).astype(np.float32)
        self.R=np.diag([3.0,3.0]).astype(np.float32)
    def predict(self, dt):
        F=np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]],dtype=np.float32)
        self.x=F@self.x; self.P=F@self.P@F.T + self.Q
    def update(self, zx, zy, mscale=1.0):
        H=np.array([[1,0,0,0],[0,1,0,0]],dtype=np.float32)
        z=np.array([[zx],[zy]],dtype=np.float32)
        R=self.R*(mscale**2)
        y=z-H@self.x; S=H@self.P@H.T + R; K=self.P@H.T@np.linalg.inv(S)
        self.x=self.x + K@y; self.P=(np.eye(4,dtype=np.float32)-K@H)@self.P
    def state(self): return float(self.x[0,0]), float(self.x[1,0])

class IrisTracker:
    """หา iris center แบบไม่พึ่งโมเดลหนัก ใช้ threshold/contour + Hough + Kalman"""
    def __init__(self):
        self.kf: Kalman2D | None = None
        self.last_t = _now_perf()
        self.roi = 180
        self.cxcy = None
        self.det_hist: list[tuple[float,int]] = []
    @staticmethod
    def _crop_aspect(img, w=640, h=480):
        if img is None: return None
        H,W=img.shape[:2]; desired=w/h; cur=W/H
        if cur>desired:
            newW=int(desired*H); off=(W-newW)//2; img=img[:,off:off+newW]
        else:
            newH=int(W/desired); off=(H-newH)//2; img=img[off:off+newH,:]
        return cv2.resize(img,(w,h)) if cv2 is not None else img
    def detect(self, frame_bgr) -> tuple[float,float] | None:
        if cv2 is None or frame_bgr is None: return None
        img=self._crop_aspect(frame_bgr); h,w=img.shape[:2]
        if self.cxcy is None: self.cxcy=(w//2, h//2)
        cx,cy=self.cxcy; roi=int(max(80,min(320,self.roi)))
        x0=max(0,cx-roi//2); y0=max(0,cy-roi//2); x1=min(w,x0+roi); y1=min(h,y0+roi)
        x0=max(0,min(x0,w-(x1-x0))); y0=max(0,min(y0,h-(y1-y0))); x1=min(w,x0+roi); y1=min(h,y0+roi)
        roi_img=img[y0:y1,x0:x1]
        if roi_img.size==0: roi_img=img; x0=y0=0; x1=w; y1=h
        g=cv2.cvtColor(roi_img,cv2.COLOR_BGR2GRAY)
        try:
            g=cv2.createCLAHE(2.0,(8,8)).apply(g)
        except Exception: pass
        g=cv2.GaussianBlur(g,(5,5),0)
        gx=cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3)
        gy=cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3)
        gmag=cv2.magnitude(gx,gy)
        hi=float(np.percentile(g,96)); med=float(np.median(g)); mask=g>hi
        if np.any(mask):
            g=g.copy(); g[mask]=int(med)
        # threshold
        thr=int(min(255,max(0,20+8)))
        _,th=cv2.threshold(g,thr,255,cv2.THRESH_BINARY_INV)
        th=cv2.morphologyEx(th,cv2.MORPH_OPEN,np.ones((3,3),np.uint8),1)
        th=cv2.morphologyEx(th,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8),1)
        cnts,_=cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        best=None; bscore=-1.0
        for c in cnts:
            area=cv2.contourArea(c)
            if area<300 or area>(roi*roi*0.55): continue
            if len(c)>=5:
                (cx_r,cy_r),(MA,ma),_ = cv2.fitEllipse(c)
                if ma<1: continue
                ratio=max(MA,ma)/max(1e-6,min(MA,ma))
                if ratio>3.5: continue
                per=cv2.arcLength(c,True); circ=4*math.pi*area/(per*per+1e-6)
                gc=[]
                for i in range(0,len(c),5):
                    px,py=int(c[i,0,0]),int(c[i,0,1])
                    if 1<=px<g.shape[1]-1 and 1<=py<g.shape[0]-1:
                        gxv=float(gx[py,px]); gyv=float(gy[py,px]); mag=float(gmag[py,px])+1e-6
                        rx=float(cx_r)-px; ry=float(cy_r)-py; rmag=math.hypot(rx,ry)+1e-6
                        cos=(( -gxv)*rx + (-gyv)*ry)/(mag*rmag)
                        gc.append(max(0.0,min(1.0,(cos+1.0)/2.0)))
                gcons=float(np.mean(gc)) if gc else 0.5
                if gcons<0.35: continue
                score=area*circ*(0.5+0.5*gcons)
                if score>bscore: bscore=score; best=(int(cx_r),int(cy_r))
        used_darkest=False
        if best is None:
            try:
                circles=cv2.HoughCircles(g,cv2.HOUGH_GRADIENT,dp=1.5,minDist=30,param1=60,param2=18,minRadius=8,maxRadius=int(roi*0.45))
                if circles is not None and len(circles[0])>0:
                    c0=circles[0][0]; best=(int(c0[0]),int(c0[1]))
            except Exception: pass
        if best is None:
            # fallback darkest window
            gf=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            step,win=10,20; best_sum=1e18; bx=by=None
            for yy in range(y0,y1-win,step):
                for xx in range(x0,x1-win,step):
                    s=float(gf[yy:yy+win,xx:xx+win].sum())
                    if s<best_sum: best_sum=s; bx=xx+win//2; by=yy+win//2
            if bx is None: return None
            cx,cy=bx,by; used_darkest=True
        else:
            cx,cy=x0+best[0], y0+best[1]
        # Kalman
        now=_now_perf(); dt=max(1e-3, now-self.last_t); self.last_t=now
        if self.kf is None: self.kf=Kalman2D(cx,cy)
        self.kf.predict(dt)
        conf = 0.25 if used_darkest else float(min(1.0,max(0.0, bscore/(roi*roi*5.0)))) if bscore>0 else 0.25
        mscale = 2.5 if used_darkest else float(np.interp(conf,[0.25,0.6,0.9],[1.8,1.0,0.7]))
        self.kf.update(cx,cy,mscale)
        cx_kf,cy_kf=self.kf.state()
        self.cxcy=(int(cx_kf),int(cy_kf))
        # adapt ROI
        if conf<0.45 or used_darkest: self.roi=min(320,int(self.roi*1.12+6))
        elif conf>0.65: self.roi=max(120,int(self.roi*0.92))
        # output normalized
        nx=float(self.cxcy[0])/float(w); ny=float(self.cxcy[1])/float(h)
        return (min(1.0,max(0.0,nx)), min(1.0,max(0.0,ny)))

class ESP32Reader(threading.Thread):
    """อ่านสตรีม MJPEG ของ ESP32cam ผ่าน OpenCV แล้วส่ง Features เข้า Queue"""
    def __init__(self, url: str, out_q: "queue.Queue[Features]", stop_event: threading.Event, iris: IrisTracker):
        super().__init__(daemon=True)
        self.url = url
        self.q = out_q
        self.stop = stop_event
        self.iris = iris

    def run(self):
        if cv2 is None:
            return
        while not self.stop.is_set():
            cap = None
            try:
                cap = cv2.VideoCapture(self.url)
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                if not cap.isOpened():
                    time.sleep(1.0)
                    continue
                while not self.stop.is_set():
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                    res = self.iris.detect(frame)
                    if res is not None:
                        nx, ny = res
                        feat = Features(nx, ny, 0.5, 0.5, True, 0.5)  # face=0.5 (ไม่ใช้), quality~กลาง
                        try:
                            self.q.put(feat, timeout=0.01)
                        except Exception:
                            pass
            except Exception:
                time.sleep(0.5)
            finally:
                if cap is not None:
                    cap.release()

class FeatureExtractor:
    def __init__(self, use_mediapipe=True):
        self.use = use_mediapipe and (mp is not None) and (cv2 is not None)
        if self.use:
            self.mesh=mp.solutions.face_mesh.FaceMesh(static_image_mode=False,refine_landmarks=True,max_num_faces=1,
                                                      min_detection_confidence=0.5,min_tracking_confidence=0.5)
        else:
            self.mesh=None
        self._emaL=self._emaR=None; self._alpha=0.05
        # landmark ids
        self.LI=[468,469,470,471,472]; self.RI=[473,474,475,476,477] if self.use else self.LI
        self.LE=[33,133,159,145]; self.RE=[362,263,386,374]
    def close(self):
        try:
            if self.mesh: self.mesh.close()
        except Exception: pass
    @staticmethod
    def _avg(ids, pts):
        xs=[pts[i][0] for i in ids if i<len(pts)]; ys=[pts[i][1] for i in ids if i<len(pts)]
        if not xs or not ys: return None
        return (sum(xs)/len(xs), sum(ys)/len(ys))
    @staticmethod
    def _eye_box(ids, pts):
        xs=[pts[i][0] for i in ids if i<len(pts)]; ys=[pts[i][1] for i in ids if i<len(pts)]
        if not xs or not ys: return None
        x0,x1=min(xs),max(xs); y0,y1=min(ys),max(ys)
        w=max(6.0,x1-x0); h=max(2.0,y1-y0); ear=h/max(6.0,w)
        return (x0,x1,y0,y1,w,h,ear)
    def _norm_in(self, p, box):
        if p is None or box is None: return (0.5,0.5)
        x0,x1,y0,y1,w,h,_=box
        nx=(p[0]-x0)/w; ny=(p[1]-y0)/h
        return (min(1.0,max(0.0,nx)), min(1.0,max(0.0,ny)))
    def extract(self, frame_bgr) -> Features | None:
        if frame_bgr is None: return None
        if not self.use:
            return Features(0.5,0.5,0.5,0.5,True,0.2)
        h,w=frame_bgr.shape[:2]
        res=self.mesh.process(cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks: return None
        lm=res.multi_face_landmarks[0]
        pts=[(lm.landmark[i].x*w, lm.landmark[i].y*h) for i in range(len(lm.landmark))]
        l_iris=self._avg(self.LI,pts); r_iris=self._avg(self.RI,pts)
        l_box=self._eye_box(self.LE,pts); r_box=self._eye_box(self.RE,pts)
        l_n=self._norm_in(l_iris,l_box) if l_iris else (0.5,0.5)
        r_n=self._norm_in(r_iris,r_box) if r_iris else (0.5,0.5)
        earL=l_box[6] if l_box else None; earR=r_box[6] if r_box else None
        def _upd(cur,base): 
            if cur is None: return base
            return cur if base is None else (base*(1.0-self._alpha)+cur*self._alpha)
        if earL and earL>0.18: self._emaL=_upd(earL,self._emaL)
        if earR and earR>0.18: self._emaR=_upd(earR,self._emaR)
        def _eye_conf(ear, base, box, ok):
            if (ear is None) or (box is None) or (not ok): return 0.0
            ratio = (ear/0.26) if base is None else (ear/max(1e-6,0.8*base)); ratio=max(0.0,min(1.4,ratio))
            bw,bh=box[4],box[5]; area=((bw*bh)/(w*h))
            return max(0.0,min(1.0, 0.7*ratio + 0.3*min(1.0, area/0.02)))
        cL=_eye_conf(earL,self._emaL,l_box,l_iris is not None)
        cR=_eye_conf(earR,self._emaR,r_box,r_iris is not None)
        wL,wR=cL,cR
        if (wL+wR)<1e-3:
            ex,ey=0.5,0.5
        else:
            ex=float((l_n[0]*wL + r_n[0]*wR)/(wL+wR))
            ey=float((l_n[1]*wL + r_n[1]*wR)/(wL+wR))
        face_ids=[1,9,152,33,263]
        fxs=[pts[i][0] for i in face_ids if i<len(pts)]; fys=[pts[i][1] for i in face_ids if i<len(pts)]
        fcx=float(sum(fxs)/len(fxs)/max(1,w)) if fxs else 0.5
        fcy=float(sum(fys)/len(fys)/max(1,h)) if fys else 0.5
        margin=min(ex,1-ex,ey,1-ey); quality=max(cL,cR)*(0.6+0.4*max(0.0,min(1.0,margin*2)))
        return Features(ex,ey,fcx,fcy, bool((earL or 0.0)>0.2 or (earR or 0.0)>0.2), float(quality))

# ========================= Model / Mapping =========================
def _poly_expand(X: np.ndarray) -> np.ndarray:
    if X.ndim==1: X=X[None,:]
    cols=[X]; n=X.shape[1]
    for i in range(n):
        for j in range(i,n):
            cols.append((X[:,i]*X[:,j])[:,None])
    return np.concatenate(cols,axis=1)

def _rmse_px(pred_xy, true_xy, W, H) -> float:
    dx=(pred_xy[:,0]-true_xy[:,0])*W; dy=(pred_xy[:,1]-true_xy[:,1])*H
    return float(np.sqrt(np.mean(dx*dx + dy*dy)))

def _mae_px(pred_xy, true_xy, W, H) -> float:
    dx=np.abs((pred_xy[:,0]-true_xy[:,0])*W); dy=np.abs((pred_xy[:,1]-true_xy[:,1])*H)
    return float(np.mean(np.sqrt(dx*dx + dy*dy)))

def _uniformity_score(pts_norm: np.ndarray) -> float:
    pts=np.asarray(pts_norm,float)
    if len(pts)<3: return 0.0
    pts=pts[np.lexsort((pts[:,1],pts[:,0]))]
    def cross(o,a,b): return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower=[]; upper=[]
    for p in pts:
        while len(lower)>=2 and cross(lower[-2], lower[-1], p)<=0: lower.pop()
        lower.append(p)
    for p in pts[::-1]:
        while len(upper)>=2 and cross(upper[-2], upper[-1], p)<=0: upper.pop()
        upper.append(p)
    hull=np.vstack([lower[:-1], upper[:-1]])
    area=0.0
    for i in range(len(hull)):
        x1,y1=hull[i]; x2,y2=hull[(i+1)%len(hull)]
        area+=x1*y2 - x2*y1
    return float(max(0.0, min(1.0, abs(area)/2.0)))

class GazeModel:
    """แปลง Features -> (x,y) ด้วยโพลีโนเมียล + compensation (bias/gain)"""
    def __init__(self):
        self.wx=None; self.wy=None
        self.comp=(1.0,0.0,1.0,0.0); self.comp_valid=False
    @staticmethod
    def feat_vec(f: Features) -> np.ndarray:
        ex,ey=float(f.eye_cx_norm),float(f.eye_cy_norm)
        fx,fy=float(f.face_cx_norm),float(f.face_cy_norm)
        fx = 0.5 + (fx-0.5)*0.3; fy = 0.5 + (fy-0.5)*0.3
        return np.array([ex,ey,fx,fy,1.0],dtype=np.float32)
    def fit(self, X: np.ndarray, Y: np.ndarray):
        Phi=_poly_expand(X); lam=1e-3
        A=Phi.T@Phi + lam*np.eye(Phi.shape[1])
        self.wx=np.linalg.solve(A, Phi.T@Y[:,0])
        self.wy=np.linalg.solve(A, Phi.T@Y[:,1])
        # fit comp
        pred=self.predict_batch(X, apply_comp=False)
        def fit_axis(p,t):
            p=np.asarray(p,float); t=np.asarray(t,float)
            var=np.var(p); 
            if var<1e-6: return 1.0,0.0
            cov=np.cov(p,t,bias=True)[0,1]; a=float(np.clip(cov/var,0.8,1.3))
            b=float(np.clip(np.mean(t)-a*np.mean(p), -0.15, 0.15))
            return a,b
        ax,bx=fit_axis(pred[:,0],Y[:,0]); ay,by=fit_axis(pred[:,1],Y[:,1])
        self.comp=(ax,bx,ay,by); self.comp_valid=True
    def predict_batch(self, X: np.ndarray, apply_comp=True) -> np.ndarray:
        Phi=_poly_expand(X); pred=np.stack([Phi@self.wx, Phi@self.wy],axis=1).astype(np.float32)
        if apply_comp and self.comp_valid:
            ax,bx,ay,by=self.comp
            pred[:,0]=ax*pred[:,0]+bx; pred[:,1]=ay*pred[:,1]+by
        return np.clip(pred,0.0,1.0)
    def predict_one(self, v: np.ndarray, apply_comp=True) -> np.ndarray:
        return self.predict_batch(v[None,:], apply_comp=apply_comp)[0]

# ========================= Calibrator FSM =========================
class CalibratorFSM:
    """IDLE -> COLLECT -> FIT -> REPORT (+optional AUTO_EXTEND)"""
    def __init__(self):
        self.state="IDLE"
        self.targets: list[tuple[float,float]]=[]
        self.idx=0
        self.X=[]; self.Y=[]; self.W=[]
        self.report: CalibrationReport | None = None
        self.auto_extended=False
    def start(self, targets: list[tuple[float,float]]):
        self.targets=list(targets); self.idx=0; self.X=[]; self.Y=[]; self.W=[]
        self.state="COLLECT"; self.report=None
    def add(self, feat_vec: np.ndarray, target: tuple[float,float], quality: float):
        if self.state!="COLLECT": return
        sx,sy=target
        q=max(0.0,min(1.0,(quality-0.20)/0.80))**2
        w=0.5+1.5*q
        self.X.append(feat_vec.tolist()); self.Y.append([float(sx),float(sy)])
        # ให้ความสำคัญกับมุม/ขอบมากขึ้น
        w *= (1.0 + 0.6*abs(sx-0.5)*2.0 + 0.6*abs(sy-0.5)*2.0) * 0.5
        self.W.append(float(w))
        self.idx += 1
        if self.idx>=len(self.targets):
            self.state="FIT"
    def fit_and_report(self, model: GazeModel, screen_wh: tuple[int,int]) -> CalibrationReport:
        self.state="FIT"
        X=np.array(self.X,dtype=np.float32); Y=np.array(self.Y,dtype=np.float32)
        model.fit(X,Y)
        pred=model.predict_batch(X, apply_comp=False)
        W,H=screen_wh
        rmse_px=_rmse_px(pred,Y,W,H)
        mae_px=_mae_px(pred,Y,W,H)
        rmse_norm=float(np.sqrt(np.mean((pred-Y)**2)))
        mae_norm=float(np.mean(np.sqrt(np.sum((pred-Y)**2,axis=1))))
        # CV (k-fold or holdout)
        n=len(X)
        if n>=9:
            k=max(2,min(5,n//3)); idx=np.arange(n); np.random.shuffle(idx); folds=np.array_split(idx,k)
            acc=[]
            for i in range(k):
                te=folds[i]; tr=np.concatenate([folds[j] for j in range(k) if j!=i])
                m=GazeModel(); m.fit(X[tr],Y[tr]); pr=m.predict_batch(X[te], apply_comp=False)
                acc.append(_rmse_px(pr,Y[te],W,H))
            rmse_cv=float(np.mean(acc))
        else:
            cut=max(2,int(0.8*n)); idx=np.arange(n); np.random.shuffle(idx); tr,te=idx[:cut],idx[cut:]
            m=GazeModel(); m.fit(X[tr],Y[tr]); pr=m.predict_batch(X[te], apply_comp=False)
            rmse_cv=float(_rmse_px(pr,Y[te],W,H))
        unif=_uniformity_score(Y)
        rep=CalibrationReport(n_points=n, rmse_px=rmse_px, rmse_cv_px=rmse_cv, mae_px=mae_px,
                              rmse_norm=rmse_norm, mae_norm=mae_norm, uniformity=unif, width=W, height=H)
        self.report=rep; self.state="REPORT"
        return rep

# ========================= Engine (core orchestrator) =========================
class GazeEngine:
    def __init__(self):
        self.model=GazeModel()
        self.fx=OneEuro(); self.fy=OneEuro()
        self.last_feat_vec=None
        self.comp_alpha=1.0; self.comp_enabled=True
        self.fps_hist=[]; self.t_prev=None
        self.last_metrics=Metrics(0.0,0.0,"fallback")
        self.eval_csv="gaze_eval_log.csv"
        self.calib=CalibratorFSM()
        self.gate={"rmse_train":45.0,"rmse_cv":55.0,"uniformity":0.50,"min_pts":9}
        self.screen=_safe_size()
        self.fallback_kx=1.8; self.fallback_ky=1.6
        self.deadzone=0.02; self.gain=1.2; self.gamma=1.0
        self.model_ready=False
    def set_screen(self, w:int, h:int): self.screen=(int(w),int(h))
    def set_comp(self, enabled:bool, alpha:float): self.comp_enabled=enabled; self.comp_alpha=float(max(0.0,min(1.0,alpha)))
    def set_gate(self, rmse_train, rmse_cv, uniformity, min_pts=9):
        self.gate={"rmse_train":float(rmse_train),"rmse_cv":float(rmse_cv),"uniformity":float(uniformity),"min_pts":int(min_pts)}
    def set_shape(self, gain, gamma, deadzone): self.gain=float(gain); self.gamma=float(gamma); self.deadzone=float(deadzone)
    def _shape(self, x:float, y:float) -> tuple[float,float]:
        # gain
        x=0.5+(x-0.5)*self.gain; y=0.5+(y-0.5)*self.gain
        # gamma
        if self.gamma!=1.0:
            def g(v):
                s=v-0.5; sign=1.0 if s>=0 else -1.0
                return 0.5 + sign*(abs(s)**self.gamma)
            x=g(x); y=g(y)
        # deadzone
        dz=self.deadzone
        if abs(x-0.5)<dz: x=0.5
        if abs(y-0.5)<dz: y=0.5
        return min(1.0,max(0.0,x)), min(1.0,max(0.0,y))
    def _fallback(self, ex,ey):
        return 0.5+(ex-0.5)*self.fallback_kx, 0.5+(ey-0.5)*self.fallback_ky
    def map_once(self, feat: Features) -> tuple[GazePoint, Metrics]:
        t0=_now_perf()
        fv=GazeModel.feat_vec(feat)
        # keep last good
        if self.last_feat_vec is None or feat.eye_open:
            self.last_feat_vec=fv
        # base pred
        if self.model_ready and (self.model.wx is not None):
            pm=self.model.predict_one(self.last_feat_vec, apply_comp=self.comp_enabled)
            # blend with fallback (เล็กน้อย) ให้ทนต่อ drift
            fb=np.array(self._fallback(fv[0],fv[1]),dtype=np.float32)
            px=(0.9*pm[0]+0.1*fb[0]); py=(0.85*pm[1]+0.15*fb[1])
        else:
            px,py=self._fallback(fv[0],fv[1])
        # shape
        px,py=self._shape(px,py)
        # smooth
        tnow=_now_perf(); sx=float(self.fx.filter(px,tnow)); sy=float(self.fy.filter(py,tnow))
        # metrics
        t1=_now_perf()
        if self.t_prev is not None:
            dt=t1-self.t_prev
            if dt>0:
                self.fps_hist.append(1.0/dt)
                if len(self.fps_hist)>90: self.fps_hist=self.fps_hist[-90:]
        self.t_prev=t1
        fps=float(np.mean(self.fps_hist)) if self.fps_hist else 0.0
        latency_ms=float((t1-t0)*1000.0)
        self.last_metrics=Metrics(fps,latency_ms, "calibrated" if self.model_ready else "fallback")
        return GazePoint(sx,sy), self.last_metrics
    # ---------- Calibration ----------
    def calib_start(self, targets: list[tuple[float,float]]): self.calib.start(targets)
    def calib_collect_if_ready(self, feat: Features, target: tuple[float,float], stable: bool):
        if self.calib.state!="COLLECT": return
        if not stable: return
        fv=GazeModel.feat_vec(feat)
        self.calib.add(fv, target, feat.quality)
    def calib_finish(self) -> CalibrationReport | None:
        if self.calib.state not in ("FIT","COLLECT"): return self.calib.report
        rep=self.calib.fit_and_report(self.model, self.screen)
        self.model_ready=True
        # log
        row={
            "ts": _now_iso(),
            "n_points": rep.n_points,
            "rmse_px": round(rep.rmse_px or 0.0,3),
            "rmse_cv_px": round(rep.rmse_cv_px or 0.0,3),
            "mae_px": round(rep.mae_px or 0.0,3),
            "rmse_norm": round(rep.rmse_norm or 0.0,6),
            "mae_norm": round(rep.mae_norm or 0.0,6),
            "uniformity": round(rep.uniformity,4),
            "passed": rep.passed(self.gate["min_pts"], self.gate["uniformity"], self.gate["rmse_train"], self.gate["rmse_cv"]),
            "fps": round(self.last_metrics.fps,2),
            "latency_ms": round(self.last_metrics.latency_ms,2),
            "screen_w": self.screen[0],
            "screen_h": self.screen[1],
        }
        try:
            p=pathlib.Path(self.eval_csv)
            if not p.exists():
                p.write_text(",".join(row.keys())+"\n"+",".join(str(row[k]) for k in row.keys())+"\n")
            else:
                with p.open("a") as f: f.write(",".join(str(row[k]) for k in row.keys())+"\n")
        except Exception: pass
        return rep
    def gate_ok(self) -> tuple[bool,str]:
        rep=self.calib.report
        if rep is None: return False,"ยังไม่คาลิเบรท"
        reasons=[]
        if rep.n_points < self.gate["min_pts"]: reasons.append(f"จุดน้อย ({rep.n_points}/{self.gate['min_pts']})")
        if rep.uniformity < self.gate["uniformity"]: reasons.append(f"uniformity {rep.uniformity:.2f} < {self.gate['uniformity']:.2f}")
        if (rep.rmse_px or 1e9) > self.gate["rmse_train"]: reasons.append(f"RMSE {rep.rmse_px:.0f}px > {self.gate['rmse_train']:.0f}px")
        if (rep.rmse_cv_px or 1e9) > self.gate["rmse_cv"]: reasons.append(f"CV {rep.rmse_cv_px:.0f}px > {self.gate['rmse_cv']:.0f}px")
        ok=len(reasons)==0
        return ok, ("ผ่าน" if ok else " , ".join(reasons))

# ========================= App State / UI =========================
class AppState:
    def __init__(self):
        self.mirror=False; self.invert_x=False; self.invert_y=False
        self.use_override=False; self.ov_w=_safe_size()[0]; self.ov_h=_safe_size()[1]
        self.gain=1.2; self.gamma=1.0; self.deadzone=0.02
        self.mouse_enable=False; self.dwell_ms=1200; self.dwell_radius=40
        self.filter_mincutoff=1.0; self.filter_beta=0.01
        self.comp_enable=True; self.comp_alpha=1.0
        self.gate_mode="Balanced"
        self.gate_rmse=45.0; self.gate_cv=55.0; self.gate_uni=0.50; self.gate_pts=9
        self.calib_points=12; self.calib_dwell_ms=1200
        self.dev_diag=False
        self.source = "Webcam"                   # "Webcam" หรือ "ESP32"
        self.esp32_url = "http://192.168.0.58:81/stream"
        # runtime
        self.shared=GazePoint(0.5,0.5)
        self.mouse_gate=False; self.mouse_reason="ยังไม่คาลิเบรท"
        self.fps=0.0; self.lat=0.0

if "APP" not in st.session_state:
    st.session_state["APP"]=AppState()
APP: AppState = st.session_state["APP"]

# ========================= Video Processor =========================
class GazeProcessor(VideoProcessorBase):
    def __init__(self):
        self.engine=GazeEngine()
        self.extractor=FeatureExtractor(use_mediapipe=True)
        self.iris=IrisTracker()
        self.mouse=MouseController(False, APP.dwell_ms, APP.dwell_radius)
        self.calib_active=False
        self.targets=[]; self.idx=0
        self.hold_start=None; self.pool=[]
    def _stable(self, feat: Features, x:float, y:float, last:tuple[float,float,float]):
        lx,ly,lt=last; dt=max(1e-3, _now_perf()-lt)
        speed=math.hypot(x-lx,y-ly)/dt
        return (feat.quality>=0.35) and (speed<1.2), speed
    def recv(self, frame: VideoFrame) -> VideoFrame:
        img=frame.to_ndarray(format="bgr24")
        if APP.mirror and cv2 is not None: img=cv2.flip(img,1)
        # apply config to engine
        if APP.use_override: self.engine.set_screen(APP.ov_w, APP.ov_h)
        self.engine.set_shape(APP.gain, APP.gamma, APP.deadzone)
        self.engine.set_comp(APP.comp_enable, APP.comp_alpha)
        self.engine.fx.mincutoff=float(APP.filter_mincutoff); self.engine.fx.beta=float(APP.filter_beta)
        self.engine.fy.mincutoff=float(APP.filter_mincutoff); self.engine.fy.beta=float(APP.filter_beta)
        self.engine.set_gate(APP.gate_rmse, APP.gate_cv, APP.gate_uni, APP.gate_pts)

        feat=self.extractor.extract(img)
        # primary mapping
        if feat is None:
            gp, met = GazePoint(0.5,0.5), Metrics(self.engine.last_metrics.fps, self.engine.last_metrics.latency_ms, self.engine.last_metrics.model)
        else:
            gp, met = self.engine.map_once(feat)
        x,y = gp.x, gp.y
        if APP.invert_x: x=1.0-x
        if APP.invert_y: y=1.0-y
        APP.shared=GazePoint(x,y); APP.fps=met.fps; APP.lat=met.latency_ms

        # --- Calibration overlay flow
        if self.calib_active:
            tx,ty = self.targets[self.idx] if (0<=self.idx<len(self.targets)) else (0.5,0.5)
            last = getattr(self,"_last_fix",(0.5,0.5,_now_perf()))
            stable, speed = (False,0.0) if feat is None else self._stable(feat, x,y,last)
            self._last_fix=(x,y,_now_perf())
            # draw HUD
            if cv2 is not None:
                h,w=img.shape[:2]; gx,gy=int(tx*w),int(ty*h)
                cv2.circle(img,(gx,gy),18,(0,170,255),2)
                cv2.putText(img,f"Calib {self.idx+1}/{len(self.targets)} | Q={0 if feat is None else feat.quality:.2f} | v={speed:.2f}",
                            (24,48), cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
            now=_now_perf()
            if stable and (feat is not None):
                if self.hold_start is None: self.hold_start=now; self.pool=[]
                self.pool.append(GazeModel.feat_vec(feat))
                elapsed=(now-self.hold_start)*1000.0
                if cv2 is not None:
                    h,w=img.shape[:2]; end=int(360*min(1.0, elapsed/APP.calib_dwell_ms))
                    cv2.ellipse(img,(int(tx*w),int(ty*h)),(22,22),0,0,end,(0,170,255),3)
                if elapsed>=APP.calib_dwell_ms:
                    fv=np.mean(np.stack(self.pool,axis=0),axis=0).astype(np.float32) if len(self.pool)>=5 else GazeModel.feat_vec(feat)
                    self.engine.calib.add(fv, (tx,ty), feat.quality)
                    self.idx += 1; self.hold_start=None; self.pool=[]
                    if self.idx>=len(self.targets):
                        rep=self.engine.calib_finish()
                        self.calib_active=False
            else:
                self.hold_start=None; self.pool=[]
        # gate & mouse
        ok, reason = self.engine.gate_ok()
        APP.mouse_gate, APP.mouse_reason = ok, reason
        self.mouse.set_enable(APP.mouse_enable)
        self.mouse.dwell_ms=APP.dwell_ms; self.mouse.dwell_radius_px=APP.dwell_radius
        self.mouse.update(x,y, do_click=(APP.mouse_enable and ok))

        # overlay gaze + metrics
        if cv2 is not None:
            h,w=img.shape[:2]; gx,gy=int(x*w),int(y*h)
            cv2.circle(img,(gx,gy),8,(0,255,0),2)
            cv2.line(img,(gx-15,gy),(gx+15,gy),(0,255,0),1)
            cv2.line(img,(gx,gy-15),(gx,gy+15),(0,255,0),1)
            info=f"FPS {APP.fps:4.1f} | Latency {APP.lat:4.1f} ms"
            cv2.putText(img,info,(w-460,28),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
        return VideoFrame.from_ndarray(img, format="bgr24")

# ========================= UI =========================
def sidebar():
    st.sidebar.title("⚙️ Controls (one-file, layered)")
    sw,sh=_safe_size()
    # Screen
    st.sidebar.subheader("🖥️ Screen")
    APP.use_override=st.sidebar.checkbox("Override screen size", value=APP.use_override)
    c1,c2=st.sidebar.columns(2)
    APP.ov_w=int(c1.number_input("Width", min_value=320, max_value=10000, value=int(APP.ov_w)))
    APP.ov_h=int(c2.number_input("Height", min_value=240, max_value=10000, value=int(APP.ov_h)))
    st.sidebar.caption(f"OS reports: {sw}×{sh}px")
    # Shaping
    st.sidebar.subheader("🎛 Shaping")
    APP.gain=float(st.sidebar.slider("Gain",0.5,2.5,APP.gain,0.05))
    APP.gamma=float(st.sidebar.slider("Gamma",0.5,2.0,APP.gamma,0.05))
    APP.deadzone=float(st.sidebar.slider("Deadzone",0.0,0.1,APP.deadzone,0.005))
    # Mouse
    st.sidebar.subheader("🖱️ Mouse")
    APP.mouse_enable=st.sidebar.toggle("Enable Mouse Control", value=APP.mouse_enable)
    APP.dwell_ms=st.sidebar.slider("Dwell Click (ms)",200,2000,APP.dwell_ms,50)
    APP.dwell_radius=st.sidebar.slider("Dwell Radius (px)",10,120,APP.dwell_radius,2)
    # Axes
    st.sidebar.subheader("🪞 Axes")
    APP.mirror=st.sidebar.checkbox("Mirror webcam", value=APP.mirror)
    APP.invert_x=st.sidebar.checkbox("Invert X", value=APP.invert_x)
    APP.invert_y=st.sidebar.checkbox("Invert Y", value=APP.invert_y)
    # Filter
    st.sidebar.markdown("---")
    st.sidebar.subheader("🪄 One Euro Filter")
    APP.filter_mincutoff=float(st.sidebar.slider("mincutoff",0.05,3.0,APP.filter_mincutoff,0.05))
    APP.filter_beta=float(st.sidebar.slider("beta",0.0,1.0,APP.filter_beta,0.01))
    # Compensation
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Compensation")
    APP.comp_enable=st.sidebar.toggle("Enable compensation", value=APP.comp_enable)
    APP.comp_alpha=float(st.sidebar.slider("Strength (alpha)",0.0,1.0,APP.comp_alpha,0.05))
    # Gate
    st.sidebar.markdown("---")
    st.sidebar.subheader("✅ Click Gate")
    mode=st.sidebar.selectbox("Mode",["Strict","Balanced","Lenient"], index=["Strict","Balanced","Lenient"].index(APP.gate_mode))
    APP.gate_mode=mode
    if mode=="Strict":
        APP.gate_rmse=30.0; APP.gate_cv=35.0; APP.gate_uni=0.55; APP.gate_pts=9
    elif mode=="Balanced":
        APP.gate_rmse=45.0; APP.gate_cv=55.0; APP.gate_uni=0.50; APP.gate_pts=9
    else:
        APP.gate_rmse=60.0; APP.gate_cv=75.0; APP.gate_uni=0.45; APP.gate_pts=9
    st.sidebar.caption(f"RMSE≤{APP.gate_rmse}px / CV≤{APP.gate_cv}px · U≥{APP.gate_uni}")
    # Calibration
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧭 Calibration")
    APP.calib_points=st.sidebar.selectbox("Points",[9,12], index=1)
    APP.calib_dwell_ms=st.sidebar.slider("Dwell per target (ms)",400,2000,APP.calib_dwell_ms,50)
    if st.sidebar.button("Start Calibration"):
        if APP.calib_points==9:
            grid=[(0.50,0.50),
                  (0.30,0.50),(0.70,0.50),
                  (0.50,0.30),(0.50,0.70),
                  (0.20,0.20),(0.80,0.20),(0.20,0.80),(0.80,0.80)]
        else:
            grid=[(0.50,0.50),
                  (0.30,0.50),(0.70,0.50),
                  (0.50,0.30),(0.50,0.70),
                  (0.15,0.15),(0.85,0.15),(0.15,0.85),(0.85,0.85),
                  (0.30,0.30),(0.70,0.30),(0.50,0.70)]
        vp=st.session_state.get("webrtc_ctx")
        if vp and vp.video_processor:
            vp.video_processor.engine.calib_start(grid)
            vp.video_processor.targets=grid
            vp.video_processor.idx=0
            vp.video_processor.calib_active=True
            st.toast("เริ่มคาลิเบรทแล้ว")

    # Dev diag
    st.sidebar.markdown("---")
    APP.dev_diag=st.sidebar.toggle("Developer diagnostics", value=APP.dev_diag)

def main():
    st.set_page_config(page_title="👀 Gaze One-File (Layered)", layout="wide")
    st.title("👀 Gaze One-File • Layered Architecture + FSM + Metrics")
    st.caption("POC ที่ยัง one-file แต่โครงชัด • FPS/Latency overlay • RMSE/MAE logging (CSV) • Gate/Screen Override")

    sidebar()

    ctx=webrtc_streamer(
        key="gaze",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video":{"width":{"ideal":640},"height":{"ideal":480},"frameRate":{"ideal":60,"min":30}},"audio":False},
        async_processing=False,
        video_processor_factory=GazeProcessor,
    )
    st.session_state["webrtc_ctx"]=ctx

    # Metrics row
    c1,c2,c3,c4,c5,c6=st.columns(6)
    with c1: st.metric("Mouse", "ON ✅" if (APP.mouse_enable and APP.mouse_gate) else ("ON ⛔" if APP.mouse_enable else "OFF"))
    with c2: st.metric("Mode", "Calibrated" if APP.mouse_gate else "Fallback")
    with c3: st.metric("Gate", "PASS" if APP.mouse_gate else "LOCK")
    with c4: st.metric("FPS", f"{APP.fps:.1f}")
    with c5: st.metric("Latency (ms)", f"{APP.lat:.1f}")
    with c6:
        sw = APP.ov_w if APP.use_override else _safe_size()[0]
        sh = APP.ov_h if APP.use_override else _safe_size()[1]
        st.metric("Screen", f"{sw}×{sh}")

    st.markdown("### Diagnostics")
    st.write({
        "gaze": (round(APP.shared.x,3), round(APP.shared.y,3)),
        "gate_reason": APP.mouse_reason,
        "mirror": APP.mirror, "invert_x": APP.invert_x, "invert_y": APP.invert_y,
        "comp": {"enabled": APP.comp_enable, "alpha": APP.comp_alpha},
    })

    # Calibration report + Log viewer
    if ctx and ctx.video_processor:
        eng=ctx.video_processor.engine
        rep=eng.calib.report
        st.subheader("Calibration Report")
        if rep is None:
            st.info("Uncalibrated — คลิกถูกล็อกจนกว่าจะคาลิเบรทผ่านเกณฑ์")
        else:
            ok, reason = eng.gate_ok()
            badge = "✅ PASS" if ok else "⛔ LOCK"
            st.markdown(f"- {badge} · RMSE(train): **{rep.rmse_px:.0f}px**, RMSE(CV): **{rep.rmse_cv_px:.0f}px**, "
                        f"MAE: **{rep.mae_px:.0f}px**, uniformity: **{rep.uniformity:.2f}**, points: **{rep.n_points}**")
            if not ok: st.caption(f"เหตุผล: {reason}")

        st.subheader("Evaluation Log (CSV)")
        p=pathlib.Path(eng.eval_csv)
        if p.exists() and pd is not None:
            try:
                df=pd.read_csv(p)
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.download_button("⬇️ Download CSV", data=df.to_csv(index=False).encode("utf-8"),
                                   file_name="gaze_eval_log.csv", mime="text/csv")
            except Exception:
                st.caption("อ่าน CSV ไม่ได้ (เปิดอยู่หรือไฟล์ล็อก?)")
        else:
            st.caption("ยังไม่มีบันทึก (ทำคาลิเบรทให้จบก่อน)")

    st.markdown("---")
    st.markdown(
        """
**สรุปสถาปัตย์ในไฟล์เดียว**
- `IrisTracker` แยก logic ตรวจ iris + Kalman (ลด state ซ่อน)
- `FeatureExtractor` รับผิดชอบ MediaPipe → `Features`
- `GazeModel` โพลีโนเมียล + compensation (เรียนรู้หลังคาลิเบรต)
- `CalibratorFSM` ทำ state machine เก็บ → fit → report
- `GazeEngine` เป็น orchestrator core (map/shape/smooth/metrics/log)
- `GazeProcessor` เป็นวงลูปเฟรม (UI overlay + mouse + gate)
        """
    )

if __name__ == "__main__":
    main()