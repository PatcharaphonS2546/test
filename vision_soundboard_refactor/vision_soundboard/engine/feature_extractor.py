from __future__ import annotations
import math
import numpy as np
from typing import Optional

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:
    import mediapipe as mp  # type: ignore
except Exception:  # pragma: no cover
    mp = None

class FeatureExtractor:
    LEFT_EYE_IDS  = [33, 133, 159, 145]
    RIGHT_EYE_IDS = [362, 263, 386, 374]
    LEFT_IRIS_IDS  = [468, 469, 470, 471, 472]
    RIGHT_IRIS_IDS = [473, 474, 475, 476, 477]

    _debug_logged = False

    def __init__(self, use_mediapipe: bool = True):
        self.use_mediapipe = use_mediapipe and (mp is not None) and (cv2 is not None)
        self.face_landmarks = None
        self.mesh = None
        self._initialized = False
        self.earL_base = 0.26
        self.earR_base = 0.26

    def _ensure_mesh(self):
        if not self._initialized and self.use_mediapipe:
            self.mp_face = mp.solutions.face_mesh
            self.mesh = self.mp_face.FaceMesh(
                static_image_mode=False,
                refine_landmarks=True,
                max_num_faces=1,
                min_detection_confidence=0.1,
                min_tracking_confidence=0.1,
            )
            self._initialized = True

    def close(self):
        if self.mesh is not None:
            self.mesh.close()

    @staticmethod
    def _median_xy(ids, pts, outlier_thresh=2.5):
        import numpy as np
        arr = np.array([pts[i][:2] for i in ids if i < len(pts)])
        if arr.shape[0] == 0:
            return None
        mx, my = np.median(arr[:,0]), np.median(arr[:,1])
        mad_x = np.median(np.abs(arr[:,0] - mx)) if arr.shape[0] else 0.0
        mad_y = np.median(np.abs(arr[:,1] - my)) if arr.shape[0] else 0.0
        mask_x = (mad_x == 0.0) | (np.abs(arr[:,0] - mx) / (mad_x + 1e-6) < outlier_thresh)
        mask_y = (mad_y == 0.0) | (np.abs(arr[:,1] - my) / (mad_y + 1e-6) < outlier_thresh)
        arr_filt = arr[mask_x & mask_y]
        if arr_filt.shape[0] == 0:
            return (mx, my)
        return (float(np.median(arr_filt[:,0])), float(np.median(arr_filt[:,1])))

    @staticmethod
    def _eye_box_metrics(ids, pts):
        arr = np.array([pts[i][:2] for i in ids if i < len(pts)])
        if arr.shape[0] == 0:
            return None
        xs = arr[:, 0]
        ys = arr[:, 1]
        x0, x1 = np.min(xs), np.max(xs)
        y0, y1 = np.min(ys), np.max(ys)
        w = max(6.0, x1 - x0)
        h = max(2.0, y1 - y0)
        ear = h / max(6.0, w)  # eye-aspect ratio
        return (x0, x1, y0, y1, w, h, ear)

    @staticmethod
    def _norm_in_box(p, box):
        if (p is None) or (box is None):
            return (0.5, 0.5)
        x0, x1, y0, y1, w, h, _ = box
        # Robust scale factor: ไม่ให้ w/h ต่ำเกินไป
        w = max(w, 6.0)
        h = max(h, 2.0)
        nx = (p[0] - x0) / w
        ny = (p[1] - y0) / h
        # Clamp ค่าให้อยู่ในช่วง 0-1
        nx = min(1.0, max(0.0, nx))
        ny = min(1.0, max(0.0, ny))
        return (nx, ny)

    def extract(self, frame_bgr) -> Optional[dict]:
        def _fallback(q=0.2, open_=False):
            return dict(
                eye_cx_norm=0.5, eye_cy_norm=0.5,
                face_cx_norm=0.5, face_cy_norm=0.5,
                eye_open=bool(open_), quality=float(q),
                yaw=0.0, pitch=0.0
            )

        if frame_bgr is None:
            return _fallback(0.2, True)

        if self.use_mediapipe:
            self._ensure_mesh()

        if (not self.use_mediapipe) or (self.mesh is None):
            return _fallback(0.1, False)

        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(frame_rgb)

        if not res.multi_face_landmarks:
            self.face_landmarks = None
            return _fallback(0.05, False)

        lm = res.multi_face_landmarks[0]
        self.face_landmarks = lm
        lmk = lm.landmark
        pts = [(p.x * w, p.y * h, p.z) for p in lmk]

        # Quality check: landmark count and confidence
        if not FeatureExtractor._debug_logged:
            print(f'[FeatureExtractor] frame shape: {frame_bgr.shape if frame_bgr is not None else None}')
            print(f'[FeatureExtractor] landmarks detected: {len(lmk)}')
        if len(lmk) < 468:
            if not FeatureExtractor._debug_logged:
                print('[FeatureExtractor] Fallback: Not enough landmarks detected:', len(lmk))
                print('[FeatureExtractor] แจ้งผู้ใช้: ไม่พบ landmark จากใบหน้า กรุณาตรวจสอบกล้องหรือแสง')
            FeatureExtractor._debug_logged = True
            return _fallback(0.05, False)
        # Check confidence for key points (if available)
        key_ids = self.LEFT_EYE_IDS + self.RIGHT_EYE_IDS + self.LEFT_IRIS_IDS + self.RIGHT_IRIS_IDS
        key_conf = []
        for i in key_ids:
            if i < len(lmk):
                p = lmk[i]
                v = getattr(p, 'visibility', None)
                if v is None:
                    v = 1.0
                key_conf.append(v)
        # ไม่เช็ค avg_conf เพราะ mediapipe v0.10.x ไม่คืน visibility
        l_iris = self._median_xy(self.LEFT_IRIS_IDS, pts)
        r_iris = self._median_xy(self.RIGHT_IRIS_IDS, pts)
        l_box  = self._eye_box_metrics(self.LEFT_EYE_IDS, pts)
        r_box  = self._eye_box_metrics(self.RIGHT_EYE_IDS, pts)
        if not FeatureExtractor._debug_logged:
            print(f'[FeatureExtractor] l_iris: {l_iris}, r_iris: {r_iris}')
            print(f'[FeatureExtractor] l_box: {l_box}, r_box: {r_box}')
            FeatureExtractor._debug_logged = True

        dx = lmk[263].x - lmk[33].x
        dy = lmk[263].y - lmk[33].y
        yaw = math.degrees(math.atan2(dy, dx))
        pitch = math.degrees(math.atan2(lmk[1].y - lmk[168].y, lmk[1].z - lmk[168].z))

        l_iris = self._median_xy(self.LEFT_IRIS_IDS, pts)
        r_iris = self._median_xy(self.RIGHT_IRIS_IDS, pts)
        l_box  = self._eye_box_metrics(self.LEFT_EYE_IDS, pts)
        r_box  = self._eye_box_metrics(self.RIGHT_EYE_IDS, pts)

        # Quality check: iris and eye box must be valid
        if (l_iris is None or l_box is None) and (r_iris is None or r_box is None):
            print('[FeatureExtractor] Fallback: iris or eye box invalid', l_iris, l_box, r_iris, r_box)
            return _fallback(0.1, False)

        l_n = self._norm_in_box(l_iris, l_box) if (l_iris and l_box) else (0.5, 0.5)
        r_n = self._norm_in_box(r_iris, r_box) if (r_iris and r_box) else (0.5, 0.5)

        earL = l_box[6] if l_box else None
        earR = r_box[6] if r_box else None
        eye_open = bool((earL or 0.0) > 0.20 or (earR or 0.0) > 0.20)

        def _eye_conf(ear, base, box, iris_ok):
            if (ear is None) or (box is None) or (not iris_ok):
                return 0.0
            if base is None: base = 0.26
            ratio = max(0.0, min(1.4, ear / max(1e-6, 0.8 * base)))
            _, _, _, _, bw, bh, _ = box
            area_norm  = (bw * bh) / max(1.0, (w * h))
            area_boost = min(1.0, area_norm / 0.02)
            return max(0.0, min(1.0, 0.7 * ratio + 0.3 * area_boost))

        cL = _eye_conf(earL, self.earL_base, l_box, l_iris is not None)
        cR = _eye_conf(earR, self.earR_base, r_box, r_iris is not None)

        for ear, attr in [(earL, "earL_base"), (earR, "earR_base")]:
            if ear is not None:
                cur = getattr(self, attr)
                setattr(self, attr, 0.9 * cur + 0.1 * ear if cur is not None else ear)

        if (cL + cR) <= 1e-6:
            ex, ey = 0.5, 0.5
        else:
            ex = float((l_n[0] * cL + r_n[0] * cR) / (cL + cR))
            ey = float((l_n[1] * cL + r_n[1] * cR) / (cL + cR))

        face_ids = [1, 9, 152, 33, 263]
        fxs = [pts[i][0] for i in face_ids if i < len(pts)]
        fys = [pts[i][1] for i in face_ids if i < len(pts)]
        fcx = float(sum(fxs) / len(fxs) / max(1, w)) if fxs else 0.5
        fcy = float(sum(fys) / len(fys) / max(1, h)) if fys else 0.5

        margin = min(ex, 1 - ex, ey, 1 - ey)
        quality = max(cL, cR) * (0.6 + 0.4 * max(0.0, min(1.0, margin * 2)))

        return dict(
            eye_cx_norm=ex, eye_cy_norm=ey,
            face_cx_norm=fcx, face_cy_norm=fcy,
            eye_open=eye_open, quality=float(quality),
            yaw=float(yaw), pitch=float(pitch),
        )
