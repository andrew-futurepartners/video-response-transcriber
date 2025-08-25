# visual_checks_lite.py  ───────────────────────────────────────────
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd


# ------------------------------------------------------------------
@dataclass
class LiteConfig:
    max_frames:      int   = 12        # frames sampled per clip
    dark_thresh:     float = 0.30      # brightness below → penalty
    face_threshold:  float = 0.30      # share of frames w/ face
    blur_thresh:     float = 150.0     # Laplacian variance threshold
    skin_ratio_veto: float = 0.50      # >50 % torso skin ⇒ veto
    # rough HSV skin-tone window (can tweak)
    skin_hsv_lower:  Tuple[int, int, int] = (0, 48, 80)
    skin_hsv_upper:  Tuple[int, int, int] = (30, 255, 255)


# ------------------------------------------------------------------
class VisualInspectorLite:
    """
    Lightweight, CPU-only highlight-reel suitability checker.
    Hard-veto (score=0) if:
      • Dark & no face
      • Shirtless (bare chest)
    Otherwise start at 100 and subtract soft penalties.
    """

    def __init__(self, **kwargs):
        self.cfg = LiteConfig(**kwargs)
        self._face = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    # ──────────────────────────────────────────────────────────────
    def analyse_video(self, path: Path) -> Dict[str, float | int | str]:
        frames = self._sample_frames(path)
        if not frames:
            return dict(score=0, flag="yes", brightness=0,
                        face_ratio=0.0, blur_ratio=1.0, skin_flag=False)

        stats = [self._analyse_frame(f) for f in frames]
        df    = pd.DataFrame(stats)

        brightness = df["brightness"].mean()
        face_ratio = df["face"].mean()
        blur_ratio = (df["blur"] < self.cfg.blur_thresh).mean()  # share blurry
        skin_flag = df["shirtless"].any()

        # NEW — duration (seconds)
        duration_sec = getattr(self, "total_frames", len(frames)) / getattr(self, "fps_est", 30)

        # ── HARD-VETO  (score = 0, flag = "yes") ───────────────────────────
        hard_fail = False

        # A) outright darkness
        if brightness < 0.15:  # frame basically black
            hard_fail = True

        # B) we never saw a “real” face (≥ 5 % of the frame)
        if face_ratio < 0.05:
            hard_fail = True

        # C) excessive blur AND no face (mush)
        if face_ratio == 0 and blur_ratio == 1.0:
            hard_fail = True

        # D) shirt-off clip
        if skin_flag:
            hard_fail = True

        # E) too short
        if duration_sec < 1:
            hard_fail = True

        if hard_fail:
            return {
                "score": 0,
                "flag": "yes",
                "brightness": round(brightness, 3),
                "face_ratio": round(face_ratio, 3),
                "blur_ratio": round(blur_ratio, 3),
                "skin_flag": bool(skin_flag),
            }
        # ───────────────────────────────────────────────────────────────────

        # soft penalties
        score = 100
        if brightness < self.cfg.dark_thresh:
            score -= 15
        if face_ratio < self.cfg.face_threshold:
            score -= 10
        if blur_ratio > 0.5:
            score -= 15
        score = max(score, 0)

        return {
            "score": int(score),
            "flag": "yes" if score < 60 else "no",
            "brightness": round(brightness, 3),
            "face_ratio": round(face_ratio, 3),
            "blur_ratio": round(blur_ratio, 3),
            "skin_flag":  bool(skin_flag),
        }

    # ──────────────────────────────────────────────────────────────
    # helpers
    # ──────────────────────────────────────────────────────────────
    def _sample_frames(self, path: Path) -> List[np.ndarray]:
        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        # ► save the real numbers for later
        self.fps_est = fps
        self.total_frames = total

        step = max(total // self.cfg.max_frames, 1)
        out = []
        for i in range(0, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if not ok:
                break
            out.append(frame)
        cap.release()
        return out

    def _analyse_frame(self, img: np.ndarray) -> Dict[str, float | bool]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # absolute brightness 0-1
        brightness = gray.mean() / 255.0

        # blur metric (variance of Laplacian)
        blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()

        # detect face in 0°, +90°, -90°
        found, bbox = self._detect_face_any_rotation(gray)

        # ── 1️⃣  Drop tiny / skin-free boxes (icons, light fixtures…) ──
        if found:
            x, y, w, h = bbox
            face_area_ratio = (w * h) / (gray.shape[0] * gray.shape[1])
            skin_face_ratio = 0.0
            if h and w:
                face_hsv = hsv[y:y + h, x:x + w]
                if face_hsv.size:
                    mask = cv2.inRange(
                        face_hsv, self.cfg.skin_hsv_lower, self.cfg.skin_hsv_upper
                    )
                    skin_face_ratio = mask.mean() / 255.0

            # need EITHER ≥5 % of frame **or** ≥5 % skin pixels
            if face_area_ratio < 0.05 and skin_face_ratio < 0.05:
                found, bbox = False, None
        # ──────────────────────────────────────────────────────────────

        # shirtless veto (needs a valid face bbox)
        shirtless = False
        if found:
            x, y, w, h = bbox
            y1 = y + h
            y2 = min(y + 3 * h, gray.shape[0])
            x1 = max(x - w, 0)
            x2 = min(x + 2 * w, gray.shape[1])
            roi = hsv[y1:y2, x1:x2]
            if roi.size:
                mask = cv2.inRange(
                    roi, self.cfg.skin_hsv_lower, self.cfg.skin_hsv_upper
                )
                skin_ratio = mask.mean() / 255.0
                shirtless = skin_ratio > 0.75

        return dict(
            brightness=brightness,
            blur=blur_val,
            face=found,
            shirtless=shirtless,
        )

    # -------------------------------------------------------------
    def _detect_face_any_rotation(self, gray: np.ndarray):
        """Return (found_bool, bbox) searching 0°, +90°, -90°."""
        for rot in (0, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_90_CLOCKWISE):
            g = gray if rot == 0 else cv2.rotate(gray, rot)
            faces = self._face.detectMultiScale(g, 1.1, 4)
            if len(faces):
                x, y, w, h = faces[0]
                if rot != 0:
                    # map bbox back to original coords
                    if rot == cv2.ROTATE_90_COUNTERCLOCKWISE:
                        x, y = y, gray.shape[1] - (x + w)
                        w, h = h, w
                    else:  # clockwise
                        x, y = gray.shape[0] - (y + h), x
                        w, h = h, w
                return True, (x, y, w, h)
        return False, None