"""Combined, low-cost camera tamper detector for SentriX."""
import cv2
import numpy as np
from app.tamper.blur import BlurDetector
from app.tamper.histogram import SceneChangeDetector

DARK_THRESHOLD = 18.0
DARK_CONFIRMATION_FRAMES = 2


def detect_darkness(frame):
    if frame is None or frame.size == 0:
        return False, 0.0
    h, w = frame.shape[:2]
    if w > 240:
        frame = cv2.resize(frame, (240, max(1, int(h * 240 / w))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    brightness = float(np.mean(gray))
    return brightness < DARK_THRESHOLD, brightness


class TamperDetector:
    def __init__(self, reference_frame):
        self.scene_change_detector = SceneChangeDetector(reference_frame)
        self.blur_detector = BlurDetector(threshold=28.0, confirm_frames=2, calibration_frames=12, ratio=0.45)
        self.dark_frames = 0
        self.last_status = "CAMERA OK"

    def _result(self, status, brightness=0.0, blur_variance=0.0, correlation=1.0):
        self.last_status = status
        return {
            "tampering": status != "CAMERA OK",
            "status": status,
            "brightness": float(brightness),
            "blur_variance": float(blur_variance),
            "histogram_correlation": float(correlation),
        }

    def detect(self, frame):
        if frame is None or frame.size == 0:
            return self._result(self.last_status)

        dark, brightness = detect_darkness(frame)
        self.dark_frames = self.dark_frames + 1 if dark else 0
        if self.dark_frames >= DARK_CONFIRMATION_FRAMES:
            self.blur_detector.reset()
            return self._result("LENS COVERED", brightness, 0.0, 0.0)

        # Preserve the existing, previously working camera-movement detector.
                # Detect camera movement using the scene detector.
        try:
            moved, correlation = self.scene_change_detector.detect(frame)
        except Exception as exc:
            print("Scene-change detection error:", exc)
            moved, correlation = False, 1.0

        if moved:
            self.blur_detector.reset()
            return self._result(
                "CAMERA MOVED",
                brightness,
                0.0,
                correlation
            )
        blurry, variance = self.blur_detector.detect(frame)
        if blurry:
            return self._result("CAMERA BLURRED", brightness, variance, correlation)
        return self._result("CAMERA OK", brightness, variance, correlation)
