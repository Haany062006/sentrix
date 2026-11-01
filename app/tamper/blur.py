"""Fast adaptive blur detection for SentriX camera tamper checks."""
import cv2
import numpy as np


class BlurDetector:
    def __init__(self, threshold=28.0, confirm_frames=2, calibration_frames=12, ratio=0.45):
        self.threshold = float(threshold)
        self.confirm_frames = max(1, int(confirm_frames))
        self.calibration_frames = max(5, int(calibration_frames))
        self.ratio = float(ratio)
        self._calibration = []
        self.reference_variance = None
        self._blur_count = 0
        self.last_variance = 0.0

    def _small_gray(self, frame):
        h, w = frame.shape[:2]
        if w > 320:
            frame = cv2.resize(frame, (320, max(1, int(h * 320 / w))), interpolation=cv2.INTER_AREA)
        if len(frame.shape) == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def detect(self, frame):
        if frame is None or frame.size == 0:
            return False, self.last_variance
        gray = self._small_gray(frame)
        variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        self.last_variance = variance

        # Establish a stable sharpness baseline from early, normal frames.
        if self.reference_variance is None:
            self._calibration.append(variance)
            if len(self._calibration) >= self.calibration_frames:
                baseline = float(np.median(self._calibration))
                self.reference_variance = max(baseline, self.threshold)
            self._blur_count = 0
            return False, variance

        is_blurry = variance < self.threshold or variance < self.reference_variance * self.ratio
        self._blur_count = self._blur_count + 1 if is_blurry else 0
        return self._blur_count >= self.confirm_frames, variance

    def reset(self):
        # Reset only the confirmation streak; preserve calibration to avoid a long blind period.
        self._blur_count = 0
