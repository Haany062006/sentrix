"""Fast adaptive blur detection for SentriX camera tamper checks."""
import cv2
import numpy as np


class BlurDetector:
    def __init__(self, threshold=6.0, confirm_frames=3, calibration_frames=12, ratio=0.20):
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

    def detect(self, frame, brightness=None):
        if frame is None or frame.size == 0:
            return False, self.last_variance

        # Do not flag blur when scene is too dark; darkness is handled by lens cover detector
        if brightness is not None and brightness < 22.0:
            self._blur_count = 0
            return False, self.last_variance

        gray = self._small_gray(frame)
        variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        self.last_variance = variance

        # Establish a stable sharpness baseline from early normal frames.
        if self.reference_variance is None:
            self._calibration.append(variance)
            if len(self._calibration) >= self.calibration_frames:
                baseline = float(np.median(self._calibration))
                self.reference_variance = max(baseline, self.threshold)
            self._blur_count = 0
            return False, variance

        # Blur requires that sharpness has collapsed significantly both below threshold
        # AND to a small fraction (< 20%) of baseline (or absolute collapse < 1.5)
        is_blurry = (variance < self.threshold and variance < self.reference_variance * self.ratio) or variance < 1.5

        if is_blurry:
            self._blur_count += 1
        else:
            self._blur_count = 0
            # Gently adapt baseline on clear frames to prevent drift over time
            if variance > self.threshold:
                self.reference_variance = 0.99 * self.reference_variance + 0.01 * variance

        return self._blur_count >= self.confirm_frames, variance

    def reset(self):
        # Reset only the confirmation streak; preserve calibration to avoid a long blind period.
        self._blur_count = 0
