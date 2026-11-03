"""Camera movement detection using ORB features and robust affine matching."""
import cv2
import numpy as np

ANALYSIS_WIDTH = 320
MIN_GOOD_MATCHES = 8
MIN_INLIER_RATIO = 0.50
MOVE_CONFIRMATION_FRAMES = 6
MIN_SHIFT_RATIO = 0.12
MIN_ROTATION_DEGREES = 6.0
MIN_SCALE_CHANGE = 0.10
RANSAC_REPROJECTION_THRESHOLD = 3.0


def _small_gray(frame):
    height, width = frame.shape[:2]
    if width > ANALYSIS_WIDTH:
        new_height = max(1, int(height * ANALYSIS_WIDTH / width))
        frame = cv2.resize(frame, (ANALYSIS_WIDTH, new_height), interpolation=cv2.INTER_AREA)
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def calculate_histogram(frame):
    gray = _small_gray(frame)
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist


class SceneChangeDetector:
    def __init__(self, reference_frame):
        if reference_frame is None or reference_frame.size == 0:
            raise ValueError("A valid reference camera frame is required.")
        self.reference_gray = _small_gray(reference_frame)
        self.reference_histogram = self._hist(self.reference_gray)
        self.orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=6, fastThreshold=10)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.reference_keypoints, self.reference_descriptors = self.orb.detectAndCompute(self.reference_gray, None)
        self.changed_frames = 0
        self.last_movement_candidate = False

    @staticmethod
    def _hist(gray):
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
        return hist

    def detect(self, current_frame):
        if current_frame is None or current_frame.size == 0:
            self.last_movement_candidate = False
            return False, 0.0

        gray = _small_gray(current_frame)
        hist = self._hist(gray)
        correlation = float(cv2.compareHist(self.reference_histogram, hist, cv2.HISTCMP_CORREL))
        moved_now = False
        self.last_movement_candidate = False
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        if (self.reference_descriptors is not None and descriptors is not None
                and len(self.reference_keypoints) >= MIN_GOOD_MATCHES
                and len(keypoints) >= MIN_GOOD_MATCHES):
            pairs = self.matcher.knnMatch(self.reference_descriptors, descriptors, k=2)
            good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
            if len(good) >= MIN_GOOD_MATCHES:
                src = np.float32([self.reference_keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst = np.float32([keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                matrix, inliers = cv2.estimateAffinePartial2D(
                    src, dst, method=cv2.RANSAC,
                    ransacReprojThreshold=RANSAC_REPROJECTION_THRESHOLD,
                    maxIters=500, confidence=0.99, refineIters=5,
                )
                if matrix is not None and inliers is not None:
                    inlier_count = int(inliers.sum())
                    inlier_ratio = inlier_count / max(len(good), 1)
                    if inlier_count >= MIN_GOOD_MATCHES and inlier_ratio >= MIN_INLIER_RATIO:
                        height, width = gray.shape
                        dx = abs(float(matrix[0, 2])) / max(width, 1)
                        dy = abs(float(matrix[1, 2])) / max(height, 1)
                        angle = abs(float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))))
                        scale = float(np.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2))
                        moved_now = (
                            dx >= MIN_SHIFT_RATIO or dy >= MIN_SHIFT_RATIO
                            or angle >= MIN_ROTATION_DEGREES
                            or abs(scale - 1.0) >= MIN_SCALE_CHANGE
                        )

        self.last_movement_candidate = moved_now
        self.changed_frames = self.changed_frames + 1 if moved_now else 0
        return self.changed_frames >= MOVE_CONFIRMATION_FRAMES, correlation


def detect_scene_change(reference_histogram, current_frame):
    current_histogram = calculate_histogram(current_frame)
    correlation = float(cv2.compareHist(reference_histogram, current_histogram, cv2.HISTCMP_CORREL))
    return correlation < 0.70, correlation
