import cv2
import numpy as np


# ============================================================
# CAMERA MOVEMENT DETECTION CONFIGURATION
# ============================================================

# Minimum number of good feature matches required.
MIN_GOOD_MATCHES = 18

# Camera movement is confirmed when the estimated transform
# indicates sufficient translation/rotation.
MIN_TRANSLATION_PIXELS = 25.0
MIN_ROTATION_DEGREES = 4.0

# Consecutive frames required before declaring movement.
CHANGE_CONFIRMATION_FRAMES = 8

# After movement is confirmed, the reference frame is updated
# so the same movement does not trigger continuously.
REFERENCE_UPDATE_AFTER_MOVEMENT = True

# Process a smaller image for faster detection.
PROCESS_WIDTH = 640


# ============================================================
# IMAGE PREPARATION
# ============================================================

def prepare_frame(frame):
    """
    Resize and convert frame to grayscale.
    """

    height, width = frame.shape[:2]

    if width > PROCESS_WIDTH:

        scale = PROCESS_WIDTH / float(width)

        new_height = int(
            height * scale
        )

        frame = cv2.resize(
            frame,
            (
                PROCESS_WIDTH,
                new_height
            ),
            interpolation=cv2.INTER_AREA
        )

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Slight blur reduces noise.
    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    return gray


# ============================================================
# HISTOGRAM
# ============================================================

def calculate_histogram(frame):

    gray = prepare_frame(frame)

    histogram = cv2.calcHist(
        [gray],
        [0],
        None,
        [256],
        [0, 256]
    )

    cv2.normalize(
        histogram,
        histogram
    )

    return histogram


# ============================================================
# FEATURE-BASED CAMERA MOVEMENT DETECTOR
# ============================================================

class SceneChangeDetector:

    def __init__(self, reference_frame):

        self.reference_frame = prepare_frame(
            reference_frame
        )

        # ORB is lightweight and suitable for real-time
        # camera movement detection.
        self.orb = cv2.ORB_create(
            nfeatures=500,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            fastThreshold=20
        )

        self.reference_keypoints, self.reference_descriptors = (
            self.orb.detectAndCompute(
                self.reference_frame,
                None
            )
        )

        self.matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING,
            crossCheck=False
        )

        self.changed_frames = 0

        # Keep histogram as a secondary measurement.
        self.reference_histogram = (
            calculate_histogram(
                reference_frame
            )
        )

        self.last_correlation = 1.0

    # ========================================================
    # ESTIMATE CAMERA MOVEMENT
    # ========================================================

    def _estimate_movement(
        self,
        current_frame
    ):

        current_gray = prepare_frame(
            current_frame
        )

        current_keypoints, current_descriptors = (
            self.orb.detectAndCompute(
                current_gray,
                None
            )
        )

        # Not enough features.
        if (
            current_descriptors is None
            or self.reference_descriptors is None
        ):

            return False, 0.0, 0.0, 0

        if (
            len(current_keypoints) < 8
            or len(self.reference_keypoints) < 8
        ):

            return False, 0.0, 0.0, 0

        try:

            matches = self.matcher.knnMatch(
                self.reference_descriptors,
                current_descriptors,
                k=2
            )

        except Exception:

            return False, 0.0, 0.0, 0

        # Lowe ratio test.
        good_matches = []

        for pair in matches:

            if len(pair) < 2:
                continue

            m, n = pair

            if m.distance < 0.72 * n.distance:

                good_matches.append(m)

        good_match_count = len(
            good_matches
        )

        if good_match_count < MIN_GOOD_MATCHES:

            return False, 0.0, 0.0, good_match_count

        # ----------------------------------------------------
        # SOURCE / DESTINATION POINTS
        # ----------------------------------------------------

        src_points = np.float32([
            self.reference_keypoints[m.queryIdx].pt
            for m in good_matches
        ]).reshape(-1, 1, 2)

        dst_points = np.float32([
            current_keypoints[m.trainIdx].pt
            for m in good_matches
        ]).reshape(-1, 1, 2)

        # ----------------------------------------------------
        # ESTIMATE AFFINE TRANSFORMATION
        # ----------------------------------------------------

        matrix, mask = cv2.estimateAffinePartial2D(
            src_points,
            dst_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=2000,
            confidence=0.99
        )

        if matrix is None:

            return False, 0.0, 0.0, good_match_count

        # ----------------------------------------------------
        # TRANSLATION
        # ----------------------------------------------------

        tx = float(
            matrix[0, 2]
        )

        ty = float(
            matrix[1, 2]
        )

        translation = (
            (tx ** 2)
            + (ty ** 2)
        ) ** 0.5

        # ----------------------------------------------------
        # ROTATION
        # ----------------------------------------------------

        rotation_radians = np.arctan2(
            matrix[1, 0],
            matrix[0, 0]
        )

        rotation_degrees = abs(
            np.degrees(
                rotation_radians
            )
        )

        # ----------------------------------------------------
        # INLIER COUNT
        # ----------------------------------------------------

        inlier_count = 0

        if mask is not None:

            inlier_count = int(
                mask.sum()
            )

        # Require enough reliable inliers.
        if inlier_count < 12:

            return (
                False,
                translation,
                rotation_degrees,
                good_match_count
            )

        movement_detected = (
            translation >= MIN_TRANSLATION_PIXELS
            or rotation_degrees >= MIN_ROTATION_DEGREES
        )

        return (
            movement_detected,
            translation,
            rotation_degrees,
            good_match_count
        )

    # ========================================================
    # DETECT
    # ========================================================

    def detect(
        self,
        current_frame
    ):

        # ----------------------------------------------------
        # HISTOGRAM CORRELATION
        # ----------------------------------------------------

        current_histogram = (
            calculate_histogram(
                current_frame
            )
        )

        correlation = cv2.compareHist(
            self.reference_histogram,
            current_histogram,
            cv2.HISTCMP_CORREL
        )

        self.last_correlation = correlation

        # ----------------------------------------------------
        # FEATURE MOVEMENT
        # ----------------------------------------------------

        (
            movement_detected,
            translation,
            rotation,
            good_matches
        ) = self._estimate_movement(
            current_frame
        )

        # ----------------------------------------------------
        # CONFIRMATION
        # ----------------------------------------------------

        if movement_detected:

            self.changed_frames += 1

        else:

            self.changed_frames = max(
                0,
                self.changed_frames - 1
            )

        moved = (
            self.changed_frames
            >= CHANGE_CONFIRMATION_FRAMES
        )

        # ----------------------------------------------------
        # UPDATE REFERENCE
        # ----------------------------------------------------

        if moved:

            if REFERENCE_UPDATE_AFTER_MOVEMENT:

                self.reference_frame = (
                    prepare_frame(
                        current_frame
                    )
                )

                (
                    self.reference_keypoints,
                    self.reference_descriptors
                ) = self.orb.detectAndCompute(
                    self.reference_frame,
                    None
                )

                self.reference_histogram = (
                    calculate_histogram(
                        current_frame
                    )
                )

            self.changed_frames = 0

        return moved, correlation


# ============================================================
# SIMPLE SCENE CHANGE FUNCTION
# ============================================================

def detect_scene_change(
    reference_histogram,
    current_frame
):

    current_histogram = (
        calculate_histogram(
            current_frame
        )
    )

    correlation = cv2.compareHist(
        reference_histogram,
        current_histogram,
        cv2.HISTCMP_CORREL
    )

    # This function is kept for compatibility with
    # any other code in the project.
    moved = correlation < -0.20

    return moved, correlation