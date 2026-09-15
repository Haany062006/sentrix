from app.tamper.brightness import detect_darkness
from app.tamper.blur import detect_blur
from app.tamper.histogram import (
    calculate_histogram,
    detect_scene_change
)


class TamperDetector:

    def __init__(self, reference_frame):

        self.reference_histogram = calculate_histogram(
            reference_frame
        )

    def detect(self, frame):

        # -----------------------------
        # 1. LENS COVERED
        # -----------------------------

        dark, brightness = detect_darkness(frame)

        # -----------------------------
        # 2. CAMERA BLURRED
        # -----------------------------

        blurry, variance = detect_blur(frame)

        # -----------------------------
        # 3. CAMERA MOVED
        # -----------------------------

        moved, correlation = detect_scene_change(
            self.reference_histogram,
            frame
        )

        # -----------------------------
        # PRIORITY
        # -----------------------------

        if dark:

            status = "LENS COVERED"

        elif blurry:

            status = "CAMERA BLURRED"

        elif moved:

            status = "CAMERA MOVED"

        else:

            status = "CAMERA OK"

        tampering = status != "CAMERA OK"

        return {
            "tampering": tampering,
            "status": status,
            "brightness": brightness,
            "blur_variance": variance,
            "histogram_correlation": correlation
        }