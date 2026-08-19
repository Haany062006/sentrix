import cv2
import numpy as np


BRIGHTNESS_THRESHOLD = 20


def detect_darkness(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    brightness = np.mean(gray)

    tampered = (
        brightness < BRIGHTNESS_THRESHOLD
    )

    return tampered, brightness