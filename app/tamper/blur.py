import cv2


BLUR_THRESHOLD = 20


def detect_blur(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    variance = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    tampered = (
        variance < BLUR_THRESHOLD
    )

    return tampered, variance