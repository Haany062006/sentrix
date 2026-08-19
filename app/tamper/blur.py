import cv2


# Lower value = more blur
BLUR_THRESHOLD = 50


def detect_blur(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    variance = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    blurred = variance < BLUR_THRESHOLD

    return blurred, variance