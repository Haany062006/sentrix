import cv2


HISTOGRAM_THRESHOLD = 0.30


def calculate_histogram(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

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


def detect_scene_change(
    reference_histogram,
    current_frame
):

    current_histogram = calculate_histogram(
        current_frame
    )

    correlation = cv2.compareHist(
        reference_histogram,
        current_histogram,
        cv2.HISTCMP_CORREL
    )

    tampered = (
        correlation <
        HISTOGRAM_THRESHOLD
    )

    return (
        tampered,
        correlation
    )