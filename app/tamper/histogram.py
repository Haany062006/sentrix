import cv2


# A scene is considered changed only when the
# correlation is very low.
HISTOGRAM_THRESHOLD = -0.20

# Number of consecutive changed frames required
# before declaring camera movement.
CHANGE_CONFIRMATION_FRAMES = 15


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


class SceneChangeDetector:

    def __init__(self, reference_frame):

        self.reference_histogram = calculate_histogram(
            reference_frame
        )

        self.changed_frames = 0

    def detect(self, current_frame):

        current_histogram = calculate_histogram(
            current_frame
        )

        correlation = cv2.compareHist(
            self.reference_histogram,
            current_histogram,
            cv2.HISTCMP_CORREL
        )

        if correlation < HISTOGRAM_THRESHOLD:

            self.changed_frames += 1

        else:

            self.changed_frames = 0

        moved = (
            self.changed_frames
            >= CHANGE_CONFIRMATION_FRAMES
        )

        return moved, correlation


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

    moved = correlation < HISTOGRAM_THRESHOLD

    return moved, correlation