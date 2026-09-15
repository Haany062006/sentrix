import cv2


class MotionDetector:
    def __init__(self, min_area=1500):
        self.min_area = min_area
        self.previous_frame = None

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.previous_frame is None:
            self.previous_frame = gray
            return False

        difference = cv2.absdiff(self.previous_frame, gray)

        _, threshold = cv2.threshold(
            difference,
            25,
            255,
            cv2.THRESH_BINARY
        )

        threshold = cv2.dilate(threshold, None, iterations=2)

        contours, _ = cv2.findContours(
            threshold,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        self.previous_frame = gray

        for contour in contours:
            area = cv2.contourArea(contour)

            if area >= self.min_area:
                return True

        return False