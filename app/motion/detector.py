import cv2


class MotionDetector:
    def __init__(self, min_area=500):
        self.previous_frame = None
        self.min_area = min_area

    def detect(self, frame):
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Reduce small image noise
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # First frame
        if self.previous_frame is None:
            self.previous_frame = gray
            return False, frame

        # Compare current frame with previous frame
        difference = cv2.absdiff(self.previous_frame, gray)

        # Convert difference to black/white
        threshold = cv2.threshold(
            difference,
            25,
            255,
            cv2.THRESH_BINARY
        )[1]

        # Make detected areas larger
        threshold = cv2.dilate(
            threshold,
            None,
            iterations=2
        )

        # Find moving areas
        contours, _ = cv2.findContours(
            threshold,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        motion_detected = False

        for contour in contours:

            if cv2.contourArea(contour) < self.min_area:
                continue

            motion_detected = True

            x, y, w, h = cv2.boundingRect(contour)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

        # Update previous frame
        self.previous_frame = gray

        return motion_detected, frame