import cv2
import os
import time
from datetime import datetime

from app.motion.detector import MotionDetector


POST_MOTION_SECONDS = 10
OUTPUT_FOLDER = "recordings"


def start_camera():

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open camera.")
        return

    # Create recordings folder
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    detector = MotionDetector()

    recording = False
    video_writer = None
    last_motion_time = None

    print("SentriX camera started.")
    print("Press Q to quit.")

    while True:

        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read frame.")
            break

        # Detect motion
        motion_detected, frame = detector.detect(frame)

        current_time = time.time()

        # --------------------------------
        # MOTION DETECTED
        # --------------------------------

        if motion_detected:

            last_motion_time = current_time

            # Start recording
            if not recording:

                timestamp = datetime.now().strftime(
                    "%Y-%m-%d_%H-%M-%S"
                )

                filename = os.path.join(
                    OUTPUT_FOLDER,
                    f"motion_{timestamp}.avi"
                )

                height, width = frame.shape[:2]

                fourcc = cv2.VideoWriter_fourcc(
                    *"XVID"
                )

                video_writer = cv2.VideoWriter(
                    filename,
                    fourcc,
                    20.0,
                    (width, height)
                )

                recording = True

                print(f"Recording started: {filename}")

        # --------------------------------
        # RECORDING
        # --------------------------------

        if recording:

            video_writer.write(frame)

            # Stop only after motion has been
            # absent for POST_MOTION_SECONDS
            if (
                last_motion_time is not None
                and current_time - last_motion_time
                >= POST_MOTION_SECONDS
            ):

                video_writer.release()
                video_writer = None

                recording = False
                last_motion_time = None

                print("Motion stopped. Recording saved.")

        # --------------------------------
        # DISPLAY STATUS
        # --------------------------------

        if recording:
            status = "RECORDING"
        elif motion_detected:
            status = "MOTION DETECTED"
        else:
            status = "MONITORING"

        cv2.putText(
            frame,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255) if recording else (0, 255, 0),
            2
        )

        cv2.imshow(
            "SentriX Camera",
            frame
        )

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # --------------------------------
    # CLEANUP
    # --------------------------------

    if video_writer is not None:
        video_writer.release()

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_camera()