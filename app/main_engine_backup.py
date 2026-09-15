import cv2
import os
import time
from datetime import datetime

from mtcnn import MTCNN
from keras_facenet import FaceNet

from app.motion.detector import MotionDetector
from app.face.recognizer import (
    load_registered_faces,
    recognize_face
)
from app.tamper.detector import TamperDetector


POST_MOTION_SECONDS = 10
OUTPUT_FOLDER = "recordings"


def run_sentrix():

    print("\n==============================")
    print("       SENTRIX STARTING")
    print("==============================")

    # --------------------------------
    # CAMERA
    # --------------------------------

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open camera.")
        return

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    # --------------------------------
    # MOTION
    # --------------------------------

    motion_detector = MotionDetector()

    # --------------------------------
    # FACE DETECTION
    # --------------------------------

    print("Loading MTCNN...")

    face_detector = MTCNN()

    # --------------------------------
    # FACE RECOGNITION
    # --------------------------------

    print("Loading registered faces...")

    registered_faces = load_registered_faces()

    embedder = None

    if registered_faces:

        print("Loading FaceNet...")

        embedder = FaceNet()

        print(
            f"{len(registered_faces)} "
            "registered face(s) loaded."
        )

    else:

        print(
            "No registered faces found."
        )

        print(
            "Face recognition will show UNKNOWN."
        )

    # --------------------------------
    # GET FIRST FRAME
    # --------------------------------

    success, first_frame = camera.read()

    if not success:

        print(
            "ERROR: Could not read first frame."
        )

        camera.release()
        return

    # --------------------------------
    # TAMPER DETECTOR
    # --------------------------------

    tamper_detector = TamperDetector(
        first_frame
    )

    # --------------------------------
    # RECORDING VARIABLES
    # --------------------------------

    recording = False

    video_writer = None

    last_motion_time = None

    # --------------------------------
    # START
    # --------------------------------

    print("\nSENTRIX is running.")
    print("Press Q to quit.\n")

    # --------------------------------
    # MAIN LOOP
    # --------------------------------

    while True:

        success, frame = camera.read()

        if not success:

            print(
                "ERROR: Could not read frame."
            )

            break

        current_time = time.time()

        # =================================
        # MOTION DETECTION
        # =================================

        motion_detected, frame = (
            motion_detector.detect(frame)
        )

        # =================================
        # TAMPER DETECTION
        # =================================

        tamper_result = tamper_detector.detect(
            frame
        )

        # =================================
        # FACE DETECTION + RECOGNITION
        # =================================

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        faces = face_detector.detect_faces(
            rgb_frame
        )

        recognized_people = []

        for face in faces:

            x, y, width, height = face["box"]

            x = max(0, x)
            y = max(0, y)

            face_image = rgb_frame[
                y:y + height,
                x:x + width
            ]

            if face_image.size == 0:
                continue

            if embedder is not None:

                name, score = recognize_face(
                    face_image,
                    embedder,
                    registered_faces
                )

            else:

                name = "UNKNOWN"
                score = 0.0

            recognized_people.append(
                {
                    "name": name,
                    "score": score
                }
            )

            # --------------------------------
            # FACE BOX
            # --------------------------------

            if name == "UNKNOWN":

                box_color = (
                    0,
                    0,
                    255
                )

            else:

                box_color = (
                    0,
                    255,
                    0
                )

            cv2.rectangle(
                frame,
                (x, y),
                (x + width, y + height),
                box_color,
                2
            )

            cv2.putText(
                frame,
                f"{name} ({score:.2f})",
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2
            )

        # =================================
        # START RECORDING
        # =================================

        if motion_detected:

            last_motion_time = current_time

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

                print(
                    f"Recording started: {filename}"
                )

        # =================================
        # WRITE RECORDING
        # =================================

        if recording:

            video_writer.write(frame)

            if (
                last_motion_time is not None
                and
                current_time - last_motion_time
                >= POST_MOTION_SECONDS
            ):

                video_writer.release()

                video_writer = None

                recording = False

                last_motion_time = None

                print(
                    "Motion stopped. "
                    "Recording saved."
                )

        # =================================
        # DISPLAY STATUS
        # =================================

        if recording:

            camera_status = "RECORDING"

        elif motion_detected:

            camera_status = "MOTION DETECTED"

        else:

            camera_status = "MONITORING"

        cv2.putText(
            frame,
            f"Camera: {camera_status}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Faces: {len(faces)}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Tamper: {tamper_result['status']}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (
                (0, 0, 255)
                if tamper_result["tampering"]
                else
                (0, 255, 0)
            ),
            2
        )

        # =================================
        # SHOW CAMERA
        # =================================

        cv2.imshow(
            "SENTRIX - Integrated System",
            frame
        )

        # =================================
        # QUIT
        # =================================

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break

    # =================================
    # CLEANUP
    # =================================

    if video_writer is not None:

        video_writer.release()

    camera.release()

    cv2.destroyAllWindows()

    print("\nSENTRIX stopped.")


if __name__ == "__main__":

    run_sentrix()