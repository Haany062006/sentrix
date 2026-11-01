import os
import cv2
import pickle
import numpy as np

from mtcnn import MTCNN
from keras_facenet import FaceNet


REGISTERED_FOLDER = "registered_faces"
SAMPLES_REQUIRED = 5


def register_person():

    name = input("Enter person's name: ").strip()

    if not name:
        print("ERROR: Name cannot be empty.")
        return

    os.makedirs(REGISTERED_FOLDER, exist_ok=True)

    detector = MTCNN()
    embedder = FaceNet()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open camera.")
        return

    embeddings = []

    print(f"\nRegistering: {name}")
    print(f"Look at the camera.")
    print(f"We need {SAMPLES_REQUIRED} good face samples.")
    print("Press Q to cancel.\n")

    while len(embeddings) < SAMPLES_REQUIRED:

        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read frame.")
            break

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        faces = detector.detect_faces(rgb_frame)

        if len(faces) == 1:

            face = faces[0]

            x, y, width, height = face["box"]

            x = max(0, x)
            y = max(0, y)

            face_image = rgb_frame[
                y:y + height,
                x:x + width
            ]

            if face_image.size == 0:
                continue

            face_image = cv2.resize(
                face_image,
                (160, 160)
            )

            face_image = np.expand_dims(
                face_image,
                axis=0
            )

            embedding = embedder.embeddings(
                face_image
            )[0]

            embeddings.append(embedding)

            print(
                f"Captured sample "
                f"{len(embeddings)}/{SAMPLES_REQUIRED}"
            )

            cv2.rectangle(
                frame,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2
            )

        elif len(faces) > 1:

            cv2.putText(
                frame,
                "Only one person should be visible",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        else:

            cv2.putText(
                frame,
                "No face detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        cv2.putText(
            frame,
            f"Samples: {len(embeddings)}/{SAMPLES_REQUIRED}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "SentriX - Face Registration",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Registration cancelled.")
            break

    camera.release()
    cv2.destroyAllWindows()

    if len(embeddings) != SAMPLES_REQUIRED:
        print("Registration not completed.")
        return

    # Average the samples into one representation
    final_embedding = np.mean(
        embeddings,
        axis=0
    )

    # Normalize the embedding
    final_embedding = final_embedding / np.linalg.norm(
        final_embedding
    )

    person_file = os.path.join(
        REGISTERED_FOLDER,
        f"{name}.pkl"
    )

    with open(person_file, "wb") as file:import os
import pickle

import cv2
import numpy as np

from mtcnn import MTCNN
from keras_facenet import FaceNet


REGISTERED_FOLDER = "registered_faces"

# 10 poses × 4 samples = 40 samples
SAMPLES_PER_POSE = 4

POSES = [
    "Look straight at the camera",
    "Turn your face slightly LEFT",
    "Turn your face more LEFT",
    "Turn your face slightly RIGHT",
    "Turn your face more RIGHT",
    "Look slightly UP",
    "Look slightly DOWN",
    "Tilt your head LEFT",
    "Tilt your head RIGHT",
    "Return to natural position"
]

TOTAL_SAMPLES = len(POSES) * SAMPLES_PER_POSE


def get_embedding(face_image, embedder):
    """
    Convert a cropped face into a normalized FaceNet embedding.
    """

    face_image = cv2.resize(
        face_image,
        (160, 160)
    )

    face_image = np.expand_dims(
        face_image,
        axis=0
    )

    embedding = embedder.embeddings(
        face_image
    )[0]

    embedding = embedding / (
        np.linalg.norm(embedding) + 1e-10
    )

    return embedding.astype(np.float32)


def register_person():

    name = input(
        "Enter person's name: "
    ).strip()

    if not name:
        print("ERROR: Name cannot be empty.")
        return

    os.makedirs(
        REGISTERED_FOLDER,
        exist_ok=True
    )

    print("\nLoading MTCNN...")
    detector = MTCNN()

    print("Loading FaceNet...")
    embedder = FaceNet()

    camera = cv2.VideoCapture(
        0,
        cv2.CAP_MSMF
    )

    if not camera.isOpened():

        print(
            "ERROR: Could not open camera."
        )

        return

    embeddings = []

    print("\n========================================")
    print("       SENTRIX FACE REGISTRATION")
    print("========================================")
    print(f"Person        : {name}")
    print(f"Total samples : {TOTAL_SAMPLES}")
    print(f"Samples/pose  : {SAMPLES_PER_POSE}")

    print("\nControls:")
    print("SPACE = capture sample")
    print("Q     = cancel")

    print("\nFollow each pose shown on screen.")
    print("Make small natural movements between captures.\n")

    # ----------------------------------------
    # PROCESS EACH POSE
    # ----------------------------------------

    for pose_index, pose_instruction in enumerate(POSES):

        pose_samples = 0

        while pose_samples < SAMPLES_PER_POSE:

            success, frame = camera.read()

            if not success:

                print(
                    "ERROR: Could not read frame."
                )

                camera.release()
                cv2.destroyAllWindows()

                return

            # MTCNN expects RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Detect faces
            faces = detector.detect_faces(
                rgb_frame
            )

            # ----------------------------------------
            # TEXT INFORMATION
            # ----------------------------------------

            cv2.putText(
                frame,
                f"POSE {pose_index + 1}/{len(POSES)}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                pose_instruction,
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Pose samples: "
                f"{pose_samples}/{SAMPLES_PER_POSE}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Total: "
                f"{len(embeddings)}/{TOTAL_SAMPLES}",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            # ----------------------------------------
            # FACE DETECTION
            # ----------------------------------------

            if len(faces) == 1:

                face = faces[0]

                x, y, width, height = face["box"]

                # Prevent invalid coordinates
                x = max(0, x)
                y = max(0, y)

                width = max(1, width)
                height = max(1, height)

                x2 = min(
                    rgb_frame.shape[1],
                    x + width
                )

                y2 = min(
                    rgb_frame.shape[0],
                    y + height
                )

                face_image = rgb_frame[
                    y:y2,
                    x:x2
                ]

                # Draw face box
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                confidence = face.get(
                    "confidence",
                    0.0
                )

                cv2.putText(
                    frame,
                    f"Face confidence: "
                    f"{confidence:.2f}",
                    (20, 175),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    "Press SPACE to capture",
                    (20, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (255, 255, 255),
                    2
                )

            elif len(faces) > 1:

                cv2.putText(
                    frame,
                    "WARNING: Only ONE person allowed",
                    (20, 175),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (0, 0, 255),
                    2
                )

            else:

                cv2.putText(
                    frame,
                    "No face detected",
                    (20, 175),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (0, 0, 255),
                    2
                )

            cv2.imshow(
                "SentriX - Face Registration",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            # ----------------------------------------
            # CANCEL
            # ----------------------------------------

            if key == ord("q"):

                print(
                    "\nRegistration cancelled."
                )

                camera.release()
                cv2.destroyAllWindows()

                return

            # ----------------------------------------
            # CAPTURE SAMPLE
            # ----------------------------------------

            if key == ord(" "):

                # Exactly one face required
                if len(faces) != 1:

                    print(
                        "Capture rejected: "
                        "exactly one face must be visible."
                    )

                    continue

                face = faces[0]

                confidence = face.get(
                    "confidence",
                    0.0
                )

                # Reject weak detections
                if confidence < 0.90:

                    print(
                        f"Capture rejected: "
                        f"face confidence "
                        f"{confidence:.2f} is too low."
                    )

                    continue

                x, y, width, height = face["box"]

                x = max(0, x)
                y = max(0, y)

                width = max(1, width)
                height = max(1, height)

                x2 = min(
                    rgb_frame.shape[1],
                    x + width
                )

                y2 = min(
                    rgb_frame.shape[0],
                    y + height
                )

                face_image = rgb_frame[
                    y:y2,
                    x:x2
                ]

                if face_image.size == 0:

                    print(
                        "Capture rejected: "
                        "invalid face region."
                    )

                    continue

                # Generate FaceNet embedding
                embedding = get_embedding(
                    face_image,
                    embedder
                )

                embeddings.append(
                    embedding
                )

                pose_samples += 1

                print(
                    f"Captured sample "
                    f"{len(embeddings)}/{TOTAL_SAMPLES} "
                    f"(Pose {pose_index + 1}, "
                    f"{pose_samples}/{SAMPLES_PER_POSE})"
                )

    # ----------------------------------------
    # CLEANUP CAMERA
    # ----------------------------------------

    camera.release()
    cv2.destroyAllWindows()

    # ----------------------------------------
    # VERIFY SAMPLE COUNT
    # ----------------------------------------

    if len(embeddings) != TOTAL_SAMPLES:

        print(
            "\nRegistration failed."
        )

        print(
            f"Expected {TOTAL_SAMPLES} samples, "
            f"got {len(embeddings)}."
        )

        return

    # ----------------------------------------
    # CONVERT TO NUMPY ARRAY
    # ----------------------------------------

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    # ----------------------------------------
    # SAVE ALL EMBEDDINGS
    # ----------------------------------------

    person_file = os.path.join(
        REGISTERED_FOLDER,
        f"{name}.pkl"
    )

    data = {
        "name": name,
        "embeddings": embeddings
    }

    with open(person_file, "wb") as file:
        pickle.dump(data, file)
    # ----------------------------------------
    # SUCCESS MESSAGE
    # ----------------------------------------

    print("\n========================================")
    print("       REGISTRATION SUCCESSFUL")
    print("========================================")
    print(f"Person        : {name}")
    print(f"Samples       : {len(embeddings)}")
    print(f"Embeddings    : {embeddings.shape}")
    print(f"Saved file    : {person_file}")
    print("========================================")


if __name__ == "__main__":
    register_person()
    pickle.dump(
            {
                "name": name,
                "embedding": final_embedding
            },
            file
        )

    print(f"\nRegistration successful!")
    print(f"Saved: {person_file}")


if __name__ == "__main__":
    register_person()