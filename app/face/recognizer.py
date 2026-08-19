import os
import pickle

import cv2
import numpy as np

from mtcnn import MTCNN
from keras_facenet import FaceNet


REGISTERED_FOLDER = "registered_faces"

# Starting threshold.
# We can tune this after testing with several people.
RECOGNITION_THRESHOLD = 0.60


def load_registered_faces():

    registered_faces = {}

    if not os.path.exists(REGISTERED_FOLDER):
        print("ERROR: registered_faces folder not found.")
        return registered_faces

    for filename in os.listdir(REGISTERED_FOLDER):

        if not filename.endswith(".pkl"):
            continue

        filepath = os.path.join(
            REGISTERED_FOLDER,
            filename
        )

        try:
            with open(filepath, "rb") as file:
                data = pickle.load(file)

            name = data["name"]

            embedding = np.asarray(
                data["embedding"],
                dtype=np.float32
            )

            registered_faces[name] = embedding

        except Exception as error:
            print(
                f"Could not load {filename}: {error}"
            )

    return registered_faces


def cosine_similarity(
    embedding1,
    embedding2
):

    embedding1 = embedding1 / (
        np.linalg.norm(embedding1) + 1e-10
    )

    embedding2 = embedding2 / (
        np.linalg.norm(embedding2) + 1e-10
    )

    return float(
        np.dot(
            embedding1,
            embedding2
        )
    )


def recognize_face(
    face_image,
    embedder,
    registered_faces
):

    # Resize face for FaceNet
    face_image = cv2.resize(
        face_image,
        (160, 160)
    )

    # Add batch dimension
    face_image = np.expand_dims(
        face_image,
        axis=0
    )

    # Generate FaceNet embedding
    embedding = embedder.embeddings(
        face_image
    )[0]

    best_name = "UNKNOWN"
    best_score = -1.0

    # Compare against every registered person
    for name, registered_embedding in registered_faces.items():

        score = cosine_similarity(
            embedding,
            registered_embedding
        )

        if score > best_score:
            best_score = score
            best_name = name

    # Apply recognition threshold
    if best_score >= RECOGNITION_THRESHOLD:
        return best_name, best_score

    return "UNKNOWN", best_score


def main():

    # --------------------------------
    # LOAD REGISTERED FACES
    # --------------------------------

    registered_faces = load_registered_faces()

    if not registered_faces:

        print("No registered faces found.")
        print("Run register.py first.")

        return

    print("\nRegistered people:")

    for name in registered_faces:
        print(f"- {name}")

    # --------------------------------
    # INITIALIZE MODELS
    # --------------------------------

    print("\nLoading MTCNN...")

    detector = MTCNN()

    print("Loading FaceNet...")

    embedder = FaceNet()

    # --------------------------------
    # OPEN CAMERA
    # --------------------------------

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        print(
            "ERROR: Could not open camera."
        )

        return

    print("\nSentriX face recognition started.")
    print("Press Q to quit.")

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

        # MTCNN expects RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Detect faces
        faces = detector.detect_faces(
            rgb_frame
        )

        # --------------------------------
        # PROCESS EACH FACE
        # --------------------------------

        for face in faces:

            x, y, width, height = face["box"]

            # Prevent negative coordinates
            x = max(0, x)
            y = max(0, y)

            # Crop face
            face_image = rgb_frame[
                y:y + height,
                x:x + width
            ]

            if face_image.size == 0:
                continue

            # Recognize face
            name, score = recognize_face(
                face_image,
                embedder,
                registered_faces
            )

            # --------------------------------
            # CHOOSE BOX COLOR
            # --------------------------------

            if name == "UNKNOWN":

                # Red in OpenCV = BGR (0, 0, 255)
                box_color = (0, 0, 255)

                label = (
                    f"UNKNOWN ({score:.2f})"
                )

            else:

                # Green in OpenCV = BGR (0, 255, 0)
                box_color = (0, 255, 0)

                label = (
                    f"{name} ({score:.2f})"
                )

            # --------------------------------
            # DRAW FACE BOX
            # --------------------------------

            cv2.rectangle(
                frame,
                (x, y),
                (x + width, y + height),
                box_color,
                2
            )

            # --------------------------------
            # DRAW LABEL
            # --------------------------------

            cv2.putText(
                frame,
                label,
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                box_color,
                2
            )

        # --------------------------------
        # DISPLAY STATUS
        # --------------------------------

        cv2.putText(
            frame,
            f"Faces detected: {len(faces)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "SentriX - Face Recognition",
            frame
        )

        # --------------------------------
        # QUIT
        # --------------------------------

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # --------------------------------
    # CLEANUP
    # --------------------------------

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()