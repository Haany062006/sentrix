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

    with open(person_file, "wb") as file:
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