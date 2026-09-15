import os
import pickle

import cv2
import numpy as np

from mtcnn import MTCNN
from keras_facenet import FaceNet


# ============================================================
# SETTINGS
# ============================================================

REGISTERED_FOLDER = "registered_faces"

# Starting value.
# We will tune this after testing.
RECOGNITION_THRESHOLD = 0.60


# ============================================================
# LOAD REGISTERED FACES
# ============================================================

def load_registered_faces():

    registered_faces = {}

    if not os.path.exists(
        REGISTERED_FOLDER
    ):

        print(
            "ERROR: registered_faces folder not found."
        )

        return registered_faces

    for filename in os.listdir(
        REGISTERED_FOLDER
    ):

        if not filename.endswith(".pkl"):
            continue

        filepath = os.path.join(
            REGISTERED_FOLDER,
            filename
        )

        try:

            with open(
                filepath,
                "rb"
            ) as file:

                data = pickle.load(file)

            name = data["name"]

            # ------------------------------------------------
            # NEW FORMAT
            # ------------------------------------------------

            if "embeddings" in data:

                embeddings = np.asarray(
                    data["embeddings"],
                    dtype=np.float32
                )

                registered_faces[name] = embeddings

            # ------------------------------------------------
            # OLD FORMAT
            # ------------------------------------------------

            elif "embedding" in data:

                embedding = np.asarray(
                    data["embedding"],
                    dtype=np.float32
                )

                # Convert old single embedding
                # into a one-element array.
                registered_faces[name] = np.expand_dims(
                    embedding,
                    axis=0
                )

            print(
                f"Loaded: {name} "
                f"({len(registered_faces[name])} embeddings)"
            )

        except Exception as error:

            print(
                f"Could not load "
                f"{filename}: {error}"
            )

    return registered_faces


# ============================================================
# COSINE SIMILARITY
# ============================================================

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


# ============================================================
# RECOGNIZE FACE
# ============================================================

def recognize_face(
    face_image,
    embedder,
    registered_faces
):

    # --------------------------------------------------------
    # RESIZE
    # --------------------------------------------------------

    face_image = cv2.resize(
        face_image,
        (160, 160)
    )

    # --------------------------------------------------------
    # ADD BATCH DIMENSION
    # --------------------------------------------------------

    face_image = np.expand_dims(
        face_image,
        axis=0
    )

    # --------------------------------------------------------
    # GENERATE LIVE EMBEDDING
    # --------------------------------------------------------

    embedding = embedder.embeddings(
        face_image
    )[0]

    best_name = "UNKNOWN"
    best_score = -1.0

    # --------------------------------------------------------
    # COMPARE WITH EVERY PERSON
    # --------------------------------------------------------

    for name, stored_embeddings in (
        registered_faces.items()
    ):

        person_best_score = -1.0

        # Compare live face with every sample
        # belonging to this person.
        for stored_embedding in stored_embeddings:

            score = cosine_similarity(
                embedding,
                stored_embedding
            )

            if score > person_best_score:

                person_best_score = score

        # Keep best person
        if person_best_score > best_score:

            best_score = person_best_score
            best_name = name

    # --------------------------------------------------------
    # APPLY THRESHOLD
    # --------------------------------------------------------

    if best_score >= RECOGNITION_THRESHOLD:

        return (
            best_name,
            best_score
        )

    return (
        "UNKNOWN",
        best_score
    )


# ============================================================
# MAIN TEST PROGRAM
# ============================================================

def main():

    # --------------------------------------------------------
    # LOAD REGISTERED FACES
    # --------------------------------------------------------

    registered_faces = (
        load_registered_faces()
    )

    if not registered_faces:

        print(
            "No registered faces found."
        )

        print(
            "Run register.py first."
        )

        return

    print()
    print("Registered people:")

    for name, embeddings in (
        registered_faces.items()
    ):

        print(
            f"- {name}: "
            f"{len(embeddings)} embeddings"
        )

    # --------------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------------

    print()
    print("Loading MTCNN...")

    detector = MTCNN()

    print(
        "Loading FaceNet..."
    )

    embedder = FaceNet()

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    camera = cv2.VideoCapture(
        0,
        cv2.CAP_MSMF
    )

    if not camera.isOpened():

        print(
            "ERROR: Could not open camera."
        )

        return

    print()
    print(
        "SENTRIX face recognition started."
    )

    print(
        "Press Q to quit."
    )

    # --------------------------------------------------------
    # LOOP
    # --------------------------------------------------------

    while True:

        success, frame = camera.read()

        if not success:

            print(
                "ERROR: Could not read frame."
            )

            break

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        faces = detector.detect_faces(
            rgb_frame
        )

        # ----------------------------------------------------
        # PROCESS FACES
        # ----------------------------------------------------

        for face in faces:

            x, y, width, height = (
                face["box"]
            )

            x = max(0, x)
            y = max(0, y)

            face_image = rgb_frame[
                y:y + height,
                x:x + width
            ]

            if face_image.size == 0:

                continue

            name, score = recognize_face(
                face_image,
                embedder,
                registered_faces
            )

            # ------------------------------------------------
            # COLOR
            # ------------------------------------------------

            if name == "UNKNOWN":

                box_color = (
                    0,
                    0,
                    255
                )

                label = (
                    f"UNKNOWN "
                    f"({score:.2f})"
                )

            else:

                box_color = (
                    0,
                    255,
                    0
                )

                label = (
                    f"{name} "
                    f"({score:.2f})"
                )

            # ------------------------------------------------
            # DRAW BOX
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x, y),
                (x + width, y + height),
                box_color,
                2
            )

            # ------------------------------------------------
            # DRAW LABEL
            # ------------------------------------------------

            cv2.putText(
                frame,
                label,
                (
                    x,
                    max(25, y - 10)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                box_color,
                2
            )

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

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
            "SENTRIX - Face Recognition",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break

    camera.release()

    cv2.destroyAllWindows()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()