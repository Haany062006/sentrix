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

# Starting threshold.
# We will tune this after seeing the actual similarity scores.
RECOGNITION_THRESHOLD = 0.60


# ============================================================
# LOAD REGISTERED FACES
# ============================================================

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

            with open(
                filepath,
                "rb"
            ) as file:

                data = pickle.load(file)

            # ------------------------------------------------
            # CHECK NAME
            # ------------------------------------------------

            name = data["name"]

            # ------------------------------------------------
            # NEW FORMAT
            # ------------------------------------------------

            if "embeddings" in data:

                embeddings = np.asarray(
                    data["embeddings"],
                    dtype=np.float32
                )

                # Make sure one embedding is also handled
                # as a list of embeddings.
                if embeddings.ndim == 1:
                    embeddings = np.expand_dims(
                        embeddings,
                        axis=0
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

                registered_faces[name] = np.expand_dims(
                    embedding,
                    axis=0
                )

            else:

                print(
                    f"WARNING: No embedding found in "
                    f"{filename}"
                )

                continue

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

    embedding1 = np.asarray(
        embedding1,
        dtype=np.float32
    )

    embedding2 = np.asarray(
        embedding2,
        dtype=np.float32
    )

    norm1 = np.linalg.norm(
        embedding1
    )

    norm2 = np.linalg.norm(
        embedding2
    )

    if norm1 < 1e-10 or norm2 < 1e-10:
        return -1.0

    embedding1 = embedding1 / norm1
    embedding2 = embedding2 / norm2

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
    # VALIDATE IMAGE
    #
    # IMPORTANT:
    # This function expects an RGB image.
    # The main camera loop converts BGR -> RGB before
    # extracting the face.
    # --------------------------------------------------------

    if face_image is None:

        return "UNKNOWN", 0.0

    if face_image.size == 0:

        return "UNKNOWN", 0.0

    # --------------------------------------------------------
    # RESIZE TO FACENET INPUT SIZE
    # --------------------------------------------------------

    face_image = cv2.resize(
        face_image,
        (160, 160),
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # CONVERT TO FLOAT32
    # --------------------------------------------------------

    face_image = face_image.astype(
        np.float32
    )

    # --------------------------------------------------------
    # ADD BATCH DIMENSION
    # Shape:
    # (160, 160, 3)
    #        ↓
    # (1, 160, 160, 3)
    # --------------------------------------------------------

    face_image = np.expand_dims(
        face_image,
        axis=0
    )

    # --------------------------------------------------------
    # GENERATE LIVE FACE EMBEDDING
    # --------------------------------------------------------

    try:

        embedding = embedder.embeddings(
            face_image
        )[0]

    except Exception as error:

        print(
            f"ERROR generating face embedding: "
            f"{error}"
        )

        return "UNKNOWN", 0.0

    # --------------------------------------------------------
    # INITIALIZE BEST MATCH
    # --------------------------------------------------------

    best_name = "UNKNOWN"
    best_score = -1.0

    # --------------------------------------------------------
    # COMPARE AGAINST EVERY REGISTERED PERSON
    # --------------------------------------------------------

    for name, stored_embeddings in (
        registered_faces.items()
    ):

        person_best_score = -1.0

        # ----------------------------------------------------
        # COMPARE AGAINST ALL EMBEDDINGS OF THIS PERSON
        # ----------------------------------------------------

        for stored_embedding in stored_embeddings:

            score = cosine_similarity(
                embedding,
                stored_embedding
            )

            if score > person_best_score:

                person_best_score = score

        # ----------------------------------------------------
        # KEEP THE BEST PERSON
        # ----------------------------------------------------

        if person_best_score > best_score:

            best_score = person_best_score
            best_name = name

    # --------------------------------------------------------
    # NO REGISTERED FACES
    # --------------------------------------------------------

    if best_score < 0:

        return "UNKNOWN", 0.0

    # --------------------------------------------------------
    # DEBUG INFORMATION
    #
    # This lets us see the actual similarity score.
    # --------------------------------------------------------

    # print(
    #     f"Best match: {best_name} "
    #     f"| Score: {best_score:.4f}"
    # )

    # --------------------------------------------------------
    # APPLY RECOGNITION THRESHOLD
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

    registered_faces = load_registered_faces()

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
    # LOAD MTCNN
    # --------------------------------------------------------

    print()
    print("Loading MTCNN...")

    detector = MTCNN()

    # --------------------------------------------------------
    # LOAD FACENET
    # --------------------------------------------------------

    print(
        "Loading FaceNet..."
    )

    embedder = FaceNet()

    # --------------------------------------------------------
    # OPEN CAMERA
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
    # CAMERA LOOP
    # --------------------------------------------------------

    while True:

        success, frame = camera.read()

        if not success:

            print(
                "ERROR: Could not read frame."
            )

            break

        # ----------------------------------------------------
        # OPENCV FRAME:
        # BGR
        #
        # MTCNN:
        # RGB
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # DETECT FACES
        # ----------------------------------------------------

        faces = detector.detect_faces(
            rgb_frame
        )

        # ----------------------------------------------------
        # PROCESS EACH FACE
        # ----------------------------------------------------

        for face in faces:

            # ------------------------------------------------
            # MTCNN CONFIDENCE
            # ------------------------------------------------

            confidence = face.get(
                "confidence",
                0.0
            )

            # Ignore very weak detections.
            if confidence < 0.90:

                continue

            # ------------------------------------------------
            # FACE BOX
            # ------------------------------------------------

            x, y, width, height = (
                face["box"]
            )

            # MTCNN can occasionally return
            # negative coordinates.
            x = max(0, x)
            y = max(0, y)

            # Prevent coordinates from exceeding frame.
            x2 = min(
                rgb_frame.shape[1],
                x + width
            )

            y2 = min(
                rgb_frame.shape[0],
                y + height
            )

            # ------------------------------------------------
            # VALIDATE BOX
            # ------------------------------------------------

            if x2 <= x or y2 <= y:

                continue

            # ------------------------------------------------
            # EXTRACT RGB FACE
            # ------------------------------------------------

            face_image = rgb_frame[
                y:y2,
                x:x2
            ]

            if face_image.size == 0:

                continue

            # ------------------------------------------------
            # RECOGNIZE
            #
            # recognize_face() expects RGB.
            # ------------------------------------------------

            name, score = recognize_face(
                face_image,
                embedder,
                registered_faces
            )

            # ------------------------------------------------
            # DISPLAY COLOR
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
            # DRAW FACE BOX
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x, y),
                (x2, y2),
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
                    max(
                        25,
                        y - 10
                    )
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

        # ----------------------------------------------------
        # SHOW CAMERA
        # ----------------------------------------------------

        cv2.imshow(
            "SENTRIX - Face Recognition",
            frame
        )

        # ----------------------------------------------------
        # QUIT
        # ----------------------------------------------------

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break

    # --------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------

    camera.release()

    cv2.destroyAllWindows()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()