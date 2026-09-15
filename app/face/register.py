import os
import cv2
import pickle
import numpy as np

from mtcnn import MTCNN
from keras_facenet import FaceNet
from app.database.db import add_person


# ============================================================
# SETTINGS
# ============================================================

DATASET_FOLDER = "face_dataset"
REGISTERED_FOLDER = "registered_faces"

SAMPLES_REQUIRED = 40

CAMERA_INDEX = 0

MIN_CONFIDENCE = 0.90
MIN_FACE_WIDTH = 80
MIN_FACE_HEIGHT = 80


# ============================================================
# REGISTER PERSON
# ============================================================

def register_person():

    name = input(
        "Enter person's name: "
    ).strip()

    if not name:

        print("ERROR: Name cannot be empty.")
        return

    # --------------------------------------------------------
    # CREATE FOLDERS
    # --------------------------------------------------------

    person_folder = os.path.join(
        DATASET_FOLDER,
        name
    )

    os.makedirs(
        person_folder,
        exist_ok=True
    )

    os.makedirs(
        REGISTERED_FOLDER,
        exist_ok=True
    )

    # --------------------------------------------------------
    # CHECK EXISTING DATASET
    # --------------------------------------------------------

    existing_images = [
        filename
        for filename in os.listdir(person_folder)
        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    image_count = len(existing_images)

    print()
    print("==========================================")
    print("       SENTRIX FACE REGISTRATION")
    print("==========================================")
    print(f"Person       : {name}")
    print(f"Dataset      : {person_folder}")
    print(f"Existing     : {image_count}")
    print(f"Required     : {SAMPLES_REQUIRED}")
    print("==========================================")
    print()

    if image_count >= SAMPLES_REQUIRED:

        print(
            "The dataset already contains "
            f"{image_count} images."
        )

        answer = input(
            "Generate embeddings from these images? (y/n): "
        ).strip().lower()

        if answer == "y":

            generate_embeddings_from_dataset(
                name,
                person_folder
            )

        return

    # --------------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------------

    print("Loading MTCNN...")

    detector = MTCNN()

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    print("Opening camera...")

    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_MSMF
    )

    if not camera.isOpened():

        print(
            "ERROR: Could not open camera."
        )

        return

    print()
    print("------------------------------------------")
    print("CAPTURE INSTRUCTIONS")
    print("------------------------------------------")
    print("Only ONE person should be visible.")
    print("Keep your face clearly visible.")
    print("Change your head angle slightly.")
    print("Change expression naturally.")
    print()
    print("Press SPACE to capture.")
    print("Press Q to cancel.")
    print("------------------------------------------")
    print()

    # --------------------------------------------------------
    # CAPTURE LOOP
    # --------------------------------------------------------

    while image_count < SAMPLES_REQUIRED:

        success, frame = camera.read()

        if not success:

            print(
                "ERROR: Could not read camera frame."
            )

            break

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        faces = detector.detect_faces(
            rgb_frame
        )

        valid_face = None

        # ----------------------------------------------------
        # EXACTLY ONE FACE
        # ----------------------------------------------------

        if len(faces) == 1:

            face = faces[0]

            confidence = face.get(
                "confidence",
                0
            )

            x, y, width, height = face["box"]

            x = max(0, x)
            y = max(0, y)

            width = max(0, width)
            height = max(0, height)

            if (
                confidence >= MIN_CONFIDENCE
                and
                width >= MIN_FACE_WIDTH
                and
                height >= MIN_FACE_HEIGHT
            ):

                valid_face = (
                    x,
                    y,
                    width,
                    height
                )

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        if valid_face is not None:

            x, y, width, height = valid_face

            cv2.rectangle(
                frame,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "FACE READY - SPACE TO CAPTURE",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        elif len(faces) > 1:

            cv2.putText(
                frame,
                "ONLY ONE FACE ALLOWED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        else:

            cv2.putText(
                frame,
                "FACE NOT DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        cv2.putText(
            frame,
            f"Images: {image_count}/{SAMPLES_REQUIRED}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "SENTRIX - Face Dataset Registration",
            frame
        )

        # ----------------------------------------------------
        # KEYBOARD
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        # SPACE
        if key == 32:

            if valid_face is None:

                print(
                    "Cannot capture. "
                    "Make sure exactly one clear face "
                    "is visible."
                )

                continue

            x, y, width, height = valid_face

            face_image = frame[
                y:y + height,
                x:x + width
            ]

            if face_image.size == 0:
                continue

            # Resize face
            face_image = cv2.resize(
                face_image,
                (160, 160)
            )

            image_count += 1

            filename = os.path.join(
                person_folder,
                f"{image_count:03d}.jpg"
            )

            cv2.imwrite(
                filename,
                face_image
            )

            print(
                f"Captured "
                f"{image_count}/{SAMPLES_REQUIRED}: "
                f"{filename}"
            )

        # Q
        elif key == ord("q"):

            print(
                "Registration cancelled."
            )

            break

    camera.release()
    cv2.destroyAllWindows()

    # --------------------------------------------------------
    # CHECK COMPLETION
    # --------------------------------------------------------

    if image_count < SAMPLES_REQUIRED:

        print()
        print(
            f"Only {image_count} images captured."
        )

        print(
            "Need "
            f"{SAMPLES_REQUIRED - image_count} more."
        )

        return

    print()
    print(
        "40 images captured successfully."
    )

    # --------------------------------------------------------
    # GENERATE EMBEDDINGS
    # --------------------------------------------------------

    generate_embeddings_from_dataset(
        name,
        person_folder
    )


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

def generate_embeddings_from_dataset(
    name,
    person_folder
):

    print()
    print("==========================================")
    print("GENERATING FACENET EMBEDDINGS")
    print("==========================================")

    image_files = sorted(
        [
            filename
            for filename in os.listdir(person_folder)
            if filename.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ]
    )

    if not image_files:

        print(
            "ERROR: No images found."
        )

        return

    print(
        f"Found {len(image_files)} images."
    )

    print(
        "Loading FaceNet..."
    )

    embedder = FaceNet()

    embeddings = []

    for index, filename in enumerate(
        image_files,
        start=1
    ):

        filepath = os.path.join(
            person_folder,
            filename
        )

        image = cv2.imread(filepath)

        if image is None:

            print(
                f"Skipping unreadable image: "
                f"{filename}"
            )

            continue

        # Convert BGR to RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # FaceNet expects 160x160
        image = cv2.resize(
            image,
            (160, 160)
        )

        image = np.expand_dims(
            image,
            axis=0
        )

        embedding = embedder.embeddings(
            image
        )[0]

        # Normalize
        embedding = embedding / (
            np.linalg.norm(embedding) + 1e-10
        )

        embeddings.append(
            embedding.astype(np.float32)
        )

        print(
            f"Processed "
            f"{index}/{len(image_files)}: "
            f"{filename}"
        )

    if not embeddings:

        print(
            "ERROR: No embeddings generated."
        )

        return

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    person_file = os.path.join(
        REGISTERED_FOLDER,
        f"{name}.pkl"
    )

    data = {
        "name": name,
        "embeddings": embeddings
    }

    with open(
        person_file,
        "wb"
    ) as file:

        pickle.dump(
            data,
            file
        )

    print()
    print("==========================================")
    print("REGISTRATION SUCCESSFUL")
    print("==========================================")
    print(f"Person     : {name}")
    print(f"Images     : {len(embeddings)}")
    print(f"Embedding  : {embeddings.shape}")
    print(f"Saved      : {person_file}")
    print("==========================================")
    print()
    add_person(
        name=name,
        role="Person"
    )

    print(
        f"Person '{name}' added to SENTRIX database."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    register_person()