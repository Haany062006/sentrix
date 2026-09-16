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

# Blur protection.
# Higher value = stricter.
MIN_BLUR_VARIANCE = 80.0

# Prevent almost identical frames from being captured.
MIN_FRAME_DIFFERENCE = 0.03

# Padding around detected face.
FACE_PADDING = 0.20


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_blur_score(image):
    """
    Measures image sharpness using Laplacian variance.
    Higher value generally means a sharper image.
    """
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()


def calculate_frame_difference(current, previous):
    """
    Calculates normalized difference between two face images.
    Returns a value between approximately 0 and 1.
    """
    if previous is None:
        return 1.0

    current_gray = cv2.cvtColor(
        current,
        cv2.COLOR_BGR2GRAY
    )

    previous_gray = cv2.cvtColor(
        previous,
        cv2.COLOR_BGR2GRAY
    )

    current_gray = cv2.resize(
        current_gray,
        (64, 64)
    )

    previous_gray = cv2.resize(
        previous_gray,
        (64, 64)
    )

    difference = cv2.absdiff(
        current_gray,
        previous_gray
    )

    return float(
        np.mean(difference) / 255.0
    )


def crop_face_with_padding(
    frame,
    box
):
    """
    Crops the detected face with a small margin.
    """
    x, y, width, height = box

    padding_x = int(
        width * FACE_PADDING
    )

    padding_y = int(
        height * FACE_PADDING
    )

    x1 = max(
        0,
        x - padding_x
    )

    y1 = max(
        0,
        y - padding_y
    )

    x2 = min(
        frame.shape[1],
        x + width + padding_x
    )

    y2 = min(
        frame.shape[0],
        y + height + padding_y
    )

    face = frame[
        y1:y2,
        x1:x2
    ]

    if face.size == 0:
        return None

    return cv2.resize(
        face,
        (160, 160)
    )


# ============================================================
# REGISTER PERSON
# ============================================================

def register_person():

    name = input(
        "Enter person's name: "
    ).strip()

    if not name:

        print(
            "ERROR: Name cannot be empty."
        )

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
            f"The dataset already contains "
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
    # LOAD MTCNN
    # --------------------------------------------------------

    print(
        "Loading MTCNN..."
    )

    detector = MTCNN()

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    print(
        "Opening camera..."
    )

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
    print("STABLE CAPTURE INSTRUCTIONS")
    print("------------------------------------------")
    print("Only ONE person should be visible.")
    print("Look directly at the camera first.")
    print("Keep your face clearly visible.")
    print("Move your head slightly between captures.")
    print("Change angle and expression naturally.")
    print()
    print("GREEN = face is ready")
    print("YELLOW = improve quality")
    print("RED = cannot capture")
    print()
    print("Press SPACE to capture.")
    print("Press Q to cancel.")
    print("------------------------------------------")
    print()

    previous_face = None

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
        blur_score = 0.0
        current_face = None

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

            x = max(
                0,
                x
            )

            y = max(
                0,
                y
            )

            width = max(
                0,
                width
            )

            height = max(
                0,
                height
            )

            # -----------------------------------------------
            # FACE QUALITY CHECK
            # -----------------------------------------------

            if (
                confidence >= MIN_CONFIDENCE
                and
                width >= MIN_FACE_WIDTH
                and
                height >= MIN_FACE_HEIGHT
            ):

                current_face = crop_face_with_padding(
                    frame,
                    (
                        x,
                        y,
                        width,
                        height
                    )
                )

                if current_face is not None:

                    blur_score = calculate_blur_score(
                        current_face
                    )

                    if blur_score >= MIN_BLUR_VARIANCE:

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

            difference = calculate_frame_difference(
                current_face,
                previous_face
            )

            if previous_face is None:

                quality_message = (
                    "FACE READY - SPACE TO CAPTURE"
                )

            elif difference >= MIN_FRAME_DIFFERENCE:

                quality_message = (
                    "GOOD VARIATION - SPACE TO CAPTURE"
                )

            else:

                quality_message = (
                    "TOO SIMILAR - MOVE SLIGHTLY"
                )

            cv2.rectangle(
                frame,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                quality_message,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Sharpness: {blur_score:.0f}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
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
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "SENTRIX - Stable Face Registration",
            frame
        )

        # ----------------------------------------------------
        # KEYBOARD
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        # ----------------------------------------------------
        # SPACE - CAPTURE
        # ----------------------------------------------------

        if key == 32:

            if valid_face is None:

                print(
                    "Cannot capture."
                )

                if len(faces) == 0:

                    print(
                        "Reason: Face not detected."
                    )

                elif len(faces) > 1:

                    print(
                        "Reason: Multiple faces detected."
                    )

                else:

                    print(
                        "Reason: Face quality is too low."
                    )

                continue

            difference = calculate_frame_difference(
                current_face,
                previous_face
            )

            if (
                previous_face is not None
                and
                difference < MIN_FRAME_DIFFERENCE
            ):

                print(
                    "Capture rejected: "
                    "too similar to previous image."
                )

                print(
                    "Move your head slightly."
                )

                continue

            # ------------------------------------------------
            # SAVE FACE
            # ------------------------------------------------

            filename = os.path.join(
                person_folder,
                f"{image_count + 1:03d}.jpg"
            )

            success = cv2.imwrite(
                filename,
                current_face
            )

            if not success:

                print(
                    "ERROR: Could not save image."
                )

                continue

            image_count += 1

            previous_face = current_face.copy()

            print(
                f"Captured "
                f"{image_count}/{SAMPLES_REQUIRED} | "
                f"Sharpness: {blur_score:.0f} | "
                f"Variation: {difference:.3f}"
            )

        # ----------------------------------------------------
        # Q - CANCEL
        # ----------------------------------------------------

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
            f"Need "
            f"{SAMPLES_REQUIRED - image_count} more."
        )

        return

    print()
    print(
        "40 quality-controlled images captured successfully."
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

        image = cv2.imread(
            filepath
        )

        if image is None:

            print(
                f"Skipping unreadable image: "
                f"{filename}"
            )

            continue

        # ----------------------------------------------------
        # CHECK BLUR AGAIN
        # ----------------------------------------------------

        blur_score = calculate_blur_score(
            image
        )

        if blur_score < MIN_BLUR_VARIANCE:

            print(
                f"Skipping blurry image: "
                f"{filename} "
                f"(sharpness={blur_score:.1f})"
            )

            continue

        # ----------------------------------------------------
        # RGB
        # ----------------------------------------------------

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = cv2.resize(
            image,
            (160, 160)
        )

        image = np.expand_dims(
            image,
            axis=0
        )

        # ----------------------------------------------------
        # FACENET
        # ----------------------------------------------------

        embedding = embedder.embeddings(
            image
        )[0]

        # ----------------------------------------------------
        # NORMALIZE
        # ----------------------------------------------------

        norm = np.linalg.norm(
            embedding
        )

        if norm < 1e-10:

            print(
                f"Skipping invalid embedding: "
                f"{filename}"
            )

            continue

        embedding = (
            embedding / norm
        )

        # ----------------------------------------------------
        # EMBEDDING QUALITY CHECK
        # ----------------------------------------------------

        if not np.all(
            np.isfinite(embedding)
        ):

            print(
                f"Skipping invalid values: "
                f"{filename}"
            )

            continue

        embeddings.append(
            embedding.astype(
                np.float32
            )
        )

        print(
            f"Processed "
            f"{index}/{len(image_files)}: "
            f"{filename}"
        )

    if not embeddings:

        print(
            "ERROR: No valid embeddings generated."
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