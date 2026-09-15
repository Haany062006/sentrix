import os
import pickle
import cv2
import numpy as np
from keras_facenet import FaceNet

from app.database.db import add_person


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_FOLDER = "face_dataset"
REGISTERED_FOLDER = "registered_faces"

IMAGE_SIZE = (160, 160)

# Process a few images at a time so CPU/RAM usage stays reasonable
BATCH_SIZE = 8


# ============================================================
# LOAD FACENET
# ============================================================

print("=" * 60)
print("SENTRIX - FaceNet Embedding Builder")
print("=" * 60)

print("\nLoading FaceNet model...")
embedder = FaceNet()
print("FaceNet loaded successfully.")


# ============================================================
# FIND PEOPLE
# ============================================================

if not os.path.exists(DATASET_FOLDER):
    print(f"\nERROR: Dataset folder not found: {DATASET_FOLDER}")
    raise SystemExit

people = [
    name
    for name in os.listdir(DATASET_FOLDER)
    if os.path.isdir(os.path.join(DATASET_FOLDER, name))
]

if not people:
    print("\nERROR: No person folders found.")
    raise SystemExit

print("\nPeople found:")

for person in people:
    print(f"  - {person}")


# ============================================================
# CREATE REGISTERED FOLDER
# ============================================================

os.makedirs(REGISTERED_FOLDER, exist_ok=True)


# ============================================================
# PROCESS EACH PERSON
# ============================================================

for person_name in people:

    person_folder = os.path.join(
        DATASET_FOLDER,
        person_name
    )

    print("\n" + "=" * 60)
    print(f"Processing: {person_name}")
    print("=" * 60)

    image_files = [
        f
        for f in os.listdir(person_folder)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    image_files.sort()

    if not image_files:
        print("No images found. Skipping.")
        continue

    print(f"Found {len(image_files)} images.")

    images = []
    valid_names = []

    # --------------------------------------------------------
    # LOAD IMAGES
    # --------------------------------------------------------

    for filename in image_files:

        path = os.path.join(
            person_folder,
            filename
        )

        image = cv2.imread(path)

        if image is None:

            print(
                f"WARNING: Could not read {filename}"
            )

            continue

        image = cv2.resize(
            image,
            IMAGE_SIZE
        )

        # OpenCV = BGR
        # FaceNet expects RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        images.append(image)
        valid_names.append(filename)

    if not images:

        print(
            "No valid images. Skipping."
        )

        continue

    images = np.array(
        images,
        dtype=np.float32
    )

    print(
        f"Valid images: {len(images)}"
    )

    print(
        "Generating embeddings..."
    )

    # --------------------------------------------------------
    # GENERATE EMBEDDINGS IN SMALL BATCHES
    # --------------------------------------------------------

    all_embeddings = []

    total = len(images)

    for start in range(
        0,
        total,
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            total
        )

        batch = images[start:end]

        print(
            f"  Processing images "
            f"{start + 1}-{end} "
            f"of {total}..."
        )

        batch_embeddings = (
            embedder.embeddings(batch)
        )

        # Normalize each embedding
        norms = np.linalg.norm(
            batch_embeddings,
            axis=1,
            keepdims=True
        )

        batch_embeddings = (
            batch_embeddings
            / np.maximum(
                norms,
                1e-10
            )
        )

        all_embeddings.extend(
            batch_embeddings
        )

    embeddings = np.array(
        all_embeddings
    )

    print(
        f"\nEmbeddings generated: "
        f"{len(embeddings)}"
    )

    print(
        f"Embedding dimension: "
        f"{embeddings.shape[1]}"
    )

    # --------------------------------------------------------
    # SAVE SAFELY
    # --------------------------------------------------------

    output_file = os.path.join(
        REGISTERED_FOLDER,
        f"{person_name}.pkl"
    )

    temp_file = (
        output_file
        + ".tmp"
    )

    data = {
        "name": person_name,
        "embeddings": embeddings
    }

    with open(
        temp_file,
        "wb"
    ) as file:

        pickle.dump(
            data,
            file
        )

    # Replace old file only after
    # successful generation
    os.replace(
        temp_file,
        output_file
    )

    print(
        "Saved successfully:"
    )

    print(
        f"  {output_file}"
    )

    # --------------------------------------------------------
    # REGISTER PERSON IN SQLITE
    # --------------------------------------------------------

    person_id = add_person(
        person_name
    )

    if person_id:

        print(
            "Database registration successful:"
        )

        print(
            f"  Person ID: {person_id}"
        )

        print(
            f"  Name: {person_name}"
        )

    else:

        print(
            "WARNING: Could not register "
            "person in database."
        )


# ============================================================
# FINISHED
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "ALL EMBEDDINGS GENERATED SUCCESSFULLY"
)

print(
    "=" * 60
)

print(
    "\nRegistered face files:"
)

for filename in os.listdir(
    REGISTERED_FOLDER
):

    if filename.lower().endswith(
        ".pkl"
    ):

        print(
            f"  - {filename}"
        )

print(
    "\nDatabase registration completed."
)

print(
    "\nYou can now test the recognizer."
)