"""Face embedding loading and recognition for SentriX.

Input contract: face_image must be an RGB image (as returned by MTCNN's
RGB-frame crop). Registered pickle files are read only; this module never
rewrites or deletes them.
"""
import os
import pickle
import cv2
import numpy as np

REGISTERED_FOLDER = "registered_faces"
RECOGNITION_THRESHOLD = 0.60


def _normalize_rows(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-10)


def load_registered_faces():
    registered_faces = {}
    if not os.path.isdir(REGISTERED_FOLDER):
        print("ERROR: registered_faces folder not found.")
        return registered_faces

    for filename in os.listdir(REGISTERED_FOLDER):
        if not filename.lower().endswith(".pkl"):
            continue
        filepath = os.path.join(REGISTERED_FOLDER, filename)
        try:
            with open(filepath, "rb") as file:
                data = pickle.load(file)
            name = data["name"]
            if "embeddings" in data:
                embeddings = _normalize_rows(data["embeddings"])
            elif "embedding" in data:
                embeddings = _normalize_rows(data["embedding"])
            else:
                print(f"Skipped {filename}: no embedding data found.")
                continue
            if embeddings.size == 0 or embeddings.shape[1] == 0:
                print(f"Skipped {filename}: empty embeddings.")
                continue
            registered_faces[name] = embeddings
            print(f"Loaded {name}: {len(embeddings)} embeddings")
        except Exception as error:
            print(f"Could not load {filename}: {error}")
    return registered_faces


def cosine_similarity(embedding1, embedding2):
    a = np.asarray(embedding1, dtype=np.float32).reshape(-1)
    b = np.asarray(embedding2, dtype=np.float32).reshape(-1)
    if a.shape != b.shape:
        return -1.0
    a = a / max(float(np.linalg.norm(a)), 1e-10)
    b = b / max(float(np.linalg.norm(b)), 1e-10)
    return float(np.dot(a, b))


def recognize_face(face_image, embedder, registered_faces):
    """Return (name, cosine score). face_image must be RGB."""
    if face_image is None or face_image.size == 0 or not registered_faces:
        return "UNKNOWN", 0.0

    # FaceNet expects RGB pixels. Do not convert to BGR in this function.
    face_image = cv2.resize(face_image, (160, 160), interpolation=cv2.INTER_AREA)
    face_batch = np.expand_dims(face_image.astype(np.uint8), axis=0)
    embedding = np.asarray(embedder.embeddings(face_batch)[0], dtype=np.float32)
    embedding = embedding / max(float(np.linalg.norm(embedding)), 1e-10)

    best_name = "UNKNOWN"
    best_score = -1.0
    for name, registered_embeddings in registered_faces.items():
        rows = _normalize_rows(registered_embeddings)
        if rows.shape[1] != embedding.shape[0]:
            continue
        scores = rows @ embedding
        # Average the best few samples instead of trusting one accidental
        # high-scoring registration sample.
        top_count = min(3, len(scores))
        person_score = float(np.mean(np.sort(scores)[-top_count:]))
        if person_score > best_score:
            best_name = name
            best_score = person_score

    if best_score >= RECOGNITION_THRESHOLD:
        return best_name, best_score
    return "UNKNOWN", best_score


def main():
    # Standalone recognition test. The integrated system uses app.main.
    from mtcnn import MTCNN
    from keras_facenet import FaceNet

    registered_faces = load_registered_faces()
    if not registered_faces:
        print("No registered faces found. Run register.py first.")
        return
    detector = MTCNN()
    embedder = FaceNet()
    camera = cv2.VideoCapture(0, cv2.CAP_MSMF)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("ERROR: Could not open camera.")
        return

    print("SentriX face recognition started. Press Q to quit.")
    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("ERROR: Could not read frame.")
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(rgb_frame) or []
            for face in faces:
                x, y, width, height = map(int, face["box"])
                x, y = max(0, x), max(0, y)
                x2 = min(rgb_frame.shape[1], x + max(1, width))
                y2 = min(rgb_frame.shape[0], y + max(1, height))
                if x2 <= x or y2 <= y:
                    continue
                name, score = recognize_face(rgb_frame[y:y2, x:x2], embedder, registered_faces)
                color = (0, 255, 0) if name != "UNKNOWN" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
                cv2.putText(frame, f"{name} ({score:.2f})", (x, max(25, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            cv2.imshow("SentriX - Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
