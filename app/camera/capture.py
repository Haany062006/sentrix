import cv2


def start_camera():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open camera.")
        return

    print("SentriX camera started.")
    print("Press Q to quit.")

    while True:
        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read frame.")
            break

        cv2.imshow("SentriX Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_camera()