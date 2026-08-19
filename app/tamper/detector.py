import cv2

from brightness import detect_darkness
from blur import detect_blur
from histogram import (
    calculate_histogram,
    detect_scene_change
)


def detect_tampering():

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        print(
            "ERROR: Could not open camera."
        )

        return

    success, frame = camera.read()

    if not success:

        print(
            "ERROR: Could not read frame."
        )

        return

    reference_histogram = (
        calculate_histogram(frame)
    )

    print(
        "SentriX tamper detection started."
    )

    print(
        "Press Q to quit."
    )

    while True:

        success, frame = camera.read()

        if not success:
            break

        dark, brightness = (
            detect_darkness(frame)
        )

        blurry, variance = (
            detect_blur(frame)
        )

        moved, correlation = (
            detect_scene_change(
                reference_histogram,
                frame
            )
        )
        tampering = False

        if dark:
            tampering = True

        if blurry and variance < 10:
            tampering = True

        if moved and correlation > 0:
            tampering = True

        if tampering:

            if dark:
                text = "LENS COVERED"

            elif blurry:
                text = "CAMERA BLURRED"

            elif moved:
                text = "CAMERA MOVED"

            else:
                text = "CAMERA OK"

            color = (
                0,
                0,
                255
            )

        else:

            text = "CAMERA OK"

            color = (
                0,
                255,
                0
            )

        cv2.putText(
            frame,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        cv2.putText(
            frame,
            f"Brightness: "
            f"{brightness:.0f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Laplacian: "
            f"{variance:.0f}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Histogram: "
            f"{correlation:.2f}",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "SentriX Tamper Detection",
            frame
        )

        if (
            cv2.waitKey(1)
            & 0xFF
            == ord("q")
        ):

            break

    camera.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":

    detect_tampering()