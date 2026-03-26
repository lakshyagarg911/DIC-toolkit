# dicUtils/seed.py
import cv2
import numpy as np


def select_seed_point(image):
    """
    Opens an interactive window for the user to click a seed point.
    Uses cv2 only for the UI window — no GPU involvement.
    Returns an (1, 1, 2) float32 numpy array compatible with the tracker.
    """
    seed_points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            seed_points.clear()
            seed_points.append((x, y))
            print("Selected seed:", seed_points)

    cv2.namedWindow("Select Seed")
    cv2.setMouseCallback("Select Seed", mouse_callback)

    while True:
        temp = image.copy()
        for pt in seed_points:
            cv2.circle(temp, pt, 6, 255, -1)

        cv2.imshow("Select Seed", temp)
        if cv2.waitKey(1) & 0xFF == 13:   # Enter to confirm
            break

    cv2.destroyWindow("Select Seed")

    if len(seed_points) == 0:
        raise RuntimeError("No seed selected.")

    return np.array(seed_points, dtype=np.float32).reshape(1, -1, 2)