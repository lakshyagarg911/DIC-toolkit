import sys
import cv2
import os
from PyQt5.QtWidgets import QApplication
from utils.loader import DataLoaderUI
from utils.roi import get_roi


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    loader_window = DataLoaderUI()

    def on_data_ready(folder_path, metadata):
        print(f"Loaded: {folder_path}")

        # Load the first frame to define ROI
        first_frame_path = os.path.join(folder_path, "frame_000000.tiff")
        image = cv2.imread(first_frame_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print("Failed to load first frame.")
            return

        polygon, mask, bbox = get_roi(image)

        if polygon is not None:
            print(f"ROI selected. Bounding box: {bbox}")
            # Next Step: Init PyTorch Engine and HDF5 Datastore
        else:
            print("ROI selection cancelled.")

    loader_window.data_loaded.connect(on_data_ready)
    loader_window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()