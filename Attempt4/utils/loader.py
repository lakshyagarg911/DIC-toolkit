import os
import json
import cv2
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QFileDialog, QLabel, QProgressBar, QMessageBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt


class VideoExtractorThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)

    def __init__(self, video_path, output_dir):
        super().__init__()
        self.video_path = video_path
        self.output_dir = output_dir

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        os.makedirs(self.output_dir, exist_ok=True)

        metadata = {
            "original_video": self.video_path,
            "fps": fps,
            "total_frames": total_frames,
            "extension": ".tiff"
        }

        with open(os.path.join(self.output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)

        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_path = os.path.join(self.output_dir, f"frame_{count:06d}.tiff")
            cv2.imwrite(frame_path, gray)

            count += 1
            if count % 10 == 0 or count == total_frames:
                self.progress.emit(int((count / total_frames) * 100))

        cap.release()
        self.finished.emit(self.output_dir)


class DataLoaderUI(QWidget):
    data_loaded = pyqtSignal(str, dict)

    def __init__(self):
        super().__init__()
        self.video_path = ""
        self.output_dir = ""
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("DIC Data Initialisation")
        self.setMinimumSize(800, 500)

        self.setStyleSheet("""
            QWidget {
                background-color: #0f172a;
                color: #f8fafc;
                font-family: 'Segoe UI', system-ui, sans-serif;
            }
            QLabel#Title {
                font-size: 32px;
                font-weight: 800;
                color: #38bdf8;
                margin-bottom: 20px;
            }
            QLabel#Info {
                font-size: 16px;
                color: #cbd5e1;
                margin: 5px;
            }
            QPushButton {
                background-color: #2563eb;
                color: white;
                font-size: 20px;
                font-weight: bold;
                border-radius: 15px;
                padding: 20px;
                margin: 10px;
                border: 2px solid transparent;
            }
            QPushButton:hover {
                background-color: #3b82f6;
                border: 2px solid #60a5fa;
            }
            QPushButton#Secondary {
                background-color: #475569;
                font-size: 16px;
                padding: 10px;
            }
            QPushButton#Secondary:hover {
                background-color: #64748b;
                border: 2px solid #94a3b8;
            }
            QPushButton#Start {
                background-color: #10b981;
            }
            QPushButton#Start:hover {
                background-color: #34d399;
                border: 2px solid #6ee7b7;
            }
            QProgressBar {
                border: 2px solid #334155;
                border-radius: 15px;
                text-align: center;
                font-size: 18px;
                font-weight: bold;
                color: #f8fafc;
                background-color: #1e293b;
                height: 40px;
                margin: 20px 10px;
            }
            QProgressBar::chunk {
                background-color: #10b981;
                border-radius: 12px;
            }
        """)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        self.label = QLabel("Select Data Source")
        self.label.setObjectName("Title")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        # Initial Button Container
        self.btn_container = QWidget()
        btn_layout = QHBoxLayout(self.btn_container)

        self.btn_video = QPushButton("🎬 Extract from Video")
        self.btn_video.setCursor(Qt.PointingHandCursor)
        self.btn_video.clicked.connect(self.select_video)
        btn_layout.addWidget(self.btn_video)

        self.btn_folder = QPushButton("📁 Load Frames Folder")
        self.btn_folder.setCursor(Qt.PointingHandCursor)
        self.btn_folder.clicked.connect(self.load_folder)
        btn_layout.addWidget(self.btn_folder)

        layout.addWidget(self.btn_container)

        # Video Extraction Confirmation Container (Hidden initially)
        self.confirm_container = QWidget()
        confirm_layout = QVBoxLayout(self.confirm_container)

        self.lbl_vid_path = QLabel("")
        self.lbl_vid_path.setObjectName("Info")
        self.lbl_vid_path.setWordWrap(True)

        self.lbl_out_path = QLabel("")
        self.lbl_out_path.setObjectName("Info")
        self.lbl_out_path.setWordWrap(True)

        self.btn_change_dir = QPushButton("⚙️ Change Output Folder")
        self.btn_change_dir.setObjectName("Secondary")
        self.btn_change_dir.setCursor(Qt.PointingHandCursor)
        self.btn_change_dir.clicked.connect(self.change_output_dir)

        self.btn_start = QPushButton("🚀 Start Extraction")
        self.btn_start.setObjectName("Start")
        self.btn_start.setCursor(Qt.PointingHandCursor)
        self.btn_start.clicked.connect(self.start_extraction)

        confirm_layout.addWidget(self.lbl_vid_path)
        confirm_layout.addWidget(self.lbl_out_path)
        confirm_layout.addWidget(self.btn_change_dir)
        confirm_layout.addWidget(self.btn_start)
        self.confirm_container.hide()

        layout.addWidget(self.confirm_container)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        self.setLayout(layout)

    def select_video(self):
        video_path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if not video_path:
            return

        self.video_path = video_path

        # Generate default folder path
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        dir_name = os.path.dirname(video_path)
        self.output_dir = os.path.join(dir_name, f"{base_name}_frames")

        # Update UI
        self.btn_container.hide()
        self.lbl_vid_path.setText(f"🎥 Video: {os.path.basename(self.video_path)}")
        self.lbl_out_path.setText(f"📁 Output: {self.output_dir}")
        self.confirm_container.show()

    def change_output_dir(self):
        new_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory", self.output_dir)
        if new_dir:
            self.output_dir = new_dir
            self.lbl_out_path.setText(f"📁 Output: {self.output_dir}")

    def start_extraction(self):
        self.confirm_container.hide()
        self.progress_bar.show()
        self.progress_bar.setValue(0)
        self.label.setText("Extracting High-Fidelity Frames...")

        self.thread = VideoExtractorThread(self.video_path, self.output_dir)
        self.thread.progress.connect(self.progress_bar.setValue)
        self.thread.finished.connect(self.on_extraction_finished)
        self.thread.start()

    def on_extraction_finished(self, folder_path):
        self.progress_bar.hide()
        self.parse_and_emit(folder_path)

    def load_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Frames Folder")
        if not folder_path:
            return
        self.parse_and_emit(folder_path)

    def parse_and_emit(self, folder_path):
        meta_path = os.path.join(folder_path, 'metadata.json')
        if not os.path.exists(meta_path):
            QMessageBox.warning(self, "Error",
                                "metadata.json not found. Please extract from a video first or ensure the folder contains valid DIC frames and metadata.")
            # Reset UI if they selected a bad folder
            self.btn_container.show()
            self.confirm_container.hide()
            self.progress_bar.hide()
            return

        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        self.data_loaded.emit(folder_path, metadata)
        self.close()