# dicUtils/launcher.py
#
# DIC Launcher — startup screen.
# Three modes:
#   1. Load from video   (uses video_frame_extractor.py)
#   2. Load from folder  (lexicographic image folder)
#   3. Load saved data   (HDF5 .h5 file from previous run)

import os
import sys

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QFrame, QLineEdit,
    QSpinBox, QComboBox, QCheckBox, QProgressBar,
    QApplication, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPixmap, QPainter, QLinearGradient

from dicUtils.settings import DICSettings, SettingsPanel

C_BG     = "#0D1117"
C_PANEL  = "#161B22"
C_BORDER = "#30363D"
C_TEXT   = "#E6EDF3"
C_MUTED  = "#8B949E"
C_ACCENT = "#00E5FF"
C_WARN   = "#FF6B35"

LAUNCHER_STYLE = f"""
    QWidget {{ background: {C_BG}; color: {C_TEXT}; font-family: Courier; }}
    QLineEdit {{
        background: #21262D; border: 1px solid {C_BORDER};
        color: {C_TEXT}; padding: 6px; border-radius: 4px;
        font-family: Courier; font-size: 11px;
    }}
    QLineEdit:focus {{ border-color: {C_ACCENT}; }}
    QPushButton {{
        background: #21262D; color: {C_TEXT};
        border: 1px solid {C_BORDER}; border-radius: 4px;
        padding: 8px 16px; font-family: Courier;
        font-weight: bold; font-size: 11px;
    }}
    QPushButton:hover  {{ background: #30363D; }}
    QPushButton#primary {{
        background: {C_ACCENT}; color: {C_BG}; border: none;
        padding: 10px 20px; font-size: 12px;
    }}
    QPushButton#primary:hover {{ background: #33ECFF; }}
    QPushButton#mode {{
        background: {C_PANEL}; color: {C_TEXT};
        border: 2px solid {C_BORDER}; border-radius: 8px;
        padding: 18px; font-size: 13px; font-weight: bold;
    }}
    QPushButton#mode:hover  {{ border-color: {C_ACCENT}; color: {C_ACCENT}; }}
    QPushButton#mode:checked {{
        border-color: {C_ACCENT}; background: #0D2030; color: {C_ACCENT};
    }}
    QSpinBox, QComboBox {{
        background: #21262D; border: 1px solid {C_BORDER};
        color: {C_TEXT}; padding: 5px; border-radius: 3px;
        font-family: Courier; font-size: 11px;
    }}
    QProgressBar {{
        background: #21262D; border: 1px solid {C_BORDER};
        border-radius: 3px; text-align: center; color: {C_TEXT};
    }}
    QProgressBar::chunk {{ background: {C_ACCENT}; border-radius: 3px; }}
    QFrame#sep {{ background: {C_BORDER}; }}
    QLabel#title {{
        color: {C_ACCENT}; font-size: 22px; font-weight: bold;
    }}
    QLabel#subtitle {{ color: {C_MUTED}; font-size: 11px; }}
    QLabel#section  {{ color: {C_ACCENT}; font-size: 10px; font-weight: bold; }}
"""


# ── Video extraction thread ───────────────────────────────────────────────────

class VideoExtractThread(QThread):
    progress = pyqtSignal(int, int)
    done     = pyqtSignal(str)    # output folder
    error    = pyqtSignal(str)

    def __init__(self, video_path, out_dir, save_every_n, ext):
        super().__init__()
        self.video_path  = video_path
        self.out_dir     = out_dir
        self.save_every_n= save_every_n
        self.ext         = ext

    def run(self):
        try:
            # Import the extractor from dicUtils
            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from dicUtils.video_frame_extractor import extract_frames_dic_lossless_cv2
            extract_frames_dic_lossless_cv2(
                video_path    = self.video_path,
                out_dir       = self.out_dir,
                image_ext     = self.ext,
                save_every_n  = self.save_every_n,
                verbose       = False,
            )
            self.done.emit(self.out_dir)
        except Exception as e:
            self.error.emit(str(e))


# ── Launcher widget ───────────────────────────────────────────────────────────

class LauncherWidget(QWidget):
    """
    Startup screen. Emits `launch_folder` or `launch_saved` when ready.
    """
    launch_folder = pyqtSignal(str, float, str)   # folder, fps, ext
    launch_saved  = pyqtSignal(str)               # h5 path

    def __init__(self, settings: DICSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.setStyleSheet(LAUNCHER_STYLE)
        self._extract_thread = None
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(48, 36, 48, 36)
        layout.setSpacing(0)

        # ── Title ─────────────────────────────────────────────────────────────
        title = QLabel("DENSE DIC")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        sub = QLabel("Digital Image Correlation — Optical Flow Analysis")
        sub.setObjectName("subtitle")
        sub.setAlignment(Qt.AlignCenter)
        layout.addWidget(sub)

        layout.addSpacing(32)

        # ── Mode buttons ──────────────────────────────────────────────────────
        mode_row = QHBoxLayout()
        mode_row.setSpacing(16)

        self._btn_video  = QPushButton("🎥\n\nLoad from Video\n\n.mp4 / .avi / .mov")
        self._btn_folder = QPushButton("🖼\n\nLoad from Folder\n\nImage sequence")
        self._btn_saved  = QPushButton("💾\n\nLoad Saved Data\n\n.h5 DIC file")

        for b in [self._btn_video, self._btn_folder, self._btn_saved]:
            b.setObjectName("mode")
            b.setCheckable(True)
            b.setMinimumHeight(130)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            mode_row.addWidget(b)

        self._btn_video.clicked.connect(lambda: self._select_mode("video"))
        self._btn_folder.clicked.connect(lambda: self._select_mode("folder"))
        self._btn_saved.clicked.connect(lambda: self._select_mode("saved"))

        layout.addLayout(mode_row)
        layout.addSpacing(24)

        # ── Config area (swapped based on mode) ──────────────────────────────
        self._config_area = QWidget()
        self._config_layout = QVBoxLayout(self._config_area)
        self._config_layout.setContentsMargins(0, 0, 0, 0)
        self._config_layout.setSpacing(10)
        layout.addWidget(self._config_area)

        layout.addSpacing(16)

        # ── Progress bar (hidden until extraction) ─────────────────────────
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._progress.setFixedHeight(20)
        layout.addWidget(self._progress)

        self._progress_lbl = QLabel("")
        self._progress_lbl.setStyleSheet(f"color: {C_MUTED}; font-size: 10px;")
        self._progress_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._progress_lbl)

        layout.addSpacing(16)

        # ── Launch button ─────────────────────────────────────────────────────
        self._launch_btn = QPushButton("▶  CONTINUE")
        self._launch_btn.setObjectName("primary")
        self._launch_btn.setFixedHeight(46)
        self._launch_btn.clicked.connect(self._on_launch)
        layout.addWidget(self._launch_btn)

        layout.addStretch()

        # Default mode
        self._mode = None
        self._select_mode("folder")

    def _clear_config(self):
        while self._config_layout.count():
            item = self._config_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _select_mode(self, mode):
        self._mode = mode
        for btn, m in [(self._btn_video,"video"),
                       (self._btn_folder,"folder"),
                       (self._btn_saved,"saved")]:
            btn.setChecked(m == mode)
        self._clear_config()
        if mode == "video":
            self._build_video_config()
        elif mode == "folder":
            self._build_folder_config()
        else:
            self._build_saved_config()

    def _row(self, label, widget):
        r = QWidget()
        h = QHBoxLayout(r)
        h.setContentsMargins(0,0,0,0)
        l = QLabel(label)
        l.setMinimumWidth(140)
        l.setStyleSheet(f"color: {C_MUTED}; font-size: 10px;")
        h.addWidget(l)
        h.addWidget(widget, stretch=1)
        return r

    def _browse_btn(self, line_edit, mode="file", filter=""):
        btn = QPushButton("Browse")
        btn.setFixedWidth(80)
        def _do():
            if mode == "file":
                p, _ = QFileDialog.getOpenFileName(self, "Select file", "", filter)
            else:
                p = QFileDialog.getExistingDirectory(self, "Select folder")
            if p:
                line_edit.setText(p)
        btn.clicked.connect(_do)
        return btn

    # ── Video config ──────────────────────────────────────────────────────────

    def _build_video_config(self):
        cl = self._config_layout

        lbl = QLabel("VIDEO SOURCE")
        lbl.setObjectName("section")
        cl.addWidget(lbl)

        # Video path
        r1 = QWidget(); h1 = QHBoxLayout(r1); h1.setContentsMargins(0,0,0,0)
        self._video_path = QLineEdit()
        self._video_path.setPlaceholderText("Path to video file...")
        h1.addWidget(self._video_path, stretch=1)
        h1.addWidget(self._browse_btn(self._video_path, "file",
                                      "Video (*.mp4 *.avi *.mov *.mkv)"))
        cl.addWidget(r1)

        # Output folder
        r2 = QWidget(); h2 = QHBoxLayout(r2); h2.setContentsMargins(0,0,0,0)
        self._video_out = QLineEdit("frames")
        self._video_out.setPlaceholderText("Output folder for extracted frames...")
        h2.addWidget(self._video_out, stretch=1)
        h2.addWidget(self._browse_btn(self._video_out, "dir"))
        cl.addWidget(r2)

        # Options
        opts = QWidget(); oh = QHBoxLayout(opts); oh.setContentsMargins(0,0,0,0)

        oh.addWidget(QLabel("Save every N frames:"))
        self._video_n = QSpinBox()
        self._video_n.setRange(1, 100); self._video_n.setValue(1)
        oh.addWidget(self._video_n)

        oh.addSpacing(16)
        oh.addWidget(QLabel("Format:"))
        self._video_ext = QComboBox()
        self._video_ext.addItems(["tiff", "png"])
        oh.addWidget(self._video_ext)

        oh.addSpacing(16)
        oh.addWidget(QLabel("FPS override:"))
        self._video_fps = QSpinBox()
        self._video_fps.setRange(1, 10000); self._video_fps.setValue(50)
        self._video_fps.setSpecialValueText("auto")
        oh.addWidget(self._video_fps)
        oh.addStretch()
        cl.addWidget(opts)



    def _extract_video(self):
        vp = self._video_path.text().strip()
        od = self._video_out.text().strip()
        if not vp or not os.path.exists(vp):
            self._progress_lbl.setText("⚠  Select a valid video file first.")
            return
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)   # indeterminate
        self._progress_lbl.setText("Extracting frames...")
        self._launch_btn.setEnabled(False)

        self._extract_thread = VideoExtractThread(
            vp, od,
            self._video_n.value(),
            self._video_ext.currentText()
        )
        self._extract_thread.done.connect(self._on_extract_done)
        self._extract_thread.error.connect(self._on_extract_error)
        self._extract_thread.start()

    def _on_extract_then_launch(self, folder):
        self._progress.setRange(0, 1); self._progress.setValue(1)
        self._progress_lbl.setText(f"✓  Extracted to: {folder}")
        self._launch_btn.setEnabled(True)
        self.launch_folder.emit(folder, self._pending_fps, self._pending_ext)

    def _on_extract_done(self, folder):
        self._progress.setRange(0, 1); self._progress.setValue(1)
        self._progress_lbl.setText(f"✓  Extracted to: {folder}")
        self._launch_btn.setEnabled(True)
        self._video_out.setText(folder)

    def _on_extract_error(self, msg):
        self._progress.setVisible(False)
        self._progress_lbl.setText(f"⚠  {msg}")
        self._launch_btn.setEnabled(True)

    # ── Folder config ─────────────────────────────────────────────────────────

    def _build_folder_config(self):
        cl = self._config_layout

        lbl = QLabel("IMAGE FOLDER")
        lbl.setObjectName("section")
        cl.addWidget(lbl)

        r1 = QWidget(); h1 = QHBoxLayout(r1); h1.setContentsMargins(0,0,0,0)
        self._folder_path = QLineEdit()
        self._folder_path.setPlaceholderText("Folder containing image sequence...")
        h1.addWidget(self._folder_path, stretch=1)
        h1.addWidget(self._browse_btn(self._folder_path, "dir"))
        cl.addWidget(r1)

        opts = QWidget(); oh = QHBoxLayout(opts); oh.setContentsMargins(0,0,0,0)
        oh.addWidget(QLabel("FPS:"))
        self._folder_fps = QSpinBox()
        self._folder_fps.setRange(1, 10000); self._folder_fps.setValue(50)
        oh.addWidget(self._folder_fps)
        oh.addSpacing(16)
        oh.addWidget(QLabel("Extension:"))
        self._folder_ext = QComboBox()
        self._folder_ext.addItems([".tiff", ".tif", ".png", ".jpg", ".bmp"])
        oh.addWidget(self._folder_ext)
        oh.addStretch()
        cl.addWidget(opts)

    # ── Saved config ──────────────────────────────────────────────────────────

    def _build_saved_config(self):
        cl = self._config_layout

        lbl = QLabel("SAVED DIC DATA")
        lbl.setObjectName("section")
        cl.addWidget(lbl)

        r1 = QWidget(); h1 = QHBoxLayout(r1); h1.setContentsMargins(0,0,0,0)
        self._saved_path = QLineEdit()
        self._saved_path.setPlaceholderText("Path to .h5 DIC file...")
        h1.addWidget(self._saved_path, stretch=1)
        h1.addWidget(self._browse_btn(self._saved_path, "file",
                                      "DIC data (*.h5 *.hdf5)"))
        cl.addWidget(r1)

    # ── Launch ────────────────────────────────────────────────────────────────

    def _on_launch(self):
        if self._mode == "video":
            vp     = self._video_path.text().strip()
            folder = self._video_out.text().strip()
            fps    = float(self._video_fps.value())
            ext    = "." + self._video_ext.currentText()
            if not vp or not os.path.exists(vp):
                self._progress_lbl.setText("⚠  Select a valid video file first.")
                return
            if not folder:
                self._progress_lbl.setText("⚠  Set output folder.")
                return
            # Extract frames first, then launch
            self._progress.setVisible(True)
            self._progress.setRange(0, 0)
            self._progress_lbl.setText("Extracting frames from video...")
            self._launch_btn.setEnabled(False)
            self._pending_fps = fps
            self._pending_ext = ext
            self._pending_folder = folder
            self._extract_thread = VideoExtractThread(
                vp, folder, self._video_n.value(),
                self._video_ext.currentText()
            )
            self._extract_thread.done.connect(self._on_extract_then_launch)
            self._extract_thread.error.connect(self._on_extract_error)
            self._extract_thread.start()
            return

        elif self._mode == "folder":
            folder = self._folder_path.text().strip()
            fps    = float(self._folder_fps.value())
            ext    = self._folder_ext.currentText()
            if not folder or not os.path.isdir(folder):
                self._progress_lbl.setText("⚠  Select a valid folder.")
                return
            self.launch_folder.emit(folder, fps, ext)

        elif self._mode == "saved":
            path = self._saved_path.text().strip()
            if not path or not os.path.exists(path):
                self._progress_lbl.setText("⚠  Select a valid .h5 file.")
                return
            self.launch_saved.emit(path)