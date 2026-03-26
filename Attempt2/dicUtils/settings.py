# dicUtils/settings.py
#
# DIC application settings — dataclass + PyQt5 settings panel widget.

import json
import os
from dataclasses import dataclass, asdict, field
from typing import List

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QSlider, QCheckBox, QGroupBox,
    QPushButton, QDialog, QScrollArea, QFrame,
    QSpinBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor


# ── Palette (duplicated here to avoid circular import) ────────────────────────
C_BG     = "#0D1117"
C_PANEL  = "#161B22"
C_BORDER = "#30363D"
C_TEXT   = "#E6EDF3"
C_MUTED  = "#8B949E"
C_ACCENT = "#00E5FF"


_SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".dic_settings.json")


# ── Settings dataclass ────────────────────────────────────────────────────────

@dataclass
class DICSettings:
    # Display
    colormap:          str   = "jet"
    overlay_alpha:     float = 0.88
    norm_mode:         str   = "percentile"   # "percentile" | "global_max"
    norm_percentile:   float = 99.0

    # Hover zoom
    hover_zoom_mode:   str   = "tooltip"      # "tooltip" | "popup" | "panel"
    hover_zoom_size:   int   = 180            # px radius of zoom region

    # Quiver
    quiver_style:      str   = "scaled"       # "scaled" | "uniform"
    quiver_density:    int   = 20             # grid step in pixels
    show_quiver:       bool  = False   # legacy — quiver now controlled via field selector

    # Playback
    default_speed:     float = 1.0

    # Save defaults
    save_every_n:      int   = 1

    COLORMAPS: List[str] = field(default_factory=lambda: [
        "jet", "plasma", "RdBu", "viridis", "inferno",
        "coolwarm", "seismic", "bwr", "turbo"
    ])

    def save(self):
        d = asdict(self)
        d.pop("COLORMAPS", None)
        with open(_SETTINGS_PATH, "w") as f:
            json.dump(d, f, indent=2)

    @classmethod
    def load(cls):
        if not os.path.exists(_SETTINGS_PATH):
            return cls()
        try:
            with open(_SETTINGS_PATH) as f:
                d = json.load(f)
            s = cls()
            for k, v in d.items():
                if hasattr(s, k):
                    setattr(s, k, v)
            return s
        except Exception:
            return cls()


# ── Settings panel widget ─────────────────────────────────────────────────────

class SettingsPanel(QDialog):
    """
    Floating settings dialog.
    Emits `changed` when any setting is modified.
    """
    changed = pyqtSignal(object)   # emits updated DICSettings

    def __init__(self, settings: DICSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("Settings")
        self.setMinimumWidth(380)
        self.setStyleSheet(f"""
            QDialog, QWidget {{
                background: {C_BG}; color: {C_TEXT}; font-family: Courier;
            }}
            QGroupBox {{
                border: 1px solid {C_BORDER}; border-radius: 4px;
                margin-top: 10px; padding-top: 8px;
                font-size: 9px; color: {C_ACCENT};
            }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 8px; }}
            QLabel  {{ font-size: 10px; }}
            QComboBox, QSpinBox, QDoubleSpinBox {{
                background: #21262D; border: 1px solid {C_BORDER};
                color: {C_TEXT}; padding: 4px; border-radius: 3px;
                font-family: Courier; font-size: 10px;
            }}
            QSlider::groove:horizontal {{
                background: #21262D; height: 4px; border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {C_ACCENT}; width: 12px; height: 12px;
                margin: -4px 0; border-radius: 6px;
            }}
            QSlider::sub-page:horizontal {{ background: {C_ACCENT}; border-radius: 2px; }}
            QCheckBox {{ font-size: 10px; spacing: 6px; }}
            QCheckBox::indicator {{
                width: 13px; height: 13px;
                border: 1px solid {C_BORDER}; border-radius: 3px;
                background: #21262D;
            }}
            QCheckBox::indicator:checked {{
                background: {C_ACCENT}; border-color: {C_ACCENT};
            }}
            QPushButton {{
                background: #21262D; color: {C_TEXT};
                border: 1px solid {C_BORDER}; border-radius: 4px;
                padding: 6px 12px; font-family: Courier;
                font-weight: bold; font-size: 10px;
            }}
            QPushButton:hover {{ background: #30363D; }}
            QPushButton#primary {{
                background: {C_ACCENT}; color: {C_BG}; border: none;
            }}
        """)
        self._build()

    def _build(self):
        s = self.settings
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(16, 16, 16, 16)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        inner = QWidget()
        il    = QVBoxLayout(inner)
        il.setSpacing(8)
        scroll.setWidget(inner)
        layout.addWidget(scroll)

        def row(label_text, widget):
            r = QWidget()
            h = QHBoxLayout(r)
            h.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel(label_text)
            lbl.setMinimumWidth(160)
            h.addWidget(lbl)
            h.addWidget(widget, stretch=1)
            return r

        def sep():
            f = QFrame(); f.setFrameShape(QFrame.HLine)
            f.setStyleSheet(f"color: {C_BORDER};")
            return f

        # ── DISPLAY ──────────────────────────────────────────────────────────
        grp = QGroupBox("DISPLAY")
        gl  = QVBoxLayout(grp)

        self._cmap = QComboBox()
        for cm in s.COLORMAPS:
            self._cmap.addItem(cm)
        self._cmap.setCurrentText(s.colormap)
        self._cmap.currentTextChanged.connect(self._on_cmap)
        gl.addWidget(row("Colormap", self._cmap))

        self._alpha_lbl = QLabel(f"{int(s.overlay_alpha*100)}%")
        self._alpha     = QSlider(Qt.Horizontal)
        self._alpha.setRange(10, 100)
        self._alpha.setValue(int(s.overlay_alpha * 100))
        self._alpha.valueChanged.connect(self._on_alpha)
        alpha_row = QWidget(); ah = QHBoxLayout(alpha_row)
        ah.setContentsMargins(0,0,0,0)
        ah.addWidget(QLabel("Overlay opacity"))
        ah.addWidget(self._alpha, stretch=1)
        ah.addWidget(self._alpha_lbl)
        gl.addWidget(alpha_row)

        self._norm = QComboBox()
        self._norm.addItems(["percentile", "global_max"])
        self._norm.setCurrentText(s.norm_mode)
        self._norm.currentTextChanged.connect(self._on_norm)
        gl.addWidget(row("Colorbar normalization", self._norm))

        self._pct = QDoubleSpinBox()
        self._pct.setRange(50.0, 100.0)
        self._pct.setSingleStep(0.5)
        self._pct.setValue(s.norm_percentile)
        self._pct.setSuffix(" %")
        self._pct.valueChanged.connect(self._on_pct)
        gl.addWidget(row("  Percentile", self._pct))

        il.addWidget(grp)

        # ── HOVER ZOOM ───────────────────────────────────────────────────────
        grp2 = QGroupBox("HOVER ZOOM")
        gl2  = QVBoxLayout(grp2)

        self._hover = QComboBox()
        self._hover.addItems(["tooltip", "popup", "panel"])
        self._hover.setCurrentText(s.hover_zoom_mode)
        self._hover.currentTextChanged.connect(self._on_hover)
        gl2.addWidget(row("Mode", self._hover))

        self._zoom_sz = QSpinBox()
        self._zoom_sz.setRange(60, 400)
        self._zoom_sz.setSingleStep(20)
        self._zoom_sz.setValue(s.hover_zoom_size)
        self._zoom_sz.setSuffix(" px")
        self._zoom_sz.valueChanged.connect(self._on_zoom_sz)
        gl2.addWidget(row("Zoom region radius", self._zoom_sz))

        il.addWidget(grp2)

        # ── QUIVER ───────────────────────────────────────────────────────────
        grp3 = QGroupBox("VELOCITY QUIVER")
        gl3  = QVBoxLayout(grp3)

        self._qstyle = QComboBox()
        self._qstyle.addItems(["scaled", "uniform"])
        self._qstyle.setCurrentText(s.quiver_style)
        self._qstyle.currentTextChanged.connect(self._on_qstyle)
        gl3.addWidget(row("Arrow style", self._qstyle))

        self._qdensity = QSpinBox()
        self._qdensity.setRange(5, 80)
        self._qdensity.setSingleStep(5)
        self._qdensity.setValue(s.quiver_density)
        self._qdensity.setSuffix(" px")
        self._qdensity.valueChanged.connect(self._on_qdensity)
        gl3.addWidget(row("Arrow grid spacing", self._qdensity))

        il.addWidget(grp3)

        # ── PLAYBACK ─────────────────────────────────────────────────────────
        grp4 = QGroupBox("PLAYBACK")
        gl4  = QVBoxLayout(grp4)

        self._speed = QComboBox()
        for spd in ["0.25", "0.5", "1.0", "2.0", "4.0"]:
            self._speed.addItem(f"{spd}×", float(spd))
        self._speed.setCurrentIndex(
            [0.25, 0.5, 1.0, 2.0, 4.0].index(s.default_speed)
            if s.default_speed in [0.25, 0.5, 1.0, 2.0, 4.0] else 2
        )
        self._speed.currentIndexChanged.connect(self._on_speed)
        gl4.addWidget(row("Default speed", self._speed))

        il.addWidget(grp4)

        # ── SAVE DEFAULTS ─────────────────────────────────────────────────────
        grp5 = QGroupBox("SAVE DEFAULTS")
        gl5  = QVBoxLayout(grp5)

        self._save_n = QSpinBox()
        self._save_n.setRange(1, 100)
        self._save_n.setValue(s.save_every_n)
        self._save_n.setSuffix("  (every N frames)")
        self._save_n.valueChanged.connect(self._on_save_n)
        gl5.addWidget(row("Save every N", self._save_n))

        il.addWidget(grp5)

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_reset = QPushButton("Reset to defaults")
        btn_reset.clicked.connect(self._reset)
        btn_save  = QPushButton("Save & Close")
        btn_save.setObjectName("primary")
        btn_save.clicked.connect(self._save_close)
        btn_row.addWidget(btn_reset)
        btn_row.addStretch()
        btn_row.addWidget(btn_save)
        layout.addLayout(btn_row)

    # ── Handlers ─────────────────────────────────────────────────────────────

    def _emit(self):
        self.changed.emit(self.settings)

    def _on_cmap(self, v):
        self.settings.colormap = v; self._emit()

    def _on_alpha(self, v):
        self.settings.overlay_alpha = v / 100.0
        self._alpha_lbl.setText(f"{v}%")
        self._emit()

    def _on_norm(self, v):
        self.settings.norm_mode = v; self._emit()

    def _on_pct(self, v):
        self.settings.norm_percentile = v; self._emit()

    def _on_hover(self, v):
        self.settings.hover_zoom_mode = v; self._emit()

    def _on_zoom_sz(self, v):
        self.settings.hover_zoom_size = v; self._emit()

    def _on_qshow(self, v):
        self.settings.show_quiver = bool(v); self._emit()

    def _on_qstyle(self, v):
        self.settings.quiver_style = v; self._emit()

    def _on_qdensity(self, v):
        self.settings.quiver_density = v; self._emit()

    def _on_speed(self, i):
        self.settings.default_speed = self._speed.itemData(i); self._emit()

    def _on_save_n(self, v):
        self.settings.save_every_n = v; self._emit()

    def _reset(self):
        self.settings = DICSettings()
        self.close()
        self._emit()

    def _save_close(self):
        self.settings.save()
        self.accept()