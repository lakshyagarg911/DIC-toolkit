# dicUtils/dense.py
#
# Dense DIC pipeline — full featured:
#   PlotSelectorDialog, ComputeThread, DICPlayerWindow
#   + hover zoom, quiver panel, settings, save/export

import cv2
import numpy as np
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as mcm
import matplotlib.colors as mcolors

from dicUtils.io import _load_gray
from dicUtils.settings import DICSettings, SettingsPanel
from dicUtils.storage import save_results, export_to_csv, export_to_npy
from dicUtils.stream  import StreamingDICStore, make_temp_h5, WINDOW_SIZE
from dicUtils.dic_runner import DICComputeThread, DIC_FIELD_CATALOGUE, scatter_to_grid
import h5py
import json

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QDialog,
    QHBoxLayout, QVBoxLayout, QGridLayout,
    QLabel, QPushButton, QCheckBox, QSlider,
    QFrame, QSizePolicy, QProgressBar, QGroupBox,
    QScrollArea, QFileDialog, QInputDialog,
    QToolTip, QMenu, QAction, QSpinBox
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QPoint, QRect, QSize
from PyQt5.QtGui import (
    QImage, QPixmap, QPainter, QColor, QPen, QBrush,
    QFont, QPainterPath, QCursor
)
from PyQt5.QtCore import QPointF, QRectF

# ── Safe dialog runner — avoids nested event loops on Windows ─────────────────

def _exec_dialog(dlg):
    """
    Run a QDialog modally without calling exec_() which causes nested
    event loops and stack overflows on Windows (0xC0000409).
    Returns QDialog.Accepted or QDialog.Rejected.
    """
    from PyQt5.QtCore import QEventLoop
    loop = QEventLoop()
    dlg.finished.connect(loop.quit)
    dlg.show()
    loop.exec_()
    return dlg.result()




# ── Farneback defaults ────────────────────────────────────────────────────────

_FB_DEFAULTS = dict(
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
)

# ── Field catalogue ───────────────────────────────────────────────────────────

FIELD_CATALOGUE = {
    "vx":     ("Velocity Vx",        "pixels/s",     True),
    "vy":     ("Velocity Vy",        "pixels/s",     True),
    "u":      ("Displacement u",     "pixels/frame", False),
    "v":      ("Displacement v",     "pixels/frame", False),
    "mag":    ("Magnitude |d|",      "pixels/frame", False),
    "exx":    ("Strain ε_xx",        "px/px",        False),
    "eyy":    ("Strain ε_yy",        "px/px",        False),
    "exy":    ("Strain ε_xy",        "px/px",        False),
    "edxx":   ("Strain rate ε̇_xx",  "px/px/s",      True),
    "edyy":   ("Strain rate ε̇_yy",  "px/px/s",      True),
    "edxy":   ("Strain rate ε̇_xy",  "px/px/s",      True),
    "quiver": ("Velocity Quiver",    "Vx,Vy arrows", False),
    # ── True DIC fields (subset-based ZNCC + Newton-Raphson) ──────────────────
    "dic_u":    ("DIC Displacement u",   "pixels",  False),
    "dic_v":    ("DIC Displacement v",   "pixels",  False),
    "dic_exx":  ("DIC Strain e_xx",      "px/px",   False),
    "dic_eyy":  ("DIC Strain e_yy",      "px/px",   False),
    "dic_exy":  ("DIC Strain e_xy",      "px/px",   False),
    "dic_zncc": ("DIC Correlation ZNCC", "[-1,1]",  False),
    "dic_ux":   ("DIC du/dx",            "px/px",   False),
    "dic_uy":   ("DIC du/dy",            "px/px",   False),
    "dic_vx":   ("DIC dv/dx",            "px/px",   False),
    "dic_vy":   ("DIC dv/dy",            "px/px",   False),
}

# Keys that are virtual (derived from other fields, not stored in HDF5)
VIRTUAL_KEYS  = {"quiver"}
DIC_KEYS      = {k for k in FIELD_CATALOGUE if k.startswith('dic_')}
# Quiver needs these base fields to be computed
QUIVER_DEPS  = {"vx", "vy"}

# ── Palette ───────────────────────────────────────────────────────────────────

C_BG    = "#0D1117"; C_PANEL = "#161B22"; C_BORDER = "#30363D"
C_TEXT  = "#E6EDF3"; C_MUTED = "#8B949E"
C_ACCENT= "#00E5FF"; C_WARN  = "#FF6B35"

STYLE = f"""
    QWidget, QDialog, QMainWindow {{
        background:{C_BG}; color:{C_TEXT}; font-family:Courier;
    }}
    QPushButton {{
        background:#21262D; color:{C_TEXT}; border:1px solid {C_BORDER};
        border-radius:4px; padding:7px 14px; font-family:Courier;
        font-weight:bold; font-size:12px;
    }}
    QPushButton:hover  {{ background:#30363D; }}
    QPushButton:pressed {{ background:{C_ACCENT}; color:{C_BG}; }}
    QPushButton#primary {{
        background:{C_ACCENT}; color:{C_BG}; border:none;
        font-size:13px; font-weight:bold; letter-spacing:1px;
    }}
    QPushButton#primary:hover {{ background:#33ECFF; border:2px solid white; }}
    QCheckBox {{ color:{C_TEXT}; font-family:Courier; font-size:10px; spacing:8px; }}
    QCheckBox::indicator {{
        width:14px; height:14px; border:1px solid {C_BORDER};
        border-radius:3px; background:#21262D;
    }}
    QCheckBox::indicator:checked {{ background:{C_ACCENT}; border-color:{C_ACCENT}; }}
    QSlider::groove:horizontal {{
        background:#21262D; height:4px; border-radius:2px;
    }}
    QSlider::handle:horizontal {{
        background:{C_ACCENT}; width:14px; height:14px;
        margin:-5px 0; border-radius:7px;
    }}
    QSlider::sub-page:horizontal {{ background:{C_ACCENT}; border-radius:2px; }}
    QProgressBar {{
        background:#21262D; border:1px solid {C_BORDER};
        border-radius:3px; text-align:center; color:{C_TEXT}; font-size:10px;
    }}
    QProgressBar::chunk {{ background:{C_ACCENT}; border-radius:3px; }}
    QGroupBox {{
        border:1px solid {C_BORDER}; border-radius:4px; margin-top:10px;
        padding-top:8px; font-family:Courier; font-size:9px; color:{C_ACCENT};
    }}
    QGroupBox::title {{ subcontrol-origin:margin; left:8px; }}
    QScrollArea {{ border:none; background:{C_BG}; }}
    QMenu {{ background:#21262D; color:{C_TEXT}; border:1px solid {C_BORDER}; }}
    QMenu::item:selected {{ background:{C_ACCENT}; color:{C_BG}; }}
"""


# ── Field computation ─────────────────────────────────────────────────────────

def _compute_fields(flow_u, flow_v, frame_step, fps):
    u   = flow_u / frame_step
    v   = flow_v / frame_step
    vx  = u * fps;  vy = v * fps
    mag = np.sqrt(u**2 + v**2)
    du_dy, du_dx = np.gradient(u)
    dv_dy, dv_dx = np.gradient(v)
    exx = du_dx;  eyy = dv_dy;  exy = 0.5*(du_dy+dv_dx)
    return {"vx":vx,"vy":vy,"u":u,"v":v,"mag":mag,
            "exx":exx,"eyy":eyy,"exy":exy,
            "edxx":exx*fps,"edyy":eyy*fps,"edxy":exy*fps}


def _normalize_field(field, vmax, settings: DICSettings):
    if settings.norm_mode == "percentile":
        p = float(np.percentile(np.abs(field), settings.norm_percentile))
        vmax_use = max(p, vmax * 0.05, 1e-9)
    else:
        vmax_use = max(vmax, 1e-9)
    norm = mcolors.Normalize(vmin=-vmax_use, vmax=vmax_use)
    cm   = mcm.get_cmap(settings.colormap)
    rgba = (cm(norm(field)) * 255).astype(np.uint8)
    return rgba, vmax_use


def _make_colorbar_pixmap(vmax, unit, height, settings: DICSettings, width=24):
    vals   = np.linspace(1, 0, height)
    cm     = mcm.get_cmap(settings.colormap)
    colors = (cm(vals) * 255).astype(np.uint8)
    label_w = 80
    result  = QPixmap(width + label_w, height)
    result.fill(QColor(C_BG))
    painter = QPainter(result)
    # Draw bar
    for y in range(height):
        r,g,b = int(colors[y,0]),int(colors[y,1]),int(colors[y,2])
        painter.fillRect(0, y, width, 1, QColor(r,g,b))
    # Ticks
    painter.setPen(QPen(QColor(C_BORDER), 1))
    for frac in [0.0, 0.5, 1.0]:
        ty = int(frac*(height-1))
        painter.drawLine(width, ty, width+5, ty)
    # Labels
    fmt = (lambda v: f"{v:.2e}") if (vmax<0.01 or vmax>999) else (lambda v: f"{v:.3f}")
    painter.setPen(QColor(C_TEXT))
    painter.setFont(QFont("Courier", 9))
    painter.drawText(width+7, 12,            f"+{fmt(vmax)}")
    painter.drawText(width+7, height//2+5,   "0")
    painter.drawText(width+7, height-3,      f"-{fmt(vmax)}")
    # Unit
    painter.setPen(QColor(C_MUTED))
    painter.setFont(QFont("Courier", 8))
    painter.save()
    painter.translate(width+label_w-2, height)
    painter.rotate(-90)
    painter.drawText(4, 0, unit)
    painter.restore()
    painter.end()
    return result


# ── Hover zoom tooltip ────────────────────────────────────────────────────────

class HoverZoomPopup(QWidget):
    """Floating zoom window that follows cursor over a FieldPanel."""
    def __init__(self, parent=None):
        super().__init__(parent, Qt.ToolTip | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(220, 220)
        self._pm   = None
        self._val  = None
        self._title= ""

    def update_content(self, pm, val, title):
        self._pm    = pm
        self._val   = val
        self._title = title
        self.update()

    def paintEvent(self, event):
        if self._pm is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        # Background
        p.setBrush(QBrush(QColor(22, 27, 34, 230)))
        p.setPen(QPen(QColor(C_ACCENT), 1.5))
        p.drawRoundedRect(self.rect().adjusted(2,2,-2,-2), 8, 8)
        # Zoomed image
        p.drawPixmap(8, 28, 204, 180, self._pm)
        # Value label
        p.setPen(QColor(C_ACCENT))
        p.setFont(QFont("Courier", 9, QFont.Bold))
        val_str = f"{self._val:.4f}" if self._val is not None else "—"
        p.drawText(QRect(8, 6, 204, 20), Qt.AlignCenter,
                   f"{self._title}  =  {val_str}")
        p.end()


# ── Field panel widget ────────────────────────────────────────────────────────

class FieldPanel(QWidget):
    hover_info = pyqtSignal(float, float, float, str)  # ix, iy, value, title

    def __init__(self, key, title, unit, settings: DICSettings, parent=None):
        super().__init__(parent)
        self.key      = key
        self.title    = title
        self.unit     = unit
        self.settings = settings
        self._bg_pm   = None
        self._ov_pm   = None
        self._cb_pm   = None
        self._poly    = None
        self._field   = None   # raw field array for value lookup
        self._vmax    = 1.0
        self._sc      = 1.0
        self._ox = self._oy = 0
        self._img_w = self._img_h = 1
        self._zoom_popup = None
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(280, 200)
        self.setMouseTracking(True)

    def set_frame(self, bg_np, field, polygon, mask, bbox, vmax):
        H, W   = bg_np.shape
        x,y,w,h= bbox
        self._field   = field
        self._vmax    = vmax
        self._img_w   = W
        self._img_h   = H

        # Background
        rgb  = cv2.cvtColor(bg_np, cv2.COLOR_GRAY2RGB)
        qi   = QImage(rgb.data, W, H, 3*W, QImage.Format_RGB888)
        self._bg_pm = QPixmap.fromImage(qi.copy())

        # Overlay
        rgba, vmax_use = _normalize_field(field, vmax, self.settings)
        self._vmax = vmax_use
        overlay_np = np.zeros((H, W, 4), dtype=np.uint8)
        pm = (mask[y:y+h, x:x+w] == 255)
        rgba[~pm] = [0,0,0,0]
        overlay_np[y:y+h, x:x+w] = rgba
        alpha = int(self.settings.overlay_alpha * 255)
        overlay_np[:,:,3] = np.where(overlay_np[:,:,3]>0, alpha, 0)
        qi2  = QImage(overlay_np.data, W, H, 4*W, QImage.Format_RGBA8888)
        self._ov_pm = QPixmap.fromImage(qi2.copy())

        self._poly = polygon
        self._bbox = bbox
        self._mask = mask

        cb_h = max(120, self.height()-40)
        self._cb_pm = _make_colorbar_pixmap(vmax_use, self.unit, cb_h, self.settings)
        self.update()

    def _canvas_to_image(self, cx, cy):
        ix = (cx - self._ox) / self._sc
        iy = (cy - self._oy) / self._sc
        return ix, iy

    def paintEvent(self, event):
        if self._bg_pm is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)

        W = self.width(); H = self.height()
        cb_w   = self._cb_pm.width() if self._cb_pm else 104
        title_h= 32
        img_w  = W - cb_w - 8
        img_h  = H - title_h

        bw = self._bg_pm.width(); bh = self._bg_pm.height()
        sc = min(img_w/bw, img_h/bh)
        dw = int(bw*sc);  dh = int(bh*sc)
        ox = (img_w-dw)//2
        oy = title_h + (img_h-dh)//2
        self._sc = sc; self._ox = ox; self._oy = oy

        # Title
        p.fillRect(0, 0, W, title_h, QColor(C_PANEL))
        p.setPen(QColor(C_ACCENT))
        p.setFont(QFont("Courier", 11, QFont.Bold))
        p.drawText(10, title_h-9, self.title)

        # Image + overlay
        p.drawPixmap(ox, oy, dw, dh, self._bg_pm)
        p.drawPixmap(ox, oy, dw, dh, self._ov_pm)

        # Polygon border
        if self._poly is not None:
            p.setPen(QPen(QColor(255,255,255,160), 1.5, Qt.DashLine))
            pts = [QPointF(pt[0]*sc+ox, pt[1]*sc+oy) for pt in self._poly]
            for i in range(len(pts)):
                p.drawLine(pts[i], pts[(i+1)%len(pts)])

        # Colorbar
        if self._cb_pm:
            cb_x = W - cb_w + 2
            cb_y = title_h + (img_h - self._cb_pm.height())//2
            p.drawPixmap(cb_x, max(title_h, cb_y), self._cb_pm)

        p.end()

    def mouseMoveEvent(self, event):
        if self._field is None:
            return
        cx, cy = event.x(), event.y()
        ix, iy = self._canvas_to_image(cx, cy)
        xi, yi = int(round(ix)), int(round(iy))
        x, y, w, h = self._bbox
        fi_x = xi - x
        fi_y = yi - y
        val = None
        if 0 <= fi_x < w and 0 <= fi_y < h:
            val = float(self._field[fi_y, fi_x])

        self.hover_info.emit(ix, iy, val if val is not None else float('nan'), self.title)

        mode = self.settings.hover_zoom_mode
        if mode in ("tooltip", "popup") and self._bg_pm is not None:
            self._show_zoom(cx, cy, ix, iy, val, mode)
        else:
            if self._zoom_popup:
                self._zoom_popup.hide()

    def _show_zoom(self, cx, cy, ix, iy, val, mode):
        sz = self.settings.hover_zoom_size
        # Crop region around cursor in image space
        img_x1 = max(0, int(ix)-sz//2)
        img_y1 = max(0, int(iy)-sz//2)
        img_x2 = min(self._img_w, img_x1+sz)
        img_y2 = min(self._img_h, img_y1+sz)

        # Composite bg+overlay at full res then crop
        bg_full = self._bg_pm.copy()
        painter = QPainter(bg_full)
        painter.drawPixmap(0, 0, self._ov_pm)
        painter.end()

        crop = bg_full.copy(img_x1, img_y1, img_x2-img_x1, img_y2-img_y1)
        zoom_pm = crop.scaled(204, 180, Qt.KeepAspectRatio,
                               Qt.FastTransformation)

        if self._zoom_popup is None:
            self._zoom_popup = HoverZoomPopup()

        self._zoom_popup.update_content(zoom_pm, val, self.title)

        if mode == "tooltip":
            gp = self.mapToGlobal(QPoint(cx+20, cy-120))
        else:
            gp = self.mapToGlobal(QPoint(cx+24, cy+8))

        self._zoom_popup.move(gp)
        self._zoom_popup.show()

    def leaveEvent(self, event):
        if self._zoom_popup:
            self._zoom_popup.hide()

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        menu.setStyleSheet(STYLE)
        a1 = menu.addAction("Save field as .npy")
        a2 = menu.addAction("Copy value at cursor")
        act = menu.exec_(event.globalPos())
        if act == a1 and self._field is not None:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save field", f"{self.key}.npy", "NumPy (*.npy)")
            if path:
                np.save(path, self._field)
        if act == a2 and self._field is not None:
            cx, cy = event.x(), event.y()
            ix, iy = self._canvas_to_image(cx, cy)
            x,y,w,h = self._bbox
            fi_x, fi_y = int(ix)-x, int(iy)-y
            if 0<=fi_x<w and 0<=fi_y<h:
                val = float(self._field[fi_y, fi_x])
                QApplication.clipboard().setText(f"{val:.6f}")


# ── Quiver panel ──────────────────────────────────────────────────────────────

class QuiverPanel(QWidget):
    """Velocity vector field overlaid on frame background."""

    def __init__(self, settings: DICSettings, parent=None):
        super().__init__(parent)
        self.settings  = settings
        self._bg_pm    = None
        self._vx       = None
        self._vy       = None
        self._bbox     = None
        self._poly     = None
        self._mask     = None
        self._vmax     = 1.0
        self._img_w    = 1
        self._img_h    = 1
        # Canvas transform — updated in paintEvent
        self._sc       = 1.0
        self._ox       = 0
        self._oy       = 0
        self._title_h  = 32
        self._zoom_popup = None
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(280, 200)
        self.setMouseTracking(True)

    def set_frame(self, bg_np, vx, vy, polygon, mask, bbox, vmax):
        H, W = bg_np.shape
        self._img_h, self._img_w = H, W
        rgb = cv2.cvtColor(bg_np, cv2.COLOR_GRAY2RGB)
        qi  = QImage(rgb.data, W, H, 3*W, QImage.Format_RGB888)
        self._bg_pm = QPixmap.fromImage(qi.copy())
        self._vx    = vx
        self._vy    = vy
        self._bbox  = bbox
        self._poly  = polygon
        self._mask  = mask
        self._vmax  = max(vmax, 1e-9)
        self.update()

    def _canvas_to_image(self, cx, cy):
        """Convert canvas pixel coords to image pixel coords."""
        ix = (cx - self._ox) / self._sc
        iy = (cy - self._oy) / self._sc
        return ix, iy

    def mouseMoveEvent(self, event):
        if self._vx is None or self._bg_pm is None:
            return
        cx, cy = event.x(), event.y()
        ix, iy = self._canvas_to_image(cx, cy)
        xi, yi = int(round(ix)), int(round(iy))
        x, y, w, h = self._bbox

        # Get values in ROI-local coords
        fi_x = xi - x
        fi_y = yi - y

        vx_val = vy_val = mag_val = None
        if 0 <= fi_x < w and 0 <= fi_y < h:
            vx_val  = float(self._vx[fi_y, fi_x])
            vy_val  = float(self._vy[fi_y, fi_x])
            mag_val = math.hypot(vx_val, vy_val)

        mode = self.settings.hover_zoom_mode
        if mode in ("tooltip", "popup"):
            self._show_quiver_zoom(cx, cy, ix, iy, vx_val, vy_val, mag_val, mode)
        else:
            if self._zoom_popup:
                self._zoom_popup.hide()

    def _show_quiver_zoom(self, cx, cy, ix, iy, vx_val, vy_val, mag_val, mode):
        sz = self.settings.hover_zoom_size
        # Crop background around cursor
        img_x1 = max(0, int(ix) - sz//2)
        img_y1 = max(0, int(iy) - sz//2)
        img_x2 = min(self._img_w, img_x1 + sz)
        img_y2 = min(self._img_h, img_y1 + sz)

        crop = self._bg_pm.copy(img_x1, img_y1,
                                img_x2-img_x1, img_y2-img_y1)
        zoom_pm = crop.scaled(244, 120, Qt.KeepAspectRatio,
                              Qt.FastTransformation)

        if self._zoom_popup is None:
            self._zoom_popup = QuiverHoverPopup()

        self._zoom_popup.update_content(zoom_pm, vx_val, vy_val, mag_val)

        if mode == "tooltip":
            gp = self.mapToGlobal(QPoint(cx + 20, cy - 110))
        else:
            gp = self.mapToGlobal(QPoint(cx + 24, cy + 8))

        self._zoom_popup.move(gp)
        self._zoom_popup.show()

    def leaveEvent(self, event):
        if self._zoom_popup:
            self._zoom_popup.hide()

    def paintEvent(self, event):
        if self._bg_pm is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        W = self.width(); H = self.height()
        title_h = 32

        bw = self._bg_pm.width(); bh = self._bg_pm.height()
        sc = min((W-8)/bw, (H-title_h)/bh)
        dw = int(bw*sc);  dh = int(bh*sc)
        ox = (W-dw)//2;   oy = title_h + (H-title_h-dh)//2
        # Store transform for mouse coordinate conversion
        self._sc = sc; self._ox = ox; self._oy = oy

        # Title
        p.fillRect(0, 0, W, title_h, QColor(C_PANEL))
        p.setPen(QColor(C_ACCENT))
        p.setFont(QFont("Courier", 11, QFont.Bold))
        p.drawText(10, title_h-9, "Velocity Field (Vx, Vy)")

        # Background (semi-transparent)
        alpha_img = QPixmap(dw, dh)
        alpha_img.fill(Qt.transparent)
        bp = QPainter(alpha_img)
        bp.setOpacity(self.settings.overlay_alpha * 0.5)
        bp.drawPixmap(0, 0, dw, dh, self._bg_pm)
        bp.end()
        p.drawPixmap(ox, oy, alpha_img)

        # Polygon mask outline
        if self._poly is not None:
            p.setPen(QPen(QColor(255,255,255,120), 1, Qt.DashLine))
            pts = [QPointF(pt[0]*sc+ox, pt[1]*sc+oy) for pt in self._poly]
            for i in range(len(pts)):
                p.drawLine(pts[i], pts[(i+1)%len(pts)])

        # Draw arrows
        if self._vx is None:
            p.end(); return

        x, y, w, h = self._bbox
        step = max(4, self.settings.quiver_density)
        cm   = mcm.get_cmap(self.settings.colormap)
        vmax = self._vmax

        for iy in range(0, h, step):
            for ix in range(0, w, step):
                if self._mask[y+iy, x+ix] == 0:
                    continue
                vx_val = float(self._vx[iy, ix])
                vy_val = float(self._vy[iy, ix])
                mag    = math.hypot(vx_val, vy_val)

                # Arrow base in canvas coords
                base_x = (x + ix) * sc + ox
                base_y = (y + iy) * sc + oy

                if self.settings.quiver_style == "scaled":
                    arrow_len = (mag / vmax) * step * sc * 1.8
                else:
                    arrow_len = step * sc * 0.9

                if mag < 1e-9:
                    continue

                # Direction
                dx = (vx_val / mag) * arrow_len
                dy = (vy_val / mag) * arrow_len

                # Color by magnitude
                norm_mag = min(mag / vmax, 1.0)
                rgba_c   = cm(norm_mag)
                color    = QColor(int(rgba_c[0]*255), int(rgba_c[1]*255),
                                  int(rgba_c[2]*255), 220)

                tip_x = base_x + dx
                tip_y = base_y + dy

                pen = QPen(color, 1.2)
                p.setPen(pen)
                p.drawLine(QPointF(base_x, base_y), QPointF(tip_x, tip_y))

                # Arrowhead
                head_len = max(3.0, arrow_len * 0.3)
                angle    = math.atan2(dy, dx)
                for side in [0.4, -0.4]:
                    hx = tip_x - head_len * math.cos(angle - side)
                    hy = tip_y - head_len * math.sin(angle - side)
                    p.drawLine(QPointF(tip_x, tip_y), QPointF(hx, hy))

        p.end()


# ── Quiver hover popup ────────────────────────────────────────────────────────

class QuiverHoverPopup(QWidget):
    """
    Floating tooltip for QuiverPanel — shows zoomed quiver arrows
    in a local neighbourhood plus Vx, Vy, magnitude values at cursor.
    """
    def __init__(self, parent=None):
        super().__init__(parent, Qt.ToolTip | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(260, 200)
        self._pm   = None   # zoomed background pixmap
        self._vx   = None
        self._vy   = None
        self._mag  = None

    def update_content(self, pm, vx_val, vy_val, mag_val):
        self._pm   = pm
        self._vx   = vx_val
        self._vy   = vy_val
        self._mag  = mag_val
        self.update()

    def paintEvent(self, event):
        if self._pm is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        W, H = self.width(), self.height()

        # Background panel
        p.setBrush(QBrush(QColor(22, 27, 34, 235)))
        p.setPen(QPen(QColor(C_WARN), 1.5))   # orange border for quiver
        p.drawRoundedRect(self.rect().adjusted(2,2,-2,-2), 8, 8)

        # Title
        p.setPen(QColor(C_WARN))
        p.setFont(QFont("Courier", 9, QFont.Bold))
        p.drawText(QRect(8, 5, W-16, 18), Qt.AlignCenter, "Velocity Quiver")

        # Zoomed image region
        img_rect_h = 120
        p.drawPixmap(8, 26, W-16, img_rect_h, self._pm)

        # Draw a single large arrow in centre of zoomed image showing direction
        if self._vx is not None and self._mag is not None and self._mag > 1e-9:
            cx = W // 2
            cy = 26 + img_rect_h // 2
            arrow_len = 40.0
            dx = (self._vx / self._mag) * arrow_len
            dy = (self._vy / self._mag) * arrow_len

            # Colour by magnitude (use orange→red for quiver popup)
            p.setPen(QPen(QColor(C_WARN), 2.5))
            p.drawLine(QPointF(cx - dx*0.5, cy - dy*0.5),
                       QPointF(cx + dx*0.5, cy + dy*0.5))
            # Arrowhead
            angle = math.atan2(dy, dx)
            for side in [0.5, -0.5]:
                hx = cx + dx*0.5 - 12 * math.cos(angle - side)
                hy = cy + dy*0.5 - 12 * math.sin(angle - side)
                p.drawLine(QPointF(cx + dx*0.5, cy + dy*0.5),
                           QPointF(hx, hy))

        # Value readout
        p.setFont(QFont("Courier", 8))
        y0 = 26 + img_rect_h + 6
        row_h = 16
        for label, val, col in [
            ("Vx",  self._vx,  C_ACCENT),
            ("Vy",  self._vy,  C_ACCENT),
            ("|V|", self._mag, C_WARN),
        ]:
            p.setPen(QColor(C_MUTED))
            p.drawText(QRect(10, y0, 30, row_h), Qt.AlignLeft, f"{label}:")
            p.setPen(QColor(col))
            val_str = f"{val:.4f}" if val is not None else "—"
            p.drawText(QRect(42, y0, W-50, row_h), Qt.AlignLeft, val_str)
            y0 += row_h

        p.end()


# ── Plot selector ─────────────────────────────────────────────────────────────

class PlotSelectorDialog(QDialog):
    def __init__(self, parent=None, already_computed=None, already_selected=None):
        """
        already_computed : set of keys already in the HDF5 store (greyed out, pre-checked)
        already_selected : set of keys currently displayed (pre-checked)
        """
        super().__init__(parent)
        self.setWindowTitle("Select Fields")
        self.setStyleSheet(STYLE)
        self._checks = {}
        self._already_computed = already_computed or set()
        self._already_selected = already_selected or set()
        self._build()

    def _build(self):
        # ── Size to 80% of screen, landscape ─────────────────────────────────
        screen = QApplication.primaryScreen().availableGeometry()
        w      = int(screen.width()  * 0.80)
        h      = int(screen.height() * 0.72)
        self.resize(w, h)
        self.move((screen.width()  - w) // 2,
                  (screen.height() - h) // 2)

        is_adding  = bool(self._already_computed)
        outer      = QVBoxLayout(self)
        outer.setSpacing(10)
        outer.setContentsMargins(18, 16, 18, 16)

        # ── Title row ─────────────────────────────────────────────────────────
        title_row = QHBoxLayout()
        title_text = "ADD / REMOVE FIELDS" if is_adding else "SELECT FIELDS TO COMPUTE"
        title = QLabel(title_text)
        title.setStyleSheet(f"color:{C_ACCENT}; font-size:13px; font-weight:bold;")
        title_row.addWidget(title)

        if is_adding:
            sub = QLabel(
                f"  —  {len(self._already_computed)} field(s) already computed  |  "
                "Cyan = instant  |  White = will compute")
            sub.setStyleSheet(f"color:{C_MUTED}; font-size:10px;")
            title_row.addWidget(sub)
        title_row.addStretch()
        outer.addLayout(title_row)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color:{C_BORDER};"); outer.addWidget(sep)

        # ── Groups arranged in a horizontal grid ──────────────────────────────
        # 3 columns: [Optical Flow cols] | [DIC cols]
        groups = [
            ("VELOCITY",           ["vx","vy","quiver"]),
            ("DISPLACEMENT",       ["u","v","mag"]),
            ("STRAIN",             ["exx","eyy","exy"]),
            ("STRAIN RATE",        ["edxx","edyy","edxy"]),
            ("TRUE DIC (ZNCC+NR)", ["dic_u","dic_v","dic_exx","dic_eyy",
                                    "dic_exy","dic_zncc"]),
            ("DIC GRADIENTS",      ["dic_ux","dic_uy","dic_vx","dic_vy"]),
        ]

        # Lay groups into a grid: 3 per row
        cols      = 3
        grid      = QGridLayout()
        grid.setSpacing(10)
        grid.setHorizontalSpacing(16)

        for idx, (gname, keys) in enumerate(groups):
            box = QGroupBox(gname)
            box.setStyleSheet(f"""
                QGroupBox {{
                    border: 1px solid {C_BORDER}; border-radius:4px;
                    margin-top:10px; padding-top:8px;
                    font-family:Courier; font-size:9px; color:{C_ACCENT};
                }}
                QGroupBox::title {{ subcontrol-origin:margin; left:8px; }}
            """)
            gl = QVBoxLayout(box)
            gl.setSpacing(5)
            gl.setContentsMargins(10, 8, 10, 8)

            for key in keys:
                name, unit, default = FIELD_CATALOGUE[key]
                already  = key in self._already_computed
                selected = key in self._already_selected

                if already:
                    cb = QCheckBox(f"{name}  [{unit}]  [computed]")
                    cb.setChecked(True)
                    cb.setStyleSheet(f"color:{C_ACCENT}; font-size:10px;")
                    cb.setToolTip("Already computed — toggle to show/hide")
                else:
                    cb = QCheckBox(f"{name}  [{unit}]")
                    cb.setChecked(default or selected)
                    cb.setStyleSheet(f"color:{C_TEXT}; font-size:10px;")
                    cb.setToolTip("Not yet computed — will run on confirm")

                self._checks[key] = cb
                gl.addWidget(cb)

            gl.addStretch()
            r, c = divmod(idx, cols)
            grid.addWidget(box, r, c)

        for c in range(cols):
            grid.setColumnStretch(c, 1)

        outer.addLayout(grid, stretch=1)

        # ── Bottom action row ─────────────────────────────────────────────────
        sep2 = QFrame(); sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet(f"color:{C_BORDER};"); outer.addWidget(sep2)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        ba = QPushButton("Select All")
        bn = QPushButton("Clear All")
        ba.setFixedWidth(110)
        bn.setFixedWidth(110)
        ba.clicked.connect(lambda: [c.setChecked(True)
                                    for c in self._checks.values()])
        bn.clicked.connect(lambda: [c.setChecked(False)
                                    for c in self._checks.values()])
        btn_row.addWidget(ba)
        btn_row.addWidget(bn)
        btn_row.addStretch()

        ok_label = "APPLY CHANGES" if is_adding else "COMPUTE & DISPLAY"
        ok = QPushButton(ok_label)
        ok.setObjectName("primary")
        ok.setFixedHeight(40)
        ok.setMinimumWidth(200)
        ok.clicked.connect(self._confirm)
        btn_row.addWidget(ok)

        outer.addLayout(btn_row)

    def _confirm(self):
        sel = [k for k, cb in self._checks.items() if cb.isChecked()]
        if not sel:
            return
        self.selected_keys = sel
        # Keys needing fresh computation
        self.new_keys = [
            k for k in sel
            if k not in self._already_computed and k not in VIRTUAL_KEYS
        ]
        # Summary print so user can see in console what will happen
        if self.new_keys:
            print(f"[Add Fields] Will compute: {self.new_keys}")
        instant = [k for k in sel if k in self._already_computed or k in VIRTUAL_KEYS]
        if instant:
            print(f"[Add Fields] Instant (already in store): {instant}")
        self.accept()

    def get_selected(self):
        return getattr(self, "selected_keys", [])

    def get_new_keys(self):
        """Keys that need to be computed (not already in HDF5)."""
        return getattr(self, "new_keys", [])


# ── Compute thread ────────────────────────────────────────────────────────────

class ComputeThread(QThread):
    progress = pyqtSignal(int, int)
    done     = pyqtSignal(object)   # emits StreamingDICStore
    error    = pyqtSignal(str)

    def __init__(self, frame_paths, polygon, mask, bbox,
                 frame_step, fps, fb_params, selected_keys, h5_path,
                 display_keys=None):
        super().__init__()
        self.frame_paths   = frame_paths
        self.polygon       = polygon
        self.mask          = mask
        self.bbox          = bbox
        self.frame_step    = frame_step
        self.fps           = fps
        self.fb            = {**_FB_DEFAULTS, **(fb_params or {})}
        self.selected_keys = selected_keys
        self.display_keys  = display_keys if display_keys is not None else list(selected_keys)
        self.h5_path       = h5_path

    def run(self):
        try:
            x,y,w,h = self.bbox
            paths    = self.frame_paths
            total    = (len(paths)-1)//self.frame_step

            store = StreamingDICStore(
                self.h5_path, self.selected_keys,
                self.polygon, self.mask, self.bbox,
                self.frame_step, self.fps,
                display_keys=self.display_keys,
            )

            prev_np = _load_gray(paths[0])

            for i, idx in enumerate(range(self.frame_step, len(paths), self.frame_step)):
                curr_np = _load_gray(paths[idx])
                flow = cv2.calcOpticalFlowFarneback(
                    prev_np, curr_np, None,
                    self.fb["pyr_scale"], self.fb["levels"], self.fb["winsize"],
                    self.fb["iterations"], self.fb["poly_n"],
                    self.fb["poly_sigma"], self.fb["flags"],
                )
                fu = flow[y:y+h, x:x+w, 0]
                fv = flow[y:y+h, x:x+w, 1]
                fields = _compute_fields(fu, fv, self.frame_step, self.fps)
                fields = {k: fields[k] for k in self.selected_keys if k in fields}
                # Write to HDF5 immediately — never accumulates in RAM
                store.append(idx, curr_np, fields)
                self.progress.emit(i+1, total)
                prev_np = curr_np

            store.finalize()
            self.done.emit(store)
        except Exception as e:
            import traceback; traceback.print_exc()
            self.error.emit(str(e))


# ── Progress dialog ───────────────────────────────────────────────────────────

class ProgressDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Computing DIC Fields")
        self.setStyleSheet(STYLE)
        self.setFixedSize(480, 150)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)
        self._t_start = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(8)

        self.label = QLabel("Running Farneback optical flow...")
        self.label.setStyleSheet(f"color:{C_TEXT}; font-size:11px;")
        layout.addWidget(self.label)

        self.bar = QProgressBar()
        self.bar.setMinimumHeight(22)
        self.bar.setTextVisible(False)
        layout.addWidget(self.bar)

        # Bottom row: frame counter (left) + ETA (right)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self.frame_label = QLabel("")
        self.frame_label.setStyleSheet(
            f"color:{C_ACCENT}; font-size:12px; font-weight:600; font-family:monospace;")
        self.eta_label = QLabel("")
        self.eta_label.setStyleSheet(f"color:{C_MUTED}; font-size:10px;")
        self.eta_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(self.frame_label)
        row.addStretch()
        row.addWidget(self.eta_label)
        layout.addLayout(row)

    def update_progress(self, cur, tot):
        import time
        if self._t_start is None:
            self._t_start = time.time()

        self.bar.setMaximum(tot)
        self.bar.setValue(cur)

        pct = int(cur / tot * 100) if tot > 0 else 0
        self.label.setText(f"Computing fields…  {pct}%")
        self.frame_label.setText(f"Frame {cur} / {tot}")

        # ETA estimate
        elapsed = time.time() - self._t_start
        if cur > 0:
            remaining = elapsed / cur * (tot - cur)
            mins, secs = divmod(int(remaining), 60)
            if mins > 0:
                eta_str = f"~{mins}m {secs:02d}s remaining"
            else:
                eta_str = f"~{secs}s remaining"
            self.eta_label.setText(eta_str)
        else:
            self.eta_label.setText("")


# ── Save dialog ───────────────────────────────────────────────────────────────

class SaveDialog(QDialog):
    def __init__(self, total_frames, settings: DICSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save / Export DIC Data")
        self.setStyleSheet(STYLE)
        self.setMinimumWidth(440)
        self._total = total_frames
        self._settings = settings
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(20,20,20,20)

        title = QLabel("SAVE / EXPORT")
        title.setStyleSheet(f"color:{C_ACCENT}; font-size:12px; font-weight:bold;")
        layout.addWidget(title)

        # Save group
        sg = QGroupBox("SAVE  (HDF5 — efficient, reloadable)")
        sl = QVBoxLayout(sg); sl.setSpacing(6)

        # Path
        pr = QWidget(); ph = QHBoxLayout(pr); ph.setContentsMargins(0,0,0,0)
        self._save_path = __import__('PyQt5.QtWidgets', fromlist=['QLineEdit']).QLineEdit()
        self._save_path.setPlaceholderText("Output .h5 path...")
        self._save_path.setText("dic_results.h5")
        browse = QPushButton("Browse"); browse.setFixedWidth(80)
        browse.clicked.connect(self._browse_save)
        ph.addWidget(self._save_path, stretch=1); ph.addWidget(browse)
        sl.addWidget(pr)

        # Mode
        mr = QWidget(); mh = QHBoxLayout(mr); mh.setContentsMargins(0,0,0,0)
        from PyQt5.QtWidgets import QRadioButton, QButtonGroup
        self._mode_all  = QRadioButton("All frames")
        self._mode_n    = QRadioButton("Every N frames")
        self._mode_custom= QRadioButton("Custom frame list")
        self._mode_all.setChecked(True)
        self._mode_group = QButtonGroup()
        for b in [self._mode_all, self._mode_n, self._mode_custom]:
            self._mode_group.addButton(b); mh.addWidget(b)
        sl.addWidget(mr)

        nr = QWidget(); nh = QHBoxLayout(nr); nh.setContentsMargins(0,0,0,0)
        nh.addWidget(QLabel("N ="))
        self._n_spin = QSpinBox(); self._n_spin.setRange(1,100)
        self._n_spin.setValue(self._settings.save_every_n)
        nh.addWidget(self._n_spin)
        nh.addSpacing(16)
        nh.addWidget(QLabel("Custom (comma-separated frame indices):"))
        from PyQt5.QtWidgets import QLineEdit
        self._custom_frames = QLineEdit()
        self._custom_frames.setPlaceholderText("e.g. 10,20,50,100")
        nh.addWidget(self._custom_frames, stretch=1)
        sl.addWidget(nr)

        save_btn = QPushButton("SAVE")
        save_btn.setObjectName("primary"); save_btn.setMinimumHeight(36)
        save_btn.clicked.connect(lambda: self.done_action("save"))
        sl.addWidget(save_btn)
        layout.addWidget(sg)

        # Export group
        eg = QGroupBox("EXPORT  (human-readable, full pixel-level data)")
        el = QVBoxLayout(eg); el.setSpacing(6)

        from PyQt5.QtWidgets import QRadioButton, QButtonGroup, QLineEdit

        # Export frame selection — same modes as save
        ex_label = QLabel("Frame selection:")
        ex_label.setStyleSheet(f"color:{C_MUTED}; font-size:9px;")
        el.addWidget(ex_label)

        emr = QWidget(); emh = QHBoxLayout(emr); emh.setContentsMargins(0,0,0,0)
        self._ex_mode_all    = QRadioButton("All frames")
        self._ex_mode_n      = QRadioButton("Every N frames")
        self._ex_mode_custom = QRadioButton("Custom frame list")
        self._ex_mode_all.setChecked(True)
        self._ex_mode_group  = QButtonGroup()
        for b in [self._ex_mode_all, self._ex_mode_n, self._ex_mode_custom]:
            self._ex_mode_group.addButton(b); emh.addWidget(b)
        el.addWidget(emr)

        enr = QWidget(); enh = QHBoxLayout(enr); enh.setContentsMargins(0,0,0,0)
        enh.addWidget(QLabel("N ="))
        self._ex_n_spin = QSpinBox(); self._ex_n_spin.setRange(1, 100)
        self._ex_n_spin.setValue(self._settings.save_every_n)
        enh.addWidget(self._ex_n_spin)
        enh.addSpacing(16)
        enh.addWidget(QLabel("Custom (comma-separated):"))
        self._ex_custom_frames = QLineEdit()
        self._ex_custom_frames.setPlaceholderText("e.g. 10,20,50,100")
        enh.addWidget(self._ex_custom_frames, stretch=1)
        el.addWidget(enr)

        er = QWidget(); eh = QHBoxLayout(er); eh.setContentsMargins(0,0,0,0)
        btn_csv = QPushButton("Export CSV (per pixel)")
        btn_npy = QPushButton("Export npy arrays")
        btn_csv.clicked.connect(lambda: self.done_action("csv"))
        btn_npy.clicked.connect(lambda: self.done_action("npy"))
        eh.addWidget(btn_csv); eh.addWidget(btn_npy)
        el.addWidget(er)
        layout.addWidget(eg)

        layout.addWidget(QPushButton("Cancel", clicked=self.reject))

        self._action = None

    def _browse_save(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save DIC data", "dic_results.h5",
                                           "HDF5 (*.h5 *.hdf5)")
        if p:
            self._save_path.setText(p)

    def done_action(self, action):
        self._action = action
        self.accept()

    def get_params(self):
        path = self._save_path.text().strip()

        # Save frame selection
        if self._mode_n.isChecked():
            mode = "every_n"; n = self._n_spin.value(); custom = None
        elif self._mode_custom.isChecked():
            mode = "custom"; n = 1
            try:
                custom = [int(v.strip()) for v in
                          self._custom_frames.text().split(",") if v.strip()]
            except ValueError:
                custom = None
        else:
            mode = "all"; n = 1; custom = None

        # Export frame selection
        if self._ex_mode_n.isChecked():
            ex_mode = "every_n"; ex_n = self._ex_n_spin.value(); ex_custom = None
        elif self._ex_mode_custom.isChecked():
            ex_mode = "custom"; ex_n = 1
            try:
                ex_custom = [int(v.strip()) for v in
                             self._ex_custom_frames.text().split(",") if v.strip()]
            except ValueError:
                ex_custom = None
        else:
            ex_mode = "all"; ex_n = 1; ex_custom = None

        return self._action, path, mode, n, custom, ex_mode, ex_n, ex_custom


# ── DIC Player window ─────────────────────────────────────────────────────────

class DICPlayerWindow(QMainWindow):

    def __init__(self, store, selected_keys, polygon, mask, bbox,
                 fps, frame_step, settings: DICSettings,
                 source_info="", frame_paths=None):
        super().__init__()
        self.store         = store       # StreamingDICStore — reads from HDF5
        self.selected_keys = selected_keys
        self.polygon       = polygon
        self.mask          = mask
        self.bbox          = bbox
        self.fps           = fps
        self.frame_step    = frame_step
        self.settings      = settings
        self.source_info   = source_info
        self.frame_paths   = frame_paths or []
        self.current_idx   = 0
        self.playing       = False
        self._vmaxes       = store._vmaxes   # pre-computed during streaming
        self._panels       = {}
        self._quiver_panel = None
        self._settings_dlg = None
        self._hover_panel  = None

        self.setWindowTitle("Dense DIC Player")
        self.setStyleSheet(STYLE)
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(int(screen.width()*0.95), int(screen.height()*0.92))
        self.move(int(screen.width()*0.025), int(screen.height()*0.04))

        self._build_ui()
        self._show_frame(0)

        self.timer = QTimer()
        self.timer.timeout.connect(self._next_frame)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QVBoxLayout(central)
        main.setContentsMargins(6,6,6,6)
        main.setSpacing(4)

        # ── Top bar ───────────────────────────────────────────────────────────
        top = QWidget()
        top.setStyleSheet(f"background:{C_PANEL}; border-radius:4px;")
        top.setFixedHeight(40)
        tl = QHBoxLayout(top); tl.setContentsMargins(12,0,12,0)

        self.info_label = QLabel("Dense DIC")
        self.info_label.setStyleSheet(
            f"color:{C_ACCENT}; font-size:13px; font-weight:bold;")
        tl.addWidget(self.info_label)
        tl.addStretch()

        # Settings gear button
        btn_settings = QPushButton("Settings")
        btn_settings.setFixedWidth(110)
        btn_settings.clicked.connect(self._open_settings)
        tl.addWidget(btn_settings)

        # Add Fields button
        btn_add = QPushButton("Add Fields")
        btn_add.setFixedWidth(130)
        btn_add.setStyleSheet("""
            QPushButton {
                background: #1A2D1A; color: #4ADE80;
                border: 1px solid #4ADE80; border-radius:4px;
                font-family:Courier; font-weight:bold; font-size:11px;
                padding: 6px;
            }
            QPushButton:hover { background: #2D4A2D; }
        """)
        btn_add.clicked.connect(self._add_fields)
        tl.addWidget(btn_add)

        # Save button
        btn_save = QPushButton("Save / Export")
        btn_save.setFixedWidth(140)
        btn_save.clicked.connect(self._open_save_dialog)
        tl.addWidget(btn_save)

        main.addWidget(top)

        # ── Hover info bar ────────────────────────────────────────────────────
        self._hover_bar = QLabel("Hover over a panel to inspect values")
        self._hover_bar.setStyleSheet(
            f"color:{C_MUTED}; font-size:10px; padding:2px 8px;")
        main.addWidget(self._hover_bar)

        # ── Panel grid ────────────────────────────────────────────────────────
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._grid_widget = QWidget()
        self.grid = QGridLayout(self._grid_widget)
        self.grid.setSpacing(4)
        self.grid.setContentsMargins(0,0,0,0)
        self._scroll.setWidget(self._grid_widget)
        main.addWidget(self._scroll, stretch=1)

        self._rebuild_grid()

        # ── Controls ──────────────────────────────────────────────────────────
        ctrl = QWidget()
        ctrl.setStyleSheet(f"background:{C_PANEL}; border-radius:4px;")
        cl = QVBoxLayout(ctrl); cl.setContentsMargins(10,6,10,6); cl.setSpacing(4)

        # Slider row
        sr = QHBoxLayout()
        self.frame_label = QLabel("Frame 0000")
        self.frame_label.setStyleSheet(
            f"color:{C_TEXT}; font-size:13px; font-weight:bold; min-width:110px;")
        sr.addWidget(self.frame_label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(0, len(self.store)-1))
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self._on_slider)
        sr.addWidget(self.slider, stretch=1)

        self.time_label = QLabel("t = 0.000 s")
        self.time_label.setStyleSheet(
            f"color:{C_MUTED}; font-size:13px; min-width:110px; "
            "qproperty-alignment:AlignRight;")
        sr.addWidget(self.time_label)
        cl.addLayout(sr)

        # Button row
        br = QHBoxLayout(); br.setSpacing(6)

        def mb(label, slot, primary=False, w=None):
            b = QPushButton(label)
            if primary: b.setObjectName("primary")
            if w: b.setFixedWidth(w)
            b.clicked.connect(slot)
            br.addWidget(b); return b

        mb("⏮",        self._go_first,   w=42)
        mb("◄◄ -10",   self._go_back10,  w=80)
        mb("◄ Prev",   self._go_prev,    w=80)
        self.play_btn = QPushButton("▶  PLAY")
        self.play_btn.setFixedWidth(140)
        self.play_btn.setFixedHeight(38)
        self.play_btn.clicked.connect(self._toggle_play)
        self.play_btn.setStyleSheet("""
            QPushButton {
                background: #00E5FF;
                color: #0D1117;
                border: none;
                border-radius: 6px;
                font-family: Courier;
                font-size: 14px;
                font-weight: bold;
                letter-spacing: 2px;
            }
            QPushButton:hover {
                background: #33ECFF;
                border: 2px solid white;
            }
            QPushButton:pressed { background: #009BB5; }
        """)
        br.addWidget(self.play_btn)
        mb("Next ►",   self._go_next,    w=80)
        mb("+10 ►►",   self._go_fwd10,   w=80)
        mb("⏭",        self._go_last,    w=42)

        br.addStretch()
        br.addWidget(QLabel("Speed:"))
        for lbl, spd in [("¼×",0.25),("½×",0.5),("1×",1.0),("2×",2.0),("4×",4.0)]:
            b = QPushButton(lbl); b.setFixedWidth(44)
            b.clicked.connect(lambda _,s=spd: self._set_speed(s))
            br.addWidget(b)

        cl.addLayout(br)
        main.addWidget(ctrl)

        self.setFocusPolicy(Qt.StrongFocus)

    def _rebuild_grid(self):
        # Clear existing
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
        self._panels = {}
        self._quiver_panel = None

        # Quiver: only show if explicitly selected AND vx/vy are available
        has_velocity = "vx" in self.selected_keys and "vy" in self.selected_keys
        show_quiver  = "quiver" in self.selected_keys and has_velocity

        # Real field keys (exclude virtual quiver key from panel list)
        field_keys = [k for k in self.selected_keys if k not in VIRTUAL_KEYS]

        n    = len(field_keys) + (1 if show_quiver else 0)
        cols = 1 if n==1 else 2 if n<=4 else 3
        rows = math.ceil(max(n, 1) / cols)

        for i, key in enumerate(field_keys):
            name, unit, _ = FIELD_CATALOGUE[key]
            panel = FieldPanel(key, name, unit, self.settings)
            panel.hover_info.connect(self._on_hover_info)
            self._panels[key] = panel
            r, c = divmod(i, cols)
            self.grid.addWidget(panel, r, c)

        if show_quiver:
            qp = QuiverPanel(self.settings)
            self._quiver_panel = qp
            idx = len(field_keys)
            r, c = divmod(idx, cols)
            self.grid.addWidget(qp, r, c)

        for c in range(cols): self.grid.setColumnStretch(c, 1)
        for r in range(rows): self.grid.setRowStretch(r, 1)

    def _on_hover_info(self, ix, iy, val, title):
        if math.isnan(val):
            self._hover_bar.setText(
                f"Cursor: x={int(ix)}  y={int(iy)}  |  outside ROI")
        else:
            self._hover_bar.setText(
                f"Cursor: x={int(ix)}  y={int(iy)}  |  {title} = {val:.5f}")

    def _show_frame(self, idx):
        total = len(self.store)
        idx   = max(0, min(idx, total-1))
        self.current_idx = idx

        # Load from streaming store (reads HDF5 window if not cached)
        frame_num, bg_np, fields = self.store.get_frame(idx)

        t = frame_num / self.fps
        last_frame = self.store.frame_index_at(total-1)

        self.info_label.setText(
            f"Dense DIC   —   Frame {frame_num:04d} / {last_frame}"
            f"   t = {t:.3f} s   |   {len(self.selected_keys)} fields"
            f"   [stream {idx+1}/{total}]")
        self.frame_label.setText(f"Frame {frame_num:04d}")
        self.time_label.setText(f"t = {t:.3f} s")

        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)

        for key, panel in self._panels.items():
            if key not in fields: continue
            panel.set_frame(bg_np, fields[key], self.polygon,
                            self.mask, self.bbox, self._vmaxes.get(key, 1.0))

        if self._quiver_panel and "vx" in fields and "vy" in fields:
            vmax_q = self._vmaxes.get("_quiver_mag", 1.0)
            self._quiver_panel.set_frame(
                bg_np, fields["vx"], fields["vy"],
                self.polygon, self.mask, self.bbox, vmax_q)

    # ── Playback ──────────────────────────────────────────────────────────────

    def _go_first(self):  self._show_frame(0)
    def _go_last(self):   self._show_frame(len(self.store)-1)
    def _go_prev(self):   self._show_frame(self.current_idx-1)
    def _go_next(self):   self._show_frame(self.current_idx+1)
    def _go_back10(self): self._show_frame(self.current_idx-10)
    def _go_fwd10(self):  self._show_frame(self.current_idx+10)
    def _on_slider(self, val): self._show_frame(val)

    def _toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self.play_btn.setText("⏸  PAUSE")
            self.play_btn.setStyleSheet("""
                QPushButton {
                    background: #FF6B35; color: white;
                    border: none; border-radius: 6px;
                    font-family: Courier; font-size: 14px;
                    font-weight: bold; letter-spacing: 2px;
                }
                QPushButton:hover { background: #FF8C5A; border: 2px solid white; }
            """)
            interval = max(1, int(1000/(self.fps/self.frame_step *
                                        self.settings.default_speed)))
            self.timer.start(interval)
        else:
            self.play_btn.setText("▶  PLAY")
            self.play_btn.setStyleSheet("""
                QPushButton {
                    background: #00E5FF; color: #0D1117;
                    border: none; border-radius: 6px;
                    font-family: Courier; font-size: 14px;
                    font-weight: bold; letter-spacing: 2px;
                }
                QPushButton:hover { background: #33ECFF; border: 2px solid white; }
                QPushButton:pressed { background: #009BB5; }
            """)
            self.timer.stop()

    def _next_frame(self):
        try:
            if self.current_idx >= len(self.store) - 1:
                self._toggle_play()
                return
            self._show_frame(self.current_idx + 1)
        except Exception as e:
            import traceback; traceback.print_exc()
            self._toggle_play()
            self._hover_bar.setText(f"ERROR - Playback: {e}")

    def _set_speed(self, m):
        self.settings.default_speed = m
        if self.playing:
            interval = max(1, int(1000/(self.fps/self.frame_step*m)))
            self.timer.setInterval(interval)

    def keyPressEvent(self, event):
        k = event.key()
        if k == Qt.Key_Right:   self._go_next()
        elif k == Qt.Key_Left:  self._go_prev()
        elif k == Qt.Key_Space: self._toggle_play()
        elif k == Qt.Key_Home:  self._go_first()
        elif k == Qt.Key_End:   self._go_last()

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0: self._go_prev()
        else: self._go_next()

    # ── Settings ──────────────────────────────────────────────────────────────

    def _add_fields(self):
        """Open field selector for uncomputed fields, compute them, merge into store."""
        if self.playing:
            self._toggle_play()

        # Keys physically in HDF5 (source of truth for what's computed)
        already_computed = set(self.store.selected_keys)
        # What's currently displayed — use store.display_keys if available
        already_selected = set(
            self.store.display_keys
            if hasattr(self.store, "display_keys")
            else self.selected_keys
        )


        selector = PlotSelectorDialog(
            already_computed=already_computed,
            already_selected=already_selected,
        )
        # Use direct show + local event loop — same pattern as ROI editor
        from PyQt5.QtCore import QEventLoop
        sel_loop = QEventLoop()
        selector.finished.connect(sel_loop.quit)
        selector.show()
        sel_loop.exec_()
        if selector.result() != QDialog.Accepted:
            return

        all_sel  = selector.get_selected()        # full desired display set
        new_keys = selector.get_new_keys()        # keys NOT yet in HDF5

        # Remove any virtual keys from new_keys (quiver is never in HDF5)
        new_keys = [k for k in new_keys if k not in VIRTUAL_KEYS]

        # Update display selection
        self.selected_keys = all_sel

        if not new_keys:
            # Nothing to compute — quiver and already-stored fields are instant
            self._rebuild_grid()
            app.processEvents()
            self._show_frame(self.current_idx)
            self._hover_bar.setText(
                f"Display updated — {len(all_sel)} field(s). No recompute needed.")
            return

        # Need to compute new fields — run a second pass over frame_paths
        if not self.frame_paths:
            self._hover_bar.setText("ERROR - Frame paths not available for recompute.")
            return

        self._hover_bar.setText(f"Computing {len(new_keys)} new field(s)...")
        app = QApplication.instance()

        progress_dlg = ProgressDialog(self)
        progress_dlg.label.setText(f"Computing: {', '.join(new_keys)}")
        progress_dlg.show()
        app.processEvents()

        thread   = AddFieldsThread(
            self.frame_paths, self.bbox,
            self.frame_step, self.fps,
            new_keys, self.store
        )
        vmaxes_accum = {k: 1e-9 for k in new_keys}

        # Main thread writes to HDF5 when frame data arrives from thread
        def on_frame_ready(frame_idx, new_data):
            name = f"frame_{frame_idx:06d}"
            try:
                with h5py.File(self.store.h5_path, "a") as f:
                    if name in f["frames"]:
                        fg = f["frames"][name]
                        for k, arr in new_data.items():
                            if k not in fg:
                                fg.create_dataset(k, data=arr,
                                    compression="gzip", compression_opts=4)
                        # Update vmaxes on the fly
                        for k, arr in new_data.items():
                            v = float(np.abs(arr).max())
                            vmaxes_accum[k] = max(vmaxes_accum[k], v)
            except Exception as e:
                print(f"[AddFields] HDF5 write error frame {frame_idx}: {e}")

        def on_progress(cur, tot):
            progress_dlg.update_progress(cur, tot)
            app.processEvents()

        def on_done():
            progress_dlg.hide()

            # Merge new physical keys into store.selected_keys
            merged_stored = list(dict.fromkeys(
                list(self.store.selected_keys) + new_keys))
            self.store.selected_keys = merged_stored

            # Merge full display selection (including virtuals) into display_keys
            self.store.display_keys = list(dict.fromkeys(
                list(self.store.display_keys) + list(self.selected_keys)))

            for k, v in vmaxes_accum.items():
                self.store._vmaxes[k] = v
            print(f"[AddFields] stored keys: {merged_stored}")
            print(f"[AddFields] display keys: {self.store.display_keys}")

            # Persist updated keys back to HDF5 metadata
            try:
                with h5py.File(self.store.h5_path, "a") as f:
                    f["metadata"].attrs["selected_keys"] = json.dumps(merged_stored)
                    f["metadata"].attrs["display_keys"]  = json.dumps(
                        self.store.display_keys)
            except Exception as e:
                print(f"[AddFields] metadata update warning: {e}")

            # Invalidate cache so new fields are read on next get_frame
            self.store._cache       = {}
            self.store._cache_start = -1
            self.store._cache_end   = -1

            self._rebuild_grid()
            self.slider.setMaximum(max(0, len(self.store) - 1))
            app.processEvents()
            self._show_frame(self.current_idx)
            self._hover_bar.setText(
                f"Added {len(new_keys)} field(s): {', '.join(new_keys)}")
        def on_error(msg):
            progress_dlg.hide()
            self._hover_bar.setText(f"ERROR - Compute: {msg}")
            print(f"[AddFields] Error: {msg}")

        thread.frame_ready.connect(on_frame_ready, Qt.QueuedConnection)
        thread.progress.connect(on_progress,    Qt.QueuedConnection)
        thread.done.connect(on_done,            Qt.QueuedConnection)
        thread.error.connect(on_error,          Qt.QueuedConnection)
        thread.start()

        from PyQt5.QtCore import QEventLoop
        add_loop = QEventLoop()
        thread.done.connect(add_loop.quit,  Qt.QueuedConnection)
        thread.error.connect(add_loop.quit, Qt.QueuedConnection)
        add_loop.exec_()
        thread.wait()

    def _open_settings(self):
        dlg = SettingsPanel(self.settings, self)
        dlg.changed.connect(self._on_settings_changed)
        from PyQt5.QtCore import QEventLoop
        loop = QEventLoop()
        dlg.finished.connect(loop.quit)
        dlg.show()
        loop.exec_()

    def _on_settings_changed(self, new_settings):
        self.settings = new_settings
        # Propagate to panels
        for panel in self._panels.values():
            panel.settings = new_settings
        if self._quiver_panel:
            self._quiver_panel.settings = new_settings
        # Rebuild grid if quiver toggle changed
        self._rebuild_grid()
        # Redraw current frame
        self._show_frame(self.current_idx)

    # ── Save / Export ─────────────────────────────────────────────────────────

    def _open_save_dialog(self):
        dlg = SaveDialog(len(self.store), self.settings, self)
        from PyQt5.QtCore import QEventLoop
        save_loop = QEventLoop()
        dlg.finished.connect(save_loop.quit)
        dlg.show()
        save_loop.exec_()
        if dlg.result() != QDialog.Accepted:
            return
        action, path, mode, n, custom, ex_mode, ex_n, ex_custom = dlg.get_params()

        action, path, mode, n, custom, ex_mode, ex_n, ex_custom = dlg.get_params()
        src_h5 = self.store.h5_path   # already on disk from streaming

        if action == "save":
            import shutil
            try:
                shutil.copy2(src_h5, path)
                self._hover_bar.setText(f"Saved to {path}")
            except Exception as e:
                self._hover_bar.setText(f"ERROR - Save failed: {e}")

        elif action == "csv":
            p = QFileDialog.getExistingDirectory(
                self, "Export CSV files to folder")
            if p:
                try:
                    export_to_csv(src_h5, p, self.selected_keys,
                                  export_mode=ex_mode, export_n=ex_n,
                                  export_custom=ex_custom)
                    self._hover_bar.setText(f"CSV exported to {p}/")
                except Exception as e:
                    self._hover_bar.setText(f"ERROR - Export failed: {e}")

        elif action == "npy":
            d = QFileDialog.getExistingDirectory(self, "Export npy to folder")
            if d:
                try:
                    export_to_npy(src_h5, d, self.selected_keys,
                                  export_mode=ex_mode, export_n=ex_n,
                                  export_custom=ex_custom)
                    self._hover_bar.setText(f"npy exported to {d}/")
                except Exception as e:
                    self._hover_bar.setText(f"ERROR - Export failed: {e}")




# ── Add-fields thread (second compute pass for new keys) ──────────────────────

class AddFieldsThread(QThread):
    """
    Re-runs Farneback and emits computed fields per frame via signal.
    The main thread writes to HDF5 — no HDF5 access from background thread.
    """
    progress    = pyqtSignal(int, int)
    frame_ready = pyqtSignal(int, dict)   # frame_idx, {key: array}
    done        = pyqtSignal()
    error       = pyqtSignal(str)

    def __init__(self, frame_paths, bbox, frame_step, fps, new_keys, store):
        super().__init__()
        self.frame_paths = frame_paths
        self.bbox        = bbox
        self.frame_step  = frame_step
        self.fps         = fps
        self.new_keys    = new_keys
        self.store       = store
        self.fb          = dict(_FB_DEFAULTS)

    def run(self):
        try:
            x, y, w, h = self.bbox
            paths       = self.frame_paths
            total       = (len(paths) - 1) // self.frame_step
            prev_np     = _load_gray(paths[0])

            for i, idx in enumerate(
                    range(self.frame_step, len(paths), self.frame_step)):

                curr_np = _load_gray(paths[idx])
                flow = cv2.calcOpticalFlowFarneback(
                    prev_np, curr_np, None,
                    self.fb["pyr_scale"], self.fb["levels"],
                    self.fb["winsize"], self.fb["iterations"],
                    self.fb["poly_n"], self.fb["poly_sigma"],
                    self.fb["flags"],
                )
                fu = flow[y:y+h, x:x+w, 0]
                fv = flow[y:y+h, x:x+w, 1]
                all_fields = _compute_fields(fu, fv, self.frame_step, self.fps)

                # Only emit requested new keys
                new_data = {k: all_fields[k].astype(np.float32)
                            for k in self.new_keys if k in all_fields}

                self.frame_ready.emit(idx, new_data)
                self.progress.emit(i + 1, total)
                prev_np = curr_np

            self.done.emit()

        except Exception as e:
            import traceback; traceback.print_exc()
            self.error.emit(str(e))

# ── Public pipeline entry point ───────────────────────────────────────────────

def run_dense_dic(frame_paths, polygon, mask, bbox, frame_step, fps,
                  fb_params=None, settings: DICSettings = None,
                  source_info=""):
    """
    Full pipeline: plot selector → streaming compute to HDF5 → player.
    Never holds more than WINDOW_SIZE frames in RAM.
    Does NOT manage the Qt event loop — caller is responsible for app.exec_().
    Returns the DICPlayerWindow (shown) or None if cancelled.
    """
    if settings is None:
        settings = DICSettings.load()

    app = QApplication.instance()
    if app is None:
        raise RuntimeError("QApplication must be created before calling run_dense_dic")

    # Plot selector
    selector = PlotSelectorDialog()
    from PyQt5.QtCore import QEventLoop

    # ── Field selector ────────────────────────────────────────────────────────
    sel_loop = QEventLoop()
    selector.finished.connect(sel_loop.quit)
    selector.show()
    sel_loop.exec_()
    if selector.result() != QDialog.Accepted:
        return None

    selected_keys     = selector.get_selected()
    dic_keys_selected = [k for k in selected_keys if k in DIC_KEYS]
    of_keys           = [k for k in selected_keys
                         if k not in VIRTUAL_KEYS and k not in DIC_KEYS]

    # Quiver needs vx+vy
    if "quiver" in selected_keys:
        for dep in QUIVER_DEPS:
            if dep not in of_keys:
                of_keys.append(dep)

    h5_path = make_temp_h5()

    # ── Step 1: Initialise HDF5 store ─────────────────────────────────────────
    # We always need the store structure (frames/bg) even if no OF fields
    # so the DIC pass has frame groups to write into.
    # If OF keys exist, ComputeThread creates the store.
    # If only DIC keys, we create a minimal store ourselves.

    store = None

    if of_keys:
        # ── Optical flow pass ─────────────────────────────────────────────────
        progress_dlg = ProgressDialog()
        progress_dlg.label.setText(
            f"Computing optical flow fields: {', '.join(of_keys)}")
        progress_dlg.show()
        app.processEvents()

        thread = ComputeThread(frame_paths, polygon, mask, bbox,
                               frame_step, fps, fb_params, of_keys,
                               h5_path, display_keys=selected_keys)
        store_holder = []

        def on_of_progress(cur, tot):
            progress_dlg.update_progress(cur, tot)
            app.processEvents()

        def on_of_done(s):
            store_holder.append(s)
            progress_dlg.hide()

        def on_of_error(msg):
            progress_dlg.hide()
            print(f"OF compute error: {msg}")

        of_loop = QEventLoop()
        thread.progress.connect(on_of_progress,  Qt.QueuedConnection)
        thread.done.connect(on_of_done,          Qt.QueuedConnection)
        thread.done.connect(of_loop.quit,        Qt.QueuedConnection)
        thread.error.connect(on_of_error,        Qt.QueuedConnection)
        thread.error.connect(of_loop.quit,       Qt.QueuedConnection)
        thread.start()
        of_loop.exec_()
        thread.wait()

        if not store_holder:
            return None
        store = store_holder[0]

    else:
        # ── DIC-only: create minimal store with bg frames ─────────────────────
        print("[DIC] No optical flow fields selected — creating frame store from images...")
        store = StreamingDICStore(
            h5_path, [], polygon, mask, bbox,
            frame_step, fps, display_keys=selected_keys)

        prev_np = _load_gray(frame_paths[0])
        total   = (len(frame_paths) - 1) // frame_step

        progress_dlg = ProgressDialog()
        progress_dlg.label.setText("Indexing frames...")
        progress_dlg.show()
        app.processEvents()

        for i, idx in enumerate(range(frame_step, len(frame_paths), frame_step)):
            curr_np = _load_gray(frame_paths[idx])
            store.append(idx, curr_np, {})   # bg only, no OF fields
            progress_dlg.update_progress(i + 1, total)
            app.processEvents()
            prev_np = curr_np

        store.finalize()
        progress_dlg.hide()
        print(f"[DIC] Frame store ready: {len(store)} frames")

    # ── Step 2: DIC pass (if dic_ fields selected) ────────────────────────────
    if dic_keys_selected:
        print(f"[DIC] Starting subset DIC: {dic_keys_selected}")

        dic_progress = ProgressDialog()
        dic_progress.label.setText(
            f"Subset DIC (ZNCC + Newton-Raphson): {', '.join(dic_keys_selected)}")
        dic_progress.show()
        app.processEvents()

        dic_thread = DICComputeThread.from_config(
            frame_paths, polygon, mask, bbox,
            frame_step, fps, dic_keys_selected,
            store.h5_path, display_keys=selected_keys,
            cfg=fb_params   # fb_params carries full config dict here
        )
        dic_vmaxes = {k: 1e-9 for k in dic_keys_selected}

        def on_dic_frame(frame_idx, dic_fields):
            with h5py.File(store.h5_path, "a") as f:
                name = f"frame_{frame_idx:06d}"
                if name in f["frames"]:
                    fg = f["frames"][name]
                    for k, arr in dic_fields.items():
                        if k not in fg:
                            fg.create_dataset(
                                k, data=arr.astype(np.float32),
                                compression="gzip", compression_opts=4)
                    for k, arr in dic_fields.items():
                        v = float(np.nanmax(np.abs(arr[~np.isnan(arr)])))                             if not np.all(np.isnan(arr)) else 1e-9
                        dic_vmaxes[k] = max(dic_vmaxes[k], v)

        def on_dic_progress(cur, tot):
            dic_progress.update_progress(cur, tot)
            app.processEvents()

        def on_dic_done():
            dic_progress.hide()
            store.selected_keys = list(dict.fromkeys(
                store.selected_keys + dic_keys_selected))
            store._vmaxes.update(dic_vmaxes)
            store._cache = {}
            store._cache_start = -1
            store._cache_end   = -1
            try:
                with h5py.File(store.h5_path, "a") as f:
                    f["metadata"].attrs["selected_keys"] = json.dumps(
                        store.selected_keys)
                    f["metadata"].attrs["display_keys"]  = json.dumps(
                        selected_keys)
            except Exception as e:
                print(f"[DIC] metadata write warning: {e}")
            print(f"[DIC] Done. vmaxes: "
                  f"{ {k: round(v, 4) for k, v in dic_vmaxes.items()} }")

        def on_dic_error(msg):
            dic_progress.hide()
            print(f"[DIC] Error: {msg}")

        dic_loop = QEventLoop()
        dic_thread.frame_ready.connect(on_dic_frame,  Qt.QueuedConnection)
        dic_thread.progress.connect(on_dic_progress,  Qt.QueuedConnection)
        dic_thread.done.connect(on_dic_done,          Qt.QueuedConnection)
        dic_thread.done.connect(dic_loop.quit,        Qt.QueuedConnection)
        dic_thread.error.connect(on_dic_error,        Qt.QueuedConnection)
        dic_thread.error.connect(dic_loop.quit,       Qt.QueuedConnection)
        dic_thread.start()
        dic_loop.exec_()
        dic_thread.wait()

    # ── Player ────────────────────────────────────────────────────────────────
    player = DICPlayerWindow(
        store, selected_keys, polygon, mask, bbox,
        fps, frame_step, settings,
        source_info=source_info, frame_paths=frame_paths
    )
    player.show()
    return player