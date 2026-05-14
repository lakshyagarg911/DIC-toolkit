# dicUtils/roi.py
#
# Professional polygon ROI editor using PyQt5.
#
# Controls:
#   Left click on canvas      — add vertex
#   Left click + drag vertex  — move vertex
#   Left click on edge        — insert vertex on that edge
#   Right click               — delete nearest vertex
#   Scroll wheel              — zoom toward cursor
#   Middle mouse drag         — pan
#   Enter / Confirm           — confirm
#   C / Clear                 — clear all
#   Z / Undo                  — remove last vertex
#   Escape                    — cancel

import cv2
import numpy as np
import math
import sys

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QFrame, QListWidget, QListWidgetItem,
    QSizePolicy
)
from PyQt5.QtCore import Qt, QPoint, QEventLoop, QRect, pyqtSignal, QSize
from PyQt5.QtGui import (
    QPainter, QColor, QPen, QBrush, QImage, QPixmap,
    QFont, QPainterPath, QPolygonF
)
from PyQt5.QtCore import QPointF


# ── Palette ───────────────────────────────────────────────────────────────────

C_BG        = QColor("#0D1117")
C_PANEL     = QColor("#161B22")
C_BORDER    = QColor("#30363D")
C_TEXT      = QColor("#E6EDF3")
C_MUTED     = QColor("#8B949E")
C_ACCENT    = QColor("#00E5FF")
C_SELECTED  = QColor("#FF6B35")
C_EDGE      = QColor("#00E5FF")
C_FILL      = QColor(0, 229, 255, 45)    # accent with low alpha
C_VERTEX_BG = QColor("#0D1117")

VERTEX_R     = 7
EDGE_HIT_D   = 10
VERTEX_HIT_D = 14


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _dist(ax, ay, bx, by):
    return math.hypot(ax-bx, ay-by)


def _dist_to_segment(px, py, ax, ay, bx, by):
    dx, dy = bx-ax, by-ay
    if dx == dy == 0:
        return math.hypot(px-ax, py-ay)
    t = max(0.0, min(1.0, ((px-ax)*dx + (py-ay)*dy) / (dx*dx+dy*dy)))
    return math.hypot(px-(ax+t*dx), py-(ay+t*dy)), t


# ── Canvas widget ─────────────────────────────────────────────────────────────

class PolygonCanvas(QWidget):

    changed = pyqtSignal()

    def __init__(self, image: np.ndarray):
        super().__init__()
        self.img_np      = image
        self.H, self.W   = image.shape
        self.points      = []        # list of [ix, iy] in IMAGE coords
        self.selected    = None      # index of selected vertex
        self.dragging    = False

        # View
        self.scale       = 1.0
        self.offset      = QPointF(0, 0)
        self._pan_last   = None

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self._build_pixmap()

    def _build_pixmap(self):
        rgb = cv2.cvtColor(self.img_np, cv2.COLOR_GRAY2RGB)
        h, w, ch = rgb.shape
        qi = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        self._base_pixmap = QPixmap.fromImage(qi)

    # ── Coord transforms ─────────────────────────────────────────────────────

    def _i2c(self, ix, iy):
        """Image → canvas."""
        return QPointF(ix * self.scale + self.offset.x(),
                       iy * self.scale + self.offset.y())

    def _c2i(self, cx, cy):
        """Canvas → image (clamped)."""
        ix = (cx - self.offset.x()) / self.scale
        iy = (cy - self.offset.y()) / self.scale
        return max(0.0, min(self.W-1, ix)), max(0.0, min(self.H-1, iy))

    def fit(self):
        """Fit image into widget with margin."""
        w, h = self.width(), self.height()
        if w < 2 or h < 2:
            return
        sx = w / self.W
        sy = h / self.H
        self.scale    = min(sx, sy) * 0.96
        self.offset   = QPointF(
            (w - self.W * self.scale) / 2,
            (h - self.H * self.scale) / 2,
        )
        self.update()

    # ── Hit testing ──────────────────────────────────────────────────────────

    def _hit_vertex(self, cx, cy, threshold=VERTEX_HIT_D):
        best_i, best_d = None, float("inf")
        for i, p in enumerate(self.points):
            cp = self._i2c(p[0], p[1])
            d  = _dist(cx, cy, cp.x(), cp.y())
            if d < threshold and d < best_d:
                best_d, best_i = d, i
        return best_i

    def _hit_edge(self, cx, cy):
        n = len(self.points)
        if n < 2:
            return None, None
        num_edges = n if n >= 3 else n-1
        best_i, best_d, best_t = None, float("inf"), 0
        for i in range(num_edges):
            a  = self._i2c(*self.points[i])
            b  = self._i2c(*self.points[(i+1) % n])
            dt = _dist_to_segment(cx, cy, a.x(), a.y(), b.x(), b.y())
            if isinstance(dt, tuple):
                d, t = dt
            else:
                d, t = dt, 0
            if d < EDGE_HIT_D and d < best_d:
                best_d, best_i, best_t = d, i, t
        return best_i, best_t

    # ── Paint ─────────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        p  = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), C_BG)

        # Image
        img_rect = QRect(
            int(self.offset.x()), int(self.offset.y()),
            int(self.W * self.scale), int(self.H * self.scale)
        )
        p.drawPixmap(img_rect, self._base_pixmap)

        n = len(self.points)
        if n == 0:
            p.end()
            return

        cpts = [self._i2c(pt[0], pt[1]) for pt in self.points]

        # Filled polygon
        if n >= 3:
            poly = QPolygonF(cpts + [cpts[0]])
            path = QPainterPath()
            path.addPolygon(poly)
            p.fillPath(path, QBrush(C_FILL))

        # Edges + midpoint hints
        pen = QPen(C_EDGE, 1.5, Qt.DashLine)
        p.setPen(pen)
        num_edges = n if n >= 3 else n-1
        for i in range(num_edges):
            a = cpts[i]
            b = cpts[(i+1) % n]
            p.drawLine(a, b)
            # midpoint dot
            mx, my = (a.x()+b.x())/2, (a.y()+b.y())/2
            p.setPen(QPen(C_EDGE, 1))
            p.setBrush(QBrush(C_PANEL))
            p.drawEllipse(QPointF(mx, my), 3.5, 3.5)
            p.setPen(pen)

        # Vertices
        for i, cp in enumerate(cpts):
            is_sel = (i == self.selected)
            color  = C_SELECTED if is_sel else C_ACCENT
            r      = VERTEX_R + 2 if is_sel else VERTEX_R

            # Glow ring
            glow_pen = QPen(QColor(color.red(), color.green(),
                                   color.blue(), 60), 6)
            p.setPen(glow_pen)
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(cp, float(r+4), float(r+4))

            # Vertex circle
            p.setPen(QPen(color, 2))
            p.setBrush(QBrush(C_VERTEX_BG))
            p.drawEllipse(cp, float(r), float(r))

            # Index label
            p.setPen(QPen(color))
            p.setFont(QFont("Courier", 7, QFont.Bold))
            p.drawText(QRect(int(cp.x())-8, int(cp.y())-8, 16, 16),
                       Qt.AlignCenter, str(i+1))

        p.end()

    # ── Mouse ─────────────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        cx, cy = event.x(), event.y()

        if event.button() == Qt.LeftButton:
            vi = self._hit_vertex(cx, cy)
            if vi is not None:
                self.selected = vi
                self.dragging = True
                self.update()
                self.changed.emit()
                return

            ei, t = self._hit_edge(cx, cy)
            if ei is not None:
                p1 = self.points[ei]
                p2 = self.points[(ei+1) % len(self.points)]
                nx = p1[0] + t*(p2[0]-p1[0])
                ny = p1[1] + t*(p2[1]-p1[1])
                self.points.insert(ei+1, [nx, ny])
                self.selected = ei+1
                self.dragging = True
                self.update()
                self.changed.emit()
                return

            ix, iy = self._c2i(cx, cy)
            self.points.append([ix, iy])
            self.selected = len(self.points)-1
            self.update()
            self.changed.emit()

        elif event.button() == Qt.RightButton:
            vi = self._hit_vertex(cx, cy, threshold=30)
            if vi is not None:
                self.points.pop(vi)
                if self.selected == vi:
                    self.selected = None
                elif self.selected and self.selected > vi:
                    self.selected -= 1
                self.update()
                self.changed.emit()

        elif event.button() == Qt.MiddleButton:
            self._pan_last = QPointF(cx, cy)

    def mouseMoveEvent(self, event):
        cx, cy = event.x(), event.y()
        if self.dragging and self.selected is not None and \
                event.buttons() & Qt.LeftButton:
            ix, iy = self._c2i(cx, cy)
            self.points[self.selected] = [ix, iy]
            self.update()
            self.changed.emit()

        elif self._pan_last is not None and event.buttons() & Qt.MiddleButton:
            dx = cx - self._pan_last.x()
            dy = cy - self._pan_last.y()
            self.offset += QPointF(dx, dy)
            self._pan_last = QPointF(cx, cy)
            self.update()

        # Cursor shape hint
        vi = self._hit_vertex(cx, cy)
        if vi is not None:
            self.setCursor(Qt.SizeAllCursor)
        else:
            ei, _ = self._hit_edge(cx, cy)
            if ei is not None:
                self.setCursor(Qt.CrossCursor)
            else:
                self.setCursor(Qt.CrossCursor)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False
        elif event.button() == Qt.MiddleButton:
            self._pan_last = None

    def wheelEvent(self, event):
        factor    = 1.12 if event.angleDelta().y() > 0 else 1/1.12
        cx, cy    = event.x(), event.y()
        new_scale = max(0.05, min(30.0, self.scale * factor))
        ratio     = new_scale / self.scale
        self.offset = QPointF(
            cx - ratio*(cx - self.offset.x()),
            cy - ratio*(cy - self.offset.y()),
        )
        self.scale = new_scale
        self.update()

    def resizeEvent(self, event):
        self.fit()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def undo(self):
        if self.points:
            self.points.pop()
            if self.selected is not None and self.selected >= len(self.points):
                self.selected = None
            self.update()
            self.changed.emit()

    def clear(self):
        self.points   = []
        self.selected = None
        self.update()
        self.changed.emit()


# ── Main window ───────────────────────────────────────────────────────────────

class ROIEditorWindow(QMainWindow):

    def __init__(self, image: np.ndarray):
        super().__init__()
        self.confirmed = False
        self.cancelled = False
        self._build_ui(image)

    def _build_ui(self, image):
        self.setWindowTitle("ROI Polygon Editor")
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background: #0D1117;
                color: #E6EDF3;
                font-family: Courier;
            }}
            QPushButton {{
                background: #21262D;
                color: #E6EDF3;
                border: 1px solid #30363D;
                border-radius: 4px;
                padding: 8px 12px;
                font-family: Courier;
                font-weight: bold;
                font-size: 11px;
            }}
            QPushButton:hover  {{ background: #30363D; }}
            QPushButton#confirm {{
                background: #00E5FF;
                color: #0D1117;
                border: none;
            }}
            QPushButton#confirm:hover {{ background: #33ECFF; }}
            QLabel {{ background: transparent; }}
            QListWidget {{
                background: #0D1117;
                border: 1px solid #30363D;
                color: #8B949E;
                font-family: Courier;
                font-size: 9px;
            }}
            QFrame#panel {{
                background: #161B22;
                border-left: 1px solid #30363D;
            }}
        """)

        # Screen size
        screen = QApplication.primaryScreen().availableGeometry()
        w = int(screen.width()  * 0.85)
        h = int(screen.height() * 0.85)
        self.resize(w, h)
        self.move((screen.width()-w)//2, (screen.height()-h)//2)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Canvas
        self.canvas = PolygonCanvas(image)
        self.canvas.changed.connect(self._on_changed)
        layout.addWidget(self.canvas, stretch=1)

        # Side panel
        panel = QFrame()
        panel.setObjectName("panel")
        panel.setFixedWidth(230)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(14, 14, 14, 14)
        panel_layout.setSpacing(6)
        layout.addWidget(panel)

        def section_label(text):
            lbl = QLabel(text)
            lbl.setStyleSheet("color: #00E5FF; font-size: 9px; font-weight: bold;")
            panel_layout.addWidget(lbl)
            sep = QFrame()
            sep.setFrameShape(QFrame.HLine)
            sep.setStyleSheet("color: #30363D;")
            panel_layout.addWidget(sep)

        def hint_label(icon, text):
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(6)
            il = QLabel(icon)
            il.setStyleSheet("color: #00E5FF; font-size: 9px; min-width: 20px;")
            tl = QLabel(text)
            tl.setStyleSheet("color: #8B949E; font-size: 8px;")
            tl.setWordWrap(True)
            h.addWidget(il)
            h.addWidget(tl, stretch=1)
            panel_layout.addWidget(w)

        section_label("CONTROLS")
        hint_label("LMB", "Click canvas → add vertex")
        hint_label("LMB", "Click vertex → select & drag")
        hint_label("LMB", "Click edge → insert vertex")
        hint_label("RMB", "Click → delete nearest vertex")
        hint_label("MMB", "Drag → pan view")
        hint_label("SCR", "Scroll → zoom toward cursor")

        panel_layout.addSpacing(8)
        section_label("VERTICES")

        self.count_label = QLabel("0 vertices")
        self.count_label.setStyleSheet(
            "color: #E6EDF3; font-size: 13px; font-weight: bold;")
        panel_layout.addWidget(self.count_label)

        self.coord_list = QListWidget()
        self.coord_list.setFixedHeight(160)
        panel_layout.addWidget(self.coord_list)

        panel_layout.addSpacing(8)
        section_label("ACTIONS")

        btn_confirm = QPushButton("✓  CONFIRM  [Enter]")
        btn_confirm.setObjectName("confirm")
        btn_confirm.clicked.connect(self._confirm)
        panel_layout.addWidget(btn_confirm)

        btn_undo = QPushButton("↩  UNDO LAST  [Z]")
        btn_undo.clicked.connect(self.canvas.undo)
        panel_layout.addWidget(btn_undo)

        btn_clear = QPushButton("✕  CLEAR  [C]")
        btn_clear.clicked.connect(self.canvas.clear)
        panel_layout.addWidget(btn_clear)

        btn_cancel = QPushButton("⊘  CANCEL  [Esc]")
        btn_cancel.clicked.connect(self._cancel)
        panel_layout.addWidget(btn_cancel)

        panel_layout.addStretch()

        self.status_label = QLabel("Click to place vertices")
        self.status_label.setStyleSheet(
            "color: #8B949E; font-size: 8px;")
        self.status_label.setWordWrap(True)
        panel_layout.addWidget(self.status_label)

        self.zoom_label = QLabel("Zoom: 100%")
        self.zoom_label.setStyleSheet("color: #30363D; font-size: 8px;")
        panel_layout.addWidget(self.zoom_label)

        # Keyboard shortcuts
        self.canvas.setFocus()

    def keyPressEvent(self, event):
        k = event.key()
        if k in (Qt.Key_Return, Qt.Key_Enter):
            self._confirm()
        elif k == Qt.Key_C:
            self.canvas.clear()
        elif k == Qt.Key_Z:
            self.canvas.undo()
        elif k == Qt.Key_Escape:
            self._cancel()

    def _on_changed(self):
        pts = self.canvas.points
        n   = len(pts)
        self.count_label.setText(f"{n} vertices")
        self.zoom_label.setText(f"Zoom: {int(self.canvas.scale*100)}%")

        self.coord_list.clear()
        for i, p in enumerate(pts):
            marker = "► " if i == self.canvas.selected else "   "
            item = QListWidgetItem(
                f"{marker}{i+1:2d}:  x={int(p[0]):4d}  y={int(p[1]):4d}")
            self.coord_list.addItem(item)
        if self.canvas.selected is not None:
            self.coord_list.setCurrentRow(self.canvas.selected)

        if n == 0:
            self.status_label.setText("Click to place vertices")
        elif n < 3:
            self.status_label.setText(f"{n} vertices — need at least 3")
        else:
            self.status_label.setText(f"{n} vertices — press Enter to confirm")

    def _confirm(self):
        if len(self.canvas.points) < 3:
            self.status_label.setText("⚠  Need at least 3 vertices!")
            return
        self.confirmed = True
        self.close()

    def _cancel(self):
        self.cancelled = True
        self.close()


# ── Public API ────────────────────────────────────────────────────────────────

def select_roi(image: np.ndarray):
    """
    Open the PyQt5 polygon ROI editor.

    Returns:
        polygon : (N, 2) int32 array of (x, y) vertices
        mask    : (H, W) uint8 binary mask — 255 inside polygon
        bbox    : (x, y, w, h) bounding box
    """
    # Use existing QApplication — never create a second one
    app = QApplication.instance()
    if app is None:
        raise RuntimeError("QApplication must exist before calling select_roi")

    win = ROIEditorWindow(image)
    win.setWindowModality(Qt.ApplicationModal)

    from PyQt5.QtCore import QEventLoop
    loop = QEventLoop()

    # Quit the local loop when window closes for any reason
    def _on_close(event):
        event.accept()
        loop.quit()
    win.closeEvent = _on_close

    win.show()
    loop.exec_()

    if win.cancelled:
        raise RuntimeError("ROI selection cancelled.")
    if not win.confirmed or len(win.canvas.points) < 3:
        raise RuntimeError("No valid ROI selected.")

    polygon = np.array([[int(p[0]), int(p[1])] for p in win.canvas.points],
                       dtype=np.int32)

    H, W = image.shape
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)

    x, y, w, h = cv2.boundingRect(polygon)
    bbox = (x, y, w, h)

    print(f"ROI: {len(polygon)} vertices  |  bbox x={x}, y={y}, w={w}, h={h}")
    return polygon, mask, bbox