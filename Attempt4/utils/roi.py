import sys
import cv2
import numpy as np
import math
from PyQt5.QtWidgets import (QDialog, QWidget, QHBoxLayout, QVBoxLayout,
                             QLabel, QPushButton, QFrame, QListWidget, QListWidgetItem,
                             QApplication)
from PyQt5.QtCore import Qt, QPointF, pyqtSignal, QRect
from PyQt5.QtGui import (QPainter, QColor, QPen, QBrush, QImage, QPixmap,
                         QPainterPath, QPolygonF)

# Theme
C_BG = QColor("#0f172a")
C_PANEL = QColor("#1e293b")
C_ACCENT = QColor("#38bdf8")
C_SELECTED = QColor("#10b981")
C_FILL = QColor(56, 189, 248, 45)


def _dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def _dist_to_seg(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    if dx == dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy)), t


class PolygonCanvas(QWidget):
    changed = pyqtSignal()

    def __init__(self, image: np.ndarray):
        super().__init__()

        if len(image.shape) == 2:
            self.img_np = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            self.img_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.H, self.W = self.img_np.shape[:2]

        self.points = []
        self.selected = None
        self.dragging = False
        self.scale = 1.0
        self.offset = QPointF(0, 0)
        self._pan_last = None

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        h, w, ch = self.img_np.shape
        self._buffer = self.img_np.tobytes()
        qimg = QImage(self._buffer, w, h, ch * w, QImage.Format_RGB888)

        self._base_pixmap = QPixmap.fromImage(qimg)

    def _i2c(self, ix, iy):
        return QPointF(ix * self.scale + self.offset.x(), iy * self.scale + self.offset.y())

    def _c2i(self, cx, cy):
        return (
            max(0.0, min(self.W - 1, (cx - self.offset.x()) / self.scale)),
            max(0.0, min(self.H - 1, (cy - self.offset.y()) / self.scale))
        )

    def fit(self):
        w, h = self.width(), self.height()
        if w < 2 or h < 2:
            return
        self.scale = min(w / self.W, h / self.H) * 0.96
        self.offset = QPointF((w - self.W * self.scale) / 2, (h - self.H * self.scale) / 2)
        self.update()

    def _hit_vertex(self, cx, cy, thresh=14):
        best_i, best_d = None, float("inf")
        for i, p in enumerate(self.points):
            cp = self._i2c(p[0], p[1])
            d = _dist(cx, cy, cp.x(), cp.y())
            if d < thresh and d < best_d:
                best_d, best_i = d, i
        return best_i

    def _hit_edge(self, cx, cy):
        n = len(self.points)
        if n < 2:
            return None, None
        best_i, best_d, best_t = None, float("inf"), 0
        for i in range(n if n >= 3 else n - 1):
            a, b = self._i2c(*self.points[i]), self._i2c(*self.points[(i + 1) % n])
            d, t = _dist_to_seg(cx, cy, a.x(), a.y(), b.x(), b.y())
            if d < 10 and d < best_d:
                best_d, best_i, best_t = d, i, t
        return best_i, best_t

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), C_BG)

        p.drawPixmap(
            QRect(int(self.offset.x()), int(self.offset.y()),
                  int(self.W * self.scale), int(self.H * self.scale)),
            self._base_pixmap
        )

        n = len(self.points)
        if n == 0:
            return

        cpts = [self._i2c(pt[0], pt[1]) for pt in self.points]

        if n >= 3:
            path = QPainterPath()
            path.addPolygon(QPolygonF(cpts + [cpts[0]]))
            p.fillPath(path, QBrush(C_FILL))

        pen = QPen(C_ACCENT, 1.5, Qt.DashLine)
        p.setPen(pen)
        for i in range(n if n >= 3 else n - 1):
            p.drawLine(cpts[i], cpts[(i + 1) % n])

        for i, cp in enumerate(cpts):
            color = C_SELECTED if i == self.selected else C_ACCENT
            r = 9 if i == self.selected else 7
            p.setPen(QPen(color, 2))
            p.setBrush(QBrush(C_BG))
            p.drawEllipse(cp, float(r), float(r))

    def mousePressEvent(self, event):
        cx, cy = event.x(), event.y()

        if event.button() == Qt.LeftButton:
            vi = self._hit_vertex(cx, cy)
            if vi is not None:
                self.selected, self.dragging = vi, True
            else:
                ei, t = self._hit_edge(cx, cy)
                if ei is not None:
                    p1, p2 = self.points[ei], self.points[(ei + 1) % len(self.points)]
                    self.points.insert(
                        ei + 1,
                        [p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1])]
                    )
                    self.selected, self.dragging = ei + 1, True
                else:
                    self.points.append(list(self._c2i(cx, cy)))
                    self.selected = len(self.points) - 1

            self.update()
            self.changed.emit()

        elif event.button() == Qt.RightButton:
            vi = self._hit_vertex(cx, cy, thresh=30)
            if vi is not None:
                self.points.pop(vi)
                if self.selected is not None and self.selected > vi:
                    self.selected -= 1
                elif self.selected == vi:
                    self.selected = None
                self.update()
                self.changed.emit()

        elif event.button() == Qt.MiddleButton:
            self._pan_last = QPointF(cx, cy)

    def mouseMoveEvent(self, event):
        cx, cy = event.x(), event.y()

        if self.dragging and self.selected is not None:
            self.points[self.selected] = list(self._c2i(cx, cy))
            self.update()
            self.changed.emit()

        elif self._pan_last and event.buttons() & Qt.MiddleButton:
            self.offset += QPointF(cx - self._pan_last.x(), cy - self._pan_last.y())
            self._pan_last = QPointF(cx, cy)
            self.update()

        self.setCursor(Qt.SizeAllCursor if self._hit_vertex(cx, cy) is not None else Qt.CrossCursor)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False
        elif event.button() == Qt.MiddleButton:
            self._pan_last = None

    def wheelEvent(self, event):
        factor = 1.12 if event.angleDelta().y() > 0 else 1 / 1.12
        new_scale = max(0.05, min(30.0, self.scale * factor))
        ratio = new_scale / self.scale

        self.offset = QPointF(
            event.x() - ratio * (event.x() - self.offset.x()),
            event.y() - ratio * (event.y() - self.offset.y())
        )

        self.scale = new_scale
        self.update()

    def resizeEvent(self, event):
        self.fit()

    def clear(self):
        self.points.clear()
        self.selected = None
        self.update()
        self.changed.emit()


class ROIDialog(QDialog):
    def __init__(self, image: np.ndarray):
        super().__init__()

        self.result_data = None
        self.setWindowTitle("DIC - Define Region of Interest")

        # Apply dark theme styling
        self.setStyleSheet("""
            QDialog, QWidget { 
                background-color: #0f172a; 
                color: #f8fafc; 
                font-family: 'Segoe UI', system-ui, sans-serif; 
            }
            QFrame#SidePanel { 
                background-color: #1e293b; 
                border-radius: 12px; 
            }
            QLabel { 
                font-size: 13px; 
                color: #cbd5e1; 
            }
            QLabel#Title { 
                font-size: 18px; 
                font-weight: bold; 
                color: #38bdf8; 
                margin-bottom: 10px; 
            }
            QListWidget { 
                background-color: #0f172a; 
                border: 1px solid #334155; 
                border-radius: 8px; 
                padding: 5px; 
                font-family: monospace; 
            }
            QPushButton { 
                background-color: #334155; 
                color: white; 
                font-weight: bold; 
                border-radius: 8px; 
                padding: 12px; 
                border: none; 
            }
            QPushButton:hover { 
                background-color: #475569; 
            }
            QPushButton#Confirm { 
                background-color: #10b981; 
            }
            QPushButton#Confirm:hover { 
                background-color: #34d399; 
            }
        """)

        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(int(screen.width() * 0.8), int(screen.height() * 0.8))

        layout = QHBoxLayout(self)

        self.canvas = PolygonCanvas(image)
        layout.addWidget(self.canvas, stretch=1)

        panel = QFrame()
        panel.setObjectName("SidePanel")
        panel.setFixedWidth(280)
        p_layout = QVBoxLayout(panel)
        p_layout.setContentsMargins(15, 15, 15, 15)
        p_layout.setSpacing(10)

        lbl_title = QLabel("ROI Editor")
        lbl_title.setObjectName("Title")
        p_layout.addWidget(lbl_title)

        lbl_controls = QLabel(
            "Controls:\n"
            "• LMB: Add / Drag Vertex\n"
            "• Edge Click: Insert Vertex\n"
            "• RMB: Delete Vertex\n"
            "• MMB: Pan Canvas\n"
            "• Scroll: Zoom In/Out"
        )
        p_layout.addWidget(lbl_controls)

        p_layout.addSpacing(10)

        self.coord_list = QListWidget()
        p_layout.addWidget(self.coord_list)

        btn_clear = QPushButton("Clear All Vertices")
        btn_clear.clicked.connect(self.canvas.clear)
        p_layout.addWidget(btn_clear)

        self.btn_ok = QPushButton("Confirm ROI")
        self.btn_ok.setObjectName("Confirm")
        self.btn_ok.clicked.connect(self._confirm)
        p_layout.addWidget(self.btn_ok)

        layout.addWidget(panel)

        self.canvas.changed.connect(self._update_ui)

    def _update_ui(self):
        self.coord_list.clear()
        for i, p in enumerate(self.canvas.points):
            item = QListWidgetItem(f"P{i + 1:02d}: ({int(p[0]):4d}, {int(p[1]):4d})")
            self.coord_list.addItem(item)

        if len(self.canvas.points) >= 3:
            self.btn_ok.setEnabled(True)
            self.btn_ok.setStyleSheet("background-color: #10b981;")
        else:
            self.btn_ok.setEnabled(False)
            self.btn_ok.setStyleSheet("background-color: #334155; color: #94a3b8;")

    def _confirm(self):
        if len(self.canvas.points) < 3:
            return

        poly = np.array(self.canvas.points, dtype=np.int32)
        mask = np.zeros((self.canvas.H, self.canvas.W), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)

        self.result_data = (poly, mask, cv2.boundingRect(poly))
        self.accept()


def get_roi(image):
    app = QApplication.instance() or QApplication(sys.argv)

    dialog = ROIDialog(image)
    if dialog.exec_() == QDialog.Accepted:
        return dialog.result_data

    return None, None, None


if __name__ == "__main__":
    img = np.zeros((500, 500), dtype=np.uint8)
    poly, mask, rect = get_roi(img)
    if poly is not None:
        print("Polygon bounds:", rect)