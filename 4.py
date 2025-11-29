import sys
import numpy as np
import cv2

from PyQt5.QtCore import (
    Qt,
    QRect,
    QPoint,
    pyqtSignal,
    QSize,
)
from PyQt5.QtGui import (
    QImage,
    QPixmap,
    QPainter,
    QPen,
    QPolygon,
)
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFileDialog,
    QSlider,
    QScrollArea,
    QMessageBox,
    QRubberBand,
)


class ImageLabel(QLabel):
    """Label แสดงภาพ + รองรับโหมด crop / polygon ROI ด้วยเมาส์"""

    rectSelected = pyqtSignal(QRect)
    polygonSelected = pyqtSignal(QPolygon)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.mode = "view"  # 'view', 'crop', 'polygon'
        self.rubber_band = None
        self.origin = QPoint()

        self.polygon_points = []
        self.drawing_polygon = False

    def set_mode(self, mode: str):
        self.mode = mode
        if mode != "polygon":
            self.polygon_points = []
            self.drawing_polygon = False
            self.update()

    def has_image(self):
        return self.pixmap() is not None

    def mousePressEvent(self, event):
        if not self.has_image():
            return

        # โหมด crop: กดเมาส์ซ้ายแล้วเริ่มสร้าง rubber band
        if self.mode == "crop" and event.button() == Qt.LeftButton:
            self.origin = event.pos()
            if self.rubber_band is None:
                self.rubber_band = QRubberBand(QRubberBand.Rectangle, self)
            # เริ่มจากสี่เหลี่ยมเล็ก ๆ
            self.rubber_band.setGeometry(QRect(self.origin, QSize()))
            self.rubber_band.show()

        # โหมด polygon ROI: คลิกเพิ่มจุด
        elif self.mode == "polygon" and event.button() == Qt.LeftButton:
            self.polygon_points.append(event.pos())
            self.drawing_polygon = True
            self.update()

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not self.has_image():
            return

        if self.mode == "crop" and self.rubber_band is not None:
            rect = QRect(self.origin, event.pos()).normalized()
            self.rubber_band.setGeometry(rect)

        self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if not self.has_image():
            return

        if self.mode == "crop" and self.rubber_band is not None:
            rect = self.rubber_band.geometry().normalized()
            self.rubber_band.hide()
            if rect.width() > 5 and rect.height() > 5:
                self.rectSelected.emit(rect)

        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        """ดับเบิลคลิกเพื่อจบการเลือก polygon ROI"""
        if self.mode == "polygon" and self.polygon_points:
            polygon = QPolygon(self.polygon_points)
            self.polygonSelected.emit(polygon)
            self.mode = "view"
            self.drawing_polygon = False
            self.update()
        super().mouseDoubleClickEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.has_image():
            return

        # วาด polygon ROI ทับบนภาพ
        if self.polygon_points:
            painter = QPainter(self)
            pen = QPen(Qt.red, 2, Qt.SolidLine)
            painter.setPen(pen)
            for i in range(len(self.polygon_points) - 1):
                painter.drawLine(self.polygon_points[i], self.polygon_points[i + 1])
            for p in self.polygon_points:
                painter.drawEllipse(p, 3, 3)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Monkey X-ray Processor (Crop / ROI / Rib Removal / Gamma)")
        self.original_image = None        # ภาพต้นฉบับ (full)
        self.current_image = None         # ภาพที่แสดง/แก้ไขปัจจุบัน (grayscale)
        self.gamma_base_image = None      # ภาพฐานสำหรับคำนวณ gamma
        self.roi_mask = None              # mask จาก polygon ROI (ถ้ามี)

        # --------- ส่วนแสดงภาพ ---------
        self.image_label = ImageLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        self.image_label.rectSelected.connect(self.on_rect_selected)
        self.image_label.polygonSelected.connect(self.on_polygon_selected)

        scroll_area = QScrollArea()
        scroll_area.setWidget(self.image_label)
        scroll_area.setWidgetResizable(True)

        # --------- ปุ่มด้านซ้าย ---------
        btn_load = QPushButton("นำภาพเข้าจากไฟล์")
        btn_load.clicked.connect(self.load_image)

        btn_reset = QPushButton("Reset ภาพให้เป็นค่าเดิม")
        btn_reset.clicked.connect(self.reset_image)

        btn_crop = QPushButton("Crop ภาพจาก cursor")
        btn_crop.clicked.connect(self.start_crop_mode)

        btn_polygon = QPushButton("Polygon ROI")
        btn_polygon.clicked.connect(self.start_polygon_mode)

        btn_rib = QPushButton("ลบซี่โครงใน ROI / ภาพ")
        btn_rib.clicked.connect(self.remove_ribs)

        btn_save = QPushButton("เซฟภาพปัจจุบัน")
        btn_save.clicked.connect(self.save_image)

        left_layout = QVBoxLayout()
        left_layout.addWidget(btn_load)
        left_layout.addWidget(btn_reset)
        left_layout.addWidget(btn_crop)
        left_layout.addWidget(btn_polygon)
        left_layout.addWidget(btn_rib)
        left_layout.addWidget(btn_save)
        left_layout.addStretch()

        left_widget = QWidget()
        left_widget.setLayout(left_layout)

        # --------- แถบ gamma ด้านขวา ---------
        self.gamma_slider = QSlider(Qt.Vertical)
        self.gamma_slider.setMinimum(20)   # 0.2
        self.gamma_slider.setMaximum(300)  # 3.0
        self.gamma_slider.setValue(100)    # 1.0
        self.gamma_slider.setTickInterval(10)
        self.gamma_slider.setTickPosition(QSlider.TicksRight)
        self.gamma_slider.valueChanged.connect(self.on_gamma_changed)

        gamma_label = QLabel("Gamma\n0.2 - 3.0")
        gamma_label.setAlignment(Qt.AlignCenter)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.gamma_slider, stretch=1)
        right_layout.addWidget(gamma_label)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        # --------- layout หลัก ---------
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        main_layout.addWidget(left_widget)
        main_layout.addWidget(scroll_area, stretch=1)
        main_layout.addWidget(right_widget)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.resize(1200, 700)

    # -----------------------------
    # ฟังก์ชันเกี่ยวกับภาพ / GUI
    # -----------------------------
    def show_message(self, text):
        QMessageBox.information(self, "Info", text)

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "เลือกไฟล์ภาพ X-ray",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)",
        )
        if not file_name:
            return

        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        if img is None:
            self.show_message("ไม่สามารถเปิดไฟล์ภาพได้")
            return

        self.original_image = img
        self.current_image = img.copy()
        self.gamma_base_image = self.current_image.copy()
        self.roi_mask = None
        self.image_label.set_mode("view")
        self.gamma_slider.setValue(100)

        self.update_image_display()
        self.show_message("โหลดภาพเรียบร้อย")

    def reset_image(self):
        if self.original_image is None:
            return
        self.current_image = self.original_image.copy()
        self.gamma_base_image = self.current_image.copy()
        self.roi_mask = None
        self.image_label.set_mode("view")
        self.gamma_slider.setValue(100)
        self.update_image_display()

    def save_image(self):
        if self.current_image is None:
            return
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "บันทึกภาพปัจจุบัน",
            "",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;All Files (*)",
        )
        if not file_name:
            return

        cv2.imwrite(file_name, self.current_image)
        self.show_message("บันทึกภาพแล้ว")

    def update_image_display(self):
        if self.current_image is None:
            self.image_label.clear()
            return

        img = self.current_image
        if img.ndim != 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        h, w = img.shape
        bytes_per_line = w
        q_img = QImage(
            img.data, w, h, bytes_per_line, QImage.Format_Grayscale8
        )
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap)
        self.image_label.adjustSize()

    # -----------------------------
    # Crop
    # -----------------------------
    def start_crop_mode(self):
        if self.current_image is None:
            self.show_message("กรุณาโหลดภาพก่อน")
            return
        self.image_label.set_mode("crop")
        self.show_message("โหมด Crop: ลากเมาส์เลือกกรอบบนภาพ แล้วปล่อยเมาส์เพื่อ Crop")

    def on_rect_selected(self, rect: QRect):
        if self.current_image is None:
            return

        img_h, img_w = self.current_image.shape

        x1 = max(0, rect.left())
        y1 = max(0, rect.top())
        x2 = min(img_w - 1, rect.right())
        y2 = min(img_h - 1, rect.bottom())

        if x2 - x1 < 10 or y2 - y1 < 10:
            self.show_message("กรอบเล็กเกินไป")
            return

        cropped = self.current_image[y1:y2 + 1, x1:x2 + 1].copy()
        self.current_image = cropped
        self.gamma_base_image = self.current_image.copy()
        self.roi_mask = None
        self.image_label.set_mode("view")
        self.gamma_slider.setValue(100)
        self.update_image_display()

    # -----------------------------
    # Polygon ROI
    # -----------------------------
    def start_polygon_mode(self):
        if self.current_image is None:
            self.show_message("กรุณาโหลดภาพก่อน")
            return
        self.roi_mask = None
        self.image_label.polygon_points = []
        self.image_label.set_mode("polygon")
        self.show_message(
            "โหมด Polygon ROI: คลิกเพิ่มจุดทีละจุดบนภาพ แล้วดับเบิลคลิกเพื่อจบ ROI"
        )

    def on_polygon_selected(self, polygon: QPolygon):
        if self.current_image is None:
            return
        if polygon.isEmpty():
            return

        img_h, img_w = self.current_image.shape
        mask = np.zeros((img_h, img_w), dtype=np.uint8)

        pts = []
        for i in range(polygon.count()):
            p = polygon.point(i)
            x = np.clip(p.x(), 0, img_w - 1)
            y = np.clip(p.y(), 0, img_h - 1)
            pts.append([x, y])

        pts = np.array(pts, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

        self.roi_mask = mask
        self.gamma_base_image = self.current_image.copy()
        self.gamma_slider.setValue(100)

        self.show_message("เลือก ROI เรียบร้อย สามารถปรับ gamma หรือกดลบซี่โครงใน ROI ได้แล้ว")

    # -----------------------------
    # Gamma correction
    # -----------------------------
    def on_gamma_changed(self, value):
        if self.gamma_base_image is None:
            return

        gamma = value / 100.0  # 0.2 - 3.0
        gamma = max(0.01, gamma)

        corrected = self.gamma_correction(self.gamma_base_image, gamma, self.roi_mask)
        self.current_image = corrected
        self.update_image_display()

    @staticmethod
    def gamma_correction(img, gamma, mask=None):
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        inv_gamma = 1.0 / gamma
        table = np.array(
            [(i / 255.0) ** inv_gamma * 255 for i in range(256)]
        ).astype("uint8")

        corrected = cv2.LUT(img, table)

        if mask is not None:
            out = img.copy()
            out[mask > 0] = corrected[mask > 0]
            return out
        else:
            return corrected

    # -----------------------------
    # Rib removal (orientation-based + inpaint)
    # -----------------------------
    def remove_ribs(self):
        if self.current_image is None:
            self.show_message("กรุณาโหลดภาพก่อน")
            return

        img = self.current_image

        # ถ้าไม่มี ROI ให้ใช้ทั้งภาพ
        if self.roi_mask is None:
            roi_mask = np.ones_like(img, dtype=np.uint8) * 255
        else:
            roi_mask = self.roi_mask

        # 1) ลด noise แล้วเพิ่ม contrast เล็กน้อย
        proc = cv2.GaussianBlur(img, (3, 3), 0)
        proc = cv2.equalizeHist(proc)

        # 2) หา edge ทั้งภาพ
        edges = cv2.Canny(proc, 30, 80)

        # 3) gradient gx, gy สำหรับคำนวณทิศทาง
        gx = cv2.Sobel(proc, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(proc, cv2.CV_32F, 0, 1, ksize=3)

        grad_angle = np.arctan2(gy, gx)             # -pi..pi
        line_angle = grad_angle + np.pi / 2.0       # ทิศของเส้น ≈ grad + 90°
        line_angle = np.mod(line_angle, np.pi)      # 0..pi

        # 4) เลือกเฉพาะเส้นที่เกือบตั้ง (90° ± 20°)
        target = np.pi / 2.0
        tol = np.deg2rad(20)
        vertical_like = np.abs(line_angle - target) < tol

        # 5) สร้าง mask ซี่โครง: เป็น edge + มุมเกือบตั้ง + อยู่ใน ROI
        rib_mask = np.zeros_like(img, dtype=np.uint8)
        rib_mask[(edges > 0) & vertical_like & (roi_mask > 0)] = 255

        # ทำให้เส้นหนาขึ้นหน่อย
        rib_mask = cv2.dilate(
            rib_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1
        )

        # 6) ใช้ inpaint กลบเส้นซี่โครง
        inpainted = cv2.inpaint(proc, rib_mask, 3, cv2.INPAINT_TELEA)

        # 7) ใส่ผลเฉพาะใน ROI กลับไปยังภาพเดิม
        result = img.copy()
        result[roi_mask > 0] = inpainted[roi_mask > 0]

        self.current_image = result
        self.gamma_base_image = self.current_image.copy()
        self.gamma_slider.setValue(100)
        self.update_image_display()
        self.show_message("ประมวลผลลบซี่โครงด้วยวิธี orientation-based เรียบร้อย")


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
