import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox,
    QSlider, QSpinBox, QCheckBox, QGroupBox, QFormLayout
)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen


# ---------- Widget แสดงรูป ที่รองรับ crop / polygon ROI ----------

class ImageLabel(QLabel):
    # ส่งสัญญาณเวลา crop เสร็จ: (x1, y1, x2, y2)
    rectFinished = pyqtSignal(int, int, int, int)
    # ส่งสัญญาณเวลา polygon ROI เสร็จ: list[QPoint]
    polyFinished = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.mode = "idle"   # "idle", "crop", "poly"
        self.start_point = None
        self.end_point = None
        self.drawing = False

        self.poly_points = []
        self.temp_point = None  # จุดล่าสุดตอนลากเมาส์เพื่อแสดงเส้น

    def set_mode(self, mode: str):
        self.mode = mode
        self.start_point = None
        self.end_point = None
        self.drawing = False
        self.temp_point = None
        if mode != "poly":
            self.poly_points = []
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.mode == "crop":
                self.drawing = True
                self.start_point = event.pos()
                self.end_point = event.pos()
                self.update()
            elif self.mode == "poly":
                # เพิ่มจุดใหม่
                self.poly_points.append(event.pos())
                self.update()

    def mouseMoveEvent(self, event):
        if self.mode == "crop" and self.drawing:
            self.end_point = event.pos()
            self.update()
        elif self.mode == "poly" and self.poly_points:
            # เก็บ temp point เพื่อวาดเส้นจากจุดสุดท้ายไปตำแหน่งเมาส์
            self.temp_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.mode == "crop" and self.drawing:
            self.drawing = False
            self.end_point = event.pos()
            # แปลงให้ได้ (x1, y1, x2, y2) แบบเรียงเล็ก→ใหญ่
            x1 = min(self.start_point.x(), self.end_point.x())
            y1 = min(self.start_point.y(), self.end_point.y())
            x2 = max(self.start_point.x(), self.end_point.x())
            y2 = max(self.start_point.y(), self.end_point.y())
            self.rectFinished.emit(x1, y1, x2, y2)
            # กลับไป idle
            self.set_mode("idle")

    def mouseDoubleClickEvent(self, event):
        # ดับเบิลคลิกปิด polygon
        if self.mode == "poly" and self.poly_points:
            self.polyFinished.emit(self.poly_points.copy())
            self.set_mode("idle")

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)

        # วาด rectangle ตอน crop
        if self.mode == "crop" and self.start_point and self.end_point:
            pen = QPen(Qt.red, 2, Qt.DashLine)
            painter.setPen(pen)
            rect = (
                min(self.start_point.x(), self.end_point.x()),
                min(self.start_point.y(), self.end_point.y()),
                abs(self.start_point.x() - self.end_point.x()),
                abs(self.start_point.y() - self.end_point.y())
            )
            painter.drawRect(*rect)

        # วาด polygon ROI
        if self.mode == "poly" and self.poly_points:
            pen = QPen(Qt.green, 2, Qt.SolidLine)
            painter.setPen(pen)
            # วาดเส้นเชื่อมจุดถาวร
            for i in range(len(self.poly_points) - 1):
                painter.drawLine(self.poly_points[i], self.poly_points[i+1])
            # วาดเส้นจากจุดสุดท้ายไป temp_point (เมาส์)
            if self.temp_point is not None:
                painter.drawLine(self.poly_points[-1], self.temp_point)


# ---------- ฟังก์ชันลบซี่โครงด้วย Otsu + morphology + inpaint ----------

def remove_ribs_otsu(
    img_gray: np.ndarray,
    vert_kernel_width: int = 5,
    vert_kernel_height: int = 40,
    spine_min_area: int = 500,
    inpaint_radius: int = 3,
    invert_threshold: bool = False,
    roi_mask: np.ndarray | None = None
) -> np.ndarray:
    """
    img_gray: ภาพ X-ray ขาวดำ uint8
    ไม่ทำ Histogram Equalization และไม่ทำ Gaussian Blur
    ถ้า roi_mask ไม่ None จะ apply เฉพาะบริเวณ roi_mask>0
    """
    img = img_gray.copy()

    # 1) Otsu threshold จากภาพ grayscale เดิมเลย
    if invert_threshold:
        flag = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    else:
        flag = cv2.THRESH_BINARY + cv2.THRESH_OTSU

    _, bone_mask = cv2.threshold(img, 0, 255, flag)

    # 2) Opening เล็ก ๆ ลบ noise จุดเล็ก ๆ
    kernel_small = np.ones((3, 3), np.uint8)
    bone_mask = cv2.morphologyEx(bone_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

    # 3) แยกกระดูกสันหลังด้วย vertical kernel
    vert_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (vert_kernel_width, vert_kernel_height)
    )
    spine_candidate = cv2.morphologyEx(bone_mask, cv2.MORPH_OPEN, vert_kernel)

    h, w = img.shape
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        spine_candidate, connectivity=8
    )

    center_x = w / 2.0
    best_label = 0
    best_dist = 1e9

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        cx, cy = centroids[label]
        if area < spine_min_area:
            continue
        dist = abs(cx - center_x)
        if dist < best_dist:
            best_dist = dist
            best_label = label

    spine_mask = np.zeros_like(bone_mask)
    spine_mask[labels == best_label] = 255

    # 4) ribs = bone - spine
    ribs_mask = cv2.subtract(bone_mask, spine_mask)
    ribs_mask = cv2.dilate(ribs_mask, kernel_small, iterations=1)

    # ถ้ามี ROI mask -> ลบซี่โครงเฉพาะใน ROI
    if roi_mask is not None:
        roi = np.zeros_like(ribs_mask)
        roi[roi_mask > 0] = 255
        ribs_mask = cv2.bitwise_and(ribs_mask, roi)

    # 5) Inpaint ลบซี่โครง
    no_rib = cv2.inpaint(img, ribs_mask, inpaint_radius, cv2.INPAINT_TELEA)

    # 6) ใส่กระดูกสันหลังกลับเข้าไปจากภาพเดิม
    result = no_rib.copy()
    result[spine_mask > 0] = img[spine_mask > 0]

    return result



# ---------- Main GUI ----------

class XRayGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("X-Ray Rib Removal (Otsu) GUI")

        # เก็บรูป
        self.original_img = None    # ภาพเดิม (gray)
        self.current_img = None     # ภาพที่แก้ไขแล้ว (gray)
        self.roi_mask = None        # polygon ROI mask (uint8)

        # สร้าง widget หลัก
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)

        # ----- ด้านซ้าย: ภาพ -----
        self.image_label = ImageLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #222;")
        main_layout.addWidget(self.image_label, stretch=3)

        # เชื่อมสัญญาณ crop / polygon
        self.image_label.rectFinished.connect(self.on_rect_finished)
        self.image_label.polyFinished.connect(self.on_poly_finished)

        # ----- ด้านขวา: control panel -----
        right_panel = QVBoxLayout()
        main_layout.addLayout(right_panel, stretch=1)

        # ปุ่มต่าง ๆ
        btn_load = QPushButton("Load Image")
        btn_reset = QPushButton("Reset Image")
        btn_crop = QPushButton("Crop (Rectangle)")
        btn_poly = QPushButton("Polygon ROI")
        btn_save = QPushButton("Save Current Image")
        btn_apply = QPushButton("Apply Rib Removal (Otsu)")

        btn_load.clicked.connect(self.load_image)
        btn_reset.clicked.connect(self.reset_image)
        btn_crop.clicked.connect(self.activate_crop_mode)
        btn_poly.clicked.connect(self.activate_poly_mode)
        btn_save.clicked.connect(self.save_image)
        btn_apply.clicked.connect(self.apply_rib_removal)

        right_panel.addWidget(btn_load)
        right_panel.addWidget(btn_reset)
        right_panel.addWidget(btn_crop)
        right_panel.addWidget(btn_poly)
        right_panel.addWidget(btn_save)
        right_panel.addWidget(btn_apply)

        # ----- กล่องพารามิเตอร์ด้านขวา -----
        param_group = QGroupBox("Parameters")
        param_layout = QFormLayout()
        param_group.setLayout(param_layout)

        # vertical kernel size
        self.spin_vert_w = QSpinBox()
        self.spin_vert_w.setRange(1, 51)
        self.spin_vert_w.setValue(5)

        self.spin_vert_h = QSpinBox()
        self.spin_vert_h.setRange(5, 200)
        self.spin_vert_h.setValue(40)

        # spine min area
        self.spin_spine_area = QSpinBox()
        self.spin_spine_area.setRange(10, 5000)
        self.spin_spine_area.setValue(500)

        # inpaint radius
        self.spin_inpaint_r = QSpinBox()
        self.spin_inpaint_r.setRange(1, 20)
        self.spin_inpaint_r.setValue(3)

        # invert threshold ใช้ในกรณีกระดูกมืดพื้นสว่าง
        self.chk_invert = QCheckBox("Invert Threshold (for dark bones)")

        param_layout.addRow("Vert Kernel Width:", self.spin_vert_w)
        param_layout.addRow("Vert Kernel Height:", self.spin_vert_h)
        param_layout.addRow("Spine Min Area:", self.spin_spine_area)
        param_layout.addRow("Inpaint Radius:", self.spin_inpaint_r)
        param_layout.addRow(self.chk_invert)

        right_panel.addWidget(param_group)

        # ช่องว่างด้านล่าง
        right_panel.addStretch()

        self.resize(1100, 700)

    # ---------- Utility แปลง numpy → QPixmap ----------

    def show_image(self, img_gray: np.ndarray):
        """แสดงภาพ grayscale บน QLabel"""
        if img_gray is None:
            return
        h, w = img_gray.shape
        # ให้ข้อมูลเป็น contiguous memory
        img_gray_c = np.ascontiguousarray(img_gray)
        qimg = QImage(
            img_gray_c.data, w, h, w, QImage.Format_Grayscale8
        )
        pix = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pix)
        self.image_label.resize(pix.size())

    # ---------- slot ของปุ่ม ----------

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not fname:
            return
        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        if img is None:
            QMessageBox.warning(self, "Error", "ไม่สามารถเปิดไฟล์ภาพได้")
            return
        self.original_img = img
        self.current_img = img.copy()
        self.roi_mask = None
        self.show_image(self.current_img)

    def reset_image(self):
        if self.original_img is None:
            return
        self.current_img = self.original_img.copy()
        self.roi_mask = None
        self.show_image(self.current_img)

    def activate_crop_mode(self):
        if self.current_img is None:
            return
        self.image_label.set_mode("crop")

    def activate_poly_mode(self):
        if self.current_img is None:
            return
        # สร้าง mask ใหม่
        self.roi_mask = None
        self.image_label.set_mode("poly")

    def save_image(self):
        if self.current_img is None:
            return
        fname, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp)"
        )
        if not fname:
            return
        cv2.imwrite(fname, self.current_img)
        QMessageBox.information(self, "Saved", "บันทึกภาพเรียบร้อยแล้ว")

    def apply_rib_removal(self):
        if self.current_img is None:
            return

        vert_w = self.spin_vert_w.value()
        vert_h = self.spin_vert_h.value()
        spine_area = self.spin_spine_area.value()
        inpaint_r = self.spin_inpaint_r.value()
        invert = self.chk_invert.isChecked()

        try:
            result = remove_ribs_otsu(
                self.current_img,
                vert_kernel_width=vert_w,
                vert_kernel_height=vert_h,
                spine_min_area=spine_area,
                inpaint_radius=inpaint_r,
                invert_threshold=invert,
                roi_mask=self.roi_mask
            )
        except Exception as e:
            QMessageBox.warning(self, "Error", f"เกิดข้อผิดพลาดเวลา process:\n{e}")
            return

        self.current_img = result
        self.show_image(self.current_img)

    # ---------- slot จาก ImageLabel (crop / polygon) ----------

    def on_rect_finished(self, x1, y1, x2, y2):
        """รับกรอบ crop จาก cursor แล้ว crop ภาพ"""
        if self.current_img is None:
            return
        h, w = self.current_img.shape

        # จำกัดไม่ให้ออกนอกภาพ
        x1 = max(0, min(w-1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h-1, y1))
        y2 = max(0, min(h, y2))

        if x2 <= x1 or y2 <= y1:
            return

        cropped = self.current_img[y1:y2, x1:x2]
        if cropped.size == 0:
            return

        self.current_img = cropped
        # original ไม่เปลี่ยน (ให้ reset กลับภาพรับเข้าตอนแรกได้)
        self.show_image(self.current_img)
        # ROI mask ถ้ามีก็ทิ้งไป เพราะขนาดภาพเปลี่ยน
        self.roi_mask = None

    def on_poly_finished(self, points):
        """รับ polygon ROI จากจุดที่คลิก แล้วสร้าง roi_mask"""
        if self.current_img is None:
            return
        h, w = self.current_img.shape

        pts = []
        for p in points:
            x = max(0, min(w-1, p.x()))
            y = max(0, min(h-1, p.y()))
            pts.append([x, y])

        if len(pts) < 3:
            return

        pts = np.array(pts, np.int32)
        pts = pts.reshape((-1, 1, 2))

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        self.roi_mask = mask

        QMessageBox.information(
            self, "ROI",
            "สร้าง Polygon ROI เรียบร้อยแล้ว.\n"
            "เมื่อกด Apply Rib Removal จะปรับเฉพาะบริเวณ ROI นี้"
        )


def main():
    app = QApplication(sys.argv)
    win = XRayGUI()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
