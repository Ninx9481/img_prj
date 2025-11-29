import sys
import cv2
import numpy as np

from PyQt5.QtCore import Qt, QRect, QPoint
from PyQt5.QtGui import QImage, QPainter, QPen
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QSlider, QGroupBox,
    QSpacerItem, QSizePolicy
)

# -------------------- Widget แสดงภาพ + รองรับ crop / polygon ROI --------------------


class ImageLabel(QLabel):
    """
    QLabel สำหรับแสดงภาพแบบ scale-to-fit และรองรับ:
    - ลากกรอบ crop (โหมด 'crop')
    - เลือก polygon ROI (โหมด 'polygon')
    ส่งพิกัดกลับในหน่วย "พิกเซลของภาพ" (ไม่ใช่พิกเซลหน้าจอ)
    """
    rectSelected = Qt.pyqtSignal(QRect)      # QRect ในพิกเซลของภาพ (view)
    polygonFinished = Qt.pyqtSignal(list)    # list[QPoint] ในพิกเซลของภาพ (view)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(400, 400)
        self.qimage = None

        # ขนาดภาพจริง
        self.image_width = 0
        self.image_height = 0

        # scale และ offset ตอนวาด
        self.scale_factor = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0

        # state การลาก/คลิก
        self.mode = "none"   # 'none', 'crop', 'polygon'
        self.dragging = False
        self.start_point_label = QPoint()
        self.current_point_label = QPoint()
        self.polygon_points_label = []

    # ---------- การตั้งค่า/โหมด ----------

    def set_image(self, qimage: QImage):
        """รับ QImage ของภาพ view ปัจจุบัน"""
        self.qimage = qimage
        if qimage is not None:
            self.image_width = qimage.width()
            self.image_height = qimage.height()
        else:
            self.image_width = 0
            self.image_height = 0
        # reset state selection
        self.set_mode("none")
        self.update()

    def set_mode(self, mode: str):
        self.mode = mode
        self.dragging = False
        self.start_point_label = QPoint()
        self.current_point_label = QPoint()
        self.polygon_points_label = []
        self.update()

    # ---------- helper: แปลงพิกัด label -> image (view) ----------

    def label_to_image_point(self, p_label: QPoint):
        """แปลงพิกัดบน widget ให้เป็นพิกัดบนภาพ (ถ้าอยู่นอกภาพให้คืน None)"""
        if self.qimage is None or self.image_width == 0 or self.image_height == 0:
            return None

        x = (p_label.x() - self.offset_x) / self.scale_factor
        y = (p_label.y() - self.offset_y) / self.scale_factor

        if x < 0 or y < 0 or x >= self.image_width or y >= self.image_height:
            return None

        return QPoint(int(x), int(y))

    # ---------- mouse events ----------

    def mousePressEvent(self, event):
        if self.qimage is None:
            return

        if event.button() == Qt.LeftButton:
            if self.mode == "crop":
                self.dragging = True
                self.start_point_label = event.pos()
                self.current_point_label = event.pos()
                self.update()
            elif self.mode == "polygon":
                # เพิ่มจุด polygon
                self.polygon_points_label.append(event.pos())
                self.update()

        elif event.button() == Qt.RightButton and self.mode == "polygon":
            # จบ polygon (ถ้ามีจุด >=3)
            if len(self.polygon_points_label) >= 3:
                img_points = []
                for p_lab in self.polygon_points_label:
                    p_img = self.label_to_image_point(p_lab)
                    if p_img is not None:
                        img_points.append(p_img)

                if len(img_points) >= 3:
                    self.polygonFinished.emit(img_points)

            self.set_mode("none")

    def mouseMoveEvent(self, event):
        if self.mode == "crop" and self.dragging:
            self.current_point_label = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if (
            event.button() == Qt.LeftButton
            and self.mode == "crop"
            and self.dragging
        ):
            self.dragging = False
            # แปลงจุดเริ่ม/จุดจบเป็นพิกัดภาพ
            p1_img = self.label_to_image_point(self.start_point_label)
            p2_img = self.label_to_image_point(self.current_point_label)

            self.set_mode("none")

            if p1_img is None or p2_img is None:
                return

            rect_img = QRect(p1_img, p2_img).normalized()
            if rect_img.width() > 5 and rect_img.height() > 5:
                self.rectSelected.emit(rect_img)

    # ---------- paintEvent: วาดภาพ (scale-to-fit) + overlay ----------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)

        if self.qimage is not None and self.image_width > 0 and self.image_height > 0:
            w_label = self.width()
            h_label = self.height()

            # คำนวณ scale ให้ภาพทั้งรูปอยู่ใน widget
            s = min(
                w_label / float(self.image_width),
                h_label / float(self.image_height)
            )
            if s <= 0:
                s = 1.0
            self.scale_factor = s

            draw_w = self.image_width * s
            draw_h = self.image_height * s

            self.offset_x = (w_label - draw_w) / 2.0
            self.offset_y = (h_label - draw_h) / 2.0

            target_rect = QRect(
                int(self.offset_x),
                int(self.offset_y),
                int(draw_w),
                int(draw_h)
            )
            painter.drawImage(target_rect, self.qimage)

        # วาดกรอบ crop ชั่วคราว
        if self.mode == "crop" and self.dragging:
            pen = QPen(Qt.red, 2, Qt.DashLine)
            painter.setPen(pen)
            rect = QRect(self.start_point_label, self.current_point_label).normalized()
            painter.drawRect(rect)

        # วาด polygon ROI ชั่วคราว
        if self.mode == "polygon" and len(self.polygon_points_label) >= 1:
            pen = QPen(Qt.green, 2)
            painter.setPen(pen)
            pts = self.polygon_points_label
            for i in range(len(pts) - 1):
                painter.drawLine(pts[i], pts[i + 1])
            for pt in pts:
                painter.drawEllipse(pt, 2, 2)


# -------------------- Main Window --------------------


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("X-ray Rib Suppression GUI")

        # ตัวแปรภาพ
        self.original_image = None      # ภาพเต็ม (numpy, grayscale)
        self.processed_image = None     # ภาพหลัง Start Process
        self.current_display_image = None  # ภาพที่แสดง + ใช้เซฟ (หลัง gamma + crop)

        # crop & ROI อยู่บนพิกัด "ภาพเต็ม"
        self.crop_rect = None           # (x, y, w, h)
        self.roi_mask = None            # boolean mask (h,w)

        # view rect (crop ปัจจุบัน) เฉพาะสำหรับ mapping
        self.view_x0 = 0
        self.view_y0 = 0
        self.view_w = 0
        self.view_h = 0

        # พารามิเตอร์
        self.gamma = 1.0
        self.rib_kernel_height = 35   # จะใช้เป็น (3, h)
        self.inpaint_radius = 3

        # ---------- widget แสดงภาพ ----------
        self.image_label = ImageLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.rectSelected.connect(self.on_rect_selected)
        self.image_label.polygonFinished.connect(self.on_polygon_finished)

        # ---------- ปุ่มด้านขวา ----------
        load_btn = QPushButton("Load Image")
        load_btn.clicked.connect(self.load_image)

        start_proc_btn = QPushButton("Start Process")
        start_proc_btn.clicked.connect(self.start_process)

        reset_crop_btn = QPushButton("Reset Crop")
        reset_crop_btn.clicked.connect(self.reset_crop)

        reset_proc_btn = QPushButton("Reset Processing")
        reset_proc_btn.clicked.connect(self.reset_processing)

        crop_btn = QPushButton("Crop (Rect)")
        crop_btn.clicked.connect(self.start_crop_mode)

        poly_btn = QPushButton("Polygon ROI")
        poly_btn.clicked.connect(self.start_polygon_mode)

        save_btn = QPushButton("Save Image")
        save_btn.clicked.connect(self.save_image)

        # ---------- แถบพารามิเตอร์ทางขวา ----------
        # 1) Gamma slider
        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setRange(10, 300)  # 0.10–3.00
        self.gamma_slider.setValue(100)      # 1.00
        self.gamma_slider.valueChanged.connect(self.on_gamma_change)
        self.gamma_label = QLabel("Gamma: 1.00")

        # 2) Rib kernel height
        self.kernel_slider = QSlider(Qt.Horizontal)
        self.kernel_slider.setRange(5, 80)      # ความสูง 5–80
        self.kernel_slider.setValue(35)
        self.kernel_slider.valueChanged.connect(self.on_kernel_change)
        self.kernel_label = QLabel("Rib kernel height: 35")

        # 3) Inpaint radius
        self.radius_slider = QSlider(Qt.Horizontal)
        self.radius_slider.setRange(1, 15)
        self.radius_slider.setValue(3)
        self.radius_slider.valueChanged.connect(self.on_radius_change)
        self.radius_label = QLabel("Inpaint radius: 3")

        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout()
        params_layout.addWidget(self.gamma_label)
        params_layout.addWidget(self.gamma_slider)
        params_layout.addSpacing(10)
        params_layout.addWidget(self.kernel_label)
        params_layout.addWidget(self.kernel_slider)
        params_layout.addSpacing(10)
        params_layout.addWidget(self.radius_label)
        params_layout.addWidget(self.radius_slider)
        params_group.setLayout(params_layout)

        # Layout ปุ่ม
        btn_layout = QVBoxLayout()
        btn_layout.addWidget(load_btn)
        btn_layout.addWidget(start_proc_btn)
        btn_layout.addWidget(reset_crop_btn)
        btn_layout.addWidget(reset_proc_btn)
        btn_layout.addWidget(crop_btn)
        btn_layout.addWidget(poly_btn)
        btn_layout.addWidget(save_btn)
        btn_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        btn_layout.addWidget(params_group)

        right_panel = QWidget()
        right_panel.setLayout(btn_layout)

        # Layout หลัก
        central = QWidget()
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.image_label, stretch=3)
        main_layout.addWidget(right_panel, stretch=1)
        central.setLayout(main_layout)

        self.setCentralWidget(central)
        self.resize(1300, 800)

    # -------------------- helper แสดงภาพ --------------------

    def show_image_on_label(self, img_gray: np.ndarray):
        """รับภาพ numpy (view หลัง crop+gamma) แล้วแสดงบน label"""
        if img_gray is None:
            return
        self.current_display_image = img_gray.copy()

        h, w = img_gray.shape
        qimg = QImage(img_gray.data, w, h, w, QImage.Format_Grayscale8).copy()
        self.image_label.set_image(qimg)

    # -------------------- pipeline อัปเดต view --------------------

    def update_view(self):
        if self.original_image is None:
            return

        # 1) base image = processed หรือ original
        base = self.processed_image if self.processed_image is not None else self.original_image
        img = base.copy()

        # 2) gamma correction (ทั้งภาพหรือเฉพาะ ROI)
        if not np.isclose(self.gamma, 1.0):
            img_norm = img.astype(np.float32) / 255.0
            corrected = np.power(img_norm, 1.0 / self.gamma)
            corrected = np.clip(corrected * 255.0, 0, 255).astype(np.uint8)

            if self.roi_mask is not None:
                img[self.roi_mask] = corrected[self.roi_mask]
            else:
                img = corrected

        # 3) crop ตาม crop_rect (บนพิกัดภาพเต็ม)
        h_full, w_full = img.shape
        if self.crop_rect is None:
            self.crop_rect = (0, 0, w_full, h_full)

        x, y, w, h = self.crop_rect
        x = max(0, min(x, w_full - 1))
        y = max(0, min(y, h_full - 1))
        w = max(1, min(w, w_full - x))
        h = max(1, min(h, h_full - y))

        self.crop_rect = (x, y, w, h)
        self.view_x0 = x
        self.view_y0 = y
        self.view_w = w
        self.view_h = h

        view = img[y:y + h, x:x + w]
        self.show_image_on_label(view)

    # -------------------- โหลดรูป --------------------

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "Open image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not fname:
            return

        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return

        self.original_image = img
        h, w = img.shape
        self.crop_rect = (0, 0, w, h)

        # reset processing ทั้งหมด
        self.processed_image = None
        self.roi_mask = None
        self.gamma = 1.0
        self.gamma_slider.setValue(100)
        self.kernel_slider.setValue(35)
        self.radius_slider.setValue(3)

        self.update_view()

    # -------------------- Start Process: ใช้โค้ดลบซี่โครง --------------------

    def start_process(self):
        if self.original_image is None:
            return

        img = self.original_image

        # -------- 1) equalize histogram --------
        img_eq = cv2.equalizeHist(img)

        # -------- 2) สร้าง mask กระดูกทั้งหมด --------
        _, bone = cv2.threshold(
            img_eq, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bone = cv2.morphologyEx(bone, cv2.MORPH_OPEN, kernel_small, iterations=1)

        # -------- 3) หา mask กระดูกสันหลัง --------
        h, w = bone.shape
        x1 = int(w * 0.25)
        x2 = int(w * 0.75)
        center_roi = bone[:, x1:x2]

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(center_roi)

        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            spine_roi_mask = (labels == largest_label).astype(np.uint8) * 255
        else:
            spine_roi_mask = np.zeros_like(center_roi, dtype=np.uint8)

        spine_mask = np.zeros_like(bone, dtype=np.uint8)
        spine_mask[:, x1:x2] = spine_roi_mask

        spine_mask = cv2.dilate(spine_mask, kernel_small, iterations=10)

        # -------- 4) หา mask ซี่โครง --------
        # ใช้ kernel สูง (3, rib_kernel_height)
        k_h = self.rib_kernel_height
        if k_h < 3:
            k_h = 3
        # ทำให้เป็นเลขคี่ เพื่อให้สมมาตร
        if k_h % 2 == 0:
            k_h += 1

        kernel_rib = cv2.getStructuringElement(cv2.MORPH_RECT, (3, k_h))
        rib_lin = cv2.morphologyEx(bone, cv2.MORPH_TOPHAT, kernel_rib)
        rib_lin = cv2.morphologyEx(rib_lin, cv2.MORPH_CLOSE, kernel_rib, iterations=1)

        _, rib_mask = cv2.threshold(
            rib_lin, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        rib_mask_no_spine = cv2.subtract(
            rib_mask,
            cv2.bitwise_and(rib_mask, spine_mask)
        )

        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        rib_mask_no_spine = cv2.dilate(rib_mask_no_spine, kernel_dilate, iterations=1)

        mask_u8 = rib_mask_no_spine.astype(np.uint8)
        mask_u8[mask_u8 > 0] = 255

        # ถ้ามี ROI ให้ประมวลผลเฉพาะใน ROI
        if self.roi_mask is not None and self.roi_mask.shape == mask_u8.shape:
            mask_u8[~self.roi_mask] = 0

        # -------- 5) Inpaint ลบซี่โครง --------
        radius = max(1, int(self.inpaint_radius))
        result = cv2.inpaint(img_eq, mask_u8, radius, cv2.INPAINT_TELEA)

        # เก็บเป็น processed image (ใช้แทน original)
        self.processed_image = result
        self.update_view()

    # -------------------- Reset / Crop / ROI --------------------

    def reset_crop(self):
        if self.original_image is None:
            return
        h, w = self.original_image.shape
        self.crop_rect = (0, 0, w, h)
        self.update_view()

    def reset_processing(self):
        if self.original_image is None:
            return
        self.processed_image = None
        self.roi_mask = None

        self.gamma = 1.0
        self.gamma_slider.setValue(100)

        self.rib_kernel_height = 35
        self.kernel_slider.setValue(35)

        self.inpaint_radius = 3
        self.radius_slider.setValue(3)

        self.update_view()

    def start_crop_mode(self):
        if self.original_image is None:
            return
        self.image_label.set_mode("crop")

    def start_polygon_mode(self):
        if self.original_image is None:
            return
        self.image_label.set_mode("polygon")

    # rectSelected: พิกัดอยู่บน "ภาพ view" (ที่ถูก crop แล้ว)
    def on_rect_selected(self, rect_view: QRect):
        if self.original_image is None:
            return

        # แปลง rect view -> rect บนภาพเต็ม
        gx = self.view_x0 + rect_view.left()
        gy = self.view_y0 + rect_view.top()
        gw = rect_view.width()
        gh = rect_view.height()

        self.crop_rect = (gx, gy, gw, gh)
        self.update_view()

    # polygonFinished: list[QPoint] พิกัดบนภาพ view
    def on_polygon_finished(self, points_view):
        if self.original_image is None:
            return
        if len(points_view) < 3:
            return

        pts = []
        for p in points_view:
            x_full = self.view_x0 + p.x()
            y_full = self.view_y0 + p.y()
            pts.append([x_full, y_full])

        pts = np.array(pts, dtype=np.int32)

        mask = np.zeros_like(self.original_image, dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        self.roi_mask = mask.astype(bool)

        self.update_view()

    # -------------------- การเปลี่ยนค่า parameter --------------------

    def on_gamma_change(self, value: int):
        if self.original_image is None:
            return
        self.gamma = value / 100.0
        self.gamma_label.setText(f"Gamma: {self.gamma:.2f}")
        self.update_view()

    def on_kernel_change(self, value: int):
        self.rib_kernel_height = value
        self.kernel_label.setText(f"Rib kernel height: {value}")

    def on_radius_change(self, value: int):
        self.inpaint_radius = value
        self.radius_label.setText(f"Inpaint radius: {value}")

    # -------------------- Save --------------------

    def save_image(self):
        if self.current_display_image is None:
            return
        fname, _ = QFileDialog.getSaveFileName(
            self, "Save image", "", "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp)"
        )
        if not fname:
            return
        cv2.imwrite(fname, self.current_display_image)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
