import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image, ImageTk


# =========================================
#   XrayMonkeyProcessor (logic ประมวลผลภาพ)
# =========================================
class XrayMonkeyProcessor:
    def __init__(self, image_path: str):
        """โหลดภาพเอกซเรย์ (ระดับเทา)"""
        self.original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.original_image is None:
            raise ValueError(f"Unable to load image: {image_path}")
        self.current_image = self.original_image.copy()
        self.height, self.width = self.original_image.shape
        self.image_path = image_path
        self.crop_coords = None  # (y1, y2, x1, x2) สำหรับ crop แบบสี่เหลี่ยม
        self.roi_mask = None     # polygon ROI mask (uint8 0/255) สำหรับ “บริเวณที่จะทำ processing”
        print(f"✓ Image loaded successfully: {self.width}x{self.height} pixels")

    # -------------------------------------------------
    # 0) Auto-crop ช่วงปอด (ใช้สัดส่วนของภาพทั้งใบ) -> ใช้ “ครอปภาพ”
    # -------------------------------------------------
    def auto_crop_lung_region(
        self,
        x_start_ratio: float = 0.20,
        x_end_ratio: float = 0.85,
        y_start_ratio: float = 0.25,
        y_end_ratio: float = 0.80,
    ):
        h, w = self.original_image.shape

        x1 = int(max(0, min(w - 1, w * x_start_ratio)))
        x2 = int(max(0, min(w,     w * x_end_ratio)))
        y1 = int(max(0, min(h - 1, h * y_start_ratio)))
        y2 = int(max(0, min(h,     h * y_end_ratio)))

        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                "The ratio value for auto-crop is invalid, resulting in an empty selection"
            )

        self.crop_coords = (y1, y2, x1, x2)
        self.roi_mask = None  # ถ้าใช้ crop แบบสี่เหลี่ยม จะไม่ใช้ polygon ROI
        self.current_image = self.original_image[y1:y2, x1:x2].copy()
        self.height, self.width = self.current_image.shape

        print(
            f"✓ Automatic lung crop: x=({x1},{x2}), y=({y1},{y2}), size={self.width}x{self.height}"
        )
        return self.current_image

    # -------------------------------------------------
    # 1) manual crop ผ่าน matplotlib (เผื่ออยากใช้ภายหลัง)
    # -------------------------------------------------
    def manual_crop_interactive(self):
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(self.original_image, cmap="gray")
        ax.set_title(
            "Drag the mouse to select the torso/lung region, then release",
            fontsize=14,
        )

        self.crop_coords = None

        def on_select(eclick, erelease):
            x1 = int(min(eclick.xdata, erelease.xdata))
            y1 = int(min(eclick.ydata, erelease.ydata))
            x2 = int(max(eclick.xdata, erelease.xdata))
            y2 = int(max(eclick.ydata, erelease.ydata))

            h, w = self.original_image.shape
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h, y2))

            if x2 <= x1 or y2 <= y1:
                print("✗ Invalid crop region")
                return

            self.crop_coords = (y1, y2, x1, x2)
            print(f"✓ Select crop: ({x1}, {y1}) ถึง ({x2}, {y2})")
            plt.close()

        RectangleSelector(
            ax,
            on_select,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
        )

        plt.tight_layout()
        plt.show()

        if self.crop_coords is not None:
            y1, y2, x1, x2 = self.crop_coords
            self.current_image = self.original_image[y1:y2, x1:x2].copy()
            self.height, self.width = self.current_image.shape
            print(f"✓ Image cropping successful: {self.width}x{self.height}")
            return True
        else:
            print("✗ No crop selected")
            return False

    def set_polygon_roi(self, mask: np.ndarray):
        """
        กำหนด polygon ROI จาก mask (0/255)
        ใช้ mask เพื่อบอก "บริเวณที่จะทำ processing" แต่ไม่ได้ครอปภาพ
        """
        if mask.shape != self.original_image.shape:
            raise ValueError("ROI mask size does not match original image")
        self.roi_mask = mask
        # ใช้ polygon ROI กับภาพเต็มเสมอ -> ยกเลิก crop แบบสี่เหลี่ยม
        self.crop_coords = None

    def reset_to_cropped(self):
        """
        รีเซ็ต current_image:
        - ถ้ามี crop_coords -> ใช้ภาพที่ถูกครอป (แบบเดิม)
        - ถ้าไม่มี -> ใช้ภาพเต็มใบ
        - roi_mask แค่เป็น mask สำหรับจำกัดบริเวณตอน processing ไม่ได้เอามาใช้ครอป
        """
        if self.crop_coords is not None:
            y1, y2, x1, x2 = self.crop_coords
            self.current_image = self.original_image[y1:y2, x1:x2].copy()
        else:
            self.current_image = self.original_image.copy()

    # -------------------------------------------------
    # 2) Enhancement (ทำเฉพาะใน ROI ถ้ามี roi_mask)
    # -------------------------------------------------
    def histogram_equalization(self):
        if self.roi_mask is not None and self.crop_coords is None:
            # processing เฉพาะในบริเวณ ROI บนภาพเต็ม
            base = self.current_image.copy()
            eq = cv2.equalizeHist(base)
            base[self.roi_mask == 255] = eq[self.roi_mask == 255]
            self.current_image = base
        else:
            # ถ้าไม่มี ROI หรือกำลังทำกับภาพที่ถูกครอป -> ทำทั้งภาพเหมือนเดิม
            self.current_image = cv2.equalizeHist(self.current_image)

        print("✓ Histogram equalization completed successfully")
        return self.current_image

    def gaussian_smoothing(self, ksize: int = 5):
        if ksize < 3:
            ksize = 3
        if ksize % 2 == 0:
            ksize += 1

        blurred = cv2.GaussianBlur(self.current_image, (ksize, ksize), 0)

        if self.roi_mask is not None and self.crop_coords is None:
            # เบลอเฉพาะใน ROI
            base = self.current_image.copy()
            base[self.roi_mask == 255] = blurred[self.roi_mask == 255]
            self.current_image = base
        else:
            self.current_image = blurred

        print(f"✓ Gaussian smoothing (kernel={ksize}x{ksize})")
        return self.current_image

    # -------------------------------------------------
    # 3) Thresholding
    # -------------------------------------------------
    def segment_bones_otsu(self):
        _, bone_mask = cv2.threshold(
            self.current_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        print("✓ Bone mask created successfully using Otsu's method")
        return bone_mask

    # -------------------------------------------------
    # 4) ลบซี่โครงด้วย Morphology (จำกัดผลเฉพาะ ROI ถ้ามี)
    # -------------------------------------------------
    def remove_ribs_morphology(
        self,
        smooth_kernel: int = 5,
        rib_length: int = 40,
        rib_thickness: int = 3,
        spine_width_ratio: float = 0.25,
    ):
        img_for_seg = self.current_image.copy()
        if smooth_kernel < 3:
            smooth_kernel = 3
        if smooth_kernel % 2 == 0:
            smooth_kernel += 1
        img_blur = cv2.GaussianBlur(img_for_seg, (smooth_kernel, smooth_kernel), 0)
        print(f"✓ Prepare the image for segmentation using Gaussian (k={smooth_kernel})")

        _, bone_mask = cv2.threshold(
            img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bone_mask_clean = cv2.morphologyEx(
            bone_mask, cv2.MORPH_CLOSE, kernel_small, iterations=1
        )
        print("✓ Refine the bone mask using morphological closing")

        if rib_length < 5:
            rib_length = 5
        if rib_thickness < 1:
            rib_thickness = 1

        kernel_rib = cv2.getStructuringElement(
            cv2.MORPH_RECT, (int(rib_length), int(rib_thickness))
        )
        ribs_mask = cv2.morphologyEx(
            bone_mask_clean, cv2.MORPH_OPEN, kernel_rib, iterations=1
        )

        h, w = bone_mask_clean.shape
        center_x = w // 2
        spine_half = int(w * spine_width_ratio / 2.0)

        left = max(center_x - spine_half, 0)
        right = min(center_x + spine_half, w)
        ribs_mask[:, left:right] = 0
        print("✓ Create the ribs mask and restrict the lateral regions (to avoid the spine)")

        bones_without_ribs_mask = cv2.subtract(bone_mask_clean, ribs_mask)

        # ถ้ามี polygon ROI -> จำกัด mask ให้อยู่เฉพาะบริเวณนั้น
        if self.roi_mask is not None and self.crop_coords is None:
            roi = self.roi_mask
            bone_mask_clean = cv2.bitwise_and(bone_mask_clean, bone_mask_clean, mask=roi)
            ribs_mask = cv2.bitwise_and(ribs_mask, ribs_mask, mask=roi)
            bones_without_ribs_mask = cv2.bitwise_and(
                bones_without_ribs_mask, bones_without_ribs_mask, mask=roi
            )

        soft_tissue_img = img_blur
        result = self.current_image.copy()

        if self.roi_mask is not None and self.crop_coords is None:
            # แทนที่ค่าซี่โครงด้วย soft tissue เฉพาะใน ROI
            replace_mask = ribs_mask  # ตอนนี้ถูก AND กับ roi แล้ว
            result[replace_mask == 255] = soft_tissue_img[replace_mask == 255]
        else:
            # เคสเดิม (ไม่มี ROI หรือทำกับภาพที่ครอปแล้ว)
            result[ribs_mask == 255] = soft_tissue_img[ribs_mask == 255]

        self.current_image = result
        print("✓ Ribs removed (replaced with soft tissue) successfully")

        return result, bone_mask_clean, ribs_mask, bones_without_ribs_mask

    # -------------------------------------------------
    # Utils
    # -------------------------------------------------
    def save_result(self, output_path: str, filename: str = "processed_xray.jpg"):
        full_path = os.path.join(output_path, filename)
        cv2.imwrite(full_path, self.current_image)
        print(f"✓ Image saved successfully: {full_path}")
        return full_path

    def get_current_image(self):
        return self.current_image

    def get_original_image(self):
        # การแสดงภาพต้นฉบับ: ถ้ามี crop ใช้ภาพครอป, ถ้าไม่มีใช้ภาพเต็ม
        if self.crop_coords is not None:
            y1, y2, x1, x2 = self.crop_coords
            return self.original_image[y1:y2, x1:x2]
        return self.original_image


# =========================================
#        CustomTkinter GUI (modern)
# =========================================
class XrayProcessorGUI:
    def __init__(self, root: ctk.CTk):
        self.root = root
        self.root.title("X-ray Image Processing")

        self.processor = None
        self.step_images = []
        self.result_tk_images = []

        self.image_canvas = None
        self.tk_image = None
        self.display_scale = 1.0

        self.crop_method = tk.StringVar(value="auto")
        self.crop_start = None
        self.crop_rect = None

        # สำหรับ Polygon ROI
        self.polygon_points = []   # list ของจุดบน canvas
        self.polygon_items = []    # id ของ objects ที่วาดบน canvas

        self.results_frame = None

        # เก็บข้อมูล slider + label เพื่อใช้ตอน reset parameter
        self.slider_controls = {}  # key -> {"slider": slider, "label": label, "var": var}

        self.main_frame = ctk.CTkScrollableFrame(
            root, width=630, height=680, corner_radius=0, fg_color="transparent"
        )
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self._build_gui()

    # ---------- UI หลัก ----------
    def _build_gui(self):
        # Title
        title_label = ctk.CTkLabel(
            self.main_frame,
            text="X-ray Image",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        title_label.pack(pady=(0, 10))

        # Image card
        image_card = ctk.CTkFrame(
            self.main_frame,
            corner_radius=16,
            fg_color=("white", "#1E1E1E"),
        )
        image_card.pack(pady=5)

        self.image_canvas = tk.Canvas(
            image_card,
            width=512,
            height=512,
            bg="black",
            highlightthickness=0,
            bd=0,
        )
        self.image_canvas.pack(padx=10, pady=10)

        self.image_canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.image_canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.image_canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.image_canvas.bind("<Double-Button-1>", self.on_canvas_double_click)

        # Status
        self.status_label = ctk.CTkLabel(
            self.main_frame,
            text="Please select an X-ray image",
            font=ctk.CTkFont(size=11),
            text_color="#8E8E93",
        )
        self.status_label.pack(pady=(8, 6))

        # Select image button
        select_btn = ctk.CTkButton(
            self.main_frame,
            text="Select an X-ray image",
            command=self.load_image,
            width=200,
            height=34,
            corner_radius=18,
        )
        select_btn.pack(pady=(0, 16))

        # Crop settings
        crop_title = ctk.CTkLabel(
            self.main_frame,
            text="Crop Settings",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        crop_title.pack(pady=(4, 4))

        crop_frame = ctk.CTkFrame(
            self.main_frame,
            corner_radius=12,
            fg_color=("white", "#1E1E1E"),
        )
        crop_frame.pack(fill="x", padx=6, pady=(0, 8))

        radio_row = ctk.CTkFrame(crop_frame, fg_color="transparent")
        radio_row.pack(pady=(8, 4))

        radio_font = ctk.CTkFont(size=12)

        ctk.CTkRadioButton(
            radio_row,
            text="Auto crop lung region",
            variable=self.crop_method,
            value="auto",
            height=24,
            radiobutton_width=14,
            radiobutton_height=14,
            font=radio_font,
        ).pack(side="left", padx=8)

        ctk.CTkRadioButton(
            radio_row,
            text="Manual crop (drag on image)",
            variable=self.crop_method,
            value="manual",
            height=24,
            radiobutton_width=14,
            radiobutton_height=14,
            font=radio_font,
        ).pack(side="left", padx=8)

        ctk.CTkRadioButton(
            radio_row,
            text="Polygon ROI (click points)",
            variable=self.crop_method,
            value="polygon",
            height=24,
            radiobutton_width=14,
            radiobutton_height=14,
            font=radio_font,
        ).pack(side="left", padx=8)

        crop_btn_row = ctk.CTkFrame(crop_frame, fg_color="transparent")
        crop_btn_row.pack(pady=(4, 10))

        ctk.CTkButton(
            crop_btn_row,
            text="Apply crop",
            command=self.apply_crop,
            width=120,
            height=30,
            corner_radius=16,
            fg_color="#007AFF",
            hover_color="#005BBB",
        ).pack(side="left", padx=6)

        ctk.CTkButton(
            crop_btn_row,
            text="Reset crop",
            command=self.reset_crop,
            width=120,
            height=30,
            corner_radius=16,
            fg_color="#E5E5EA",
            hover_color="#D1D1D6",
            text_color="#1D1D1F",
        ).pack(side="left", padx=6)

        # Parameters
        param_title = ctk.CTkLabel(
            self.main_frame,
            text="Processing Parameters",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        param_title.pack(pady=(10, 4))

        param_card = ctk.CTkFrame(
            self.main_frame,
            corner_radius=12,
            fg_color=("white", "#1E1E1E"),
        )
        param_card.pack(fill="x", padx=6, pady=(0, 10))

        # ----- สร้างตัวแปรพารามิเตอร์ -----
        self.params = {
            "use_hist_eq": tk.BooleanVar(value=True),
            "smooth_kernel": tk.IntVar(value=5),
            "rib_length": tk.IntVar(value=40),
            "rib_thickness": tk.IntVar(value=3),
            "spine_width_percent": tk.IntVar(value=25),
        }

        # แถว Use Histogram Equalization + ปุ่ม Reset parameters (บรรทัดเดียวกัน)
        hist_row = ctk.CTkFrame(param_card, fg_color="transparent")
        hist_row.pack(fill="x", padx=10, pady=(8, 4))

        ctk.CTkLabel(
            hist_row,
            text="Use Histogram Equalization",
            anchor="w",
        ).pack(side="left", padx=(0, 10))

        ctk.CTkCheckBox(
            hist_row,
            text="",
            variable=self.params["use_hist_eq"],
            onvalue=True,
            offvalue=False,
            height=22,
            checkbox_width=14,
            checkbox_height=14,
        ).pack(side="left")

        reset_param_btn = ctk.CTkButton(
            hist_row,
            text="Reset params",
            command=self.reset_parameters,
            width=100,
            height=24,
            corner_radius=12,
            fg_color="#E5E5EA",
            hover_color="#D1D1D6",
            text_color="#1D1D1F",
        )
        reset_param_btn.pack(side="right")

        # slider helper
        def slider_callback(value, var, value_label):
            iv = int(round(float(value)))
            var.set(iv)
            value_label.configure(text=str(iv))

        def add_slider(parent, key, label_text, var, frm, to):
            row = ctk.CTkFrame(parent, fg_color="transparent")
            row.pack(fill="x", padx=10, pady=(4, 4))

            top_row = ctk.CTkFrame(row, fg_color="transparent")
            top_row.pack(fill="x")

            ctk.CTkLabel(top_row, text=label_text, anchor="w").pack(side="left")

            value_label = ctk.CTkLabel(
                top_row, text=str(var.get()), anchor="e", width=40
            )
            value_label.pack(side="right")

            slider = ctk.CTkSlider(
                row,
                from_=frm,
                to=to,
                number_of_steps=int(to - frm),
                command=lambda v: slider_callback(v, var, value_label),
            )
            slider.set(var.get())
            slider.pack(fill="x")

            # เก็บไว้ใช้ตอน reset_parameters
            self.slider_controls[key] = {
                "slider": slider,
                "label": value_label,
                "var": var,
            }

        add_slider(
            param_card,
            "smooth_kernel",
            "Gaussian kernel size (blur before segmentation)",
            self.params["smooth_kernel"],
            3,
            21,
        )
        add_slider(
            param_card,
            "rib_length",
            "Structuring element length for ribs",
            self.params["rib_length"],
            10,
            120,
        )
        add_slider(
            param_card,
            "rib_thickness",
            "Structuring element thickness for ribs",
            self.params["rib_thickness"],
            1,
            9,
        )
        add_slider(
            param_card,
            "spine_width_percent",
            "Spine region width (%)",
            self.params["spine_width_percent"],
            10,
            40,
        )

        # Action buttons
        action_row = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        action_row.pack(pady=14)

        ctk.CTkButton(
            action_row,
            text="Process image",
            command=self.process_image,
            width=160,
            height=34,
            corner_radius=18,
        ).pack(side="left", padx=6)

        ctk.CTkButton(
            action_row,
            text="Reset processing",
            command=self.reset_processing,
            width=160,
            height=34,
            corner_radius=18,
            fg_color="#E5E5EA",
            hover_color="#D1D1D6",
            text_color="#1D1D1F",
        ).pack(side="left", padx=6)

        ctk.CTkButton(
            action_row,
            text="Show results",
            command=self.show_results,
            width=160,
            height=34,
            corner_radius=18,
            fg_color="#FFD60A",
            hover_color="#E5C009",
            text_color="#1D1D1F",
        ).pack(side="left", padx=6)

        # Save button
        self.save_row = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.save_row.pack(pady=(4, 10))

        ctk.CTkButton(
            self.save_row,
            text="Save results",
            command=self.save_results,
            width=220,
            height=32,
            corner_radius=18,
        ).pack()

    # ===== helper แสดงภาพบน canvas =====
    def show_image_on_canvas(self, img_gray):
        if img_gray is None:
            return

        h, w = img_gray.shape
        max_size = 500
        scale = min(max_size / w, max_size / h, 1.0)
        disp_w, disp_h = int(w * scale), int(h * scale)

        img_resized = cv2.resize(img_gray, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(img_rgb)
        self.tk_image = ImageTk.PhotoImage(pil_img)

        self.image_canvas.config(width=disp_w, height=disp_h)
        self.image_canvas.delete("all")
        self.image_canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

        self.display_scale = scale
        self.crop_start = None
        self.crop_rect = None

        # ล้าง polygon วาดบน canvas
        self.polygon_points = []
        self.polygon_items = []

    # ===== helpers สำหรับ polygon ROI =====
    def clear_polygon(self):
        for item in self.polygon_items:
            self.image_canvas.delete(item)
        self.polygon_points = []
        self.polygon_items = []

    def add_polygon_point(self, x, y):
        r = 3
        dot = self.image_canvas.create_oval(x - r, y - r, x + r, y + r,
                                            fill="#FFD60A", outline="")
        self.polygon_items.append(dot)

        if self.polygon_points:
            x0, y0 = self.polygon_points[-1]
            line = self.image_canvas.create_line(x0, y0, x, y, fill="#FFD60A", width=2)
            self.polygon_items.append(line)

        self.polygon_points.append((x, y))

    def finalize_polygon(self):
        if self.processor is None or self.display_scale <= 0:
            return
        if len(self.polygon_points) < 3:
            messagebox.showwarning("Warning", "Please select at least 3 points for polygon ROI.")
            return

        # ปิด polygon บน canvas (แค่ให้เห็นตอนเลือก)
        x0, y0 = self.polygon_points[-1]
        x_first, y_first = self.polygon_points[0]
        line = self.image_canvas.create_line(x0, y0, x_first, y_first,
                                             fill="#FFD60A", width=2)
        self.polygon_items.append(line)

        # แปลงพิกัด canvas -> พิกัดภาพเต็ม
        pts = []
        for x, y in self.polygon_points:
            px = int(x / self.display_scale)
            py = int(y / self.display_scale)
            pts.append([px, py])

        h, w = self.processor.original_image.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)

        # กำหนด polygon ROI เป็นบริเวณที่จะทำ processing (ไม่ครอปภาพ)
        self.processor.set_polygon_roi(mask)

        # แสดงภาพเต็มตามปกติ (ยังไม่ตัดขอบนอก ROI ทิ้ง)
        self.show_image_on_canvas(self.processor.get_original_image())
        self.status_label.configure(text="ROI (polygon for processing) selected", text_color="#0A84FF")

    # ===== events manual / polygon crop =====
    def on_canvas_press(self, event):
        if self.processor is None:
            return

        method = self.crop_method.get()

        if method == "manual":
            self.crop_start = (event.x, event.y)
            if self.crop_rect is not None:
                self.image_canvas.delete(self.crop_rect)
                self.crop_rect = None
        elif method == "polygon":
            self.add_polygon_point(event.x, event.y)

    def on_canvas_drag(self, event):
        if self.processor is None:
            return
        if self.crop_method.get() != "manual":
            return
        if self.crop_start is None:
            return

        x0, y0 = self.crop_start
        x1, y1 = event.x, event.y
        if self.crop_rect is None:
            self.crop_rect = self.image_canvas.create_rectangle(
                x0, y0, x1, y1, outline="#0A84FF", width=2
            )
        else:
            self.image_canvas.coords(self.crop_rect, x0, y0, x1, y1)

    def on_canvas_release(self, event):
        if self.processor is None:
            return
        if self.crop_method.get() != "manual":
            return
        if self.crop_start is None:
            return

        x0, y0 = self.crop_start
        x1, y1 = event.x, event.y
        self.crop_start = None

        if self.display_scale <= 0:
            return

        x_min = int(min(x0, x1) / self.display_scale)
        x_max = int(max(x0, x1) / self.display_scale)
        y_min = int(min(y0, y1) / self.display_scale)
        y_max = int(max(y0, y1) / self.display_scale)

        h, w = self.processor.original_image.shape
        x_min = max(0, min(w - 1, x_min))
        x_max = max(0, min(w, x_max))
        y_min = max(0, min(h - 1, y_min))
        y_max = max(0, min(h, y_max))

        if x_max <= x_min or y_max <= y_min:
            messagebox.showwarning("Warning", "Invalid crop region")
            return

        self.processor.crop_coords = (y_min, y_max, x_min, x_max)
        self.processor.roi_mask = None  # ถ้าครอป ก็เลิกใช้ polygon ROI
        self.processor.reset_to_cropped()

        self.show_image_on_canvas(self.processor.get_current_image())
        self.status_label.configure(text="ROI (manual crop) selected", text_color="#0A84FF")

    def on_canvas_double_click(self, event):
        """จบการเลือก polygon ROI เมื่อ double-click"""
        if self.processor is None:
            return
        if self.crop_method.get() != "polygon":
            return
        self.finalize_polygon()

    # ===== ปุ่ม Apply crop (auto) =====
    def apply_crop(self):
        if self.processor is None:
            messagebox.showwarning("Incomplete data", "Please select an image first")
            return

        method = self.crop_method.get()

        if method == "auto":
            try:
                self.processor.auto_crop_lung_region()
                self.show_image_on_canvas(self.processor.get_current_image())
                self.status_label.configure(text="ROI (auto crop) selected", text_color="#0A84FF")
            except Exception as e:
                messagebox.showerror("Error", f"Auto-crop failed: {e}")
        elif method == "manual":
            messagebox.showinfo(
                "Manual crop",
                "Select 'Manual crop' then drag the mouse over the image to choose ROI.",
            )
        elif method == "polygon":
            messagebox.showinfo(
                "Polygon ROI",
                "Select 'Polygon ROI' then click points on the image.\nDouble-click to finish the polygon.\nProcessing will be limited to that region (image not cropped).",
            )

    # ===== ปุ่ม Reset crop =====
    def reset_crop(self):
        if self.processor is None:
            messagebox.showwarning("Incomplete data", "Please select an image first")
            return

        self.processor.crop_coords = None
        self.processor.roi_mask = None
        self.processor.reset_to_cropped()

        self.clear_polygon()
        self.show_image_on_canvas(self.processor.get_original_image())
        self.status_label.configure(
            text="Crop/ROI has been reset to full image", text_color="#8E8E93"
        )

    # ===== ปุ่ม Reset processing =====
    def reset_processing(self):
        if self.processor is None:
            messagebox.showwarning("Incomplete data", "Please select an image first")
            return

        self.processor.reset_to_cropped()
        self.step_images = []

        self.show_image_on_canvas(self.processor.get_current_image())
        self.status_label.configure(
            text="Processing has been reset", text_color="#8E8E93"
        )

        if self.results_frame is not None:
            self.results_frame.pack_forget()
            self.results_frame = None
        self.result_tk_images.clear()

    # ===== ปุ่ม Reset Parameters =====
    def reset_parameters(self):
        # ค่า default
        defaults = {
            "use_hist_eq": True,
            "smooth_kernel": 5,
            "rib_length": 40,
            "rib_thickness": 3,
            "spine_width_percent": 25,
        }

        # ตั้งค่าให้ตัวแปร
        self.params["use_hist_eq"].set(defaults["use_hist_eq"])

        for key in ["smooth_kernel", "rib_length", "rib_thickness", "spine_width_percent"]:
            value = defaults[key]
            self.params[key].set(value)
            ctrl = self.slider_controls.get(key)
            if ctrl is not None:
                ctrl["slider"].set(value)
                ctrl["label"].configure(text=str(value))

        self.status_label.configure(
            text="Parameters reset to default", text_color="#8E8E93"
        )

    # ===== callbacks หลัก =====
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select X-ray image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")],
        )
        if not file_path:
            return

        try:
            self.processor = XrayMonkeyProcessor(file_path)
            self.status_label.configure(
                text=f"Loaded: {os.path.basename(file_path)}", text_color="#34C759"
            )

            self.processor.crop_coords = None
            self.processor.roi_mask = None
            self.crop_method.set("auto")

            self.show_image_on_canvas(self.processor.get_original_image())
            self.step_images = []

            if self.results_frame is not None:
                self.results_frame.pack_forget()
                self.results_frame = None
            self.result_tk_images.clear()

        except Exception as e:
            messagebox.showerror("Error", f"Unable to load image: {e}")
            self.status_label.configure(
                text="Image loading failed", text_color="#FF3B30"
            )
            self.processor = None
            self.step_images = []

    def process_image(self):
        if self.processor is None:
            messagebox.showwarning("Incomplete data", "Please select an image first")
            return

        try:
            # เริ่มจากภาพตาม crop ปัจจุบัน (หรือเต็มภาพ)
            self.processor.reset_to_cropped()
            self.step_images = [
                ("Original / ROI", self.processor.get_current_image().copy())
            ]

            if self.params["use_hist_eq"].get():
                self.processor.histogram_equalization()
                self.step_images.append(
                    ("Histogram Equalization", self.processor.get_current_image().copy())
                )

            smooth_k = self.params["smooth_kernel"].get()
            self.processor.gaussian_smoothing(smooth_k)
            self.step_images.append(
                (f"Gaussian Blur (k={smooth_k})", self.processor.get_current_image().copy())
            )

            rib_len = self.params["rib_length"].get()
            rib_th = self.params["rib_thickness"].get()
            spine_ratio = self.params["spine_width_percent"].get() / 100.0

            result, bone_mask, ribs_mask, bones_no_ribs = (
                self.processor.remove_ribs_morphology(
                    smooth_kernel=smooth_k,
                    rib_length=rib_len,
                    rib_thickness=rib_th,
                    spine_width_ratio=spine_ratio,
                )
            )

            self.step_images.append(("Bone Mask (clean)", bone_mask))
            self.step_images.append(("Rib Mask", ribs_mask))
            self.step_images.append(("Bones without Ribs (mask)", bones_no_ribs))
            self.step_images.append(("Final Result", result.copy()))

            self.status_label.configure(
                text="Image processing completed", text_color="#34C759"
            )

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during processing: {e}")
            self.status_label.configure(text="Processing failed", text_color="#FF3B30")

    def show_results(self):
        if not self.step_images:
            messagebox.showwarning("Incomplete data", "Please process the image first")
            return

        if self.results_frame is None:
            self.results_frame = ctk.CTkFrame(
                self.main_frame,
                corner_radius=12,
                fg_color=("white", "#1E1E1E"),
            )
            self.results_frame.pack(
                before=self.save_row,
                pady=(0, 10),
                padx=12,
                fill="x",
            )

        for w in self.results_frame.winfo_children():
            w.destroy()
        self.result_tk_images.clear()

        max_width = 480

        for title, img in self.step_images:
            item_frame = ctk.CTkFrame(self.results_frame, fg_color="transparent")
            item_frame.pack(pady=6, fill="x")

            lbl = ctk.CTkLabel(
                item_frame,
                text=title,
                font=ctk.CTkFont(size=12, weight="bold"),
            )
            lbl.pack(pady=(6, 2))

            h, w = img.shape
            scale = min(max_width / w, 1.0)
            disp_w, disp_h = int(w * scale), int(h * scale)
            img_resized = cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
            pil_img = Image.fromarray(img_rgb)

            tk_img = ctk.CTkImage(light_image=pil_img, size=(disp_w, disp_h))
            self.result_tk_images.append(tk_img)

            img_label = ctk.CTkLabel(item_frame, image=tk_img, text="")
            img_label.pack(padx=20, pady=(0, 8))

    def save_results(self):
        if self.processor is None or not self.step_images:
            messagebox.showwarning("Incomplete data", "No results available to save")
            return

        output_dir = filedialog.askdirectory(title="Select a folder to save the results")
        if not output_dir:
            return

        try:
            for idx, (title, img) in enumerate(self.step_images):
                safe_title = title.replace(" ", "_").replace("/", "_")
                filename = f"step{idx:02d}_{safe_title}.jpg"
                full_path = os.path.join(output_dir, filename)
                cv2.imwrite(full_path, img)
                print(f"✓ Saved: {full_path}")

            self.processor.current_image = self.step_images[-1][1].copy()
            self.processor.save_result(output_dir, "final_result.jpg")

            self.status_label.configure(
                text=f"All results saved to: {output_dir}", text_color="#34C759"
            )

        except Exception as e:
            messagebox.showerror("Error", f"Unable to save results: {e}")
            self.status_label.configure(
                text="Saving results failed", text_color="#FF3B30"
            )


# =========================================
#                   main
# =========================================
def main():
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")

    root = ctk.CTk(fg_color="#F2F2F7")
    root.geometry("650x700")
    root.resizable(False, False)

    app = XrayProcessorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()