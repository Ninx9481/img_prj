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

        self.crop_coords = None  # (y1, y2, x1, x2)

        # polygon ROI (สำหรับจำกัดพื้นที่ image processing)
        self.roi_polygon = None       # np.ndarray Nx2 (x, y)
        self.roi_mask = None          # mask 0/255 รูปเดียวกับ original

        print(f"✓ Image loaded successfully: {self.width}x{self.height} pixels")

    # -------------------------------------------------
    # 0) Auto-crop ช่วงปอด (ใช้สัดส่วนของภาพทั้งใบ)
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

    def reset_to_cropped(self):
        """ใช้ภาพในกรอบ crop ถ้ามี ไม่งั้นใช้ภาพเต็ม"""
        if self.crop_coords is not None:
            y1, y2, x1, x2 = self.crop_coords
            self.current_image = self.original_image[y1:y2, x1:x2].copy()
        else:
            self.current_image = self.original_image.copy()

    # -------------------------------------------------
    # ROI polygon (ใช้จำกัดบริเวณประมวลผล แต่ไม่ครอป)
    # -------------------------------------------------
    def set_roi_polygon(self, points_xy):
        """
        points_xy: list[(x,y)] ในพิกเซลของภาพเต็ม
        """
        if len(points_xy) < 3:
            self.roi_polygon = None
            self.roi_mask = None
            print("✗ ROI polygon requires at least 3 points")
            return

        pts = np.array(points_xy, dtype=np.int32)
        mask = np.zeros_like(self.original_image, dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)

        self.roi_polygon = pts
        self.roi_mask = mask

        print(f"✓ ROI polygon updated with {len(points_xy)} points")

    def clear_roi_polygon(self):
        self.roi_polygon = None
        self.roi_mask = None
        print("✓ ROI polygon cleared")

    def _update_with_roi(self, new_img):
        """
        ถ้ามี roi_mask และไม่ได้ crop: แทนที่เฉพาะในบริเวณ ROI
        ถ้าไม่มี: ใช้ทั้งภาพ
        """
        if self.roi_mask is not None and self.crop_coords is None:
            base = self.current_image.copy()
            roi = self.roi_mask
            base[roi == 255] = new_img[roi == 255]
            self.current_image = base
        else:
            self.current_image = new_img

        self.height, self.width = self.current_image.shape
        return self.current_image

    # -------------------------------------------------
    # 2) Enhancement
    # -------------------------------------------------
    def clahe_equalization(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        """ใช้ CLAHE (local histogram)"""
        img = self.current_image
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced = clahe.apply(img)
        print("✓ CLAHE (local histogram) completed successfully")
        return self._update_with_roi(enhanced)

    def histogram_equalization_global(self):
        """global hist eq (ถ้าอยากลองแบบง่าย)"""
        img = self.current_image
        enhanced = cv2.equalizeHist(img)
        print("✓ Global histogram equalization completed")
        return self._update_with_roi(enhanced)

    def denoise_median_bilateral_gaussian(
        self,
        ksize_median: int = 3,
        bilateral_d: int = 9,
        bilateral_sigma_color: float = 75,
        bilateral_sigma_space: float = 75,
        ksize_gauss: int = 5,
    ):
        """
        median -> bilateral -> Gaussian chain
        """
        img = self.current_image

        if ksize_median < 3:
            ksize_median = 3
        if ksize_median % 2 == 0:
            ksize_median += 1

        if ksize_gauss < 3:
            ksize_gauss = 3
        if ksize_gauss % 2 == 0:
            ksize_gauss += 1

        med = cv2.medianBlur(img, ksize_median)
        bil = cv2.bilateralFilter(
            med, d=bilateral_d, sigmaColor=bilateral_sigma_color, sigmaSpace=bilateral_sigma_space
        )
        gauss = cv2.GaussianBlur(bil, (ksize_gauss, ksize_gauss), 0)

        print(
            f"✓ Denoise (median k={ksize_median}, bilateral d={bilateral_d}, Gaussian k={ksize_gauss})"
        )
        return self._update_with_roi(gauss)

    # -------------------------------------------------
    # 3) Thresholding / Bone segmentation
    # -------------------------------------------------
    def segment_bones_otsu(self, img=None):
        """Otsu บนภาพ (img) ถ้าไม่ส่งจะใช้ current_image"""
        if img is None:
            img = self.current_image
        _, bone_mask = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        print("✓ Bone mask created successfully using Otsu's method")
        return bone_mask

    # -------------------------------------------------
    # Helper: opening by reconstruction (approx)
    # -------------------------------------------------
    def _opening_by_reconstruction(self, src, kernel, iterations=1):
        """Approximate opening-by-reconstruction ด้วย erosion ตามด้วย dilation constrained"""
        eroded = cv2.erode(src, kernel, iterations=iterations)
        reconstructed = cv2.dilate(eroded, kernel, iterations=iterations)
        reconstructed[reconstructed > 0] = 255
        return reconstructed

    # -------------------------------------------------
    # 4) ลบซี่โครงด้วย Morphology + refined ribs mask
    # -------------------------------------------------
    def remove_ribs_morphology(
        self,
        smooth_kernel: int = 5,
        rib_length: int = 40,
        rib_thickness: int = 3,
        spine_width_ratio: float = 0.25,
    ):
        """
        ใช้ภาพ self.current_image (ซึ่งก่อนหน้านี้อาจผ่าน CLAHE + denoise แล้ว)
        เพื่อ:
          1) สร้าง bone mask ด้วย Otsu
          2) หา rib mask ด้วย white top-hat หลายมุม + distance transform ให้เป็นแถบบาง
          3) blend soft tissue ใต้ ribs_mask กลับเข้าไปในภาพเดิม
        """
        img_for_seg = self.current_image.copy()

        # --- smooth ก่อนหา bone ---
        if smooth_kernel < 3:
            smooth_kernel = 3
        if smooth_kernel % 2 == 0:
            smooth_kernel += 1

        img_blur = cv2.GaussianBlur(img_for_seg, (smooth_kernel, smooth_kernel), 0)
        print(f"✓ Prepare the image for segmentation using Gaussian (k={smooth_kernel})")

        # --- bone mask ด้วย Otsu ---
        _, bone_mask = cv2.threshold(
            img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bone_mask_clean = cv2.morphologyEx(
            bone_mask, cv2.MORPH_CLOSE, kernel_small, iterations=1
        )
        print("✓ Refine the bone mask using morphological closing")

        # ถ้ามี ROI และไม่ crop: จำกัด bone mask เฉพาะใน ROI
        if self.roi_mask is not None and self.crop_coords is None:
            bone_mask_clean = cv2.bitwise_and(bone_mask_clean, self.roi_mask)

        # --- สร้าง ribs mask ด้วย white top-hat หลายมุม ---
        if rib_length < 5:
            rib_length = 5
        if rib_thickness < 1:
            rib_thickness = 1

        def make_line_kernel(length, thickness, angle_deg):
            size = max(length, thickness) + 2
            k = np.zeros((size, size), dtype=np.uint8)
            cv2.line(
                k,
                (0, size // 2),
                (size - 1, size // 2),
                255,
                thickness,
            )
            M = cv2.getRotationMatrix2D((size / 2, size / 2), angle_deg, 1.0)
            k_rot = cv2.warpAffine(k, M, (size, size))
            k_rot[k_rot > 0] = 255
            return k_rot

        angles = [0, 20, -20, 40, -40, 60, -60]
        tophat_combined = np.zeros_like(img_blur, dtype=np.uint8)

        for ang in angles:
            kernel_rib = make_line_kernel(int(rib_length), int(rib_thickness), ang)
            th = cv2.morphologyEx(img_blur, cv2.MORPH_TOPHAT, kernel_rib)
            tophat_combined = np.maximum(tophat_combined, th)

        _, ribs_mask_raw = cv2.threshold(
            tophat_combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # จำกัดเฉพาะส่วนที่ทับกับ bone_mask_clean
        ribs_mask_raw = cv2.bitwise_and(ribs_mask_raw, bone_mask_clean)

        # spine protection
        h, w = ribs_mask_raw.shape
        center_x = w // 2
        spine_half = int(w * spine_width_ratio / 2.0)
        left = max(center_x - spine_half, 0)
        right = min(center_x + spine_half, w)
        ribs_mask_raw[:, left:right] = 0

        # จำกัดใน ROI ถ้ามี (และไม่ crop)
        if self.roi_mask is not None and self.crop_coords is None:
            ribs_mask_raw = cv2.bitwise_and(ribs_mask_raw, self.roi_mask)

        # --- refine rib mask ให้เป็นแถบแคบตามแนวซี่โครง ---
        dist = cv2.distanceTransform(ribs_mask_raw, cv2.DIST_L2, 3)

        center_thresh = max(1.0, rib_thickness / 2.0)
        _, ribs_center = cv2.threshold(dist, center_thresh, 255, cv2.THRESH_BINARY)
        ribs_center = np.uint8(ribs_center)

        dilate_iter = max(1, rib_thickness // 2)
        ribs_mask_thin = cv2.dilate(
            ribs_center, np.ones((3, 3), np.uint8), iterations=dilate_iter
        )

        ribs_mask = ribs_mask_thin
        print("✓ Ribs mask refined to thin bands along ribs")

        bones_without_ribs_mask = cv2.subtract(bone_mask_clean, ribs_mask)
        bones_without_ribs_mask = self._opening_by_reconstruction(
            bones_without_ribs_mask, kernel_small
        )

        # --- สร้างผลลัพธ์: blend soft tissue + ภาพเดิม เฉพาะใต้ ribs_mask ---
        soft_tissue_img = img_blur
        result = self.current_image.copy()

        alpha = 0.8  # 0 = ใช้ soft tissue ล้วน, 1 = ใช้ภาพเดิม
        blend = (alpha * soft_tissue_img + (1 - alpha) * result).astype(np.uint8)
        result[ribs_mask == 255] = blend[ribs_mask == 255]

        self._update_with_roi(result)
        print("✓ Ribs suppressed using thin mask + soft-tissue blending")

        return self.current_image, bone_mask_clean, ribs_mask, bones_without_ribs_mask

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
        return self.original_image

    def draw_roi_outline_on(self, img, color=255, thickness=2):
        """วาดเส้น polygon ROI ลงบน img (in-place)"""
        if self.roi_polygon is None:
            return img
        pts = self.roi_polygon.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
        return img


# =========================================
#        CustomTkinter GUI (modern)
# =========================================
class XrayProcessorGUI:
    def __init__(self, root: ctk.CTk):
        self.root = root
        self.root.title("X-ray Image Processing")

        self.processor = None
        self.image_canvas = None
        self.tk_image = None
        self.display_scale = 1.0

        # crop / ROI mode
        self.crop_method = tk.StringVar(value="none")  # none / auto / manual
        self.roi_mode = tk.BooleanVar(value=False)

        # canvas state
        self.crop_start = None
        self.crop_rect = None

        # ROI points (ในพิกเซลของภาพเต็ม)
        self.roi_points = []          # list[(x,y)]
        self.roi_preview_items = []   # canvas item ids

        # ---------- main frame ใช้ grid ซ้าย-ขวา ----------
        self.main_frame = ctk.CTkFrame(
            root, corner_radius=0, fg_color="transparent"
        )
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # row 0 = toolbar ด้านบน, row 1 = แผงซ้าย/ขวา
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=3)  # ซ้ายกว้างกว่า
        self.main_frame.grid_columnconfigure(1, weight=1)  # ขวา = พาเนลพารามิเตอร์

        self.left_panel = None
        self.right_panel = None

        self.status_label = None

        self._build_gui()

    # ---------- UI หลัก ----------
    def _build_gui(self):
        # ==========================
        #  Toolbar ด้านบน (ปุ่มต่าง ๆ)
        # ==========================
        toolbar = ctk.CTkFrame(
            self.main_frame,
            corner_radius=0,
            fg_color=("white", "#1E1E1E"),
        )
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        toolbar.grid_columnconfigure(0, weight=1)

        # Title ซ้ายบน
        title_label = ctk.CTkLabel(
            toolbar,
            text="X-ray Image Processing",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        title_label.grid(row=0, column=0, padx=10, pady=6, sticky="w")

        # แถวปุ่มด้านขวาของ toolbar
        button_row = ctk.CTkFrame(toolbar, fg_color="transparent")
        button_row.grid(row=0, column=1, padx=10, pady=6, sticky="e")

        # ปุ่มต่าง ๆ อยู่ด้านบน
        select_btn = ctk.CTkButton(
            button_row,
            text="Select image",
            command=self.load_image,
            width=120,
            height=30,
            corner_radius=18,
        )
        select_btn.pack(side="left", padx=4)

        process_btn = ctk.CTkButton(
            button_row,
            text="Process image",
            command=self.process_image,
            width=130,
            height=30,
            corner_radius=18,
        )
        process_btn.pack(side="left", padx=4)

        reset_btn = ctk.CTkButton(
            button_row,
            text="Reset processing",
            command=self.reset_processing,
            width=130,
            height=30,
            corner_radius=18,
            fg_color="#E5E5EA",
            hover_color="#D1D1D6",
            text_color="#1D1D1F",
        )
        reset_btn.pack(side="left", padx=4)

        save_btn = ctk.CTkButton(
            button_row,
            text="Save result",
            command=self.save_result,
            width=120,
            height=30,
            corner_radius=18,
        )
        save_btn.pack(side="left", padx=4)

        # ==========================
        #  แพเนลซ้าย: แสดงภาพ + สถานะ
        # ==========================
        self.left_panel = ctk.CTkFrame(
            self.main_frame,
            corner_radius=16,
            fg_color=("white", "#1E1E1E"),
        )
        self.left_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 8))

        self.image_canvas = tk.Canvas(
            self.left_panel,
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

        self.status_label = ctk.CTkLabel(
            self.left_panel,
            text="Please select an X-ray image",
            font=ctk.CTkFont(size=11),
            text_color="#8E8E93",
        )
        self.status_label.pack(pady=(0, 10))

        # ==========================
        #  แพเนลขวา: Crop/ROI + พารามิเตอร์
        # ==========================
        self.right_panel = ctk.CTkFrame(
            self.main_frame,
            corner_radius=12,
            fg_color=("white", "#1E1E1E"),
        )
        self.right_panel.grid(row=1, column=1, sticky="ns")

        # ----- Crop & ROI settings -----
        crop_title = ctk.CTkLabel(
            self.right_panel,
            text="Crop & ROI Settings",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        crop_title.pack(pady=(10, 4))

        crop_frame = ctk.CTkFrame(
            self.right_panel,
            corner_radius=12,
            fg_color=("white", "#1E1E1E"),
        )
        crop_frame.pack(fill="x", padx=6, pady=(0, 8))

        radio_row = ctk.CTkFrame(crop_frame, fg_color="transparent")
        radio_row.pack(pady=(8, 0), fill="x")

        radio_font = ctk.CTkFont(size=12)

        ctk.CTkRadioButton(
            radio_row,
            text="No crop",
            variable=self.crop_method,
            value="none",
            height=24,
            radiobutton_width=14,
            radiobutton_height=14,
            font=radio_font,
        ).pack(side="left", padx=8)

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
            text="Manual crop (drag rect)",
            variable=self.crop_method,
            value="manual",
            height=24,
            radiobutton_width=14,
            radiobutton_height=14,
            font=radio_font,
        ).pack(side="left", padx=8)

        roi_row = ctk.CTkFrame(crop_frame, fg_color="transparent")
        roi_row.pack(pady=(4, 4), fill="x")

        ctk.CTkCheckBox(
            roi_row,
            text="Polygon ROI (click points on image)",
            variable=self.roi_mode,
            onvalue=True,
            offvalue=False,
        ).pack(side="left", padx=8)

        crop_btn_row = ctk.CTkFrame(crop_frame, fg_color="transparent")
        crop_btn_row.pack(pady=(4, 10))

        ctk.CTkButton(
            crop_btn_row,
            text="Apply crop",
            command=self.apply_crop,
            width=115,
            height=30,
            corner_radius=16,
            fg_color="#007AFF",
            hover_color="#005BBB",
        ).pack(side="left", padx=4)

        ctk.CTkButton(
            crop_btn_row,
            text="Reset crop/ROI",
            command=self.reset_crop_and_roi,
            width=135,
            height=30,
            corner_radius=16,
            fg_color="#E5E5EA",
            hover_color="#D1D1D6",
            text_color="#1D1D1F",
        ).pack(side="left", padx=4)

        # ----- Processing parameters -----
        param_title = ctk.CTkLabel(
            self.right_panel,
            text="Processing Parameters",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        param_title.pack(pady=(8, 4))

        param_card = ctk.CTkFrame(
            self.right_panel,
            corner_radius=12,
            fg_color=("white", "#1E1E1E"),
        )
        param_card.pack(fill="x", padx=6, pady=(0, 10))

        # ค่า default
        self.params_default = {
            "use_hist_eq": True,
            "smooth_kernel": 5,
            "rib_length": 40,
            "rib_thickness": 3,
            "spine_width_percent": 25,
        }

        self.params = {
            "use_hist_eq": tk.BooleanVar(value=self.params_default["use_hist_eq"]),
            "smooth_kernel": tk.IntVar(value=self.params_default["smooth_kernel"]),
            "rib_length": tk.IntVar(value=self.params_default["rib_length"]),
            "rib_thickness": tk.IntVar(value=self.params_default["rib_thickness"]),
            "spine_width_percent": tk.IntVar(value=self.params_default["spine_width_percent"]),
        }

        # top row: hist eq + reset button
        top_row = ctk.CTkFrame(param_card, fg_color="transparent")
        top_row.pack(fill="x", padx=10, pady=(8, 4))

        ctk.CTkCheckBox(
            top_row,
            text="Use Histogram Equalization (CLAHE)",
            variable=self.params["use_hist_eq"],
            onvalue=True,
            offvalue=False,
        ).pack(side="left")

        ctk.CTkButton(
            top_row,
            text="Reset params",
            width=100,
            height=26,
            corner_radius=14,
            fg_color="#E5E5EA",
            hover_color="#D1D1D6",
            text_color="#1D1D1F",
            command=self.reset_params,
        ).pack(side="right")

        # slider helper
        def slider_callback(value, var, value_label):
            iv = int(round(float(value)))
            var.set(iv)
            value_label.configure(text=str(iv))

        def add_slider(parent, label_text, var, frm, to):
            row = ctk.CTkFrame(parent, fg_color="transparent")
            row.pack(fill="x", padx=10, pady=(4, 4))

            top_row2 = ctk.CTkFrame(row, fg_color="transparent")
            top_row2.pack(fill="x")

            ctk.CTkLabel(top_row2, text=label_text, anchor="w").pack(side="left")

            value_label = ctk.CTkLabel(
                top_row2, text=str(var.get()), anchor="e", width=40
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

        add_slider(
            param_card,
            "Gaussian kernel size (blur before segmentation)",
            self.params["smooth_kernel"],
            3,
            21,
        )
        add_slider(
            param_card,
            "Structuring element length for ribs",
            self.params["rib_length"],
            10,
            120,
        )
        add_slider(
            param_card,
            "Structuring element thickness for ribs",
            self.params["rib_thickness"],
            1,
            9,
        )
        add_slider(
            param_card,
            "Spine region width (%)",
            self.params["spine_width_percent"],
            10,
            40,
        )

    # ===== helper แสดงภาพบน canvas =====
    def _draw_roi_preview_on_canvas(self):
        """วาดจุด/เส้น ROI ลงใน canvas จาก self.roi_points"""
        for item in self.roi_preview_items:
            self.image_canvas.delete(item)
        self.roi_preview_items.clear()

        if not self.roi_points or self.display_scale <= 0:
            return

        scaled_pts = [
            (int(x * self.display_scale), int(y * self.display_scale))
            for (x, y) in self.roi_points
        ]

        # วาดจุด
        for (sx, sy) in scaled_pts:
            r = 3
            item = self.image_canvas.create_oval(
                sx - r, sy - r, sx + r, sy + r,
                outline="white", fill="white"
            )
            self.roi_preview_items.append(item)

        # วาดเส้นต่อกัน
        for i in range(1, len(scaled_pts)):
            x0, y0 = scaled_pts[i - 1]
            x1, y1 = scaled_pts[i]
            item = self.image_canvas.create_line(
                x0, y0, x1, y1,
                fill="white", width=2
            )
            self.roi_preview_items.append(item)

        # polygon ปิดแล้ว: ต่อจุดสุดท้ายหากับจุดแรก
        if self.processor is not None and self.processor.roi_polygon is not None:
            x0, y0 = scaled_pts[-1]
            x1, y1 = scaled_pts[0]
            item = self.image_canvas.create_line(
                x0, y0, x1, y1,
                fill="white", width=2
            )
            self.roi_preview_items.append(item)

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

        self._draw_roi_preview_on_canvas()

    # ===== events manual crop + ROI =====
    def on_canvas_press(self, event):
        if self.processor is None:
            return

        # โหมด ROI: คลิกเพิ่มจุด
        if self.roi_mode.get():
            x_img = int(event.x / self.display_scale)
            y_img = int(event.y / self.display_scale)
            self.roi_points.append((x_img, y_img))
            self.processor.set_roi_polygon(self.roi_points)
            self._draw_roi_preview_on_canvas()
            self.status_label.configure(
                text=f"ROI point added: ({x_img}, {y_img})",
                text_color="#0A84FF",
            )
            return

        # manual crop rect
        if self.crop_method.get() == "manual":
            self.crop_start = (event.x, event.y)
            if self.crop_rect is not None:
                self.image_canvas.delete(self.crop_rect)
                self.crop_rect = None

    def on_canvas_drag(self, event):
        if self.processor is None:
            return
        if self.crop_start is None or self.crop_method.get() != "manual":
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
        if self.crop_start is None or self.crop_method.get() != "manual":
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
        self.processor.reset_to_cropped()

        self.show_image_on_canvas(self.processor.get_current_image())
        self.status_label.configure(text="Crop (manual rectangle) selected", text_color="#0A84FF")

    # ===== ปุ่ม Apply crop =====
    def apply_crop(self):
        if self.processor is None:
            messagebox.showwarning("Incomplete data", "Please select an image first")
            return

        mode = self.crop_method.get()
        if mode == "auto":
            try:
                self.processor.auto_crop_lung_region()
                self.show_image_on_canvas(self.processor.get_current_image())
                self.status_label.configure(text="Crop (auto lung region) applied", text_color="#0A84FF")
            except Exception as e:
                messagebox.showerror("Error", f"Auto-crop failed: {e}")
        elif mode == "none":
            self.processor.crop_coords = None
            self.processor.reset_to_cropped()
            self.show_image_on_canvas(self.processor.get_current_image())
            self.status_label.configure(
                text="Crop disabled (full image)", text_color="#8E8E93"
            )
        else:
            messagebox.showinfo(
                "Manual crop",
                "Drag a rectangle on the image (Manual crop mode) to select crop region.",
            )

    def reset_crop_and_roi(self):
        if self.processor is None:
            messagebox.showwarning("Incomplete data", "Please select an image first")
            return

        # reset crop
        self.processor.crop_coords = None
        self.processor.reset_to_cropped()
        self.crop_method.set("none")

        # reset ROI
        self.roi_points.clear()
        if self.processor is not None:
            self.processor.clear_roi_polygon()
        for item in self.roi_preview_items:
            self.image_canvas.delete(item)
        self.roi_preview_items.clear()

        self.show_image_on_canvas(self.processor.get_current_image())
        self.status_label.configure(
            text="Crop and ROI have been reset", text_color="#8E8E93"
        )

    # ===== reset params =====
    def reset_params(self):
        self.params["use_hist_eq"].set(self.params_default["use_hist_eq"])
        self.params["smooth_kernel"].set(self.params_default["smooth_kernel"])
        self.params["rib_length"].set(self.params_default["rib_length"])
        self.params["rib_thickness"].set(self.params_default["rib_thickness"])
        self.params["spine_width_percent"].set(self.params_default["spine_width_percent"])
        self.status_label.configure(
            text="Processing parameters reset to default", text_color="#8E8E93"
        )

    # ===== ปุ่ม Reset processing =====
    def reset_processing(self):
        if self.processor is None:
            messagebox.showwarning("Incomplete data", "Please select an image first")
            return

        self.processor.reset_to_cropped()
        self.show_image_on_canvas(self.processor.get_current_image())
        self.status_label.configure(
            text="Processing has been reset", text_color="#8E8E93"
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

            self.show_image_on_canvas(self.processor.get_original_image())
            self.processor.crop_coords = None
            self.crop_method.set("none")

            # reset ROI
            self.roi_points.clear()
            self.processor.clear_roi_polygon()
            for item in self.roi_preview_items:
                self.image_canvas.delete(item)
            self.roi_preview_items.clear()

        except Exception as e:
            messagebox.showerror("Error", f"Unable to load image: {e}")
            self.status_label.configure(
                text="Image loading failed", text_color="#FF3B30"
            )
            self.processor = None

    def _overlay_roi_if_any(self, img):
        if self.processor is None:
            return img
        out = img.copy()
        self.processor.draw_roi_outline_on(out, color=255, thickness=2)
        return out

    def process_image(self):
        if self.processor is None:
            messagebox.showwarning("Incomplete data", "Please select an image first")
            return

        try:
            # เริ่มจากภาพที่ crop แล้ว (ถ้ามี)
            self.processor.reset_to_cropped()

            # hist eq / CLAHE
            if self.params["use_hist_eq"].get():
                self.processor.clahe_equalization()

            # denoise
            smooth_k = self.params["smooth_kernel"].get()
            self.processor.denoise_median_bilateral_gaussian(
                ksize_median=3,
                bilateral_d=9,
                bilateral_sigma_color=75,
                bilateral_sigma_space=75,
                ksize_gauss=smooth_k,
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

            # แสดงเฉพาะผลสุดท้าย
            final_img = self._overlay_roi_if_any(result.copy())
            self.processor.current_image = final_img.copy()
            self.show_image_on_canvas(final_img)

            self.status_label.configure(
                text="Image processing completed", text_color="#34C759"
            )

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during processing: {e}")
            self.status_label.configure(text="Processing failed", text_color="#FF3B30")

    def save_result(self):
        if self.processor is None:
            messagebox.showwarning("Incomplete data", "No result to save")
            return

        output_dir = filedialog.askdirectory(title="Select a folder to save the result")
        if not output_dir:
            return

        try:
            self.processor.save_result(output_dir, "final_result.jpg")
            self.status_label.configure(
                text=f"Result saved to: {output_dir}", text_color="#34C759"
            )
        except Exception as e:
            messagebox.showerror("Error", f"Unable to save result: {e}")
            self.status_label.configure(
                text="Saving result failed", text_color="#FF3B30"
            )


# =========================================
#                   main
# =========================================
def main():
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")

    root = ctk.CTk(fg_color="#F2F2F7")
    # ปรับให้กว้างพอสำหรับ panel ขวา
    root.geometry("1100x700")
    root.resizable(False, False)

    app = XrayProcessorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
