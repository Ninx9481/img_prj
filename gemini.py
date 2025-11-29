import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image, ImageTk
import math

# =========================================
#   AdvancedXrayProcessor (logic ประมวลผลภาพ)
# =========================================
class AdvancedXrayProcessor:
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
    # 0) Crop & ROI (เหมือนเดิม)
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
        self.roi_mask = None
        self.current_image = self.original_image[y1:y2, x1:x2].copy()
        self.height, self.width = self.current_image.shape

        print(
            f"✓ Automatic lung crop: x=({x1},{x2}), y=({y1},{y2}), size={self.width}x{self.height}"
        )
        return self.current_image

    def set_polygon_roi(self, mask: np.ndarray):
        if mask.shape != self.original_image.shape:
            raise ValueError("ROI mask size does not match original image")
        self.roi_mask = mask
        self.crop_coords = None

    def reset_to_cropped(self):
        if self.crop_coords is not None:
            y1, y2, x1, x2 = self.crop_coords
            self.current_image = self.original_image[y1:y2, x1:x2].copy()
        else:
            self.current_image = self.original_image.copy()
        self.height, self.width = self.current_image.shape

    # -------------------------------------------------
    # 1) New Enhancement: Homomorphic Filtering
    # -------------------------------------------------
    def _homomorphic_filter(self, img, d0=50.0, gamma_l=0.5, gamma_h=2.0, c=1.0):
        # 1. Log transform
        img_log = np.log1p(np.float64(img) + 1e-6)

        # 2. DFT (Fourier Transform)
        dft = cv2.dft(img_log, flags=cv2.DFT_COMPLEX_OUTPUT)

        # 3. Shift zero-frequency component to the center
        dft_shift = np.fft.fftshift(dft)

        # 4. Create Homomorphic Filter
        rows, cols = img.shape
        crow, ccol = rows // 2, cols // 2
        
        u, v = np.meshgrid(np.arange(cols), np.arange(rows))
        D_sq = (u - ccol)**2 + (v - crow)**2

        # Gaussian High-pass Filter
        H_uv = 1.0 - np.exp(-c * (D_sq / (d0**2)))
        
        # Homomorphic Filter H_homo(u,v) = (gamma_h - gamma_l) * H_uv + gamma_l
        H_homo = (gamma_h - gamma_l) * H_uv + gamma_l
        H_homo = H_homo[:, :, np.newaxis] # Reshape to 2-channel

        # 5. Apply the filter in frequency domain
        dft_filtered = dft_shift * H_homo

        # 6. Inverse Shift & 7. Inverse DFT
        dft_ishift = np.fft.ifftshift(dft_filtered)
        img_back = cv2.idft(dft_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        
        # 8. Exponential (Inverse Log) transform
        img_homo = np.expm1(img_back)

        # 9. Normalize and convert back to uint8
        img_homo = np.clip(img_homo, 0, 255)
        return np.uint8(img_homo)

    def apply_enhancement(self, method: str, params: dict):
        # *** โค้ดถูกแก้ไขให้รองรับเฉพาะ Homomorphic/Hist_eq/none
        # แต่ใน GUI จะเลือก Homomorphic เป็นค่าเริ่มต้นและบังคับใช้
        
        if method == "hist_eq":
            processed = cv2.equalizeHist(self.current_image)
            self.current_image = processed
            print("✓ Histogram equalization completed successfully")
        
        elif method == "homomorphic":
            d0 = params["homo_d0"]
            gl = params["homo_gamma_l"]
            gh = params["homo_gamma_h"]

            processed = self._homomorphic_filter(self.current_image, d0=d0, gamma_l=gl, gamma_h=gh)
            self.current_image = processed

            print(f"✓ Homomorphic Filtering completed (D0={d0}, GL={gl}, GH={gh})")

        elif method == "none":
            print("✓ No enhancement applied.")

        # Apply ROI mask if polygon ROI is used (only when not cropped)
        if self.roi_mask is not None and self.crop_coords is None:
            base = self.original_image.copy() # ใช้ภาพเดิมที่ไม่ถูก Enhace เป็นฐาน
            # ใช้ Mask กับภาพที่ถูก Enhace แล้ว
            base[self.roi_mask == 255] = self.current_image[self.roi_mask == 255]
            self.current_image = base
            print("✓ Enhancement limited to Polygon ROI area.")


        return self.current_image


    # -------------------------------------------------
    # 2) New Noise Reduction: Bilateral Filtering (Edge-Preserving)
    # -------------------------------------------------
    def apply_noise_reduction(self, method: str, params: dict):
        # *** โค้ดถูกแก้ไขให้รองรับเฉพาะ Bilateral/Gaussian/none
        # แต่ใน GUI จะเลือก Bilateral เป็นค่าเริ่มต้นและบังคับใช้
        
        if method == "bilateral":
            # Bilateral Filter: ลด Noise พร้อมรักษาขอบ
            d = int(params["bilateral_d"])
            sigma_c = int(params["bilateral_sigma_c"])
            sigma_s = int(params["bilateral_sigma_s"])

            blurred = cv2.bilateralFilter(self.current_image, d, sigma_c, sigma_s)
            
            self.current_image = blurred

            print(f"✓ Bilateral Filtering (d={d}, σc={sigma_c}, σs={sigma_s})")
        
        elif method == "gaussian":
            # Gaussian Smoothing
            ksize = 5 # Fixed for simplicity in this flow, GUI controls only for Bilateral
            blurred = cv2.GaussianBlur(self.current_image, (ksize, ksize), 0)
            self.current_image = blurred
            print(f"✓ Gaussian smoothing (kernel={ksize}x{ksize})")
        
        elif method == "none":
            print("✓ No noise reduction applied.")
        
        # Apply ROI mask if polygon ROI is used (only when not cropped)
        if self.roi_mask is not None and self.crop_coords is None and method != "none":
            # ต้องมั่นใจว่า current_image ตอนนี้คือผลลัพธ์ที่ถูก Enhance & Filtered เฉพาะใน ROI
            # ในโค้ดนี้เราทำ Enhancement ใน ROI ไปแล้ว ดังนั้นส่วนนี้ไม่จำเป็นต้องทำซ้ำ
            pass

        return self.current_image

    # -------------------------------------------------
    # 3) New Segmentation: Adaptive Thresholding (เหมือนเดิม)
    # -------------------------------------------------
    def segment_bones_adaptive(self, block_size: int = 11, c: int = 2):
        if block_size < 3: block_size = 3
        if block_size % 2 == 0: block_size += 1

        # ADAPTIVE_THRESH_GAUSSIAN_C ใช้ Local Weighted Mean (Gaussian)
        # THRESH_BINARY_INV ใช้เพื่อให้ส่วนที่สว่าง (กระดูก) เป็น 255
        bone_mask = cv2.adaptiveThreshold(
            self.current_image, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV,
            block_size, c
        )
        print(f"✓ Bone mask created using Adaptive Thresholding (Block={block_size}, C={c})")
        return bone_mask

    # -------------------------------------------------
    # 4) Canny Edge Detection (New Utility) (เหมือนเดิม)
    # -------------------------------------------------
    def get_canny_edge(self, threshold1: int = 50, threshold2: int = 150):
        # Canny ควรใช้กับภาพที่ผ่านการลด Noise แล้ว
        edges = cv2.Canny(self.current_image, threshold1, threshold2)
        print(f"✓ Canny Edge Detection completed (T1={threshold1}, T2={threshold2})")
        return edges

    # -------------------------------------------------
    # 5) Rib Removal (อัปเดตไปใช้ Adaptive Thresholding) (เหมือนเดิม)
    # -------------------------------------------------
    def remove_ribs_morphology(
        self,
        adapt_block_size: int = 11,
        adapt_c: int = 2,
        rib_length: int = 40,
        rib_thickness: int = 3,
        spine_width_ratio: float = 0.25,
    ):
        # 1. Image for Soft Tissue (ภาพที่ผ่าน Enhancement และ Noise Reduction มาแล้ว)
        soft_tissue_img = self.current_image.copy()

        # 2. Segmentation (ใช้ Adaptive Thresholding)
        bone_mask = self.segment_bones_adaptive(adapt_block_size, adapt_c)
        
        # 3. Refine Bone Mask (Morphological Closing)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bone_mask_clean = cv2.morphologyEx(
            bone_mask, cv2.MORPH_CLOSE, kernel_small, iterations=1
        )
        print("✓ Refine the bone mask using morphological closing")

        # 4. Extract Ribs Mask (Morphological Opening)
        if rib_length < 5: rib_length = 5
        if rib_thickness < 1: rib_thickness = 1

        kernel_rib = cv2.getStructuringElement(
            cv2.MORPH_RECT, (int(rib_length), int(rib_thickness))
        )
        ribs_mask = cv2.morphologyEx(
            bone_mask_clean, cv2.MORPH_OPEN, kernel_rib, iterations=1
        )

        # 5. Suppress Spine Region in Ribs Mask 
        h, w = bone_mask_clean.shape
        center_x = w // 2
        spine_half = int(w * spine_width_ratio / 2.0)

        left = max(center_x - spine_half, 0)
        right = min(center_x + spine_half, w)
        ribs_mask[:, left:right] = 0
        print("✓ Create the ribs mask and restrict the lateral regions (to avoid the spine)")

        # 6. Find Bones without Ribs Mask (Useful for visualization)
        bones_without_ribs_mask = cv2.subtract(bone_mask_clean, ribs_mask)

        # 7. Apply ROI (ถ้ามี Polygon ROI)
        if self.roi_mask is not None and self.crop_coords is None:
            # ใช้ ROI mask ที่มีขนาดเท่าภาพเต็ม
            roi = self.roi_mask
            bone_mask_clean = cv2.bitwise_and(self.original_image, bone_mask_clean, mask=roi)
            ribs_mask = cv2.bitwise_and(self.original_image, ribs_mask, mask=roi)

        # 8. Rib Suppression (แทนที่ค่าพิกเซล)
        result = self.current_image.copy()

        replace_mask = ribs_mask
        result[replace_mask == 255] = soft_tissue_img[replace_mask == 255]

        self.current_image = result
        print("✓ Ribs removed (replaced with soft tissue) successfully")

        # ส่ง Bone Mask ที่เป็น 0/255 กลับไปสำหรับการแสดงผล (ในขนาดที่ถูก Crop/เต็มภาพตามสถานะปัจจุบัน)
        if self.crop_coords is not None:
             y1, y2, x1, x2 = self.crop_coords
             bone_mask_clean_disp = bone_mask_clean[y1:y2, x1:x2]
             ribs_mask_disp = ribs_mask[y1:y2, x1:x2]
             bones_without_ribs_mask_disp = bones_without_ribs_mask[y1:y2, x1:x2]
        else:
             bone_mask_clean_disp = bone_mask_clean
             ribs_mask_disp = ribs_mask
             bones_without_ribs_mask_disp = bones_without_ribs_mask


        return result, bone_mask_clean_disp, ribs_mask_disp, bones_without_ribs_mask_disp

    # -------------------------------------------------
    # 4) Utils (เหมือนเดิม)
    # -------------------------------------------------
    def save_result(self, output_path: str, filename: str = "processed_xray.jpg"):
        full_path = os.path.join(output_path, filename)
        cv2.imwrite(full_path, self.current_image)
        print(f"✓ Image saved successfully: {full_path}")
        return full_path

    def get_current_image(self):
        return self.current_image

    def get_original_image(self):
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
        self.root.title("Advanced X-ray Image Processing")

        self.processor = None
        self.step_images = []
        self.result_tk_images = []

        self.image_canvas = None
        self.tk_image = None
        self.display_scale = 1.0

        self.crop_method = tk.StringVar(value="auto")
        self.crop_start = None
        self.crop_rect = None

        self.polygon_points = []
        self.polygon_items = []

        self.results_frame = None

        self.slider_controls = {}
        self.float_slider_controls = {} 

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
            text="Advanced X-ray Image Processing",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        title_label.pack(pady=(0, 10))

        # Image card & Canvas
        image_card = ctk.CTkFrame(self.main_frame, corner_radius=16, fg_color=("white", "#1E1E1E"))
        image_card.pack(pady=5)
        self.image_canvas = tk.Canvas(image_card, width=512, height=512, bg="black", highlightthickness=0, bd=0)
        self.image_canvas.pack(padx=10, pady=10)
        self.image_canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.image_canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.image_canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.image_canvas.bind("<Double-Button-1>", self.on_canvas_double_click)

        # Status & Select Button
        self.status_label = ctk.CTkLabel(self.main_frame, text="Please select an X-ray image", font=ctk.CTkFont(size=11), text_color="#8E8E93")
        self.status_label.pack(pady=(8, 6))
        
        # Primary Action Button (Select Image) - ใช้สีตาม Theme Default
        select_btn = ctk.CTkButton(self.main_frame, text="Select an X-ray image", command=self.load_image, width=200, height=34, corner_radius=18)
        select_btn.pack(pady=(0, 16))

        # Crop settings (เหมือนเดิม)
        crop_title = ctk.CTkLabel(self.main_frame, text="Crop/ROI Settings", font=ctk.CTkFont(size=14, weight="bold"))
        crop_title.pack(pady=(4, 4))
        crop_frame = ctk.CTkFrame(self.main_frame, corner_radius=12, fg_color=("white", "#1E1E1E"))
        crop_frame.pack(fill="x", padx=6, pady=(0, 8))
        radio_row = ctk.CTkFrame(crop_frame, fg_color="transparent")
        radio_row.pack(pady=(8, 4))
        radio_font = ctk.CTkFont(size=12)
        ctk.CTkRadioButton(radio_row, text="Auto crop lung region", variable=self.crop_method, value="auto", height=24, radiobutton_width=14, radiobutton_height=14, font=radio_font).pack(side="left", padx=8)
        ctk.CTkRadioButton(radio_row, text="Manual crop (drag on image)", variable=self.crop_method, value="manual", height=24, radiobutton_width=14, radiobutton_height=14, font=radio_font).pack(side="left", padx=8)
        ctk.CTkRadioButton(radio_row, text="Polygon ROI (click points)", variable=self.crop_method, value="polygon", height=24, radiobutton_width=14, radiobutton_height=14, font=radio_font).pack(side="left", padx=8)
        crop_btn_row = ctk.CTkFrame(crop_frame, fg_color="transparent")
        crop_btn_row.pack(pady=(4, 10))
        
        # Primary Action Button (Apply crop)
        ctk.CTkButton(crop_btn_row, text="Apply crop", command=self.apply_crop, width=120, height=30, corner_radius=16).pack(side="left", padx=6)
        
        # Neutral Action Button (Reset crop)
        ctk.CTkButton(
            crop_btn_row, 
            text="Reset crop", 
            command=self.reset_crop, 
            width=120, 
            height=30, 
            corner_radius=16, 
            fg_color="#3A3A3C", 
            hover_color="#48484A", 
            text_color="white" 
        ).pack(side="left", padx=6)

        # Parameters
        param_title = ctk.CTkLabel(self.main_frame, text="Processing Parameters (Advanced)", font=ctk.CTkFont(size=14, weight="bold"))
        param_title.pack(pady=(10, 4))
        param_card = ctk.CTkFrame(self.main_frame, corner_radius=12, fg_color=("white", "#1E1E1E"))
        param_card.pack(fill="x", padx=6, pady=(0, 10))

        # ----- สร้างตัวแปรพารามิเตอร์ (FIXED: Homomorphic และ Bilateral) -----
        self.params = {
            "enhancement_method": tk.StringVar(value="homomorphic"), # FIXED: Homomorphic
            "filter_method": tk.StringVar(value="bilateral"),        # FIXED: Bilateral
            "homo_d0": tk.DoubleVar(value=50.0),
            "homo_gamma_l": tk.DoubleVar(value=0.5),
            "homo_gamma_h": tk.DoubleVar(value=2.0),
            "bilateral_d": tk.IntVar(value=9),
            "bilateral_sigma_c": tk.IntVar(value=75),
            "bilateral_sigma_s": tk.IntVar(value=75),
            "adapt_block_size": tk.IntVar(value=11), 
            "adapt_c": tk.IntVar(value=2),           
            "rib_length": tk.IntVar(value=40),
            "rib_thickness": tk.IntVar(value=3),
            "spine_width_percent": tk.IntVar(value=25),
            "canny_t1": tk.IntVar(value=50),
            "canny_t2": tk.IntVar(value=150),
        }
        
        # Helper functions for UI (เหมือนเดิม)
        def slider_callback_int(value, var, value_label):
            iv = int(round(float(value)))
            if var.get() != iv: 
                var.set(iv)
                value_label.configure(text=str(iv))

        def slider_callback_float(value, var, value_label):
            fv = round(float(value), 2)
            if var.get() != fv:
                var.set(fv)
                value_label.configure(text=f"{fv:.2f}")

        def add_slider(parent, key, label_text, var, frm, to, step=1):
            is_float = isinstance(var, tk.DoubleVar)
            row = ctk.CTkFrame(parent, fg_color="transparent")
            row.pack(fill="x", padx=10, pady=(4, 4))

            top_row = ctk.CTkFrame(row, fg_color="transparent")
            top_row.pack(fill="x")

            ctk.CTkLabel(top_row, text=label_text, anchor="w").pack(side="left")

            if is_float:
                value_label = ctk.CTkLabel(top_row, text=f"{var.get():.2f}", anchor="e", width=40)
            else:
                value_label = ctk.CTkLabel(top_row, text=str(var.get()), anchor="e", width=40)
            value_label.pack(side="right")
            
            steps = int(round((to - frm) / step)) if step > 0 else 0

            slider = ctk.CTkSlider(
                row,
                from_=frm,
                to=to,
                number_of_steps=steps,
                command=lambda v: (
                    slider_callback_float(v, var, value_label) if is_float else slider_callback_int(v, var, value_label)
                ),
            )
            slider.set(var.get())
            slider.pack(fill="x")

            control_dict = {"slider": slider, "label": value_label, "var": var, "is_float": is_float}
            if is_float:
                self.float_slider_controls[key] = control_dict
            else:
                self.slider_controls[key] = control_dict

        # Neutral Action Button (Reset params)
        reset_param_btn = ctk.CTkButton(
            param_card,
            text="Reset params",
            command=self.reset_parameters,
            width=100,
            height=24,
            corner_radius=12,
            fg_color="#3A3A3C", # Mid-dark gray
            hover_color="#48484A", # Slightly lighter hover
            text_color="white", # Light text
        )
        reset_param_btn.pack(side="right", padx=10, pady=(8, 4))

        # --- Enhancement Method (FIXED) ---
        enh_label = ctk.CTkLabel(
            param_card, 
            text="Enhancement Method: Homomorphic Filtering (Recommended)", 
            anchor="w", 
            font=ctk.CTkFont(size=13, weight="bold"), 
            text_color="#34C759" # Green for "Recommended/Success"
        )
        enh_label.pack(fill="x", padx=10, pady=(8, 4))

        # Homomorphic Sliders
        add_slider(param_card, "homo_d0", "Homomorphic Cutoff Frequency D0", self.params["homo_d0"], 10, 100)
        add_slider(param_card, "homo_gamma_l", "Homomorphic Low Gain (γL)", self.params["homo_gamma_l"], 0.1, 1.0, step=0.05)
        add_slider(param_card, "homo_gamma_h", "Homomorphic High Gain (γH)", self.params["homo_gamma_h"], 1.0, 5.0, step=0.1)

        ctk.CTkFrame(param_card, height=1, fg_color="#4A4A4A").pack(fill="x", padx=10, pady=8) # Separator

        # --- Noise Reduction Method (FIXED) ---
        filter_label = ctk.CTkLabel(
            param_card, 
            text="Noise Reduction Filter: Bilateral Filtering (Recommended)", 
            anchor="w", 
            font=ctk.CTkFont(size=13, weight="bold"), 
            text_color="#34C759" # Green for "Recommended/Success"
        )
        filter_label.pack(fill="x", padx=10, pady=(8, 4))

        # Bilateral Filter Sliders
        add_slider(param_card, "bilateral_d", "Bilateral Filter Diameter (d)", self.params["bilateral_d"], 3, 21, step=2) 
        add_slider(param_card, "bilateral_sigma_c", "Bilateral Sigma Color (σc)", self.params["bilateral_sigma_c"], 10, 150)
        add_slider(param_card, "bilateral_sigma_s", "Bilateral Sigma Space (σs)", self.params["bilateral_sigma_s"], 10, 150)

        ctk.CTkFrame(param_card, height=1, fg_color="#4A4A4A").pack(fill="x", padx=10, pady=8) # Separator

        # --- Segmentation (Adaptive Thresholding) --- (เหมือนเดิม)
        add_slider(param_card, "adapt_block_size", "Adaptive Threshold Block Size (k, must be odd)", self.params["adapt_block_size"], 3, 51, step=2)
        add_slider(param_card, "adapt_c", "Adaptive Threshold Constant (C)", self.params["adapt_c"], 0, 20)

        ctk.CTkFrame(param_card, height=1, fg_color="#4A4A4A").pack(fill="x", padx=10, pady=8) # Separator

        # --- Morphology (Same as original) (เหมือนเดิม) ---
        add_slider(param_card, "rib_length", "Structuring element length for ribs", self.params["rib_length"], 10, 120)
        add_slider(param_card, "rib_thickness", "Structuring element thickness for ribs", self.params["rib_thickness"], 1, 9)
        add_slider(param_card, "spine_width_percent", "Spine region width (%)", self.params["spine_width_percent"], 10, 40)

        ctk.CTkFrame(param_card, height=1, fg_color="#4A4A4A").pack(fill="x", padx=10, pady=8) # Separator
        
        # --- Canny Edge --- (เหมือนเดิม)
        add_slider(param_card, "canny_t1", "Canny Edge Threshold 1 (T1)", self.params["canny_t1"], 0, 255)
        add_slider(param_card, "canny_t2", "Canny Edge Threshold 2 (T2)", self.params["canny_t2"], 0, 255)
        
        # Action buttons 
        action_row = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        action_row.pack(pady=14)
        
        # Primary Action Button (Process image)
        ctk.CTkButton(action_row, text="Process image", command=self.process_image, width=160, height=34, corner_radius=18).pack(side="left", padx=6)
        
        # Neutral Action Button (Reset processing)
        ctk.CTkButton(
            action_row, 
            text="Reset processing", 
            command=self.reset_processing, 
            width=160, 
            height=34, 
            corner_radius=18, 
            fg_color="#3A3A3C", 
            hover_color="#48484A", 
            text_color="white" 
        ).pack(side="left", padx=6)
        
        # Secondary/Emphasis Button (Show results)
        ctk.CTkButton(action_row, text="Show results", command=self.show_results, width=160, height=34, corner_radius=18, fg_color="#FFD60A", hover_color="#E5C009", text_color="#1D1D1F").pack(side="left", padx=6)

        # Save button 
        self.save_row = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.save_row.pack(pady=(4, 10))
        # Primary Action Button (Save results)
        ctk.CTkButton(self.save_row, text="Save results", command=self.save_results, width=220, height=32, corner_radius=18).pack()

    # ---------- Helper Functions (Canvas/Crop/ROI) (เหมือนเดิม) ----------

    def show_image_on_canvas(self, img_gray):
        if img_gray is None: return
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
        self.polygon_points = []
        self.polygon_items = []
    
    def clear_polygon(self):
        for item in self.polygon_items: self.image_canvas.delete(item)
        self.polygon_points = []
        self.polygon_items = []

    def add_polygon_point(self, x, y):
        r = 3
        dot = self.image_canvas.create_oval(x - r, y - r, x + r, y + r, fill="#FFD60A", outline="")
        self.polygon_items.append(dot)
        if self.polygon_points:
            x0, y0 = self.polygon_points[-1]
            line = self.image_canvas.create_line(x0, y0, x, y, fill="#FFD60A", width=2)
            self.polygon_items.append(line)
        self.polygon_points.append((x, y))

    def finalize_polygon(self):
        if self.processor is None or self.display_scale <= 0: return
        if len(self.polygon_points) < 3:
            messagebox.showwarning("Warning", "Please select at least 3 points for polygon ROI.")
            return

        x0, y0 = self.polygon_points[-1]
        x_first, y_first = self.polygon_points[0]
        line = self.image_canvas.create_line(x0, y0, x_first, y_first, fill="#FFD60A", width=2)
        self.polygon_items.append(line)

        pts = []
        for x, y in self.polygon_points:
            px = int(x / self.display_scale)
            py = int(y / self.display_scale)
            pts.append([px, py])

        h, w = self.processor.original_image.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)

        self.processor.set_polygon_roi(mask)
        self.show_image_on_canvas(self.processor.get_original_image())
        self.status_label.configure(text="ROI (polygon for processing) selected", text_color="#0A84FF")

    def on_canvas_press(self, event):
        if self.processor is None: return
        method = self.crop_method.get()
        if method == "manual":
            self.crop_start = (event.x, event.y)
            if self.crop_rect is not None: self.image_canvas.delete(self.crop_rect); self.crop_rect = None
        elif method == "polygon":
            self.add_polygon_point(event.x, event.y)

    def on_canvas_drag(self, event):
        if self.processor is None or self.crop_method.get() != "manual" or self.crop_start is None: return
        x0, y0 = self.crop_start
        x1, y1 = event.x, event.y
        if self.crop_rect is None:
            self.crop_rect = self.image_canvas.create_rectangle(x0, y0, x1, y1, outline="#0A84FF", width=2)
        else:
            self.image_canvas.coords(self.crop_rect, x0, y0, x1, y1)

    def on_canvas_release(self, event):
        if self.processor is None or self.crop_method.get() != "manual" or self.crop_start is None: return
        x0, y0 = self.crop_start
        x1, y1 = event.x, event.y
        self.crop_start = None
        if self.display_scale <= 0: return
        
        x_min = int(min(x0, x1) / self.display_scale)
        x_max = int(max(x0, x1) / self.display_scale)
        y_min = int(min(y0, y1) / self.display_scale)
        y_max = int(max(y0, y1) / self.display_scale)
        h, w = self.processor.original_image.shape
        x_min = max(0, min(w - 1, x_min)); x_max = max(0, min(w, x_max))
        y_min = max(0, min(h - 1, y_min)); y_max = max(0, min(h, y_max))

        if x_max <= x_min or y_max <= y_min: messagebox.showwarning("Warning", "Invalid crop region"); return

        self.processor.crop_coords = (y_min, y_max, x_min, x_max)
        self.processor.roi_mask = None
        self.processor.reset_to_cropped()
        self.show_image_on_canvas(self.processor.get_current_image())
        self.status_label.configure(text="ROI (manual crop) selected", text_color="#0A84FF")

    def on_canvas_double_click(self, event):
        if self.processor is None or self.crop_method.get() != "polygon": return
        self.finalize_polygon()

    def apply_crop(self):
        if self.processor is None: messagebox.showwarning("Incomplete data", "Please select an image first"); return
        method = self.crop_method.get()
        if method == "auto":
            try:
                self.processor.auto_crop_lung_region()
                self.show_image_on_canvas(self.processor.get_current_image())
                self.status_label.configure(text="ROI (auto crop) selected", text_color="#0A84FF")
            except Exception as e:
                messagebox.showerror("Error", f"Auto-crop failed: {e}")
        elif method == "manual":
            messagebox.showinfo("Manual crop", "Select 'Manual crop' then drag the mouse over the image to choose ROI.")
        elif method == "polygon":
            messagebox.showinfo("Polygon ROI", "Select 'Polygon ROI' then click points on the image.\nDouble-click to finish the polygon.\nProcessing will be limited to that region (image not cropped).")

    def reset_crop(self):
        if self.processor is None: messagebox.showwarning("Incomplete data", "Please select an image first"); return
        self.processor.crop_coords = None
        self.processor.roi_mask = None
        self.processor.reset_to_cropped()
        self.clear_polygon()
        self.show_image_on_canvas(self.processor.get_original_image())
        self.status_label.configure(text="Crop/ROI has been reset to full image", text_color="#8E8E93")

    def reset_processing(self):
        if self.processor is None: messagebox.showwarning("Incomplete data", "Please select an image first"); return
        self.processor.reset_to_cropped()
        self.step_images = []
        self.show_image_on_canvas(self.processor.get_current_image())
        self.status_label.configure(text="Processing has been reset", text_color="#8E8E93")
        if self.results_frame is not None: self.results_frame.pack_forget(); self.results_frame = None
        self.result_tk_images.clear()

    def reset_parameters(self):
        # ค่า default
        defaults = {
            "enhancement_method": "homomorphic", # FIXED Default
            "filter_method": "bilateral",        # FIXED Default
            "homo_d0": 50.0,
            "homo_gamma_l": 0.5,
            "homo_gamma_h": 2.0,
            "bilateral_d": 9,
            "bilateral_sigma_c": 75,
            "bilateral_sigma_s": 75,
            "adapt_block_size": 11,
            "adapt_c": 2,
            "rib_length": 40,
            "rib_thickness": 3,
            "spine_width_percent": 25,
            "canny_t1": 50,
            "canny_t2": 150,
        }

        # ตั้งค่าให้ตัวแปร string
        self.params["enhancement_method"].set(defaults["enhancement_method"])
        self.params["filter_method"].set(defaults["filter_method"])

        # ตั้งค่าให้ slider
        for key, value in defaults.items():
            if key in self.slider_controls:
                ctrl = self.slider_controls[key]
                ctrl["var"].set(value)
                ctrl["slider"].set(value)
                ctrl["label"].configure(text=str(value))
            elif key in self.float_slider_controls:
                ctrl = self.float_slider_controls[key]
                ctrl["var"].set(value)
                ctrl["slider"].set(value)
                ctrl["label"].configure(text=f"{value:.2f}")

        self.status_label.configure(text="Parameters reset to default", text_color="#8E8E93")

    # ---------- Main Logic Callbacks (เหมือนเดิม) ----------

    def load_image(self):
        file_path = filedialog.askopenfilename(title="Select X-ray image", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")])
        if not file_path: return
        try:
            self.processor = AdvancedXrayProcessor(file_path)
            self.status_label.configure(text=f"Loaded: {os.path.basename(file_path)}", text_color="#34C759")
            self.processor.crop_coords = None
            self.processor.roi_mask = None
            self.crop_method.set("auto")
            self.show_image_on_canvas(self.processor.get_original_image())
            self.step_images = []
            if self.results_frame is not None: self.results_frame.pack_forget(); self.results_frame = None
            self.result_tk_images.clear()
        except Exception as e:
            messagebox.showerror("Error", f"Unable to load image: {e}")
            self.status_label.configure(text="Image loading failed", text_color="#FF3B30")
            self.processor = None; self.step_images = []

    def process_image(self):
        if self.processor is None: messagebox.showwarning("Incomplete data", "Please select an image first"); return
        
        self.reset_processing() # Reset ก่อนเริ่ม
        self.step_images = [("Original / ROI", self.processor.get_current_image().copy())]

        try:
            # 1. Enhancement (FIXED: Homomorphic)
            enh_method = self.params["enhancement_method"].get()
            enh_params = {
                "homo_d0": self.params["homo_d0"].get(),
                "homo_gamma_l": self.params["homo_gamma_l"].get(),
                "homo_gamma_h": self.params["homo_gamma_h"].get(),
            }
            self.processor.apply_enhancement(enh_method, enh_params)
            self.step_images.append((f"1. Enhancement ({enh_method})", self.processor.get_current_image().copy()))

            # 2. Noise Reduction (FIXED: Bilateral)
            filter_method = self.params["filter_method"].get()
            filter_params = {
                "bilateral_d": self.params["bilateral_d"].get(),
                "bilateral_sigma_c": self.params["bilateral_sigma_c"].get(),
                "bilateral_sigma_s": self.params["bilateral_sigma_s"].get(),
            }
            self.processor.apply_noise_reduction(filter_method, filter_params)
            self.step_images.append((f"2. Noise Reduction ({filter_method})", self.processor.get_current_image().copy()))
            
            # 3. Segmentation & Rib Suppression
            adapt_block = self.params["adapt_block_size"].get()
            adapt_c = self.params["adapt_c"].get()
            rib_len = self.params["rib_length"].get()
            rib_th = self.params["rib_thickness"].get()
            spine_ratio = self.params["spine_width_percent"].get() / 100.0
            
            # Run the updated rib removal
            result_img, bone_mask_clean, ribs_mask, bones_without_ribs_mask = self.processor.remove_ribs_morphology(
                adapt_block, adapt_c, rib_len, rib_th, spine_ratio
            )
            
            # 4. Canny Edge Output (Applied to the final processed image before replacement)
            canny_t1 = self.params["canny_t1"].get()
            canny_t2 = self.params["canny_t2"].get()
            canny_edge_map = self.processor.get_canny_edge(canny_t1, canny_t2)
            
            # Add all output steps for display
            self.step_images.append(("3. Adaptive Bone Mask", bone_mask_clean))
            self.step_images.append(("4. Ribs Mask", ribs_mask))
            self.step_images.append(("5. Bones without Ribs Mask", bones_without_ribs_mask))
            self.step_images.append(("6. Rib Suppression Result", result_img))
            self.step_images.append(("7. Canny Edge Map", canny_edge_map))


            self.show_image_on_canvas(result_img)
            self.status_label.configure(text="Processing completed successfully. Click 'Show results' for details.", text_color="#34C759")

        except Exception as e:
            messagebox.showerror("Processing Error", f"An error occurred during processing: {e}")
            self.status_label.configure(text="Processing failed", text_color="#FF3B30")
            self.processor.reset_to_cropped()
            self.show_image_on_canvas(self.processor.get_current_image())
            self.step_images = []

    def show_results(self):
        if not self.step_images:
            messagebox.showwarning("No Results", "Please run 'Process image' first to generate results.")
            return

        # Simple separate window for results (for demonstration/simplicity)
        results_window = ctk.CTkToplevel(self.root)
        results_window.title("Processing Steps and Masks")

        scrollable_frame = ctk.CTkScrollableFrame(results_window, width=900, height=600)
        scrollable_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        self.result_tk_images.clear() # Clear previous images
        
        row, col = 0, 0
        for name, img in self.step_images:
            # Convert OpenCV image to PhotoImage for display
            if img.ndim == 2:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = img
            
            # Resize image for display
            h, w = img_rgb.shape[:2]
            max_dim = 250
            scale = min(max_dim / w, max_dim / h, 1.0)
            disp_w, disp_h = int(w * scale), int(h * scale)
            img_resized = cv2.resize(img_rgb, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

            pil_img = Image.fromarray(img_resized)
            tk_img = ImageTk.PhotoImage(pil_img)
            self.result_tk_images.append(tk_img) # Store reference to prevent garbage collection

            # Image Label
            label = ctk.CTkLabel(scrollable_frame, text=name, image=tk_img, compound="top", font=ctk.CTkFont(size=14, weight="bold"))
            label.grid(row=row, column=col, padx=10, pady=10, sticky="n")

            col += 1
            if col > 2: # 3 images per row
                col = 0
                row += 1

    def save_results(self):
        if self.processor is None or not self.step_images:
            messagebox.showwarning("No Results", "Please run 'Process image' first.")
            return

        output_dir = filedialog.askdirectory(title="Select Output Directory")
        if not output_dir: return

        try:
            base_filename = os.path.splitext(os.path.basename(self.processor.image_path))[0]
            for i, (name, img) in enumerate(self.step_images):
                safe_name = name.replace(" ", "_").replace("/", "_").replace(".", "_")
                filename = f"{i:02d}_{base_filename}_{safe_name}.png"
                full_path = os.path.join(output_dir, filename)
                cv2.imwrite(full_path, img)

            # Save the final result using the dedicated function
            final_filename = f"{base_filename}_FINAL_SUPPRESSED.png"
            self.processor.save_result(output_dir, final_filename)
            
            messagebox.showinfo("Success", f"All {len(self.step_images)} result images saved to: {output_dir}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save images: {e}")

# =========================================
#        ส่วนเริ่มต้นโปรแกรม (Entry Point)
# =========================================
if __name__ == "__main__":
    # 1. ตั้งค่ารูปลักษณ์ (Dark mode)
    ctk.set_appearance_mode("Dark") 
    
    # 2. ตั้งค่าธีมสี
    ctk.set_default_color_theme("blue")
    
    # 3. สร้างหน้าต่างหลัก (root window)
    root = ctk.CTk()
    
    # 4. สร้าง instance ของ Class GUI
    app = XrayProcessorGUI(root)
    
    # 5. เริ่มต้น loop การทำงานของ GUI
    root.mainloop()