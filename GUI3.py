import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

import tkinter as tk  # ยังใช้สำหรับ Canvas และ event บางส่วน
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
        x2 = int(max(0, min(w, w * x_end_ratio)))
        y1 = int(max(0, min(h - 1, h * y_start_ratio)))
        y2 = int(max(0, min(h, h * y_end_ratio)))

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
    # 1) manual crop ผ่าน matplotlib (ยังเก็บไว้เผื่อใช้)
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
    # 2) Enhancement
    # -------------------------------------------------
    def histogram_equalization(self):
        self.current_image = cv2.equalizeHist(self.current_image)
        print("✓ Histogram equalization completed successfully")
        return self.current_image

    def gaussian_smoothing(self, ksize: int = 5):
        if ksize < 3:
            ksize = 3
        if ksize % 2 == 0:
            ksize += 1
        self.current_image = cv2.GaussianBlur(self.current_image, (ksize, ksize), 0)
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
    # 4) ลบซี่โครงด้วย Morphology
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

        soft_tissue_img = img_blur
        result = self.current_image.copy()
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
        if self.crop_coords is not None:
            y1, y2, x1, x2 = self.crop_coords
            return self.original_image[y1:y2, x1:x2]
        return self.original_image

    def display_results(self, step_images):
        n = len(step_images)
        if n == 0:
            return

        if n == 1:
            title, img = step_images[0]
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(img, cmap="gray")
            ax.set_title(title, fontsize=12)
            ax.axis("off")
            plt.tight_layout()
            plt.show()
            return

        k = n - 1
        rows = 3
        cols = int(np.ceil(k / 2.0))
        if cols < 1:
            cols = 1

        base_size = 4
        fig, axes = plt.subplots(rows, cols, figsize=(base_size * cols, base_size * rows))

        axes = np.array(axes)
        if axes.ndim == 1:
            axes = axes.reshape(rows, cols)

        idx = 0
        for r in range(2):
            for c in range(cols):
                ax = axes[r, c]
                if idx < k:
                    title, img = step_images[idx]
                    ax.imshow(img, cmap="gray")
                    ax.set_title(title, fontsize=12)
                    ax.axis("off")
                    idx += 1
                else:
                    ax.axis("off")

        final_title, final_img = step_images[-1]
        mid_c = cols // 2

        for c in range(cols):
            ax = axes[2, c]
            if c == mid_c:
                ax.imshow(final_img, cmap="gray")
                ax.set_title(final_title, fontsize=12)
                ax.axis("off")
            else:
                ax.axis("off")

        plt.tight_layout()
        plt.show()


# =========================================
#        CustomTkinter GUI (modern)
# =========================================
class XrayProcessorGUI:
    def __init__(self, root: ctk.CTk):
        self.root = root
        self.root.title("X-ray Image Processing")

        self.processor = None
        self.step_images = []

        self.image_canvas = None
        self.tk_image = None
        self.display_scale = 1.0

        self.crop_method = tk.StringVar(value="auto")
        self.crop_start = None
        self.crop_rect = None

        # scrollable main area
        self.main_frame = ctk.CTkScrollableFrame(
            root, width=630, height=680, corner_radius=0, fg_color="transparent"
        )
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self._build_gui()

    # ---------- UI หลัก ----------
    def _build_gui(self):
        # ===== Title =====
        title_label = ctk.CTkLabel(
            self.main_frame,
            text="X-ray Image",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        title_label.pack(pady=(0, 10))

        # ===== Image area =====
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

        # ===== Status =====
        self.status_label = ctk.CTkLabel(
            self.main_frame,
            text="Please select an X-ray image",
            font=ctk.CTkFont(size=11),
            text_color="#8E8E93",
        )
        self.status_label.pack(pady=(8, 6))

        # ===== Select image button =====
        select_btn = ctk.CTkButton(
            self.main_frame,
            text="Select an X-ray image",
            command=self.load_image,
            width=200,
            height=34,
            corner_radius=18,
        )
        select_btn.pack(pady=(0, 16))

        # ===== Crop section =====
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

        # radio buttons
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

        # crop buttons
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

        # ===== Parameters section =====
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

        self.params = {
            "use_hist_eq": tk.BooleanVar(value=True),
            "smooth_kernel": tk.IntVar(value=5),
            "rib_length": tk.IntVar(value=40),
            "rib_thickness": tk.IntVar(value=3),
            "spine_width_percent": tk.IntVar(value=25),
        }

        # Hist eq
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

        # ---- Slider helper (มีตัวเลขแสดงค่า) ----
        def slider_callback(value, var, value_label):
            iv = int(round(float(value)))
            var.set(iv)
            value_label.configure(text=str(iv))

        def add_slider(parent, label_text, var, frm, to):
            row = ctk.CTkFrame(parent, fg_color="transparent")
            row.pack(fill="x", padx=10, pady=(4, 4))

            # แถวบน: label + value
            top_row = ctk.CTkFrame(row, fg_color="transparent")
            top_row.pack(fill="x")

            ctk.CTkLabel(top_row, text=label_text, anchor="w").pack(
                side="left"
            )

            value_label = ctk.CTkLabel(
                top_row, text=str(var.get()), anchor="e", width=40
            )
            value_label.pack(side="right")

            # แถวล่าง: slider
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

        # ===== Action buttons =====
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
        save_row = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        save_row.pack(pady=(4, 10))

        ctk.CTkButton(
            save_row,
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

    # ===== events manual crop =====
    def on_canvas_press(self, event):
        if self.processor is None or self.crop_method.get() != "manual":
            return
        self.crop_start = (event.x, event.y)
        if self.crop_rect is not None:
            self.image_canvas.delete(self.crop_rect)
            self.crop_rect = None

    def on_canvas_drag(self, event):
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
        if self.crop_start is None or self.crop_method.get() != "manual":
            return
        x0, y0 = self.crop_start
        x1, y1 = event.x, event.y
        self.crop_start = None

        if self.processor is None or self.display_scale <= 0:
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
        self.status_label.configure(text="ROI (manual) selected", text_color="#0A84FF")

    # ===== ปุ่ม Apply crop (auto) =====
    def apply_crop(self):
        if self.processor is None:
            messagebox.showwarning("Incomplete data", "Please select an image first")
            return

        if self.crop_method.get() == "auto":
            try:
                self.processor.auto_crop_lung_region()
                self.show_image_on_canvas(self.processor.get_current_image())
                self.status_label.configure(text="ROI (auto) selected", text_color="#0A84FF")
            except Exception as e:
                messagebox.showerror("Error", f"Auto-crop failed: {e}")
        else:
            messagebox.showinfo(
                "Manual crop",
                "Select 'Manual crop' then drag the mouse over the image to choose ROI.",
            )

    # ===== ปุ่ม Reset crop =====
    def reset_crop(self):
        if self.processor is None:
            messagebox.showwarning("Incomplete data", "Please select an image first")
            return

        self.processor.crop_coords = None
        self.processor.reset_to_cropped()

        self.show_image_on_canvas(self.processor.get_current_image())
        self.status_label.configure(
            text="Crop has been reset to full image", text_color="#8E8E93"
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
            self.step_images = []
            self.processor.crop_coords = None
            self.crop_method.set("auto")

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
            messagebox.showinfo("Success", "Image processing completed!")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during processing: {e}")
            self.status_label.configure(text="Processing failed", text_color="#FF3B30")

    def show_results(self):
        if not self.step_images or len(self.step_images) < 2:
            messagebox.showwarning("Incomplete data", "Please process the image first")
            return
        self.processor.display_results(self.step_images)

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

            messagebox.showinfo("Success", "Results saved successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Unable to save results: {e}")
            self.status_label.configure(
                text="Saving results failed", text_color="#FF3B30"
            )


# =========================================
#                   main
# =========================================
def main():
    ctk.set_appearance_mode("light")          # ลอง "dark" ได้
    ctk.set_default_color_theme("blue")      # หรือ "green", "dark-blue"

    root = ctk.CTk()
    root.geometry("650x700")
    root.resizable(False, False)

    app = XrayProcessorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
