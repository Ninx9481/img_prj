import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import tkinter as tk
from tkinter import (
    filedialog,
    messagebox,
    Scale,
    Button,
    Label,
    Frame,
    HORIZONTAL,
    BooleanVar,
)

from PIL import Image, ImageTk  # ใช้แสดงภาพบน Tkinter


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
            raise ValueError("The ratio value for auto-crop is invalid, resulting in an empty selection")

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
        ax.set_title("Drag the mouse to select the torso/lung region, then release", fontsize=14)

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

        rect_selector = RectangleSelector(
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
#             Tkinter GUI (with scrollbar)
# =========================================
class XrayProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("X-ray image processing program")
        self.root.geometry("650x700")  # ขนาดคงที่ (เดี๋ยวล็อกใน main)

        self.processor = None
        self.step_images = []

        self.image_canvas = None
        self.tk_image = None
        self.display_scale = 1.0

        self.crop_method = tk.StringVar(value="auto")
        self.crop_start = None
        self.crop_rect = None

        # สำหรับ scrollable area
        self.canvas = None
        self.content = None
        self.content_id = None

        self._build_scrollable_area()
        self._build_gui()

    # ---------- สร้างพื้นที่เลื่อน ----------
    def _build_scrollable_area(self):
        outer = Frame(self.root)
        outer.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(outer)
        vscroll = tk.Scrollbar(outer, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vscroll.set)

        vscroll.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        # frame ข้างในที่เราจะเอา widget ทั้งหมดไว้
        self.content = Frame(self.canvas)

        # วาง content ให้อยู่กึ่งกลางด้านบน (anchor="n")
        self.content_id = self.canvas.create_window(
            (0, 0), window=self.content, anchor="n"
        )

        # อัปเดต scrollregion ตามขนาด content
        self.content.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )

        # จัดให้อยู่ตรงกลางทุกครั้งที่ canvas resize
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # เลื่อนด้วย mouse wheel
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_canvas_resize(self, event):
        """จัด content ให้อยู่ตรงกลางแนวนอน และยืดความกว้างตาม canvas"""
        canvas_width = event.width
        self.canvas.coords(self.content_id, canvas_width / 2, 0)
        self.canvas.itemconfig(self.content_id, width=canvas_width)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # ---------- UI หลัก ----------
    def _build_gui(self):
        # ===== หัวข้อด้านบนสุด =====
        title_label = Label(
            self.content,
            text="X-ray Image",
            font=("Arial", 20, "bold"),
            fg="black",
        )
        title_label.pack(pady=10)

        # ===== พื้นที่แสดงภาพ =====
        img_frame = Frame(self.content, bd=2, relief="sunken")
        img_frame.pack(pady=5)

        self.image_canvas = tk.Canvas(img_frame, width=512, height=512, bg="black")
        self.image_canvas.pack()

        self.image_canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.image_canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.image_canvas.bind("<ButtonRelease-1>", self.on_canvas_release)

        # ===== แถบสถานะ (ย้ายมาอยู่เหนือปุ่มเลือกภาพ) =====
        self.status_label = Label(
            self.content,
            text="Please select an X-ray image",
            font=("Arial", 10),
            fg="blue",
        )
        self.status_label.pack(pady=(8, 4))

        # ===== ปุ่มเลือกภาพ =====
        btn_frame = Frame(self.content)
        btn_frame.pack(pady=6)

        Button(
            btn_frame,
            text="Select an X-ray image",
            command=self.load_image,
            width=20,
            height=2,
            bg="lightblue",
        ).pack()

        # ===== หัวข้อ วิธีครอปภาพ =====
        method_label = Label(
            self.content,
            text="== Choose how to crop the image ==",
            font=("Arial", 14, "bold"),
        )
        method_label.pack(pady=(15, 5))

        # เฉพาะ radio ไว้ในแถวบน
        crop_frame = Frame(self.content)
        crop_frame.pack(pady=(0, 5))

        tk.Radiobutton(
            crop_frame,
            text="Auto crop lung region",
            variable=self.crop_method,
            value="auto",
        ).pack(side="left", padx=5)

        tk.Radiobutton(
            crop_frame,
            text="Manual crop (drag on image)",
            variable=self.crop_method,
            value="manual",
        ).pack(side="left", padx=5)

        # ปุ่ม Apply / Reset crop แยกมาอยู่ด้านล่าง
        crop_btn_frame = Frame(self.content)
        crop_btn_frame.pack(pady=(0, 10))

        Button(
            crop_btn_frame,
            text="Apply crop",
            command=self.apply_crop,
            width=12,
        ).pack(side="left", padx=10)

        Button(
            crop_btn_frame,
            text="Reset crop",
            command=self.reset_crop,
            width=12,
        ).pack(side="left", padx=10)

        # ===== พารามิเตอร์ =====
        self.params = {
            "use_hist_eq": BooleanVar(value=True),
            "smooth_kernel": tk.IntVar(value=5),
            "rib_length": tk.IntVar(value=40),
            "rib_thickness": tk.IntVar(value=3),
            "spine_width_percent": tk.IntVar(value=25),
        }

        param_frame = Frame(self.content)
        param_frame.pack(pady=10, padx=20, fill="both", expand=True)

        Label(
            param_frame,
            text="=== Adjust the parameters as shown in the image. ===",
            font=("Arial", 12, "bold"),
        ).pack()

        hist_frame = Frame(param_frame)
        hist_frame.pack(fill="x", pady=5)
        Label(hist_frame, text="Use Histogram Equalization:").pack(side="left", padx=5)
        tk.Checkbutton(
            hist_frame, variable=self.params["use_hist_eq"], onvalue=True, offvalue=False
        ).pack(side="left")

        Label(param_frame, text="Gaussian kernel size (image blurring before segmentation):").pack(
            anchor="w"
        )
        Scale(
            param_frame,
            from_=3,
            to=21,
            orient=HORIZONTAL,
            variable=self.params["smooth_kernel"],
        ).pack(fill="x", padx=10)

        Label(param_frame, text="Structuring element length for ribs:").pack(anchor="w")
        Scale(
            param_frame,
            from_=10,
            to=120,
            orient=HORIZONTAL,
            variable=self.params["rib_length"],
        ).pack(fill="x", padx=10)

        Label(param_frame, text="Structuring element thickness for ribs:").pack(anchor="w")
        Scale(
            param_frame,
            from_=1,
            to=9,
            orient=HORIZONTAL,
            variable=self.params["rib_thickness"],
        ).pack(fill="x", padx=10)

        Label(param_frame, text="Spine region width (%)").pack(anchor="w")
        Scale(
            param_frame,
            from_=10,
            to=40,
            orient=HORIZONTAL,
            variable=self.params["spine_width_percent"],
        ).pack(fill="x", padx=10)

        # ===== ปุ่มประมวลผล/รีเซ็ต/แสดงผล/บันทึก =====
        process_frame = Frame(self.content)
        process_frame.pack(pady=15)

        Button(
            process_frame,
            text="Process image",
            command=self.process_image,
            width=18,
            height=2,
            bg="lightgreen",
        ).pack(side="left", padx=5)

        Button(
            process_frame,
            text="Reset processing",
            command=self.reset_processing,
            width=18,
            height=2,
            bg="lightgray",
        ).pack(side="left", padx=5)

        Button(
            process_frame,
            text="Show results",
            command=self.show_results,
            width=18,
            height=2,
            bg="lightyellow",
        ).pack(side="left", padx=5)

        save_frame = Frame(self.content)
        save_frame.pack(pady=10)

        Button(
            save_frame,
            text="Save results",
            command=self.save_results,
            width=20,
            height=2,
            bg="lightcoral",
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
                x0, y0, x1, y1, outline="red"
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

        # ไม่แตะต้อง step_images เพื่อไม่กระทบผลประมวลผลที่มีอยู่
        self.show_image_on_canvas(self.processor.get_current_image())
        self.status_label.config(text="✓ ROI (manual) selected", fg="green")

    # ===== ปุ่ม Apply crop (auto) =====
    def apply_crop(self):
        if self.processor is None:
            messagebox.showwarning("Incomplete data", "Please select an image first")
            return

        if self.crop_method.get() == "auto":
            try:
                self.processor.auto_crop_lung_region()
                # ไม่แตะต้อง step_images เช่นกัน
                self.show_image_on_canvas(self.processor.get_current_image())
                self.status_label.config(text="✓ ROI (auto) selected", fg="green")
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

        # รีเซ็ตแค่ crop ไม่ยุ่งกับ step_images (ไม่ลบผลประมวลผล)
        self.processor.crop_coords = None
        self.processor.reset_to_cropped()

        self.show_image_on_canvas(self.processor.get_current_image())
        self.status_label.config(text="✓ Crop has been reset to full image", fg="green")

    # ===== ปุ่ม Reset processing =====
    def reset_processing(self):
        if self.processor is None:
            messagebox.showwarning("Incomplete data", "Please select an image first")
            return

        # กลับไปภาพตาม crop ปัจจุบัน และล้างผลประมวลผลทั้งหมด
        self.processor.reset_to_cropped()
        self.step_images = []

        self.show_image_on_canvas(self.processor.get_current_image())
        self.status_label.config(text="✓ Processing has been reset", fg="green")

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
            self.status_label.config(
                text=f"✓ Load image: {os.path.basename(file_path)}", fg="green"
            )

            self.show_image_on_canvas(self.processor.get_original_image())
            self.step_images = []
            self.processor.crop_coords = None
            self.crop_method.set("auto")

        except Exception as e:
            messagebox.showerror("Error", f"Unable to load image: {e}")
            self.status_label.config(text="✗ Image loading failed", fg="red")
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

            # ไม่อัปเดตรูปบน canvas เพื่อให้เห็น ROI เดิม
            self.status_label.config(text="✓ Image processing completed", fg="green")
            messagebox.showinfo("Success", "Image processing completed!")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during processing: {e}")
            self.status_label.config(text="✗ Processing failed", fg="red")

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
                print(f"✓ บันทึก: {full_path}")

            self.processor.current_image = self.step_images[-1][1].copy()
            self.processor.save_result(output_dir, "final_result.jpg")

            self.status_label.config(
                text=f"✓ All results saved at: {output_dir}",
                fg="green",
            )

            messagebox.showinfo("Success", "Results saved successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Unable to save results: {e}")
            self.status_label.config(text="✗ Saving results failed", fg="red")


# =========================================
#                   main
# =========================================
def main():
    root = tk.Tk()
    # ขนาดคงที่ + ล็อกไม่ให้ resize
    root.geometry("650x700")
    root.resizable(False, False)

    app = XrayProcessorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
