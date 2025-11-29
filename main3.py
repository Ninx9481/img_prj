import os
import cv2
import numpy as np

import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image, ImageTk


# =======================================
#   Logic ประมวลผลภาพ (รวมไอเดียจาก Image.py)
# =======================================
class XrayMonkeyProcessor:
    def __init__(self):
        self.original_image = None      # step 1
        self.current_image = None       # step 6 (สุดท้าย)
        self.height = 0
        self.width = 0

        # cropping / ROI
        self.crop_coords = None         # (x1, y1, x2, y2)
        self.roi_mask = None            # mask 0/255 เช่น polygon หรือ rect

        # เก็บ step สำหรับ save 6 steps (แนวจาก Image.py)
        self.step_original = None       # step 1
        self.step_cropped = None        # step 2
        self.step_polygon = None        # step 3 (ภาพโชว์ ROI/polygon)
        self.current_binary = None      # step 4
        self.current_weighted = None    # step 5
        # step 6 ใช้ self.current_image

    # ---------- load / reset ----------
    def load_image(self, path: str):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Unable to load image: {path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.original_image = img_rgb
        self.current_image = img_rgb.copy()
        self.height, self.width = img_rgb.shape[:2]

        # reset roi/crop
        self.crop_coords = None
        self.roi_mask = None

        # reset steps
        self.step_original = self.original_image.copy()
        self.step_cropped = self.original_image.copy()
        self.step_polygon = None
        self.current_binary = None
        self.current_weighted = None

    def reset_image(self):
        if self.original_image is None:
            return
        self.current_image = self.original_image.copy()
        self.height, self.width = self.current_image.shape[:2]
        self.crop_coords = None
        self.roi_mask = None

        # reset step ย่อย
        self.step_cropped = self.original_image.copy()
        self.step_polygon = None
        self.current_binary = None
        self.current_weighted = None

    # ---------- crop / ROI ----------
    def set_crop_rect(self, x1, y1, x2, y2):
        if self.original_image is None:
            return

        x1, x2 = sorted([max(0, x1), min(self.width, x2)])
        y1, y2 = sorted([max(0, y1), min(self.height, y2)])

        if x2 <= x1 or y2 <= y1:
            return

        self.crop_coords = (x1, y1, x2, y2)
        cropped = self.original_image[y1:y2, x1:x2].copy()
        self.current_image = cropped
        self.step_cropped = cropped.copy()
        self.height, self.width = cropped.shape[:2]

        # ถ้ามี ROI mask อยู่เก่า ต้อง crop mask ด้วย
        if self.roi_mask is not None:
            self.roi_mask = self.roi_mask[y1:y2, x1:x2]

    def clear_roi(self):
        self.roi_mask = None
        self.step_polygon = None

    def set_roi_polygon(self, points_xy):
        """
        points_xy: list ของ (x, y) ในพิกเซลของภาพ current_image
        สร้าง mask 0/255
        """
        if self.current_image is None or len(points_xy) < 3:
            return

        h, w = self.current_image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(points_xy, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

        self.roi_mask = mask

        # ทำภาพ visual ROI เพื่อเก็บเป็น step_polygon (step 3)
        overlay = self.current_image.copy()
        color = (255, 0, 0)
        cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=2)
        # ทับ overlay บนภาพเดิมเบาๆ
        alpha = 0.4
        poly_vis = cv2.addWeighted(overlay, alpha, self.current_image, 1 - alpha, 0)
        self.step_polygon = poly_vis

    # ---------- helper ----------
    def _apply_roi_if_any(self, img_gray_or_rgb):
        """
        ถ้ามี roi_mask → ใช้เฉพาะส่วน mask == 255
        ถ้าไม่มี → ส่งกลับทั้งภาพ
        """
        if self.roi_mask is None:
            return img_gray_or_rgb

        if img_gray_or_rgb.ndim == 2:
            out = img_gray_or_rgb.copy()
            out[self.roi_mask == 0] = 0
        else:
            out = img_gray_or_rgb.copy()
            out[self.roi_mask == 0] = 0
        return out

    # ---------- image processing (แนวจาก Image.py) ----------
    def apply_otsu_auto(self):
        """
        Step 4: Otsu Threshold ของภาพ grayscale (จาก current_image)
        คืนค่า threshold ที่ได้
        """
        if self.current_image is None:
            return None

        # ทำเป็น gray ก่อน
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2GRAY)

        # จำกัดเฉพาะ ROI ถ้ามี
        gray_roi = self._apply_roi_if_any(gray)

        otsu_val, binary = cv2.threshold(
            gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        self.current_binary = binary.copy()

        # เอา binary ไปเก็บเป็น current_image แยก channel เพื่อแสดง
        self.current_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        self.height, self.width = self.current_image.shape[:2]

        return int(otsu_val)

    def apply_manual_threshold(self, thresh_val: int):
        """
        ปรับ threshold ด้วยค่าจาก slider (ต่อจาก Otsu)
        """
        if self.current_image is None:
            return

        gray = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2GRAY)
        gray_roi = self._apply_roi_if_any(gray)

        _, binary = cv2.threshold(gray_roi, thresh_val, 255, cv2.THRESH_BINARY)
        self.current_binary = binary.copy()
        self.current_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        self.height, self.width = self.current_image.shape[:2]

    def apply_weighted_mask(self, alpha: float = 0.5):
        """
        Step 5: ทำ weighted mask ทับบน original (แนวจาก Image.py)
        current_binary ต้องมีแล้ว
        """
        if self.current_binary is None or self.step_original is None:
            return

        binary_3ch = cv2.cvtColor(self.current_binary, cv2.COLOR_GRAY2RGB)

        # Resize กลับไปขนาด original ถ้าจำเป็น (simplify: resize ด้วย INTER_NEAREST)
        h0, w0 = self.step_original.shape[:2]
        if binary_3ch.shape[:2] != (h0, w0):
            binary_3ch = cv2.resize(binary_3ch, (w0, h0), interpolation=cv2.INTER_NEAREST)

        # ทำ overlay: จุดที่เป็น white → highlight
        colored_mask = np.zeros_like(binary_3ch)
        colored_mask[self.current_binary > 0] = (255, 0, 0)  # แดง

        result = cv2.addWeighted(self.step_original, 1.0, colored_mask, alpha, 0)
        self.current_weighted = result.copy()
        self.current_image = result.copy()
        self.height, self.width = self.current_image.shape[:2]

    # ---------- helper แปลงไปแสดงบน Tk ----------
    def get_tk_image(self, max_size=(900, 900)):
        """
        return (tk_image, display_width, display_height)
        """
        if self.current_image is None:
            return None, 0, 0

        img = self.current_image
        h, w = img.shape[:2]

        scale = min(max_size[0] / w, max_size[1] / h, 1.0)
        new_w = int(w * scale)
        new_h = int(h * scale)

        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(img_resized)
        tk_img = ImageTk.PhotoImage(pil_img)

        return tk_img, new_w, new_h

    # ---------- save 6 steps ----------
    def save_all_steps(self, base_path: str):
        """
        บันทึก step_1..6 ตามแนว Image.py
        base_path: path ที่ user เลือก (เช่น C:/out/result.png)
        จะใช้ชื่อแบบ result_01_original.png ...
        """
        if self.step_original is None:
            return

        base_name, ext = os.path.splitext(base_path)
        if not ext:
            ext = ".png"

        saved = []

        def _save_if_not_none(img, suffix):
            if img is not None:
                fn = f"{base_name}_{suffix}{ext}"
                cv2.imwrite(fn, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                saved.append(suffix)

        _save_if_not_none(self.step_original, "01_original")
        _save_if_not_none(self.step_cropped, "02_cropped")
        _save_if_not_none(self.step_polygon, "03_polygon")
        if self.current_binary is not None:
            fn = f"{base_name}_04_otsu_binary{ext}"
            cv2.imwrite(fn, self.current_binary)
            saved.append("04_otsu_binary")

        if self.current_weighted is not None:
            _save_if_not_none(self.current_weighted, "05_weighted_mask")

        _save_if_not_none(self.current_image, "06_final_result")

        return saved


# =======================================
#   GUI (โครงสร้างจาก GUI8.py + Theme CTk)
# =======================================
class XrayProcessorGUI:
    def __init__(self, root: ctk.CTk):
        self.root = root
        self.root.title("Monkey X-ray Combined GUI")
        self.root.geometry("1280x720")

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.processor = XrayMonkeyProcessor()

        # canvas
        self.image_canvas = None
        self.tk_image = None
        self.display_scale = 1.0

        # crop state
        self.crop_start = None
        self.crop_rect = None
        self.crop_rect_id = None

        # สร้าง layout หลัก แบบซ้ายรูป-ขวา control
        self._build_layout()

    # ---------- layout ----------
    def _build_layout(self):
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=3)
        self.root.grid_columnconfigure(1, weight=2)

        # left panel (ภาพ)
        self.left_panel = ctk.CTkFrame(self.root)
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        # top toolbar
        toolbar = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        toolbar.pack(fill="x", pady=(4, 4))

        ctk.CTkButton(
            toolbar, text="📂 Load Image", command=self.load_image, width=120
        ).pack(side="left", padx=2)

        ctk.CTkButton(
            toolbar, text="💾 Save Final", command=self.save_final, width=120
        ).pack(side="left", padx=2)

        ctk.CTkButton(
            toolbar, text="📑 Save 6 Steps", command=self.save_all_steps, width=140
        ).pack(side="left", padx=2)

        ctk.CTkButton(
            toolbar, text="🔄 Reset", command=self.reset_image, width=100
        ).pack(side="left", padx=2)

        # canvas frame
        canvas_frame = ctk.CTkFrame(self.left_panel)
        canvas_frame.pack(fill="both", expand=True, pady=(4, 4))

        self.image_canvas = tk.Canvas(
            canvas_frame, bg="#1e1e1e", highlightthickness=0
        )
        self.image_canvas.pack(fill="both", expand=True)

        # bind event สำหรับ crop rect
        self.image_canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.image_canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.image_canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        # status bar
        self.status_label = ctk.CTkLabel(
            self.left_panel, text="Ready", anchor="w"
        )
        self.status_label.pack(fill="x", pady=(2, 4), padx=4)

        # right panel (controls)
        self.right_panel = ctk.CTkFrame(self.root)
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)

        self._build_right_controls()

    def _build_right_controls(self):
        self.right_panel.grid_rowconfigure(10, weight=1)

        # ---- Step 1: ROI / Crop ----
        title1 = ctk.CTkLabel(
            self.right_panel, text="Step 1: Select Area (Crop / ROI)",
            font=ctk.CTkFont(size=15, weight="bold")
        )
        title1.grid(row=0, column=0, sticky="w", padx=8, pady=(4, 2))

        crop_frame = ctk.CTkFrame(self.right_panel)
        crop_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        crop_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            crop_frame, text="ลากเมาส์บนรูปเพื่อ crop สี่เหลี่ยม",
            anchor="w"
        ).grid(row=0, column=0, sticky="w", padx=6, pady=(4, 4))

        ctk.CTkButton(
            crop_frame, text="❌ Clear ROI",
            command=self.clear_roi,
        ).grid(row=1, column=0, sticky="ew", padx=6, pady=(2, 6))

        # ---- Step 2: Processing (Otsu + Threshold) ----
        title2 = ctk.CTkLabel(
            self.right_panel, text="Step 2: Processing",
            font=ctk.CTkFont(size=15, weight="bold")
        )
        title2.grid(row=2, column=0, sticky="w", padx=8, pady=(4, 2))

        proc_frame = ctk.CTkFrame(self.right_panel)
        proc_frame.grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 8))
        proc_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkButton(
            proc_frame, text="✨ Auto Otsu",
            command=self.apply_otsu,
        ).grid(row=0, column=0, sticky="ew", padx=6, pady=(4, 4))

        # slider manual threshold (ต่อจาก Otsu)
        self.thresh_var = tk.IntVar(value=128)

        row_slider = ctk.CTkFrame(proc_frame, fg_color="transparent")
        row_slider.grid(row=1, column=0, sticky="ew", padx=6, pady=(4, 4))
        row_slider.grid_columnconfigure(0, weight=1)

        self.thresh_label = ctk.CTkLabel(row_slider, text="Threshold: 128")
        self.thresh_label.grid(row=0, column=0, sticky="w")

        thresh_slider = ctk.CTkSlider(
            proc_frame, from_=0, to=255, number_of_steps=255,
            command=self.on_thresh_change
        )
        thresh_slider.set(128)
        thresh_slider.grid(row=2, column=0, sticky="ew", padx=6, pady=(4, 4))
        self.thresh_slider = thresh_slider

        # ---- Step 3: Weighted Mask ----
        title3 = ctk.CTkLabel(
            self.right_panel, text="Step 3: Weighted Mask",
            font=ctk.CTkFont(size=15, weight="bold")
        )
        title3.grid(row=4, column=0, sticky="w", padx=8, pady=(4, 2))

        mask_frame = ctk.CTkFrame(self.right_panel)
        mask_frame.grid(row=5, column=0, sticky="ew", padx=8, pady=(0, 8))

        self.alpha_var = tk.DoubleVar(value=0.5)
        self.alpha_label = ctk.CTkLabel(
            mask_frame, text="Alpha: 0.50"
        )
        self.alpha_label.grid(row=0, column=0, sticky="w", padx=6, pady=(4, 2))

        alpha_slider = ctk.CTkSlider(
            mask_frame, from_=0.1, to=1.0, number_of_steps=90,
            command=self.on_alpha_change
        )
        alpha_slider.set(0.5)
        alpha_slider.grid(row=1, column=0, sticky="ew", padx=6, pady=(2, 4))
        self.alpha_slider = alpha_slider

        ctk.CTkButton(
            mask_frame, text="🩻 Apply Weighted Mask (Step 5)",
            command=self.apply_weighted_mask
        ).grid(row=2, column=0, sticky="ew", padx=6, pady=(4, 6))

    # ---------- file actions ----------
    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if not path:
            return
        try:
            self.processor.load_image(path)
            self.status_label.configure(text=f"Loaded: {os.path.basename(path)}")
            self.update_canvas_image()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def save_final(self):
        if self.processor.current_image is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPG", "*.jpg")]
        )
        if not path:
            return

        img_bgr = cv2.cvtColor(self.processor.current_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, img_bgr)
        messagebox.showinfo("Saved", f"Final image saved to:\n{path}")

    def save_all_steps(self):
        if self.processor.step_original is None:
            return

        path = filedialog.asksaveasfilename(
            title="Save Steps (Base Filename)",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPG", "*.jpg")]
        )
        if not path:
            return

        saved = self.processor.save_all_steps(path)
        if saved:
            messagebox.showinfo(
                "Saved",
                f"Successfully saved {len(saved)} steps:\n" + "\n".join(saved)
            )

    def reset_image(self):
        self.processor.reset_image()
        self.status_label.configure(text="Reset to original")
        self.update_canvas_image()

    # ---------- ROI / Crop events ----------
    def clear_roi(self):
        self.processor.clear_roi()
        self.status_label.configure(text="ROI cleared")
        # ลบสี่เหลี่ยมบน canvas ถ้ามี
        if self.crop_rect_id is not None:
            self.image_canvas.delete(self.crop_rect_id)
            self.crop_rect_id = None
        self.update_canvas_image()

    def on_mouse_down(self, event):
        if self.processor.current_image is None:
            return
        self.crop_start = (event.x, event.y)
        if self.crop_rect_id is not None:
            self.image_canvas.delete(self.crop_rect_id)
            self.crop_rect_id = None

    def on_mouse_drag(self, event):
        if self.crop_start is None:
            return
        x0, y0 = self.crop_start
        x1, y1 = event.x, event.y

        if self.crop_rect_id is not None:
            self.image_canvas.coords(self.crop_rect_id, x0, y0, x1, y1)
        else:
            self.crop_rect_id = self.image_canvas.create_rectangle(
                x0, y0, x1, y1, outline="red"
            )

    def on_mouse_up(self, event):
        if self.crop_start is None or self.processor.current_image is None:
            return

        x0, y0 = self.crop_start
        x1, y1 = event.x, event.y
        self.crop_start = None

        if self.tk_image is None:
            return

        # แปลงจาก coords บน canvas -> coords จริงในภาพ
        img_w = self.tk_image.width()
        img_h = self.tk_image.height()

        # หา offset ตรงกลาง canvas
        c_w = self.image_canvas.winfo_width()
        c_h = self.image_canvas.winfo_height()

        offset_x = (c_w - img_w) // 2
        offset_y = (c_h - img_h) // 2

        x0_img = int(x0 - offset_x)
        y0_img = int(y0 - offset_y)
        x1_img = int(x1 - offset_x)
        y1_img = int(y1 - offset_y)

        # scale ระหว่างภาพจริงและภาพแสดง
        h_real, w_real = self.processor.current_image.shape[:2]
        scale_x = w_real / img_w
        scale_y = h_real / img_h

        x0_real = int(x0_img * scale_x)
        y0_real = int(y0_img * scale_y)
        x1_real = int(x1_img * scale_x)
        y1_real = int(y1_img * scale_y)

        # ถ้า drag เส้นเดียว แทบไม่ crop ไม่ทำอะไร
        if abs(x1_real - x0_real) < 5 or abs(y1_real - y0_real) < 5:
            return

        self.processor.set_crop_rect(x0_real, y0_real, x1_real, y1_real)
        self.status_label.configure(text="Cropped selected area (Step 2)")
        self.update_canvas_image()

    # ---------- Processing ----------
    def apply_otsu(self):
        val = self.processor.apply_otsu_auto()
        if val is not None:
            self.thresh_var.set(val)
            self.thresh_label.configure(text=f"Threshold (Otsu): {val}")
            self.thresh_slider.set(val)
            self.status_label.configure(text=f"Auto Otsu value: {val}")
            self.update_canvas_image()

    def on_thresh_change(self, value):
        v_int = int(float(value))
        self.thresh_var.set(v_int)
        self.thresh_label.configure(text=f"Threshold: {v_int}")
        self.processor.apply_manual_threshold(v_int)
        self.update_canvas_image()

    def on_alpha_change(self, value):
        v_float = float(value)
        self.alpha_var.set(v_float)
        self.alpha_label.configure(text=f"Alpha: {v_float:.2f}")

    def apply_weighted_mask(self):
        alpha = self.alpha_var.get()
        self.processor.apply_weighted_mask(alpha)
        self.status_label.configure(text=f"Applied weighted mask (alpha={alpha:.2f})")
        self.update_canvas_image()

    # ---------- canvas update ----------
    def update_canvas_image(self):
        self.image_canvas.delete("all")

        tk_img, w, h = self.processor.get_tk_image(max_size=(1000, 900))
        if tk_img is None:
            return

        self.tk_image = tk_img

        c_w = self.image_canvas.winfo_width()
        c_h = self.image_canvas.winfo_height()

        x = (c_w - w) // 2
        y = (c_h - h) // 2
        self.image_canvas.create_image(x, y, anchor="nw", image=self.tk_image)

        # ล้าง rect เก่า
        self.crop_rect_id = None


if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    root = ctk.CTk()
    app = XrayProcessorGUI(root)
    root.mainloop()
