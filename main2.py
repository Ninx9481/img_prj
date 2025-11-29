import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk

class MonkeyXRayApp:
    def __init__(self, root):
        self.root = root
        self.root.title("X-ray Image Processing")
        self.root.geometry("1280x720")
        self.root.state('zoomed') 

        # --- ตัวแปรจัดการภาพ ---
        self.original_image = None   
        self.base_image = None       
        self.current_image = None    
        self.tk_image = None         

        # --- พารามิเตอร์การแสดงผล ---
        self.scale = 1.0        
        self.offset_x = 0       
        self.offset_y = 0       

        # --- ตัวแปร ROI / Mask ---
        self.roi_mask = None    
        self.image_before_roi = None 
        
        self.mode = None        
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.rect_id = None
        
        self.poly_points = []   
        self.temp_points = []   

        # ================= GUI SETUP (ใช้สไตล์จาก GUI8) =================
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        # main frame
        self.main_frame = ctk.CTkFrame(self.root, corner_radius=0)
        self.main_frame.pack(fill="both", expand=True)

        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=3)
        self.main_frame.grid_columnconfigure(1, weight=1)

        # toolbar เหมือน GUI8
        toolbar = ctk.CTkFrame(
            self.main_frame,
            corner_radius=0,
            fg_color=("white", "#1E1E1E"),
        )
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        toolbar.grid_columnconfigure(0, weight=1)

        title_label = ctk.CTkLabel(
            toolbar,
            text="X-ray Image Processing",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        title_label.grid(row=0, column=0, padx=16, pady=6, sticky="w")

        # ปุ่มบน toolbar (ใช้ logic เดิมจาก main.py)
        btn_frame = ctk.CTkFrame(toolbar, fg_color="transparent")
        btn_frame.grid(row=0, column=1, padx=12, pady=6, sticky="e")

        ctk.CTkButton(
            btn_frame,
            text="📂 Load image",
            width=110,
            command=self.load_image,
        ).pack(side="left", padx=4)

        ctk.CTkButton(
            btn_frame,
            text="💾 Save image",
            width=110,
            command=self.save_image,
        ).pack(side="left", padx=4)

        ctk.CTkButton(
            btn_frame,
            text="↺ Reset all",
            width=110,
            fg_color="#FF9F0A",
            hover_color="#FFB340",
            command=self.reset_image,
        ).pack(side="left", padx=4)

        # ========== พื้นที่ภาพ (ซ้าย) ==========
        canvas_container = ctk.CTkFrame(self.main_frame, fg_color="#1C1C1E", corner_radius=18)
        canvas_container.grid(row=1, column=0, sticky="nsew", padx=(16, 8), pady=(0, 16))

        self.canvas = tk.Canvas(canvas_container, bg="#111111", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=8)

        # ========== แผง control (ขวา) ==========
        self.panel = ctk.CTkFrame(self.main_frame, fg_color=("white", "#2C2C2E"), corner_radius=18)
        self.panel.grid(row=1, column=1, sticky="nsew", padx=(8, 16), pady=(0, 16))
        self.panel.grid_rowconfigure(3, weight=1)  # ให้ส่วน slider ขยายได้

        header_label = ctk.CTkLabel(
            self.panel,
            text="X-Ray Visualization",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        header_label.grid(row=0, column=0, columnspan=1, padx=12, pady=(12, 6), sticky="w")

        # 1) File Group (ตอนนี้อยู่บน toolbar แล้ว เลยไม่ต้องทำซ้ำ)

        # 2) Selection tools
        group_edit = ctk.CTkFrame(self.panel, fg_color="transparent")
        group_edit.grid(row=1, column=0, sticky="ew", padx=12, pady=(4, 8))
        group_edit.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            group_edit,
            text="Step 1 : Select area",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 4))

        self.btn_crop = ctk.CTkButton(
            group_edit,
            text="✂ Rectangle Crop",
            command=self.activate_crop,
        )
        self.btn_crop.grid(row=1, column=0, sticky="ew", pady=2)

        self.btn_poly = ctk.CTkButton(
            group_edit,
            text="⬠ Polygon ROI",
            command=self.activate_polygon,
            fg_color="#FF3B30",
            hover_color="#FF5E57",
        )
        self.btn_poly.grid(row=2, column=0, sticky="ew", pady=2)

        ctk.CTkButton(
            group_edit,
            text="✖ Clear Selection",
            command=self.clear_roi_mask,
            fg_color="#E5E5EA",
            hover_color="#D1D1D6",
            text_color="#1D1D1F",
        ).grid(row=3, column=0, sticky="ew", pady=(2, 0))

        # 3) Processing (Otsu + slider)
        group_proc = ctk.CTkFrame(self.panel, fg_color="transparent")
        group_proc.grid(row=2, column=0, sticky="ew", padx=12, pady=(4, 8))
        group_proc.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            group_proc,
            text="Step 2 : Threshold highlight",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 4))

        ctk.CTkButton(
            group_proc,
            text="✨ Auto Otsu Level",
            command=self.apply_otsu_auto,
        ).grid(row=1, column=0, sticky="ew", pady=(0, 6))

        ctk.CTkLabel(
            group_proc,
            text="Threshold Level Adjustment",
        ).grid(row=2, column=0, sticky="w")

        self.thresh_val = tk.IntVar(value=0)

        # slider ของ customtkinter ใช้ callback แยก เพื่อ sync กับ logic เดิม
        def _slider_changed(value):
            self.thresh_val.set(int(value))
            self.on_thresh_change(value)

        self.scale_thresh = ctk.CTkSlider(
            group_proc,
            from_=0,
            to=255,
            number_of_steps=255,
            command=_slider_changed,
        )
        self.scale_thresh.grid(row=3, column=0, sticky="ew", pady=(4, 4))

        ctk.CTkLabel(
            group_proc,
            text="*Shows original details in highlighted bone area*",
            font=ctk.CTkFont(size=11),
            text_color="#8E8E93",
            wraplength=220,
            justify="left",
        ).grid(row=4, column=0, sticky="w", pady=(2, 0))

        # Status bar ด้านล่าง panel
        self.status_label = ctk.CTkLabel(
            self.panel,
            text="Ready",
            anchor="w",
        )
        self.status_label.grid(row=4, column=0, sticky="ew", padx=12, pady=(8, 12))

        # Events บน canvas
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<Configure>", lambda e: self.update_display())

    # ================= CORE FUNCTIONS =================

    def update_base_image(self):
        if self.original_image is None: 
            return
        self.base_image = self.original_image.copy()
        self.current_image = self.base_image.copy()
        self.roi_mask = None
        self.image_before_roi = None
        self.poly_points = []
        self.temp_points = []
        self.rect_id = None
        self.clear_overlays()
        self.update_display()
        self.status_label.configure(text="Base image reset")

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"),
                ("All files", "*.*")
            ]
        )
        if not file_path:
            return

        img_bgr = cv2.imread(file_path)
        if img_bgr is None:
            messagebox.showerror("Error", "Cannot load image")
            return
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.original_image = img_rgb
        self.update_base_image()
        self.status_label.configure(text=f"Loaded: {file_path}")

    def save_image(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "No image to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg;*.jpeg"),
                ("BMP files", "*.bmp"),
                ("TIFF files", "*.tif;*.tiff"),
                ("All files", "*.*")
            ]
        )
        if not file_path:
            return
        
        img_bgr = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(file_path, img_bgr)
        self.status_label.configure(text=f"Saved: {file_path}")

    def reset_image(self):
        if self.original_image is None:
            return
        self.update_base_image()
        self.thresh_val.set(0)
        self.scale_thresh.set(0)
        self.status_label.configure(text="Reset all changes")

    def clear_roi_mask(self):
        self.roi_mask = None
        self.image_before_roi = None
        self.poly_points = []
        self.temp_points = []
        self.rect_id = None
        self.mode = None
        self.drawing = False

        # ล้าง overlay บน canvas
        self.clear_overlays()
        # Refresh Threshold on whole image
        self.on_thresh_change(self.thresh_val.get())
        self.status_label.config(text="Selection Cleared")

    def clear_overlays(self):
        if self.canvas is not None:
            self.canvas.delete("rect_roi")
            self.canvas.delete("poly_roi")

    def update_display(self):
        if self.current_image is None: return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 2 or ch < 2: return

        ih, iw = self.current_image.shape[:2]
        ratio_w = cw / iw
        ratio_h = ch / ih
        self.scale = min(ratio_w, ratio_h, 1.0) 

        new_w = int(iw * self.scale)
        new_h = int(ih * self.scale)
        self.offset_x = (cw - new_w) // 2
        self.offset_y = (ch - new_h) // 2

        pil_img = Image.fromarray(self.current_image)
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(pil_img)

        self.canvas.delete("img_tag")
        self.canvas.create_image(cw//2, ch//2, anchor=tk.CENTER, image=self.tk_image, tags="img_tag")
        self.canvas.tag_lower("img_tag")

    def get_real_coords(self, cx, cy):
        if self.current_image is None: return 0, 0
        rx = (cx - self.offset_x) / self.scale
        ry = (cy - self.offset_y) / self.scale
        h, w = self.current_image.shape[:2]
        rx = max(0, min(rx, w))
        ry = max(0, min(ry, h))
        return int(rx), int(ry)

    def activate_crop(self):
        self.mode = "rect"
        self.drawing = False
        self.status_label.configure(text="Rectangle Crop Mode: drag on image")
        self.btn_crop.configure(fg_color="#74b9ff")
        self.btn_poly.configure(fg_color="#FF3B30")

    def activate_polygon(self):
        self.mode = "poly"
        self.drawing = False
        self.poly_points = []
        self.temp_points = []
        self.clear_overlays()
        self.status_label.configure(text="Polygon ROI Mode: left-click to add points, right-click to close")
        self.btn_poly.configure(fg_color="#ff7675")
        self.btn_crop.configure(fg_color="#3498db")

    # ================= MOUSE EVENTS =================

    def on_mouse_down(self, event):
        if self.current_image is None: return
        if self.mode == "rect":
            self.drawing = True
            self.start_x, self.start_y = event.x, event.y
            self.clear_overlays()
            self.rect_id = self.canvas.create_rectangle(
                self.start_x, self.start_y, self.start_x, self.start_y,
                outline="yellow", width=2, tags="rect_roi"
            )
        elif self.mode == "poly":
            rx, ry = self.get_real_coords(event.x, event.y)
            self.poly_points.append((rx, ry))
            self.temp_points.append((event.x, event.y))
            if len(self.temp_points) > 1:
                self.canvas.create_line(
                    self.temp_points[-2][0], self.temp_points[-2][1],
                    self.temp_points[-1][0], self.temp_points[-1][1],
                    fill="cyan", width=2, tags="poly_roi"
                )

    def on_mouse_drag(self, event):
        if self.mode == "rect" and self.drawing and self.rect_id is not None:
            self.canvas.coords(self.rect_id, self.start_x, self.start_y, event.x, event.y)

    def on_mouse_up(self, event):
        if self.mode == "rect" and self.drawing:
            self.drawing = False
            end_x, end_y = event.x, event.y
            x1, y1 = min(self.start_x, end_x), min(self.start_y, end_y)
            x2, y2 = max(self.start_x, end_x), max(self.start_y, end_y)
            rx1, ry1 = self.get_real_coords(x1, y1)
            rx2, ry2 = self.get_real_coords(x2, y2)

            if rx2 - rx1 > 10 and ry2 - ry1 > 10:
                self.image_before_roi = self.base_image.copy()
                self.roi_mask = np.zeros(self.base_image.shape[:2], dtype=np.uint8)
                self.roi_mask[ry1:ry2, rx1:rx2] = 255
                self.status_label.config(text=f"Rectangle ROI set: ({rx1},{ry1})-({rx2},{ry2})")
            else:
                self.status_label.config(text="Rectangle too small, ignored")
                self.roi_mask = None
                self.image_before_roi = None
                self.clear_overlays()

            self.on_thresh_change(self.thresh_val.get())

    def on_right_click(self, event):
        if self.mode == "poly" and len(self.poly_points) >= 3:
            self.image_before_roi = self.base_image.copy()
            self.roi_mask = np.zeros(self.base_image.shape[:2], dtype=np.uint8)
            pts = np.array(self.poly_points, dtype=np.int32)
            cv2.fillPoly(self.roi_mask, [pts], 255)
            self.status_label.config(text=f"Polygon ROI set with {len(self.poly_points)} points")
            self.clear_overlays()
            self.on_thresh_change(self.thresh_val.get())
        elif self.mode == "poly":
            self.status_label.config(text="Polygon canceled")
            self.poly_points = []
            self.temp_points = []
            self.clear_overlays()

    # ================= LOGIC: FADED HIGHLIGHT =================

    def apply_otsu_auto(self):
        if self.base_image is None: return
        
        # 1. แปลงภาพเป็น Grayscale
        gray = cv2.cvtColor(self.base_image, cv2.COLOR_RGB2GRAY)
        
        # 2. คำนวณ Otsu (ตอนนี้ใช้ทั้งภาพ)
        otsu_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 3. อัพเดตค่าในตัวแปร Slider
        val = int(otsu_val)
        self.thresh_val.set(val)
        self.scale_thresh.set(val)
        
        # 4. ประมวลผลภาพทันที
        self.on_thresh_change(val)
        
        self.status_label.config(text=f"Auto Otsu Value: {val}")

    def on_thresh_change(self, val):
        if self.base_image is None: return
        
        threshold_value = int(val)
        
        # 1. เตรียมข้อมูล (ใช้ base_image เป็นหลัก)
        img_source_rgb = self.base_image.copy()
        # แปลงเป็น Grayscale เพื่อคำนวณ
        img_gray = cv2.cvtColor(img_source_rgb, cv2.COLOR_RGB2GRAY)
        
        # 2. สร้าง Hard Binary Mask (0 หรือ 255) ตามค่า Slider
        _, mask_binary = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY)
        
        # --- ส่วนที่เพิ่มเติม: สร้าง Weighted Mask / Alpha ---
        
        # 3. ทำให้ Mask นุ่มนวลขึ้นด้วย Gaussian Blur
        blur_amount = (21, 21) 
        weighted_mask = cv2.GaussianBlur(mask_binary, blur_amount, 0)
        
        # 4. แปลงเป็น Alpha Channel (ค่า 0.0 - 1.0)
        alpha = weighted_mask.astype(np.float32) / 255.0
        
        # 5. เตรียม Layer สำหรับ Blending (แปลงเป็น float32 เพื่อการคำนวณ)
        img_float = img_gray.astype(np.float32)
        
        # 6. กำหนดภาพ Background ให้เป็นเวอร์ชันที่มืดลงเล็กน้อย (เช่น 60% ของเดิม)
        dark_background = img_float * 0.6
        
        # 7. สร้างภาพ Highlight (กระดูก) ให้สว่างขึ้น (เช่น 1.3 เท่า แต่ไม่เกิน 255)
        bright_foreground = np.clip(img_float * 1.3, 0, 255)
        
        # 8. Blend ด้วยสมการ:
        #    output = alpha * bright_foreground + (1 - alpha) * dark_background
        blended = alpha * bright_foreground + (1 - alpha) * dark_background
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        # 9. แปลงกลับเป็น RGB เพื่อแสดงผล
        processed_rgb = cv2.cvtColor(blended, cv2.COLOR_GRAY2RGB)
        
        if self.roi_mask is not None:
            mask_3ch = cv2.cvtColor(self.roi_mask, cv2.COLOR_GRAY2RGB)
            final_image = np.where(mask_3ch == 255, processed_rgb, img_source_rgb)
        else:
            final_image = processed_rgb

        self.current_image = final_image
        self.update_display()

if __name__ == "__main__":
    root = ctk.CTk()
    app = MonkeyXRayApp(root)
    root.mainloop()
