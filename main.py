import tkinter as tk
from tkinter import filedialog, messagebox, Scale, HORIZONTAL
import cv2
import numpy as np
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
        
        # ** ความเข้มของพื้นหลัง (0.0 = ดำสนิท, 1.0 = เท่าเดิม, 0.3 = จางลงเหลือ 30%) **
        self.fade_factor = 0.3       

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

        # ================= GUI SETUP =================
        self.panel = tk.Frame(root, width=300, bg="#f4f4f4", padx=10, pady=10, relief=tk.RAISED, bd=2)
        self.panel.pack(side=tk.RIGHT, fill=tk.Y)
        self.panel.pack_propagate(False)

        tk.Label(self.panel, text="X-Ray Visualization", font=("Helvetica", 16, "bold"), bg="#f4f4f4").pack(pady=(0, 20))

        # 1. File Group
        group_file = tk.LabelFrame(self.panel, text="File", font=("Arial", 10, "bold"), bg="#f4f4f4", padx=5, pady=5)
        group_file.pack(fill=tk.X, pady=5)
        
        tk.Button(group_file, text="📂 Load Image", command=self.load_image, bg="#dfe6e9", height=2).pack(fill=tk.X, pady=2)
        tk.Button(group_file, text="💾 Save Image", command=self.save_image, bg="#81ecec", height=2).pack(fill=tk.X, pady=2)
        tk.Button(group_file, text="↺ Reset All", command=self.reset_image, bg="#fab1a0").pack(fill=tk.X, pady=2)

        # 2. Selection Tools
        group_edit = tk.LabelFrame(self.panel, text="Step 1: Select Area", font=("Arial", 10, "bold"), bg="#f4f4f4", padx=5, pady=5)
        group_edit.pack(fill=tk.X, pady=10)
        
        self.btn_crop = tk.Button(group_edit, text="✂ Rectangle Crop", command=self.activate_crop, bg="#a29bfe", height=2)
        self.btn_crop.pack(fill=tk.X, pady=2)
        
        self.btn_poly = tk.Button(group_edit, text="⬠ Polygon ROI", command=self.activate_polygon, bg="#ff7675", height=2)
        self.btn_poly.pack(fill=tk.X, pady=2)
        
        tk.Button(group_edit, text="✖ Clear Selection", command=self.clear_roi_mask, bg="#ffcccc").pack(fill=tk.X, pady=2)

        # 3. Processing
        group_proc = tk.LabelFrame(self.panel, text="Step 2: Threshold Highlight", font=("Arial", 10, "bold"), bg="#f4f4f4", padx=5, pady=5)
        group_proc.pack(fill=tk.X, pady=10)

        tk.Button(group_proc, text="✨ Auto Otsu Level", command=self.apply_otsu_auto, bg="#ffeaa7", height=2).pack(fill=tk.X, pady=5)

        tk.Label(group_proc, text="Threshold Level Adjustment:", bg="#f4f4f4").pack(anchor=tk.W, pady=(5,0))
        
        self.thresh_val = tk.IntVar(value=0)
        self.scale_thresh = Scale(group_proc, from_=0, to=255, orient=HORIZONTAL, 
                                  variable=self.thresh_val, command=self.on_thresh_change, bg="#f4f4f4")
        self.scale_thresh.pack(fill=tk.X)
        
        tk.Label(group_proc, text="*Shows original details in highlight\nand fades the background*", font=("Arial", 8), fg="gray", bg="#f4f4f4").pack(pady=5)

        # Status Bar
        self.status_label = tk.Label(self.panel, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        # Canvas
        self.canvas_frame = tk.Frame(root, bg="#2d3436")
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, bg="#2d3436", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Events
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<Configure>", lambda e: self.update_display())

    # ================= CORE FUNCTIONS =================

    def update_base_image(self):
        if self.current_image is not None:
            self.base_image = self.current_image.copy()

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif")])
        if path:
            img = cv2.imread(path)
            if img is not None:
                self.original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.current_image = self.original_image.copy()
                self.update_base_image()
                self.roi_mask = None
                self.image_before_roi = None
                self.update_display()
                self.status_label.config(text=f"Loaded: {path.split('/')[-1]}")

    def save_image(self):
        if self.current_image is None: return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPG", "*.jpg")])
        if path:
            save_img = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, save_img)
            messagebox.showinfo("Success", "Image Saved Successfully!")

    def reset_image(self):
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.update_base_image()
            self.roi_mask = None
            self.image_before_roi = None
            self.scale_thresh.set(127)
            self.mode = None
            self.canvas.delete("all")
            self.update_display()
            self.status_label.config(text="Reset to Original")

    def clear_roi_mask(self):
        if self.image_before_roi is not None:
            self.current_image = self.image_before_roi.copy()
            self.image_before_roi = None
            self.update_base_image()
            
        self.roi_mask = None
        # Refresh Threshold on whole image
        self.on_thresh_change(self.thresh_val.get())
        self.status_label.config(text="Selection Cleared")

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

    # ================= CROP & POLYGON TOOLS =================

    def activate_crop(self):
        self.mode = 'crop'
        self.canvas.config(cursor="cross")
        self.status_label.config(text="Mode: Rectangle Crop")
        self.clear_overlays()

    def activate_polygon(self):
        self.mode = 'polygon'
        self.canvas.config(cursor="dot")
        self.poly_points = []
        self.temp_points = []
        self.status_label.config(text="Polygon: Left Click to add, Right Click to finish")
        self.clear_overlays()
        if self.roi_mask is not None: self.clear_roi_mask()

    def clear_overlays(self):
        self.canvas.delete("selection")
        self.canvas.delete("poly")

    def on_mouse_down(self, event):
        if self.current_image is None: return

        if self.mode == 'crop':
            self.drawing = True
            self.start_x = event.x
            self.start_y = event.y
            self.canvas.delete("selection")
            self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, 
                                                        outline='yellow', width=1, tags="selection")

        elif self.mode == 'polygon':
            real_x, real_y = self.get_real_coords(event.x, event.y)
            self.poly_points.append((real_x, real_y))
            self.temp_points.append((event.x, event.y))
            
            r = 2
            self.canvas.create_oval(event.x-r, event.y-r, event.x+r, event.y+r, fill='cyan', outline='cyan', tags="poly")
            if len(self.temp_points) > 1:
                self.canvas.create_line(self.temp_points[-2], self.temp_points[-1], fill='cyan', width=1, tags="poly")

    def on_mouse_drag(self, event):
        if self.mode == 'crop' and self.drawing:
            self.canvas.coords("selection", self.start_x, self.start_y, event.x, event.y)

    def on_mouse_up(self, event):
        if self.mode == 'crop' and self.drawing:
            self.drawing = False
            x1, y1 = self.get_real_coords(self.start_x, self.start_y)
            x2, y2 = self.get_real_coords(event.x, event.y)
            
            rx1, rx2 = sorted([x1, x2])
            ry1, ry2 = sorted([y1, y2])
            
            if (rx2 - rx1) > 10 and (ry2 - ry1) > 10:
                if messagebox.askyesno("Confirm Crop", "Crop this area?"):
                    self.current_image = self.current_image[ry1:ry2, rx1:rx2]
                    self.update_base_image()
                    self.roi_mask = None 
                    self.image_before_roi = None
                    self.update_display()
                    self.mode = None
                    self.canvas.config(cursor="arrow")
            self.canvas.delete("selection")

    def on_right_click(self, event):
        if self.mode == 'polygon' and len(self.poly_points) > 2:
            self.canvas.create_line(self.temp_points[-1], self.temp_points[0], fill='cyan', width=1, tags="poly")
            
            h, w = self.current_image.shape[:2]
            self.roi_mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array([self.poly_points], dtype=np.int32)
            cv2.fillPoly(self.roi_mask, pts, 255) 

            # Backup
            self.image_before_roi = self.current_image.copy()
            self.update_base_image() # Base now has the line drawn if we kept it, but wait...
            
            # วาดเส้น ROI ลงบน Base เลยเพื่อให้เห็นขอบเขตชัดเจนเวลาปรับ
            cv2.polylines(self.base_image, [pts], True, (0, 255, 255), thickness=1)

            self.update_display()
            self.mode = None
            self.canvas.config(cursor="arrow")
            self.status_label.config(text="Area Selected. Effect applies INSIDE area.")
            self.clear_overlays()
            
            # เรียกใช้ Threshold ทันทีเมื่อเลือกเสร็จ
            self.on_thresh_change(self.thresh_val.get())

    # ================= LOGIC: FADED HIGHLIGHT =================

    def apply_otsu_auto(self):
        if self.base_image is None: return
        
        # 1. แปลงภาพเป็น Grayscale
        gray = cv2.cvtColor(self.base_image, cv2.COLOR_RGB2GRAY)
        
        # 2. คำนวณ Otsu
        if self.roi_mask is not None:
            # กรณีมี ROI ใช้ Otsu ทั้งภาพเพื่อความเสถียร (หรือจะใช้ masked array ก็ได้แต่วิธีนี้ง่ายกว่า)
            otsu_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            otsu_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 3. อัพเดตค่าในตัวแปร Slider
        val = int(otsu_val)
        self.thresh_val.set(val)
        
        # 4. [สำคัญ!] สั่งให้ประมวลผลภาพทันที เพื่อให้ภาพบนจอเปลี่ยน
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
        # ksize (Kernal Size) ต้องเป็นเลขคี่ ยิ่งมากยิ่งฟุ้ง (เช่น (15,15), (31,31))
        # ปรับค่านี้เพื่อเพิ่ม/ลดความฟุ้งของขอบกระดูก
        blur_amount = (21, 21) 
        weighted_mask = cv2.GaussianBlur(mask_binary, blur_amount, 0)
        
        # 4. แปลงเป็น Alpha Channel (ค่า 0.0 - 1.0)
        # พื้นที่สีขาวจะมี alpha ใกล้ 1, สีดำใกล้ 0, ขอบฟุ้งๆ จะอยู่ระหว่างกลาง
        alpha = weighted_mask.astype(np.float32) / 255.0
        
        # 5. เตรียม Layer สำหรับ Blending (แปลงเป็น float32 เพื่อการคำนวณ)
        img_float = img_gray.astype(np.float32)
        
        # Layer 1: Foreground (ส่วนสว่าง/กระดูก) - ใช้ภาพเดิม
        foreground = img_float
        
        # Layer 2: Background (ส่วนมืด/เนื้อเยื่อ) - ทำให้จางลง
        background = img_float * self.fade_factor
        
        # 6. Apply Alpha Blending (Standard Formula)
        # Final = (FG * alpha) + (BG * (1 - alpha))
        # ตรงไหน alpha สูงจะเห็น foreground ชัด, ตรงไหน alpha ต่ำจะเห็น background จางๆ
        blended_float = (foreground * alpha) + (background * (1.0 - alpha))
        
        # --------------------------------------------------

        # แปลงกลับเป็น uint8
        output_uint8 = np.clip(blended_float, 0, 255).astype(np.uint8)
        
        # แปลงกลับเป็น RGB เพื่อแสดงผล (ผลลัพธ์จะเป็นโทนขาวดำที่นุ่มนวล)
        processed_rgb = cv2.cvtColor(output_uint8, cv2.COLOR_GRAY2RGB)
        
        # 7. Apply ROI Logic (ถ้ามีการเลือกพื้นที่)
        if self.roi_mask is not None:
            mask_3ch = cv2.cvtColor(self.roi_mask, cv2.COLOR_GRAY2RGB)
            # ใน ROI -> ใช้ภาพที่ Blend แล้ว
            # นอก ROI -> ใช้ภาพสีต้นฉบับ (base_image)
            final_image = np.where(mask_3ch == 255, processed_rgb, img_source_rgb)
        else:
            final_image = processed_rgb

        self.current_image = final_image
        self.update_display()

if __name__ == "__main__":
    root = tk.Tk()
    app = MonkeyXRayApp(root)
    root.mainloop()