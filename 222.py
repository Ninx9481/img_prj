import tkinter as tk
from tkinter import filedialog, messagebox, Scale, HORIZONTAL
import cv2
import numpy as np
from PIL import Image, ImageTk
import os

class MonkeyXRayApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Monkey X-Ray: Final Complete")
        self.root.geometry("1280x720")
        self.root.state('zoomed') 

        # --- ตัวแปรจัดการภาพหลัก ---
        self.original_image = None   
        self.base_image = None       
        self.current_image = None    
        self.tk_image = None         

        # --- ตัวแปรเก็บภาพแต่ละขั้นตอน (6 Steps) ---
        self.step_original = None    # 1. Original Loaded
        self.step_cropped = None     # 2. After Crop
        self.step_polygon = None     # 3. Visual with Polygon Lines
        self.current_binary = None   # 4. Otsu Binary
        self.current_weighted = None # 5. Weighted Mask
        # Step 6 is self.current_image (Final)

        # --- พารามิเตอร์การแสดงผล ---
        self.scale = 1.0        
        self.offset_x = 0       
        self.offset_y = 0
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
        self.panel = tk.Frame(root, width=320, bg="#f4f4f4", padx=10, pady=10, relief=tk.RAISED, bd=2)
        self.panel.pack(side=tk.RIGHT, fill=tk.Y)
        self.panel.pack_propagate(False)

        tk.Label(self.panel, text="X-Ray Analysis Tools", font=("Helvetica", 16, "bold"), bg="#f4f4f4", fg="black").pack(pady=(0, 20))

        # 1. File Group
        group_file = tk.LabelFrame(self.panel, text="File Operations", font=("Arial", 10, "bold"), bg="#f4f4f4", fg="black", padx=5, pady=5)
        group_file.pack(fill=tk.X, pady=5)
        
        tk.Button(group_file, text="📂 Load Image", command=self.load_image, bg="#dfe6e9", height=2).pack(fill=tk.X, pady=2)
        tk.Button(group_file, text="💾 Save Final Only", command=self.save_image, bg="#81ecec", height=2).pack(fill=tk.X, pady=2)
        
        # ปุ่ม Save ทุกขั้นตอน
        btn_steps = tk.Button(group_file, text="📑 Save All 6 Steps", command=self.save_process_steps, bg="#fab1a0", height=2)
        btn_steps.pack(fill=tk.X, pady=2)
        
        tk.Button(group_file, text="↺ Reset All", command=self.reset_image, bg="#ffdd59").pack(fill=tk.X, pady=2)

        # 2. Selection Tools
        group_edit = tk.LabelFrame(self.panel, text="Step 1: Select Area", font=("Arial", 10, "bold"), bg="#f4f4f4", fg="black", padx=5, pady=5)
        group_edit.pack(fill=tk.X, pady=10)
        
        self.btn_crop = tk.Button(group_edit, text="✂ Rectangle Crop", command=self.activate_crop, bg="#a29bfe", height=2)
        self.btn_crop.pack(fill=tk.X, pady=2)
        
        self.btn_poly = tk.Button(group_edit, text="⬠ Polygon ROI", command=self.activate_polygon, bg="#ff7675", height=2)
        self.btn_poly.pack(fill=tk.X, pady=2)
        
        tk.Button(group_edit, text="✖ Clear Selection", command=self.clear_roi_mask, bg="#ffcccc").pack(fill=tk.X, pady=2)

        # 3. Processing
        group_proc = tk.LabelFrame(self.panel, text="Step 2: Processing", font=("Arial", 10, "bold"), bg="#f4f4f4", fg="black", padx=5, pady=5)
        group_proc.pack(fill=tk.X, pady=10)

        tk.Button(group_proc, text="✨ Auto Otsu Level", command=self.apply_otsu_auto, bg="#ffeaa7", height=2).pack(fill=tk.X, pady=5)

        tk.Label(group_proc, text="Threshold Adjustment:", bg="#f4f4f4", fg="black").pack(anchor=tk.W, pady=(5,0))
        
        self.thresh_val = tk.IntVar(value=0)
        self.scale_thresh = Scale(group_proc, from_=0, to=255, orient=HORIZONTAL, 
                                  variable=self.thresh_val, command=self.on_thresh_change, bg="#f4f4f4", fg="black")
        self.scale_thresh.pack(fill=tk.X)
        
        tk.Label(group_proc, text="*Generates weighted mask\nfor soft blending*", font=("Arial", 8), fg="gray", bg="#f4f4f4").pack(pady=5)

        self.status_label = tk.Label(self.panel, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas_frame = tk.Frame(root, bg="#2d3436")
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, bg="#2d3436", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Events
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        
        # --- แก้ไขส่วนนี้เพื่อรองรับ MAC และ Double Click ---
        # 1. คลิกขวาแบบ Windows/Linux
        self.canvas.bind("<Button-3>", self.finish_polygon) 
        # 2. คลิกขวาแบบ Mac (บางเวอร์ชันใช้ Button-2)
        self.canvas.bind("<Button-2>", self.finish_polygon) 
        # 3. ดับเบิ้ลคลิกซ้าย (Double Click)
        self.canvas.bind("<Double-Button-1>", self.finish_polygon)
        # 4. ปุ่ม Enter (เผื่อไว้)
        self.root.bind("<Return>", self.finish_polygon) 
        
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
                
                # --- [SAVE STATE] Step 1 & 2 Init ---
                self.step_original = self.original_image.copy()
                self.step_cropped = self.original_image.copy() # ยังไม่ crop ก็คือภาพเดิม
                self.step_polygon = None
                self.current_binary = None
                self.current_weighted = None
                
                self.update_display()
                self.status_label.config(text=f"Loaded: {path.split('/')[-1]}")

    def save_image(self):
        if self.current_image is None: return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPG", "*.jpg")])
        if path:
            save_img = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, save_img)
            messagebox.showinfo("Success", "Final Image Saved.")

    def save_process_steps(self):
        """ Save all 6 Steps """
        if self.step_original is None: return

        # ถามชื่อไฟล์หลัก
        path = filedialog.asksaveasfilename(title="Save Steps (Base Filename)", defaultextension=".png", 
                                            filetypes=[("PNG", "*.png"), ("JPG", "*.jpg")])
        if path:
            base_name = os.path.splitext(path)[0]
            ext = os.path.splitext(path)[1]
            if not ext: ext = ".png"
            
            saved_list = []
            try:
                # 1. Original
                if self.step_original is not None:
                    cv2.imwrite(f"{base_name}_01_original{ext}", cv2.cvtColor(self.step_original, cv2.COLOR_RGB2BGR))
                    saved_list.append("01_original")

                # 2. Cropped
                if self.step_cropped is not None:
                    cv2.imwrite(f"{base_name}_02_cropped{ext}", cv2.cvtColor(self.step_cropped, cv2.COLOR_RGB2BGR))
                    saved_list.append("02_cropped")

                # 3. Polygon Visual
                if self.step_polygon is not None:
                    cv2.imwrite(f"{base_name}_03_polygon{ext}", cv2.cvtColor(self.step_polygon, cv2.COLOR_RGB2BGR))
                    saved_list.append("03_polygon")

                # 4. Otsu Binary
                if self.current_binary is not None:
                    cv2.imwrite(f"{base_name}_04_otsu_binary{ext}", self.current_binary)
                    saved_list.append("04_otsu_binary")
                
                # 5. Weighted Mask
                if self.current_weighted is not None:
                    cv2.imwrite(f"{base_name}_05_weighted_mask{ext}", self.current_weighted)
                    saved_list.append("05_weighted_mask")
                
                # 6. Final Result
                if self.current_image is not None:
                    cv2.imwrite(f"{base_name}_06_final_result{ext}", cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR))
                    saved_list.append("06_final_result")

                messagebox.showinfo("Saved", f"Successfully saved {len(saved_list)} steps:\n" + "\n".join(saved_list))
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {str(e)}")

    def reset_image(self):
        """ รีเซ็ตภาพกลับไปยังสถานะเริ่มต้น โดยไม่เปิดหน้าต่างเลือกไฟล์ใหม่ """
        if self.original_image is not None:
            # 1. คืนค่าภาพจาก original
            self.current_image = self.original_image.copy()
            self.update_base_image()
            
            # 2. เคลียร์ค่าตัวแปร Logic ทั้งหมด
            self.roi_mask = None
            self.image_before_roi = None
            
            # 3. รีเซ็ต Step การบันทึก
            self.step_original = self.original_image.copy()
            self.step_cropped = self.original_image.copy()
            self.step_polygon = None
            self.current_binary = None
            self.current_weighted = None
            
            # 4. รีเซ็ต GUI
            self.scale_thresh.set(0)
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
        self.step_polygon = None # Clear Polygon step
        
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
                    
                    # --- [SAVE STATE] Step 2: Cropped ---
                    self.step_cropped = self.current_image.copy()
                    self.step_polygon = None # Crop ใหม่ Polygon เก่าหาย
                    
                    self.update_display()
                    self.mode = None
                    self.canvas.config(cursor="arrow")
            self.canvas.delete("selection")

    def finish_polygon(self, event=None):
        # ตรวจสอบว่ากำลังวาด Polygon อยู่หรือไม่
        if self.mode == 'polygon' and len(self.poly_points) > 2:
            
            # วาดเส้นปิดใน Canvas (เส้นร่าง)
            self.canvas.create_line(self.temp_points[-1], self.temp_points[0], fill='cyan', width=1, tags="poly")
            
            h, w = self.current_image.shape[:2]
            self.roi_mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array([self.poly_points], dtype=np.int32)
            cv2.fillPoly(self.roi_mask, pts, 255) 

            # Backup
            self.image_before_roi = self.current_image.copy()
            self.update_base_image() 
            
            # วาดเส้น ROI ลงบนภาพจริง
            cv2.polylines(self.base_image, [pts], True, (0, 255, 255), thickness=1)
            
            # [SAVE STATE] Step 3: Polygon Visual
            self.step_polygon = self.base_image.copy()

            self.update_display()
            self.mode = None
            self.canvas.config(cursor="arrow")
            self.status_label.config(text="Area Selected.")
            self.clear_overlays()
            
            # Auto Update Threshold
            self.on_thresh_change(self.thresh_val.get())

    # ================= LOGIC: ALPHA BLENDING & SAVING =================

    def apply_otsu_auto(self):
        if self.base_image is None: return
        
        gray = cv2.cvtColor(self.base_image, cv2.COLOR_RGB2GRAY)
        if self.roi_mask is not None:
             otsu_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            otsu_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        val = int(otsu_val)
        self.thresh_val.set(val)
        self.on_thresh_change(val)
        self.status_label.config(text=f"Auto Otsu Value: {val}")

    def on_thresh_change(self, val):
        if self.base_image is None: return
        
        threshold_value = int(val)
        img_source_rgb = self.base_image.copy()
        img_gray = cv2.cvtColor(img_source_rgb, cv2.COLOR_RGB2GRAY)
        
        # 1. Step 4: Binary Otsu/Threshold
        _, mask_binary = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY)
        
        # [SAVE STATE]
        self.current_binary = mask_binary.copy() 
        
        # 2. Step 5: Weighted Mask (Alpha)
        blur_amount = (21, 21)
        weighted_mask = cv2.GaussianBlur(mask_binary, blur_amount, 0)
        
        # [SAVE STATE]
        self.current_weighted = weighted_mask.copy()
        
        # 3. Calculation
        alpha = weighted_mask.astype(np.float32) / 255.0
        img_float = img_gray.astype(np.float32)
        
        foreground = img_float
        background = img_float * self.fade_factor
        
        blended_float = (foreground * alpha) + (background * (1.0 - alpha))
        
        output_uint8 = np.clip(blended_float, 0, 255).astype(np.uint8)
        processed_rgb = cv2.cvtColor(output_uint8, cv2.COLOR_GRAY2RGB)
        
        # 4. Apply ROI
        if self.roi_mask is not None:
            mask_3ch = cv2.cvtColor(self.roi_mask, cv2.COLOR_GRAY2RGB)
            final_image = np.where(mask_3ch == 255, processed_rgb, img_source_rgb)
        else:
            final_image = processed_rgb

        # Step 6: Final Result (Current Image)
        self.current_image = final_image
        self.update_display()

if __name__ == "__main__":
    root = tk.Tk()
    app = MonkeyXRayApp(root)
    root.mainloop()