# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import filedialog, messagebox

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector, PolygonSelector

import numpy as np
import cv2


class XrayApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Monkey X-ray Rib Removal (Gabor Filter)")

        # ----------------- state -----------------
        self.original_image = None   # ภาพต้นฉบับ
        self.current_image = None    # ภาพที่ถูกแก้ไขปัจจุบัน
        self.roi_mask = None         # mask จาก polygon ROI

        self.rect_selector = None
        self.poly_selector = None

        # ----------------- layout หลัก -----------------
        top_frame = tk.Frame(root)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        main_frame = tk.Frame(root)
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        canvas_frame = tk.Frame(main_frame)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        control_frame = tk.Frame(main_frame, padx=5, pady=5,
                                 relief=tk.GROOVE, bd=2)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # ----------------- figure matplotlib -----------------
        self.fig = Figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')

        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ----------------- ปุ่มด้านบน -----------------
        tk.Button(top_frame, text="นำเข้าภาพ",
                  command=self.load_image).pack(side=tk.LEFT, padx=2, pady=2)

        tk.Button(top_frame, text="รีเซ็ตภาพ",
                  command=self.reset_image).pack(side=tk.LEFT, padx=2, pady=2)

        tk.Button(top_frame, text="ครอปจากเมาส์",
                  command=self.activate_crop).pack(side=tk.LEFT, padx=2, pady=2)

        tk.Button(top_frame, text="Polygon ROI",
                  command=self.activate_polygon_roi).pack(side=tk.LEFT, padx=2, pady=2)

        tk.Button(top_frame, text="เซฟภาพ",
                  command=self.save_image).pack(side=tk.LEFT, padx=2, pady=2)

        # ----------------- แถบพารามิเตอร์ Gabor -----------------
        gabor_frame = tk.LabelFrame(control_frame, text="Gabor filter",
                                    padx=5, pady=5)
        gabor_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        self.theta_var = tk.IntVar(value=0)       # องศา
        self.lambda_var = tk.IntVar(value=15)     # ความยาวคลื่น
        self.sigma_var = tk.DoubleVar(value=4.0)  # sigma
        self.gamma_var = tk.IntVar(value=5)       # 0.1 - 2.0 (คูณ 0.1)
        self.ksize_var = tk.IntVar(value=31)      # ขนาด kernel (เลขคี่)
        self.thresh_var = tk.IntVar(value=50)     # threshold หา mask ซี่โครง

        # theta
        tk.Label(gabor_frame, text="Theta (deg)").pack(anchor="w")
        tk.Scale(gabor_frame, from_=0, to=180, orient=tk.HORIZONTAL,
                 variable=self.theta_var).pack(fill=tk.X)

        # lambda
        tk.Label(gabor_frame, text="Lambda (px)").pack(anchor="w")
        tk.Scale(gabor_frame, from_=5, to=50, orient=tk.HORIZONTAL,
                 variable=self.lambda_var).pack(fill=tk.X)

        # sigma
        tk.Label(gabor_frame, text="Sigma").pack(anchor="w")
        tk.Scale(gabor_frame, from_=1, to=20, resolution=0.5,
                 orient=tk.HORIZONTAL, variable=self.sigma_var).pack(fill=tk.X)

        # gamma
        tk.Label(gabor_frame, text="Gamma (x0.1)").pack(anchor="w")
        tk.Scale(gabor_frame, from_=1, to=20, orient=tk.HORIZONTAL,
                 variable=self.gamma_var).pack(fill=tk.X)

        # kernel size
        tk.Label(gabor_frame, text="Kernel size").pack(anchor="w")
        tk.Scale(gabor_frame, from_=7, to=101, orient=tk.HORIZONTAL,
                 variable=self.ksize_var).pack(fill=tk.X)

        # threshold
        tk.Label(gabor_frame, text="Threshold").pack(anchor="w")
        tk.Scale(gabor_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                 variable=self.thresh_var).pack(fill=tk.X)

        tk.Button(gabor_frame, text="ใช้ Gabor + ลบซี่โครง",
                  command=self.apply_gabor_and_remove_ribs).pack(
            fill=tk.X, pady=5
        )

    # =========================================================
    # utilities
    # =========================================================
    def clear_selectors(self):
        """ปิด RectangleSelector / PolygonSelector ถ้ายังเปิดอยู่"""
        if self.rect_selector is not None:
            self.rect_selector.set_active(False)
            self.rect_selector = None
        if self.poly_selector is not None:
            self.poly_selector.set_active(False)
            self.poly_selector = None

    def update_display(self):
        """วาดภาพ current_image ลงบน axes"""
        if self.current_image is None:
            return
        self.ax.clear()
        self.ax.axis('off')
        self.ax.imshow(self.current_image, cmap='gray', vmin=0, vmax=255)
        self.canvas.draw_idle()

    # =========================================================
    # ปุ่ม: โหลดภาพ
    # =========================================================
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"),
                ("All files", "*.*"),
            ]
        )
        if not file_path:
            return

        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            messagebox.showerror("Error", "ไม่สามารถอ่านไฟล์ภาพได้")
            return

        self.original_image = img
        self.current_image = img.copy()
        self.roi_mask = None
        self.clear_selectors()
        self.update_display()

    # =========================================================
    # ปุ่ม: รีเซ็ตภาพ
    # =========================================================
    def reset_image(self):
        if self.original_image is None:
            return
        self.current_image = self.original_image.copy()
        self.roi_mask = None
        self.clear_selectors()
        self.update_display()

    # =========================================================
    # ปุ่ม: เซฟภาพ
    # =========================================================
    def save_image(self):
        if self.current_image is None:
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg;*.jpeg"),
                ("BMP", "*.bmp"),
            ],
        )
        if not file_path:
            return
        cv2.imwrite(file_path, self.current_image)

    # =========================================================
    # ปุ่ม: ครอปจากเมาส์
    # =========================================================
    def activate_crop(self):
        if self.current_image is None:
            messagebox.showinfo("Info", "กรุณานำเข้าภาพก่อน")
            return

        self.clear_selectors()

        # NOTE: เวอร์ชันใหม่ของ Matplotlib ไม่ใช้ drawtype แล้ว
        self.rect_selector = RectangleSelector(
            self.ax,
            self.on_rect_select,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords='data',
            interactive=False,
        )
        self.canvas.draw_idle()
        messagebox.showinfo(
            "Crop",
            "ลากเมาส์บนภาพเพื่อเลือกกรอบสำหรับ crop แล้วปล่อยเมาส์",
        )

    def on_rect_select(self, eclick, erelease):
        if self.current_image is None:
            return
        # เช็กเผื่อคลิกนอก axes
        if (
            eclick.xdata is None
            or eclick.ydata is None
            or erelease.xdata is None
            or erelease.ydata is None
        ):
            return

        x1, y1 = int(round(eclick.xdata)), int(round(eclick.ydata))
        x2, y2 = int(round(erelease.xdata)), int(round(erelease.ydata))
        self.crop_image_coords(x1, y1, x2, y2)

        # ปิด selector หลังจาก crop เสร็จ
        if self.rect_selector is not None:
            self.rect_selector.set_active(False)
            self.rect_selector = None
        self.canvas.draw_idle()

    def crop_image_coords(self, x1, y1, x2, y2):
        h, w = self.current_image.shape
        xs = sorted([x1, x2])
        ys = sorted([y1, y2])
        x1, x2 = max(0, xs[0]), min(w, xs[1])
        y1, y2 = max(0, ys[0]), min(h, ys[1])

        if x2 <= x1 or y2 <= y1:
            messagebox.showwarning("Crop", "กรอบเล็กเกินไปหรือไม่ถูกต้อง")
            return

        self.current_image = self.current_image[y1:y2, x1:x2].copy()
        self.roi_mask = None
        self.update_display()

    # =========================================================
    # ปุ่ม: Polygon ROI
    # =========================================================
    def activate_polygon_roi(self):
        if self.current_image is None:
            messagebox.showinfo("Info", "กรุณานำเข้าภาพก่อน")
            return

        self.clear_selectors()

        self.poly_selector = PolygonSelector(
            self.ax,
            self.on_polygon_select,
            useblit=True,  # ยังใช้ได้ใน Matplotlib เวอร์ชันปัจจุบัน
        )
        self.canvas.draw_idle()
        messagebox.showinfo(
            "Polygon ROI",
            "คลิกทีละจุดล้อมบริเวณที่ต้องการ แล้วดับเบิลคลิกเพื่อจบ",
        )

    def on_polygon_select(self, verts):
        if self.current_image is None:
            return

        h, w = self.current_image.shape
        pts = np.array(verts, dtype=np.int32)

        # กันไม่ให้ออกนอกภาพ
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)

        self.roi_mask = mask

        if self.poly_selector is not None:
            self.poly_selector.set_active(False)
            self.poly_selector = None

        self.canvas.draw_idle()

    # =========================================================
    # ปุ่ม: ใช้ Gabor + ลบซี่โครง
    # =========================================================
    def apply_gabor_and_remove_ribs(self):
        if self.current_image is None:
            messagebox.showinfo("Info", "กรุณานำเข้าภาพก่อน")
            return

        img = self.current_image
        if img.ndim == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img

        # อ่านค่าพารามิเตอร์จาก slider
        ksize = int(self.ksize_var.get())
        if ksize % 2 == 0:
            ksize += 1  # kernel size ต้องเป็นเลขคี่

        sigma = float(self.sigma_var.get())
        lambd = float(self.lambda_var.get())
        gamma = float(self.gamma_var.get()) / 10.0  # แปลงจาก 1-20 เป็น 0.1-2.0
        theta_deg = float(self.theta_var.get())
        theta = np.deg2rad(theta_deg)

        # สร้าง Gabor kernel
        kernel = cv2.getGaborKernel(
            (ksize, ksize),
            sigma,
            theta,
            lambd,
            gamma,
            0,
            ktype=cv2.CV_32F,
        )

        # filter
        img_float = img_gray.astype(np.float32) / 255.0
        filtered = cv2.filter2D(img_float, cv2.CV_32F, kernel)

        # normalize กลับเป็น 8-bit
        filtered_norm = cv2.normalize(
            filtered, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
        )
        filtered_uint8 = filtered_norm.astype(np.uint8)

        # threshold เพื่อหา mask ของส่วนที่ตอบสนองต่อ Gabor (ซี่โครง)
        thresh_val = int(self.thresh_var.get())
        _, mask = cv2.threshold(
            filtered_uint8, thresh_val, 255, cv2.THRESH_BINARY
        )

        # จำกัดเฉพาะใน ROI ถ้ามี polygon ROI
        if self.roi_mask is not None:
            roi_255 = (self.roi_mask * 255).astype(np.uint8)
            mask = cv2.bitwise_and(mask, roi_255)

        # ใช้ inpaint ลบเส้นที่อยู่ใน mask (ซี่โครง)
        inpainted = cv2.inpaint(img_gray, mask, 3, cv2.INPAINT_TELEA)

        self.current_image = inpainted
        self.update_display()


if __name__ == "__main__":
    root = tk.Tk()
    app = XrayApp(root)
    root.mainloop()
