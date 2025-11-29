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
        self.root.title("Monkey X-ray – Gabor soft rib suppression")

        # ---------- state ----------
        self.original_image = None
        self.current_image = None
        self.roi_mask = None

        self.rect_selector = None
        self.poly_selector = None

        # ---------- layout ----------
        top_frame = tk.Frame(root)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        main_frame = tk.Frame(root)
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        canvas_frame = tk.Frame(main_frame)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        control_frame = tk.Frame(main_frame, padx=5, pady=5,
                                 relief=tk.GROOVE, bd=2)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # ---------- matplotlib canvas ----------
        self.fig = Figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ---------- top buttons ----------
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

        # ---------- Gabor controls ----------
        gabor_frame = tk.LabelFrame(control_frame, text="Gabor soft rib suppression",
                                    padx=5, pady=5)
        gabor_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        self.theta_var = tk.IntVar(value=80)      # มุมกลางของซี่โครง
        self.lambda_var = tk.IntVar(value=15)     # wavelength
        self.sigma_var = tk.DoubleVar(value=4.0)  # sigma
        self.gamma_var = tk.IntVar(value=5)       # 0.5 จริง ๆ
        self.ksize_var = tk.IntVar(value=31)      # kernel size
        self.alpha_var = tk.IntVar(value=80)      # ความแรงในการลบ (%)
        self.spine_width_var = tk.IntVar(value=60)  # ระยะป้องกันสันหลัง

        tk.Label(gabor_frame, text="Theta center (deg)").pack(anchor="w")
        tk.Scale(gabor_frame, from_=0, to=180, orient=tk.HORIZONTAL,
                 variable=self.theta_var).pack(fill=tk.X)

        tk.Label(gabor_frame, text="Lambda (px)").pack(anchor="w")
        tk.Scale(gabor_frame, from_=5, to=50, orient=tk.HORIZONTAL,
                 variable=self.lambda_var).pack(fill=tk.X)

        tk.Label(gabor_frame, text="Sigma").pack(anchor="w")
        tk.Scale(gabor_frame, from_=1, to=20, resolution=0.5,
                 orient=tk.HORIZONTAL, variable=self.sigma_var).pack(fill=tk.X)

        tk.Label(gabor_frame, text="Gamma (x0.1)").pack(anchor="w")
        tk.Scale(gabor_frame, from_=1, to=20, orient=tk.HORIZONTAL,
                 variable=self.gamma_var).pack(fill=tk.X)

        tk.Label(gabor_frame, text="Kernel size").pack(anchor="w")
        tk.Scale(gabor_frame, from_=7, to=101, orient=tk.HORIZONTAL,
                 variable=self.ksize_var).pack(fill=tk.X)

        tk.Label(gabor_frame, text="Suppression strength (%)").pack(anchor="w")
        tk.Scale(gabor_frame, from_=0, to=200, orient=tk.HORIZONTAL,
                 variable=self.alpha_var).pack(fill=tk.X)

        tk.Label(gabor_frame, text="Spine protection width (px)").pack(anchor="w")
        tk.Scale(gabor_frame, from_=0, to=200, orient=tk.HORIZONTAL,
                 variable=self.spine_width_var).pack(fill=tk.X)

        tk.Button(gabor_frame, text="กดเพื่อลดซี่โครง (Gabor soft)",
                  command=self.apply_gabor_soft).pack(fill=tk.X, pady=5)

    # =========================================================
    # utilities
    # =========================================================
    def clear_selectors(self):
        if self.rect_selector is not None:
            self.rect_selector.set_active(False)
            self.rect_selector = None
        if self.poly_selector is not None:
            self.poly_selector.set_active(False)
            self.poly_selector = None

    def update_display(self):
        if self.current_image is None:
            return
        self.ax.clear()
        self.ax.axis('off')
        self.ax.imshow(self.current_image, cmap='gray', vmin=0, vmax=255)
        self.canvas.draw_idle()

    # =========================================================
    # load / reset / save
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

    def reset_image(self):
        if self.original_image is None:
            return
        self.current_image = self.original_image.copy()
        self.roi_mask = None
        self.clear_selectors()
        self.update_display()

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
    # crop
    # =========================================================
    def activate_crop(self):
        if self.current_image is None:
            messagebox.showinfo("Info", "กรุณานำเข้าภาพก่อน")
            return
        self.clear_selectors()
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
    # polygon ROI
    # =========================================================
    def activate_polygon_roi(self):
        if self.current_image is None:
            messagebox.showinfo("Info", "กรุณานำเข้าภาพก่อน")
            return
        self.clear_selectors()
        self.poly_selector = PolygonSelector(
            self.ax,
            self.on_polygon_select,
            useblit=True,
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
    # Gabor soft suppression
    # =========================================================
    def apply_gabor_soft(self):
        if self.current_image is None:
            messagebox.showinfo("Info", "กรุณานำเข้าภาพก่อน")
            return

        img = self.current_image
        if img.ndim == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img
        img_norm = img_gray.astype(np.float32) / 255.0

        # พารามิเตอร์จาก slider
        ksize = int(self.ksize_var.get())
        if ksize % 2 == 0:
            ksize += 1
        sigma = float(self.sigma_var.get())
        lambd = float(self.lambda_var.get())
        gamma = float(self.gamma_var.get()) / 10.0
        theta_center = float(self.theta_var.get())
        alpha = float(self.alpha_var.get()) / 100.0
        spine_width = float(self.spine_width_var.get())

        # ทำ Gabor หลายมุมรอบ ๆ theta_center
        orientations = [theta_center - 20.0, theta_center, theta_center + 20.0]
        responses = []
        for tdeg in orientations:
            theta = np.deg2rad(tdeg)
            kernel = cv2.getGaborKernel(
                (ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F
            )
            resp = cv2.filter2D(img_norm, cv2.CV_32F, kernel)
            responses.append(np.abs(resp))

        # รวมเป็น rib map
        R = responses[0]
        for r in responses[1:]:
            R = np.maximum(R, r)

        # จำกัดใน ROI ถ้ามี
        if self.roi_mask is not None:
            R = R * self.roi_mask.astype(np.float32)

        # ป้องกันกระดูกสันหลัง: ทำ weight ให้ผลน้อยใกล้สันหลัง
        h, w = img_gray.shape
        col_profile = img_gray.mean(axis=0)
        spine_x = int(np.argmax(col_profile))
        if spine_width > 0:
            x = np.arange(w, dtype=np.float32)
            dx = (x - spine_x) / (spine_width + 1e-6)
            weight = 1.0 - np.exp(-0.5 * dx * dx)  # 0 ใกล้สันหลัง, 1 ไกล
            R *= weight[None, :]

        # ทำให้ rib map นุ่ม ๆ และ normalize
        R_blur = cv2.GaussianBlur(R, (5, 5), 0)
        R_norm = R_blur / (R_blur.max() + 1e-6)

        # ลบ rib map ออกจากภาพ
        I_new = img_norm - alpha * R_norm
        I_new = np.clip(I_new, 0.0, 1.0)

        self.current_image = (I_new * 255).astype(np.uint8)
        self.update_display()


if __name__ == "__main__":
    root = tk.Tk()
    app = XrayApp(root)
    root.mainloop()
