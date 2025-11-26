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


class XrayMonkeyProcessor:
    def __init__(self, image_path: str):
        """โหลดภาพเอกซเรย์ (ระดับเทา)"""
        self.original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.original_image is None:
            raise ValueError(f"ไม่สามารถโหลดภาพได้: {image_path}")
        self.current_image = self.original_image.copy()
        self.height, self.width = self.original_image.shape
        self.image_path = image_path
        self.crop_coords = None  # (y1, y2, x1, x2) บริเวณที่ใช้ประมวลผล (เช่น ปอด)
        print(f"✓ โหลดภาพสำเร็จ: {self.width}x{self.height} pixels")

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
        """
        ครอปภาพช่วงปอดโดยอัตโนมัติ (ใช้สัดส่วนของภาพทั้งใบ)
        - เหมาะกับภาพ X-ray ท่า lateral ที่จัดท่าใกล้เคียงกัน
        - ถ้าจำเป็นสามารถเปลี่ยน ratio ได้ในโค้ด

        พิกัดทั้งหมดคำนวณบน original_image เพื่อไม่ให้ส่วนอื่นเสียหาย
        """
        h, w = self.original_image.shape

        # แปลง ratio -> พิกัด pixel
        x1 = int(max(0, min(w - 1, w * x_start_ratio)))
        x2 = int(max(0, min(w, w * x_end_ratio)))
        y1 = int(max(0, min(h - 1, h * y_start_ratio)))
        y2 = int(max(0, min(h, h * y_end_ratio)))

        if x2 <= x1 or y2 <= y1:
            raise ValueError("ค่า ratio สำหรับ auto-crop ไม่ถูกต้อง ทำให้กรอบว่าง")

        self.crop_coords = (y1, y2, x1, x2)
        self.current_image = self.original_image[y1:y2, x1:x2].copy()
        self.height, self.width = self.current_image.shape

        print(
            f"✓ Auto-crop ปอด: x=({x1},{x2}), y=({y1},{y2}), size={self.width}x{self.height}"
        )
        return self.current_image

    # -------------------------------------------------
    # 1) เลือกบริเวณ (crop) ด้วยเมาส์
    # -------------------------------------------------
    def manual_crop_interactive(self):
        """ให้ผู้ใช้เลือก crop ด้วยเมาส์ (บนภาพเต็ม)"""
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(self.original_image, cmap="gray")
        ax.set_title("ลากเมาส์เพื่อเลือกบริเวณลำตัว/ปอด แล้วปล่อย", fontsize=14)

        self.crop_coords = None

        def on_select(eclick, erelease):
            x1 = int(min(eclick.xdata, erelease.xdata))
            y1 = int(min(eclick.ydata, erelease.ydata))
            x2 = int(max(eclick.xdata, erelease.xdata))
            y2 = int(max(eclick.ydata, erelease.ydata))

            # จำกัดให้อยู่ในภาพ
            h, w = self.original_image.shape
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h, y2))

            if x2 <= x1 or y2 <= y1:
                print("✗ กรอบ crop ไม่ถูกต้อง")
                return

            self.crop_coords = (y1, y2, x1, x2)
            print(f"✓ เลือก crop: ({x1}, {y1}) ถึง ({x2}, {y2})")
            plt.close()

        rect_selector = RectangleSelector(
            ax,
            on_select,
            useblit=True,
            button=[1],  # left mouse button
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
            print(f"✓ ตัดส่วนภาพสำเร็จ: {self.width}x{self.height}")
            return True
        else:
            print("✗ ไม่ได้เลือก crop")
            return False

    def reset_to_cropped(self):
        """รีเซ็ต current_image ให้เป็นภาพหลัง crop (หรือภาพต้นฉบับถ้ายังไม่ crop)"""
        if self.crop_coords is not None:
            y1, y2, x1, x2 = self.crop_coords
            self.current_image = self.original_image[y1:y2, x1:x2].copy()
        else:
            self.current_image = self.original_image.copy()

    # -------------------------------------------------
    # 2) การปรับแต่งภาพตามบทที่ 3 (Enhancement)
    # -------------------------------------------------
    def histogram_equalization(self):
        """Histogram Equalization ตามบทที่ 3"""
        self.current_image = cv2.equalizeHist(self.current_image)
        print("✓ ทำ Histogram Equalization สำเร็จ")
        return self.current_image

    def gaussian_smoothing(self, ksize: int = 5):
        """
        Gaussian smoothing (Spatial Filter)
        ใช้ลด noise / ทำให้ภาพนุ่มขึ้นก่อน segmentation
        """
        if ksize < 3:
            ksize = 3
        if ksize % 2 == 0:
            ksize += 1  # kernel ต้องเป็นเลขคี่
        self.current_image = cv2.GaussianBlur(self.current_image, (ksize, ksize), 0)
        print(f"✓ Gaussian smoothing (kernel={ksize}x{ksize})")
        return self.current_image

    # -------------------------------------------------
    # 3) การแบ่งส่วนกระดูกด้วย Thresholding (บทที่ 7)
    # -------------------------------------------------
    def segment_bones_otsu(self):
        """ใช้ Otsu's thresholding เพื่อแยกกระดูกออกจากพื้นหลัง"""
        _, bone_mask = cv2.threshold(
            self.current_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        print("✓ สร้าง bone mask ด้วย Otsu's method สำเร็จ")
        return bone_mask

    # -------------------------------------------------
    # 4) การลบซี่โครงด้วย Morphology (บทที่ 7)
    # -------------------------------------------------
    def remove_ribs_morphology(
        self,
        smooth_kernel: int = 5,
        rib_length: int = 40,
        rib_thickness: int = 3,
        spine_width_ratio: float = 0.25,
    ):
        """
        ลบซี่โครงโดยใช้วิธีตามบทที่ 3 และ 7 เท่านั้น:
        - Gaussian smoothing
        - Otsu thresholding
        - Morphological closing/opening

        แนวคิด:
        1) ทำให้ภาพนุ่มลงด้วย Gaussian (ลด noise)
        2) ใช้ Otsu หา bone mask
        3) Morphological closing ให้กระดูกต่อเนื่องขึ้น
        4) Morphological opening ด้วย kernel แนวนอนดึงซี่โครงออกมา
        5) จำกัด ribs mask เฉพาะด้านซ้าย/ขวา ไม่ยุ่งกับแถบตรงกลาง (กระดูกสันหลัง)
        6) แทนค่าพิกเซลบริเวณซี่โครงด้วยค่าจากภาพเบลอ (soft tissue)
        """
        # 1) smoothing
        img_for_seg = self.current_image.copy()
        if smooth_kernel < 3:
            smooth_kernel = 3
        if smooth_kernel % 2 == 0:
            smooth_kernel += 1
        img_blur = cv2.GaussianBlur(img_for_seg, (smooth_kernel, smooth_kernel), 0)
        print(f"✓ เตรียมภาพสำหรับ segmentation ด้วย Gaussian (k={smooth_kernel})")

        # 2) Otsu thresholding เพื่อหากระดูกทั้งหมด
        _, bone_mask = cv2.threshold(
            img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # 2.1) closing เล็กน้อยให้กระดูกต่อเนื่อง
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bone_mask_clean = cv2.morphologyEx(
            bone_mask, cv2.MORPH_CLOSE, kernel_small, iterations=1
        )
        print("✓ ปรับ bone mask ด้วย morphological closing")

        # 3) หา rib mask ด้วย opening + kernel แนวนอน
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

        # 4) ไม่ให้ลบกระดูกในแถบตรงกลาง (กระดูกสันหลัง)
        left = max(center_x - spine_half, 0)
        right = min(center_x + spine_half, w)
        ribs_mask[:, left:right] = 0
        print("✓ สร้าง ribs mask และจำกัดบริเวณด้านข้าง (กันกระดูกสันหลัง)")

        # 5) กระดูกอื่นที่ไม่ใช่ซี่โครง (เช่น กระดูกสันหลัง)
        bones_without_ribs_mask = cv2.subtract(bone_mask_clean, ribs_mask)

        # 6) ทำภาพผลลัพธ์: แทนค่าตรงซี่โครงด้วยค่าจากภาพเบลอ (soft tissue)
        soft_tissue_img = img_blur
        result = self.current_image.copy()
        result[ribs_mask == 255] = soft_tissue_img[ribs_mask == 255]

        self.current_image = result
        print("✓ ลบซี่โครง (แทนค่าด้วย soft tissue) สำเร็จ")

        return result, bone_mask_clean, ribs_mask, bones_without_ribs_mask

    # -------------------------------------------------
    # Utils
    # -------------------------------------------------
    def save_result(self, output_path: str, filename: str = "processed_xray.jpg"):
        full_path = os.path.join(output_path, filename)
        cv2.imwrite(full_path, self.current_image)
        print(f"✓ บันทึกภาพสำเร็จ: {full_path}")
        return full_path

    def get_current_image(self):
        return self.current_image

    def get_original_image(self):
        """คืนภาพในกรอบ crop (ถ้ามี) ไม่ใช่ทั้งภาพเต็ม"""
        if self.crop_coords is not None:
            y1, y2, x1, x2 = self.crop_coords
            return self.original_image[y1:y2, x1:x2]
        return self.original_image

    def display_results(self, step_images):
        """
        แสดงผลลัพธ์เป็น 3 แถว:
        - แถวที่ 1–2: รูปทุกขั้นตอน (ยกเว้นรูปสุดท้าย)
        - แถวที่ 3: แสดงรูปสุดท้าย (Final Result) อยู่ตรงกลางแถว
        และย่อขนาดหน้าต่างลงประมาณ 20%
        """
        n = len(step_images)
        if n == 0:
            return

        # กรณีมีรูปเดียว แสดงปกติ
        if n == 1:
            title, img = step_images[0]
            fig, ax = plt.subplots(figsize=(4, 4))  # ย่อ 20%
            ax.imshow(img, cmap="gray")
            ax.set_title(title, fontsize=12)
            ax.axis("off")
            plt.tight_layout()
            plt.show()
            return

        # จำนวนรูปย่อย (ไม่รวมรูปสุดท้าย)
        k = n - 1
        rows = 3
        cols = int(np.ceil(k / 2.0))
        if cols < 1:
            cols = 1

        # ย่อหน้าต่างลง 20% จากขนาดฐาน 5 → 4
        base_size = 4
        fig, axes = plt.subplots(rows, cols, figsize=(base_size * cols, base_size * rows))

        axes = np.array(axes)
        if axes.ndim == 1:
            axes = axes.reshape(rows, cols)

        # ----------------------------
        # แถวที่ 1–2: รูปขั้นตอน
        # ----------------------------
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

        # ----------------------------
        # แถวที่ 3: รูปสุดท้ายกลางแถว
        # ----------------------------
        final_title, final_img = step_images[-1]
        mid_c = cols // 2  # ตำแหน่งคอลัมน์ตรงกลาง

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


class XrayProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("โปรแกรมประมวลผลภาพเอกซเรย์ของลิง (ใช้วิธีจากสไลด์เท่านั้น)")
        self.root.geometry("650x850")

        self.processor = None
        self.step_images = []

        self._build_gui()

    def _build_gui(self):
        btn_frame = Frame(self.root)
        btn_frame.pack(pady=10)

        Button(
            btn_frame,
            text="เลือกภาพเอกซเรย์",
            command=self.load_image,
            width=20,
            height=2,
            bg="lightblue",
        ).pack()

        # ตัวแปรพารามิเตอร์ (ให้ผู้ใช้ปรับได้)
        self.params = {
            "use_hist_eq": BooleanVar(value=True),
            "smooth_kernel": tk.IntVar(value=5),
            "rib_length": tk.IntVar(value=40),
            "rib_thickness": tk.IntVar(value=3),
            "spine_width_percent": tk.IntVar(value=25),  # 25% ของความกว้างภาพ
        }

        param_frame = Frame(self.root)
        param_frame.pack(pady=10, padx=20, fill="both", expand=True)

        Label(
            param_frame,
            text="=== ปรับพารามิเตอร์ตามภาพ ===",
            font=("Arial", 12, "bold"),
        ).pack()

        # Histogram equalization on/off
        hist_frame = Frame(param_frame)
        hist_frame.pack(fill="x", pady=5)
        Label(hist_frame, text="ใช้ Histogram Equalization:").pack(side="left", padx=5)
        tk.Checkbutton(
            hist_frame, variable=self.params["use_hist_eq"], onvalue=True, offvalue=False
        ).pack(side="left")

        # Smoothing kernel
        Label(param_frame, text="ขนาด Gaussian kernel (เบลอภาพก่อนแบ่งส่วน):").pack(
            anchor="w"
        )
        Scale(
            param_frame,
            from_=3,
            to=21,
            orient=HORIZONTAL,
            variable=self.params["smooth_kernel"],
        ).pack(fill="x", padx=10)

        # Rib length
        Label(param_frame, text="ความยาว structuring element สำหรับซี่โครง:").pack(
            anchor="w"
        )
        Scale(
            param_frame,
            from_=10,
            to=120,
            orient=HORIZONTAL,
            variable=self.params["rib_length"],
        ).pack(fill="x", padx=10)

        # Rib thickness
        Label(param_frame, text="ความหนา structuring element สำหรับซี่โครง:").pack(
            anchor="w"
        )
        Scale(
            param_frame,
            from_=1,
            to=9,
            orient=HORIZONTAL,
            variable=self.params["rib_thickness"],
        ).pack(fill="x", padx=10)

        # Spine width
        Label(param_frame, text="ความกว้างบริเวณกระดูกสันหลัง (%)").pack(anchor="w")
        Scale(
            param_frame,
            from_=10,
            to=40,
            orient=HORIZONTAL,
            variable=self.params["spine_width_percent"],
        ).pack(fill="x", padx=10)

        # ปุ่มประมวลผล / แสดงผล
        process_frame = Frame(self.root)
        process_frame.pack(pady=15)

        Button(
            process_frame,
            text="ประมวลผลภาพ",
            command=self.process_image,
            width=20,
            height=2,
            bg="lightgreen",
        ).pack(side="left", padx=5)

        Button(
            process_frame,
            text="แสดงผลลัพธ์",
            command=self.show_results,
            width=20,
            height=2,
            bg="lightyellow",
        ).pack(side="left", padx=5)

        # ปุ่มบันทึก
        save_frame = Frame(self.root)
        save_frame.pack(pady=10)

        Button(
            save_frame,
            text="บันทึกผลลัพธ์",
            command=self.save_results,
            width=20,
            height=2,
            bg="lightcoral",
        ).pack()

        # status
        self.status_label = Label(
            self.root, text="กรุณาเลือกภาพเอกซเรย์", font=("Arial", 10), fg="blue"
        )
        self.status_label.pack(pady=10)

    # ---------- ฟังก์ชัน preview ROI ----------
    def preview_roi(self):
        """แสดงตัวอย่างบริเวณที่ถูกครอป (ROI) ให้ผู้ใช้ดู"""
        if self.processor is None:
            return

        roi = self.processor.get_current_image()
        if roi is None or roi.size == 0:
            messagebox.showwarning("คำเตือน", "ยังไม่มีบริเวณให้แสดงตัวอย่าง")
            return

        plt.figure(figsize=(6, 6))
        plt.imshow(roi, cmap="gray")
        plt.title("ตัวอย่างบริเวณที่จะใช้ประมวลผล (ROI)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    # ---------------- GUI callbacks ----------------
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="เลือกภาพเอกซเรย์",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")],
        )
        if not file_path:
            return

        try:
            self.processor = XrayMonkeyProcessor(file_path)
            self.status_label.config(
                text=f"✓ โหลดภาพ: {os.path.basename(file_path)}", fg="green"
            )

            # วนลูปให้เลือกวิธี crop + preview ได้จนกว่าจะพอใจ
            while True:
                use_auto = messagebox.askyesno(
                    "เลือกวิธี Crop",
                    "ต้องการให้โปรแกรมเลือกช่วงปอดอัตโนมัติหรือไม่?\n"
                    "(Yes = auto-crop ช่วงปอด, No = เลือกบริเวณเองด้วยเมาส์)",
                )

                # reset กลับภาพเต็มก่อนทุกครั้ง
                self.processor.current_image = self.processor.original_image.copy()
                self.processor.crop_coords = None

                if use_auto:
                    # auto-crop ช่วงปอด
                    try:
                        self.processor.auto_crop_lung_region()
                    except Exception as e:
                        messagebox.showerror(
                            "ข้อผิดพลาด", f"Auto-crop ล้มเหลว: {e}\nโปรดลองใหม่อีกครั้ง"
                        )
                        continue
                else:
                    # manual crop ด้วยเมาส์
                    ok = self.processor.manual_crop_interactive()
                    if not ok:
                        retry = messagebox.askyesno(
                            "ไม่ได้เลือกกรอบ",
                            "ยังไม่ได้เลือกบริเวณใดเลย\nต้องการลองครอปใหม่อีกครั้งหรือไม่?",
                        )
                        if retry:
                            continue
                        else:
                            # ใช้ทั้งภาพ
                            self.processor.current_image = (
                                self.processor.original_image.copy()
                            )
                            self.processor.crop_coords = None

                # แสดง preview ROI
                self.preview_roi()
                confirm = messagebox.askyesno(
                    "ยืนยันกรอบ",
                    "ต้องการใช้กรอบนี้เป็นบริเวณสำหรับประมวลผลหรือไม่?",
                )
                if confirm:
                    break
                # ถ้าไม่ยืนยัน ให้วนเลือกวิธีใหม่อีกรอบ

            self.step_images = [
                ("Original / ROI", self.processor.get_current_image().copy())
            ]
            self.status_label.config(text="✓ เลือกบริเวณ ROI เรียบร้อย", fg="green")

        except Exception as e:
            messagebox.showerror("ข้อผิดพลาด", f"ไม่สามารถโหลดภาพ: {e}")
            self.status_label.config(text="✗ โหลดภาพล้มเหลว", fg="red")
            self.processor = None
            self.step_images = []

    def process_image(self):
        if self.processor is None:
            messagebox.showwarning("ข้อมูลไม่สมบูรณ์", "กรุณาเลือกภาพก่อน")
            return

        try:
            # เริ่มจากภาพหลัง crop ทุกครั้ง
            self.processor.reset_to_cropped()
            self.step_images = [
                ("Original / ROI", self.processor.get_current_image().copy())
            ]

            # 1) Histogram Equalization (optional)
            if self.params["use_hist_eq"].get():
                self.processor.histogram_equalization()
                self.step_images.append(
                    ("Histogram Equalization", self.processor.get_current_image().copy())
                )

            # 2) Gaussian smoothing ก่อน segmentation
            smooth_k = self.params["smooth_kernel"].get()
            self.processor.gaussian_smoothing(smooth_k)
            self.step_images.append(
                (f"Gaussian Blur (k={smooth_k})", self.processor.get_current_image().copy())
            )

            # 3) ลบซี่โครงด้วย morphological method
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

            self.status_label.config(text="✓ ประมวลผลภาพเสร็จสิ้น", fg="green")
            messagebox.showinfo("สำเร็จ", "ประมวลผลภาพเสร็จสิ้น!")

        except Exception as e:
            messagebox.showerror("ข้อผิดพลาด", f"เกิดข้อผิดพลาดระหว่างประมวลผล: {e}")
            self.status_label.config(text="✗ ประมวลผลล้มเหลว", fg="red")

    def show_results(self):
        if not self.step_images or len(self.step_images) < 2:
            messagebox.showwarning("ข้อมูลไม่สมบูรณ์", "กรุณาประมวลผลภาพก่อน")
            return
        self.processor.display_results(self.step_images)

    def save_results(self):
        if self.processor is None or not self.step_images:
            messagebox.showwarning("ข้อมูลไม่สมบูรณ์", "ยังไม่มีผลลัพธ์ให้บันทึก")
            return

        output_dir = filedialog.askdirectory(title="เลือกโฟลเดอร์สำหรับบันทึกผลลัพธ์")
        if not output_dir:
            return

        try:
            # บันทึกแต่ละขั้นตอน (เฉพาะ ROI)
            for idx, (title, img) in enumerate(self.step_images):
                safe_title = title.replace(" ", "_").replace("/", "_")
                filename = f"step{idx:02d}_{safe_title}.jpg"
                full_path = os.path.join(output_dir, filename)
                cv2.imwrite(full_path, img)
                print(f"✓ บันทึก: {full_path}")

            # บันทึกผลลัพธ์สุดท้ายเป็นชื่อมาตรฐาน
            self.processor.current_image = self.step_images[-1][1].copy()
            self.processor.save_result(output_dir, "final_result.jpg")

            self.status_label.config(
                text=f"✓ บันทึกผลลัพธ์ทั้งหมดที่: {output_dir}", fg="green"
            )
            messagebox.showinfo("สำเร็จ", "บันทึกผลลัพธ์สำเร็จ!")

        except Exception as e:
            messagebox.showerror("ข้อผิดพลาด", f"ไม่สามารถบันทึกผลลัพธ์: {e}")
            self.status_label.config(text="✗ บันทึกผลลัพธ์ล้มเหลว", fg="red")


def main():
    root = tk.Tk()
    app = XrayProcessorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()