import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import tkinter as tk
from tkinter import filedialog, messagebox, Scale, Button, Label, Frame, HORIZONTAL
from tkinter import BooleanVar


class XrayMonkeyProcessor:
    def __init__(self, image_path: str):
        """โหลดภาพเอกซเรย์ (ระดับเทา)"""
        self.original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.original_image is None:
            raise ValueError(f"ไม่สามารถโหลดภาพได้: {image_path}")
        self.current_image = self.original_image.copy()
        self.height, self.width = self.original_image.shape
        self.image_path = image_path
        self.crop_coords = None  # (y1, y2, x1, x2)
        print(f"✓ โหลดภาพสำเร็จ: {self.width}x{self.height} pixels")

    # -----------------------------
    # 1) เลือกบริเวณลำตัว (crop)
    # -----------------------------
    def manual_crop_interactive(self):
        """ให้ผู้ใช้เลือก crop บริเวณลำตัวของลิงด้วยเมาส์"""
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(self.original_image, cmap="gray")
        ax.set_title("ลากเมาส์เพื่อเลือกบริเวณลำตัวของลิง แล้วปล่อย", fontsize=14)

        self.crop_coords = None

        def on_select(eclick, erelease):
            x1 = int(min(eclick.xdata, erelease.xdata))
            y1 = int(min(eclick.ydata, erelease.ydata))
            x2 = int(max(eclick.xdata, erelease.xdata))
            y2 = int(max(eclick.ydata, erelease.ydata))

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(self.width, x2)
            y2 = min(self.height, y2)

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

    # ---------------------------------
    # 2) การปรับแต่งภาพตามบทที่ 3 (Enhancement)
    # ---------------------------------
    def histogram_equalization(self):
        """
        ปรับคอนทราสต์ด้วย Histogram Equalization
        ตามแนวคิดในบทที่ 3 (Image Enhancement in Spatial Domain)
        """
        self.current_image = cv2.equalizeHist(self.current_image)
        print("✓ ทำ Histogram Equalization สำเร็จ")
        return self.current_image

    def gaussian_smoothing(self, ksize: int = 5):
        """
        Smoothing Spatial Filter (Gaussian)
        ใช้ลด noise / ทำให้ภาพนุ่มขึ้นก่อนทำ segmentation (บทที่ 3)
        """
        if ksize < 3:
            ksize = 3
        if ksize % 2 == 0:
            ksize += 1  # kernel ต้องเป็นเลขคี่
        self.current_image = cv2.GaussianBlur(self.current_image, (ksize, ksize), 0)
        print(f"✓ Gaussian smoothing (kernel={ksize}x{ksize})")
        return self.current_image

    # ---------------------------------
    # 3) การแบ่งส่วนกระดูกด้วย Thresholding (บทที่ 7)
    # ---------------------------------
    def segment_bones_otsu(self):
        """
        ใช้ Otsu's thresholding เพื่อแยกกระดูกออกจากพื้นหลัง
        บทที่ 7 (Thresholding with Otsu’s Method)
        """
        # Otsu จะเลือกค่า threshold ที่เหมาะที่สุดอัตโนมัติ
        _, bone_mask = cv2.threshold(
            self.current_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        print("✓ สร้าง bone mask ด้วย Otsu's method สำเร็จ")
        return bone_mask

    # ---------------------------------
    # 4) การลบซี่โครงด้วยวิธี Morphology (บทที่ 7)
    # ---------------------------------
    def remove_ribs_morphology(
        self,
        smooth_kernel: int = 5,
        rib_length: int = 40,
        rib_thickness: int = 3,
        spine_width_ratio: float = 0.25,
    ):
        """
        ลบซี่โครงโดยใช้วิธีตามบทที่ 3 และ 7 เท่านั้น:
        - Gaussian smoothing (บทที่ 3)
        - Otsu thresholding (บทที่ 7)
        - Morphological opening/closing (บทที่ 7)

        แนวคิด:
        1) ทำให้ภาพนุ่มลงด้วย Gaussian (ลด noise เล็ก ๆ)
        2) ใช้ Otsu หา bone mask
        3) ใช้ morphological opening ด้วย kernel รูปสี่เหลี่ยมแนวนอน
           เพื่อดึงเฉพาะโครงสร้างที่มีลักษณะเป็นแท่งยาวแนวนอน (ซี่โครง)
        4) จำกัดบริเวณซี่โครงให้อยู่แค่ด้านซ้าย/ขวา ไม่ยุ่งกับแถบตรงกลาง (กระดูกสันหลัง)
        5) ทำภาพใหม่โดยแทนค่าพิกเซลตรงบริเวณซี่โครงด้วยค่าในภาพที่ผ่านการ blur
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

        # ทำ closing เล็กน้อยให้กระดูกต่อเนื่องขึ้น
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
        print("✓ สร้าง ribs mask ด้วย morphological opening และจำกัดบริเวณด้านข้าง")

        # 5) กระดูกอื่นที่ไม่ใช่ซี่โครง (เช่น กระดูกสันหลัง)
        bones_without_ribs_mask = cv2.subtract(bone_mask_clean, ribs_mask)

        # 6) ทำภาพผลลัพธ์: แทนค่าตรงซี่โครงด้วยค่าจากภาพเบลอ (soft tissue)
        soft_tissue_img = img_blur  # ใช้ภาพเบลอแทนพื้นผิวนุ่ม ๆ
        result = self.current_image.copy()
        result[ribs_mask == 255] = soft_tissue_img[ribs_mask == 255]

        self.current_image = result
        print("✓ ลบซี่โครง (แทนค่าด้วย soft tissue) สำเร็จ")

        # return ภาพและ mask ต่าง ๆ เผื่อเอาไปโชว์เป็น step
        return result, bone_mask_clean, ribs_mask, bones_without_ribs_mask

    # ---------------------------------
    # Utils
    # ---------------------------------
    def save_result(self, output_path: str, filename: str = "processed_xray.jpg"):
        full_path = os.path.join(output_path, filename)
        cv2.imwrite(full_path, self.current_image)
        print(f"✓ บันทึกภาพสำเร็จ: {full_path}")
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
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        if n == 1:
            axes = [axes]

        for ax, (title, img) in zip(axes, step_images):
            ax.imshow(img, cmap="gray")
            ax.set_title(title, fontsize=10)
            ax.axis("off")

        plt.tight_layout()
        plt.show()


class XrayProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("โปรแกรมประมวลผลภาพเอกซเรย์ของลิง (ใช้วิธีจากสไลด์เท่านั้น)")
        self.root.geometry("650x850")

        self.processor: XrayMonkeyProcessor | None = None
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

            # ให้ผู้ใช้เลือก crop เฉพาะลำตัว
            if messagebox.askyesno("Crop ภาพ", "ต้องการเลือกบริเวณลำตัวของลิงหรือไม่?"):
                if self.processor.manual_crop_interactive():
                    self.status_label.config(
                        text="✓ เลือกบริเวณลำตัวสำเร็จ", fg="green"
                    )
                else:
                    self.status_label.config(text="✗ ไม่ได้เลือกบริเวณลำตัว", fg="red")

            # รีเซ็ต step images
            self.step_images = [("Original / Cropped", self.processor.get_current_image().copy())]

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
            self.step_images = [("Original / Cropped", self.processor.get_current_image().copy())]

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

            result, bone_mask, ribs_mask, bones_no_ribs = self.processor.remove_ribs_morphology(
                smooth_kernel=smooth_k,
                rib_length=rib_len,
                rib_thickness=rib_th,
                spine_width_ratio=spine_ratio,
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
            # บันทึกแต่ละขั้นตอน
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
