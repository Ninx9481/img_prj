import cv2
import numpy as np
import os

# ============================================================
#   ฟังก์ชันช่วยถาม/พิมพ์/เซฟภาพ
# ============================================================

def ask_bool(question, default=True):
    d = "Y/n" if default else "y/N"
    ans = input(f"{question} ({d}): ").strip().lower()
    if ans == "":
        return default
    return ans in ["y", "yes", "true", "1"]

def ask_int(question, default, min_val=None, max_val=None, must_odd=False):
    while True:
        ans = input(f"{question} [default={default}]: ").strip()
        if ans == "":
            value = default
        else:
            try:
                value = int(ans)
            except:
                print("กรุณากรอกจำนวนเต็ม")
                continue
        if must_odd and value % 2 == 0:
            print("ต้องเป็นเลขคี่")
            continue
        if min_val is not None and value < min_val:
            print(f"ต้องไม่น้อยกว่า {min_val}")
            continue
        if max_val is not None and value > max_val:
            print(f"ต้องไม่มากกว่า {max_val}")
            continue
        return value

def ask_float(question, default, min_val=None, max_val=None):
    while True:
        ans = input(f"{question} [default={default}]: ").strip()
        if ans == "":
            value = default
        else:
            try:
                value = float(ans)
            except:
                print("กรุณากรอกตัวเลข")
                continue
        if min_val is not None and value < min_val:
            print(f"ต้องไม่น้อยกว่า {min_val}")
            continue
        if max_val is not None and value > max_val:
            print(f"ต้องไม่มากกว่า {max_val}")
            continue
        return value

def save_img(img, filename, outdir="outputs", save=True):
    """เซฟรูปถ้า save=True"""
    if not save:
        return

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, filename)

    img_to_save = img
    if img_to_save.dtype != np.uint8:
        img_to_save = cv2.normalize(img_to_save, None, 0, 255,
                                    cv2.NORM_MINMAX).astype(np.uint8)

    cv2.imwrite(path, img_to_save)
    print("Saved:", path)

# ============================================================
#   ประมวลผลเฉพาะ ROI (ลำตัว) ลบซี่โครงใน polygon
# ============================================================

def process_torso(torso, params, outdir, tag, save_steps=True):
    """
    torso: ภาพลำตัว (crop แล้ว, grayscale 2D)
    params: dict ค่าพารามิเตอร์
    return: ภาพลำตัวหลังลบซี่โครง
    """

    # -------- 1) Enhancement --------
    clahe = cv2.createCLAHE(
        clipLimit=params["clahe_clip"],
        tileGridSize=(params["clahe_tile"], params["clahe_tile"])
    )
    torso_enh = clahe.apply(torso)
    torso_den = cv2.medianBlur(torso_enh, params["median_ksize"])

    save_img(torso_enh, f"{tag}_1_enh.png", outdir, save_steps)
    save_img(torso_den, f"{tag}_2_denoise.png", outdir, save_steps)

    # -------- 2) Bone mask ด้วย Otsu --------
    blur = cv2.GaussianBlur(torso_den,
                            (params["gauss_ksize"], params["gauss_ksize"]), 0)
    _, bone = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # ให้ขาวเป็นกระดูก ถ้าไม่ใช่ให้ invert
    if (bone == 255).any() and (bone == 0).any():
        if torso_den[bone == 255].mean() < torso_den[bone == 0].mean():
            bone = cv2.bitwise_not(bone)

    kernel_small = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (params["small_kernel"], params["small_kernel"])
    )
    bone = cv2.morphologyEx(bone, cv2.MORPH_CLOSE, kernel_small)

    # ลบวัตถุเล็ก ๆ
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bone)
    clean = np.zeros_like(bone)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= params["min_area"]:
            clean[labels == i] = 255
    bone = clean

    save_img(bone, f"{tag}_3_boneMask.png", outdir, save_steps)

    # -------- 3) แยกกระดูกสันหลัง / ซี่โครง --------
    kernel_vert = cv2.getStructuringElement(
        cv2.MORPH_RECT, (params["vert_width"], params["vert_length"])
    )
    spine = cv2.morphologyEx(bone, cv2.MORPH_OPEN, kernel_vert)
    spine = cv2.dilate(
        spine,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (params["spine_dilate"], params["spine_dilate"])),
        iterations=1
    )
    ribs = cv2.bitwise_and(bone, cv2.bitwise_not(spine))

    ribs_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (params["ribs_open"], params["ribs_open"])
    )
    ribs = cv2.morphologyEx(ribs, cv2.MORPH_OPEN, ribs_kernel)

    save_img(spine, f"{tag}_4_spine.png", outdir, save_steps)
    save_img(ribs, f"{tag}_5_ribs.png", outdir, save_steps)

    # -------- 4) สร้าง polygon (convex hull) รอบซี่โครง --------
    contours, _ = cv2.findContours(ribs, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    poly_mask = np.zeros_like(torso)

    if len(contours) > 0:
        pts = np.vstack(contours)
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(poly_mask, hull, 255)
    else:
        poly_mask = ribs.copy()

    save_img(poly_mask, f"{tag}_6_polygon.png", outdir, save_steps)

    # -------- 5) Inpaint ซี่โครงเฉพาะใน polygon --------
    ribs_in_poly = cv2.bitwise_and(ribs, poly_mask)

    torso_bgr = cv2.cvtColor(torso_den, cv2.COLOR_GRAY2BGR)
    inpaint_bgr = cv2.inpaint(
        torso_bgr, ribs_in_poly, params["inpaint_radius"], cv2.INPAINT_TELEA
    )
    inpaint_gray = cv2.cvtColor(inpaint_bgr, cv2.COLOR_BGR2GRAY)

    save_img(ribs_in_poly, f"{tag}_7_ribs_in_poly.png", outdir, save_steps)
    save_img(inpaint_gray, f"{tag}_7_inpaint.png", outdir, save_steps)

    # -------- 6) รวม inside/outside polygon ในกรอบ crop --------
    final_torso = torso.copy()
    final_torso[poly_mask == 255] = inpaint_gray[poly_mask == 255]

    save_img(final_torso, f"{tag}_8_final_torso.png", outdir, save_steps)

    return final_torso

# ============================================================
#                      MAIN (auto crop + save option)
# ============================================================

def main():
    print("=== Auto-crop Monkey X-ray & Remove Ribs (no GUI) ===")

    # 1) เลือกไฟล์ภาพ
    print("แนะนำ: ให้วางไฟล์ภาพไว้โฟลเดอร์เดียวกับสคริปต์นี้")
    img_name = input("พิมพ์ชื่อไฟล์ภาพ (เช่น monkey_xray.png): ").strip()
    if img_name == "":
        print("ไม่ได้ระบุไฟล์ จบโปรแกรม")
        return

    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("อ่านไฟล์ไม่สำเร็จ ตรวจสอบชื่อไฟล์/ตำแหน่ง")
        return

    H, W = img.shape
    print(f"อ่านภาพสำเร็จ ขนาด = {W} x {H}")

    # 2) ถามว่าจะเซฟไฟล์ขั้นตอนต่าง ๆ หรือไม่
    save_steps = ask_bool("ต้องการบันทึกรูปผลลัพธ์ทั้งหมดไหม?", True)
    outdir = "outputs"

    # 3) ใช้ภาพเต็มเพื่อหา ribs mask เบื้องต้น (สำหรับ auto-crop)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enh_full = clahe.apply(img)
    den_full = cv2.medianBlur(enh_full, 3)

    blur_full = cv2.GaussianBlur(den_full, (5, 5), 0)
    _, bone_full = cv2.threshold(
        blur_full, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    if (bone_full == 255).any() and (bone_full == 0).any():
        if den_full[bone_full == 255].mean() < den_full[bone_full == 0].mean():
            bone_full = cv2.bitwise_not(bone_full)

    bone_full = cv2.morphologyEx(
        bone_full,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )

    kernel_vert_full = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 35))
    spine_full = cv2.morphologyEx(bone_full, cv2.MORPH_OPEN, kernel_vert_full)
    spine_full = cv2.dilate(
        spine_full,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1
    )
    ribs_full = cv2.bitwise_and(bone_full, cv2.bitwise_not(spine_full))
    ribs_full = cv2.morphologyEx(
        ribs_full,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )

    ys, xs = np.where(ribs_full > 0)
    if len(xs) == 0 or len(ys) == 0:
        print("ไม่พบ ribs ในภาพ (mask ว่าง) ใช้ครึ่งล่างของภาพแทน")
        x_min, x_max = int(W * 0.2), int(W * 0.9)
        y_min, y_max = int(H * 0.3), int(H * 0.8)
    else:
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        pad_x = int(0.05 * W)  # padding 5% รอบด้าน
        pad_y = int(0.05 * H)

        x_min = max(x_min - pad_x, 0)
        x_max = min(x_max + pad_x, W - 1)
        y_min = max(y_min - pad_y, 0)
        y_max = min(y_max + pad_y, H - 1)

    torso = img[y_min:y_max+1, x_min:x_max+1]
    save_img(torso, "auto_crop_torso.png", outdir, save_steps)
    print(f"Auto-crop ROI: x=[{x_min},{x_max}], y=[{y_min},{y_max}]")

    # 4) ลูปให้ปรับพารามิเตอร์ได้หลายรอบ
    run = 1
    while True:
        print(f"\n===== รอบที่ {run}: กำหนดพารามิเตอร์ =====")
        print("(กด Enter = ใช้ค่ามาตรฐานที่แนะนำ)")

        params = {
            "clahe_clip":   ask_float("CLAHE clipLimit", 2.0),
            "clahe_tile":   ask_int("CLAHE tileGridSize", 8, 2),

            "median_ksize": ask_int("Median filter kernel (odd)", 3, 1, must_odd=True),
            "gauss_ksize":  ask_int("Gaussian blur kernel (odd)", 5, 1, must_odd=True),

            "small_kernel": ask_int("Small morphology kernel (odd)", 3, 1, must_odd=True),
            "min_area":     ask_int("min_area สำหรับลบจุดเล็ก", 80, 1),

            "vert_width":   ask_int("ความกว้าง kernel spine", 5, 1),
            "vert_length":  ask_int("ความยาว kernel spine", 35, 3),
            "spine_dilate": ask_int("ขนาด dilate spine (odd)", 7, 1, must_odd=True),

            "ribs_open":    ask_int("kernel เปิด ribs (odd)", 3, 1, must_odd=True),

            "inpaint_radius": ask_int("inpaint radius", 3, 1),
        }

        tag = f"run{run}"

        torso_result = process_torso(torso, params, outdir, tag, save_steps)

        # เอากลับไปใส่ในภาพเต็ม
        full_result = img.copy()
        full_result[y_min:y_max+1, x_min:x_max+1] = torso_result

        save_img(full_result, f"{tag}_full_result.jpg", outdir, save_steps)
        # เซฟชื่อหลัก (ถ้า save_steps=False จะไม่สร้างไฟล์)
        save_img(full_result, "monkey_xray_no_ribs_final.jpg", outdir, save_steps)

        again = ask_bool("ต้องการลองปรับพารามิเตอร์ใหม่อีกรอบไหม?", False)
        if not again:
            print("จบโปรแกรม ถ้าเลือกเซฟ ให้เปิดโฟลเดอร์ outputs ดูรูปแต่ละขั้นตอน")
            break

        run += 1

if __name__ == "__main__":
    main()
