import cv2
import numpy as np
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Simple rib suppression by vertical blur in selected ROI"
    )
    parser.add_argument("--input", "-i", required=True, help="path รูป X-ray (jpg/png)")
    parser.add_argument("--output", "-o", default="output_blur_roi.jpg",
                        help="ไฟล์ผลลัพธ์")

    # ยิ่งใหญ่ ซี่โครงยิ่งจาง แต่ปอดจะเนียนขึ้น
    parser.add_argument("--blur_len", type=int, default=41,
                        help="ความยาว kernel แนวตั้ง (เลขคี่)")
    # สัดส่วนภาพเบลอใน ROI
    parser.add_argument("--alpha", type=float, default=0.9,
                        help="0=ไม่เปลี่ยน, 1=ใช้ภาพเบลอล้วนใน ROI")

    return parser.parse_args()


def select_roi(img):
    clone = img.copy()
    cv2.namedWindow("Select lung ROI", cv2.WINDOW_NORMAL)
    print(">> ลากกรอบสี่เหลี่ยมคร่าว ๆ รอบ lung field แล้วกด ENTER/SPACE")
    r = cv2.selectROI("Select lung ROI", clone, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select lung ROI")
    x, y, w, h = r
    if w == 0 or h == 0:
        raise RuntimeError("ไม่ได้เลือก ROI")
    return x, y, w, h


def blur_ribs_in_roi(roi_bgr, blur_len=41, alpha=0.9):
    # grayscale
    if len(roi_bgr.shape) == 3:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi_bgr.copy()

    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # เพิ่ม contrast นิดหนึ่ง
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    # blur แนวตั้ง (1 x blur_len)
    if blur_len % 2 == 0:
        blur_len += 1
    blurred = cv2.GaussianBlur(gray_eq, (1, blur_len), 0)

    # ผสมภาพเบลอ + ภาพเดิม ใน ROI
    alpha = max(0.0, min(1.0, alpha))
    beta = 1.0 - alpha
    mixed = cv2.addWeighted(blurred, alpha, gray_eq, beta, 0.0)

    result_bgr = cv2.cvtColor(mixed, cv2.COLOR_GRAY2BGR)
    return result_bgr


def main():
    args = parse_arguments()

    img = cv2.imread(args.input, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    if img is None:
        print("❌ ไม่พบไฟล์ภาพ:", args.input)
        return

    try:
        x, y, w, h = select_roi(img)
    except RuntimeError as e:
        print("❌", e)
        return

    roi = img[y:y+h, x:x+w]

    print(f"Processing ROI... blur_len={args.blur_len}, alpha={args.alpha}")
    result_roi = blur_ribs_in_roi(
        roi,
        blur_len=args.blur_len,
        alpha=args.alpha
    )

    # เอาผลไปวางคืนในภาพใหญ่
    output = img.copy()
    output[y:y+h, x:x+w] = result_roi

    # แสดง Original vs Result
    vis_original = img.copy()
    cv2.putText(vis_original, "Original", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    vis_result = output.copy()
    cv2.putText(vis_result, "Ribs Suppressed (blur ROI)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    h0 = vis_original.shape[0]
    vis_result = cv2.resize(vis_result, (vis_result.shape[1], h0))
    show = np.hstack((vis_original, vis_result))

    cv2.namedWindow("Rib Blur ROI", cv3.WINDOW_NORMAL)  # <- ถ้า error เปลี่ยนเป็น cv2
    cv2.imshow("Rib Blur ROI", show)

    print("\n--- กด 's' เพื่อบันทึกผลลัพธ์, 'ESC' เพื่อออก ---")
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k == ord('s'):
            cv2.imwrite(args.output, output)
            print("✅ Saved:", args.output)
            break
        elif k == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()