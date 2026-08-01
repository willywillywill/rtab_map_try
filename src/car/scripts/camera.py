#!/usr/bin/env python3
import cv2


def main():
    video_device = "/dev/video0"

    cap = cv2.VideoCapture(video_device)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera: {video_device}")
        print("請先檢查：")
        print("ls /dev/video*")
        print("v4l2-ctl --list-devices")
        return

    print(f"[INFO] Opened camera: {video_device}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("[WARN] Failed to read frame")
            continue

        cv2.imshow("Astra RGB Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()