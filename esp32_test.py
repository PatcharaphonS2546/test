import cv2

URL = "http://10.63.100.193:81/stream"  # เปลี่ยนเป็น URL ของคุณ

cap = cv2.VideoCapture(URL)
if not cap.isOpened():
    print("ไม่สามารถเชื่อมต่อกล้องได้")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("รับภาพไม่สำเร็จ")
        break
    frame = cv2.flip(frame, 0)
    cv2.imshow("ESP32-CAM", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()