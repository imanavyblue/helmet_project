from flask import Flask, render_template, Response, jsonify
import cv2
import math
import cvzone
from ultralytics import YOLO

# กำหนดเส้นทางของวิดีโอและโมเดล YOLO
video_path = "media/C0008.mp4"  # วิดีโอตัวอย่าง (หรือใช้กล้อง)
cap = cv2.VideoCapture(video_path)
model = YOLO("wieght/best.pt")

# ตั้งชื่อคลาสที่ต้องการตรวจจับ
classNames = ['With Helmet', 'Without Helmet']

app = Flask(__name__)

detected_classes = []

def generate_frames():
    global detected_classes
    while True:
        success, img = cap.read()
        if not success:
            break
        results = model(img, stream=True)
        detected_classes = []  # รีเซ็ตข้อมูลการตรวจจับทุกครั้งที่อ่านเฟรมใหม่
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h))
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])

                # เก็บข้อมูลคลาสที่ตรวจจับได้
                detected_classes.append(f'{classNames[cls]}: {conf:.2f}')

                # วาดกรอบและแสดงข้อมูลบนเฟรม
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=2, thickness=3)

        # แปลงภาพเป็น JPEG
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_detected_classes')
def get_detected_classes():
    return jsonify(detected_classes)

if __name__ == "__main__":
    app.run(debug=True)
