from flask import Flask, render_template, Response, redirect, url_for
from ultralytics import YOLO
import cv2
import math

app = Flask(__name__)
cap = None
model = None
classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

def init_camera():
    global cap
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

def init_model():
    global model
    model = YOLO('C:/Users/sandi/Downloads/best_det.pt')

@app.route('/')
def admin():
    if cap is not None:
        cap.release()
    return render_template('admin.html')

def gen(camera):
    while True:
        success, img = camera.read()
        results = model(img, stream=True)

        for r in results:
            boxes = r.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                confidence = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]

                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.putText(img, class_name, org, font, fontScale, color, thickness)

            ret, frame = cv2.imencode('.jpg', img)
            if not ret:
                break

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')

@app.route('/camera')
def camera():
    init_camera()
    init_model()
    return render_template('camera.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(cap), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
