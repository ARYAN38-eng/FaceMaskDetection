from flask import Flask,render_template,Response,url_for
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
model=load_model('facemaskdetection.h5')

app= Flask(__name__)

def detect_face_mask(frame):
    # Resize the frame to the expected input size (224x224)
    resized_frame = cv2.resize(frame, (224, 224))
    # Preprocess the frame if needed (e.g., normalization)
    # Perform prediction
    y_pred = model.predict(np.expand_dims(resized_frame, axis=0))
    class_label = "This Person is wearing mask on his face" if y_pred[0][0] < 0.5 else "Not Wearing Mask!!!"
    return class_label

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        class_label = detect_face_mask(frame)
        cv2.putText(frame, class_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/videofeed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)