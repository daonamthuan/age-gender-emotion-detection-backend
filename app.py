import cv2
import numpy as np
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
from ultralytics import YOLO
import base64


app = Flask(__name__)
# Thiết lập timeout cho các request
app.config['PERMANENT_SESSION_LIFETIME'] = 600  # Timeout 10 phút
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 600  # Timeout 10 phút
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load YOLOv8 model for face detection
yolo_model = YOLO('yolov8n_face.pt')

# Load the age, gender, and emotion models
age_model = load_model('models/age_model.h5')
gender_model = load_model('models/gender_model.h5')
emotion_model = load_model('models/emotion_model.h5')

# Labels on Age, Gender and Emotion to be predicted
age_ranges = ['1-2', '3-9', '10-20', '21-27', '28-45', '46-65', '66-116']
gender_ranges = ['male', 'female']
emotion_ranges = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

def get_face_detections(image):
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Invalid image array")
    
    results = yolo_model(image)
    
    face_bounding_boxes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width = x2 - x1
            height = y2 - y1
            face_bounding_boxes.append((x1, y1, width, height))
    
    print("================================== Boundingbox =====================================")
    print(face_bounding_boxes)
    return face_bounding_boxes


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No image file in request"}), 400
    
    file = request.files['file']
    image = Image.open(file.stream)
    image = np.array(image)
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # Detect faces
    face_detections = get_face_detections(image)

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    predictions = []
    i = 0
    for (x, y, w, h) in face_detections:        
        face_img = gray_image[y:y+h, x:x+w]

        # sửa lại preprocessing chỗ này
        emotion_img = cv2.resize(face_img, (48, 48), interpolation = cv2.INTER_AREA)
        if np.sum([emotion_img])!=0:
                roi = emotion_img.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)

                prediction = emotion_model.predict(roi)[0]
                output_emotion=emotion_ranges[prediction.argmax()]
        
        gender_img = cv2.resize(face_img, (100, 100), interpolation = cv2.INTER_AREA)
        gender_image_array = np.array(gender_img)
        gender_input = np.expand_dims(gender_image_array, axis=0)
        output_gender=gender_ranges[np.argmax(gender_model.predict(gender_input))]

        age_image = cv2.resize(face_img, (200, 200), interpolation = cv2.INTER_AREA)
        age_input = age_image.reshape(-1, 200, 200, 1)
        output_age = age_ranges[np.argmax(age_model.predict(age_input))]


        predictions.append({
            'x': x,
            'y': y,
            'width': w,
            'height': h,
            'age': output_age,
            'gender': output_gender,
            'emotion': output_emotion
        })

    return jsonify(predictions)


@socketio.on('image')
def handle_image(data):
    image_data = data['image']
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    image = np.array(image)
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    face_detections = get_face_detections(image)

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    predictions = []
    for (x, y, w, h) in face_detections:
        face_img = gray_image[y:y+h, x:x+w]

        # sửa lại preprocessing chỗ này
        emotion_img = cv2.resize(face_img, (48, 48), interpolation=cv2.INTER_AREA)
        emotion_image_array = np.array(emotion_img)
        emotion_input = np.expand_dims(emotion_image_array, axis=0)
        output_emotion = emotion_ranges[np.argmax(emotion_model.predict(emotion_input))]

        gender_img = cv2.resize(face_img, (100, 100), interpolation=cv2.INTER_AREA)
        gender_image_array = np.array(gender_img)
        gender_input = np.expand_dims(gender_image_array, axis=0)
        output_gender = gender_ranges[np.argmax(gender_model.predict(gender_input))]

        age_image = cv2.resize(face_img, (200, 200), interpolation=cv2.INTER_AREA)
        age_input = age_image.reshape(-1, 200, 200, 1)
        output_age = age_ranges[np.argmax(age_model.predict(age_input))]

        predictions.append({
            'x': x,
            'y': y,
            'width': w,
            'height': h,
            'age': output_age,
            'gender': output_gender,
            'emotion': output_emotion
        })

    emit('prediction', predictions)

if __name__ == '__main__':
    app.run(debug=False)
