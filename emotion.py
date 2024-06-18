import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from ultralytics import YOLO
from keras.models import load_model

model = YOLO('yolov8n_face.pt')

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Load your emotion, gender, and age models here
export_dir='emotion.h5'
emotion_model = load_model('models/emotion_model.h5')


# Define ranges or labels for emotion, gender, and age
emotion_ranges= ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Perform face detection with YOLO
    results = model(frame)
    print("Result from yolo model: \n", results.boxes)
    
    # Iterate through detected faces
    for result in results:
        boxes = result.boxes.xyxy
        
        for box in boxes:
            x1, y1, x2, y2 = box.tolist()
        # Extract face region
            face = frame[int(y1):int(y2), int(x1):int(x2)]

            # Perform emotion, gender, and age prediction
            # Assuming gray scale image for emotion, gender, and age models
            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # Resize face for emotion recognition (48x48), gender (100x100), and age (200x200)
            emotion_img = cv2.resize(gray_face, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([emotion_img])!=0:
                roi = emotion_img.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)

                prediction = emotion_model.predict(roi)[0]
                label=emotion_ranges[prediction.argmax()]


            # Draw rectangle around the face and label with attributes
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f'{label}, {int(x1), int(y1), int(x2),int(y2)}'
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Face Detection', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
