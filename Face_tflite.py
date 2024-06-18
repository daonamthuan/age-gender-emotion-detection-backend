import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import gc
import tempfile
import time
from keras.models import load_model

st.title("Đếm Người với Mô hình TFLite")

age_model_dir = 'age_model_pretrained.h5'
age_model = load_model(age_model_dir)
age_ranges = ['1-2', '3-9', '10-20', '21-27', '28-45', '46-65', '66-116']

options_1 = ["Ảnh", "Video", "Webcam"]
selected_option = st.selectbox("Nguồn:", options_1)

options_2 = ["MobileNetSSD"]
selected_option_2 = st.selectbox("Mô hình:", options_2)  

threshold = st.slider("Ngưỡng phát hiện", 0.0, 1.0)

if selected_option_2 == "MobileNetSSD":
    interpreter = tf.lite.Interpreter(model_path="detect.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    floating_model = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:
    pass

def detect_and_predict_age(image, imH, imW):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

    count_people = 0
    ages = []

    for i in range(len(scores)):
        if scores[i] >= threshold:
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))

            count_people += 1

            face = image[ymin:ymax, xmin:xmax]
            img_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            age_image = cv2.resize(img_gray, (200, 200), interpolation=cv2.INTER_AREA)
            age_input = age_image.reshape(-1, 200, 200, 1)
            output_age = age_ranges[np.argmax(age_model.predict(age_input))]
            ages.append(output_age)
            
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
            cv2.putText(image, f'Age: {output_age}', (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 4)

    return image, count_people, ages

if selected_option == "Ảnh":
    placeholder = st.empty()
    uploaded_file = st.file_uploader("Chọn ảnh...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        imH, imW, _ = image.shape
        image, count_people, ages = detect_and_predict_age(image, imH, imW)

        st.image(image, channels="BGR")
        placeholder.text(f"Số người: {count_people}, Tuổi: {', '.join(ages)}")

elif selected_option == "Video":
    video = st.file_uploader("Chọn video...", type=["mp4"])
    if video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video.read())
        path = tfile.name
        cap = cv2.VideoCapture(path)
        imW = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        imH = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 5)
        stframe = st.empty()
        placeholder = st.empty()
        fps_placeholder = st.empty()
        frame_counter = 0
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Lỗi khi đọc khung hình")
                break
            
            frame, count_people, ages = detect_and_predict_age(frame, imH, imW)
            stframe.image(frame, channels="BGR")
            placeholder.text(f"Số người: {count_people}, Tuổi: {', '.join(ages)}")

            frame_counter += 1
            if frame_counter % 20 == 0:
                fps = frame_counter / (time.time() - start_time)
                fps_placeholder.text(f"FPS: {fps:.2f}")
                frame_counter = 0
                start_time = time.time()

        cap.release()
        cv2.destroyAllWindows()

elif selected_option == "Webcam":
    if 'run' not in st.session_state:
        st.session_state.run = False

    def start_capture():
        st.session_state.run = True

    def stop_capture():
        st.session_state.run = False
        cap.release()
        cv2.destroyAllWindows()

    start_button = st.button("Bắt đầu quay", on_click=start_capture)
    stop_button = st.button("Dừng quay", on_click=stop_capture)

    cap = None
    frame_counter = 0
    placeholder = st.empty()
    stframe = st.empty()
    fps_placeholder = st.empty()
    start_time = time.time()

    if st.session_state.run:
        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FPS, 5)
        cap.set(3, 640)
        cap.set(4, 480)
        while st.session_state.run and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Lỗi khi đọc khung hình")
                break
            
            frame, count_people, ages = detect_and_predict_age(frame, 480, 640)
            stframe.image(frame, channels="BGR")
            placeholder.text(f"Số người: {count_people}, Tuổi: {', '.join(ages)}")

            frame_counter += 1
            if frame_counter % 20 == 0:
                fps = frame_counter / (time.time() - start_time)
                fps_placeholder.text(f"FPS: {fps:.2f}")
                frame_counter = 0
                start_time = time.time()

            gc.collect()

        cap.release()
        cv2.destroyAllWindows()
