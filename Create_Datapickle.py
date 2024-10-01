import os
import pickle
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'
data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(class_dir):
        continue

    for img_path in os.listdir(class_dir):
        img_file = os.path.join(class_dir, img_path)
        if not os.path.isfile(img_file):
            continue

        img = cv2.imread(img_file)
        if img is None:
            print(f"Error reading image {img_file}. Skipping...")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                x_ = [lm.x for lm in hand_landmarks.landmark]
                y_ = [lm.y for lm in hand_landmarks.landmark]

                min_x, min_y = min(x_), min(y_)

                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min_x)
                    data_aux.append(lm.y - min_y)

                data.append(data_aux)
                labels.append(dir_)

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Data processing complete. Data saved to 'data.pickle'.")
