import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the pre-trained model
with open('./model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

# Initialize video capture
cap = cv2.VideoCapture(2)  # Ensure the correct camera index is used

if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

# Initialize MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Expanded labels dictionary for ASL alphabets
labels_dict = {i: chr(65 + i) for i in range(26)}
expected_features = 42  # 21 landmarks * 2 (x and y coordinates)

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Extract landmark coordinates
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            # Check if the correct number of features are extracted
            if len(data_aux) != expected_features:
                print(f"Skipping frame: Expected {expected_features} features, but got {len(data_aux)}")
                continue

            # Calculate bounding box for the hand
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10  # Adjusted to include the entire hand
            y2 = int(max(y_) * H) + 10  # Adjusted to include the entire hand

            # Predict the gesture
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            # Draw bounding box and prediction on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit the loop if 'q' is pressed
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
