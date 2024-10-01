import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Add the total number of classes (including new signs)
number_of_classes = 26  # A-Z alphabets
dataset_size = 300

cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        cv2.putText(frame, 'Ready? Press "Q" to start!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        img_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(img_path, frame)
        counter += 1

    print(f'Done collecting data for class {j}')

cap.release()
cv2.destroyAllWindows()
