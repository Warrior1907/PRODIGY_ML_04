import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Ask user for gesture label
label_name = input("Enter gesture label (e.g., thumbs_up, fist, peace): ").strip()

# Initialize data holders
data = []
labels = []

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access webcam.")
    exit()

print("Webcam initialized. Press 'q' to stop recording gesture.")

# Start capturing frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark features
            landmark_list = []
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])
            data.append(landmark_list)
            labels.append(label_name)

    # Show number of samples collected
    cv2.putText(frame, f"Samples: {len(data)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show video frame
    cv2.imshow("Collecting Gesture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close window
cap.release()
cv2.destroyAllWindows()

# Save the collected data
filename = f"{label_name}_gesture_data.pkl"
with open(filename, "wb") as f:
    pickle.dump((data, labels), f)

print(f"Saved {len(data)} samples to {filename}")
