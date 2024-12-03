import os
import mediapipe as mp
import cv2
import pickle

DATA_DIR = './data'

# Initialize MediaPipe hands and drawing utilities
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)

data = []
labels = []

# Loop through directories and images in the DATA_DIR
for dir_name in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR, dir_name)
    
    # Ensure directory is valid
    if os.path.isdir(class_dir):
        for img_path in os.listdir(class_dir):
            data_aux = []  # Auxiliary list for storing features
            img = cv2.imread(os.path.join(class_dir, img_path))
            if img is None:
                continue  # Skip if the image cannot be read

            # Convert image to RGB (MediaPipe expects RGB)
            img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Process the image to find hand landmarks
            results = hands.process(img_RGB)

            # Extract hand landmarks if available
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        data_aux.append(lm.x)
                        data_aux.append(lm.y)
            
            # Ensure a fixed-length feature vector (84 elements: 21 landmarks * 2 hands * x and y)
            while len(data_aux) < 42:  # Pad with zeros if fewer landmarks
                data_aux.append(0.0)
            if len(data_aux) > 42:  # Truncate if more landmarks (unlikely with 2 hands max)
                data_aux = data_aux[:42]

            # Append the processed data and label
            data.append(data_aux)
            labels.append(dir_name)

# Save the data and labels into a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
