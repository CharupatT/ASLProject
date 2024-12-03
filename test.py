import cv2
import mediapipe as mp
import pickle
import numpy as np

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize webcam (ensure the correct camera index; 0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=True,  # For live video, set to False
    min_detection_confidence=0.3
)

while True:
    data_aux = []  # To store landmarks for the frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Convert frame to RGB for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hand landmarks
    result = hands.process(frame_rgb)

    # If hand landmarks are detected
    if result.multi_hand_landmarks:
        # Use only the first detected hand
        hand_landmarks = result.multi_hand_landmarks[0]
        for lm in hand_landmarks.landmark:
            data_aux.append(lm.x)
            data_aux.append(lm.y)

        # Check if the feature vector matches the model's expectations
        if len(data_aux) == 42:  # Ensure it matches the training dimensions
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = prediction[0]

            # Display the prediction on the frame
            cv2.putText(frame, f'Prediction: {predicted_character}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Draw landmarks on the frame
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

    # Show the frame with prediction and landmarks
    cv2.imshow('Hand Gesture Recognition (Random Forest)', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows() 