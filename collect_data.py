import os
import cv2

# Configuration
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3
dataset_size = 200

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

done = False

# Data collection loop


class_dir = os.path.join(DATA_DIR, "Z")
if not os.path.exists(class_dir):
    os.makedirs(class_dir)

print('Collecting data for class {}')
count = 0
while count < dataset_size:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        done = True
        break

    # Display the frame
    cv2.imshow("Frame", frame)

        # Save the captured frame to the appropriate class folder
    frame_path = os.path.join(class_dir, f"Z{count}.jpg")
    cv2.imwrite(frame_path, frame)

    count += 1

        # Break if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        done = True
        break



# Cleanup
cap.release()
cv2.destroyAllWindows()

print("Data collection complete.")
