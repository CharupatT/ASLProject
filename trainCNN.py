import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))

filtered_data = []
filtered_labels = []
print(f"First few feature lengths: {[len(features) for features in data_dict['data'][:5]]}")

for features, label in zip(data_dict['data'], data_dict['labels']):
    if len(features) == 42:  # Replace with the correct length
        filtered_data.append(features)
        filtered_labels.append(label)

# Ensure data is homogeneous in shape
data = np.array(filtered_data, dtype=np.float32)
labels = np.array(filtered_labels, dtype=str)

print(f"Number of samples: {len(data)}")
print(f"Number of labels: {len(labels)}")

# Reshape data for CNN (e.g., into pseudo-images of shape 7x6)
data = data.reshape(-1, 7, 6, 1)  # Adding a channel dimension

# Encode labels into integers
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Convert labels to one-hot encoding
labels = to_categorical(labels)

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Define CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(7, 6, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(labels.shape[1], activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=16)

# Evaluate the model
y_predict = model.predict(x_test)
y_pred_classes = np.argmax(y_predict, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_test_classes, y_pred_classes)
print(f"Test score: {accuracy * 100:.2f}%")

# Save the model
model.save('cnn_model.h5')
print("Model saved as cnn_model.h5")
