import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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

# print(f"Feature data shape: {data.shape}")
# print(f"Labels shape: {labels.shape}")

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Test model
y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)

print(f"Test score: {score * 100:.2f}%")

f = open('model.p', 'wb')
pickle.dump({'model': model, 'labels': labels}, f)
f.close()
