import pickle
import numpy as np
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load all .pkl gesture files
all_data = []
all_labels = []

for file in glob.glob("*_gesture_data.pkl"):
    with open(file, "rb") as f:
        data, labels = pickle.load(f)
        all_data.extend(data)
        all_labels.extend(labels)
    print(f"✅ Loaded {len(labels)} samples from {file}")

# Convert to arrays
X = np.array(all_data)
y = np.array(all_labels)

# Split into training/testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")

# Save the trained model
with open("gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as 'gesture_model.pkl'")
