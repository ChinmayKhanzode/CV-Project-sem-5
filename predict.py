import mediapipe as mp
import cv2
import os
import pandas as pd
from sklearn.svm import SVC

df = pd.read_csv("pose_dataset.csv")
# Train the model
X = df.iloc[:, :-1]
y = df["target"]
model = SVC(kernel="poly")
model.fit(X, y)

def predict_pose(image_path, model):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        temp = []
        for lm in results.pose_landmarks.landmark:
            temp += [lm.x, lm.y, lm.z, lm.visibility]
        prediction = model.predict([temp])
        predicted_pose = list(poses.keys())[list(poses.values()).index(prediction[0])]
        cv2.putText(img, predicted_pose, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
        cv2.imshow("Pose Detection", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No pose detected in the image.")

# Test the pose detection
predict_pose("DATASET/TEST/goddess/00000000.jpg", model)