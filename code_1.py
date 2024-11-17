import mediapipe as mp
import cv2
import os
import pandas as pd
from sklearn.svm import SVC

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils
points = mp_pose.PoseLandmark

# Dataset path
path = "DATASET/TRAIN"  # Update with your dataset path
poses = {"plank": 0, "goddess": 1, "downdog" : 2, "tree" : 3, "warrior2" : 4}  # Update with all relevant pose classes
data = []

# Extract pose landmarks
for pose_name, label in poses.items():
    pose_path = os.path.join(path, pose_name)
    for img_name in os.listdir(pose_path):
        temp = []
        img = cv2.imread(os.path.join(pose_path, img_name))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                temp += [lm.x, lm.y, lm.z, lm.visibility]
            temp.append(label)
            data.append(temp)


columns = [f"{str(p)[13:]}_{coord}" for p in points for coord in ["x", "y", "z", "vis"]] + ["target"]
df = pd.DataFrame(data, columns=columns)
df.to_csv("pose_dataset.csv", index=False)

# Train the model
X = df.iloc[:, :-1]
y = df["target"]
model = SVC(kernel="poly")
model.fit(X, y)

def predict_pose_with_lines(image_path, model):
    
    
    # Load MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    pose = mp_pose.Pose()

    # Load the image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process pose landmarks
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        # Draw pose skeleton on the image
        mp_draw.draw_landmarks(
            img, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,  # Draw connections
            mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),  # Keypoints
            mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)  # Connections
        )

        # Extract landmarks for prediction
        temp = []
        for lm in results.pose_landmarks.landmark:
            temp += [lm.x, lm.y, lm.z, lm.visibility]
        
        # Predict the pose
        prediction = model.predict([temp])
        predicted_pose = list(poses.keys())[list(poses.values()).index(prediction[0])]

        # Annotate the image with the predicted pose
        cv2.putText(
            img, 
            predicted_pose, 
            (50, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (255, 255, 0), 
            3
        )

        # Display the image with pose skeleton
        cv2.imshow("Pose Detection with Skeleton", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No pose detected in the image.")

# Test the function
predict_pose_with_lines("DATASET/TEST/goddess/00000000.jpg", model)
# predict_pose_with_lines("DATASET/TEST/tree/00000031.jpg", model)
# predict_pose_with_lines("DATASET/TEST/warrior2/00000012.jpg", model)

