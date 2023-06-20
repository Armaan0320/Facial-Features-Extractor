import cv2
import os

# Path to the folder containing the input images
image_folder = "ffhq-dataset-images"

# Load the cascade classifiers
eye_cascade = cv2.CascadeClassifier("cascades/haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier("cascades/haarcascade_mcs_mouth.xml")

# Lists to store the bounding box coordinates
eye_boxes = []
mouth_boxes = []

# Iterate over all images in the folder
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Read the image
        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Detect mouth
        mouth = mouth_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=11)

        # Store bounding box coordinates for eyes
        for (ex, ey, ew, eh) in eyes:
            eye_boxes.append((ex, ey, ew, eh))

        # Store bounding box coordinates for mouth
        if len(mouth) > 0:
            (mx, my, mw, mh) = mouth[0]
            mouth_boxes.append((mx, my, mw, mh))

        # Draw bounding boxes around eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(image, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Draw bounding box around mouth
        if len(mouth) > 0:
            (mx, my, mw, mh) = mouth[0]
            cv2.rectangle(image, (mx, my), (mx + mw, my + mh), (255, 0, 0), 2)

        # Show the processed image
        cv2.imshow("Image", image)
        cv2.waitKey(0)

# Cleanup
cv2.destroyAllWindows()

# Print the bounding box coordinates
print("Eye bounding boxes:")
for box in eye_boxes:
    print(box)

print("\nMouth bounding boxes:")
for box in mouth_boxes:
    print(box)
