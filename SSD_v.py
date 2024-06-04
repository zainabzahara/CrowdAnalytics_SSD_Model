import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Paths to the model and configuration file
model_path = 'mobilenet_iter_73000.caffemodel'
config_path = 'deploy.prototxt'

# Ensure the model files exist
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
if not os.path.isfile(config_path):
    raise FileNotFoundError(f"Config file not found: {config_path}")

# Load the pre-trained model
net = cv2.dnn.readNetFromCaffe(config_path, model_path)


# Function to detect persons in an image
def detect_persons(image):
    if image is None:
        raise ValueError("Image not loaded correctly. Please check the file path and ensure the image exists.")

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    persons = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            if idx == 15:  # Class label for 'person' in COCO dataset is 15
                persons += 1
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                label = f"Person: {confidence:.2f}"
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image, persons


# Load an image from your dataset
dataset_dir = r'H:\Zainab Zahara\HackerSpace\ML_Project1\part_A_final\train_data\images'
print(f"Loading images from: {dataset_dir}")

for image_name in os.listdir(dataset_dir):
    image_path = os.path.join(dataset_dir, image_name)
    if not image_path.endswith('.jpg'):  # Skip non-image files
        continue

    # Load and process the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        continue

    # Detect persons in the image
    result_image, num_persons = detect_persons(image)
    print(f"Number of persons detected in {image_name}: {num_persons}")

    # Display the result
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title(image_name)
    plt.show()
