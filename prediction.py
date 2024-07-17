# prediction.py

import numpy as np
import glob
from PIL import Image
import tensorflow as tf
import cv2
import os
from model import AS_Net

# Parameters
height = 192
width = 256
channels = 3

# Load test images
Dataset_add = 'dataset_isic18/'
Test_add = 'ISIC2018_Task1-2_Test_Input'
Test_list = glob.glob(Dataset_add + Test_add + '/*.jpg')


def preprocess_image(img_path):
    image = Image.open(img_path).convert('RGB')
    image = np.array(image, dtype=np.float32)
    image = cv2.resize(image, (width, height))
    image = tf.image.adjust_gamma(image / 255., gamma=1.6)
    return image


# Preprocess all test images
Test_data = np.array([preprocess_image(img_path) for img_path in Test_list])

# Load the trained model
model = AS_Net()
model.load_weights('./checkpoint_best/weights.weights_best.weights.h5')

# Make predictions
predictions = model.predict(Test_data, batch_size=16, verbose=1)

# Create a directory to save predictions if it doesn't exist
os.makedirs('predictions', exist_ok=True)

# Define thresholds for classification
low_threshold = 0.1  # Below this is considered low probability
high_threshold = 0.3  # Above this is considered high probability

# Process predictions and determine lesion presence
for i, pred in enumerate(predictions):
    # Calculate the percentage of pixels predicted as lesion
    lesion_percentage = np.mean(pred) * 100

    # Determine the likelihood of lesion presence
    if lesion_percentage < low_threshold:
        likelihood = "Low"
    elif lesion_percentage > high_threshold:
        likelihood = "High"
    else:
        likelihood = "Moderate"

    # Get the original image filename
    original_filename = os.path.basename(Test_list[i])

    # Save the prediction as an image
    img = Image.fromarray((pred.squeeze() * 255).astype(np.uint8))
    img = img.convert('L')  # Convert to grayscale
    img.save(f'predictions/pred_{original_filename}')

    # Print the result
    print(f"Image {original_filename}: {lesion_percentage:.2f}% likelihood of lesion (Likelihood: {likelihood})")

    # Save this information to a file
    with open('predictions/results.txt', 'a') as f:
        f.write(f"{original_filename}: {lesion_percentage:.2f}% likelihood of lesion (Likelihood: {likelihood})\n")

print("Predictions completed and saved.")
