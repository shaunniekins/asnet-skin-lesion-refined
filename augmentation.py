# augmentation.py

import math
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Define paths
file_dir1 = "./dataset_isic18/ISIC2018_Task1-2_Training_Input"
save_path1 = "./dataset_isic18/preview/img"
file_dir2 = "./dataset_isic18/ISIC2018_Task1_Training_GroundTruth"
save_path2 = "./dataset_isic18/preview/mask"

# Create save directories if they don't exist
os.makedirs(save_path1, exist_ok=True)
os.makedirs(save_path2, exist_ok=True)

shape = (192, 256)
seed = 40

# Set generator parameters
datagen = ImageDataGenerator(
    fill_mode='nearest',
    rotation_range=90,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    data_format='channels_last')

# Check if directories exist and have images
if not os.path.exists(file_dir1) or len(os.listdir(file_dir1)) == 0:
    print(f"Error: Directory {file_dir1} does not exist or is empty")
if not os.path.exists(file_dir2) or len(os.listdir(file_dir2)) == 0:
    print(f"Error: Directory {file_dir2} does not exist or is empty")

# Generate augmented images
gen1 = datagen.flow_from_directory(
    os.path.dirname(file_dir1),
    classes=[os.path.basename(file_dir1)],
    target_size=shape,
    batch_size=15,
    class_mode=None,
    save_to_dir=save_path1,
    save_prefix='aug_',
    seed=seed,
    save_format='jpg')

gen2 = datagen.flow_from_directory(
    os.path.dirname(file_dir2),
    classes=[os.path.basename(file_dir2)],
    target_size=shape,
    batch_size=15,
    class_mode=None,
    save_to_dir=save_path2,
    save_prefix='aug_',
    seed=seed,
    save_format='png')

# Generate augmented data
num_files = len([f for f in os.listdir(file_dir1) if f.endswith('.jpg') or f.endswith('.png')])
step = math.ceil(num_files / 15)
# Augment data by 4 times
for i in range(4 * step):
    gen1.next()
    gen2.next()

print("Augmentation completed.")