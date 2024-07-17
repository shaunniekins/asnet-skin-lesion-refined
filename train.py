# train.py

from model import AS_Net
from loss import WBEC
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from keras.callbacks import ModelCheckpoint
from PIL import Image
import glob
import numpy as np
import tensorflow as tf
import os
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

height = 192
width = 256


def read_from_paths(image_path_list, mask_path_list):
    images = []
    masks = []
    for img_path, mask_path in zip(image_path_list, mask_path_list):
        image = Image.open(img_path).convert('RGB')
        image = np.array(image, dtype=np.float32)
        # Resize the image to a fixed size
        image = cv2.resize(image, (width, height))
        image = tf.image.adjust_gamma(image / 255., gamma=1.6)

        mask = Image.open(mask_path)
        mask = np.array(mask, dtype=np.float32)
        # Resize the mask to a fixed size
        mask = cv2.resize(mask, (width, height),
                          interpolation=cv2.INTER_NEAREST)
        mask = np.expand_dims(mask, -1)

        images.append(image)
        masks.append(mask / 255)
    images_array = np.array(images)
    masks_array = np.array(masks)
    return images_array, masks_array


Dataset_add = 'dataset_isic18/'
Tr_add = 'ISIC2018_Task1-2_Training_Input'
Tr_ms_add = 'ISIC2018_Task1_Training_GroundTruth'

Tr_list = glob.glob(Dataset_add + Tr_add + '/*.jpg')
Tr_ms_list = glob.glob(Dataset_add + Tr_ms_add + '/*.png')

val_data = np.load('data_val.npy').astype(dtype=np.float32)
val_mask = np.load('mask_val.npy').astype(dtype=np.float32)

val_data = tf.image.adjust_gamma(val_data / 255., gamma=1.6)
val_mask = np.expand_dims(val_mask, axis=-1)
val_mask = val_mask / 255.

batch_size = 3 #16
nb_epoch = 10
steps_per_epoch = int(np.ceil(len(Tr_list)/batch_size))


def generator(all_image_list, all_mask_list):
    cnt = 0
    while True:
        images_array, masks_array = read_from_paths(all_image_list[cnt*batch_size:(cnt+1)*batch_size],
                                                    all_mask_list[cnt*batch_size:(cnt+1)*batch_size])
        yield images_array, masks_array

        cnt = (cnt + 1) % steps_per_epoch   # total_size/batch_size
        if cnt == 0:
            state = np.random.get_state()
            np.random.shuffle(all_image_list)
            np.random.set_state(state)
            np.random.shuffle(all_mask_list)


# Build model
model = AS_Net()

weights_path = './checkpoint/weights.weights.h5'
if os.path.exists(weights_path):
    model.load_weights(weights_path)
else:
    print(f"No weights found at {weights_path}, initializing from scratch.")

initial_learning_rate = 1e-4
decay_steps = 10000
decay_rate = 0.9  # double check if correct value

lr_schedule = ExponentialDecay(
    initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True)

optimizer = Adam(learning_rate=lr_schedule)

model.compile(optimizer=optimizer,
              loss=WBEC(), metrics=['binary_accuracy'])

mcp_save = ModelCheckpoint(
    filepath='./checkpoint/weights.weights.h5', save_weights_only=True)
mcp_save_best = ModelCheckpoint('./checkpoint_best/weights.weights_best.weights.h5',
                                verbose=1, save_best_only=True, save_weights_only=True, mode='min')

history = model.fit(x=generator(Tr_list, Tr_ms_list),
                    epochs=nb_epoch,
                    verbose=1,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=(val_data, val_mask), callbacks=[mcp_save, mcp_save_best])
print(model.optimizer.get_config())
