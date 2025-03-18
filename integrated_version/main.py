import os
import glob
import math
import numpy as np
import tensorflow as tf
from keras import Model, Input, backend
from keras.applications import VGG16
from keras.layers import (
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    Activation,
    UpSampling2D,
    concatenate,
    Multiply,
    GlobalAveragePooling2D,
    Dense,
    Reshape,
)
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
from keras.losses import Loss
from PIL import Image
from sklearn.metrics import (
    f1_score,
    jaccard_score,
    precision_recall_curve,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt


# #################### Model Definition ####################


def AS_Net(input_size=(192, 256, 3)):
    inputs = Input(input_size)
    VGGnet = VGG16(weights="imagenet", include_top=False, input_shape=(192, 256, 3))
    output1 = Model(inputs=VGGnet.inputs, outputs=VGGnet.get_layer(index=2).output)(
        inputs
    )
    output2 = Model(inputs=VGGnet.inputs, outputs=VGGnet.get_layer(index=5).output)(
        inputs
    )
    output3 = Model(inputs=VGGnet.inputs, outputs=VGGnet.get_layer(index=9).output)(
        inputs
    )
    output4 = Model(inputs=VGGnet.inputs, outputs=VGGnet.get_layer(index=13).output)(
        inputs
    )
    output5 = Model(inputs=VGGnet.inputs, outputs=VGGnet.get_layer(index=17).output)(
        inputs
    )

    merge1 = concatenate(
        [output4, UpSampling2D((2, 2), interpolation="bilinear")(output5)], axis=-1
    )
    SAM1 = SAM(filters=1024)(merge1)
    CAM1 = CAM(filters=1024)(merge1)

    merge21 = concatenate(
        [output3, UpSampling2D((2, 2), interpolation="bilinear")(SAM1)], axis=-1
    )
    merge22 = concatenate(
        [output3, UpSampling2D((2, 2), interpolation="bilinear")(CAM1)], axis=-1
    )
    SAM2 = SAM(filters=512)(merge21)
    CAM2 = CAM(filters=512)(merge22)

    merge31 = concatenate(
        [output2, UpSampling2D((2, 2), interpolation="bilinear")(SAM2)], axis=-1
    )
    merge32 = concatenate(
        [output2, UpSampling2D((2, 2), interpolation="bilinear")(CAM2)], axis=-1
    )
    SAM3 = SAM(filters=256)(merge31)
    CAM3 = CAM(filters=256)(merge32)

    merge41 = concatenate(
        [output1, UpSampling2D((2, 2), interpolation="bilinear")(SAM3)], axis=-1
    )
    merge42 = concatenate(
        [output1, UpSampling2D((2, 2), interpolation="bilinear")(CAM3)], axis=-1
    )
    SAM4 = SAM(filters=128)(merge41)
    CAM4 = CAM(filters=128)(merge42)

    output = Synergy()((SAM4, CAM4))
    output = Activation("sigmoid")(output)

    model = Model(inputs=inputs, outputs=output)

    return model


class SAM(Model):
    def __init__(self, filters=1024):
        super(SAM, self).__init__()
        self.filters = filters

        self.conv1 = Conv2D(
            self.filters // 4,
            3,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )
        self.conv2 = Conv2D(
            self.filters // 4,
            3,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )
        self.conv3 = Conv2D(
            self.filters // 4,
            3,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )

        self.conv4 = Conv2D(
            self.filters // 4, 1, activation="relu", kernel_initializer="he_normal"
        )

        self.pool1 = MaxPooling2D((2, 2))
        self.upsample1 = UpSampling2D((2, 2), interpolation="bilinear")
        self.W1 = Conv2D(
            self.filters // 4, 1, activation="sigmoid", kernel_initializer="he_normal"
        )
        self.pool2 = MaxPooling2D((4, 4))
        self.upsample2 = UpSampling2D((4, 4), interpolation="bilinear")
        self.W2 = Conv2D(
            self.filters // 4, 1, activation="sigmoid", kernel_initializer="he_normal"
        )

    def call(self, inputs):
        out1 = self.conv3(self.conv2(self.conv1(inputs)))
        out2 = self.conv4(inputs)

        merge1 = self.W1(Activation("relu")(self.upsample1(self.pool1(out2))))
        merge2 = self.W2(Activation("relu")(self.upsample2(self.pool2(out2))))

        out3 = merge1 + merge2

        y = Multiply()([out1, out3]) + out2

        return y


class CAM(Model):
    def __init__(self, filters, reduction_radio=16):
        super(CAM, self).__init__()
        self.filters = filters

        self.conv1 = Conv2D(
            self.filters // 4,
            3,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )
        self.conv2 = Conv2D(
            self.filters // 4,
            3,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )
        self.conv3 = Conv2D(
            self.filters // 4,
            3,
            activation="relu",
            padding="same",
            kernel_initializer="he_normal",
        )

        self.conv4 = Conv2D(
            self.filters // 4, 1, activation="relu", kernel_initializer="he_normal"
        )

        self.gpool = GlobalAveragePooling2D()
        self.fc1 = Dense(
            self.filters // (4 * reduction_radio), activation="relu", use_bias=False
        )
        self.fc2 = Dense(self.filters // 4, activation="sigmoid", use_bias=False)
        self.reshape = Reshape((1, 1, self.filters))

    def call(self, inputs):
        out1 = self.conv3(self.conv2(self.conv1(inputs)))
        out2 = self.conv4(inputs)
        out3 = self.fc2(self.fc1(self.gpool(out2)))

        y = Multiply()([out1, out3]) + out2

        return y


class Synergy(Model):
    def __init__(self, alpha=0.5, belta=0.5):
        super(Synergy, self).__init__()
        # Make alpha and beta trainable parameters instead of fixed values
        self.alpha = tf.Variable(alpha, trainable=True, name="alpha", dtype=tf.float32)
        self.beta = tf.Variable(belta, trainable=True, name="beta", dtype=tf.float32)
        self.conv = Conv2D(1, 3, padding="same", kernel_initializer="he_normal")
        self.bn = BatchNormalization()

    def call(self, inputs):
        x, y = inputs
        # Use the trainable parameters for weighting
        inputs = self.alpha * x + self.beta * y
        y = self.bn(self.conv(inputs))
        return y


# #################### Loss Function ####################


class WBEC(Loss):
    def __init__(self, weight=2.5):
        super(WBEC, self).__init__()
        self.weight = weight

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        epslion_ = tf.constant(backend.epsilon(), y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, epslion_, 1.0 - epslion_)

        wbce = self.weight * y_true * tf.math.log(y_pred + backend.epsilon())
        wbce += (1 - y_true) * tf.math.log(1 - y_pred + backend.epsilon())

        return -wbce


# #################### Data Preparation ####################


def prepare_isic_data(
    dataset_path="dataset_isic18/", height=192, width=256, channels=3
):
    """
    Prepares the ISIC 2018 dataset by resizing images and masks
    and saving them as numpy arrays.
    """

    Tr_add = os.path.join(dataset_path, "ISIC2018_Task1-2_Training_Input")
    Tr_ms_add = os.path.join(dataset_path, "ISIC2018_Task1_Training_GroundTruth")
    Va_add = os.path.join(dataset_path, "ISIC2018_Task1-2_Validation_Input")
    Va_ms_add = os.path.join(dataset_path, "ISIC2018_Task1_Validation_GroundTruth")

    Tr_list = glob.glob(os.path.join(Tr_add, "*.jpg"))
    Tr_ms_list = glob.glob(os.path.join(Tr_ms_add, "*.png"))
    Va_list = glob.glob(os.path.join(Va_add, "*.jpg"))
    Va_ms_list = glob.glob(os.path.join(Va_ms_add, "*.png"))

    print(
        f"Training images: {len(Tr_list)}, Training masks: {len(Tr_ms_list)}, "
        f"Validation images: {len(Va_list)}, Validation masks: {len(Va_ms_list)}"
    )

    Data_train_2018 = np.zeros([len(Tr_list), height, width, channels])
    Label_train_2018 = np.zeros([len(Tr_ms_list), height, width])
    Data_validate_2018 = np.zeros([len(Va_list), height, width, channels])
    Label_validate_2018 = np.zeros([len(Va_ms_list), height, width])

    for idx, img_path in enumerate(Tr_list):
        print(f"Processing training image {idx + 1}/{len(Tr_list)}")
        img = Image.open(img_path).convert("RGB")
        img = np.array(img.resize((width, height))).astype(dtype=np.float32)
        Data_train_2018[idx, :, :, :] = img

        img2 = Image.open(Tr_ms_list[idx])
        img2 = np.array(img2.resize((width, height))).astype(dtype=np.float32)
        Label_train_2018[idx, :, :] = img2

    for idx, img_path in enumerate(Va_list):
        print(f"Processing validation image {idx + 1}/{len(Va_list)}")
        img = Image.open(img_path).convert("RGB")
        img = np.array(img.resize((width, height))).astype(dtype=np.float32)
        Data_validate_2018[idx, :, :, :] = img

        img2 = Image.open(Va_ms_list[idx])
        img2 = np.array(img2.resize((width, height))).astype(dtype=np.float32)
        Label_validate_2018[idx, :, :] = img2

    print("Reading ISIC 2018 finished")

    np.save("data_train", Data_train_2018)
    np.save("data_val", Data_validate_2018)
    np.save("mask_train", Label_train_2018)
    np.save("mask_val", Label_validate_2018)


# #################### Data Augmentation ####################


def augment_data(
    file_dir1="./dataset_isic18/ISIC2018_Task1-2_Training_Input",
    save_path1="./dataset_isic18/preview/img",
    file_dir2="./dataset_isic18/ISIC2018_Task1_Training_GroundTruth",
    save_path2="./dataset_isic18/preview/mask",
    shape=(192, 256),
    seed=40,
):
    """
    Augments the training data using keras ImageDataGenerator.
    """

    if not os.path.exists(save_path1):
        os.makedirs(save_path1)
    if not os.path.exists(save_path2):
        os.makedirs(save_path2)

    datagen = image.ImageDataGenerator(
        fill_mode="nearest",
        rotation_range=270,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        data_format="channels_last",
    )
    gen1 = datagen.flow_from_directory(
        file_dir1,
        target_size=shape,
        batch_size=15,
        class_mode="input",
        save_to_dir=save_path1,
        save_prefix="tran_",
        seed=seed,
        save_format="jpg",
    )
    gen2 = datagen.flow_from_directory(
        file_dir2,
        target_size=shape,
        batch_size=15,
        class_mode="input",
        save_to_dir=save_path2,
        save_prefix="tran_",
        seed=seed,
        save_format="png",
    )

    step = math.ceil(len(gen1.classes) / gen1.batch_size)
    for i in range(4 * step):
        gen1.next()
        gen2.next()


# #################### Training ####################


def read_from_paths(image_path_list, mask_path_list):
    """
    Reads images and masks from given paths and returns numpy arrays.
    """
    images = []
    masks = []
    for img_path, mask_path in zip(image_path_list, mask_path_list):
        image = Image.open(img_path).convert("RGB")
        image = np.array(image, dtype=np.float32)
        image = tf.image.adjust_gamma(image / 255.0, gamma=1.6)

        mask = Image.open(mask_path)
        mask = np.array(mask, dtype=np.float32)
        mask = np.expand_dims(mask, -1)

        images.append(image)
        masks.append(mask / 255)
    images_array = np.array(images)
    masks_array = np.array(masks)
    return images_array, masks_array


def train_model(
    dataset_path="dataset_isic18/",
    checkpoint_path="./checkpoint/weights.hdf5",
    checkpoint_best_path="./checkpoint_best/weights_best.hdf5",
    height=192,
    width=256,
    batch_size=16,
    nb_epoch=75,
):
    """
    Trains the AS-Net model.
    """

    Tr_add = os.path.join(dataset_path, "ISIC2018_Task1-2_Training_Input")
    Tr_ms_add = os.path.join(dataset_path, "ISIC2018_Task1_Training_GroundTruth")

    Tr_list = glob.glob(os.path.join(Tr_add, "*.jpg"))
    Tr_ms_list = glob.glob(os.path.join(Tr_ms_add, "*.png"))

    val_data = np.load("data_val.npy").astype(dtype=np.float32)
    val_mask = np.load("mask_val.npy").astype(dtype=np.float32)

    val_data = tf.image.adjust_gamma(val_data / 255.0, gamma=1.6)
    val_mask = np.expand_dims(val_mask, axis=-1)
    val_mask = val_mask / 255.0

    steps_per_epoch = int(np.ceil(len(Tr_list) / batch_size))

    def generator(all_image_list, all_mask_list):
        cnt = 0
        while True:
            images_array, masks_array = read_from_paths(
                all_image_list[cnt * batch_size : (cnt + 1) * batch_size],
                all_mask_list[cnt * batch_size : (cnt + 1) * batch_size],
            )
            yield images_array, masks_array

            cnt = (cnt + 1) % steps_per_epoch
            if cnt == 0:
                state = np.random.get_state()
                np.random.shuffle(all_image_list)
                np.random.set_state(state)
                np.random.shuffle(all_mask_list)

    model = AS_Net()
    # model.load_weights(checkpoint_path) # Load weights if needed
    model.compile(
        optimizer=Adam(learning_rate=1e-4, decay=1e-7),
        loss=WBEC(),
        metrics=["binary_accuracy"],
    )

    mcp_save = ModelCheckpoint(checkpoint_path, save_weights_only=True)
    mcp_save_best = ModelCheckpoint(
        checkpoint_best_path,
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
    )

    history = model.fit(
        x=generator(Tr_list, Tr_ms_list),
        epochs=nb_epoch,
        verbose=1,
        steps_per_epoch=steps_per_epoch,
        validation_data=(val_data, val_mask),
        callbacks=[mcp_save, mcp_save_best],
    )
    print(model.optimizer.get_config())


# #################### Evaluation ####################


def evaluate_model(
    checkpoint_best_path="./checkpoint_best/weights_best.hdf5",
    output_folder="output/",
):
    """
    Evaluates the trained AS-Net model on the validation set.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    te_data = np.load("data_val.npy").astype(np.float32)
    te_mask = np.load("mask_val.npy").astype(np.float32)
    te_mask = np.expand_dims(te_mask, axis=-1)

    te_data = tf.image.adjust_gamma(te_data / 255.0, gamma=1.6)
    te_mask /= 255.0

    print("ISIC18 Validation Dataset loaded")

    model = AS_Net()
    model.load_weights(checkpoint_best_path)
    predictions = model.predict(te_data, batch_size=8, verbose=1)

    y_scores = predictions.reshape(
        predictions.shape[0]
        * predictions.shape[1]
        * predictions.shape[2]
        * predictions.shape[3],
        1,
    )

    y_true = te_mask.reshape(
        te_mask.shape[0] * te_mask.shape[1] * te_mask.shape[2] * te_mask.shape[3], 1
    )

    y_scores = np.where(y_scores > 0.5, 1, 0)
    y_true = np.where(y_true > 0.5, 1, 0)

    # Area under the ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    AUC_ROC = roc_auc_score(y_true, y_scores)
    print("\nArea under the ROC curve: " + str(AUC_ROC))
    roc_curve = plt.figure()
    plt.plot(fpr, tpr, "-", label="Area Under the Curve (AUC = %0.4f)" % AUC_ROC)
    plt.title("ROC curve")
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.legend(loc="lower right")
    plt.savefig(output_folder + "ROC.png")

    # Precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    AUC_prec_rec = np.trapz(precision, recall)
    print("\nArea under Precision-Recall curve: " + str(AUC_prec_rec))
    prec_rec_curve = plt.figure()
    plt.plot(
        recall,
        precision,
        "-",
        label="Area Under the Curve (AUC = %0.4f)" % AUC_prec_rec,
    )
    plt.title("Precision - Recall curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower right")
    plt.savefig(output_folder + "Precision_recall.png")

    # Confusion matrix
    threshold_confusion = 0.5
    print(
        "\nConfusion matrix:  Custom threshold (for positive) of "
        + str(threshold_confusion)
    )
    y_pred = np.empty((y_scores.shape[0]))
    for i in range(y_scores.shape[0]):
        if y_scores[i] >= threshold_confusion:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    confusion = confusion_matrix(y_true, y_pred)
    print(confusion)
    accuracy = 0
    if float(np.sum(confusion)) != 0:
        accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
    print("Global Accuracy: " + str(accuracy))
    specificity = 0
    if float(confusion[0, 0] + confusion[0, 1]) != 0:
        specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
    print("Specificity: " + str(specificity))
    sensitivity = 0
    if float(confusion[1, 1] + confusion[1, 0]) != 0:
        sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
    print("Sensitivity: " + str(sensitivity))
    precision_value = 0
    if float(confusion[1, 1] + confusion[0, 1]) != 0:
        precision_value = float(confusion[1, 1]) / float(
            confusion[1, 1] + confusion[0, 1]
        )
    print("Precision: " + str(precision_value))

    # Jaccard similarity index
    jaccard_index = jaccard_score(y_true, y_pred)
    print("\nJaccard similarity score: " + str(jaccard_index))

    # F1 score
    F1_score = f1_score(
        y_true, y_pred, labels=None, average="binary", sample_weight=None
    )
    print("\nF1 score (F-measure): " + str(F1_score))

    # Save the results
    file_perf = open(os.path.join(output_folder, "performances.txt"), "w")
    file_perf.write(
        "Area under the ROC curve: "
        + str(AUC_ROC)
        + "\nArea under Precision-Recall curve: "
        + str(AUC_prec_rec)
        + "\nJaccard similarity score: "
        + str(jaccard_index)
        + "\nF1 score (F-measure): "
        + str(F1_score)
        + "\n\nConfusion matrix:"
        + str(confusion)
        + "\nACCURACY: "
        + str(accuracy)
        + "\nSENSITIVITY: "
        + str(sensitivity)
        + "\nSPECIFICITY: "
        + str(specificity)
        + "\nPRECISION: "
        + str(precision_value)
    )
    file_perf.close()

    # Save 10 results with error rate lower than threshold
    threshold = 300
    predictions = np.where(predictions > 0.5, 1, 0)
    te_mask = np.where(te_mask > 0.5, 1, 0)
    good_prediction = np.zeros([predictions.shape[0], 1], np.uint8)
    id_m = 0
    for idx in range(predictions.shape[0]):
        esti_sample = predictions[idx]
        true_sample = te_mask[idx]
        esti_sample = esti_sample.reshape(
            esti_sample.shape[0] * esti_sample.shape[1] * esti_sample.shape[2], 1
        )
        true_sample = true_sample.reshape(
            true_sample.shape[0] * true_sample.shape[1] * true_sample.shape[2], 1
        )
        er = 0
        for idy in range(true_sample.shape[0]):
            if esti_sample[idy] != true_sample[idy]:
                er = er + 1
        if er < threshold:
            good_prediction[id_m] = idx
            id_m += 1

    fig, ax = plt.subplots(10, 3, figsize=[15, 15])

    for idx in range(10):
        ax[idx, 0].imshow(te_data[good_prediction[idx, 0]])
        ax[idx, 1].imshow(np.squeeze(te_mask[good_prediction[idx, 0]]), cmap="gray")
        ax[idx, 2].imshow(np.squeeze(predictions[good_prediction[idx, 0]]), cmap="gray")

    plt.savefig(os.path.join(output_folder, "sample_results.png"))


# #################### Main Execution ####################

if __name__ == "__main__":
    # Prepare data (this will take time)
    prepare_isic_data()
    # Augment data (optional, also takes time)
    augment_data()
    # Train the model
    train_model()
    # Evaluate the model
    evaluate_model()

    print("All tasks completed!")
