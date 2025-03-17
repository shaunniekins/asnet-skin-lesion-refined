# Attention-Synergy-Network

This is an unofficial implementation of AS-Net: Attention Synergy Network for skin lesion segmentation. The original paper can be found [here](https://doi.org/10.1016/j.eswa.2022.117112).

## Project Structure

- `model.py` - Implementation of the AS-Net architecture
- `loss.py` - Custom Weighted Binary Cross-Entropy loss function
- `train.py` - Script for training the model
- `dataset_isic18.py` - Script to prepare ISIC 2018 dataset
- `augmentation.py` - Script for data augmentation
- `evaluate.py` - Script for model evaluation

## Directory Structure

- `dataset_isic18/` - Contains the ISIC 2018 dataset
  - `ISIC2018_Task1-2_Training_Input/` - Training images
  - `ISIC2018_Task1_Training_GroundTruth/` - Training masks
  - `ISIC2018_Task1-2_Validation_Input/` - Validation images
  - `ISIC2018_Task1_Validation_GroundTruth/` - Validation masks
- `checkpoint/` - Saves model weights during training
- `checkpoint_best/` - Saves the best model weights
- `output/` - Stores evaluation results and performance metrics
- `predictions/` - Stores model predictions

## Datasets

You can download the ISIC 2018 train dataset from [this](https://challenge.isic-archive.com/data/#2018) link.

## Setup and Process

1. Clone this repository:

   ```bash
   git clone <repository-url>
   cd asnet-original-version
   ```

2. Set up the environment:

   ```bash
   python3 -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   ```

3. Download and organize the ISIC 2018 dataset into the following structure:

   ```dir
   dataset_isic18/
   ├── ISIC2018_Task1-2_Training_Input/
   ├── ISIC2018_Task1_Training_GroundTruth/
   ├── ISIC2018_Task1-2_Validation_Input/
   └── ISIC2018_Task1_Validation_GroundTruth/
   ```

4. Prepare the dataset:

   ```bash
   python dataset_isic18.py
   ```

   This script resizes images to 192×256 and saves processed data as .npy files.

5. (Optional) Augment the dataset:

   ```bash
   python augmentation.py
   ```

   This generates additional training samples through rotations, flips, and zoom transformations.

6. Train the model:

   ```bash
   python train.py
   ```

   Training progress and model weights will be saved in the `checkpoint/` directory.

7. Evaluate the model:

   ```bash
   python evaluate.py
   ```

   This will generate performance metrics (Jaccard index, F1 score, AUC, etc.) and sample visualizations in the `output/` directory.

## Model Performance

After evaluation, the following metrics are computed:

- Area under the ROC curve
- Area under Precision-Recall curve
- Jaccard similarity score
- F1 score
- Accuracy, Sensitivity, Specificity, and Precision

Visual results including ROC curve, Precision-Recall curve, and sample predictions are saved in the `output/` directory.

> Note: This implementation has been modified to serve as a basis for another research topic. Changes were made to accommodate module updates and specific requirements.
