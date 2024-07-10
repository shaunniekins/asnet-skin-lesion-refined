# Attention-Synergy-Network (Personal Reconfiguration and Reimplementation)

This is an unofficial implementation of AS-Net: Attention Synergy Network for skin lesion segmentation. The original paper can be found [here](https://doi.org/10.1016/j.eswa.2022.117112).

## Dataset

Download the ISIC 2018 train dataset from [this link](https://challenge.isic-archive.com/data/#2018).

## Setup and Process

1. Prepare the dataset:
   - Create a `dataset_isic18` directory.
   - Inside `dataset_isic18`, create the following subdirectories and add data from the ISIC 2018 dataset:
     - ISIC2018_Task1_Training_GroundTruth
     - ISIC2018_Task1_Validation_GroundTruth
     - ISIC2018_Task1-2_Training_Input
     - ISIC2018_Task1-2_Validation_Input

2. Set up the environment:
   ```
   python3 -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   ```

3. Create directories for checkpoints:
   ```
   mkdir checkpoint checkpoint_best output predictions
   ```

4. Prepare the ISIC 2018 dataset:
   ```
   python3 Prepare_ISIC2018.py
   ```

5. (Optional) Data augmentation:
   ```
   python3 augmentation.py
   ```
   This step increases the dataset size and improves robustness by generating augmented images and masks.

6. Train the model:
   ```
   python3 train.py
   ```

7. Evaluate the model:
   ```
   python3 evaluate.py
   ```

8. Generate predictions:
   ```
   python3 prediction.py
   ```


> Note: This implementation has been modified to serve as a basis for another research topic. Changes were made to accommodate module updates and specific requirements.