# AS-Net Integrated Implementation

This directory contains an integrated implementation of AS-Net for skin lesion segmentation using the ISIC 2018 dataset. This version consolidates all the necessary components into a single file for easier execution.

## Contents

- `main.py` - The complete implementation including:
  - Model architecture (AS-Net with SAM and CAM attention modules)
  - Custom loss function (WBEC)
  - Data preparation utilities
  - Training pipeline
  - Evaluation metrics

## How to Use

### As a Python Script

Run the entire pipeline by executing:

```bash
python main.py
```

This will sequentially:

1. Prepare the ISIC dataset
2. Augment the training data (optional)
3. Train the AS-Net model
4. Evaluate performance on validation data

## Dataset Requirements

The code expects the ISIC 2018 dataset in the following structure:

- `dataset_isic18/ISIC2018_Task1-2_Training_Input/`
- `dataset_isic18/ISIC2018_Task1_Training_GroundTruth/`
- `dataset_isic18/ISIC2018_Task1-2_Validation_Input/`
- `dataset_isic18/ISIC2018_Task1_Validation_GroundTruth/`

## Directory Structure

Before running, create the following directories for checkpoints and output:

```bash
mkdir -p checkpoint checkpoint_best output
```
