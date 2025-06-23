# Bachelor End Project - Matching Multi-stain tissue sections of skincancer WSIs
A Dual-Image Vision Transformer (VIT) adapted to make a match prediction for image pairs with varying input sizes.

This repository contains code to support the BEP: ***"Deep Learning for Multi-Stain Tissue Section Matching in Whole Slide Images"***

![Example workflow including Dual-Image ViT application](figures/Overview_model_processing_cropped.png)


## Project overview

This project proposes a deep learning model for matching tissue cross-sections across differently stained whole-slide images (WSIs), such as H&E and IHC, using a Dual-Image ViT. The model aims to support pathologists by enabling robust cross-stain matching and serves as a foundational component for future multi-modal diagnostic pipelines.

**Key contributions include:**

- A dual-stream ViT architecture for stain-invariant representation learning  
- Patch-based training and ranking pipeline for image pair similarity  
- Evaluation across varying match-to-non-match ratios to simulate real-world data imbalance  
- Performance analysis based on recall@k, median rank, and case-level accuracy

The model is suitable for integration as a preprocessing module in automated diagnostic systems and can potentially help reduce manual workload in pathology workflows.

## Requirements

To run this project, you will need the following documents and data (excluded from this repository):
- H&E and IHC WSI crops (data used for this research)
- Ground-truth rotations file (json) for IHC crops
- Ground-truth rotations file (json) for H&E crops
- Matches file (json) including all correct matches
- Configurations file containing all paths and hyperparameters

## Repository structure
This repository is structered in the following way:
```bash
BEP_tissue_alignment/
├── preprocessing/                                      # Folder containing scripts for preprocessing H&E and IHC WSI crops into model input based on the json files
│   ├── mask_crops.py                                   # Removes background and replaces with uniform color (crops → crops_masked)
│   ├── correct_rotations_padding.py                    # Rotates and pads the images and masks based on binary masks and rotation files (crops_masked → _rotated)
│   ├── select_matched_images.py                        # Selects matched images and masks based on matches file (_rotated → _matched)
│   └── select_size.py                                  # Filters data on pixel threshold to fit GPU memory, based on matches file and matched images (matched.csv → filtered.csv)
├── training/                                           # Folder containing scripts for training and folder with required model architecture, parameters and utils
│   ├── external/                                       # Folder containing required model architecture, parameters and (dataset)utils
│   │   ├── dataset_utils.py                            # Script with utils for initialising the training and valitation dataset
│   │   ├── DualBranchViT.py                            # Model architecture for the Dual-Branch ViT (not yet used in this research)
│   │   ├── DualInputViT.py                             # Model architecture for the Dual-Image ViT used in this research
│   │   ├── ViT_utils.py                                # Script with utils functions for the model architecture initialisation
│   │   └── vit_wee_patch16_reg1_gap_256.sbb_in1k.pth   # Pre-trained model parameters
│   └── training.py                                     # Script for training the ViT model
├── evaluation/                                         # Folder containing the scripts for evaluating the retrieval performance
│   ├── eval_ranking.py                                 # Applies the traied model to all possible matches and saves rankings and metrics
│   └── analyse_scores.py                               # Computes futher details metrics based on the saved rankings
├── figures/                                            # Folder containing any figures supporting the README file
├── config.py                                           # Configuration file containing all paths, hyperparameters and variables; is not yet created, but is required in order to run the scripts
├── requitements.txt                                    # Document containing all required packages in order to run the scripts
└── README.md                                           # Overview of repository information and instructions about the scripts

```

## Getting started

### 1. Clone the repository
To get started, download the repository to your machine: 
```bash
git clone https://github.com/evipaulides/BEP_tissue_alignment.git
cd BEP_tissue_alignment
```

###  2. Create a virtual environment
To create a new (pip) environment, run the following: 
```bash
python -m venv BEP_venv
BEP_venv\Scripts\activate    
```

### 3. Install required packages
Install the required packages inyo your virtual environment by running the following command:
```bash
pip install -r requirements.txt
```

## Project walk-through
A walk-through of how to run the scripts to preprocess the data into model input, train the ViT and evaluate the model's performance.

### 0. Set-up
First, create a virtual environment, install all packages from the `requirements.txt` file and create a `config.py` file with all necessary paths, variables and parameters.

### 1. Preprocessing - Preperating raw data into model input
#### 1.1. mask_crops.py
Run this script to remove the WSI background and replace it with a uniform background color (that matches a standard WSI background) to remove any visible adjacent cross-sections from the image. The input: Path to the directory containing images of a specific train, path to the directory containing the corresponding masks and the path to the directory to save the masked crops. Note: this needs to be runned for both stainings.

#### 1.2. correct_rotations_padding.py
Run this script to rotate the tissue sections to align the tissue layers horizontally, with the epidermis at the top. Then, the crops and masks are padded into suitable model input based on the rotations files. The input: Rotations file, paths to the directory with crops masked, path to the directory with masks and the paths to the directories of the rotated, padded images and masks to save the processed data. Note: this needs to be runned for both stainings.

#### 1.3. select_matched_images.py
Run this script to select the images that are part of a match pair, based on the matches file. The input: the matches file, paths to the rotated, padded images directories and paths to the matched images directories to save the selected images.

#### 1.4. select_size.py (optional)
Run this script to filter the train, validation and test data files to the allowed amount of pixels on the running device. The script creates another csv file that only contains the matches that both meet the allowed image size. The input: the matches file and the paths to the matched images directories. Note: this needs to be runned for all 3 types of data files (train, val, test).

### 2. Training - training the model to the optimal parameters
#### 2.1. training.py
Based on the training and validation data (captured in csv files) and the matched images, the model can now be trained to predict whether tissue sections represent the same specimin (match) or not (non-match). The trained model parameters are saved.

### 3. Evaluation - evaluating the model on the retrieval performance
#### 3.1. eval_ranking.py
The model is applied to all possible image pairs. For each query image, match probabilities were computed against all candidate images of the opposite staining modality. The candidates were then ranked in descending order based on predicted probabilities. Metrics are computed and saved, as well as all rankings. metrics include:
- Recall@k : k = 1,5,10. Indicating the percentage of queries in which the correct counterpart was found amoung the top k results
- Mean and median rank of the true match and IQR: to provide insight into overall retrieval depth.

#### 3.2. analyse_scores.py
Run this script to provide a more detailed analysis of the computed rankings. The script loads the predicted rankings and calculates the patient-level metrics, as well as the correct in-case retrieval.


## Contributors
Evi Paulides <br>
Bachelor End Project, IMAG/e
