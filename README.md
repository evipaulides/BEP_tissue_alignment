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

