# Bachelor End Project - Matching Multi-stain tissue sections of skincancer WSIs
A Dual-Image Vision Transformer (VIT) adapted to make a match prediction for image pairs with varying input sizes.

This repository contains all code to support the paper: ***"Deep Learning for Multi-Stain Tissue Section Matching in Whole Slide Images"***

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

