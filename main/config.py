import torch

# define paths
train_csv = "data/data_split/train_filtered2.csv"
val_csv = "data/data_split/val_filtered2.csv"
test_csv = "data/data_split/test_matches.csv"

he_dir = "data/HE_images_matched"
he_mask_dir = "data/HE_masks_matched"
ihc_dir = "data/IHC_images_matched"
ihc_mask_dir = "data/IHC_masks_matched"

state_dict_path = "main/external/vit_wee_patch16_reg1_gap_256.sbb_in1k.pth"
saved_model_path = "checkpoints/2025-06-03 13.32.48_DualInputViT_ep2.pth"

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# define training settings
epochs = 50
gradient_accumulation_steps = 128
learning_rate = 3e-5
match_prob = 0.5
pred_threshold = 0.5

# define model settings
model_architecture = "DualInputViT" # "DualBranchViT"
patch_shape = 16
input_dim = 3
embed_dim = 256
n_classes = 1
depth = 14
n_heads = 4
mlp_ratio = 5
load_pretrained_param = True

# define outputs
prediction_dir = "predictions"