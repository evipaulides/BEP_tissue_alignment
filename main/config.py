

# Config
EPOCHS = 3
BATCH_SIZE = 1
LR = 3e-5
ACCUMULATION_STEPS = 2
CHECKPOINT_DIR = "checkpoints"
RANDOM_SEED = 42
MODEL_NAME = "vit_base_patch16_224"
PATCH_SIZE = 16

# Prepared splits
train_csv = "data/data_split/train_matches.csv"
val_csv = "data/data_split/val_matches.csv"
test_csv = "data/data_split/test_matches.csv"

he_dir = "data/HE_images_matched"
ihc_dir = "data/IHC_images_matched"
