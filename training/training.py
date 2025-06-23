import logging
import random
from datetime import datetime
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from external.DualBranchViT import DualBranchViT
from external.DualInputViT import DualInputViT
from external.dataset_utils import PairDataset
from external.ViT_utils import convert_state_dict

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

def seed_worker(worker_id):
    """
    Seed worker processes for reproducibility.
    """
    worker_seed = config.RANDOM_SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train_model(epochs, train_dataloader, val_dataloader, model, optimizer, device, gradient_accumulation_steps, pred_threshold, logger, scheduler=None):
    """
    Train the model for a specified number of epochs.

    Args:
        epochs (int): Number of epochs to train.
        train_dataloader (DataLoader): DataLoader for training data.
        val_dataloader (DataLoader): DataLoader for validation data.
        model (nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        device (torch.device): Device to run the model on.
        gradient_accumulation_steps (int): Number of steps to accumulate gradients.
        pred_threshold (float): Threshold for predictions.
        logger (logging.Logger): Logger for logging training information.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.

    Returns:
        loss_dict (dict): Dictionary containing training and validation losses, accuracies, and learning rates.
    """
    loss_dict = {
        'train_loss': [],
        'train_step': [],
        'val_loss': [],
        'val_step': [],
        'val_accuracy': [],
        'learning_rate': [],
    }
    for epoch in range(epochs):
        accummulated_loss = 0
        # Training step
        for step, (he_img, he_pos, ihc_img, ihc_pos, label) in enumerate(train_dataloader):
            
            if False: # Set to True to visualize images
                print(label)
                plt.imshow(he_img[0, ...].numpy().transpose((1,2,0)))
                plt.show()
                plt.imshow(ihc_img[0, ...].numpy().transpose((1,2,0)))
                plt.show()

            logit = model(
                he_img.to(device), 
                he_pos.to(device), 
                ihc_img.to(device), 
                ihc_pos.to(device),
                stain_embed = False,
                pos_embed = True,
            )
            loss = F.binary_cross_entropy_with_logits(logit[0], label.to(device))
            loss /= gradient_accumulation_steps
            accummulated_loss += loss.item()
                    
            # perform the backwards pass
            loss.backward()

            if (step+1) % gradient_accumulation_steps == 0:
                # update the network parameters and reset the gradient
                optimizer.step()
                optimizer.zero_grad() # set the gradient to 0 again
                logger.info(f'{epoch}, {step} - train loss: {accummulated_loss}')
                # store loss information
                loss_dict['train_loss'].append(accummulated_loss)
                loss_dict['train_step'].append(step+(epochs*len(train_dataset)))
                accummulated_loss = 0
        
        # Validation step
        model.eval()
        best_val_loss = None
        with torch.no_grad():
            accummulated_loss = 0
            correct = 0
            for step, (he_img, he_pos, ihc_img, ihc_pos, label) in enumerate(val_dataloader):
                
                if False:
                    print(label)
                    plt.imshow(he_img[0, ...].numpy().transpose((1,2,0)))
                    plt.show()
                    plt.imshow(ihc_img[0, ...].numpy().transpose((1,2,0)))
                    plt.show()

                logit = model(
                    he_img.to(device), 
                    he_pos.to(device), 
                    ihc_img.to(device), 
                    ihc_pos.to(device),
                    stain_embed = False,
                    pos_embed = True,
                )
                loss = F.binary_cross_entropy_with_logits(logit[0], label.to(device))
                accummulated_loss += loss.item()

                pred = F.sigmoid(logit)
                if (pred > pred_threshold).item() is bool(label.to('cpu')):
                    correct += 1

            # store loss information
            accummulated_loss /= len(val_dataloader)
            accuracy = correct/len(val_dataset)
            logger.info(f'{epoch} - val loss: {accummulated_loss}')
            logger.info(f'{epoch} - val accuracy: {accuracy}')
            logger.info(f'{epoch} - Learning Rate: {scheduler.optimizer.param_groups[0]['lr']:.2e}')
            loss_dict['val_loss'].append(accummulated_loss)
            loss_dict['val_step'].append((epochs+1)*len(train_dataset))
            loss_dict['val_accuracy'].append(accuracy)
            loss_dict['learning_rate'].append(scheduler.optimizer.param_groups[0]['lr'])
            # save model if validation loss is the lowest so far
            if best_val_loss is None or accummulated_loss < best_val_loss:
                best_val_loss = accummulated_loss
                torch.save(model.state_dict(), f'{start_time}_{model_architecture}_ep{epoch}.pth')

            if scheduler is not None:
                scheduler.step(accummulated_loss)
    return loss_dict

if __name__ == '__main__':

    # Set the random seed for reproducibility
    RANDOM_SEED = config.RANDOM_SEED
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(RANDOM_SEED)

    # define paths
    train_csv = config.train_csv
    val_csv = config.val_csv

    he_dir = config.he_dir
    he_mask_dir = config.he_mask_dir
    ihc_dir = config.ihc_dir
    ihc_mask_dir = config.ihc_mask_dir

    state_dict_path = config.state_dict_path

    device = config.device

    # define training settings
    epochs = config.epochs
    gradient_accumulation_steps = config.gradient_accumulation_steps
    learning_rate = config.learning_rate
    match_prob = config.match_prob
    pred_threshold = config.pred_threshold

    # define model settings
    model_architecture = config.model_architecture
    patch_shape = config.patch_shape
    input_dim = config.input_dim
    embed_dim = config.embed_dim
    n_classes = config.n_classes
    depth = config.depth
    n_heads = config.n_heads
    dropout_prob = config.dropout_prob
    mlp_ratio = config.mlp_ratio
    load_pretrained_param = config.load_pretrained_param

    # initialize model
    pretrained_state_dict = convert_state_dict(torch.load(state_dict_path))
    if model_architecture == "DualInputViT":
        model = DualInputViT(
            patch_shape = patch_shape, 
            input_dim = input_dim,
            embed_dim = embed_dim, 
            n_classes = n_classes,
            depth = depth,
            n_heads = n_heads,
            mlp_ratio = mlp_ratio,
            dropout_prob = dropout_prob,
            pytorch_attn_imp = False,
            init_values = 1e-5,
            act_layer = nn.GELU
        )
        if load_pretrained_param:
            model.load_state_dict(pretrained_state_dict, strict=False)
    elif model_architecture == "DualBranchViT":
        model = DualBranchViT(
            patch_shape = patch_shape, 
            input_dim = input_dim,
            embed_dim = embed_dim, 
            n_classes = n_classes,
            depth = depth,
            n_heads = n_heads,
            mlp_ratio = mlp_ratio,
            pytorch_attn_imp = False,
            init_values = 1e-5,
            act_layer = nn.GELU
        )
        if load_pretrained_param:
            model.he_encoder.load_state_dict(pretrained_state_dict, strict=False)
            model.ihc_encoder.load_state_dict(pretrained_state_dict, strict=False)

    model.to(device)
    model.train()

    # initialize optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=8, factor=0.5)

    # initilize datasets and dataloaders
    train_dataset = PairDataset(
        csv_path=train_csv,
        he_dir=he_dir,
        he_mask_dir=he_mask_dir,
        ihc_dir=ihc_dir,
        ihc_mask_dir=ihc_mask_dir,
        patch_size=16,
        match_prob=match_prob,
        transform=True,
    )
    train_dataloader = DataLoader(train_dataset, shuffle=True, worker_init_fn=seed_worker, generator=g)

    # initilize datasets and dataloaders
    val_dataset = PairDataset(
        csv_path=val_csv,
        he_dir=he_dir,
        he_mask_dir=he_mask_dir,
        ihc_dir=ihc_dir,
        ihc_mask_dir=ihc_mask_dir,
        patch_size=16,
        match_prob=match_prob,
        transform=False,
    )
    val_dataloader = DataLoader(val_dataset, shuffle=False, worker_init_fn=seed_worker, generator=g)

    # define logger
    start_time = datetime.now().strftime("%Y-%m-%d %H.%M.%S")
    logging.basicConfig(
        level=logging.INFO,
        filename=f'{start_time}_training_log.txt',
        format='%(asctime)s - %(message)s',
        datefmt='%d/%m/%Y %I:%M:%S %p',
        encoding='utf-8',
    )
    logger = logging.getLogger(__name__)

    print(f'Starting training with {start_time}_{model_architecture} model on {device} for {epochs} epochs - with match_prob {match_prob} and lr patience {scheduler.patience}')

    loss_dict = train_model(
        epochs=epochs, 
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader, 
        model=model, 
        optimizer=optimizer, 
        device=device, 
        gradient_accumulation_steps=gradient_accumulation_steps, 
        pred_threshold=pred_threshold, 
        logger=logger,
        scheduler=scheduler
    )

    # save loss information
    torch.save(loss_dict, f'{start_time}_loss.pth')