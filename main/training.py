import logging
import random
from datetime import datetime

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

import config

if __name__ == '__main__':

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

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

    # initilize datasets and dataloaders
    train_dataset = PairDataset(
        csv_path=train_csv,
        he_dir=he_dir,
        he_mask_dir=he_mask_dir,
        ihc_dir=ihc_dir,
        ihc_mask_dir=ihc_mask_dir,
        patch_size=16,
        match_prob=match_prob,
        transform=False,
    )
    train_dataloader = DataLoader(train_dataset, shuffle=True)

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
    val_dataloader = DataLoader(val_dataset, shuffle=False)

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

    loss_dict = {
        'train_loss': [],
        'train_step': [],
        'val_loss': [],
        'val_step': [],
        'val_accuracy': [],
    }
    for epoch in range(epochs):
        accummulated_loss = 0
        for step, (he_img, he_pos, ihc_img, ihc_pos, label) in enumerate(train_dataloader):
            
            if True:
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
            loss_dict['val_loss'].append(accummulated_loss)
            loss_dict['val_step'].append((epochs+1)*len(train_dataset))
            loss_dict['val_accuracy'].append(accuracy)
            # save model if validation loss is the lowest so far
            if best_val_loss is None or accummulated_loss < best_val_loss:
                best_val_loss = accummulated_loss
                torch.save(model.state_dict(), f'{start_time}_{model_architecture}_ep{epoch}.pth')
    # save loss information
    torch.save(loss_dict, f'{start_time}_loss.pth')