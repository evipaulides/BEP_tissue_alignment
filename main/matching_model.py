import torch
import torch.nn as nn
import timm
from external.ViT import ViT, convert_state_dict
from typing import Callable




class MatchingModel(nn.Module):
    def __init__(self, 
                 model_path,
                 patch_shape = 16, 
                 input_dim = 3,
                 embed_dim = 256, 
                 n_classes = None,
                 depth = 14,
                 n_heads = 4,
                 mlp_ratio = 5,
                 pytorch_attn_imp = False,
                 init_values = 1e-5,
                 act_layer: Callable = nn.GELU):
        super().__init__()
#ViT param initatiliseren

        self.encoder_he = ViT(patch_shape=patch_shape,
                              input_dim=input_dim,
                              embed_dim=embed_dim, 
                              n_classes=n_classes,
                              depth=depth,
                              n_heads=n_heads,
                              mlp_ratio=mlp_ratio,
                              pytorch_attn_imp=pytorch_attn_imp,
                              init_values=init_values,
                              act_layer=act_layer)
        self.encoder_ihc = ViT(patch_shape=patch_shape,
                               input_dim=input_dim,
                              embed_dim=embed_dim, 
                              n_classes=n_classes,
                              depth=depth,
                              n_heads=n_heads,
                              mlp_ratio=mlp_ratio,
                              pytorch_attn_imp=pytorch_attn_imp,
                              init_values=init_values,
                              act_layer=act_layer)

        self.custom_state_dict(path=model_path)
        
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            act_layer(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, x1, pos1, x2, pos2):
        # Encode HE image
        z1 = self.encoder_he(x1, pos1)

        # Encode IHC image
        z2 = self.encoder_ihc(x2, pos2)

        # z1 and z2 are of shape (B, D)
        x = torch.cat([z1, z2], dim=1)  # shape (B, 2D)
        return self.net(x)              # shape (B, 1), match probability
    
    def custom_state_dict(self, path):
        state_dict = torch.load(path)
        converted_state_dict = convert_state_dict(state_dict=state_dict)
        converted_state_dict['pos_embedder.pos_embed'] = self.encoder_he.state_dict()['pos_embedder.pos_embed']
        self.encoder_he.load_state_dict(converted_state_dict)
        self.encoder_ihc.load_state_dict(converted_state_dict)

