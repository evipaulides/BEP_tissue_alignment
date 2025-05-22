import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import config


class PatchEmbedder(nn.Module):
    def __init__(self, patch_size=16, in_channels=3, embed_dim=768, dropout=0.1):
        super(PatchEmbedder, self).__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.dropout = dropout

        # Convolutional layer to create patches
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x)  # Shape: (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2, 3).transpose(1, 2) # Shape: (B, num_patches, embed_dim)

        return x

class PositionalEmbedder(nn.Module):
    max_position_index = 100
    repeat_section_embedding = True

    def __init__(self, embed_dim: int, dropout_prob: float):
        super().__init__()
        # initialize instance attributes
        self.embed_dim = embed_dim
        self.pos_embed = nn.Parameter(
            data=torch.zeros((self.max_position_index+1, self.embed_dim//2)), requires_grad=False) # (max_position_index+1, embed_dim//2)
        X = torch.arange(self.max_position_index+1, dtype=torch.float32).reshape(-1, 1) # (max_position_index+1, 1)
        X = X / torch.pow(10000, torch.arange(0, self.embed_dim//2, 2, dtype=torch.float32) / (self.embed_dim//2))
        self.pos_embed[:, 0::2] = torch.sin(X)
        self.pos_embed[:, 1::2] = torch.cos(X)

        # initialize dropout layer
        self.pos_drop = nn.Dropout(p=dropout_prob)
        
    def forward(self, x: torch.Tensor, pos: torch.Tensor): # Through the model
        pos = torch.round(pos).to(int)
        if torch.max(pos[:, :, 1:]) > self.max_position_index:
            raise ValueError(
                'Maximum requested position index exceeds the prepared position indices.'
            )
        # get the number of items in the batch and the number of tokens in the sequence
        B, S, _ = pos.shape
        device = self.pos_embed.get_device()
        if device == -1:
            device = 'cpu'

        # define embeddings for x and y dimension
        embeddings = [self.pos_embed[pos[:, :, 0], :],
                      self.pos_embed[pos[:, :, 1], :]]
        # add a row of zeros as padding in case the embedding dimension has an odd length
        if self.embed_dim % 2 == 1:
            embeddings.append(torch.zeros((B, S, 1), device=device))

        # prepare positional embedding
        pos_embedding = torch.concat(embeddings, dim=-1)

        # account for [CLS] token
        pos_embedding = torch.concatenate(
            [torch.zeros((B, 1, self.embed_dim), device=device), pos_embedding], dim=1,
        )
        
        # plt.imshow(pos_embedding[0, ...])
        # plt.show()

        # check if the shape of the features and positional embeddings match
        if x.shape != pos_embedding.shape:
            raise ValueError(
                'Shape of features and positional embedding tensors do not match.',
            )
        # add the combined embedding to each element in the sequence
        x = self.pos_drop(x+pos_embedding)
        
        return x
    
class MLPHead(nn.Module):
    def __init__(self, in_dim=768, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim//2),
            nn.ReLU(),
            nn.Linear(in_dim//2, out_dim)
        ) # Dropout nog toevoegen?

    def forward(self, x):
        return self.net(x)
    
class MultiStainContrastiveModel(nn.Module):
    def __init__(self, MODEL_NAME, patch_size=16, dropout_prob=0.1):	
        super().__init__()
        self.patch_size = patch_size
        self.dropout_prob = dropout_prob

        # Load the pre-trained model
        vit = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
        self.embed_dim = vit.embed_dim

        # Define the patch embedding layer
        self.patch_embed = PatchEmbedder(patch_size=patch_size, in_channels=3, embed_dim=self.embed_dim, dropout=dropout_prob)

        # Define the positional embedding layer
        self.pos_embed = PositionalEmbedder(embed_dim=self.embed_dim, dropout_prob=dropout_prob)

        # Reuse the ViT blocks and normalization layer
        self.blocks = vit.blocks
        self.norm = vit.norm

        # initialize cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        # Define the MLP head
        self.mlp_head = MLPHead(in_dim=self.embed_dim, out_dim=256)

    def forward(self, x: torch.Tensor, pos: torch.Tensor):
        # Patch embedding
        x = self.patch_embed(x)

        # create and add the [CLS] token to the sequence of embeddings 
        # (one for each item in the batch)
        # [1, 1, D] -> [B, 1, D]
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        # [B, S, D] -> [B, 1+S, D]
        x = torch.cat((cls_tokens, x), dim=1)

        # Positional embedding
        x = self.pos_embed(x, pos)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Normalization
        x = self.norm(x)

        # MLP head
        x = x[:, 0]  # Select the CLS token
        output = self.mlp_head(x)  # Ik weet niet of dit nog aangepast moet worden naar mijn probleem

        return output