# ---------------------------------------------------------------------------
# Contains the code for Aesthetic Scorer V2
# Ref: https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/fe88a163f4661b4ddabba0751ff645e2e620746e/simple_inference.py
# ---------------------------------------------------------------------------

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

from PIL import Image
from pathlib import Path
# from importlib import resources
from transformers import CLIPModel

ASSETS_PATH = Path("../assets")

class MLPDiff(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )


    def forward(self, embed):
        return self.layers(embed)


class AestheticScorerDiff(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp = MLPDiff()
        state_dict = torch.load(ASSETS_PATH.joinpath("sac+logos+ava1-l14-linearMSE.pth"))
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.eval()

    def __call__(self, images):
        device = next(self.parameters()).device
        embed = self.clip.get_image_features(pixel_values=images)
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1)
    
class AestheticScorer(torch.nn.Module):
    
    def __init__(self,
                 aesthetic_target=None,
                 grad_scale=0,
                 device=None,
                 accelerator=None,
                 torch_dtype=None):
        super().__init__()

        self.grad_scale = grad_scale
        self.aesthetic_target = aesthetic_target

        self.target_size = 224
        self.normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                    std=[0.26862954, 0.26130258, 0.27577711])
        
        self.scorer = AestheticScorerDiff(dtype=torch_dtype).to(device, dtype=torch_dtype)
        self.scorer.requires_grad_(False)

    def score(self, im_pix_un):

        if isinstance(im_pix_un, Image.Image):
            im_pix_un = transforms.ToTensor()(im_pix_un)
            im_pix_un = im_pix_un.unsqueeze(0)

        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1) 
        im_pix = torchvision.transforms.Resize(self.target_size)(im_pix)
        im_pix = self.normalize(im_pix).to(im_pix_un.dtype)
        rewards = self.scorer(im_pix)

        return rewards

    def loss_fn(self, im_pix_un):
        rewards = self.score(im_pix_un)
        if self.aesthetic_target is None: # default maximization
            loss = -1 * rewards
        else:
            # using L1 to keep on same scale
            loss = abs(rewards - self.aesthetic_target)
        return loss
