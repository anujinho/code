
# import clip
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF\

from clip import clip
from PIL import Image

class ClipScorer(nn.Module):
    def __init__(self):
        super(ClipScorer, self).__init__()

        clip_model, clip_preprocess = clip.load("RN50")
        # clip_model, clip_preprocess = clip.load('ViT-L/14')
        print(clip_preprocess)
        clip_model.eval()
        for param in clip_model.parameters():
            param.requires_grad = False

        self.model = clip_model.to('cuda')
        self.trans = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x, y):
        x = (x + 1) * 0.5
        x = TF.resize(x, (224, 224), interpolation=TF.InterpolationMode.BICUBIC)
        x = self.trans(x)
        x = x.to('cuda')

        image_features = self.model.encode_image(x)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        logits_per_image = 100 * image_features @ y.t()
        return -1 * logits_per_image

    def encode(self, x):
        x = (x + 1) * 0.5
        x = TF.resize(x, (224, 224), interpolation=TF.InterpolationMode.BICUBIC)
        x = self.trans(x)
        x = x.to('cuda')

        image_features = self.model.encode_image(x)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        return image_features
    
    def score(self, im_pix, target):

        if isinstance(im_pix, Image.Image):
            im_pix = transforms.ToTensor()(im_pix)
            im_pix = im_pix.unsqueeze(0)

        curr_target = target.repeat(im_pix.shape[0], 1)

        return - (self(im_pix, target)).squeeze(1) 
    
    def loss_fn(self, im_pix, target):

        if isinstance(im_pix, Image.Image):
            im_pix = transforms.ToTensor()(im_pix)
            im_pix = im_pix.unsqueeze(0)

        curr_target = target.repeat(im_pix.shape[0], 1)

        return self(im_pix, target)
