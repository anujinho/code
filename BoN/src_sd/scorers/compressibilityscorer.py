# ---------------------------------------------------------------------------
# Contains the code for Compressibility Measurement
# ---------------------------------------------------------------------------

# import clip
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF\

from clip import clip
from PIL import Image
import io

class CompressibilityScorer(nn.Module):
    def __init__(self):
        super(CompressibilityScorer, self).__init__()

    def score(self, images):
        
        if isinstance(images, Image.Image):
            images = transforms.ToTensor()(images)
            images = images.unsqueeze(0)
        
        images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers] # Size in kilobytes

        
        return torch.tensor([-1*s for s in sizes])

    
    def loss_fn(self, images):

        return - self.score(images)
