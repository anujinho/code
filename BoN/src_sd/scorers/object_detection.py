# ---------------------------------------------------------------------------
# Contains the code for Object Detection
# Ref: https://github.com/arpitbansal297/Universal-Guided-Diffusion/blob/main/stable-diffusion-guided/scripts/object_detection.py
# ---------------------------------------------------------------------------

import torch
import torchvision.transforms as transforms

from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from PIL import Image
    
class ObjectDetectionScorer(torch.nn.Module):
    
    def __init__(self,
                 device=None,
                 accelerator=None,
                 torch_dtype=None):
        super().__init__()

        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.scorer = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
        self.preprocess = weights.transforms()
        for param in self.scorer.parameters():
            param.requires_grad = False
        self.categories = weights.meta["categories"]

    def score(self, im_pix_un):

        if isinstance(im_pix_un, Image.Image):
            im_pix_un = transforms.ToTensor()(im_pix_un)
            im_pix_un = im_pix_un.unsqueeze(0)
    
        self.scorer.eval()
        inter = self.preprocess((im_pix_un + 1) * 0.5)
        return self.scorer(inter)
    
    def loss_fn(self, im_pix_un, gt):

        def set_bn_to_eval(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()

        self.scorer.train()
        self.scorer.backbone.eval()
        self.scorer.apply(set_bn_to_eval)
        inter = self.preprocess((im_pix_un + 1) * 0.5)
        loss = self.scorer(inter, gt)
        return loss['loss_classifier'] + loss['loss_objectness'] + loss['loss_rpn_box_reg']
