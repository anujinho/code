# ---------------------------------------------------------------------------
# Contains the code for Human Preference Scorer V2
# ---------------------------------------------------------------------------

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

from PIL import Image
from pathlib import Path
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer


class HPSScorer(torch.nn.Module):

    def __init__(self,
                 device=None,
                 inference_dtype=None):
        super().__init__()
        
        model_name = "ViT-H-14"

        self.model, _, _ = create_model_and_transforms(
            model_name,
            'laion2B-s32B-b79K',
            precision=inference_dtype,
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )    
        
        self.device= device
        
        
        # checkpoint_path = f"{os.path.expanduser('~')}/.cache/huggingface/hub/models--xswu--HPSv2/snapshots/697403c78157020a1ae59d23f111aa58ced35b0a/HPS_v2_compressed.pt"
        checkpoint_path = Path("../assets/HPS_v2_compressed.pt")
        
        # force download of model via score
        # hpsv2.score([], "")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.tokenizer = get_tokenizer(model_name)
        self.model = self.model.to(self.device, dtype=inference_dtype)
        self.model.eval()

        self.target_size =  224
        self.normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                    std=[0.26862954, 0.26130258, 0.27577711])
        

    def score(self, im_pix, prompts):

        if isinstance(im_pix, Image.Image):
            im_pix = transforms.ToTensor()(im_pix)
            im_pix = im_pix.unsqueeze(0)

        x_var = torchvision.transforms.Resize(self.target_size)(im_pix)
        x_var = self.normalize(x_var).to(im_pix.dtype)        
        caption = self.tokenizer(prompts)
        caption = caption.to(self.device)
        outputs = self.model(x_var, caption)
        image_features, text_features = outputs["image_features"], outputs["text_features"]
        logits = image_features @ text_features.T
        scores = torch.diagonal(logits)

        return scores
    
    def loss_fn(self, im_pix, prompts):

        scores = self.score(im_pix, prompts)
        loss = 1.0 - scores

        return  loss
