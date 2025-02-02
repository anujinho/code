
import os
import cv2
import PIL
import json
import torch
import random
import numpy as np
import torchvision
import argparse
import torch.nn as nn

from pathlib import Path
from PIL import Image
from omegaconf import OmegaConf
from torch.utils import data
from itertools import islice
from helper import OptimizerDetails
from torchvision import transforms, utils
from pytorch_lightning import seed_everything
from ldm.util import instantiate_from_config
from transformers import CLIPModel, AutoFeatureExtractor
from ldm.models.diffusion.ddim_with_grad import DDIMSamplerWithGrad
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

ASSETS_PATH = Path("../../BoN/assets/")

# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept

class Dataset(data.Dataset):
    def __init__(self, folder, image_size, data_aug=False, exts=['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        random.shuffle(self.paths)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = img.convert('RGB')
        img = img.resize((self.image_size, self.image_size), resample=PIL.Image.LANCZOS)

        return self.transform(img)

def return_cv2(img, path):
    black = [255, 255, 255]
    img = (img + 1) * 0.5
    utils.save_image(img, path, nrow=1)
    img = cv2.imread(path)
    img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
    return img

def cycle(dl):
    while True:
        for data in dl:
            yield data

import os
import errno
def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass


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
                 device=None,
                 accelerator=None,
                 torch_dtype=None):
        super().__init__()

        self.aesthetic_target = aesthetic_target

        self.target_size = 224
        self.normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                    std=[0.26862954, 0.26130258, 0.27577711])
        
        self.scorer = AestheticScorerDiff(dtype=torch_dtype).to(device, dtype=torch_dtype)

    def score(self, im_pix_un):

        if isinstance(im_pix_un, Image.Image):
            im_pix_un = transforms.ToTensor()(im_pix_un)
            im_pix_un = im_pix_un.unsqueeze(0)

        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1) 
        im_pix = torchvision.transforms.Resize(self.target_size)(im_pix)
        im_pix = self.normalize(im_pix).to(im_pix_un.dtype)
        rewards = self.scorer(im_pix)

        return rewards

    def forward(self, im_pix_un, y, return_score=False):
        
        if return_score:
            return self.score(im_pix_un)
        
        rewards = self.score(im_pix_un)
        if self.aesthetic_target is None: # default maximization
            loss = -1 * rewards
        else:
            # using L1 to keep on same scale
            loss = abs(rewards - self.aesthetic_target)
        return loss

def get_optimation_details(args):

    l_func = AestheticScorer(device='cuda')
    l_func.eval()
    for param in l_func.parameters():
        param.requires_grad = False
    l_func = torch.nn.DataParallel(l_func)

    # guidance_func = AestheticScorer(aesthetic_target = 10, device='cuda')
    operation = OptimizerDetails()

    operation.num_steps = args.optim_num_steps
    operation.operation_func = None

    operation.optimizer = 'Adam'
    operation.lr = args.optim_lr
    operation.loss_func = l_func

    operation.max_iters = args.optim_max_iters
    operation.loss_cutoff = args.optim_loss_cutoff

    operation.guidance_3 = args.optim_forward_guidance
    operation.guidance_2 = args.optim_backward_guidance

    operation.optim_guidance_3_wt = args.optim_forward_guidance_wt
    operation.original_guidance = args.optim_original_conditioning

    operation.warm_start = args.optim_warm_start
    operation.print = args.optim_print
    operation.print_every = 5
    operation.folder = args.optim_folder

    return operation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )


    parser.add_argument("--optim_lr", default=1e-2, type=float)
    parser.add_argument('--optim_max_iters', type=int, default=1)
    parser.add_argument('--optim_mask_type', type=int, default=1)
    parser.add_argument("--optim_loss_cutoff", default=0.00001, type=float)
    parser.add_argument('--optim_forward_guidance', action='store_true', default=False)
    parser.add_argument('--optim_backward_guidance', action='store_true', default=False)
    parser.add_argument('--optim_original_conditioning', action='store_true', default=False)
    parser.add_argument("--optim_forward_guidance_wt", default=5.0, type=float)
    parser.add_argument("--optim_tv_loss", default=None, type=float)
    parser.add_argument('--optim_warm_start', action='store_true', default=False)
    parser.add_argument('--optim_print', action='store_true', default=False)
    parser.add_argument('--optim_folder', default='./temp/')
    parser.add_argument("--optim_num_steps", nargs="+", default=[1], type=int)
    parser.add_argument('--text_type', type=int, default=1)
    parser.add_argument("--text", default=None)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument('--face_folder', default='./data/face_data')

    parser.add_argument('--fr_crop', action='store_true')
    parser.add_argument('--center_face', action='store_true')
    parser.add_argument("--trials", default=50, type=int)
    parser.add_argument("--indexes", nargs="+", default=[0, 1, 2], type=int)

    opt = parser.parse_args()
    results_folder = opt.optim_folder
    create_folder(results_folder)

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.eval()


    sampler = DDIMSamplerWithGrad(model)
    operation = get_optimation_details(opt)

    image_size = 256
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])

    batch_size = opt.batch_size

    torch.set_grad_enabled(False)

    # Load prompts
    with open(ASSETS_PATH.joinpath('eval_simple_animals.txt'), 'r') as fp:
        prompts = [line.strip() for line in fp.readlines()]

    # prompts = [opt.text]

    for prompt in prompts:

        print(prompt)

        # Run the pipeline

        offset = 0
        num_images_per_prompt = opt.trials

        savepath = Path(results_folder).joinpath(prompt)
        if Path.exists(savepath):

            images = [x for x in savepath.iterdir() if x.suffix == '.png']
            num_gen_images = len(images)
            if num_gen_images == num_images_per_prompt:
                print(f'Images found. Skipping prompt.')
                continue

            elif num_gen_images < num_images_per_prompt:
                offset = num_gen_images
                num_images_per_prompt -= num_gen_images
                print(f'Found {num_gen_images} images. Generating {num_images_per_prompt} more.')


        if not Path.exists(savepath):
            Path.mkdir(savepath, exist_ok=True, parents=True)

        torch.cuda.empty_cache()

        # rewards = []

        uc = None
        if opt.scale != 1.0:
            uc = model.module.get_learned_conditioning(batch_size * [""])
        c = model.module.get_learned_conditioning(batch_size * [prompt])
        for multiple_tries in range(num_images_per_prompt):
            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
            samples_ddim, start_zt = sampler.sample(S=opt.ddim_steps,
                                                conditioning=c,
                                                batch_size=batch_size,
                                                shape=shape,
                                                verbose=False,
                                                unconditional_guidance_scale=opt.scale,
                                                unconditional_conditioning=uc,
                                                eta=opt.ddim_eta,
                                                operated_image=None,
                                                operation=operation)

            x_samples_ddim = model.module.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

            reward = operation.loss_func(x_samples_ddim, None, return_score=True)

            # rewards.extend(reward.detach().cpu().tolist())

            utils.save_image(x_samples_ddim, f'{savepath}/new_img_{multiple_tries + offset}.png')

            if Path.exists(savepath.joinpath("rewards.json")): # append rewards to file

                rewards = None
                with open(savepath.joinpath("rewards.json"), 'r') as fp:
                    rewards = json.load(fp)
                
                rewards.extend(reward.detach().cpu().tolist())

                with open(savepath.joinpath("rewards.json"), 'w') as fp:
                    json.dump(rewards, fp)

            else: # create new rewards file

                rewards = []
                rewards.extend(reward.detach().cpu().tolist())

                with open(savepath.joinpath("rewards.json"), 'w') as fp:
                    json.dump(rewards, fp)


if __name__ == "__main__":
    main()