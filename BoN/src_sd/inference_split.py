# ---------------------------------------------------------------------------
# Contains the code to run inference with specified guidance method
# ---------------------------------------------------------------------------

import os
import logging
import warnings
import random
import torch
import copy
import numpy as np

from PIL import Image
from tqdm.auto import tqdm
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from pipelines import (
    BoNSDPipeline, 
    IBoNSDPipeline, 
    IBoNSDPipelineI2I, 
    SDPipelineI2I, 
    UncondSDPipeline, 
    GradSDPipeline, 
    BoNSDPipelineI2I,
    GradSDPipelineI2I,
    CoDeSDPipeline,
    CoDeSDPipelineI2I,
    prepare_image, 
    encode
)
from scorers import (
    AestheticScorer, 
    HPSScorer, 
    FaceRecognitionScorer, 
    ClipScorer,
    CompressibilityScorer
)

from typing import Optional
from argparse import ArgumentParser, Namespace
from diffusers import DDIMScheduler, DDPMScheduler
from diffusers.utils.torch_utils import randn_tensor

logger = logging.getLogger("guided-diff")

NUM_RETRY = 3

def get_parser() -> ArgumentParser:
    """Get parser.

    Returns:
        ArgumentParser: The parser object.
    """
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=False,
                        help="Path to a model config file")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="<DEBUG, INFO, WARNING, ERROR>")

    return parser


def run_experiment(config):

    # Set device
    device = (
        "cuda"
        if torch.cuda.is_available() and config.project.accelerator in ['cuda', 'auto']
        else "cpu"
    )

    logger.info('Using accelerator %s.' % (device))

    # Set pipeline
    model_id = config.guidance.basemodel
    if config.guidance.method == "bon":
        pipe = BoNSDPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16).to(device)
    elif config.guidance.method == "c_bon":
        pipe = BoNSDPipelineI2I.from_pretrained(
            model_id, torch_dtype=torch.float16).to(device)
    elif config.guidance.method == "ibon_i2i":
        pipe = IBoNSDPipelineI2I.from_pretrained(
            model_id, torch_dtype=torch.float16).to(device)
    elif config.guidance.method == "c_code" or config.guidance.method == "c_code_b1":
        pipe = CoDeSDPipelineI2I.from_pretrained(
            model_id, torch_dtype=torch.float16).to(device)
    elif config.guidance.method == "i2i":
        pipe = SDPipelineI2I.from_pretrained(
            model_id, torch_dtype=torch.float16).to(device)
    elif config.guidance.method == 'uncond':
        pipe = UncondSDPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16).to(device)
    elif config.guidance.method == 'grad':
        pipe = GradSDPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16).to(device)
        pipe.set_guidance(config.guidance.guidance_scale)
    elif config.guidance.method == 'grad_i2i':
        pipe = GradSDPipelineI2I.from_pretrained(
            model_id, torch_dtype=torch.float16).to(device)
        pipe.set_guidance(config.guidance.guidance_scale)
    elif config.guidance.method == "ibon":
        pipe = IBoNSDPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16).to(device)
    elif config.guidance.method == "code" or config.guidance.method == "code_b1":
        pipe = CoDeSDPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16).to(device)        
    else:
        raise NotImplementedError(f'{config.guidance.method} pipeline not found!')

    # freeze parameters of models to save more memory
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    pipe.set_genbatch(config.guidance.genbatch)
    pipe.set_retry(NUM_RETRY)

    logger.info(f'Loaded {pipe}')

    # Change to DDPM scheduler
    pipe.scheduler = DDPMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing")

    # Set scorer
    if config.guidance.scorer == "aesthetic":
        scorer = AestheticScorer()
    elif config.guidance.scorer == "compress":
        scorer = CompressibilityScorer()
    elif config.guidance.scorer == "facedetector":
        scorer = FaceRecognitionScorer()
    elif config.guidance.scorer in ["styletransfer", "strokegen"]:
        scorer = ClipScorer()
    else:
        scorer = HPSScorer()

    pipe.setup_scorer(scorer)

    logger.info('Loaded Scorer Model')

    # Set project path
    savepath = Path(config.project.path).joinpath(config.project.name)

    # Initial noise samples
    num_images_per_prompt = config.guidance.num_images_per_prompt
    num_channels_latents = pipe.unet.config.in_channels
    generator = torch.Generator(device=device).manual_seed(config.project.seed)

    # Load prompts
    with open(Path(config.project.promptspath), 'r') as fp:
        prompts = [line.strip() for line in fp.readlines()]

    # Select n prompts
    prompts = [prompts[idx]  for idx in config.guidance.prompt_idxs]
    logger.info(prompts)

    if isinstance(scorer, FaceRecognitionScorer):
        # target_dir = [x for x in Path(
        #     '../assets/face_data/celeb').iterdir() if x.is_file()]
        
        target_map = {
            "0": '../assets/face_data/celeb/og_img_4.png',
            "1": '../assets/face_data/celeb/og_img_6.png',
            "2": '../assets/face_data/celeb/og_img_8.png',
            # "3": '../assets/face_data/celeb/og_img_9.png',
            # "4": '../assets/face_data/celeb/og_img_10.png',
        }
    

    if isinstance(scorer, ClipScorer) and config.guidance.scorer != 'strokegen':
        # target_dir = [x for x in Path(
        #     '../assets/style_folder/styles').iterdir() if x.is_file()]
        
        target_map = {
            "0": '../assets/style_folder/styles/style_0.jpg',
            "1": '../assets/style_folder/styles/style_1.jpg',
            "2": '../assets/style_folder/styles/style_2.jpg',
            # "3": '../assets/style_folder/styles/style_3.jpg',
            # "4": '../assets/style_folder/styles/style_4.jpg',
        }
        

    elif isinstance(scorer, ClipScorer) and config.guidance.scorer == 'strokegen':
        # target_dir = [x for x in Path(
        #     '../assets/style_folder/styles').iterdir() if x.is_file()]
        
        target_map = {
            "0": '../assets/stroke_gen/stroke_img_0.jpg',
            "1": '../assets/stroke_gen/stroke_img_1.jpg',
            # "2": '../assets/stroke_gen/stroke_img_2.jpg',
            "2": '../assets/stroke_gen/stroke_img_3.jpg',
            # "4": '../assets/stroke_gen/stroke_img_4.jpg',
        }

    elif isinstance(scorer, CompressibilityScorer):
        # target_dir = [x for x in Path(
        #     '../assets/style_folder/styles').iterdir() if x.is_file()]
        
        target_map = {
            "0": '../assets/compressibility/compress/resized/img_0.jpg',
            "1": '../assets/compressibility/compress/resized/img_1.jpg',
            "2": '../assets/compressibility/compress/resized/img_2.jpg',
        }
    # breakpoint()  
    # Select n targets
    if 'target_idxs' in config.guidance.keys():
        target_dir = [Path(target_map[str(idx)])  for idx in config.guidance.target_idxs]
        logger.info(target_dir)
    

    if isinstance(pipe, CoDeSDPipelineI2I) or isinstance(pipe, SDPipelineI2I)\
        or isinstance(pipe, BoNSDPipelineI2I) or isinstance(pipe, GradSDPipelineI2I):

        # if config.input_image:
        # if (isinstance(scorer, FaceRecognitionScorer) or isinstance(scorer, ClipScorer)) or isinstance(scorer, CompressibilityScorer):
        #     # If target image is used for init_noise conditioning:

            percent_noise = config.guidance.percent_noise
            timestep = torch.Tensor(
                [int(pipe.scheduler.config.num_train_timesteps * percent_noise)]).to(pipe.device).long()

            for idx, target_img in enumerate(target_dir):

                loaded_img = Image.open(fp=target_img)
                pipe.set_target(loaded_img)

                curr_path = savepath.joinpath(
                    target_img.stem).joinpath('images')
                if not Path.exists(curr_path):
                    Path.mkdir(curr_path, exist_ok=True, parents=True)

                pipe.set_project_path(curr_path)

                loaded_img.save(curr_path.joinpath('target.png'))

                for prompt in prompts:

                    ######## error handling starts >>>>>>>>>>>>>>>>>>>>>>>>>>>
                    seed_everything(config.project.seed)
                    is_failed = True
                    counter = 0
                    while is_failed:

                        if counter > 0:
                            seed_everything(config.project.seed + (1000 * counter))
                            logger.info(f'Retrying Prompt {prompt}')

                        else:   
                            logger.info(f'Prompt {prompt}')

                        is_failed = False
                        counter += 1

                        ######## error handling ends <<<<<<<<<<<<<<<<<<<<<<<<<<<<<

                        # Resume generation
                        offset = 0  # start numbering of images
                        num_images_curr_prompt = num_images_per_prompt  # samples for current prompt

                        prompt_path = curr_path.joinpath(prompt)
                        if Path.exists(prompt_path):
                            images = [x for x in prompt_path.iterdir()
                                    if x.suffix == '.png']
                            num_gen_images = len(images)
                            if num_gen_images >= num_images_curr_prompt:
                                logger.info(f'Images found. Skipping prompt.')
                                continue

                            elif num_gen_images < num_images_curr_prompt:
                                offset = num_gen_images
                                num_images_curr_prompt -= num_gen_images
                                logger.info(f'Found {num_gen_images} images. Generating {num_images_curr_prompt} more.')

                        torch.cuda.empty_cache()

                        curr_target = copy.deepcopy(loaded_img)
                        curr_target = encode(curr_target, pipe.vae)

                        noise = torch.randn(curr_target.shape).to(pipe.device)
                        if percent_noise > 0.999:
                            curr_target = noise.half()
                        else:
                            curr_target = pipe.scheduler.add_noise(
                                original_samples=curr_target, noise=noise, timesteps=timestep).half()

                        curr_target = curr_target.repeat(
                            num_images_curr_prompt, 1, 1, 1)

                        # Run the pipeline
                        is_failed = pipe(
                            offset=offset,
                            percent_noise=percent_noise,
                            prompt=prompt,  # What to generate
                            n_samples=config.guidance.num_samples,
                            block_size=config.guidance.block_size,
                            latents=curr_target,
                            num_images_per_prompt=num_images_curr_prompt,
                            num_inference_steps=config.guidance.num_inference_steps,
                            num_try=counter
                        )

    else:

            curr_path = savepath.joinpath('images')
            if not Path.exists(curr_path):
                Path.mkdir(curr_path, exist_ok=True, parents=True)

            pipe.set_project_path(curr_path)

            for prompt in prompts:

                ######## error handling starts >>>>>>>>>>>>>>>>>>>>>>>>>>>

                seed_everything(config.project.seed)
                                
                is_failed = True
                counter = 0
                while is_failed:

                    if counter > 0:
                        seed_everything(config.project.seed + (1000 * counter))
                        logger.info(f'Retrying Prompt {prompt}')

                    else:   
                        logger.info(f'Prompt {prompt}')

                    is_failed = False
                    counter += 1

                    ######## error handling ends <<<<<<<<<<<<<<<<<<<<<<<<<<<<<

                    # Resume generation
                    offset = 0  # start numbering of images
                    num_images_curr_prompt = num_images_per_prompt  # samples for current prompt

                    prompt_path = curr_path.joinpath(prompt)
                    if Path.exists(prompt_path):
                        images = [x for x in prompt_path.iterdir()
                                if x.suffix == '.png']
                        num_gen_images = len(images)
                        if num_gen_images >= num_images_curr_prompt:
                            logger.info(f'Images found. Skipping prompt.')
                            continue

                        elif num_gen_images < num_images_curr_prompt:
                            offset = num_gen_images
                            num_images_curr_prompt -= num_gen_images
                            logger.info(f'Found {num_gen_images} images. Generating {num_images_curr_prompt} more.')

                    torch.cuda.empty_cache()

                    shape = (num_images_curr_prompt, num_channels_latents,
                            pipe.unet.config.sample_size, pipe.unet.config.sample_size)
                    init_noise = randn_tensor(
                        shape, generator=generator, device=pipe._execution_device, dtype=pipe.text_encoder.dtype)

                    # Run the pipeline
                    is_failed = pipe(
                        offset=offset,
                        prompt=prompt,  # What to generate
                        n_samples=config.guidance.num_samples,
                        block_size=config.guidance.block_size,
                        latents=init_noise,
                        num_images_per_prompt=num_images_curr_prompt,
                        num_inference_steps=config.guidance.num_inference_steps,
                        num_try=counter
                    )



def _select_seed_randomly(min_seed_value: int = 0, max_seed_value: int = 255) -> int:
    return random.randint(min_seed_value, max_seed_value)

def seed_everything(seed: Optional[int] = None):
    """
    Function that sets seed for pseudo-random number generators in:
    pytorch, numpy, python.random
    In addition, sets the following environment variables:

    - `PL_GLOBAL_SEED`: will be passed to spawned subprocesses (e.g. ddp_spawn backend).

    Args:
        seed: the integer value seed for global random state in Lightning.
            If `None`, will read seed from `PL_GLOBAL_SEED` env variable
            or select it randomly.
    """
    if seed is None:
        seed = _select_seed_randomly()

    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    logger.info(f'Random seed {seed} has been set.')

    return seed

def main(args: Namespace):

    # def main(config):

    config = OmegaConf.load(args.config)

    print(config.project.name)

    if args.log_level == "ERROR":
        warnings.filterwarnings("ignore")

    # Set up logger
    log_path = Path(config.project.path).joinpath(config.project.name)

    if not Path.exists(log_path):
        Path.mkdir(log_path, exist_ok=True, parents=True)

    if 'target_idxs' in config.keys():
        log_path = log_path.joinpath(f'log_t{config.guidance.target_idxs[0]}_p{config.guidance.prompt_idxs[0]}.txt')
    else:
        log_path = log_path.joinpath(f'log_p{config.guidance.prompt_idxs[0]}.txt')

    logging.basicConfig(level=logging.INFO,
                        filemode='w',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filename=log_path)
    logger = logging.getLogger()

    logger.info('Log file is %s.' % (log_path))

    # Set seed for reproducibility
    seed = config.project.seed
    seed_everything(seed)
    
    # os.environ["PL_GLOBAL_SEED"] = str(seed)
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

    run_experiment(config=config)

    handlers = logger.handlers[:]
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()



if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
