import json
import copy
import torch
import torchvision

from pathlib import Path
from tqdm.auto import tqdm
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from typing import Union, Optional, List, Callable, Dict, Any
from scorers import HPSScorer, AestheticScorer, FaceRecognitionScorer, ClipScorer

class CoDeSDPipeline(StableDiffusionPipeline):
    @torch.no_grad()
    def __call__(
        self,
        offset: int = 5,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        num_try: Optional[int] = 1,
        guidance_scale: float = 7.5,
        n_samples: int = 5,
        block_size: int = 5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.
        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
        Examples:
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # Generate in batches
        assert n_samples % self.genbatch == 0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            n_samples * self.genbatch,
            # num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop for BoN 
        is_failed = True
        num_batch = num_images_per_prompt // self.genbatch
        
        for batch_iter in tqdm(range(num_batch), total=num_batch):

            curr_samples = latents[batch_iter * self.genbatch: (batch_iter + 1) * self.genbatch]
            curr_samples = curr_samples.repeat(n_samples, 1, 1, 1) # (n_samples, 4, 64, 64)
            
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            for i, t in enumerate(timesteps):

                # expand the latents if we are doing classifier free guidance
                # latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                
                latent_model_input = torch.cat([curr_samples] * 2) if do_classifier_free_guidance else curr_samples
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                # latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                curr_samples = self.scheduler.step(noise_pred, t, curr_samples, **extra_step_kwargs).prev_sample

                prev_timestep = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps

                if ((i + 1) % block_size == 0) or (t == timesteps[-1]): # at the end of block do BoN
                    
                    if t > timesteps[-1]: # If not final step use estimates x0
                        latent_model_input = torch.cat([curr_samples] * 2) if do_classifier_free_guidance else curr_samples
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                        # predict the noise residual
                        noise_pred = self.unet(
                            latent_model_input,
                            prev_timestep,
                            encoder_hidden_states=prompt_embeds,
                            cross_attention_kwargs=cross_attention_kwargs,
                        ).sample

                        # perform guidance
                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                        pred_original_temp = self.scheduler.step(noise_pred, prev_timestep, curr_samples, **extra_step_kwargs).pred_original_sample
                        rewards = self.compute_scores(pred_original_temp, prompt)
                    else:
                        rewards = self.compute_scores(curr_samples, prompt)
                    

                    rewards = torch.cat([x.unsqueeze(0) for x in rewards.chunk(n_samples)], dim=0) # (n_samples, self.genbatch)
                    select_ind = torch.max(rewards, dim=0)[1]

                    gen_sample = copy.deepcopy(curr_samples)
                    gen_sample = torch.cat([x.unsqueeze(0) for x in gen_sample.chunk(n_samples)], dim=0) # (n_samples, self.genbatch, 4, 64, 64)
                    gen_sample = gen_sample.permute(1,0,2,3,4)
                    curr_samples = torch.cat([x[select_ind[idx]].unsqueeze(0) for idx, x in enumerate(gen_sample)], dim=0) # TODO: Make it efficient

                    if t > timesteps[-1]: # If not the end replicate n times
                        curr_samples = curr_samples.repeat(n_samples, 1, 1, 1) # (n_samples, 4, 64, 64)

            try:
                self.save_outputs(curr_samples, start=(batch_iter*self.genbatch)+offset, prompt=prompt, num_try=num_try)
            except:
                is_failed = True
                continue
            
            store_rewards = []
            savepath = Path(self.path.joinpath(prompt)).joinpath("rewards.json")
            if Path.exists(savepath): # if exists append else create new
                with open(savepath, 'r') as fp:
                    store_rewards = json.load(fp)

            rewards = rewards.permute(1,0)
            rewards = torch.cat([x[select_ind[idx]].unsqueeze(0) for idx, x in enumerate(rewards)], dim=0) # TODO: Make it efficient
            store_rewards.extend(rewards.cpu().numpy().tolist())

            with open(savepath, 'w') as fp:
                json.dump(store_rewards, fp)

        return is_failed

    def set_genbatch(self, genbatch: int = 5):
        self.genbatch = genbatch

    def set_retry(self, retry: int = 0):
        self.retry = retry
    
    def set_project_path(self, path):
        self.path = path

    def setup_scorer(self, scorer):
        self.scorer = scorer

    def save_outputs(self, latent, prompt, start, num_try):
        decoded_latents = self.decode_latents(latent)
        image_pils = self.numpy_to_pil(decoded_latents)

        decoded_latents = torch.from_numpy(decoded_latents).permute(0,3,1,2)

        try:
            if isinstance(self.scorer, HPSScorer):
                prompts = [prompt] * len(decoded_latents)
                rewards = self.scorer.score(decoded_latents, prompts)
            elif isinstance(self.scorer, FaceRecognitionScorer)or\
                isinstance(self.scorer, ClipScorer):
                rewards = self.scorer.score(decoded_latents, self.target_img)
            else:
                rewards = self.scorer.score(decoded_latents)
        except:
            if num_try < self.retry:
                raise ValueError()
            else:
                rewards = torch.tensor([- torch.inf] * decoded_latents.shape[0])

        savepath = Path(self.path.joinpath(prompt))
        if not Path.exists(savepath):
            Path.mkdir(savepath, exist_ok=True, parents=True)

        for idx in range(len(image_pils)):
            image_pils[idx].save(savepath.joinpath(f"{start + idx}.png"))

        return rewards.detach().cpu().tolist()
    
    def set_target(self, target_img):
        if isinstance(self.scorer, ClipScorer):
            target_img = torchvision.transforms.ToTensor()(target_img)
            self.target_img = self.scorer.encode(target_img.unsqueeze(0))
        else:
            self.target_img = torchvision.transforms.ToTensor()(target_img)
        # print(self.target_img.shape)

    @torch.no_grad()
    def compute_scores(self, latent, prompt):
        decoded_latents = self.decode_latents(latent)
        decoded_latents = torch.from_numpy(decoded_latents).permute(0,3,1,2)

        if isinstance(self.scorer, HPSScorer):
            
            prompts = [prompt] * len(decoded_latents)
            out = self.scorer.score(decoded_latents, prompts)

        elif isinstance(self.scorer, FaceRecognitionScorer) or\
              isinstance(self.scorer, ClipScorer):
            
            out = self.scorer.score(decoded_latents, self.target_img)
        else:
            out = self.scorer.score(decoded_latents)

        return out
