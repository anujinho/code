import os
import copy
import numpy as np

from pathlib import Path
from omegaconf import OmegaConf

_METHODS = ['code_b1', 'bon', 'code'] # 'ibon', ibon_i2i', 'bon', 'uncond', 'i2i', 'bon_i2i', 'code', 'c_code'

_SCORERS = {
    # 'aesthetic': '../assets/eval_simple_animals.txt', 
    # 'hpsv2': '../assets/hps_v2_all_eval.txt', 
    # 'facedetector': '../assets/face.txt', 
    # 'styletransfer': '../assets/style.txt',
    'compress': '../assets/compressibility.txt',
    }

def create_function():

    template = OmegaConf.load('template.yaml')
    config_dir = Path('.')

    for method in _METHODS:

        curr_path = config_dir.joinpath(method)

        if not Path.exists(curr_path):
            Path.mkdir(curr_path, parents=True)

        for scorer in _SCORERS.keys():

            print(f'{method} {scorer}')

            if method == 'uncond':

                curr_config = copy.deepcopy(template)
                curr_config.project.name = f'{method}_{scorer}'
                curr_config.project.promptspath = _SCORERS[scorer]

                curr_config.guidance.method = method
                curr_config.guidance.scorer = scorer

                savepath = curr_path.joinpath(f'{curr_config.project.name}.yaml')
                OmegaConf.save(curr_config, savepath)

            elif method == 'code_b1':

                for num_samples in [40]: # [25, 50, 100, 200, 500]:

                    for block_size in [1]: # [5, 10, 20, 50, 100]

                        curr_config = copy.deepcopy(template)
                        m = method.split('_')[0]
                        curr_config.project.name = f'{m}{num_samples}_b{block_size}_{scorer}'
                        curr_config.project.promptspath = _SCORERS[scorer]

                        curr_config.guidance.method = method
                        curr_config.guidance.scorer = scorer
                        curr_config.guidance.num_samples = num_samples
                        curr_config.guidance.block_size = block_size

                        savepath = curr_path.joinpath(f'{curr_config.project.name}.yaml')
                        OmegaConf.save(curr_config, savepath)

            elif method in ['ibon', 'code']:

                for num_samples in [100]: # [25, 50, 100, 200, 500]:

                    for block_size in [5]:

                        curr_config = copy.deepcopy(template)
                        curr_config.project.name = f'{method}{num_samples}_b{block_size}_{scorer}'
                        curr_config.project.promptspath = _SCORERS[scorer]

                        curr_config.guidance.method = method
                        curr_config.guidance.scorer = scorer
                        curr_config.guidance.num_samples = num_samples
                        curr_config.guidance.block_size = block_size

                        savepath = curr_path.joinpath(f'{curr_config.project.name}.yaml')
                        OmegaConf.save(curr_config, savepath)

            elif method == 'bon':

                for num_samples in [40]: # [25, 50, 100, 200, 500]:

                    curr_config = copy.deepcopy(template)
                    curr_config.project.name = f'{method}{num_samples}_{scorer}'
                    curr_config.project.promptspath = _SCORERS[scorer]

                    curr_config.guidance.method = method
                    curr_config.guidance.scorer = scorer
                    curr_config.guidance.num_samples = num_samples

                    savepath = curr_path.joinpath(f'{curr_config.project.name}.yaml')
                    OmegaConf.save(curr_config, savepath)

            elif method == 'bon_i2i':

                for num_samples in [5, 10, 20, 30, 40]: # [25, 50, 100, 200, 500]:

                    for percent_noise in np.arange(0.4, 0.9, 0.1): # np.arange(0.4, 1.0, 0.1):

                        curr_config = copy.deepcopy(template)
                        curr_config.project.name = f'{method}{num_samples}_r{int(float(round(percent_noise,1))*10)}_{scorer}'
                        curr_config.project.promptspath = _SCORERS[scorer]

                        curr_config.guidance.method = method
                        curr_config.guidance.scorer = scorer
                        curr_config.guidance.num_samples = num_samples
                        curr_config.guidance.num_gen_target_images_per_prompt = num_samples
                        curr_config.guidance.percent_noise = float(round(percent_noise,1))

                        savepath = curr_path.joinpath(f'{curr_config.project.name}.yaml')
                        OmegaConf.save(curr_config, savepath)

            elif method == 'grad':
                gs = np.arange(0.5,1.5,0.5) if 'style' in scorer else np.arange(100, 600, 100)
                for guidance_scale in gs: # [25, 50, 100, 200, 500]:

                    curr_config = copy.deepcopy(template)
                    curr_config.project.name = f'{method}{guidance_scale}_{scorer}'
                    curr_config.project.promptspath = _SCORERS[scorer]

                    curr_config.guidance.method = method
                    curr_config.guidance.scorer = scorer
                    curr_config.guidance.guidance_scale = int(guidance_scale)

                    savepath = curr_path.joinpath(f'{curr_config.project.name}.yaml')
                    OmegaConf.save(curr_config, savepath)

            elif method == 'grad_i2i':

                for guidance_scale in np.arange(100, 600, 100): # [25, 50, 100, 200, 500]:

                    for percent_noise in np.arange(0.5, 0.9, 0.1): # np.arange(0.4, 1.0, 0.1):

                        curr_config = copy.deepcopy(template)
                        curr_config.project.name = f'{method}{guidance_scale}_r{int(float(round(percent_noise,1))*10)}_{scorer}'
                        curr_config.project.promptspath = _SCORERS[scorer]

                        curr_config.guidance.method = method
                        curr_config.guidance.scorer = scorer
                        curr_config.guidance.guidance_scale = int(guidance_scale)
                        curr_config.guidance.percent_noise = float(round(percent_noise,1))

                        savepath = curr_path.joinpath(f'{curr_config.project.name}.yaml')
                        OmegaConf.save(curr_config, savepath)

            elif method in ['ibon_i2i', 'c_code']:

                for num_samples in [10, 20, 30, 40]: # [25, 50, 100, 200, 500]:

                    for block_size in [5, 20, 50, 100]: # [5, 10, 25]:

                        for percent_noise in np.arange(0.5, 0.9, 0.1): # np.arange(0.4, 1.0, 0.1):

                            curr_config = copy.deepcopy(template)
                            curr_config.project.name = f'{method}{num_samples}_b{block_size}_r{int(float(round(percent_noise,1))*10)}_{scorer}'
                            curr_config.project.promptspath = _SCORERS[scorer]

                            curr_config.guidance.method = method
                            curr_config.guidance.scorer = scorer
                            curr_config.guidance.num_samples = num_samples
                            curr_config.guidance.block_size = block_size
                            curr_config.guidance.num_gen_target_images_per_prompt = num_samples
                            curr_config.guidance.percent_noise = float(round(percent_noise,1))

                            savepath = curr_path.joinpath(f'{curr_config.project.name}.yaml')
                            OmegaConf.save(curr_config, savepath)

            elif method == 'i2i':

                for percent_noise in np.arange(0.4, 1.0, 0.1):

                    curr_config = copy.deepcopy(template)
                    curr_config.project.name = f'{method}_r{int(float(round(percent_noise,1))*10)}_{scorer}'
                    curr_config.project.promptspath = _SCORERS[scorer]

                    curr_config.guidance.method = method
                    curr_config.guidance.scorer = scorer
                    curr_config.guidance.percent_noise = float(round(percent_noise,1))

                    savepath = curr_path.joinpath(f'{curr_config.project.name}.yaml')
                    OmegaConf.save(curr_config, savepath)

if __name__ == '__main__':
    create_function()
