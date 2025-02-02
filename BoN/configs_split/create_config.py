import os
import copy
import numpy as np

from pathlib import Path
from omegaconf import OmegaConf

_METHODS = ['c_code_b1'] # ['c_bon', 'i2i'] # 'ibon', ibon_i2i', 'bon', 'uncond', 'i2i', 'bon_i2i', 'code', 'c_code'

_SCORERS = {
    # 'aesthetic': '../assets/eval_simple_animals.txt', 
    # 'hpsv2': '../assets/hps_v2_all_eval.txt', 
    'facedetector': '../assets/face.txt', 
    'styletransfer': '../assets/style.txt',
    # 'strokegen': '../assets/stroke.txt',
    # 'compress': '../assets/compressibility.txt',
    }

def create_function():

    template = OmegaConf.load('template.yaml')
    config_dir = Path('.')

    for method in _METHODS:

        curr_path = config_dir.joinpath(method)

        if not Path.exists(curr_path):
            Path.mkdir(curr_path, parents=True)

        for scorer in _SCORERS.keys():

            num_prompts = 3 if scorer == 'facedetector' else 4
            num_targets = 3

            print(f'{method} {scorer}')

            if method == 'uncond':

                for prompt_idx in range(num_prompts):

                    for target_idx in range(num_targets):

                        curr_config = copy.deepcopy(template)
                        curr_config.project.name = f'{method}_{scorer}'
                        curr_config.project.promptspath = _SCORERS[scorer]

                        curr_config.guidance.method = method
                        curr_config.guidance.scorer = scorer
                        curr_config.guidance.target_idxs = [target_idx]
                        curr_config.guidance.prompt_idxs = [prompt_idx]

                        filename = f'{method}_p{prompt_idx}_t{target_idx}_{scorer}'
                        savepath = curr_path.joinpath(f'{filename}.yaml')
                        OmegaConf.save(curr_config, savepath)
            
            elif method in ['code_b1']:

                for num_samples in [10, 20, 30, 100]: # [10, 20, 30, 40]: # [25, 50, 100, 200, 500]:

                    for block_size in [1]: # [5, 10, 20, 50, 100]

                        for prompt_idx in range(num_prompts):

                            if scorer in ['facedetector', 'styletransfer', 'strokegen']:
                                for target_idx in range(num_targets):

                                    curr_config = copy.deepcopy(template)
                                    m = method.split('_')[0]
                                    curr_config.project.name = f'{m}{num_samples}_b{block_size}_{scorer}'
                                    curr_config.project.promptspath = _SCORERS[scorer]

                                    curr_config.guidance.method = method
                                    curr_config.guidance.scorer = scorer
                                    curr_config.guidance.num_samples = num_samples
                                    curr_config.guidance.block_size = block_size
                                    curr_config.guidance.target_idxs = [target_idx]
                                    curr_config.guidance.prompt_idxs = [prompt_idx]

                                    filename = f'{m}{num_samples}_p{prompt_idx}_t{target_idx}_b{block_size}_{scorer}'
                                    savepath = curr_path.joinpath(f'{filename}.yaml')
                                    OmegaConf.save(curr_config, savepath)
                            
                            else:
                                curr_config = copy.deepcopy(template)
                                m = method.split('_')[0]
                                curr_config.project.name = f'{m}{num_samples}_b{block_size}_{scorer}'
                                curr_config.project.promptspath = _SCORERS[scorer]

                                curr_config.guidance.method = method
                                curr_config.guidance.scorer = scorer
                                curr_config.guidance.num_samples = num_samples
                                curr_config.guidance.block_size = block_size
                                curr_config.guidance.prompt_idxs = [prompt_idx]

                                filename = f'{m}{num_samples}_p{prompt_idx}_b{block_size}_{scorer}'
                                savepath = curr_path.joinpath(f'{filename}.yaml')
                                OmegaConf.save(curr_config, savepath)


            elif method in ['c_code_b1']:

                for num_samples in [10, 20, 30]: # [10, 20, 30, 40]: # [25, 50, 100, 200, 500]:
                    
                    pc = [0.7] if 'face' in scorer else [0.6]
                    for percent_noise in pc:
                        for block_size in [1]: # [5, 10, 20, 50, 100]

                            for prompt_idx in range(num_prompts):

                                for target_idx in range(num_targets):

                                    curr_config = copy.deepcopy(template)
                                    m = 'c_code'
                                    curr_config.project.name = f'{m}_{num_samples}_b{block_size}_r{int(float(round(percent_noise,1))*10)}_{scorer}'
                                    curr_config.project.promptspath = _SCORERS[scorer]

                                    curr_config.guidance.method = method
                                    curr_config.guidance.scorer = scorer
                                    curr_config.guidance.num_samples = num_samples
                                    curr_config.guidance.block_size = block_size
                                    curr_config.guidance.target_idxs = [target_idx]
                                    curr_config.guidance.prompt_idxs = [prompt_idx]
                                    curr_config.guidance.percent_noise = float(round(percent_noise,1))

                                    filename = f'{m}_{num_samples}_p{prompt_idx}_t{target_idx}_b{block_size}_r{int(float(round(percent_noise,1))*10)}_{scorer}'
                                    savepath = curr_path.joinpath(f'{filename}.yaml')
                                    OmegaConf.save(curr_config, savepath)

                                
            elif method in ['code']:

                for num_samples in [10, 20, 30, 40]: #[10, 20, 30, 40]: # [10, 20, 30, 40]:

                    for block_size in [5]: # [5, 10, 20, 50, 100]

                        for prompt_idx in range(num_prompts):

                            if scorer in ['facedetector', 'styletransfer', 'strokegen']:
                                for target_idx in range(num_targets):

                                    curr_config = copy.deepcopy(template)
                                    curr_config.project.name = f'{method}{num_samples}_b{block_size}_{scorer}'
                                    curr_config.project.promptspath = _SCORERS[scorer]

                                    curr_config.guidance.method = method
                                    curr_config.guidance.scorer = scorer
                                    curr_config.guidance.num_samples = num_samples
                                    curr_config.guidance.block_size = block_size
                                    curr_config.guidance.target_idxs = [target_idx]
                                    curr_config.guidance.prompt_idxs = [prompt_idx]

                                    filename = f'{method}{num_samples}_p{prompt_idx}_t{target_idx}_b{block_size}_{scorer}'
                                    savepath = curr_path.joinpath(f'{filename}.yaml')
                                    OmegaConf.save(curr_config, savepath)

                            else:
                                curr_config = copy.deepcopy(template)
                                curr_config.project.name = f'{method}{num_samples}_b{block_size}_{scorer}'
                                curr_config.project.promptspath = _SCORERS[scorer]

                                curr_config.guidance.method = method
                                curr_config.guidance.scorer = scorer
                                curr_config.guidance.num_samples = num_samples
                                curr_config.guidance.block_size = block_size
                                curr_config.guidance.prompt_idxs = [prompt_idx]

                                filename = f'{method}{num_samples}_p{prompt_idx}_b{block_size}_{scorer}'
                                savepath = curr_path.joinpath(f'{filename}.yaml')
                                OmegaConf.save(curr_config, savepath)



            elif method == 'bon':

                for num_samples in [10, 20, 30, 100]: # [5, 10, 20, 30, 40]: # [25, 50, 100, 200, 500]:

                    for prompt_idx in range(num_prompts):

                        if scorer in ['facedetector', 'styletransfer', 'strokegen']:
                            for target_idx in range(num_targets):

                                curr_config = copy.deepcopy(template)
                                curr_config.project.name = f'{method}{num_samples}_{scorer}'
                                curr_config.project.promptspath = _SCORERS[scorer]

                                curr_config.guidance.method = method
                                curr_config.guidance.scorer = scorer
                                curr_config.guidance.num_samples = num_samples
                                curr_config.guidance.target_idxs = [target_idx]
                                curr_config.guidance.prompt_idxs = [prompt_idx]

                                filename = f'{method}{num_samples}_p{prompt_idx}_t{target_idx}_{scorer}'
                                savepath = curr_path.joinpath(f'{filename}.yaml')
                                OmegaConf.save(curr_config, savepath)

                        else:
                            curr_config = copy.deepcopy(template)
                            curr_config.project.name = f'{method}{num_samples}_{scorer}'
                            curr_config.project.promptspath = _SCORERS[scorer]

                            curr_config.guidance.method = method
                            curr_config.guidance.scorer = scorer
                            curr_config.guidance.num_samples = num_samples
                            curr_config.guidance.prompt_idxs = [prompt_idx]

                            filename = f'{method}{num_samples}_p{prompt_idx}_{scorer}'
                            savepath = curr_path.joinpath(f'{filename}.yaml')
                            OmegaConf.save(curr_config, savepath)

            
            elif method == 'c_bon':

                for num_samples in [10, 20, 30, 100]: # [10, 20, 30, 40]: # [5, 10, 20, 30, 40]: # [25, 50, 100, 200, 500]:

                    for percent_noise in [0.8]: #  np.arange(0.5, 0.9, 0.1)
                        for prompt_idx in range(num_prompts):

                            for target_idx in range(num_targets):

                                curr_config = copy.deepcopy(template)
                                curr_config.project.name = f'{method}{num_samples}_r{int(float(round(percent_noise,1))*10)}_{scorer}'
                                curr_config.project.promptspath = _SCORERS[scorer]

                                curr_config.guidance.method = method
                                curr_config.guidance.scorer = scorer
                                curr_config.guidance.num_samples = num_samples
                                curr_config.guidance.target_idxs = [target_idx]
                                curr_config.guidance.prompt_idxs = [prompt_idx]
                                curr_config.guidance.percent_noise = float(round(percent_noise,1))

                                filename = f'{method}{num_samples}_p{prompt_idx}_t{target_idx}_r{float(round(percent_noise,1))}_{scorer}'
                                savepath = curr_path.joinpath(f'{filename}.yaml')
                                OmegaConf.save(curr_config, savepath)
            
            elif method == 'i2i':

                for percent_noise in [0.6, 0.7, 0.8]: # np.arange(0.5, 0.9, 0.1):
                    for prompt_idx in range(num_prompts):

                        for target_idx in range(num_targets):

                            curr_config = copy.deepcopy(template)
                            curr_config.project.name = f'{method}_r{int(float(round(percent_noise,1))*10)}_{scorer}'
                            curr_config.project.promptspath = _SCORERS[scorer]

                            curr_config.guidance.method = method
                            curr_config.guidance.scorer = scorer
                            curr_config.guidance.target_idxs = [target_idx]
                            curr_config.guidance.prompt_idxs = [prompt_idx]
                            curr_config.guidance.percent_noise = float(round(percent_noise,1))

                            filename = f'{method}_p{prompt_idx}_t{target_idx}_r{float(round(percent_noise,1))}_{scorer}'
                            savepath = curr_path.joinpath(f'{filename}.yaml')
                            OmegaConf.save(curr_config, savepath)

            elif method == 'grad':
            
                for guidance_scale in [1.3]: # np.arange(100, 600, 100): # [25, 50, 100, 200, 500]:

                    for prompt_idx in range(num_prompts):

                        if scorer in ['facedetector', 'styletransfer', 'strokegen']:
                            for target_idx in range(num_targets):

                                curr_config = copy.deepcopy(template)
                                curr_config.project.name = f'{method}{int(float(round(guidance_scale,1))*10)}_{scorer}'
                                curr_config.project.promptspath = _SCORERS[scorer]

                                curr_config.guidance.method = method
                                curr_config.guidance.scorer = scorer
                                curr_config.guidance.guidance_scale = float(guidance_scale)
                                curr_config.guidance.target_idxs = [target_idx]
                                curr_config.guidance.prompt_idxs = [prompt_idx]
                                
                                filename = f'{method}{int(float(round(guidance_scale,1))*10)}_p{prompt_idx}_t{target_idx}_{scorer}'
                                savepath = curr_path.joinpath(f'{filename}.yaml')
                                OmegaConf.save(curr_config, savepath)

                        else:
                            curr_config = copy.deepcopy(template)
                            curr_config.project.name = f'{method}{int(float(round(guidance_scale,1))*10)}_{scorer}'
                            curr_config.project.promptspath = _SCORERS[scorer]

                            curr_config.guidance.method = method
                            curr_config.guidance.scorer = scorer
                            curr_config.guidance.guidance_scale = float(guidance_scale)
                            curr_config.guidance.prompt_idxs = [prompt_idx]
                            
                            filename = f'{method}{int(float(round(guidance_scale,1))*10)}_p{prompt_idx}_{scorer}'
                            savepath = curr_path.joinpath(f'{filename}.yaml')
                            OmegaConf.save(curr_config, savepath)


                
            elif method in ['c_code']:

                num_prompts = 3 if scorer == 'facedetector' else 4
                num_targets = 3

                for num_samples in [10, 20, 30 , 40]: # [10, 20, 30, 40]:

                    for block_size in [5]: # [5, 10, 20, 50, 100]

                        # pc = [0.6] if 'style' in scorer else [0.7]
                        for percent_noise in [0.8]:
                            # pt = [4,5] if 'style' in scorer else [3,4]
                            for prompt_idx in range(num_prompts):# pt 

                                for target_idx in range(num_targets):

                                    curr_config = copy.deepcopy(template)
                                    curr_config.project.name = f'{method}{num_samples}_b{block_size}_r{int(float(round(percent_noise,1))*10)}_{scorer}'
                                    curr_config.project.promptspath = _SCORERS[scorer]

                                    curr_config.guidance.method = method
                                    curr_config.guidance.scorer = scorer
                                    curr_config.guidance.num_samples = num_samples
                                    curr_config.guidance.block_size = block_size
                                    curr_config.guidance.target_idxs = [target_idx]
                                    curr_config.guidance.prompt_idxs = [prompt_idx]
                                    curr_config.guidance.percent_noise = float(round(percent_noise,1))

                                    filename = f'{method}{num_samples}_p{prompt_idx}_t{target_idx}_b{block_size}_r{float(round(percent_noise,1))}_{scorer}'
                                    savepath = curr_path.joinpath(f'{filename}.yaml')
                                    OmegaConf.save(curr_config, savepath)

if __name__ == '__main__':
    create_function()
