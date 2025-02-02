import os
import re

from pathlib import Path

def create_job_file(config, export_path):
    """Create the slurm job file
    """

    template_path = Path('pytorch.job')
    lines = []

    with open(template_path, 'r') as fp:
        for line in fp:
            lines.append(line)

    updated_lines = []

    for line in lines:
        if re.search("output", line):
            updated_lines.append(line.replace("slurm", config))

        elif re.search("--config", line):
            
            temp = line

            temp = temp.replace("method", export_path.stem)
            temp = temp.replace("config_file", config)
            updated_lines.append(temp)

        else:
            updated_lines.append(line)

    with open(export_path.joinpath(f'{config}.sbatch'), 'w') as fp:
        for line in updated_lines:
            fp.write(line)

    return


def main():
    """Generate job files for available configurations
    """

    config_dir = Path('../configs_split')

    methods = [x for x in config_dir.iterdir() if x.is_dir()]

    for method in methods:

        if method.stem not in ['c_code_b1']:
            continue

        curr_path = Path(method.stem)
        # print(curr_path)

        if not Path.exists(curr_path):
            Path.mkdir(curr_path, parents=True, exist_ok=True)

        configs = [x for x in method.iterdir() if x.is_file()]
        # breakpoint()
        for config in configs:   
            # if ('aesthetic' in config.stem) or ('style' in config.stem) or ('style' in config.stem):
            #     continue
            if ('compress' in config.stem) or ('stroke' in config.stem):
                continue
            print(config.stem)
            create_job_file(config.stem, curr_path)

if __name__ == '__main__':
    main()
