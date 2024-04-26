import os
import random
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn

import video_llama.tasks as tasks
from video_llama.common.config import Config
from video_llama.common.dist_utils import get_rank, init_distributed_mode
from video_llama.common.logger import setup_logger
from video_llama.common.registry import registry
from video_llama.common.utils import now
from video_llama.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)

from video_llama.tasks import *
from video_llama.models import *
from video_llama.runners import *
from video_llama.processors import *
from video_llama.datasets.builders import *

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--options",  nargs="+", help="overwrite params in xxx.config (only for run and model). Example: --options 'ckpt=aaa' 'ckpt_2=bbb'")
    args = parser.parse_args()
    return args

def setup_seeds(config): 
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base")) # 'video_llama.runners.runner_base.RunnerBase'
    return runner_cls

def main():

    cfg = Config(parse_args())

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    max_epoch = cfg.run_cfg['max_epoch'] 
    iters_per_epoch = cfg.run_cfg['iters_per_epoch'] 
    warmup_steps = cfg.run_cfg['warmup_steps'] 
    job_id = f'{now()}_{max_epoch}_{iters_per_epoch}_{warmup_steps}'

    # print logging files
    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)
    setup_logger() 
    cfg.pretty_print()

    # load task and start training
    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)
    runner = get_runner_class(cfg)(
        cfg=cfg,
        job_id=job_id, 
        task=task, 
        model=model, 
        datasets=datasets
    )
    runner.train()


##################################################################################################
################ Different CUDA version needs different install approach #########################
##################################################################################################
## machine1: 
# conda env create -f environment.yml

## machine2: cuda=11.7
# conda create --name=videollama3 python=3.9
# conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
# pip install -r requirement.txt # Note: remove already installed packages in requirement.txt

## machine3：cuda=10.2
# conda create --name=videollama3 python=3.9
# conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
# conda install transformers
# pip install -r requirement.txt # Note: remove already installed packages in requirement.txt

## other errors：libGL.so.1 errors in opencv, then yum install mesa-libGL
##################################################################################################
if __name__ == "__main__":
    main()
