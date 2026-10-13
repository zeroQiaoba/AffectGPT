import os
import time
import glob
import random
import argparse
import numpy as np
from pathlib import Path

import torch
from datetime import datetime
import torch.backends.cudnn as cudnn

import my_affectgpt.tasks as tasks
from my_affectgpt.common.config import Config
from my_affectgpt.common.dist_utils import get_rank, init_distributed_mode
from my_affectgpt.common.logger import setup_logger
from my_affectgpt.common.registry import registry
from my_affectgpt.common.optims import LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler
from my_affectgpt.tasks import *
from my_affectgpt.models import *
from my_affectgpt.runners import *
from my_affectgpt.processors import *
from my_affectgpt.datasets.builders import *


def search_for_ckpt_root(root_candidates):
    if len(root_candidates) == 0:
        return ''
    
    # keep the root that holds the most checkpoint files
    maxcount = 0
    targetroot = ''
    for root in root_candidates:
        count = len([path for path in os.listdir(root) if path.startswith('checkpoint_')])
        print (root, '==>', count)
        if count > maxcount:
            maxcount = count
            targetroot = root
    print ('================================================')
    print (f'Targetroot: epoch range: 0-{maxcount-1}')
    
    # report when the last checkpoint of the target root was created
    last_file = sorted(glob.glob(targetroot + '/checkpoint*'))[-1]
    file_stat = Path(last_file).stat()
    creation_time = file_stat.st_ctime
    print("Targetroot: Last ckpt creation time:", datetime.fromtimestamp(creation_time))
    print ('================================================')
    return targetroot


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
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base")) # 'affectgpt.runners.runner_base.RunnerBase'
    return runner_cls


def main():

    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--options",  nargs="+", help="overwrite params in xxx.config (only for run and model). Example: --options 'ckpt=aaa' 'ckpt_2=bbb'")
    args = parser.parse_args()
    cfg = Config(args)

    ## R1 training supports batch_size=1 only, and has to start from a pre-trained checkpoint
    if cfg.model_cfg.arch.startswith('affectgpt_r1'):
        assert cfg.run_cfg.batch_size_train == 1
        try:
            cfg_name = os.path.basename(args.cfg_path)[:-len('.yaml')]
            if cfg_name.count('continue') == 1:
                cfg_name = cfg_name.split('stage2')[0]+'stage1' # drop everything after r1stage2
                root_candidates = glob.glob(os.path.join('output', cfg_name+'*', cfg_name+'*'))
                ckpt3_root = search_for_ckpt_root(root_candidates)
                ckpt3_path = glob.glob("%s/*%06d*.pth" %(ckpt3_root, int(cfg.model_cfg.continue_epoch)))[0]
            elif cfg_name.count('continue') == 2:
                cfg_name = "_continue".join(cfg_name.split('_continue')[:-1]) # drop everything after the second 'continue'
                root_candidates = glob.glob(os.path.join('output', cfg_name, cfg_name+'*'))
                ckpt3_root = search_for_ckpt_root(root_candidates)
                ckpt3_path = glob.glob("%s/*%06d*.pth" %(ckpt3_root, int(cfg.model_cfg.continue_epoch)))[0]
            print (ckpt3_path)
            cfg.model_cfg.ckpt_3 = ckpt3_path
        except:
            raise ValueError("Random initialization")

    # distributed training: one shared job_id puts every rank in the same process group, so an NCCL barrier cannot hang
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    job_name = os.path.basename(args.cfg_path)[:-len('.yaml')]
    job_id = f"{job_name}_{datetime.now().strftime('%Y%m%d%H%M')[:-1]}"
    print (job_id)

    # set up distributed mode, seeds and logger
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


if __name__ == "__main__":
    main()

