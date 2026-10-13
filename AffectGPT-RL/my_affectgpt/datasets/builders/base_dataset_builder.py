import os
import shutil
import logging
import warnings

import torch.distributed as dist
from omegaconf import OmegaConf

import my_affectgpt.common.utils as utils
from my_affectgpt.common.dist_utils import is_dist_avail_and_initialized, is_main_process
from my_affectgpt.common.registry import registry
from my_affectgpt.processors.base_processor import BaseProcessor

class BaseDatasetBuilder:
    train_dataset_cls, eval_dataset_cls = None, None

    def __init__(self, dataset_cfg=None, model_cfg=None):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.model_cfg = model_cfg
        self.vis_processors = {"train": BaseProcessor(), "eval": BaseProcessor()}
        self.txt_processors = {"train": BaseProcessor(), "eval": BaseProcessor()}
        self.img_processors = {"train": BaseProcessor(), "eval": BaseProcessor()}

    # load vis_processors and text_processors using config files
    def build_processors(self):
        vis_proc_cfg = self.model_cfg.get("vis_processor")
        txt_proc_cfg = self.model_cfg.get("txt_processor")
        img_proc_cfg = self.model_cfg.get("img_processor")

        if vis_proc_cfg is not None:
            vis_train_cfg = vis_proc_cfg.get("train") # {'name': 'alpro_video_train', 'n_frms': 8, 'image_size': 224}
            vis_eval_cfg = vis_proc_cfg.get("eval")   # None
            self.vis_processors["train"] = self._build_proc_from_cfg(vis_train_cfg) # alpro_video_train class initialized with vis_train_cfg
            self.vis_processors["eval"]  = self._build_proc_from_cfg(vis_eval_cfg)   # None

        if txt_proc_cfg is not None:
            txt_train_cfg = txt_proc_cfg.get("train") # {'name': 'blip_caption'}
            txt_eval_cfg = txt_proc_cfg.get("eval")   # None
            self.txt_processors["train"] = self._build_proc_from_cfg(txt_train_cfg) # blip_caption class
            self.txt_processors["eval"]  = self._build_proc_from_cfg(txt_eval_cfg)   # None
        
        if img_proc_cfg is not None:
            img_train_cfg = img_proc_cfg.get("train") # {'name': 'blip_caption'}
            img_eval_cfg = img_proc_cfg.get("eval")   # None
            self.img_processors["train"] = self._build_proc_from_cfg(img_train_cfg) # blip_caption class
            self.img_processors["eval"]  = self._build_proc_from_cfg(img_eval_cfg)   # None

    # cfg is None: return None; cfg is not None: parse its content
    @staticmethod
    def _build_proc_from_cfg(cfg):
        return (
            registry.get_processor_class(cfg.name).from_config(cfg)
            if cfg is not None
            else None
        )
