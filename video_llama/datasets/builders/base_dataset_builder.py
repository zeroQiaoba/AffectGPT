import logging
import os
import shutil
import warnings

import torch.distributed as dist
from omegaconf import OmegaConf
from torchvision.datasets.utils import download_url

import video_llama.common.utils as utils
from video_llama.common.dist_utils import is_dist_avail_and_initialized, is_main_process
from video_llama.common.registry import registry
from video_llama.processors.base_processor import BaseProcessor

class BaseDatasetBuilder:
    train_dataset_cls, eval_dataset_cls = None, None

    def __init__(self, cfg=None):
        super().__init__()
        self.config = cfg
        self.data_type = self.config.data_type
        self.vis_processors = {"train": BaseProcessor(), "eval": BaseProcessor()}
        self.text_processors = {"train": BaseProcessor(), "eval": BaseProcessor()}

    # load vis_processors and text_processors using config files
    def build_processors(self):
        vis_proc_cfg = self.config.get("vis_processor")
        txt_proc_cfg = self.config.get("text_processor")

        if vis_proc_cfg is not None:
            vis_train_cfg = vis_proc_cfg.get("train") # if value is not exists, return None
            vis_eval_cfg = vis_proc_cfg.get("eval")
            self.vis_processors["train"] = self._build_proc_from_cfg(vis_train_cfg)
            self.vis_processors["eval"] = self._build_proc_from_cfg(vis_eval_cfg)

        if txt_proc_cfg is not None:
            txt_train_cfg = txt_proc_cfg.get("train")
            txt_eval_cfg = txt_proc_cfg.get("eval")
            self.text_processors["train"] = self._build_proc_from_cfg(txt_train_cfg)
            self.text_processors["eval"] = self._build_proc_from_cfg(txt_eval_cfg)

    @staticmethod
    def _build_proc_from_cfg(cfg):
        return (
            registry.get_processor_class(cfg.name).from_config(cfg)
            if cfg is not None
            else None
        )
