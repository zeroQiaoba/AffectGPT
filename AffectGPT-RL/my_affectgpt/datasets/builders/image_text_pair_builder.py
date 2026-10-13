import os
import logging
import warnings

from my_affectgpt.common.registry import registry
from my_affectgpt.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from my_affectgpt.datasets.datasets.ovmerdplus_dataset import OVMERDPlus_Dataset
from my_affectgpt.datasets.datasets.mer2023 import MER2023_Dataset
from my_affectgpt.datasets.datasets.mer2024 import MER2024_Dataset
from my_affectgpt.datasets.datasets.meld import MELD_Dataset
from my_affectgpt.datasets.datasets.cmumosi  import CMUMOSI_Dataset
from my_affectgpt.datasets.datasets.cmumosei import CMUMOSEI_Dataset
from my_affectgpt.datasets.datasets.sims import SIMS_Dataset
from my_affectgpt.datasets.datasets.simsv2 import SIMSv2_Dataset
from my_affectgpt.datasets.datasets.iemocap import IEMOCAPFour_Dataset
from my_affectgpt.datasets.datasets.mercaptionplus_dataset import MERCaptionPlus_Dataset
from my_affectgpt.datasets.datasets.mer25ov1000_dataset import MER25OV1000_Dataset

###################################################
###################################################
###################################################
@registry.register_builder("ovmerdplus")
class OVMERDPlusBuilder(BaseDatasetBuilder):
    train_dataset_cls = OVMERDPlus_Dataset

    def build_datasets(self):
        logging.info("Building datasets OVMERDPlus")
        self.build_processors()

        datasets = dict()
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            txt_processor=self.txt_processors["train"],
            img_processor=self.img_processors["train"],
            dataset_cfg=self.dataset_cfg,
            model_cfg=self.model_cfg,
            )
        return datasets


###################################################
###################################################
###################################################
@registry.register_builder("mer2023")
class MER2023Builder(BaseDatasetBuilder):
    train_dataset_cls = MER2023_Dataset

    def build_datasets(self):
        logging.info("Building datasets MER2023")
        self.build_processors()

        datasets = dict()
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            txt_processor=self.txt_processors["train"],
            img_processor=self.img_processors["train"],
            dataset_cfg=self.dataset_cfg,
            model_cfg=self.model_cfg,
            )
        return datasets


@registry.register_builder("mer2024")
class MER2024Builder(BaseDatasetBuilder):
    train_dataset_cls = MER2024_Dataset

    def build_datasets(self):
        logging.info("Building datasets MER2024")
        self.build_processors()

        datasets = dict()
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            txt_processor=self.txt_processors["train"],
            img_processor=self.img_processors["train"],
            dataset_cfg=self.dataset_cfg,
            model_cfg=self.model_cfg,
            )
        return datasets



@registry.register_builder("meld")
class MELDBuilder(BaseDatasetBuilder):
    train_dataset_cls = MELD_Dataset

    def build_datasets(self):
        logging.info("Building datasets MELD")
        self.build_processors()

        datasets = dict()
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            txt_processor=self.txt_processors["train"],
            img_processor=self.img_processors["train"],
            dataset_cfg=self.dataset_cfg,
            model_cfg=self.model_cfg,
            )
        return datasets


@registry.register_builder("iemocapfour")
class IEMOCAPFourBuilder(BaseDatasetBuilder):
    train_dataset_cls = IEMOCAPFour_Dataset

    def build_datasets(self):
        logging.info("Building datasets IEMOCAPFour")
        self.build_processors()

        datasets = dict()
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            txt_processor=self.txt_processors["train"],
            img_processor=self.img_processors["train"],
            dataset_cfg=self.dataset_cfg,
            model_cfg=self.model_cfg,
            )
        return datasets
    

@registry.register_builder("cmumosi")
class CMUMOSIBuilder(BaseDatasetBuilder):
    train_dataset_cls = CMUMOSI_Dataset

    def build_datasets(self):
        logging.info("Building datasets CMUMOSI")
        self.build_processors()

        datasets = dict()
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            txt_processor=self.txt_processors["train"],
            img_processor=self.img_processors["train"],
            dataset_cfg=self.dataset_cfg,
            model_cfg=self.model_cfg,
            )
        return datasets



@registry.register_builder("cmumosei")
class CMUMOSEIBuilder(BaseDatasetBuilder):
    train_dataset_cls = CMUMOSEI_Dataset

    def build_datasets(self):
        logging.info("Building datasets CMUMOSEI")
        self.build_processors()

        datasets = dict()
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            txt_processor=self.txt_processors["train"],
            img_processor=self.img_processors["train"],
            dataset_cfg=self.dataset_cfg,
            model_cfg=self.model_cfg,
            )
        return datasets


@registry.register_builder("sims")
class SIMSBuilder(BaseDatasetBuilder):
    train_dataset_cls = SIMS_Dataset

    def build_datasets(self):
        logging.info("Building datasets SIMS")
        self.build_processors()

        datasets = dict()
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            txt_processor=self.txt_processors["train"],
            img_processor=self.img_processors["train"],
            dataset_cfg=self.dataset_cfg,
            model_cfg=self.model_cfg,
            )
        return datasets


@registry.register_builder("simsv2")
class SIMSv2Builder(BaseDatasetBuilder):
    train_dataset_cls = SIMSv2_Dataset

    def build_datasets(self):
        logging.info("Building datasets SIMSv2")
        self.build_processors()

        datasets = dict()
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            txt_processor=self.txt_processors["train"],
            img_processor=self.img_processors["train"],
            dataset_cfg=self.dataset_cfg,
            model_cfg=self.model_cfg,
            )
        return datasets


@registry.register_builder("mercaptionplus")
class MERCaptionPlus_Builder(BaseDatasetBuilder):
    train_dataset_cls = MERCaptionPlus_Dataset

    def build_datasets(self):
        logging.info("Building datasets MERCaptionPlus_Dataset")
        self.build_processors()

        datasets = dict()
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            txt_processor=self.txt_processors["train"],
            img_processor=self.img_processors["train"],
            dataset_cfg=self.dataset_cfg,
            model_cfg=self.model_cfg,
            )
        return datasets


@registry.register_builder("mer25ov1000")
class MER25OV1000_Builder(BaseDatasetBuilder):
    train_dataset_cls = MER25OV1000_Dataset

    def build_datasets(self):
        logging.info("Building datasets MER25OV1000_Dataset")
        self.build_processors()

        datasets = dict()
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            txt_processor=self.txt_processors["train"],
            img_processor=self.img_processors["train"],
            dataset_cfg=self.dataset_cfg,
            model_cfg=self.model_cfg,
            )
        return datasets


