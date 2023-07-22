import os
import logging
import warnings

from video_llama.common.registry import registry
from video_llama.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from video_llama.datasets.datasets.cc_sbu_dataset import CCSBUAlignDataset
from video_llama.datasets.datasets.emotion_instruct_dataset import Emotion_Reasoning_Dataset


@registry.register_builder("cc_sbu_align")
class CCSBUAlignBuilder(BaseDatasetBuilder):
    train_dataset_cls = CCSBUAlignDataset

    def build_datasets(self):
        logging.info("Building datasets...")
        self.build_processors()

        datasets = dict()
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_path=self.config.build_info.ann_path,
            vis_root=self.config.build_info.vis_root,
            max_length=self.config.max_length,
            num_audio_query_token=self.config.num_audio_query_token,
            num_video_query_token=self.config.num_video_query_token,
            tokenizer_name=self.config.tokenizer_name
            )
        return datasets


@registry.register_builder("emotion_reason_instruct")
class EmoReasonBuilder(BaseDatasetBuilder):
    train_dataset_cls = Emotion_Reasoning_Dataset

    def build_datasets(self):
        logging.info("Building datasets...")
        self.build_processors()

        datasets = dict()
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            max_length = self.config.max_length,
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_path=self.config.build_info.ann_path,
            vis_root=self.config.build_info.vis_root,
            num_video_query_token=self.config.num_video_query_token,
            num_audio_query_token=self.config.num_audio_query_token,
            tokenizer_name=self.config.tokenizer_name
            )
        return datasets
    
