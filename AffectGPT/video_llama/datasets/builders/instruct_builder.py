import os
import logging
import warnings

from video_llama.common.registry import registry
from video_llama.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from video_llama.datasets.datasets.llava_instruct_dataset import Instruct_Dataset
from video_llama.datasets.datasets.video_instruct_dataset import Video_Instruct_Dataset


@registry.register_builder("webvid_instruct")
class WebvidInstruct_Builder(BaseDatasetBuilder):
    train_dataset_cls = Video_Instruct_Dataset

    def build_datasets(self):
        self.build_processors()

        datasets = dict()
        dataset_cls = self.train_dataset_cls
        datasets["train"] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            vis_root=self.config.build_info.vis_root,
            ann_path=self.config.build_info.ann_path,
            num_audio_query_token = self.config.num_audio_query_token,
            num_video_query_token = self.config.num_video_query_token,
            tokenizer_name = self.config.tokenizer_name,
            max_length = self.config.max_length,
        )
        return datasets


@registry.register_builder("llava_instruct")
class LlavaInstruct_Builder(BaseDatasetBuilder):
    train_dataset_cls = Instruct_Dataset

    def build_datasets(self):
        self.build_processors()

        datasets = dict()
        dataset_cls = self.train_dataset_cls
        datasets["train"] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            vis_root=self.config.build_info.vis_root,
            ann_path=self.config.build_info.ann_path,
            num_audio_query_token = self.config.num_audio_query_token,
            num_video_query_token = self.config.num_video_query_token,
            tokenizer_name = self.config.tokenizer_name,
            max_length = self.config.max_length,
        )

        return datasets

