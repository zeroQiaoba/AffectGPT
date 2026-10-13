"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import cv2
import decord
import numpy as np
import random as rnd
from omegaconf import OmegaConf

import torch
from torchvision import transforms

from decord import VideoReader
from my_affectgpt.processors import transforms_video
from my_affectgpt.processors.base_processor import BaseProcessor
from my_affectgpt.processors.randaugment import VideoRandomAugment
from my_affectgpt.processors import functional_video as F
from my_affectgpt.common.registry import registry


MAX_INT = registry.get("MAX_INT")
decord.bridge.set_bridge("torch")

## video -> sampled frames
def load_video(video_path, n_frms=MAX_INT, height=-1, width=-1, sampling="uniform", return_msg=False):
    decord.bridge.set_bridge("torch")
    vr = VideoReader(uri=video_path, height=height, width=width)

    vlen = len(vr)
    start, end = 0, vlen

    n_frms_update = min(n_frms, vlen) # for vlen < n_frms, only read vlen

    if sampling == "uniform": # uniform sampling
        indices = np.arange(start, end, vlen / n_frms_update).astype(int).tolist()
    elif sampling == "headtail": # sample half randomly from the front, half from the back
        indices_h = sorted(rnd.sample(range(vlen // 2), n_frms_update // 2))
        indices_t = sorted(rnd.sample(range(vlen // 2, vlen), n_frms_update // 2))
        indices = indices_h + indices_t
    else:
        raise NotImplementedError

    #########################################
    ## for vlen < n_frms, pad into n_frms
    while len(indices) < n_frms:
        indices.append(indices[-1])
    #########################################

    # get_batch -> T, H, W, C
    temp_frms = vr.get_batch(indices) # this block errors: [h264 @ 0xc97e880] mmco: unref short failure => usually a problem with the video itself
    tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
    frms = tensor_frms.permute(3, 0, 1, 2).float()  # (C, T, H, W)

    if not return_msg:
        return frms

    fps = float(vr.get_avg_fps())
    sec = ", ".join([str(round(f / fps, 1)) for f in indices])
    msg = f"The video contains {len(indices)} frames sampled at {sec} seconds. "
    return frms, msg


## read and sample face info [this one not included]
def load_face(face_npy, n_frms=MAX_INT, height=-1, width=-1, sampling="uniform", return_msg=False):
    
    faces = np.load(face_npy)
    faces = [cv2.resize(face, (width, height)) for face in faces] # [seqlen, 224, 224, 3]

    vlen = len(faces)
    start, end = 0, vlen

    n_frms_update = min(n_frms, vlen) # for vlen < n_frms, only read vlen

    if sampling == "uniform": # uniform sampling
        indices = np.arange(start, end, vlen / n_frms_update).astype(int).tolist()
    elif sampling == "headtail": # sample half randomly from the front, half from the back
        indices_h = sorted(rnd.sample(range(vlen // 2), n_frms_update // 2))
        indices_t = sorted(rnd.sample(range(vlen // 2, vlen), n_frms_update // 2))
        indices = indices_h + indices_t
    else:
        raise NotImplementedError

    #########################################
    ## for vlen < n_frms, pad into n_frms
    while len(indices) < n_frms:
        indices.append(indices[-1])
    #########################################

    # get_batch -> T, H, W, C
    temp_frms = np.array(faces)[indices]
    tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
    frms = tensor_frms.permute(3, 0, 1, 2).float()  # (C, T, H, W)

    if not return_msg:
        return frms

    msg = "We read faces in this time."
    return frms, msg


# set the default image normalization processor
class AlproVideoBaseProcessor(BaseProcessor):
    def __init__(self, mean=None, std=None, n_frms=MAX_INT):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)
        self.normalize = transforms_video.NormalizeVideo(mean, std)
        self.n_frms = n_frms


## these functions ToUint8 / ToTHWC are custom-written
class ToUint8(object):
    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor.to(torch.uint8)

    def __repr__(self):
        return self.__class__.__name__


class ToTHWC(object):
    """
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (C, T, H, W)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (T, H, W, C)
    """

    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor.permute(1, 2, 3, 0)

    def __repr__(self):
        return self.__class__.__name__


class ResizeVideo(object):
    def __init__(self, target_size, interpolation_mode="bilinear"):
        self.target_size = target_size
        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: central cropping of video clip. Size is
            (C, T, crop_size, crop_size)
        """
        return F.resize(clip, self.target_size, self.interpolation_mode)

    def __repr__(self):
        return self.__class__.__name__ + "(resize_size={0})".format(self.target_size)


# default: image_size=224; n_frms=8;
@registry.register_processor("alpro_video_train")
class AlproVideoTrainProcessor(AlproVideoBaseProcessor):
    def __init__(
        self,
        image_size=384,
        mean=None,
        std=None,
        min_scale=0.5,
        max_scale=1.0,
        n_frms=MAX_INT,
    ):
        super().__init__(mean=mean, std=std, n_frms=n_frms)

        self.image_size = image_size

        self.transform = transforms.Compose(
            [
                # Video size is (C, T, H, W) -> after random crop, resize to the same scale as the input image (C, T, H, W)
                transforms_video.RandomResizedCropVideo(
                    image_size,
                    scale=(min_scale, max_scale),
                    interpolation_mode="bicubic",
                ),
                ToTHWC(),  # C, T, H, W -> T, H, W, C
                ToUint8(),
                transforms_video.ToTensorVideo(),  # T, H, W, C -> C, T, H, W
                self.normalize, # relies on the base class image (mean, std) processor
            ]
        )

    def __call__(self, vpath):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: video clip after transforms. Size is (C, T, size, size).
        """
        clip = load_video(
            video_path=vpath,
            n_frms=self.n_frms,
            height=self.image_size,
            width=self.image_size,
            sampling="headtail",
        )

        return self.transform(clip)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()
        image_size = cfg.get("image_size", 256)
        mean = cfg.get("mean", None)
        std = cfg.get("std", None)
        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)
        n_frms = cfg.get("n_frms", MAX_INT)
        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
            n_frms=n_frms,
        )


@registry.register_processor("alpro_video_eval")
class AlproVideoEvalProcessor(AlproVideoBaseProcessor):
    def __init__(self, image_size=256, mean=None, std=None, n_frms=MAX_INT):
        super().__init__(mean=mean, std=std, n_frms=n_frms)

        self.image_size = image_size

        # Input video size is (C, T, H, W)
        self.transform = transforms.Compose(
            [   
                ## at eval time the random crop is removed to keep the eval stage consistent
                # frames will be resized during decord loading.
                ToUint8(),  # C, T, H, W
                ToTHWC(),  # T, H, W, C
                transforms_video.ToTensorVideo(),  # C, T, H, W
                self.normalize,  # C, T, H, W
            ]
        )

    def __call__(self, vpath):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: video clip after transforms. Size is (C, T, size, size).
        """
        clip = load_video(
            video_path=vpath,
            n_frms=self.n_frms,
            height=self.image_size,
            width=self.image_size,
        )

        return self.transform(clip)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()
        image_size = cfg.get("image_size", 256)
        mean = cfg.get("mean", None)
        std = cfg.get("std", None)
        n_frms = cfg.get("n_frms", MAX_INT)
        return cls(image_size=image_size, mean=mean, std=std, n_frms=n_frms)
