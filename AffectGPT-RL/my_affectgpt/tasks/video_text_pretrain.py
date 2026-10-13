"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from my_affectgpt.common.registry import registry
from my_affectgpt.tasks.base_task import BaseTask


@registry.register_task("video_text_pretrain")
class VideoTextPretrainTask(BaseTask): # everything is inherited from the video_text_pretrain task
    def __init__(self):
        super().__init__()

    def evaluation(self, model, data_loader, cuda_enabled=True):
        pass
