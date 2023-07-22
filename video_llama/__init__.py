"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import sys
from omegaconf import OmegaConf

from video_llama.tasks import *
from video_llama.models import *
from video_llama.processors import *
from video_llama.datasets.builders import *
from video_llama.common.registry import registry

root_dir = os.path.dirname(os.path.abspath(__file__))
registry.register_path("library_root", root_dir)

repo_root = os.path.join(root_dir, "..")
registry.register_path("repo_root", repo_root)

cache_root = os.path.join(repo_root, "cache")
registry.register_path("cache_root", cache_root)

registry.register("MAX_INT", sys.maxsize)

registry.register("SPLIT_NAMES", ["train", "val", "test"])
