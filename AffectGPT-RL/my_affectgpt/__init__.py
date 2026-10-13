"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import sys
from omegaconf import OmegaConf

from my_affectgpt.tasks import *
from my_affectgpt.models import *
from my_affectgpt.processors import *
from my_affectgpt.datasets.builders import *
from my_affectgpt.common.registry import registry

root_dir = os.path.dirname(os.path.abspath(__file__))
registry.register_path("library_root", root_dir)

repo_root = os.path.join(root_dir, "..")
registry.register_path("repo_root", repo_root)

cache_root = os.path.join(repo_root, "cache")
registry.register_path("cache_root", cache_root)

registry.register("MAX_INT", sys.maxsize)

registry.register("SPLIT_NAMES", ["train", "val", "test"])
