# *_*coding:utf-8 *_*
import os

#########################################################################################
## Global paths
#########################################################################################
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(REPO_ROOT, 'dataset') # all data (video/audio, subtitles, labels) is read from dataset/ inside this repository
EMOTION_WHEEL_ROOT = os.path.join(REPO_ROOT, 'emotion_wheel')
OUTSIDE_WHEEL_MAPPING = os.path.join(EMOTION_WHEEL_ROOT, 'wheel_mapping.npz') # read from the original xlsx; now loads a pre-compressed mapping file instead
BERT_BASE_UNCASED = 'models/bert-base-uncased'


#######################
## Storage paths for all models
#######################
PATH_TO_LLM = {
    'Qwen25': 'models/Qwen2.5-7B-Instruct',
}

PATH_TO_VISUAL = {
    'CLIP_VIT_LARGE': 'models/clip-vit-large-patch14',
}

PATH_TO_AUDIO = {
    'HUBERT_LARGE':  'models/chinese-hubert-large',
}


#######################
## Storage paths for all datasets
#######################
## the three datasets share the dataset/ dir inside this repository; only the label file differs
DATA_DIR = {
    'OVMERDPlus':     DATASET_ROOT,
    'MER25OV1000':    DATASET_ROOT,
    'MERCaptionPlus': DATASET_ROOT,
}

## temporary override, only to make the model run locally
PATH_TO_RAW_AUDIO      = {key: os.path.join(DATASET_ROOT, 'audio')               for key in DATA_DIR}
PATH_TO_RAW_VIDEO      = {key: os.path.join(DATASET_ROOT, 'video')               for key in DATA_DIR}
PATH_TO_RAW_FACE       = {key: os.path.join(DATASET_ROOT, 'openface_face')       for key in DATA_DIR}
PATH_TO_TRANSCRIPTIONS = {key: os.path.join(DATASET_ROOT, 'subtitle_chieng.csv') for key in DATA_DIR}

## OV labels (columns: name, openset), one file per dataset
PATH_TO_LABEL = {
    'OVMERDPlus':     os.path.join(DATASET_ROOT, 'ovmerdplus-emotion.csv'),    # OV-MERD+, 532 samples
    'MER25OV1000':    os.path.join(DATASET_ROOT, 'mer2025ov1000-emotion.csv'),     # MER2025-OV, 1000 samples
    'MERCaptionPlus': os.path.join(DATASET_ROOT, 'mercaptionplus-emotion.csv'),# MER-Caption+, 31327 samples
}

## Emotion descriptions (columns: name, reason), used by the cold-start description task
## when the file is missing, dataset loading falls back to 'not provided'
PATH_TO_REASON = {
    'MERCaptionPlus': os.path.join(DATASET_ROOT, 'mercaptionplus-description.csv'),
}

#######################
## store global values
#######################
DEFAULT_IMAGE_PATCH_TOKEN = '<ImageHere>'
DEFAULT_AUDIO_PATCH_TOKEN = '<AudioHere>'
DEFAULT_FRAME_PATCH_TOKEN = '<FrameHere>'
DEFAULT_FACE_PATCH_TOKEN  = '<FaceHere>'
DEFAULT_MULTI_PATCH_TOKEN = '<MultiHere>'
IGNORE_INDEX = -100
