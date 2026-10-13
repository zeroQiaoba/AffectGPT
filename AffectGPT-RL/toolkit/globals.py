## emotion mapping functions
emos_mer = ['neutral', 'angry', 'happy', 'sad', 'worried',  'surprise']
emo2idx_mer, idx2emo_mer = {}, {}
for ii, emo in enumerate(emos_mer): emo2idx_mer[emo] = ii
for ii, emo in enumerate(emos_mer): idx2emo_mer[ii] = emo

import sys
sys.path.append("../..")
import config

################## Audio Model ######################
# huggingface
HUBERT_BASE_CHINESE = 'chinese-hubert-base' # https://huggingface.co/TencentGameMate/chinese-hubert-base
HUBERT_LARGE_CHINESE = 'chinese-hubert-large' # https://huggingface.co/TencentGameMate/chinese-hubert-large
WAV2VEC2_BASE_CHINESE = 'chinese-wav2vec2-base' # https://huggingface.co/TencentGameMate/chinese-wav2vec2-base
WAV2VEC2_LARGE_CHINESE = 'chinese-wav2vec2-large' # https://huggingface.co/TencentGameMate/chinese-wav2vec2-large
WAV2VEC2_BASE = 'wav2vec2-base-960h' # https://huggingface.co/facebook/wav2vec2-base-960h
WAV2VEC2_LARGE = 'wav2vec2-large-960h' # https://huggingface.co/facebook/wav2vec2-large-960h
WAVLM_BASE = 'wavlm-base' # https://huggingface.co/microsoft/wavlm-base
WAVLM_LARGE = 'wavlm-large' # https://huggingface.co/microsoft/wavlm-large
WHISPER_BASE = 'whisper-base' # https://huggingface.co/openai/whisper-base
WHISPER_LARGE = 'whisper-large-v2' # https://huggingface.co/openai/whisper-large-v2
DATA2VEC_AUDIO_BASE = 'data2vec-audio-base-960h' # https://huggingface.co/facebook/data2vec-audio-base-960h
DATA2VEC_AUDIO_LARGE = 'data2vec-audio-large' # https://huggingface.co/facebook/data2vec-audio-large

# other models
IS09 = 'IS09'
IS10 = 'IS10'
IS13 = 'IS13'
eGeMAPS = 'eGeMAPS'
WAV2VEC_LARGE_Z = 'wav2vec-large-z'
WAV2VEC_LARGE_C = 'wav2vec-large-c'
VGGISH = 'vggish'
EMOTION2VEC = 'emotion2vec'
EMOTION2VEC_FINETUNE = 'emotion2vec-finetune'

WHOLE_AUDIO = [WAVLM_BASE, WAVLM_LARGE, HUBERT_BASE_CHINESE, HUBERT_LARGE_CHINESE,
               WAV2VEC2_BASE_CHINESE, WAV2VEC2_LARGE_CHINESE, WAV2VEC2_BASE,
               WAV2VEC2_LARGE, DATA2VEC_AUDIO_BASE, DATA2VEC_AUDIO_LARGE, WHISPER_BASE, WHISPER_LARGE, IS09, IS10, IS13,
               eGeMAPS, WAV2VEC_LARGE_Z, WAV2VEC_LARGE_C, VGGISH, EMOTION2VEC, EMOTION2VEC_FINETUNE]


##################### Text Model #####################
# English Model
BERT_BASE = 'bert-base-cased'
BERT_LARGE = 'bert-large-cased'
BERT_BASE_UNCASED = 'bert-base-uncased'
BERT_LARGE_UNCASED = 'bert-large-uncased'
ALBERT_BASE = 'albert-base-v2'
ALBERT_LARGE = 'albert-large-v2'
ALBERT_XXLARGE = 'albert-xxlarge-v2'
ROBERTA_BASE = 'roberta-base'
ROBERTA_LARGE = 'roberta-large'
ELECTRA_BASE = 'electra-base-discriminator'
ELECTRA_LARGE = 'electra-large-discriminator'
XLNET_BASE = 'xlnet-base-cased'
XLNET_LARGE = 'xlnet-large-cased'
T5_BASE = 't5-base'
T5_LARGE = 't5-large'
DEBERTA_BASE = 'deberta-base'
DEBERTA_LARGE = 'deberta-large'
DEBERTA_XLARGE = 'deberta-v2-xlarge'
DEBERTA_XXLARGE = 'deberta-v2-xxlarge'

# Chinese Model
BERT_BASE_CHINESE = 'bert-base-chinese' # https://huggingface.co/bert-base-chinese
ROBERTA_BASE_CHINESE = 'chinese-roberta-wwm-ext' # https://huggingface.co/hfl/chinese-roberta-wwm-ext
ROBERTA_LARGE_CHINESE = 'chinese-roberta-wwm-ext-large' # https://huggingface.co/hfl/chinese-roberta-wwm-ext-large
DEBERTA_LARGE_CHINESE = 'deberta-chinese-large' # https://huggingface.co/WENGSYX/Deberta-Chinese-Large
ELECTRA_SMALL_CHINESE = 'chinese-electra-180g-small' # https://huggingface.co/hfl/chinese-electra-180g-small-discriminator
ELECTRA_BASE_CHINESE  = 'chinese-electra-180g-base' # https://huggingface.co/hfl/chinese-electra-180g-base-discriminator
ELECTRA_LARGE_CHINESE = 'chinese-electra-180g-large' # https://huggingface.co/hfl/chinese-electra-180g-large-discriminator
XLNET_BASE_CHINESE = 'chinese-xlnet-base' # https://huggingface.co/hfl/chinese-xlnet-base
MACBERT_BASE_CHINESE = 'chinese-macbert-base' # https://huggingface.co/hfl/chinese-macbert-base
MACBERT_LARGE_CHINESE = 'chinese-macbert-large' # https://huggingface.co/hfl/chinese-macbert-large
PERT_BASE_CHINESE = 'chinese-pert-base' # https://huggingface.co/hfl/chinese-pert-base
PERT_LARGE_CHINESE = 'chinese-pert-large' # https://huggingface.co/hfl/chinese-pert-large
LERT_SMALL_CHINESE = 'chinese-lert-small' # https://huggingface.co/hfl/chinese-lert-small
LERT_BASE_CHINESE  = 'chinese-lert-base' # https://huggingface.co/hfl/chinese-lert-base
LERT_LARGE_CHINESE = 'chinese-lert-large' # https://huggingface.co/hfl/chinese-lert-large
GPT2_CHINESE = 'gpt2-chinese-cluecorpussmall' # https://huggingface.co/uer/gpt2-chinese-cluecorpussmall
CLIP_CHINESE = 'taiyi-clip-roberta-chinese' # https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-Roberta-102M-Chinese
WENZHONG_GPT2_CHINESE = 'wenzhong2-gpt2-chinese' # https://huggingface.co/IDEA-CCNL/Wenzhong2.0-GPT2-3.5B-chinese
ALBERT_TINY_CHINESE = 'albert_chinese_tiny' # https://huggingface.co/clue/albert_chinese_tiny
ALBERT_SMALL_CHINESE = 'albert_chinese_small' # https://huggingface.co/clue/albert_chinese_small
SIMBERT_BASE_CHINESE = 'simbert-base-chinese' # https://huggingface.co/WangZeJun/simbert-base-chinese

# maybe multi-linguish
MPNET_BASE = 'paraphrase-multilingual-mpnet-base-v2' # https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2
LLAMA_7B  = 'llama-7b-hf' # https://huggingface.co/decapoda-research/llama-7b-hf
LLAMA_13B = 'llama-13b-hf' # https://huggingface.co/decapoda-research/llama-13b-hf
LLAMA2_7B = 'llama-2-7b' # https://huggingface.co/meta-llama/Llama-2-7b
LLAMA2_13B = 'Llama-2-13b-hf' # https://huggingface.co/NousResearch/Llama-2-13b-hf
VICUNA_7B  = 'vicuna-7b-v0' # https://huggingface.co/lmsys/vicuna-7b-delta-v0
VICUNA_13B = 'stable-vicuna-13b' # https://huggingface.co/CarperAI/stable-vicuna-13b-delta
ALPACE_13B = 'chinese-alpaca-2-13b' # https://huggingface.co/ziqingyang/chinese-alpaca-2-13b
MOSS_7B = 'moss-base-7b' # https://huggingface.co/fnlp/moss-base-7b
STABLEML_7B = 'stablelm-base-alpha-7b-v2' # https://huggingface.co/stabilityai/stablelm-base-alpha-7b-v2
BLOOM_7B = 'bloom-7b1' # https://huggingface.co/bigscience/bloom-7b1
CHATGLM2_6B = 'chatglm2-6b' # https://huggingface.co/THUDM/chatglm2-6b
FALCON_7B = 'falcon-7b' # https://huggingface.co/tiiuae/falcon-7b
BAICHUAN_7B = 'Baichuan-7B' # https://huggingface.co/baichuan-inc/Baichuan-7B
BAICHUAN_13B = 'Baichuan-13B-Base' # https://huggingface.co/baichuan-inc/Baichuan-13B-Base
BAICHUAN2_7B = 'Baichuan2-7B-Base' # https://huggingface.co/baichuan-inc/Baichuan2-7B-Base
BAICHUAN2_13B = 'Baichuan2-13B-Base' # https://huggingface.co/baichuan-inc/Baichuan2-13B-Base
OPT_13B = 'opt-13b' # https://huggingface.co/facebook/opt-13b

WHOLE_TEXT = [BERT_BASE_CHINESE, ROBERTA_BASE_CHINESE, ROBERTA_LARGE_CHINESE, DEBERTA_LARGE_CHINESE,
              ELECTRA_SMALL_CHINESE, ELECTRA_BASE_CHINESE, ELECTRA_LARGE_CHINESE, XLNET_BASE_CHINESE,
              MACBERT_BASE_CHINESE, MACBERT_LARGE_CHINESE, PERT_BASE_CHINESE, PERT_LARGE_CHINESE,
              LERT_SMALL_CHINESE, LERT_BASE_CHINESE, LERT_LARGE_CHINESE, GPT2_CHINESE, CLIP_CHINESE,
              WENZHONG_GPT2_CHINESE, ALBERT_TINY_CHINESE, ALBERT_SMALL_CHINESE, SIMBERT_BASE_CHINESE,
              MPNET_BASE, LLAMA_7B, LLAMA_13B, LLAMA2_7B, LLAMA2_13B, VICUNA_7B, VICUNA_13B, ALPACE_13B, 
              MOSS_7B, STABLEML_7B, BLOOM_7B, CHATGLM2_6B, FALCON_7B, BAICHUAN_7B, BAICHUAN_13B, BAICHUAN2_7B, 
              BAICHUAN2_13B, OPT_13B, BERT_BASE, BERT_LARGE, BERT_BASE_UNCASED, BERT_LARGE_UNCASED, ALBERT_BASE, ALBERT_LARGE, 
              ALBERT_XXLARGE, ROBERTA_BASE, ROBERTA_LARGE, ELECTRA_BASE, ELECTRA_LARGE, XLNET_BASE, 
              XLNET_LARGE, T5_BASE, T5_LARGE, DEBERTA_BASE, DEBERTA_LARGE, DEBERTA_XLARGE, DEBERTA_XXLARGE]


################## Image Model ######################
# huggingface
CLIP_VIT_BASE = 'clip-vit-base-patch32' # https://huggingface.co/openai/clip-vit-base-patch32
CLIP_VIT_LARGE = 'clip-vit-large-patch14' # https://huggingface.co/openai/clip-vit-large-patch14
DATA2VEC_VISUAL = 'data2vec-vision-base-ft1k' # https://huggingface.co/facebook/data2vec-vision-base-ft1k
VIDEOMAE_BASE = 'videomae-base' # https://huggingface.co/MCG-NJU/videomae-base
VIDEOMAE_LARGE = 'videomae-large' # https://huggingface.co/MCG-NJU/videomae-large
EVA_BASE = 'eva02_base_patch14_224'

# others
MANet = 'manet'
EMONET = 'emonet' 
RESNET50FACE = 'resnet50face'
SENET50FACE = 'senet50face'
MSCELEB = 'msceleb'
IMAGENet = 'imagenet'
VIDEOMAE_BASE_SUN1 = 'videomae-base-K400-mer2023-299'
VIDEOMAE_BASE_SUN2 = 'videomae-base-VoxCeleb2-99'
DINOV2_LARGE = 'dinov2-large'

WHOLE_IMAGE = [CLIP_VIT_BASE, CLIP_VIT_LARGE, DATA2VEC_VISUAL, VIDEOMAE_BASE, VIDEOMAE_LARGE, EVA_BASE, 
               MANet, EMONET, RESNET50FACE, SENET50FACE, MSCELEB, IMAGENet,
               VIDEOMAE_BASE_SUN1, VIDEOMAE_BASE_SUN2, DINOV2_LARGE]


################## Model Name Mapping ######################
featname_mapping = {
    'eGeMAPS': 'eGeMAPS',
    'IS09': 'IS09',
    'vggish': 'VGGish',
    'wav2vec-large-c': 'wav2vec-large',
    'data2vec-audio-base-960h': 'data2vec-base',
    'data2vec-audio-large': 'data2vec-large',
    'chinese-wav2vec2-base': 'wav2vec 2.0-base',
    'chinese-wav2vec2-large': 'wav2vec 2.0-large',
    'whisper-base': 'Whisper-base',
    'whisper-large-v2': 'Whisper-large',
    'wavlm-base': 'WavLM-base',
    'wavlm-large': 'WavLM-large',
    'chinese-hubert-base': 'HUBERT-base',
    'chinese-hubert-large': 'HUBERT-large',
    'emotion2vec': 'emotion2vec',
    'emotion2vec-finetune': 'emotion2vec (finetune)',

    'msceleb': 'ResNet-MSCeleb',
    'imagenet': 'ResNet-ImageNet',
    'emonet': 'EmoNet',
    'senet50face': 'SENet-FER2013', 
    'videomae-base': 'VideoMAE-base',
    'videomae-large': 'VideoMAE-large',
    'resnet50face': 'ResNet-FER2013',
    'eva02_base_patch14_224': 'EVA-02-base',
    'manet': 'MANet-RAFDB',
    'clip-vit-base-patch32': 'CLIP-base',
    'clip-vit-large-patch14': 'CLIP-large',
    'dinov2-large': 'DINOv2-large',
    'videomae-base-K400-mer2023-299': 'VideoMAE-base (MER2023)',
    'videomae-base-VoxCeleb2-99':     'VideoMAE-base (VoxCeleb2)',

    'albert_chinese_small': 'ALBERT-small',
    'opt-13b': 'OPT-13B',
    'chinese-xlnet-base': 'XLNet-base',
    'llama-13b-hf': 'Llama-13B',
    'moss-base-7b': 'MOSS-7B',
    'stable-vicuna-13b': 'Vicuna-13B',
    'deberta-chinese-large': 'DeBERTa-large',
    'stablelm-base-alpha-7b-v2': 'StableLM-7B',
    'Llama-2-13b-hf': 'Llama2-13B',
    'chinese-pert-base': 'PERT-base',
    'chinese-pert-large': 'PERT-large',
    'chinese-electra-180g-base': 'ELECTRA-base',
    'chinese-electra-180g-large': 'ELECTRA-large',
    'falcon-7b': 'Falcon-7B',
    'bert-base-chinese': 'BERT-base',
    'chatglm2-6b': 'ChatGLM2-6B',
    'paraphrase-multilingual-mpnet-base-v2': 'Sentence-BERT',
    'chinese-macbert-base':  'MacBERT-base',
    'chinese-macbert-large': 'MacBERT-large', # pre-error and now changed => 论文中的marbert-base 实际上是marbert-large
    'chinese-lert-base': 'LERT-base',
    'chinese-lert-large': 'LERT-large',
    'chinese-alpaca-2-13b': 'Alpaca2-13B',
    'bloom-7b1': 'BLOOM-7B',
    'chinese-roberta-wwm-ext': 'RoBERTa-base',
    'chinese-roberta-wwm-ext-large': 'RoBERTa-large',
    'Baichuan-13B-Base': 'Baichuan-13B',
}


featname_mapping_reverse = {}
for key in featname_mapping:
    value = featname_mapping[key]
    featname_mapping_reverse[value] = key


################## Audio Noise Path ######################
PATH_TO_NOISE = '/share/home/lianzheng/emotion-data/musan/audio-select'


# ################## Feature Ranking ######################
## case1: Ranking for MERBench
# AUDIO_RANK_LOW2HIGH = [
#     "eGeMAPS", "IS09", "VGGish", "wav2vec-large", "data2vec-base",
#     "wav2vec 2.0-large", "wav2vec 2.0-base", "WavLM-base", "Whisper-base",
#     "HUBERT-base", "WavLM-large", "Whisper-large", "HUBERT-large",	
# ]

# TEXT_RANK_LOW2HIGH = [
#     "OPT-13B", "ALBERT-small", "XLNet-base", "Llama-13B", "Vicuna-13B", "DeBERTa-large",
#     "StableLM-7B", "MOSS-7B", "Llama2-13B", "PERT-base", "ELECTRA-base", "Falcon-7B",
#     "ChatGLM2-6B", "MacBERT-base", "Sentence-BERT", "LERT-base", "BLOOM-7B", "RoBERTa-large", 
#     "Baichuan-13B",
# ]

# IMAGR_RANK_LOW2HIGH = [
#     "ResNet-MSCeleb", "ResNet-ImageNet", "EmoNet", "VideoMAE-base", "VideoMAE-large", "SENet-FER2013", 
#     "ResNet-FER2013", "DINOv2-large", "EVA-02-base", "CLIP-base", "MANet-RAFDB", "CLIP-large", 
# ]

## case2: Ranking for MER2024
AUDIO_RANK_LOW2HIGH = [
    'eGeMAPS', 'VGGish', 'Whisper-base', 
    'emotion2vec', 'Whisper-large', 'wav2vec 2.0-base', 'wav2vec 2.0-large', 'HUBERT-base', 'HUBERT-large', 
]

TEXT_RANK_LOW2HIGH = [
    'XLNet-base', 'ELECTRA-large', 'MOSS-7B', 'PERT-large', 'PERT-base', 'LERT-large', 'ELECTRA-base', 'LERT-base', 'RoBERTa-base', 
    'MacBERT-base', 'RoBERTa-large', 'ChatGLM2-6B', 'MacBERT-large', 'BLOOM-7B', 'Baichuan-13B', 
]

IMAGR_RANK_LOW2HIGH = [
    'VideoMAE-base', 'EmoNet', 'VideoMAE-large', 'DINOv2-large', 'SENet-FER2013', 'ResNet-FER2013', 
    'MANet-RAFDB', 'EVA-02-base', 'CLIP-base', 'VideoMAE-base (VoxCeleb2)', 'VideoMAE-base (MER2023)', 'CLIP-large', 
]
