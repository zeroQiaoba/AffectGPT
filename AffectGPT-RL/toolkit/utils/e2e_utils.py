import os
from toolkit.globals import *

from transformers import GPT2Model, AutoModel
from transformers import GPT2Tokenizer, BertTokenizer, AutoTokenizer
from transformers import AutoFeatureExtractor, Wav2Vec2FeatureExtractor, AutoModelForCausalLM


def load_e2e_pretrain_processor(model_name):
    model_dir  = os.path.join(config.PATH_TO_PRETRAINED_MODELS, f'transformers/{model_name}')
    ## for audio
    if model_name in WHOLE_AUDIO:
        processor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
    ## for text
    elif model_name in [DEBERTA_LARGE_CHINESE, ALBERT_TINY_CHINESE, ALBERT_SMALL_CHINESE]:
        processor = BertTokenizer.from_pretrained(model_dir, use_fast=False)
    elif model_name in [WENZHONG_GPT2_CHINESE]:
        processor = GPT2Tokenizer.from_pretrained(model_dir, use_fast=False)
    elif model_name in [LLAMA_7B, LLAMA_13B, LLAMA2_7B, VICUNA_7B, VICUNA_13B, ALPACE_13B, OPT_13B, BLOOM_7B]:
        processor = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    elif model_name in [LLAMA2_13B]:
        processor = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    elif model_name in [CHATGLM2_6B, MOSS_7B]:
        processor = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
    elif model_name in [BAICHUAN_7B, BAICHUAN_13B, BAICHUAN2_7B, BAICHUAN2_13B]:
        processor = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
    elif model_name in [STABLEML_7B]:
        processor = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    elif model_name in WHOLE_TEXT:
        processor = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    ## for visual
    elif model_name in WHOLE_IMAGE:
        processor  = AutoFeatureExtractor.from_pretrained(model_dir)
    else:
        print (f'model_name has not been merged in e2e training!!')
    return processor

def load_e2e_pretrain_model(model_name):
    model_dir = os.path.join(config.PATH_TO_PRETRAINED_MODELS, f'transformers/{model_name}')
    ## for audio
    if model_name in WHOLE_AUDIO:
        model = AutoModel.from_pretrained(model_dir)
    ## for text
    elif model_name in [DEBERTA_LARGE_CHINESE, ALBERT_TINY_CHINESE, ALBERT_SMALL_CHINESE]:
        model = AutoModel.from_pretrained(model_dir)
    elif model_name in [WENZHONG_GPT2_CHINESE]:
        model = GPT2Model.from_pretrained(model_dir)
    elif model_name in [LLAMA_7B, LLAMA_13B, LLAMA2_7B, VICUNA_7B, VICUNA_13B, ALPACE_13B, OPT_13B, BLOOM_7B]:
        model = AutoModel.from_pretrained(model_dir)
    elif model_name in [LLAMA2_13B]:
        model = AutoModel.from_pretrained(model_dir, use_safetensors=False)
    elif model_name in [CHATGLM2_6B, MOSS_7B]:
        model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
    elif model_name in [BAICHUAN_7B, BAICHUAN_13B, BAICHUAN2_7B, BAICHUAN2_13B]:
        model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
    elif model_name in [STABLEML_7B]:
        model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
    elif model_name in WHOLE_TEXT:
        model = AutoModel.from_pretrained(model_dir)
    ## for visual
    elif model_name in WHOLE_IMAGE:
        model = AutoModel.from_pretrained(model_dir)
    else:
        print (f'model_name has not been merged in e2e training!!')
    return model

## test support language
def test_and_verify_support_language():
    sentence_chi = '今天天气真好！'
    sentence_eng = 'The whether is good!'
    
    for model_name in WHOLE_TEXT:
        print (f'====== {model_name} ======')
        try:
            tokenizer = load_e2e_pretrain_processor(model_name)
            print (sentence_chi)
            inputs = tokenizer(sentence_chi, return_tensors='pt')
            print (tokenizer.decode(inputs['input_ids'][0]))
            print (sentence_eng)
            inputs = tokenizer(sentence_eng, return_tensors='pt')
            print (tokenizer.decode(inputs['input_ids'][0]))
        except:
            print (f'{model_name} processing has errors!')


# run -d toolkit/utils/e2e_utils.py
if __name__ == '__main__':
    # 可以在 nlpr 上也测试看看，从而分析支持的 language
    test_and_verify_support_language()
    # bert-base-chinese: only chinese
    # bert-base-uncased: only English
