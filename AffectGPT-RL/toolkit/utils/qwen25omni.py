# ============================================================================= # 
## Qwen25-Omni 仅用于 thinking rank 的作用 (所有函数名都用 aaa_qwen25omni 以示区分)
# ============================================================================= # 
import os
import re
import cv2
import math
import time
import tqdm
import glob
import base64
import numpy as np
from scipy.optimize import minimize
from qwen_omni_utils import process_mm_info

import warnings
warnings.filterwarnings('ignore') # 不显示 warning

# ====================================================== # 
############          模型基本调用策略         ############
# ====================================================== # 
def func_postprocess_qwen25omni(response):
    response = response.strip()
    if response.startswith("输入"):   response = response[len("输入"):]
    if response.startswith("输出"):   response = response[len("输出"):]
    if response.startswith("翻译"):   response = response[len("翻译"):]
    if response.startswith("让我们来翻译一下："): response = response[len("让我们来翻译一下："):]
    if response.startswith("output"): response = response[len("output"):]
    if response.startswith("Output"): response = response[len("Output"):]
    if response.startswith("input"): response = response[len("input"):]
    if response.startswith("Input"): response = response[len("Input"):]
    response = response.strip()
    if response.startswith(":"):  response = response[len(":"):]
    if response.startswith("："): response = response[len("："):]
    response = response.strip()
    response = response.replace('\n', '') # remove \n
    response = response.strip()
    return response


# 调用 qwen25omni 完成以下任务
def get_completion_qwen25omni(model, tokenizer, prompt, video_path, input_type):
    
    assert input_type in ['text', 'audiovideo']
    assert isinstance(prompt, str)

    if input_type == 'text':
        message = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        USE_AUDIO_IN_VIDEO = True
        
    elif input_type == 'audiovideo':
        message = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        USE_AUDIO_IN_VIDEO = True

    text = tokenizer.apply_chat_template([message], add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info([message], use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = tokenizer(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = inputs.to(model.device).to(model.dtype)
    generated_ids = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ] # 提出 out_ids 部分内容
    batch_responses = tokenizer.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    ) # 解码出文字
    response = batch_responses[0]
    response = func_postprocess_qwen25omni(response)
    return response




# ================================================================= # 
#      依据 (answer1, answer2, preference) 获取 answer list 排名     #
# ================================================================= # 
def func_postprocess_preference_qwen25omni(response):
    response = response.lower()
    if response.startswith('a1'):
        return 'a1'
    elif response.startswith('a2'):
        return 'a2'
    elif response.startswith('same'):
        return 'same'
    ## 其他不规范的格式输出: [a1, a2, same] 哪个在最前面，就输出哪个作为 answer
    elif response.find('a1') != -1 or response.find('a2') != -1 or response.find('same') != -1:
        keys  = ['a1', 'a2', 'same']
        poses = [response.find('a1'), response.find('a2'), response.find('same')]
        poses = [10000 if item == -1 else item for item in poses]
        return keys[np.argmin(poses)]
    else: # 其他无法判断的情况
        return 'same'

def get_modellists_qwen25omni(pairwise_compare):
    model_lists = []
    for model1 in pairwise_compare:
        for model2 in pairwise_compare[model1]:
            model_lists.extend([model1, model2])
    model_lists = sorted(list(set(model_lists)))
    return model_lists


def Bradley_Terry_with_ties_update_softer_qwen25omni(pairwise_compare):
    beta = 1.0 # 平局参数，可以设为可调参数（后续可扩展）

    model_lists = get_modellists_qwen25omni(pairwise_compare)
    n_models = len(model_lists) # 待比较的模型数量

    def negative_log_likelihood(lambdas):
        lambdas = np.maximum(lambdas, 1e-10)  # 避免数值问题（除零或负数）
        ll = 0.0

        for i in range(n_models):
            for j in range(i+1, n_models):
                model_i = model_lists[i]
                model_j = model_lists[j]
                win_num = pairwise_compare[model_i][model_j]['win']
                lose_num= pairwise_compare[model_i][model_j]['lose']
                tie_num = pairwise_compare[model_i][model_j]['tie']

                if win_num > 0:
                    denom = lambdas[i] + lambdas[j] + beta
                    prob = lambdas[i] / denom
                    ll += win_num * np.log(prob)

                if lose_num > 0:
                    denom = lambdas[i] + lambdas[j] + beta
                    prob = lambdas[j] / denom
                    ll += lose_num * np.log(prob)

                if tie_num > 0:
                    denom = lambdas[i] + lambdas[j] + beta
                    prob = beta / denom
                    ll += tie_num * np.log(prob)

        return -ll  # 返回负对数似然（因为我们要最小化）

    x0 = np.ones(n_models) # 初始值：所有 lambda = 1.0
    result = minimize(
        fun=negative_log_likelihood,
        x0=x0,
        method='L-BFGS-B',  # 或 'BFGS', 'Nelder-Mead'
        tol=1e-6
    )
    estimated_lambdas = result.x
    return estimated_lambdas


def extract_think_ranks_samplecalling_qwen25omni(model, tokenizer, thinks, video_path):
    pairwise_compare = {}
    for ii in range(len(thinks)):
        for jj in range(ii+1, len(thinks)):
            model1 = f"model{ii}"
            model2 = f"model{jj}"
            think_ii = thinks[ii]
            think_jj = thinks[jj]
            prompt = f"""We provide two descriptions for a given input: \
a1: \"{think_ii}\". \
a2: \"{think_jj}\". \
Please determinate which one is better aligned with the input content. \
If both of them equally align with the input content, please output 'same'. \
Therefore, the output should be a1, a2, or same. Please direct output the answer without additional reasoning process."""
            response = get_completion_qwen25omni(model, tokenizer, prompt, video_path, 'audiovideo')
            preference = func_postprocess_preference_qwen25omni(response)

            # store (model1, model2, preference)
            if model1 not in pairwise_compare:
                pairwise_compare[model1] = {}
            if model2 not in pairwise_compare:
                pairwise_compare[model2] = {}
            if model2 not in pairwise_compare[model1]:
                pairwise_compare[model1][model2] = {'win':0, 'lose':0, 'tie': 0}
            if model1 not in pairwise_compare[model2]:
                pairwise_compare[model2][model1] = {'win':0, 'lose':0, 'tie': 0}
            
            if preference == 'a1':
                pairwise_compare[model1][model2]['win']   += 1
                pairwise_compare[model2][model1]['lose']  += 1
            elif preference == 'a2':
                pairwise_compare[model1][model2]['lose']  += 1
                pairwise_compare[model2][model1]['win']   += 1
            elif preference == 'same':
                pairwise_compare[model1][model2]['tie'] += 1
                pairwise_compare[model2][model1]['tie'] += 1

    theta_estimated = Bradley_Terry_with_ties_update_softer_qwen25omni(pairwise_compare)
    index_sorted = np.argsort(-theta_estimated)

    return index_sorted




# ============================== # 
#     依据 reason 获取 openset    #
# ============================== # 
def get_openset_qwen25omni(sentence):
    match = re.search(r'\[(.*?)\]', sentence)
    if match:
        result = match.group(1)
        return result
    else:
        return ""
        
def extract_reason_to_openset_qwen25omni(model, tokenizer, reason):
    
    prompt = f"""Please assume the role of an expert in the field of emotions. \
We provide clues that may be related to the emotions of the characters. Based on the provided clues, please identify the emotional states of the main character. \
The main character is the one with the most detailed clues. \
Please separate different emotional categories with commas and output only the clearly identifiable emotional categories in a list format. \
Please remember that you can only ouput at most four labels for each input. \
If none are identified, please output an empty list. \
Input: We cannot recognize his emotional state; Output: [] \
Input: His emotional state is happy, sad, and angry; Output: [happy, sad, angry] \
Input: {reason}; Output: """
    response = get_completion_qwen25omni(model, tokenizer, prompt, None, 'text')
    openset = get_openset_qwen25omni(response) # 只返回 [xxx] 中间的部分，因为 qwen25omni 格式遵循能力较弱
    return openset
