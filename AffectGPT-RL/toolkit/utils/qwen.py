
import os
import cv2
import math
import time
import tqdm
import glob
import base64
import numpy as np


# ====================================================== # 
############          模型基本调用策略         ############
# ====================================================== # 
def func_postprocess_qwen(response):
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


# 采用qwen完成接口调用：同时支持 prompt 或者 prompt_list
def get_completion_qwen(model, tokenizer, prompt):
    
    assert isinstance(prompt, str)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    response = func_postprocess_qwen(response)
    # print(f"Prompt: {prompt} \n Response: {response}")
    return response


# 依赖于 vllm 
def get_completion_qwen_bacth(llm, sampling_params, tokenizer, prompt_list):
    
    assert isinstance(prompt_list, list)

    message_batch = []
    for prompt in prompt_list:
        message_batch.append([{"role": "user", "content": prompt}])

    text_batch = tokenizer.apply_chat_template(
        message_batch,
        tokenize=False,
        add_generation_prompt=True,
    )

    outputs = llm.generate(text_batch, sampling_params)
    
    # => batch_responses
    batch_responses = []
    for output in outputs:
        prompt = output.prompt
        response = output.outputs[0].text
        response = func_postprocess_qwen(response)
        batch_responses.append(response)
        # print(f"Prompt: {prompt} \n Response: {response}")
    return batch_responses



# ========================= # 
##      基本操作：翻译      ##
# ========================= # 
def translate_chi2eng_qwen(model=None, tokenizer=None, llm=None, sampling_params=None, reason=None, batch_reasons=None):
    
    def func_prompt_template(reason):
        prompt = f"""Please translate the Chinese input into English. Please ensure the translated results does not contain any Chinese words.
Input: 高兴; Output: happy \
Input: 生气; Output: angry \
Input: {reason}; Output: """
        return prompt

    ## process for reason
    if reason is not None:
        prompt = func_prompt_template(reason)
        response = get_completion_qwen(model, tokenizer, prompt)
        return response

    ## process for reason_list
    if batch_reasons is not None:
        prompt_list = []
        for reason in batch_reasons:
            prompt = func_prompt_template(reason)
            prompt_list.append(prompt)
        response_list = get_completion_qwen_bacth(llm, sampling_params, tokenizer, prompt_list)
        return response_list


def translate_eng2chi_qwen(model=None, tokenizer=None, llm=None, sampling_params=None, reason=None, batch_reasons=None):
    
    def func_prompt_template(reason):
        prompt = f"""Please translate the English input into Chinese.
Input: happy; Output: 高兴 \
Input: angry; Output: 生气 \
Input: {reason}; Output: """
        return prompt

    ## process for reason
    if reason is not None:
        prompt = func_prompt_template(reason)
        response = get_completion_qwen(model, tokenizer, prompt)
        return response

    ## process for reason_list
    if batch_reasons is not None:
        prompt_list = []
        for reason in batch_reasons:
            prompt = func_prompt_template(reason)
            prompt_list.append(prompt)
        response_list = get_completion_qwen_bacth(llm, sampling_params, tokenizer, prompt_list)
        return response_list




# ========================== # 
##      reason merging      ##
# ========================== # 
def reason_merge_qwen(model=None, tokenizer=None, llm=None, sampling_params=None, 
                      reason=None, subtitle=None, batch_reasons=None, batch_subtitles=None):
    
    def func_prompt_template(reason, subtitle):
        
        assert subtitle != "", 'Error: subtitle cannot be empty.'
        
        if reason != '':
            reason_merge = ""
            reason_merge += f"Clue: {reason}；"
            reason_merge += f"Subtitle: {subtitle}"
            prompt = f"Please assume the role of an expert in the field of emotions. \
    We have provided clues from the video that may be related to the characters' emotional states. \
    In addition, we have also provided the subtitle content of the video. \
    Please merge all these information to infer the emotional states of the characters, and provide reasoning for your inferences. \
    Input: {reason_merge}\
    Output:"
        else:
            reason_merge = ""
            reason_merge += f"Subtitle: {subtitle}"
            prompt = f"Please assume the role of an expert in the field of emotions.\
    We have provided the subtitle content of the video.\
    Please infer the emotional states of the characters, and provide reasoning process for your inferences.\
    Input: {reason_merge}\
    Output:"
        return prompt

    ## process for reason
    if reason is not None:
        prompt = func_prompt_template(reason, subtitle)
        response = get_completion_qwen(model, tokenizer, prompt)
        return response

    ## process for reason_list
    if batch_reasons is not None:
        prompt_list = []
        for reason, subtitle in zip(batch_reasons, batch_subtitles):
            prompt = func_prompt_template(reason, subtitle)
            prompt_list.append(prompt)
        response_list = get_completion_qwen_bacth(llm, sampling_params, tokenizer, prompt_list)
        return response_list



############################################################################
############################################################################
############################################################################
## 后面这些都是跟标签和评价相关的部分

# ====================================================== # 
##      reason -> (onehot, rank, openset, valence)      ##
# ====================================================== # 
def reason_to_onehot_qwen(model=None, tokenizer=None, llm=None, sampling_params=None, reason=None, batch_reasons=None):
    
    # 1. 给 few-shot 的结果看起来更合理一些 => 至少格式看着正确一些
    # 2. 增加一个相对复杂的shot看看结果
    # 3. 再次强调看看呢？ => 依旧是很多输出 mix 的结果，说明模型的指令追随能力一般般
    def func_prompt_template(reason):
        prompt = f"""Please act as an expert in the field of emotions. \
We provide clues that related to the character's emotions. Based on the provided clues, please identify the emotional states of the main character. \
The main character is the one with the most detailed clues. \
Please select one of the following emotion labels that best matches the given clues: [happy, angry, worried, sad, surprise, neutral]. \
We would like to emphasize that please must only output one label from the above candidates: [happy, angry, worried, sad, surprise, neutral]. You cannot output label outside these candidates, like mixed, happiness. \
Input: We cannot recognize his emotional state; Output: neutral \
Input: His emotional state is joyful, happiness, anger; Output: happy \
Input: While the woman in the video appears to be in a positive emotional state, the audio suggests that the speaker might be experiencing anxiety or nervousness, particularly when discussing the shopping card; Output: worried \
Input: The character likely experiences a range of positive emotions including excitement, enthusiasm, and confidence. They might feel motivated and inspired by the topic they are discussing, demonstrating a high level of engagement and investment; Output: happy \
Input: {reason}; Output: """
        return prompt

    # 标签后处理: 删除结尾处的 “句号”
    def func_onehot_label_polish(onehot):
        onehot = onehot.split('.')[0]
        return onehot

    ## process for reason
    if reason is not None:
        prompt = func_prompt_template(reason)
        response = get_completion_qwen(model, tokenizer, prompt)
        response = func_onehot_label_polish(response)
        return response

    ## process for reason_list
    if batch_reasons is not None:
        prompt_list = []
        for reason in batch_reasons:
            prompt = func_prompt_template(reason)
            prompt_list.append(prompt)
        response_list = get_completion_qwen_bacth(llm, sampling_params, tokenizer, prompt_list)
        for ii, response in enumerate(response_list):
            response_list[ii] = func_onehot_label_polish(response)
        return response_list

def reason_to_rank_qwen(model=None, tokenizer=None, llm=None, sampling_params=None, reason=None, batch_reasons=None):
    
    def func_prompt_template(reason):
        prompt = f"""Please assume the role of an expert in the emotional domain. We provide clues that may be related to the emotions of the character. \
            Based on the provided clues, identify the emotional states of the main character. \
            We provide a set of emotional candidates, please rank them in order of likelihood from high to low. \
            The candidate set is [happy, angry, worried, sad, surprise, neutral]. Please directly output the ranking results. \
            Input: {reason}; Output: """
        return prompt

    ## process for reason
    if reason is not None:
        prompt = func_prompt_template(reason)
        response = get_completion_qwen(model, tokenizer, prompt)
        return response

    ## process for reason_list
    if batch_reasons is not None:
        prompt_list = []
        for reason in batch_reasons:
            prompt = func_prompt_template(reason)
            prompt_list.append(prompt)
        response_list = get_completion_qwen_bacth(llm, sampling_params, tokenizer, prompt_list)
        return response_list


def reason_to_openset_qwen(model=None, tokenizer=None, llm=None, sampling_params=None, reason=None, batch_reasons=None):
    
    def func_prompt_template(reason):
        prompt = f"""Please assume the role of an expert in the field of emotions. \
We provide clues that may be related to the emotions of the characters. Based on the provided clues, please identify the emotional states of the main character. \
The main character is the one with the most detailed clues. \
Please separate different emotional categories with commas and output only the clearly identifiable emotional categories in a list format. \
Please remember that you can only ouput at most four labels for each input. \
If none are identified, please output an empty list. \
Input: We cannot recognize his emotional state; Output: [] \
Input: His emotional state is happy, sad, and angry; Output: [happy, sad, angry] \
Input: {reason}; Output: """
        return prompt

    ## process for reason
    if reason is not None:
        prompt = func_prompt_template(reason)
        response = get_completion_qwen(model, tokenizer, prompt)
        return response

    ## process for reason_list
    if batch_reasons is not None:
        prompt_list = []
        for reason in batch_reasons:
            prompt = func_prompt_template(reason)
            prompt_list.append(prompt)
        response_list = get_completion_qwen_bacth(llm, sampling_params, tokenizer, prompt_list)
        return response_list



def reason_to_valence_qwen(model=None, tokenizer=None, llm=None, sampling_params=None, reason=None, batch_reasons=None):
    
    def func_prompt_template(reason):
        prompt = f"""Please identify the overall positive or negative emotional polarity of the main characters.  \
The output should be a ﬂoating-point number ranging from -1 to 1.  \
Here, -1 indicates extremely negative emotions, 0 indicates neutral emotions, and 1 indicates extremely positive emotions.  \
Please provide your judgment as a ﬂoating-point number.  \
Input: I am very happy; Output: 1  \
Input: I am very angry; Output: -1 \
Input: I am neutral; Output: 0 \
Input: {reason}; Output: """
        return prompt

    ## process for reason
    if reason is not None:
        prompt = func_prompt_template(reason)
        response = get_completion_qwen(model, tokenizer, prompt)
        return response

    ## process for reason_list
    if batch_reasons is not None:
        prompt_list = []
        for reason in batch_reasons:
            prompt = func_prompt_template(reason)
            prompt_list.append(prompt)
        response_list = get_completion_qwen_bacth(llm, sampling_params, tokenizer, prompt_list)
        return response_list


# 目标：判断一个输入的内容是否包含相应的线索描述
def reason_to_withcontent_qwen(model=None, tokenizer=None, llm=None, sampling_params=None, reason=None, batch_reasons=None):
    
    def func_prompt_template(reason):
        prompt = f"""Please identify whether a given description contain informative clues. The output should be either True or False.\
Input: the tone is low and seems to be accompanied by a sob; Output: True; \
Input: In the audio, the voice of this character is calm and soothing, with a gentle tone; Output: True; \
Input: Her mouth moves slightly, indicating that she is seemingly having a conversation with someone on the right side of the frame, who is not visible. The specific emotions need to be analyzed in conjunction with additional context; Output: True; \
Input: It is impossible to determine the emotional state; Output: False; \
Input: In the audio, ... ; Output: False; \
Input: {reason}; Output: """
        return prompt

    ## process for reason
    if reason is not None:
        prompt = func_prompt_template(reason)
        response = get_completion_qwen(model, tokenizer, prompt)
        return response

    ## process for reason_list
    if batch_reasons is not None:
        prompt_list = []
        for reason in batch_reasons:
            prompt = func_prompt_template(reason)
            prompt_list.append(prompt)
        response_list = get_completion_qwen_bacth(llm, sampling_params, tokenizer, prompt_list)
        # post-polish
        for ii, response in enumerate(response_list):
            assert response.startswith('True') or response.startswith('False')
            if response.startswith('True'):
                response_list[ii] = True
            elif response.startswith('False'):
                response_list[ii] = False
        return response_list


# ========================================== # 
##      openset -> (onehot, sentiment)      ##
# ========================================== # 
def openset_to_onehot_qwen(model=None, tokenizer=None, llm=None, sampling_params=None, reason=None, batch_reasons=None):
    def func_prompt_template(reason):
        prompt = f"""Please act as an expert in the field of emotions. \
            We provide a few words to describe the emotions of a character. \
            Please choose the emotion label from the following list that is closest to the given words: happy, angry, worried, sad, surprise, neutral.
            Input: [joyful]; Output: happy \
            Input: []; Output: neutral \
            Input: {reason}; Output: """
        return prompt

    ## process for reason
    if reason is not None:
        prompt = func_prompt_template(reason)
        response = get_completion_qwen(model, tokenizer, prompt)
        return response

    ## process for reason_list
    if batch_reasons is not None:
        prompt_list = []
        for reason in batch_reasons:
            prompt = func_prompt_template(reason)
            prompt_list.append(prompt)
        response_list = get_completion_qwen_bacth(llm, sampling_params, tokenizer, prompt_list)
        return response_list

def openset_to_sentiment_qwen(model=None, tokenizer=None, llm=None, sampling_params=None, reason=None, batch_reasons=None):
    def func_prompt_template(reason):
        prompt = f"""Please act as an expert in the field of emotions. \
            We provide a few words to describe the emotions of a character. \
            Please choose the most likely sentiment from the given candidates: [positive, negative, neutral] \
            Please direct output answer without analyzing process. \
            Input: [joyful]; Output: positive \
            Input: []; Output: neutral \
            Input: {reason}; Output: """
        return prompt

    ## process for reason
    if reason is not None:
        prompt = func_prompt_template(reason)
        response = get_completion_qwen(model, tokenizer, prompt)
        return response

    ## process for reason_list
    if batch_reasons is not None:
        prompt_list = []
        for reason in batch_reasons:
            prompt = func_prompt_template(reason)
            prompt_list.append(prompt)
        response_list = get_completion_qwen_bacth(llm, sampling_params, tokenizer, prompt_list)
        return response_list


