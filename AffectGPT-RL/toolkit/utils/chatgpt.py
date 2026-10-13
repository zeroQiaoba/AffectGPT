
import os
import cv2
import math
import time
import tqdm
import glob
import base64
import numpy as np

import openai

# avoid RPD errors
global_index = 0
candidate_keys = ["sk-xxx"]
openai.api_key = candidate_keys[global_index]

# 单次调用
def func_get_completion(prompt, model="gpt-3.5-turbo-16k-0613"):
    try:
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0, # this is the degree of randomness
            max_tokens=1000,
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print ('发生错误：', e) # change key to avoid RPD
        global global_index # 修改全局变量
        global_index = (global_index + 1) % len(candidate_keys)
        print (f'========== key index: {global_index} ==========')
        openai.api_key = candidate_keys[global_index]
        return ''

# 多次调用，避免网络异常
def get_completion(prompt, model, maxtry=5):
    response = ''
    try_number = 0
    while len(response) == 0:
        try_number += 1
        if try_number == maxtry: 
            print (f'fail for {maxtry} times')
            break
        response = func_get_completion(prompt, model)
    return response

# chatgpt输出结果后处理
def func_postprocess_chatgpt(response):
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

# ---------------------------------------------------------------------
## convert image/video into GPT4 support version
def func_image_to_base64(image_path, grey_flag=False): # support more types
    image = cv2.imread(image_path)
    if grey_flag:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return func_opencv_to_base64(image)

def func_opencv_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    return base64_image

# deal with text
def func_nyp_to_text(npy_path):
    text = np.load(npy_path).tolist()
    text = text.strip()
    text = text.replace('\n', '') # remove \n
    text = text.replace('\t', '') # remove \t
    text = text.strip()
    return text

# support video or uniform sampled frames
# outputs: (nframe<=4, h, w, c)
def sample_frames_from_video(video_path, samplenum=3):
    if os.path.isdir(video_path):
        select_frames = sorted(glob.glob(video_path + '/*'))
        select_frames = select_frames[:samplenum]
        select_frames = [cv2.imread(item) for item in select_frames]
    else: # is video
        # read frames from videos
        frames = []
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if ret == False: break
            frames.append(frame)
        cap.release()
        while len(frames) < samplenum: # ensure large than samplenum
            frames.append(frames[-1])
        print (f'frame numbers: {len(frames)}')
        
        tgt_length = int(len(frames)/samplenum)*samplenum
        frames = frames[:tgt_length]
        indices = np.arange(0, len(frames), int(len(frames) / samplenum)).astype(int).tolist()
        print ('sample indexes: ', indices)
        assert len(indices) == samplenum
        select_frames = [frames[index] for index in indices]

    assert len(select_frames) == samplenum
    return select_frames

# # !! only support video inputs
# # => does not have much difference compared to 'sample stategy v1'
# def sample_frames_from_video_v2(video_path, samplenum=3):
#     print ('sample stategy v2')
#     # read frames from videos
#     frames = []
#     cap = cv2.VideoCapture(video_path)
#     while True:
#         ret, frame = cap.read()
#         if ret == False: break
#         frames.append(frame)
#     cap.release()

#     # sample frames
#     frames = np.array(frames)
#     vlen = len(frames)
#     indices = np.linspace(0, vlen-1, samplenum).astype(int).tolist() # 包含首位均匀采样
#     print (f'raw frames: {vlen}, samplenum: {samplenum}, sampled index: {indices}')
#     return frames[indices]


# ---------------------------------------------------------------------
## Translation
# ---------------------------------------------------------------------
def get_translate_eng2chi(text, model='gpt-3.5-turbo-16k-0613'):
    if len(text) == 0:
        return ""
    text = text.replace('\n', '')
    prompt = f"""
              请将以下输入翻译为中文：
              
              输入：{text}

              输出：
              """
    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    print (text)
    print (response)
    return response


def get_translate_chi2eng(text, model='gpt-3.5-turbo-16k-0613'):
    if len(text)==0:
        return ""
    text = text.replace('\n', '')
    prompt = f"""
              请将以下输入翻译为英文：
              
              输入：我爱你

              输出：I love you

              输入：{text}

              输出：
              """
    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    print (text)
    print (response)
    return response


# ---------------------------------------------------------------------
## Caption
# ---------------------------------------------------------------------
def get_image_caption_per(image_path, model='gpt-4-vision-preview'):

    prompt = [
                {
                    "type": "text", 
                    "text": "Please provide a detailed description of this image."},
                {
                    "type": "image",
                    "image": func_image_to_base64(image_path),
                },
            ]

    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    print (response)
    return response

def get_image_caption_batch(image_paths, model='gpt-4-vision-preview'):

    prompt = [
                {
                    "type": "text", 
                    "text": "Please play the role of an image description expert. We provide multiple images. \
                             Please provide a detailed description of these images. The output format should be {'name':, 'result':} for each image."
                },
            ]
    for ii, image_path in enumerate(image_paths):
        prompt.append(
            {
                "type":  f"image-{ii+1}",
                "image": func_image_to_base64(image_path),
            }
        )
    # for item in prompt: print (item['type']) # debug

    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    print (response)
    return response

def get_video_caption_per(video_path, model='gpt-4-vision-preview'):

    samplenum = 3
    frames = sample_frames_from_video(video_path, samplenum = samplenum)
    
    prompt = [
                {
                    "type": "text", 
                    "text": f"Please play the role of a video description expert. We provide {samplenum} uniformly sampled frames for this video. \
                             These frames are sorted chronologically. Please consider the temporal relationship between these frames and provide a detailed description for this video. \
                             In the description, please ignore the speaker's identity and focus on the video content. \
                             The output format should be {'name':, 'result':}."
                },
                {
                    "type": "image1",
                    "image": func_opencv_to_base64(frames[0]),
                },
                {
                    "type": "image2",
                    "image": func_opencv_to_base64(frames[1]),
                },
                {
                    "type": "image3",
                    "image": func_opencv_to_base64(frames[2]),
                },
                {
                    "type": "image4",
                    "image": func_opencv_to_base64(frames[3]),
                },
            ]
    
    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    print (response)
    return response


def get_video_caption_batch(video_paths, model='gpt-4-vision-preview'):

    prompt = [
                {
                    "type": "text", 
                    "text": "Please play the role of a video description expert. We provide multiple videos, each with four temporally uniformly sampled frames.\
                             These frames are sorted chronologically. Please consider the temporal relationship between these frames and provide a detailed description for each video. \
                             In the description, please ignore the speaker's identity and focus on the video content. \
                             The output format of each video should be {'name':, 'result':}."
                }
            ]
    
    for ii, video_path in enumerate(video_paths):
        video_frames = sample_frames_from_video(video_path, samplenum=4)
        for jj, video_frame in enumerate(video_frames):
            prompt.append(
                    {
                        "type":  f"video{ii+1}_image{jj+1}",
                        "image": func_opencv_to_base64(video_frame),
                    },
            )
    # for item in prompt: print (item['type']) # debug

    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    print (response)
    return response


# ---------------------------------------------------------------------
## Emotion
# ---------------------------------------------------------------------
# 20 images per time [原始]
# def get_image_emotion_batch(image_paths, candidate_list, sleeptime=0, grey_flag=False, model='gpt-4-vision-preview'):
#     prompt = [
#                 {
#                     "type":  "text", 
#                     "text": f"Please play the role of a facial expression classification expert. We provide {len(image_paths)} images. Please ignore the speaker's identity and focus on the facial expression. \
#                               For each image, please sort the provided categories from high to low according to the top 5 similarity with the input image. \
#                               Here are the optional categories: {candidate_list}. Please ignore the speaker's identity and focus on the facial expression. The output format should be {{'name':, 'result':}} for each image."
#                 }
#             ]
#     for ii, image_path in enumerate(image_paths):
#         prompt.append(
#             {
#                 "type":  f"image-{ii+1}",
#                 "image": func_image_to_base64(image_path, grey_flag),
#             }
#         )
#     print (prompt[0]['text']) # debug
#     for item in prompt: print (item['type']) # debug
#     time.sleep(sleeptime)
#     response = get_completion(prompt, model)
#     response = func_postprocess_chatgpt(response)
#     print (response)
#     return response

## 临时测试：分析不同 prompt template 的影响 [只在SFEW上分析不同template的影响] => 不同的template，所采用的metric也是不一样的
def get_image_emotion_batch(image_paths, candidate_list, sleeptime=0, template='case0', grey_flag=False, model='gpt-4-vision-preview'):
    if template == 'case0': # 原始的prompt
        prompt = [
                    {
                        "type":  "text", 
                        "text": f"Please play the role of a facial expression classification expert. We provide {len(image_paths)} images. Please ignore the speaker's identity and focus on the facial expression. \
                                For each image, please sort the provided categories from high to low according to the top 5 similarity with the input image. \
                                Here are the optional categories: {candidate_list}. Please ignore the speaker's identity and focus on the facial expression. The output format should be {{'name':, 'result':}} for each image."
                    }
                ]
    elif template == 'case1': # 删除 Please play the role of a facial expression classification expert. 分析其影响
        prompt = [
                    {
                        "type":  "text", 
                        "text": f"We provide {len(image_paths)} images. Please ignore the speaker's identity and focus on the facial expression. \
                                For each image, please sort the provided categories from high to low according to the top 5 similarity with the input image. \
                                Here are the optional categories: {candidate_list}. Please ignore the speaker's identity and focus on the facial expression. The output format should be {{'name':, 'result':}} for each image."
                    }
                ]
    elif template == 'case2': # 改成选择 best category，而不是排序
        prompt = [
                    {
                        "type":  "text", 
                        "text": f"Please play the role of a facial expression classification expert. We provide {len(image_paths)} images. Please ignore the speaker's identity and focus on the facial expression. \
                                For each image, please select the most likely category according to the correlation with the input image. \
                                Here are the optional categories: {candidate_list}. Please ignore the speaker's identity and focus on the facial expression. The output format should be {{'name':, 'result':}} for each image."
                    }
                ]
    
    for ii, image_path in enumerate(image_paths):
        prompt.append(
            {
                "type":  f"image-{ii+1}",
                "image": func_image_to_base64(image_path, grey_flag),
            }
        )
    print (prompt[0]['text']) # debug
    for item in prompt: print (item['type']) # debug
    time.sleep(sleeptime)
    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    print (response)
    return response


def get_evoke_emotion_batch(image_paths, candidate_list, sleeptime=0, model='gpt-4-vision-preview'):
    prompt = [
                {
                    "type":  "text", 
                    "text": f"Please play the role of a emotion recognition expert. We provide {len(image_paths)} images. \
                              Please recognize sentiments evoked by these images (i.e., guess how viewer might emotionally feel after seeing these images.) \
                              If there is a person in the image, ignore that person's identity. \
                              For each image, please sort the provided categories from high to low according to the similarity with the input image. \
                              Here are the optional categories: {candidate_list}. If there is a person in the image, ignore that person's identity. \
                              The output format should be {{'name':, 'result':}} for each image."
                }
            ]
    for ii, image_path in enumerate(image_paths):
        prompt.append(
            {
                "type":  f"image-{ii+1}",
                "image": func_image_to_base64(image_path),
            }
        )
    print (prompt[0]['text']) # debug
    for item in prompt: print (item['type']) # debug
    time.sleep(sleeptime)
    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    print (response)
    return response

def get_micro_emotion_batch(image_paths, candidate_list, sleeptime=0, model='gpt-4-vision-preview'):
    prompt = [
                {
                    "type":  "text", 
                    "text": f"Please play the role of a micro-expression recognition expert. We provide {len(image_paths)} images. Please ignore the speaker's identity and focus on the facial expression. \
                              For each image, please sort the provided categories from high to low according to the similarity with the input image. \
                              The expression may not be obvious, please pay attention to the details of the face. \
                              Here are the optional categories: {candidate_list}. Please ignore the speaker's identity and focus on the facial expression. The output format should be {{'name':, 'result':}} for each image."
                }
            ]
    for ii, image_path in enumerate(image_paths):
        prompt.append(
            {
                "type":  f"image-{ii+1}",
                "image": func_image_to_base64(image_path),
            }
        )
    print (prompt[0]['text']) # debug
    for item in prompt: print (item['type']) # debug
    time.sleep(sleeptime)
    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    print (response)
    return response

# 20 images per time
def get_audio_emotion_batch(image_paths, candidate_list, sleeptime=0, model='gpt-4-vision-preview'):
    prompt = [
                {
                    "type":  "text", 
                    "text": f"Please play the role of a audio expression classification expert. We provide {len(image_paths)} audios, each with an image of Mel spectrogram. \
                              Please ignore the speaker's identity and recognize the speaker's expression from the provided Mel spectrogram. \
                              For each sample, please sort the provided categories from high to low according to the top 5 similarity with the input. \
                              Here are the optional categories: {candidate_list}. The output format should be {{'name':, 'result':}} for each audio."
                }
            ]
    for ii, image_path in enumerate(image_paths):
        prompt.append(
            {
                "type":  f"audio-{ii+1}",
                "image": func_image_to_base64(image_path),
            }
        )
    print (prompt[0]['text']) # debug
    for item in prompt: print (item['type']) # debug
    time.sleep(sleeptime)
    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    print (response)
    return response


def get_text_emotion_batch(npy_paths, candidate_list, sleeptime=0, model='gpt-4-vision-preview'):
    prompt = [
                {
                    "type":  "text", 
                    "text": f"Please play the role of a textual emotion classification expert. We provide {len(npy_paths)} texts. \
                              Please recognize the speaker's expression from the provided text. \
                              For each text, please sort the provided categories from high to low according to the top 5 similarity with the input. \
                              Here are the optional categories: {candidate_list}. The output format should be {{'name':, 'result':}} for each text."
                }
            ]
    for ii, npy_path in enumerate(npy_paths):
        prompt.append(
            {
                "type":  f"text",
                "text": f"{func_nyp_to_text(npy_path)}",
            }
        )
    print (prompt[0]['text']) # debug
    for item in prompt: print (item['type']) # debug
    time.sleep(sleeptime)
    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    print (response)
    return response


# 20 images per time
def get_video_emotion_batch(video_paths, candidate_list, sleeptime=0, samplenum=3, model='gpt-4-vision-preview'):
    prompt = [
                {
                    "type":  "text", 
                    "text": f"Please play the role of a video expression classification expert. We provide {len(video_paths)} videos, each with {samplenum} temporally uniformly sampled frames. Please ignore the speaker's identity and focus on their facial expression. \
                              For each video, please sort the provided categories from high to low according to the top 5 similarity with the input video. \
                              Here are the optional categories: {candidate_list}. Please ignore the speaker's identity and focus on the facial expression. The output format should be {{'name':, 'result':}} for each video."
                }
            ]
    
    for ii, video_path in enumerate(video_paths):
        video_frames = sample_frames_from_video(video_path, samplenum)  # 支持图像文件夹和视频输入
        # video_frames = sample_frames_from_video_v2(video_path, samplenum) # 只支持视频输入 [和上面的没有本质差别]
        for jj, video_frame in enumerate(video_frames):
            prompt.append(
                    {
                        "type":  f"video{ii+1}_image{jj+1}",
                        "image": func_opencv_to_base64(video_frame),
                    },
            )
    print (prompt[0]['text']) # debug
    for item in prompt: print (item['type']) # debug
    time.sleep(sleeptime)
    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    print (response)
    return response


def get_multi_emotion_batch(video_paths, candidate_list, sleeptime=0, model='gpt-4-vision-preview'):
    prompt = [
                {
                    "type":  "text", 
                    "text": f"Please play the role of a video expression classification expert. We provide {len(video_paths)} videos, each with the speaker's content and three temporally uniformly sampled frames.\
                              Please ignore the speaker's identity and focus on their emotions. Please ignore the speaker's identity and focus on their emotions. \
                              For each video, please sort the provided categories from high to low according to the top 5 similarity with the input video. \
                              Here are the optional categories: {candidate_list}. Please ignore the speaker's identity and focus on their emotions. The output format should be {{'name':, 'result':}} for each video."
                }
            ]
    
    for ii, video_path in enumerate(video_paths):
        # convert video_path to text path
        split_paths = video_path.split('/')
        split_paths[-2] = 'text'
        split_paths[-1] = split_paths[-1].rsplit('.', 1)[0] + '.npy'
        text_path = "/".join(split_paths)
        assert os.path.exists(text_path)
        prompt.append(
                {
                    "type": "text",
                    "text": f"{func_nyp_to_text(text_path)}",
                },
        )

        # read frames
        video_frames = sample_frames_from_video(video_path, samplenum=3)
        for jj, video_frame in enumerate(video_frames):
            prompt.append(
                    {
                        "type":  f"video{ii+1}_image{jj+1}",
                        "image": func_opencv_to_base64(video_frame),
                    },
            )
       
    print (prompt[0]['text']) # debug
    for item in prompt: print (item['type']) # debug
    time.sleep(sleeptime)
    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    print (response)
    return response


# ---------------------------------------------------------------------
## Image-text Sentiment Analysis on Social Media
# ---------------------------------------------------------------------
def get_social_multi_batch(video_paths, candidate_list, sleeptime=0, model='gpt-4-vision-preview'):
    prompt = [
                {
                    "type":  "text", 
                    "text": f"Please play the role of an emotion recognition expert. We provide {len(video_paths)} image-text pairs.\
                        Please analyze how he will feel if he post this image-text pair on social media. If there is a person in the image, ignore that person's identity.\
                        For each image-text pair, please sort the provided categories from high to low according to the similarity with the input image-text pair. \
                        Here are the optional categories: {candidate_list}. The output format should be {{'name':, 'result':}} for each image-text pair."
                }
            ]
    
    for ii, video_path in enumerate(video_paths):
        # convert video_path to text path
        item1, item2 = os.path.split(video_path)
        item0, item1 = os.path.split(item1)
        item1 = 'text'
        item2 = item2.rsplit('.', 1)[0] + '.npy'
        text_path = os.path.join(item0, item1, item2)
        assert os.path.exists(text_path)
        prompt.append(
                {
                    "type": "image",
                    "image": func_image_to_base64(video_path),
                },
        )
        prompt.append(
                {
                    "type": "text",
                    "text": f"{func_nyp_to_text(text_path)}",
                },
        )
    
    print (prompt[0]['text']) # debug
    for item in prompt: print (item['type']) # debug
    time.sleep(sleeptime)
    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    print (response)
    return response


def get_social_image_batch(video_paths, candidate_list, sleeptime=0, model='gpt-4-vision-preview'):
    prompt = [
                {
                    "type":  "text", 
                    "text": f"Please play the role of an emotion recognition expert. We provide {len(video_paths)} images.\
                        Please analyze how he will feel if he post this image on social media. If there is a person in the image, ignore that person's identity.\
                        For each image, please sort the provided categories from high to low according to the similarity with the input image. \
                        Here are the optional categories: {candidate_list}. The output format should be {{'name':, 'result':}} for each image."
                }
            ]
    
    for ii, video_path in enumerate(video_paths):
        prompt.append(
                {
                    "type": "image",
                    "image": func_image_to_base64(video_path),
                },
        )
    
    print (prompt[0]['text']) # debug
    for item in prompt: print (item['type']) # debug
    time.sleep(sleeptime)
    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    print (response)
    return response


def get_social_text_batch(npy_paths, candidate_list, sleeptime=0, model='gpt-4-vision-preview'):
    prompt = [
                {
                    "type":  "text", 
                    "text": f"Please play the role of an emotion recognition expert. We provide {len(npy_paths)} texts.\
                        Please analyze how he will feel if he post this text on social media.\
                        For each text, please sort the provided categories from high to low according to the similarity with the input text. \
                        Here are the optional categories: {candidate_list}. The output format should be {{'name':, 'result':}} for each text."
                }
            ]
    
    for ii, npy_path in enumerate(npy_paths):
        prompt.append(
                {
                    "type": "text",
                    "text": f"{func_nyp_to_text(npy_path)}",
                },
        )
    
    print (prompt[0]['text']) # debug
    for item in prompt: print (item['type']) # debug
    time.sleep(sleeptime)
    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    print (response)
    return response



# ---------------------------------------------------------------------
## Emotion Reasoning
# ---------------------------------------------------------------------
def get_text_reason(text, sleeptime=0, model='gpt-3.5-turbo-16k-0613'):

    prompt = [
                {
                    "type": "text", 
                    "text": f"请假设作为情感领域的专家。我们有一段文本，请分析从哪些内容中可以推测出人物的情感状态，并给出推理依据。文本内容为：{text}"
                }
            ]
    print (prompt[0]['text']) # debug
    for item in prompt: print (item['type']) # debug
    time.sleep(sleeptime)
    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    print (response)
    return response


def get_video_reason(video_path, sleeptime=0, samplenum=4, model='gpt-4-vision-preview'):

    frames = sample_frames_from_video(video_path, samplenum)
    
    if samplenum > 1:
        prompt = [
                    {
                        "type": "text",
                        "text": f"请假设作为情感领域的专家，重点关注图像中人物面部表情、肢体动作、所处环境、发生事件等和人物情感相关的线索，并进行详细描述，最终预测视频中人物的情感状态。\
                                在描述过程中，请忽略人物的身份信息。在描述过程中，请忽略人物的身份信息。在描述过程中，请忽略人物的身份信息。尽量提供可能的情感线索。\
                                我们从视频中均匀采样了{samplenum}帧，按照时间顺序排列分别为image1到image{samplenum}。\
                                描述过程中，请考虑帧之间的时序关系，并给出这段视频的完整描述。\
                                不要用第一张图片、第二张图片这种描述，而是采用开头、中间、结尾等随着时间推移的描述。"
                    }
                ]
        for ii, frame in enumerate(frames):
            prompt.append(
                {
                    "type": f"image{ii+1}",
                    "image": func_opencv_to_base64(frame),
                },
            )
    else: # samplenum=1
        prompt = [
                    {
                        "type": "text",
                        "text": f"请假设作为情感领域的专家，重点关注图像中人物面部表情、肢体动作、所处环境、发生事件等和人物情感相关的线索，并进行详细描述，最终预测人物的情感状态。\
                                在描述过程中，请忽略人物的身份信息。在描述过程中，请忽略人物的身份信息。在描述过程中，请忽略人物的身份信息。尽量提供可能的情感线索。"
                    },
                    {
                        "type": f"image",
                        "image": func_opencv_to_base64(frames[0]),
                    },
                ]

    print (prompt[0]['text']) # debug
    for item in prompt: print (item['type']) # debug
    time.sleep(sleeptime)
    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    print (response)
    return response


## 合并同类型的reason，比如多段audio reason合并
def get_merge_reason_audio(reasons, sleeptime=0, model='gpt-3.5-turbo-16k-0613'):

    reason_merge = ""
    for ii in range(len(reasons)):
        reason_merge += f"text{ii+1}:{reasons[ii]};"
    reason_merge = reason_merge[:-1]

    prompt = [
                {
                    "type": "text", 
                    "text": f"我们有{len(reasons)}段描述，有的是中文描述，有的是英文描述。\
                            请将所有英文描述转成中文，再将{len(reasons)}段描述进行合并，删除重复的表述，得到一段完整的描述。\
                            输入：{reason_merge} 输出："
                }
            ]
    print (prompt[0]['text']) # debug
    for item in prompt: print (item['type']) # debug
    time.sleep(sleeptime)
    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    print (response)
    return response


## 结合音频线索+视频线索，分析字幕中的情感原因
def update_text_reason(video_reason, audio_reason, subtitle, sleeptime=0, model='gpt-3.5-turbo-16k-0613'):

    ## case1: 有视频线索 # (audio, video, subtitle) -> text reason
    if video_reason != '' and audio_reason != '': 
        reason_merge = ""
        reason_merge += f"视频线索：{video_reason}；"
        reason_merge += f"音频线索：{audio_reason}；"
        reason_merge += f"字幕内容：{subtitle}"

        prompt = [
                    {
                        "type": "text", 
                        "text": f"请假设作为情感领域的专家。我们提供了音频、视频中可能与人物情感相关的线索。此外，我们还提供了视频原始的字幕内容。\
请分析从哪些字幕内容中可以推测出人物的情感状态，并给出推理依据。在推理过程中，请结合音频线索和视频线索进行分析。\
输入：视频线索： 在视频的开头，我们看到一位女士坐在沙发上，面对着旁边的人，她瞪大眼睛，嘴巴微张，表现的有些疑惑和惊讶。到了视频的中间部分，女士逐渐笑了起来，嘴角上扬，眼睛弯曲，表现出开心的情绪。同时，女士耸肩和低头的动作表现出了不好意思的情绪。\
音频线索： 角色先是笑了一声，之后说话时语调柔和，语气平稳。\
字幕内容： 真看不出来，你这个人还挺贫。\
输出：字幕内容：“真看不出来，你这个人还挺贫。” 这句话可能是女士对旁边人的一种评价或反应。\
根据音频线索中描述的角色语调柔和、语气平稳，以及视频线索中女士表现出的笑容和开心的情绪，我们可以推断这句话可能带有一种轻松或者幽默的语气。\
因此，这句话可能并非负面评价，而是一种调侃或者开玩笑的表达方式，与女士整体展现的积极情绪相符合。\
输入：{reason_merge}\
输出："
                    }
                ]
    
    # (audio, subtitle) -> text reason
    elif video_reason == '' and audio_reason != '':
        reason_merge = ""
        reason_merge += f"音频线索：{audio_reason}；"
        reason_merge += f"字幕内容：{subtitle}"

        prompt = [
                    {
                        "type": "text", 
                        "text": f"请假设作为情感领域的专家。我们提供了音频中可能与人物情感相关的线索。此外，我们还提供了视频原始的字幕内容。\
请分析从哪些字幕内容中可以推测出人物的情感状态，并给出推理依据。在推理过程中，请结合音频线索进行分析。\
输入：音频线索： 角色先是笑了一声，之后说话时语调柔和，语气平稳。\
字幕内容： 真看不出来，你这个人还挺贫。\
输出：字幕内容：“真看不出来，你这个人还挺贫。” 这句话可能是女士对旁边人的一种评价或反应。\
根据音频线索中描述的笑声和柔和平稳的语调，以及视频字幕内容中的“真看不出来，你这个人还挺贫”可以推测出人物的情感状态。\
首先，角色先是笑了一声，这表明他可能处于一种轻松或愉快的情绪状态，笑声通常与积极的情绪相关联。\
接着，字幕内容中的“真看不出来，你这个人还挺贫”可能表达了一种嘲讽或调侃的语气。尽管语气平稳，但由于使用了“挺贫”这一贬义词汇，可以推断出角色在表达时带有一定的讽刺和负面情绪。\
因此，综合音频线索和字幕内容，可以推断出角色的情感状态为带有调侃和嘲讽的轻松或愉快，但同时也带有一定的负面情绪。\
输入：{reason_merge}\
输出："
                    }
                ]

    # (video, subtitle) -> text reason
    elif video_reason != '' and audio_reason == '':
        reason_merge = ""
        reason_merge += f"视频线索：{video_reason}；"
        reason_merge += f"字幕内容：{subtitle}"

        prompt = [
                    {
                        "type": "text", 
                        "text": f"请假设作为情感领域的专家。我们提供了视频中可能与人物情感相关的线索。此外，我们还提供了视频原始的字幕内容。\
请分析从哪些字幕内容中可以推测出人物的情感状态，并给出推理依据。在推理过程中，请结合视频线索进行分析。\
输入：视频线索： 在视频的开头，我们看到一位女士坐在沙发上，面对着旁边的人，她瞪大眼睛，嘴巴微张，表现的有些疑惑和惊讶。到了视频的中间部分，女士逐渐笑了起来，嘴角上扬，眼睛弯曲，表现出开心的情绪。同时，女士耸肩和低头的动作表现出了不好意思的情绪。\
字幕内容： 真看不出来，你这个人还挺贫。\
输出：字幕内容：“真看不出来，你这个人还挺贫。” 这句话可能是女士对旁边人的一种评价或反应。\
根据视频线索中女士表现出的笑容和开心的情绪，我们可以推断这句话可能带有一种轻松的语气。\
因此，这句话可能并非负面评价，而是一种调侃或者开玩笑的表达方式，与女士整体展现的积极情绪相符合。\
输入：{reason_merge}\
输出："
                    }
                ]
    
    # 全为空，则返回error
    else: # audio_reason and video_reason 全为空
        prompt = [
                {
                    "type": "text", 
                    "text": f"请假设作为情感领域的专家。我们有一段文本，请分析从哪些内容中可以推测出人物的情感状态，并给出推理依据。文本内容为：{subtitle}"
                }
            ]
        
    print (prompt[0]['text']) # debug
    for item in prompt: print (item['type']) # debug
    time.sleep(sleeptime)
    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    print (response)
    return response


def update_text_reason_v666(reason, subtitle, sleeptime=0, model='gpt-3.5-turbo-16k-0613'):

    if reason != '':
        reason_merge = ""
        reason_merge += f"视频线索：{reason}；"
        reason_merge += f"字幕内容：{subtitle}"

        prompt = [
                    {
                        "type": "text", 
                        "text": f"请假设作为情感领域的专家。我们提供了视频中可能与人物情感相关的线索。此外，我们还提供了视频原始的字幕内容。\
    请分析从哪些字幕内容中可以推测出人物的情感状态，并给出推理依据。在推理过程中，请结合视频线索进行分析。\
    输入：{reason_merge}\
    输出："
                    }
                ]
    else:
        reason_merge = ""
        reason_merge += f"字幕内容：{subtitle}"

        prompt = [
                    {
                        "type": "text", 
                        "text": f"请假设作为情感领域的专家。我们提供了视频原始的字幕内容。\
    请分析从哪些字幕内容中可以推测出人物的情感状态，并给出推理依据。\
    输入：{reason_merge}\
    输出："
                    }
                ]
    
    print (prompt[0]['text']) # debug
    for item in prompt: print (item['type']) # debug
    time.sleep(sleeptime)
    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    print (response)
    return response

# ---------------------------------------------------------------------
## Emotion Reasoning Result Evaluation
# ---------------------------------------------------------------------
def get_reason_to_discrete(reason, candidate_list, sleeptime=0, model='gpt-3.5-turbo-16k-0613'):
    prompt = [
                {
                    "type":  "text", 
                    ## original prompt
                    "text": f"请假设作为情感领域的专家。我们提供了可能与人物情感相关的线索。请依据提供的线索识别主要人物的情感状态。\
                              我们提供了情感候选集合，请按照可能性从高到低进行排序。请直接输出排序结果。\
                              候选集合为：{candidate_list}。输入：{reason}。输出："

                    ## CoT prompt
                    # "text": f"请假设作为情感领域的专家。我们提供了可能与人物情感相关的线索。请依据提供的线索识别人物的情感状态，并提供分析依据。\
                    #           我们提供了情感候选集合，请按照可能性从高到低进行排序。\
                    #           候选集合为：{candidate_list}。输入：{reason}。输出格式：{{rank:, evidence:}}"

                    ## original prompt：
                    # "text": f"请假设作为情感领域的专家。我们提供了与人物情感相关的所有线索。请依据提供的线索识别描述中主要人物的情感状态。\
                    #           我们提供了情感候选集合，请按照可能性从高到低进行排序。请直接输出排序结果，排序结果中不能包括不在候选集中的情感标签。\
                    #           候选集合为：{candidate_list}。输入：{reason}。输出："
                }
            ]
    print (prompt[0]['text']) # debug
    for item in prompt: print (item['type']) # debug
    time.sleep(sleeptime)
    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    print (response)
    return response

# valence: [-5, 5]
def get_reason_to_valence(reason, sleeptime=0, model='gpt-3.5-turbo-16k-0613'):
    prompt = [
                {
                    "type":  "text", 
                    "text": f"请假设作为情感领域的专家。我们提供了可能与人物情感相关的线索。请依据提供的线索识别主要人物的情绪正负向。\
                              输出的数值范围是-5到+5之间的浮点数。其中，-5表示情绪非常负向，0表示情绪为中性，+5表示情绪非常正向。整体上，数值越大，情绪越正向; 数值越小，情绪越负向。\
                              请根据你的判断，输出带两位小数点的浮点数。请直接输出数值结果，不包括分析过程。\
                              输入：{reason}。输出："
                }
            ]
    print (prompt[0]['text']) # debug
    for item in prompt: print (item['type']) # debug
    time.sleep(sleeptime)
    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    print (response)
    return response

def get_reason_to_openset(multi_reason, sleeptime=0, model='gpt-3.5-turbo-16k-0613', lang='chi'):
    
    if lang == 'chi':
        prompt = [
                    {
                        "type": "text",
                        "text": f"请假设作为情感领域的专家。我们提供了可能与人物情感相关的线索。请依据提供的线索识别主要人物的情感状态。\
不同的情感类别之间用逗号隔开。仅输出比较明确的情感类别，输出格式为list形式。如果没有则输出为空list。\
输入：{multi_reason}。输出："
                    }
                ]
    elif lang == 'eng':
          ## 这个出来的格式会有些问题
#         prompt = [
#                     {
#                         "type": "text",
#                         "text": f"Please assume the role of an expert in the field of emotions. \
# We provide clues that may be related to the emotions of the characters. Based on the provided clues, please identify the emotional states of the main characters. \
# Please separate different emotional categories with commas and output only the clearly identifiable emotional categories in a list format. \
# If none are identified, please output an empty list.\
# Input: {multi_reason}; Output: "
#                     }
#                 ]
        ## 我们增加 few-shot 看看结果咋样 [需要进一步测试看看！！]
        prompt = [
                    {
                        "type": "text",
                        "text": f"Please assume the role of an expert in the field of emotions. \
We provide clues that may be related to the emotions of the characters. Based on the provided clues, please identify the emotional states of the main characters. \
Please separate different emotional categories with commas and output only the clearly identifiable emotional categories in a list format. \
If none are identified, please output an empty list.\
Input: We cannot recognize his emotional state; Output: []\
Input: His emotional state is happy, sad, and angry; Output: [happy, sad, angry]\
Input: {multi_reason}; Output: "
                    }
                ]
    
    print (prompt[0]['text']) # debug
    for item in prompt: print (item['type']) # debug
    time.sleep(sleeptime)
    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    print (response)
    return response


def get_reason_to_gesture(video_reason, sleeptime=0, model='gpt-3.5-turbo-16k-0613'):
    prompt = [
                {
                    "type": "text", 
                    "text": f"请假设作为情感领域的专家。我们提供了视频中与人物情感相关的线索。请输出描述中涉及的面部运动和肢体动作，不同动作之间用逗号隔开。输出格式为list形式。\
输入：开头的画面中，我们看到一个男性坐在一个室内环境中，背景看起来像是一个办公室或者图书室，有书架和文件。男性戴着眼镜，紧紧皱着眉头，嘴角向下，眼眸低垂，显得十分低落。他嘴巴微微运动，目光没有直视对方，好像在思考或回忆着一些事情。结尾的画面中， 男性的表情更加低落，眼睛也同时看向下方，表现得很懊悔。\
输出：['紧紧皱着眉头', '嘴角向下', '眼眸低垂', '嘴巴微微运动', '目光没有直视对方', '眼睛也同时看向下方']\
输入：{video_reason}。\
输出："
                }
            ]
    print (prompt[0]['text']) # debug
    for item in prompt: print (item['type']) # debug
    time.sleep(sleeptime)
    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    print (response)
    return response


# def get_reason_to_modality(video_reason, audio_reason, text_reason, emotion, flag='rank', sleeptime=0, model='gpt-3.5-turbo-16k-0613'):
#     reason_merge = ""
#     reason_merge += f"线索1：{video_reason}；\n"
#     reason_merge += f"线索2：{audio_reason}；\n"
#     reason_merge += f"线索3：{text_reason}；\n"
#     reason_merge += f"情感表情：{emotion}；\n"

#     if flag == 'rank':
#         prompt = [
#                     {
#                         "type": "text", 
#                         "text": f"请假设作为情感领域的专家。我们提供了三条线索以及情感标签。请判断哪条线索最有可能预测得到这个情感标签，按照可能性从高到低进行排序。\
#                             输出内容包含排序结果和分析过程。排序结果用list形式，list中只包含{{线索1, 线索2, 线索3}}三个元素。输出格式为: {{'rank':, 'anaylze':}}\
#                             输入：{reason_merge}。输出："
#                     }
#                 ]
#     elif flag == 'identical':
#         prompt = [
#                     {
#                         "type": "text", 
#                         "text": f"请假设作为情感领域的专家。我们提供了三条线索以及情感标签。请判断哪些线索可以预测得到这个情感标签，输出内容包含分析过程与结果集合。\
#                             结果集合为list形式，结果集合属于{{线索1, 线索2, 线索3}}的全集或子集。输出格式为: {{'result':, 'anaylze':}}\
#                             输入：{reason_merge}。输出："
#                     }
#                 ]
        
#     print (prompt[0]['text']) # debug
#     for item in prompt: print (item['type']) # debug
#     time.sleep(sleeptime)
#     response = get_completion(prompt, model)
#     response = func_postprocess_chatgpt(response)
#     print (response)
#     return response


def get_openset_overlap_rate(gt_openset, pred_openset, sleeptime=0, model='gpt-3.5-turbo-16k-0613'):
    prompt = [
                {
                    "type": "text", 
                    "text": f"请假设作为情感领域的专家。我们提供了两个情感标签的集合，请计算两个集合之间的重叠率。\
                        输出的数值范围是0到1之间的浮点数。数值越小，重叠率越低；数值越大，重叠率越高。在重叠率的计算中，如果不同集合的情感之间存在一定相似性，也算是一种重叠。\
                        请根据你的判断，输出带两位小数点的浮点数。请直接输出数值结果，不包括分析过程。\
                        集合1：{gt_openset}。集合2：{pred_openset}。输出："
                }
            ]
    print (prompt[0]['text']) # debug
    for item in prompt: print (item['type']) # debug
    time.sleep(sleeptime)
    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    print (response)
    return response


# => 测试下来，这个prompt是最好的
def get_openset_synonym(gt_openset, pred_openset, sleeptime=0, model='gpt-3.5-turbo-16k-0613'):
    merge_openset = list(set(gt_openset) | set(pred_openset)) # 删除了完全一样的单词，降低聚类难度
    prompt = [
                {
                    "type": "text", 
                    "text": f"Please assume the role of an expert in the field of emotions. We provide a set of emotions. \
Please group the emotions, with each group containing emotions with the same meaning. \
Directly output the results. The output format should be a list containing multiple lists. \
Input: ['Agree', 'agreement', 'Relaxed', 'acceptance', 'pleasant', 'relaxed', 'Accept', 'positive', 'Happy'] Output: [['Agree', 'agreement', 'Accept', 'acceptance'], ['Relaxed', 'relaxed'],['pleasant', 'positive', 'Happy']] \
Input: {merge_openset} Output:"
                }
            ]
    print (prompt[0]['text']) # debug
    for item in prompt: print (item['type']) # debug
    time.sleep(sleeptime)
    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    print (response)
    return response


def get_sentence_overlap_rate(gt, pred, sleeptime=0, model='gpt-3.5-turbo-16k-0613'):
    prompt = [
                {
                    "type": "text", 
                    "text": f"请假设作为情感领域的专家。我们提供了两段关于主要人物情感线索的描述，请计算两个线索之间的相似度。\
                        输出的数值范围是0到1之间的浮点数。数值越小，相似度越低；数值越大，相似度越高。\
                        请根据你的判断，输出带两位小数点的浮点数。请直接输出数值结果，不包括分析过程。\
                        描述1：{gt}。描述2：{pred}。输出："
                }
            ]
    print (prompt[0]['text']) # debug
    for item in prompt: print (item['type']) # debug
    time.sleep(sleeptime)
    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    print (response)
    return response


# 采用更强大的 gpt-4o 进行这种文本处理操作，结果会更加稳定一点
def get_synonym(text, sleeptime=0, model='gpt-3.5-turbo-16k-0613'):
    if len(text) == 0:
        return ""
    text = text.replace('\n', '')
    prompt = f"""
              Please output the synonyms of the following word in a list format. Please directly return the answer:
              
              Input: 'insecure'

              Output: ['uncertain', 'unsure', 'unconfident', 'self-doubting']

              Input: {text}

              Output: 
              """
    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    print (text)
    print (response)
    return response


def get_different_format(text, prompt_type='case1', model='gpt-3.5-turbo-16k-0613'):
    if len(text) == 0:
        return ""
    text = text.replace('\n', '')

    ## one-shot example1:
    if prompt_type == 'case1':
        prompt = f"""
                Please output different forms of the following word in a list format. Please directly return the answer:
                
                Input: 'victimised'

                Output: ['victimised', 'victimising', 'victimises', 'victimise', 'victimisation', 'victimized', 'victimizing', 'victimizes', 'victimize', 'victimization']

                Input: {text}

                Output: 
                """

    ## one-shot example2:
    elif prompt_type == 'case2':
        prompt = f"""
                Please output different forms of the following word in a list format. Please directly return the answer:
                
                Input: 'hopeful'

                Output: ['hope', 'hopes', 'hoping', 'hoped', 'hopeful', 'hopefuls', 'hopefully', 'hopefulness']

                Input: {text}

                Output: 
                """

    response = get_completion(prompt, model)
    response = func_postprocess_chatgpt(response)
    print (text)
    print (response)
    return response


if __name__ == '__main__':
    ## for ghelper
    # set http_proxy=http://127.0.0.1:9981
    # set https_proxy=http://127.0.0.1:9981
    ## for clash for windows
    # set http_proxy=http://127.0.0.1:7890
    # set https_proxy=http://127.0.0.1:7890


    # ------------------ pre testing on different modalities -----------------
    ## text input [test ok]
    eng = "This is a very long sentence composed of exactly one thousand words without any punctuation \
            and it continues to grow as more words are added to reach the desired length which might seem quite"
    get_translate_eng2chi(eng)

    ## image input [test ok]
    # image_paths = ["/root/image1.jpg", "/root/image2.jpg", 
    #                "/root/image3.png", "/root/image4.png", 
    #                "/root/image5.png", "/root/image6.png"]
    # get_image_caption_batch(image_paths)

    ## video input [test ok]
    # video_paths = ["/root/sample_00000000.avi",
    #                "/root/sample_00000001.mp4",
    #                "/root/sample_00000002.mp4",
    #                "/root/sample_00000004.mp4",
    #                "/root/sample_00000005.mp4",
    #                "/root/sample_00004102.avi"]
    # get_video_caption_batch(video_paths)

    
    # ------------------ reasoning testing -----------------
    ## for AffectGPT process => test ok, very strong baseline
    # text = "寡人要是韩王，先拿你的人头祭奠那些冤死士卒。"
    # get_text_reason(text)
    # video_path = "E:\\Dataset\\mer2023-dataset\\test1\\sample_00000008.avi"
    # get_video_reason(video_path)