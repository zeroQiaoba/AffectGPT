import tqdm
import config
from toolkit.utils.read_files import *
from toolkit.utils.functions import *
from my_affectgpt.evaluation.ew_metric import *

dataset = 'Video-ChatGPT'

##################################################
## keep only the emotion-related caption information
##################################################
def get_candidate_video_types(video_root):
    whole_types = []
    for video_path in tqdm.tqdm(glob.glob(video_root + '/*')):
        video_type = video_path.split('.')[-1]
        whole_types.append(video_type)
    print (set(whole_types))
    
# video_root = os.path.join(config.DATA_DIR[dataset], 'training_videos')
# get_candidate_video_types(video_root)
'''
'mkv', 'mp4', 'webm'
'''

def judge_whether_video_exist(video_root, name, candidates=['mp4', 'mkv', 'webm']):
    for video_type in candidates:
        videopath = os.path.join(video_root, name+'.'+video_type)
        if os.path.exists(videopath):
            return videopath
    return ""

from decord import VideoReader
def func_get_video_duration(videopath):
    vr = VideoReader(uri=videopath)
    fps = vr.get_avg_fps()
    duration = len(vr) / fps
    return duration

def convert_json_to_csv(video_root, label_path, save_path):

    # read infos
    with open(label_path, 'r', encoding='utf-8') as f:
        datas = json.load(f)
    
    store_names, store_qas, store_durations = [], [], []
    for ii, data in tqdm.tqdm(enumerate(datas)):
        name = data['video_id']
        question  = data['q'].replace('\n', ' ').replace('\t', ' ')
        answer = data['a'].replace('\n', ' ').replace('\t', ' ')

        try:
            videopath = judge_whether_video_exist(video_root, name)
            duration  =  func_get_video_duration(videopath)
            store_durations.append(duration)
        except:
            print ('error: ', videopath)

        store_names.append(name)
        store_qas.append([question, answer])
        if ii == 100: break

    print (f'sample number: {len(store_names)}')      # 100k QA pairs
    print (f'sample number: {len(set(store_names))}') # 13303 samples
    print (f'min duration: {min(store_durations)}')
    print (f'max duration: {max(store_durations)}')
    print (f'mean duration: {np.mean(store_durations)}')
    # min duration: 15.048366666666666
    # max duration: 235.302
    # mean duration: 114.38374598838188
    # => these are too long to suit MER content, which is mostly about understanding short videos,
    #    so long videos are not a good fit



## Step1: label conversion [drop the conversation part so questions do not depend on context]
video_root = os.path.join(config.DATA_DIR[dataset], 'video')
label_path = os.path.join(config.DATA_DIR[dataset], 'VideoInstruct_Dataset.json')
save_path  = os.path.join(config.DATA_DIR[dataset], 'label.csv')
convert_json_to_csv(video_root, label_path, save_path)
# => the dataset has both video and audio + video types



def perseve_emotion_related_description(label_path, save_path):

    ## step1: read label_path
    names = func_read_key_from_csv(label_path, 'name')
    questions = func_read_key_from_csv(label_path, 'question')
    answers = func_read_key_from_csv(label_path, 'answer')
    nq2a = {}
    for (name, question, answer) in zip(names, questions, answers):
        nq = f'{name}+{question}'
        nq2a[nq] = answer
    print (f'sample number: {len(nq2a)}') # 6636 [ok]
    
    # step2 filter caption: recognize sentiment from the caption
    llm, tokenizer, sampling_params = func_read_batch_calling_model(modelname='Qwen25')
    # 1. get openset
    nq2openset = {}
    whole_nqs, whole_opensets = extract_openset_batchcalling(name2reason=nq2a, 
                                                             llm=llm, tokenizer=tokenizer, sampling_params=sampling_params)
    for nq, openset in zip(whole_nqs, whole_opensets):
        nq2openset[nq] = openset
    # 2. save
    save_temp = os.path.join(config.DATA_DIR[dataset], 'nq2openset.npy')
    np.save(save_temp, nq2openset)
        


## step3: 


def remove_not_used_videos(label_path, video_root):

    # number of videos present in the labels
    names = func_read_key_from_csv(label_path, 'name')
    questions = func_read_key_from_csv(label_path, 'q')
    answers = func_read_key_from_csv(label_path, 'a')
    print ('labeled sample number: ', len(set(names))) # 7905

    # number of videos present in the path
    video_paths = []
    for video_subroot in tqdm.tqdm(glob.glob(video_root + '/*')):
        for video_path in glob.glob(video_subroot + '/*'):
            video_paths.append(video_path)
    print (f'video sample in root number: {len(video_paths)}') # 7905

## step4: delete videos not used anywhere => [this is the already-filtered file]


