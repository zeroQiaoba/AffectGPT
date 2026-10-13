import tqdm
import config
from scipy.io import wavfile
import numpy as np
from toolkit.utils.read_files import *
from toolkit.utils.functions import *
from my_affectgpt.evaluation.ew_metric import *

dataset = 'MAFW'


#################################################
## label processing: descriptive_text.xlsx -> label.csv
#################################################
def convert_json_to_csv(video_root, label_path, save_path):

    # read infos
    store_names, store_captions = [], []
    df = pd.read_excel(label_path)
    for _, row in df.iterrows():
        name = row['name']
        caption = row['english']
        caption = caption.replace('\n', ' ').replace('\t', ' ').strip()
        store_names.append(name)
        store_captions.append(caption)

    print (f'qa number: {len(store_names)}') # 140750 (name, q, a)
    print (f'sample number: {len(set(store_names))}') # 140750 unique samples

    # check video exists
    for name in store_names:
        videopath = os.path.join(video_root, name)
        assert os.path.exists(videopath)

    # store into save_path
    columns = ['name', 'caption']
    data = np.column_stack([store_names, store_captions])
    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(save_path, index=False)


## Step1: convert labels to caption-related labels



########################################
## video -> audio
########################################
def func_split_audio_multiprocess(argv=None, video_path=None, audio_path=None):
    video_path, audio_path = argv
    cmd = "%s -loglevel quiet -y -i %s -ar 16000 -ac 1 %s" %(config.PATH_TO_FFMPEG, video_path, audio_path)
    os.system(cmd)

def main_split_audio_multiprocess(video_root, save_root):
    
    if not os.path.exists(save_root): 
        os.makedirs(save_root)
            
    params = []
    for video_path in tqdm.tqdm(glob.glob(video_root + '/*')):
        audioname = os.path.basename(video_path)[:-4] + '.wav'
        audio_path = os.path.join(save_root, audioname)
        params.append((video_path, audio_path))
    
    with multiprocessing.Pool(processes=8) as pool:
        list(tqdm.tqdm(pool.imap(func_split_audio_multiprocess, params), total=len(params)))

## Step2: extract audio from video


def find_error_reasons(label_path, video_root, audio_root):

    ## the extracted audio and video do not match => the source itself has no audio track

    ## check whether all labeled samples have audio => they do not

    ## fill broken audio with an 8s silence clip
    for video_path in tqdm.tqdm(glob.glob(video_root + '/*')):
        audioname = os.path.basename(video_path)[:-4] + '.wav'
        audio_path = os.path.join(audio_root, audioname)
        if not os.path.exists(audio_path):
            print (audio_path)
            samplerate = 16000
            duration = 8
            silence = np.zeros(samplerate * duration, dtype=np.int16)
            wavfile.write(audio_path, samplerate, silence)


## some samples in this dataset have only video, no audio
# => fill these broken audio files with an 8s silence clip



########################################
## audio -> subtitle
########################################
import whisper
def extract_subtitle(audio_root, save_csv):

    model = whisper.load_model("base")

    name2key = {}
    for audio_path in tqdm.tqdm(glob.glob(audio_root + '/*')):
        name = os.path.basename(audio_path)
        subtitle = model.transcribe(audio_path)['text'].strip()
        name2key[name] = subtitle

    names = [name for name in name2key]
    keynames = ['english']
    func_write_key_to_csv(save_csv, names, name2key, keynames)

## Step3: extract subtitles from audio [it is multilingual, so recognition may be error-prone]



########################################
## video errors
########################################
def func_video_conversion_multiprocess(argv=None, video_path=None, save_path=None):
    video_path, save_path = argv
    cmd = "%s -loglevel quiet -y -i \"%s\" \"%s\"" %(config.PATH_TO_FFMPEG, video_path, save_path)
    os.system(cmd)


def main_video_conversion_multiprocess(video_root, save_root):
    if not os.path.exists(save_root): 
        os.makedirs(save_root)

    params = []
    for video_path in glob.glob(video_root + '/*'):
        videoname = os.path.basename(video_path)
        save_video = os.path.join(save_root, videoname)
        params.append((video_path, save_video))

    with multiprocessing.Pool(processes=8) as pool:
        list(tqdm.tqdm(pool.imap(func_video_conversion_multiprocess, params), total=len(params)))

## 1. video conversion
        

def postprocess_error_conversion_videos(video_root, save_root):
    for video_path in glob.glob(video_root + '/*'):
        videoname = os.path.basename(video_path)
        save_video = os.path.join(save_root, videoname)
        if not os.path.exists(save_video):
            print (save_video)
            shutil.copy(video_path, save_video)


## 2. some videos fail to convert, so copy the original video


########################################
## ov label extraction
########################################
def extract_ov_labels(label_path, save_path):
    names = func_read_key_from_csv(label_path, 'name')
    answers = func_read_key_from_csv(label_path, 'caption')
    name2reason = {}
    for (name, answer) in zip(names, answers):
        name2reason[name] = answer

    # name2reason => openset
    llm, tokenizer, sampling_params = func_read_batch_calling_model(modelname='Qwen25')
    extract_openset_batchcalling(name2reason=name2reason, store_npz=save_path,
                                 llm=llm, tokenizer=tokenizer, sampling_params=sampling_params)

def calculate_statistic_for_ov(save_path):
    fileitems = np.load(save_path, allow_pickle=True)['fileitems']
    print (f'sample number: {len(fileitems)}') # 54326

    whole_ovs = []
    for item in fileitems:
        whole_ovs.extend(string_to_list(item))
    print (f'unique label: {len(set(whole_ovs))}') # 1074

    # count labels appearing more than 6 times
    label_filter = []
    label2count = func_label_distribution(whole_ovs)
    for label in label2count:
        if label2count[label] >= 6:
            label_filter.append(label)
    print (f'at least appear 6 times: {len(set(label_filter))}') # 282


## step4: extract ov labels from label.csv
label_path = os.path.join(config.DATA_DIR[dataset], 'label.csv')
save_path = label_path[:-4]+'-ov.npz'
calculate_statistic_for_ov(save_path)

