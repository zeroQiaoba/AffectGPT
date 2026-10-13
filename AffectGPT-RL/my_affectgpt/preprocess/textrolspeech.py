import tqdm
import config
from scipy.io import wavfile
import numpy as np
from toolkit.utils.read_files import *
from toolkit.utils.functions import *
from my_affectgpt.evaluation.ew_metric import *

dataset = 'TextrolSpeech'

#################################################
## label processing: -> (label.csv, audio)
#################################################
def label_conversion(label_path, save_audio, save_label):

    if not os.path.exists(save_audio):
        os.makedirs(save_audio)

    ## read infos => (path, style, content)
    paths  = func_read_key_from_csv(label_path, 'wav_fp')
    styles = func_read_key_from_csv(label_path, 'style_prompt')
    contents = func_read_key_from_csv(label_path, 'content_prompt')
    
    ## => (store_names, store_paths, store_captions, store_contents)
    store_names, store_paths, store_captions, store_contents = [], [], [], []
    for ii in tqdm.tqdm(range(len(paths))):
        # 1. read (path, style)
        path  = paths[ii]
        style = styles[ii].replace('\n', ' ').replace('\t', ' ').strip()
        content = contents[ii].replace('\n', ' ').replace('\t', ' ').strip()
        
        # 2. path => audio_path
        audio_path = ''
        if path.startswith('emotion_dataset'):
            audio_root = '/data3/renyong/zeroqiaoba/dataset/TextrolSpeech/audio-old'
            audio_path = os.path.join(audio_root, path)
            newname = '_'.join(path.split('/')[1:])
            newname = newname.replace(' ', '_')
            newname = newname.replace('(', '' )
            newname = newname.replace(')', '' )
            audio_name = f'emotion_{newname}'

        elif path.startswith('VCTK-Corpus'):
            audio_root = '/data3/renyong/Dataset/VCTK/24kwav'
            audio_path = os.path.join(audio_root, os.path.basename(path)[:-4]+'_mic2.wav')
            audio_name = f'vctk_{os.path.basename(audio_path)}'
            
        elif path.startswith('libritts'):
            audio_root = '/data3/renyong/Dataset/LibriTTS/LibriTTS'
            audio_path = os.path.join(audio_root, path.split('/', 1)[1])
            audio_name = f'libritts_{os.path.basename(audio_path)}'
            
        if audio_path=='' or (not os.path.exists(audio_path)):
            continue
        else:
            store_names.append(audio_name)
            store_paths.append(audio_path)
            store_captions.append(style)
            store_contents.append(content)

    ## save audio root
    for (name, path, caption) in tqdm.tqdm(zip(store_names, store_paths, store_captions)):
        new_path = os.path.join(save_audio, name)
        shutil.copy(path, new_path)
    
    ## save label.csv
    columns = ['name', 'caption', 'subtitle']
    data = np.column_stack([store_names, store_captions, store_contents])
    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(save_label, index=False)

## Step1: convert labels to caption descriptions [235486 samples]


#########################################################
## Split the label.csv into three subfolders for validation
#########################################################
def label_split_into_subset(label_path):
    names     = func_read_key_from_csv(label_path, 'name')
    captions  = func_read_key_from_csv(label_path, 'caption')
    subtitles = func_read_key_from_csv(label_path, 'subtitle')

    for prefix in ['emotion', 'vctk', 'libritts']:
        names_prefix, captions_prefix, subtitles_prefix = [], [], []
        for (name, caption, subtitle) in zip(names, captions, subtitles):
            if name.startswith(prefix):
                names_prefix.append(name)
                captions_prefix.append(caption)
                subtitles_prefix.append(subtitle)

        save_label = label_path[:-4] + f'_{prefix}.csv'
        columns = ['name', 'caption', 'subtitle']
        data = np.column_stack([names_prefix, captions_prefix, subtitles_prefix])
        df = pd.DataFrame(data=data, columns=columns)
        df.to_csv(save_label, index=False)

## Step2: dataset split
