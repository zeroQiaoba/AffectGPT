import tqdm
import config
from scipy.io import wavfile
import numpy as np
from toolkit.utils.read_files import *
from toolkit.utils.functions import *
from my_affectgpt.evaluation.ew_metric import *

dataset = 'PromptSpeech'

#################################################
## label processing: -> (label.csv, audio)
#################################################
def label_conversion(label_path, save_audio, save_label):

    if not os.path.exists(save_audio):
        os.makedirs(save_audio)

    ## read infos => (path, style, content)
    names  = func_read_key_from_csv(label_path, 'item_name')
    styles = func_read_key_from_csv(label_path, 'style_prompt')
    contents = func_read_key_from_csv(label_path, 'content_prompt')
    
    ## => (store_names, store_paths, store_captions, store_contents)
    store_names, store_paths, store_captions, store_contents = [], [], [], []
    for ii in tqdm.tqdm(range(len(names))):
        # 1. read (path, style)
        name  = names[ii]
        style = styles[ii].replace('\n', ' ').replace('\t', ' ').strip()
        content = contents[ii].replace('\n', ' ').replace('\t', ' ').strip()
        
        # 2. path => audio_path
        audio_root = '/data3/renyong/Dataset/LibriTTS/LibriTTS'
        path1, path2 = name.split('_')[:2]
        audio_path1 = os.path.join(audio_root, 'train-clean-100', path1, path2, name+'.wav')
        audio_path2 = os.path.join(audio_root, 'train-clean-360', path1, path2, name+'.wav')
        audio_path3 = os.path.join(audio_root, 'test-clean', path1, path2, name+'.wav')
        audio_path4 = os.path.join(audio_root, 'dev-clean',  path1, path2, name+'.wav')
        if os.path.exists(audio_path1):
            audio_path = audio_path1
        elif os.path.exists(audio_path2):
            audio_path = audio_path2
        elif os.path.exists(audio_path3):
            audio_path = audio_path3
        elif os.path.exists(audio_path4):
            audio_path = audio_path4
        audio_name = os.path.basename(audio_path)
            
        store_names.append(audio_name)
        store_paths.append(audio_path)
        store_captions.append(style)
        store_contents.append(content)
    
    print (len(store_names)) # 26588
    print (len(store_paths)) # 26588
    print (len(set(store_names))) # 26588
    print (len(set(store_paths))) # 26588

    ## save audio root
    for (name, path) in tqdm.tqdm(zip(store_names, store_paths)):
        new_path = os.path.join(save_audio, name)
        shutil.copy(path, new_path)
    
    ## save label.csv
    columns = ['name', 'caption', 'subtitle']
    data = np.column_stack([store_names, store_captions, store_contents])
    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(save_label, index=False)

## Step1: convert labels to caption descriptions [26588 samples]
