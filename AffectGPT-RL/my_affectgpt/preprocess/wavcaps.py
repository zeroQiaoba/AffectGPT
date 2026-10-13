import tqdm
import config
from toolkit.utils.read_files import *
from toolkit.utils.functions import *
from my_affectgpt.evaluation.ew_metric import *

dataset = 'WavCaps'

##################################################
## keep only the emotion-related caption information
##################################################
def convert_json_to_csv(wav_root, label_path, save_path):

    # read infos
    store_names, store_captions = [], []
    with open(label_path, 'r', encoding='utf-8') as f:
        datas = json.load(f)
    for data in tqdm.tqdm(datas):
        name    = data['name']
        caption = data['description']
        caption = caption.replace('\n', ' ').replace('\t', ' ').strip()
        store_names.append(name)
        store_captions.append(caption)

    print (f'qa number: {len(store_names)}') # 140750 (name, q, a)
    print (f'sample number: {len(set(store_names))}') # 140750 unique samples

    # store into save_path
    columns = ['name', 'caption']
    data = np.column_stack([store_names, store_captions])
    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(save_path, index=False)


## Step1: label conversion [drop the conversation part so questions do not depend on context]


def perseve_emotion_related_description(label_path, save_path):

    ## step1: read label_path
    names = func_read_key_from_csv(label_path, 'name')
    captions = func_read_key_from_csv(label_path, 'caption')
    name2caption = {}
    for (name, caption) in zip(names, captions):
        name2caption[name] = caption
    print (f'sample number: {len(name2caption)}') # 140750
    

    ## step3: statistics
    save_temp = os.path.join(config.DATA_DIR[dataset], 'name2openset.npy')
    name2openset = np.load(save_temp, allow_pickle=True).tolist()
    print (f'sample number: {len(name2openset)}') # 140750
    emo_names = [name for name in name2openset if string_to_list(name2openset[name]) != []]
    print (f'emotion sample number: {len(emo_names)}') # 5945

    ## step4: result storage
    names_filter, captions_filter = [], []
    for name in name2caption:
        if name in emo_names:
            names_filter.append(name)
            captions_filter.append(name2caption[name])
    columns = ['name', 'caption']
    data = np.column_stack([names_filter, captions_filter])
    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(save_path, index=False)

## step2: 


def remove_not_used_images(label_path, audio_root):

    # number of videos present in the labels
    names = func_read_key_from_csv(label_path, 'name')
    print ('labeled sample number: ', len(set(names))) # 140750

    # number of videos present in the path
    audio_paths = []
    for audio_subroot in glob.glob(audio_root + '/*'):
        for audio_path in glob.glob(audio_subroot + '/*'):
            audio_paths.append(audio_path)
    print (f'audio sample in root number: {len(audio_paths)}') # 140838

    # delete files [actually every sample is labeled]
    for audio_path in tqdm.tqdm(audio_paths):
        audioname = "/".join(audio_path.split('/')[-2:])
        if audioname not in names:
            os.remove(audio_path)


# step3: delete videos not used anywhere => [this is the already-filtered file]





##################################################
## one dataset was not fully downloaded, so handle the freesound files separately
##################################################
def convert_json_to_csv_fsd(wav_root, label_path, save_path):

    # read infos
    store_names, store_captions = [], []
    with open(label_path, 'r', encoding='utf-8') as f:
        datas = json.load(f)['data']
    for data in tqdm.tqdm(datas):
        # print (data)
        name = 'FreeSound_flac/' + str(data['id']) + '.flac'
        assert os.path.exists(os.path.join(wav_root, name))
        caption = data['description']
        caption = caption.replace('\n', ' ').replace('\t', ' ').strip()
        store_names.append(name)
        store_captions.append(caption)

    print (f'qa number: {len(store_names)}') # 262300 (name, q, a)
    print (f'sample number: {len(set(store_names))}') # 262300 unique samples

    # store into save_path
    columns = ['name', 'caption']
    data = np.column_stack([store_names, store_captions])
    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(save_path, index=False)


## Step1: label conversion [drop the conversation part so questions do not depend on context]


def perseve_emotion_related_description_fsd(label_path, save_path):

    ## step1: read label_path
    names = func_read_key_from_csv(label_path, 'name')
    captions = func_read_key_from_csv(label_path, 'caption')
    name2caption = {}
    for (name, caption) in zip(names, captions):
        name2caption[name] = caption
    print (f'sample number: {len(name2caption)}') # 262300
    

    ## step3: statistics
    save_temp = os.path.join(config.DATA_DIR[dataset], 'name2openset.npy')
    name2openset = np.load(save_temp, allow_pickle=True).tolist()
    print (f'sample number: {len(name2openset)}') # 262300
    emo_names = [name for name in name2openset if string_to_list(name2openset[name]) != []]
    print (f'emotion sample number: {len(emo_names)}') # 29821

    ## step4: result storage
    names_filter, captions_filter = [], []
    for name in name2caption:
        if name in emo_names:
            names_filter.append(name)
            captions_filter.append(name2caption[name])
    columns = ['name', 'caption']
    data = np.column_stack([names_filter, captions_filter])
    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(save_path, index=False)



## step3: manually merge the two csv files
# => label.csv label-onlyemotion.csv
    

########################################################
## some wavcaps data is unreadable, which crashes the program
########################################################
from my_affectgpt.models.ImageBind.data import load_audio, transform_audio
# WavCaps is not part of this release, so its dataset module is gone from my_affectgpt.datasets.datasets

## check1: check whether all files exist [they all exist]

## check3: find audio files that cannot be opened [fewer scripts, faster error lookup]
'''
AudioSet_SL/YVcu0pVF1npM.flac
AudioSet_SL/Y06-g5jz-OGc.flac
AudioSet_SL/Y3sSblRfEG2o.flac
AudioSet_SL/YWudGD6ZHRoY.flac
AudioSet_SL/YFli8wjBFV2M.flac
AudioSet_SL/YmW3S0u8bj58.flac
'''

## after deleting those, errors remain: audio exists but raw audio does not
## check again: whether every audio is readable and yields the same data.shape
'''
FreeSound_flac/640185.flac
FreeSound_flac/641577.flac
'''