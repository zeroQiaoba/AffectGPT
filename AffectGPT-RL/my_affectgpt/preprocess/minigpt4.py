import tqdm
import config
from toolkit.utils.read_files import *
from toolkit.utils.functions import *



##################################################
## keep only the emotion-related caption information in MINIGPT-4
##################################################
def convert_json_to_csv(json_path, save_path):

    # read infos
    name2caption = {}
    with open(json_path,'r',encoding='utf-8') as f:
        datas = json.load(f)['annotations']
    for data in tqdm.tqdm(datas):
        name = data["image_id"]
        caption = data["caption"]
        name2caption[name] = caption
    print (f'sample number: {len(name2caption)}') # sample number: 3439
    print (name2caption)

    # store into json_path
    names = [name for name in name2caption]
    print (save_path)
    func_write_key_to_csv(save_path, names, name2caption, 'caption')



def perseve_emotion_related_description(label_path, save_path):

    # step1: read label_path
    name2caption = {}
    names = func_read_key_from_csv(label_path, 'name')
    captions = func_read_key_from_csv(label_path, 'caption')
    for (name, caption) in zip(names, captions):
        name2caption[name] = caption

        
    # step3: statistics
    save_temp = os.path.join(config.DATA_DIR['MiniGPT4'], 'name2openset.npy')
    name2openset = np.load(save_temp, allow_pickle=True).tolist()
    print (f'sample number: {len(name2openset)}')
    emonames = [name for name in name2openset if string_to_list(name2openset[name]) != []]
    print (f'emotion sample number: {len(emonames)}') # emotion sample number: 892

    # step4: store the results
    name2caption_filter = {}
    for name in name2caption:
        if name in emonames:
            name2caption_filter[name] = name2caption[name]
    names = [name for name in name2caption_filter]
    func_write_key_to_csv(save_path, names, name2caption_filter, 'caption')




def postprocess(csv_path):
    name2caption = {}
    names = func_read_key_from_csv(csv_path, 'name')
    captions = func_read_key_from_csv(csv_path, 'caption')
    for (name, caption) in zip(names, captions):
        caption = caption.replace('\n', ' ')
        caption = caption.replace('\t', ' ')
        name2caption[name] = caption
    names = [name for name in name2caption]
    func_write_key_to_csv(csv_path, names, name2caption, 'caption')
    


'''
python affectgpt/preprocess/minigpt4.py
'''