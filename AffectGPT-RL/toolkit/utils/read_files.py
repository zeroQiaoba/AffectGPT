import os
import json
import math
import random
import numpy as np
import pandas as pd
import tqdm

## read pkl
# videoIDs, videoSpeakers, videoLabels, videoText, videoAudio, videoVisual1, videoSentence, trainVid, \
#     testVid = pickle.load(open(pkl_path, "rb"), encoding='latin1')

## write pkl
# pickle.dump([videoIDs, videoSpeakers, videoLabelsNew, videoTextNew, videoAudioNew, videoVisualNew, videoSentence, trainVid, \
#             testVid], open(save_path, 'wb'))

## read txt
# with open(output_path, encoding='utf8') as f: lines = [line.strip() for line in f]
# lines = [line for line in lines if len(line)!=0]

## write txt
# file_object = open('thefile.txt', 'w')
# file_object.write(all_the_text)
# file_object.close()

## read csv file
# df_label = pd.read_csv(label_file)
# meta_columns = ['timestamp', 'segment_id']
# metas = df_label[meta_columns].values # change to numpy
# label_timestamps = metas[:,0]
# df = pd.concat(segment_dfs) ## concat different csv files
# for _, row in df.iterrows(): ## read for each row
#     word = row['word']

## write csv file
# meta_columns = ['timestamp', 'segment_id']
# columns = meta_columns + [str(i) for i in range(embedding_dim)] # x,x,0,1,2,3,4,5,...
# data = np.column_stack([metas, aligned_embeddings])
# df = pd.DataFrame(data=data, columns=columns)
# df[meta_columns] = df[meta_columns].astype(np.int64)
# df.to_csv(csv_file, index=False)

## read json
# with open("../config/record.json",'r') as load_f:
#   load_dict = json.load(load_f)

## write json
# with open("../config/record.json","w") as f:
#   json.dump(new_dict,f)



# 功能1：只支持一个keyname
def func_labelstudio_init_key(keyname, names, values, save_path=""):
    whole_json = []
    for ii, name in enumerate(names):
        # s3_path = f's3://zeroqiaoba-first/video3/{name}.webm' # case1 [ok]
        # s3_path = f's3://zeroqiaoba/video5/{name}.webm'
        # s3_path = f's3://zeroqiaoba-first\\video3\\{name}.webm' # case2 [unwork]
        s3_path = f'/data/local-files/?d=video_webm/{name}.webm' # local storage
        onefile_json = {}
        onefile_json['id'] = ii
        onefile_json['data'] = {}
        onefile_json['data']['video'] = s3_path
        onefile_json['data'][keyname] = values[ii]
        onefile_json['annotations'] = []
        onefile_json['predictions'] = []
        whole_json.append(onefile_json)
    ## save whole_json
    with open(save_path, "w") as f: 
        json.dump(whole_json, f)
    return whole_json


# 功能1：给一个json文件增加一个key
def func_labelstudio_update_key(json_path, val_name, name2val):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        video = item['data']['video']
        videoname = os.path.basename(video).rsplit('.', 1)[0] # 对于 case1 [ok]
        # videoname = video.split('\\')[-1].rsplit('.', 1)[0] # case2 [unwork]
        item['data'][val_name] = name2val[videoname]
    
    with open(json_path, "w") as f:
        json.dump(data, f)
        

# 功能：将一个json分割到多个json，并存储在store_root中
def func_labelstudio_split_json(json_path, store_root, split_num=8, shuffle=True):
    if not os.path.exists(store_root):
        os.makedirs(store_root)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if shuffle:
        data = func_shuffle_list_data(data)

    subset_number = math.ceil(len(data)/split_num)
    for ii in range(split_num):
        sub_data = data[ii*subset_number:(ii+1)*subset_number]

        save_path = os.path.join(store_root, f'split-{ii}.json')
        with open(save_path, "w") as f:
            json.dump(sub_data, f)

# 功能：将一个list文件分成多份，存储在store_root中
def func_split_list_data(data, store_root, split_num=8, shuffle=True):
    if not os.path.exists(store_root):
        os.makedirs(store_root)

    if shuffle:
        data = func_shuffle_list_data(data)

    subset_number = math.ceil(len(data)/split_num)
    for ii in range(split_num):
        sub_data = data[ii*subset_number:(ii+1)*subset_number]

        save_path = os.path.join(store_root, f'split-{ii}.npy')
        np.save(save_path, sub_data)


# 功能2：读取key值对应的 name2key [因为可能存在多个values，所以返回的values都变成list格式了]
def func_labelstudio_read_key(json_path):
    with open(json_path,'r',encoding='utf-8') as f:
        data = json.load(f)
    
    name2val = {}
    for item in data:
        values = []

        ## analyze videoname
        videopath = item['data']['video']
        videoname = os.path.basename(videopath).rsplit('.', 1)[0]
        # case1: sample_00001189.webm
        # case2: def5d5b7-sample_00001189.webm
        videoname_split = videoname.split('-', 1)
        if len(videoname_split) == 2:
            videoname = videoname_split[1]
        elif len(videoname_split) == 1:
            videoname = videoname_split[0]
        else:
            print (videoname)
            raise ValueError('videoname has some errors!!')
        
        ## analyze annotations
        keys, values = [], []
        annotations = item['annotations']
        assert len(annotations) == 1
        result = annotations[0]['result']
        for ii in range(len(result)): # result 可能有多个 value

            # 分析 choices 内容
            if 'choices' in result[ii]['value']:
                item = result[ii]['value']['choices']
                keyname = result[ii]['from_name']
                values.append(item)
                keys.append(keyname)
                
            # 分析 text 内容
            if 'text' in result[ii]['value']:
                item = result[ii]['value']['text']
                keyname = result[ii]['from_name']
                values.append(item)
                keys.append(keyname)

        name2val[videoname] = (keys, values)
    return name2val


def func_shuffle_list_data(whole_json):
	indices = np.arange(len(whole_json))
	random.shuffle(indices)

	new_json = []
	for index in indices:
		new_json.append(whole_json[index])
	return new_json


# 功能3：从csv中读取特定的key对应的值
def func_read_key_from_csv(csv_path, key):
    values = []
    df = pd.read_csv(csv_path)
    # for _, row in df.iterrows():
    for _, row in df.iterrows():
        if key not in row:
            values.append("")
        else:
            value = row[key]
            if pd.isna(value): value=""
            values.append(value)
    return values


# names[ii] -> keys=name2key[names[ii]], containing keynames
def func_write_key_to_csv(csv_path, names, name2key, keynames):
    ## specific case: only save names
    if len(name2key) == 0 or len(keynames) == 0:
        df = pd.DataFrame(data=names, columns=['name'])
        df.to_csv(csv_path, index=False)
        return

    ## other cases:
    if isinstance(keynames, str):
        keynames = [keynames]
    assert isinstance(keynames, list)
    columns = ['name'] + keynames

    values = []
    for name in names:
        value = name2key[name]
        values.append(value)
    values = np.array(values)
    # ensure keynames is mapped
    if len(values.shape) == 1:
        assert len(keynames) == 1
    else:
        assert values.shape[-1] == len(keynames)
    data = np.column_stack([names, values])

    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(csv_path, index=False)
    

# 仅限于utf-8
def func_read_text_file(file_path):
    try:
        with open(file_path, encoding='utf8') as f: lines = [line.strip() for line in f]
        lines = [line for line in lines if len(line)!=0]
        return lines
    except:
        with open(file_path, encoding='ansi') as f: lines = [line.strip() for line in f]
        lines = [line for line in lines if len(line)!=0]
        return lines

##############################################################################################
## names[ii] -> values[ii], 可能存在多个values，写到keyname+{jj} 中，返回json内容，存储是后面存储的
# whole_json = func_labelstudio_init_key(keyname, names, values)

## 给一个json_path增加一个key，并按照原始路径保存到json_path
# func_labelstudio_update_key(json_path, val_name, name2val)

## 功能：将一个json分割到多个json，并存储在store_root中
# func_labelstudio_split_json(json_path, store_root, split_num=8, shuffle=True)

## 功能：将一个list数据分割成split_num
# func_split_list_data(data, store_root, split_num=8, shuffle=True)

## 功能：读取key值对应的 name2key，可能有多个values值
# name2val = func_labelstudio_read_key(json_path)

## 将json信息打乱
## new_json = func_shuffle_list_data(whole_json)

## 功能：从csv中读取特定的key对应的值
# func_read_key_from_csv(csv_path, key)

## names[ii] -> keys=name2key[names[ii]], containing keynames -> csv_path
## func_write_key_to_csv(csv_path, names, name2key, keynames)
##############################################################################################