
import os
import tqdm
import glob
import shutil
import random

import numpy as np

from toolkit.globals import *
from toolkit.utils.functions import *
from toolkit.utils.read_files import func_write_key_to_csv
from toolkit.utils.read_files import func_read_key_from_csv

def merge_all_subtitles(language='chinese'):
    whole_chis = []
    for dataset in ['MER2023', 'IEMOCAPSix', 'CMUMOSI', 'CMUMOSEI', 'SIMS', 'MELD', 'SIMSv2']:
        trans_path = config.PATH_TO_TRANSCRIPTIONS[dataset]
        chis = func_read_key_from_csv(trans_path, language)
        print (f'sample number: {len(chis)}')
        print (f'chi: {chis[10]}')
        whole_chis.extend(chis)
    print (f'all sample number: {len(whole_chis)}')

    names = []
    name2key = {}
    for ii in range(len(whole_chis)):
        names.append(ii)
        name2key[ii] = [whole_chis[ii]]
    keynames = [language]
    func_write_key_to_csv(f'{language}_whole_dataset.csv', names, name2key, keynames)


def generate_tts_audio_folder(tts_root, language='chinese'):
    
    start_ids = 0
    for dataset in ['MER2023', 'IEMOCAPSix', 'CMUMOSI', 'CMUMOSEI', 'SIMS', 'MELD', 'SIMSv2']:
        ## 读取 (names, chis, ids)
        trans_path = config.PATH_TO_TRANSCRIPTIONS[dataset]
        chis = func_read_key_from_csv(trans_path, language)
        names = func_read_key_from_csv(trans_path, 'name')
        assert len(chis) == len(names)
        print (f'sample number: {len(chis)}')
        print (f'chi: {chis[10]}')
        ids = np.arange(start_ids, start_ids + len(names), 1)
        assert len(ids) == len(names)
        start_ids += len(names)

        ## 将合成语音复制到指定文件夹
        save_root = config.PATH_TO_RAW_AUDIO[dataset] + f'-tts{language[:3]}'
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        copy_from_tts = 0
        for ii in tqdm.tqdm(range(len(names))):
            tts_path = os.path.join(tts_root, str(ids[ii])+'.wav')
            if not os.path.exists(tts_path):
                continue
            save_path = os.path.join(save_root, names[ii]+'.wav')
            if os.path.exists(save_path):
                continue
            shutil.copy(tts_path, save_path)
            copy_from_tts += 1
        print (f'copy_from_tts number: {copy_from_tts}')
        
        ## 对于合成失败的语音，则直接用原始的音频文件
        copy_from_origin = 0
        audio_root = config.PATH_TO_RAW_AUDIO[dataset]
        for audio_path in tqdm.tqdm(glob.glob(audio_root + '/*')):
            audio_name = os.path.basename(audio_path)
            save_path = os.path.join(save_root, audio_name)
            if os.path.exists(save_path):
                continue
            shutil.copy(audio_path, save_path)
            copy_from_origin += 1
        print (f'copy_from_origin number: {copy_from_origin}')

        ## 统计音频是否完整
        print (f'all audio number: {copy_from_tts + copy_from_origin}')
        print (f'all audio number: {len(os.listdir(save_root))}')


def check_correctness(language='chinese'):
    for dataset in ['MER2023', 'IEMOCAPSix', 'CMUMOSI', 'CMUMOSEI', 'SIMS', 'MELD', 'SIMSv2']:
        trans_path = config.PATH_TO_TRANSCRIPTIONS[dataset]
        chis = func_read_key_from_csv(trans_path, language)
        names = func_read_key_from_csv(trans_path, 'name')
        audio_root = config.PATH_TO_RAW_AUDIO[dataset] + f'-tts{language[:3]}'

        # random select one sample
        save_root = './temptemp-ttsaudio'
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        index = random.randint(0, len(names)-1)
        name, chi = names[index], chis[index]
        save_path = os.path.join(save_root, name+'.wav')
        audio_path = os.path.join(audio_root, name+'.wav')
        shutil.copy(audio_path, save_path)
        print (name, chi)


def convert_audio_to_16k():
    for dataset in ['MER2023', 'IEMOCAPSix', 'CMUMOSI', 'CMUMOSEI', 'SIMS', 'MELD', 'SIMSv2']:
        for suffix in ['-ttschi', '-ttseng']:
            audio_root = config.PATH_TO_RAW_AUDIO[dataset] + suffix
            save_root = config.PATH_TO_RAW_AUDIO[dataset] + suffix + '16k'
            func_split_audio_from_video_16k(audio_root, save_root)


def generate_cmd_lines_for_extractors():
    for dataset in ['MER2023', 'IEMOCAPSix', 'CMUMOSI', 'CMUMOSEI', 'SIMS', 'MELD', 'SIMSv2']:
        for feature in ['data2vec-audio-base-960h', 'wavlm-large', 'chinese-hubert-large']:
            for tts_lang in ['english', 'chinese']:
                cmd = f"python -u extract_audio_huggingface.py --dataset={dataset} --feature_level='UTTERANCE' --model_name={feature} --tts_lang={tts_lang} --gpu=0"
                print (cmd)


if __name__ == '__main__':
    
    ## 将所有数据集的中文文本合一起
    # merge_all_subtitles('chinese')
    # merge_all_subtitles('english')

    ## 将合成的音频差分到相应数据集的文件夹中
    # chinese_root = '/share/home/lianzheng/emotion-data/TTS-data/Chinese/results_lianzheng'
    # generate_tts_audio_folder(chinese_root, language='chinese')

    # english_root = '/share/home/lianzheng/emotion-data/TTS-data/English/syn'
    # generate_tts_audio_folder(english_root, language='english')

    ## 7个数据集，随机选择一个文件，检查合成的音频，是否和字幕文件是对齐的 [check ok]
    # check_correctness(language='chinese')
    # check_correctness(language='english')

    ## 将音频转成16k语音
    convert_audio_to_16k()

    ## 生成特征提取的cmd脚本
    # generate_cmd_lines_for_extractors()
    '''
cd chinese-mer-2023/feature_extraction/audio
python -u extract_audio_huggingface.py --dataset=MER2023 --feature_level='UTTERANCE' --model_name=data2vec-audio-base-960h --tts_lang=english --gpu=0
python -u extract_audio_huggingface.py --dataset=MER2023 --feature_level='UTTERANCE' --model_name=data2vec-audio-base-960h --tts_lang=chinese --gpu=0
python -u extract_audio_huggingface.py --dataset=MER2023 --feature_level='UTTERANCE' --model_name=wavlm-large --tts_lang=english --gpu=0
python -u extract_audio_huggingface.py --dataset=MER2023 --feature_level='UTTERANCE' --model_name=wavlm-large --tts_lang=chinese --gpu=0
python -u extract_audio_huggingface.py --dataset=MER2023 --feature_level='UTTERANCE' --model_name=chinese-hubert-large --tts_lang=english --gpu=0
python -u extract_audio_huggingface.py --dataset=MER2023 --feature_level='UTTERANCE' --model_name=chinese-hubert-large --tts_lang=chinese --gpu=0
python -u extract_audio_huggingface.py --dataset=IEMOCAPSix --feature_level='UTTERANCE' --model_name=data2vec-audio-base-960h --tts_lang=english --gpu=0
python -u extract_audio_huggingface.py --dataset=IEMOCAPSix --feature_level='UTTERANCE' --model_name=data2vec-audio-base-960h --tts_lang=chinese --gpu=0
python -u extract_audio_huggingface.py --dataset=IEMOCAPSix --feature_level='UTTERANCE' --model_name=wavlm-large --tts_lang=english --gpu=0
python -u extract_audio_huggingface.py --dataset=IEMOCAPSix --feature_level='UTTERANCE' --model_name=wavlm-large --tts_lang=chinese --gpu=0
python -u extract_audio_huggingface.py --dataset=IEMOCAPSix --feature_level='UTTERANCE' --model_name=chinese-hubert-large --tts_lang=english --gpu=0
python -u extract_audio_huggingface.py --dataset=IEMOCAPSix --feature_level='UTTERANCE' --model_name=chinese-hubert-large --tts_lang=chinese --gpu=0
python -u extract_audio_huggingface.py --dataset=CMUMOSI --feature_level='UTTERANCE' --model_name=data2vec-audio-base-960h --tts_lang=english --gpu=0
python -u extract_audio_huggingface.py --dataset=CMUMOSI --feature_level='UTTERANCE' --model_name=data2vec-audio-base-960h --tts_lang=chinese --gpu=0
python -u extract_audio_huggingface.py --dataset=CMUMOSI --feature_level='UTTERANCE' --model_name=wavlm-large --tts_lang=english --gpu=0
python -u extract_audio_huggingface.py --dataset=CMUMOSI --feature_level='UTTERANCE' --model_name=wavlm-large --tts_lang=chinese --gpu=0
python -u extract_audio_huggingface.py --dataset=CMUMOSI --feature_level='UTTERANCE' --model_name=chinese-hubert-large --tts_lang=english --gpu=0
python -u extract_audio_huggingface.py --dataset=CMUMOSI --feature_level='UTTERANCE' --model_name=chinese-hubert-large --tts_lang=chinese --gpu=0
python -u extract_audio_huggingface.py --dataset=CMUMOSEI --feature_level='UTTERANCE' --model_name=data2vec-audio-base-960h --tts_lang=english --gpu=0
python -u extract_audio_huggingface.py --dataset=CMUMOSEI --feature_level='UTTERANCE' --model_name=data2vec-audio-base-960h --tts_lang=chinese --gpu=0
python -u extract_audio_huggingface.py --dataset=CMUMOSEI --feature_level='UTTERANCE' --model_name=wavlm-large --tts_lang=english --gpu=0
python -u extract_audio_huggingface.py --dataset=CMUMOSEI --feature_level='UTTERANCE' --model_name=wavlm-large --tts_lang=chinese --gpu=0
python -u extract_audio_huggingface.py --dataset=CMUMOSEI --feature_level='UTTERANCE' --model_name=chinese-hubert-large --tts_lang=english --gpu=0
python -u extract_audio_huggingface.py --dataset=CMUMOSEI --feature_level='UTTERANCE' --model_name=chinese-hubert-large --tts_lang=chinese --gpu=0
python -u extract_audio_huggingface.py --dataset=SIMS --feature_level='UTTERANCE' --model_name=data2vec-audio-base-960h --tts_lang=english --gpu=0
python -u extract_audio_huggingface.py --dataset=SIMS --feature_level='UTTERANCE' --model_name=data2vec-audio-base-960h --tts_lang=chinese --gpu=0
python -u extract_audio_huggingface.py --dataset=SIMS --feature_level='UTTERANCE' --model_name=wavlm-large --tts_lang=english --gpu=0
python -u extract_audio_huggingface.py --dataset=SIMS --feature_level='UTTERANCE' --model_name=wavlm-large --tts_lang=chinese --gpu=0
python -u extract_audio_huggingface.py --dataset=SIMS --feature_level='UTTERANCE' --model_name=chinese-hubert-large --tts_lang=english --gpu=0
python -u extract_audio_huggingface.py --dataset=SIMS --feature_level='UTTERANCE' --model_name=chinese-hubert-large --tts_lang=chinese --gpu=0
python -u extract_audio_huggingface.py --dataset=MELD --feature_level='UTTERANCE' --model_name=data2vec-audio-base-960h --tts_lang=english --gpu=0
python -u extract_audio_huggingface.py --dataset=MELD --feature_level='UTTERANCE' --model_name=data2vec-audio-base-960h --tts_lang=chinese --gpu=0
python -u extract_audio_huggingface.py --dataset=MELD --feature_level='UTTERANCE' --model_name=wavlm-large --tts_lang=english --gpu=0
python -u extract_audio_huggingface.py --dataset=MELD --feature_level='UTTERANCE' --model_name=wavlm-large --tts_lang=chinese --gpu=0
python -u extract_audio_huggingface.py --dataset=MELD --feature_level='UTTERANCE' --model_name=chinese-hubert-large --tts_lang=english --gpu=0
python -u extract_audio_huggingface.py --dataset=MELD --feature_level='UTTERANCE' --model_name=chinese-hubert-large --tts_lang=chinese --gpu=0
python -u extract_audio_huggingface.py --dataset=SIMSv2 --feature_level='UTTERANCE' --model_name=data2vec-audio-base-960h --tts_lang=english --gpu=0
python -u extract_audio_huggingface.py --dataset=SIMSv2 --feature_level='UTTERANCE' --model_name=data2vec-audio-base-960h --tts_lang=chinese --gpu=0
python -u extract_audio_huggingface.py --dataset=SIMSv2 --feature_level='UTTERANCE' --model_name=wavlm-large --tts_lang=english --gpu=0
python -u extract_audio_huggingface.py --dataset=SIMSv2 --feature_level='UTTERANCE' --model_name=wavlm-large --tts_lang=chinese --gpu=0
python -u extract_audio_huggingface.py --dataset=SIMSv2 --feature_level='UTTERANCE' --model_name=chinese-hubert-large --tts_lang=english --gpu=0
python -u extract_audio_huggingface.py --dataset=SIMSv2 --feature_level='UTTERANCE' --model_name=chinese-hubert-large --tts_lang=chinese --gpu=0
    '''
