import os
import glob
import tqdm
import pickle
from toolkit.globals import *
from toolkit.utils.chatgpt import *
from toolkit.utils.functions import *
from toolkit.utils.read_files import *

# t: ms
def convert_time(t):
    t = int(t)
    ms = t % 1000
    t = math.floor(t / 1000)
    h = math.floor(t / 3600)
    m = math.floor((t - h * 3600) / 60)
    s = t - 3600 * h - 60 * m
    return '%02d:%02d:%02d.%03d' % (h, m, s, ms)

# split short video from long video
def split_video_by_start_end_CMUMOSEI(data_root, save_root):
    ## video number 3837
    trans_root = os.path.join(data_root, 'Transcript/Segmented/Combined') # 3837 samples
    video_root = os.path.join(data_root, 'Videos/Full/Combined')     # 3837 samples
    ffmpeg_path = config.PATH_TO_FFMPEG

    if not os.path.exists(save_root): os.makedirs(save_root)
    video_paths = glob.glob(video_root+'/*')
    print (f'processing videos: {len(video_paths)}')
    for ii, video_path in enumerate(video_paths):
        print (f'{ii+1}/{len(video_paths)}  {video_path}')

        video_name = os.path.basename(video_path)[:-4]
        trans_path = os.path.join(trans_root, video_name+'.txt')
        assert os.path.exists(trans_path), f'{trans_path} not exists!!'

        ## read lines
        with open(trans_path, encoding='utf8') as f: lines = [line.strip() for line in f]
        lines = [line for line in lines if len(line)!=0]
        for ii, line in enumerate(lines):
            name1, name2, start, end, sentence = line.split('___', 4)
            name = f'{name1}_{name2}'
            subvideo_path = os.path.join(save_root, name+'.mp4')
            if os.path.exists(subvideo_path): continue

            start = convert_time(float(start)*1000)
            end = convert_time(float(end)*1000)

            cmd = f'{ffmpeg_path} -nostats -loglevel 0 -ss {start} -to {end} -accurate_seek -i "{video_path}" -vcodec copy -acodec copy "{subvideo_path}" -y'
            os.system(cmd)

# only preserve transcription.csv exist videos
def select_videos_for_cmumosei(data_root, video_root, save_root):
    trans_file = os.path.join(data_root, 'transcription.csv')
    if not os.path.exists(save_root): os.makedirs(save_root)

    error_lines = []
    df = pd.read_csv(trans_file)
    for idx, row in tqdm.tqdm(df.iterrows()):
        name = row['name']
        video_path = os.path.join(video_root, name+'.mp4')
        if not os.path.exists(video_path):
            error_lines.append(name)
        else:
            save_path = os.path.join(save_root, name+'.mp4')
            cmd = f'cp {video_path} {save_path}'
            os.system(cmd)
    print (f'error samples: {len(error_lines)}')


def generate_transcription(label_path, save_path):
    ## read pkl file
    names, eng_sentences = [], []
    videoIDs, _, _, videoSentences, _, _, _ = pickle.load(open(label_path, "rb"), encoding='latin1')
    for vid in videoIDs:
        names.extend(videoIDs[vid])
        eng_sentences.extend(videoSentences[vid])
    print (f'whole sample number: {len(names)}')
    
    ## write to csv file
    name2key = {}
    for ii, name in enumerate(names):
        name2key[name] = [eng_sentences[ii]]
    func_write_key_to_csv(save_path, names, name2key, ['english'])


def read_train_val_test(label_path, data_type):
    names, labels = [], []
    assert data_type in ['train', 'val', 'test']
    videoIDs, videoLabels, _, _, trainVids, valVids, testVids = pickle.load(open(label_path, "rb"), encoding='latin1')
    if data_type == 'train': vids = trainVids
    if data_type == 'val':   vids = valVids
    if data_type == 'test':  vids = testVids
    for vid in vids:
        names.extend(videoIDs[vid])
        labels.extend(videoLabels[vid])
    return names, labels


def normalize_dataset_format(data_root, save_root):
    # gain paths
    label_path = os.path.join(save_root, 'CMUMOSEI_features_raw_2way.pkl')
    assert os.path.exists(label_path), f'must has a pre-processed label file'

    # # gain (names, labels)
    train_names, train_labels = read_train_val_test(label_path, 'train')
    val_names,   val_labels   = read_train_val_test(label_path, 'val')
    test_names,  test_labels  = read_train_val_test(label_path, 'test')
    print (f'train: {len(train_names)}')
    print (f'val:   {len(val_names)}')
    print (f'test:  {len(test_names)}')
    
    ## output path
    save_video_temp  = os.path.join(save_root, 'video')
    save_video_final = os.path.join(save_root, 'subvideo')
    save_label = os.path.join(save_root, 'label.npz')
    save_trans = os.path.join(save_root, 'transcription.csv')
    if not os.path.exists(save_root):  os.makedirs(save_root)
    if not os.path.exists(save_video_temp): os.makedirs(save_video_temp)
    if not os.path.exists(save_video_final): os.makedirs(save_video_final)

    ## generate new transcripts [TODO: test]
    generate_transcription(label_path, save_trans)

    # save videos [TODO: test]
    split_video_by_start_end_CMUMOSEI(data_root, save_video_temp)
    select_videos_for_cmumosei(data_root, save_video_temp, save_video_final)
    
    ## generate label path
    whole_corpus = {}
    for name, videonames, labels in [('train', train_names, train_labels),
                                     ('val',   val_names,   val_labels  ),
                                     ('test',  test_names,  test_labels )]:
        whole_corpus[name] = {}
        print (f'{name}: sample number: {len(videonames)}')
        for ii, videoname in enumerate(videonames):
            whole_corpus[name][videoname] = {'emo': 0, 'val': labels[ii]}
            
    np.savez_compressed(save_label,
                        train_corpus=whole_corpus['train'],
                        val_corpus=whole_corpus['val'],
                        test_corpus=whole_corpus['test'])


if __name__ == '__main__':
    ## linux
    # data_root = '/share/home/lianzheng/emotion-data/CMU-MOSEI/Raw'
    # save_root = '/share/home/lianzheng/chinese-mer-2023/dataset/cmumosei-process'
    # normalize_dataset_format(data_root, save_root)

    ## window => 直接用ghelper客户端 => clash and ghelper 的端口号是不一样的 [linux无法翻墙]
    # set http_proxy=http://127.0.0.1:9981
    # set https_proxy=http://127.0.0.1:9981
    # translate transcript
    # data_root = 'H:\\desktop\\Multimedia-Transformer\\chinese-mer-2023\\dataset\\cmumosei-process'
    # trans_path = os.path.join(data_root, 'transcription.csv')
    # save_path  = os.path.join(data_root, 'transcription-engchi.csv')
    # polish_path = os.path.join(data_root, 'transcription-engchi-polish.csv')
    # func_translate_transcript(trans_path, save_path) # 还是需要人为检查一下，尤其是带“翻译|输入|输出”的单词
    # func_translate_transcript_polish(trans_path, save_path, polish_path) # 再次检测一下遗漏的部分

    # check for label range => should be [-3, 3]
    label_path = config.PATH_TO_LABEL['CMUMOSEI']
    train_corpus = np.load(label_path, allow_pickle=True)['train_corpus'].tolist()
    val_corpus = np.load(label_path, allow_pickle=True)['val_corpus'].tolist()
    test_corpus = np.load(label_path, allow_pickle=True)['test_corpus'].tolist()
    corpus = {**train_corpus, **val_corpus, **test_corpus}
    labels = [corpus[name]['val'] for name in corpus]
    print (f'min:{min(labels)}; max:{max(labels)}')
    '[-3, 3]'