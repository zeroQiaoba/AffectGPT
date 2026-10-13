import os
import shutil
import pickle
from toolkit.utils.chatgpt import *
from toolkit.utils.functions import *
from toolkit.utils.read_files import *


def generate_transcription(label_path, save_path):
    ## read pkl file
    names, eng_sentences = [], []
    videoIDs, _, _, videoSentences, _, _, _ = pickle.load(open(label_path, "rb"), encoding='latin1')
    for vid in videoIDs:
        names.extend(videoIDs[vid])
        eng_sentences.extend(videoSentences[vid])
    print (f'whole sample number: {len(names)}')
    
    # translate eng2chi
    chi_sentences = []
    for eng in eng_sentences:
        # chi = get_translate_eng2chi(eng, model='gpt-3.5-turbo-16k-0613')
        chi = get_translate_eng2chi(eng, model='gpt-4-0613')
        chi_sentences.append(chi)

    ## write to csv file
    name2key = {}
    for ii, name in enumerate(names):
        name2key[name] = [chi_sentences[ii], eng_sentences[ii]]
    func_write_key_to_csv(save_path, names, name2key, ['chinese', 'english'])


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
    label_path = os.path.join(save_root, 'CMUMOSI_features_raw_2way.pkl')
    assert os.path.exists(label_path), f'must has a pre-processed label file'
    video_root = os.path.join(data_root, 'Video/Segmented')

    # gain (names, labels)
    train_names, train_labels = read_train_val_test(label_path, 'train')
    val_names,   val_labels   = read_train_val_test(label_path, 'val')
    test_names,  test_labels  = read_train_val_test(label_path, 'test')
    print (f'train number: {len(train_names)}')
    print (f'val   number: {len(val_names)}')
    print (f'test  number: {len(test_names)}')
    
    ## output path
    save_video = os.path.join(save_root, 'subvideo')
    save_label = os.path.join(save_root, 'label.npz')
    save_trans = os.path.join(save_root, 'transcription.csv')
    if not os.path.exists(save_root):  os.makedirs(save_root)
    if not os.path.exists(save_video): os.makedirs(save_video)

    ## generate new transcripts
    generate_transcription(label_path, save_trans)

    ## generate label path
    whole_corpus = {}
    for name, videonames, labels in [('train', train_names, train_labels),
                                     ('val',   val_names,   val_labels  ),
                                     ('test',  test_names,  test_labels )]:
        whole_corpus[name] = {}
        for ii, videoname in enumerate(videonames):
            whole_corpus[name][videoname] = {'emo': 0, 'val': labels[ii]}
            
            # move video
            video_path = os.path.join(video_root, videoname+'.mp4')
            save_path  = os.path.join(save_video, videoname+'.mp4')
            shutil.copy(video_path, save_path)
            
    np.savez_compressed(save_label,
                        train_corpus=whole_corpus['train'],
                        val_corpus=whole_corpus['val'],
                        test_corpus=whole_corpus['test'])


if __name__ == '__main__':
    ## 直接用ghelper客户端 => clash and ghelper 的端口号是不一样的
    # set http_proxy=http://127.0.0.1:9981
    # set https_proxy=http://127.0.0.1:9981
    # data_root = 'G:\\CMU-MOSI\\Raw'
    # save_root = 'E:\\Dataset\\cmumosi-process'
    # normalize_dataset_format(data_root, save_root)

    # data_root = 'H:\\desktop\\Multimedia-Transformer\\chinese-mer-2023\\dataset\\cmumosi-process'
    # trans_path = os.path.join(data_root, 'transcription.csv')
    # polish_path = os.path.join(data_root, 'transcription-engchi-polish.csv')
    # func_translate_transcript_polish_merge(trans_path, polish_path) # 再次检测一下遗漏的部分

    # check for label range => should be [-3, 3]
    label_path = config.PATH_TO_LABEL['CMUMOSI']
    train_corpus = np.load(label_path, allow_pickle=True)['train_corpus'].tolist()
    val_corpus = np.load(label_path, allow_pickle=True)['val_corpus'].tolist()
    test_corpus = np.load(label_path, allow_pickle=True)['test_corpus'].tolist()
    corpus = {**train_corpus, **val_corpus, **test_corpus}
    labels = [corpus[name]['val'] for name in corpus]
    print (f'min:{min(labels)}; max:{max(labels)}')
    '[-3, 3]'