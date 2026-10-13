import os
import shutil
from toolkit.utils.functions import *
from toolkit.utils.read_files import *

def func_convert_name_to_newname(video_id, clip_id):
    newname = video_id + '_%04d' %(clip_id)
    return newname

def func_merge_id_to_path(video_id, clip_id, video_root):
    video_path = os.path.join(video_root, video_id, '%04d.mp4' %(clip_id))
    return video_path

# label_path -> (video_paths, labels)
def read_labels(label_path, video_root):
    video_ids = func_read_key_from_csv(label_path, 'video_id')
    clip_ids = func_read_key_from_csv(label_path, 'clip_id')
    labels = func_read_key_from_csv(label_path, 'label')
    print (f'label range  ->  min:{min(labels)}  max:{max(labels)}')
    print (f'whole sample number: {len(labels)}')

    video_paths = []
    for ii in range(len(video_ids)):
        video_path = func_merge_id_to_path(video_ids[ii], clip_ids[ii], video_root)
        video_paths.append(video_path)

    return video_paths, labels

# 只读取 idx_path 对应的 items并返回
def gain_sub_items(video_paths, labels, idx_path):
    indexes = func_read_key_from_csv(idx_path, 'index')
    video_paths = np.array(video_paths)[indexes]
    labels = np.array(labels)[indexes]
    print (f'subset sample number: {len(labels)}')
    return video_paths, labels

# 转化为 newname 对应的 trans
def update_transcription(trans_path, save_path):
    video_ids = func_read_key_from_csv(trans_path, 'video_id')
    clip_ids = func_read_key_from_csv(trans_path, 'clip_id')
    chi_subtitles = func_read_key_from_csv(trans_path, 'Chinese')
    eng_subtitles = func_read_key_from_csv(trans_path, 'English')
    print (f'whole sample number: {len(video_ids)}')

    newnames = []
    for ii in range(len(video_ids)):
        newname = func_convert_name_to_newname(video_ids[ii], clip_ids[ii])
        newnames.append(newname)
    
    name2key = {}
    for ii, name in enumerate(newnames):
        name2key[name] = [chi_subtitles[ii], eng_subtitles[ii]]
    func_write_key_to_csv(save_path, newnames, name2key, ['chinese', 'english'])


# ------------------- main process -------------------
def normalize_dataset_format(data_root, save_root):
    # gain paths
    video_root = os.path.join(data_root, 'Raw')
    label_path = os.path.join(data_root, 'metadata/sentiment/label_M.csv')
    train_idx_path = os.path.join(data_root, 'metadata/train_index.csv')
    val_idx_path = os.path.join(data_root, 'metadata/val_index.csv')
    test_idx_path = os.path.join(data_root, 'metadata/test_index.csv')
    trans_path = os.path.join(data_root, 'metadata/Translation.csv')

    # read all items
    video_paths, labels = read_labels(label_path, video_root)
    train_video, train_label = gain_sub_items(video_paths, labels, train_idx_path)
    val_video,   val_label   = gain_sub_items(video_paths, labels, val_idx_path)
    test_video,  test_label  = gain_sub_items(video_paths, labels, test_idx_path)

    ## output path
    save_video = os.path.join(save_root, 'video')
    save_label = os.path.join(save_root, 'label.npz')
    save_trans = os.path.join(save_root, 'transcription.csv')
    if not os.path.exists(save_root):  os.makedirs(save_root)
    if not os.path.exists(save_video): os.makedirs(save_video)

    ## generate new transcripts
    update_transcription(trans_path, save_trans)

    ## generate label path
    whole_corpus = {}
    for name, video_paths, labels in [('train', train_video, train_label),
                                      ('val',   val_video,   val_label  ),
                                      ('test',  test_video,  test_label )]:
        whole_corpus[name] = {}        
        print (f'{name}: sample number: {len(video_paths)}')
        for ii, video_path in enumerate(video_paths):
            video_name = video_path.split('/')[-2]
            clip_name  = video_path.split('/')[-1]
            save_path  = os.path.join(save_video, f'{video_name}_{clip_name}')
            shutil.copy(video_path, save_path)

            save_name  = os.path.basename(save_path)[:-4]
            whole_corpus[name][save_name] = {'emo': 0, 'val': labels[ii]}
            
    np.savez_compressed(save_label,
                        train_corpus=whole_corpus['train'],
                        val_corpus=whole_corpus['val'],
                        test_corpus=whole_corpus['test'])

if __name__ == '__main__':
    # data_root = '/data/lianzheng/chinese-mer-2023/CH-SIMS'
    # save_root = '/data/lianzheng/chinese-mer-2023/CH-SIMS-process'
    # normalize_dataset_format(data_root, save_root)

    # data_root = 'H:\\desktop\\Multimedia-Transformer\\chinese-mer-2023\\dataset\\sims-process'
    # trans_path = os.path.join(data_root, 'transcription.csv')
    # polish_path = os.path.join(data_root, 'transcription-engchi-polish.csv')
    # func_translate_transcript_polish_merge(trans_path, polish_path)

    # check for label range => should be [-1, 1]
    label_path = config.PATH_TO_LABEL['SIMS']
    train_corpus = np.load(label_path, allow_pickle=True)['train_corpus'].tolist()
    val_corpus = np.load(label_path, allow_pickle=True)['val_corpus'].tolist()
    test_corpus = np.load(label_path, allow_pickle=True)['test_corpus'].tolist()
    corpus = {**train_corpus, **val_corpus, **test_corpus}
    labels = [corpus[name]['val'] for name in corpus]
    print (f'min:{min(labels)}; max:{max(labels)}')
    '[-1, 1]'