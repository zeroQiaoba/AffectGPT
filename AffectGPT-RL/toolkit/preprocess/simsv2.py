import os
import shutil
from toolkit.utils.functions import *
from toolkit.utils.read_files import *

def func_merge_id_to_path(video_id, clip_id, video_root):
    video_path = os.path.join(video_root, video_id, clip_id+'.mp4')
    return video_path

def func_convert_path_to_newname(video_id, clip_id):
    newname = f'{video_id}_{clip_id}'
    return newname

# label_path -> (video_paths, labels)
def read_labels(label_path, video_root):

    # read all items
    video_ids = func_read_key_from_csv(label_path, 'video_id')
    clip_ids = func_read_key_from_csv(label_path, 'clip_id')
    chis = func_read_key_from_csv(label_path, 'text')
    labels = func_read_key_from_csv(label_path, 'label')
    modes = func_read_key_from_csv(label_path, 'mode')

    print (f'label range ->  min:{min(labels)}  max:{max(labels)}')
    print (f'whole sample number: {len(labels)}')
    print ('modes: ', set(modes))

    newnames, videopaths = [], []
    for ii in range(len(video_ids)):
        newname = func_convert_path_to_newname(video_ids[ii], clip_ids[ii])
        videopath = func_merge_id_to_path(video_ids[ii], clip_ids[ii], video_root)
        newnames.append(newname)
        videopaths.append(videopath)
    print (f'whole sample number: {len(set(newnames))}')
    return chis, labels, modes, videopaths, newnames


# ------------------- main process -------------------
def normalize_dataset_format(data_root, save_root):
    # gain paths
    video_root = os.path.join(data_root, 'Raw')
    label_path = os.path.join(data_root, 'meta.csv')
    
    # read all items
    chis, labels, modes, videopaths, newnames = read_labels(label_path, video_root)

    ## output path
    save_video = os.path.join(save_root, 'video')
    save_label = os.path.join(save_root, 'label.npz')
    save_trans = os.path.join(save_root, 'transcription.csv')
    if not os.path.exists(save_root):  os.makedirs(save_root)
    if not os.path.exists(save_video): os.makedirs(save_video)

    ## generate new transcripts
    name2key = {}
    for ii, newname in enumerate(newnames):
        name2key[newname] = [chis[ii]]
    func_write_key_to_csv(save_trans, newnames, name2key, ['chinese'])

    ## copy videos
    for ii, videopath in enumerate(videopaths):
        assert videopath.endswith('.mp4')
        savepath = os.path.join(save_video, newnames[ii]+'.mp4')
        shutil.copy(videopath, savepath)

    ## generate label path
    whole_corpus = {}
    for ii, newname in enumerate(newnames):
        mode = modes[ii] # [train, valid, test]
        if mode not in whole_corpus: 
            whole_corpus[mode] = {}
        whole_corpus[mode][newname] = {'emo': 0, 'val': labels[ii]}

    np.savez_compressed(save_label,
                        train_corpus=whole_corpus['train'],
                        val_corpus=whole_corpus['valid'],
                        test_corpus=whole_corpus['test'])

if __name__ == '__main__':
    # data_root = 'I:\\CH-SIMS-v2\\zip\\supervised\\ch-simsv2s'
    # save_root = 'E:\\Dataset\\simsv2-process'
    # normalize_dataset_format(data_root, save_root)

    # data_root = 'E:\\Dataset\\simsv2-process'
    # trans_path = os.path.join(data_root, 'transcription.csv')
    # polish_path = os.path.join(data_root, 'transcription-engchi-polish.csv')
    # func_translate_transcript_polish_merge(trans_path, polish_path)
    # func_translate_transcript_polish_merge(polish_path, '')

    # check for label range => should be [-1, 1]
    label_path = config.PATH_TO_LABEL['SIMSv2']
    train_corpus = np.load(label_path, allow_pickle=True)['train_corpus'].tolist()
    val_corpus = np.load(label_path, allow_pickle=True)['val_corpus'].tolist()
    test_corpus = np.load(label_path, allow_pickle=True)['test_corpus'].tolist()
    corpus = {**train_corpus, **val_corpus, **test_corpus}
    labels = [corpus[name]['val'] for name in corpus]
    print (f'min:{min(labels)}; max:{max(labels)}')
    '[-1, 1]'