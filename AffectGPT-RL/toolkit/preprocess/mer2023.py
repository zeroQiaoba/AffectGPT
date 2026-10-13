import os
import glob
import shutil
from toolkit.globals import *
from toolkit.utils.read_files import *
from toolkit.utils.functions import *

def normalize_dataset_format(data_root, save_root):
    ## input path
    train_video, train_label = os.path.join(data_root, 'train'), os.path.join(data_root, 'train-label.csv')
    test1_video, test1_label = os.path.join(data_root, 'test1'), os.path.join(data_root, 'test1-label.csv')
    test2_video, test2_label = os.path.join(data_root, 'test2'), os.path.join(data_root, 'test2-label.csv')
    test3_video, test3_label = os.path.join(data_root, 'test3'), os.path.join(data_root, 'test3-label.csv')

    ## output path
    save_video = os.path.join(save_root, 'video')
    save_label = os.path.join(save_root, 'label-6way.npz')
    if not os.path.exists(save_root):  os.makedirs(save_root)
    if not os.path.exists(save_video): os.makedirs(save_video)

    ## generate label path
    whole_corpus = {}
    for name, video_root, label_path in [('train', train_video, train_label),
                                         ('test1', test1_video, test1_label),
                                         ('test2', test2_video, test2_label),
                                         ('test3', test3_video, test3_label)]:
        
        whole_corpus[name] = {}
        names = func_read_key_from_csv(label_path, 'name')
        emos  = func_read_key_from_csv(label_path, 'discrete')
        vals  = func_read_key_from_csv(label_path, 'valence')
        # process for test3 [test3 do not have vals]
        if name == 'test3': vals = [-10] * len(names)
        print (f'{name}: sample number: {len(names)}')
        for ii in range(len(names)):
            whole_corpus[name][names[ii]] = {'emo': emos[ii], 'val': vals[ii]}
            # copy video
            video_path = glob.glob(os.path.join(video_root, f'{names[ii]}*'))[0]
            video_name = os.path.basename(video_path)
            new_path = os.path.join(save_video, video_name)
            shutil.copy(video_path, new_path)

    np.savez_compressed(save_label,
                        train_corpus=whole_corpus['train'],
                        test1_corpus=whole_corpus['test1'],
                        test2_corpus=whole_corpus['test2'],
                        test3_corpus=whole_corpus['test3'])

if __name__ == '__main__':
    # data_root = '/data/lianzheng/chinese-mer-2023/mer2023-dataset'
    # save_root = '/data/lianzheng/chinese-mer-2023/mer2023-dataset-process'
    # normalize_dataset_format(data_root, save_root)

    ## check for label mapping
    label_path = config.PATH_TO_LABEL['MER2023']
    train_corpus = np.load(label_path, allow_pickle=True)['train_corpus'].tolist()
    labels = [train_corpus[name]['emo'] for name in train_corpus]
    func_discrte_label_distribution(labels)
    '''
    mer_mapping: ['neutral', 'angry', 'happy', 'sad', 'worried',  'surprise']
    '''