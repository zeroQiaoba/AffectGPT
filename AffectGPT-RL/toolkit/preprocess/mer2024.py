import os
import glob
import shutil
from toolkit.globals import *
from toolkit.utils.read_files import *
from toolkit.utils.functions import *


## 解析 train_label
def read_train_label(train_label):
    train_names = func_read_key_from_csv(train_label, 'name')
    train_emos  = func_read_key_from_csv(train_label, 'discrete')
    print(f'train: {len(train_names)}')
    return train_names, train_emos

## 解析 semi_label
def split_semi_label(semi_label):
    names = func_read_key_from_csv(semi_label, 'name')
    emos  = func_read_key_from_csv(semi_label, 'discrete')
    print(f'semi: {len(names)}')

    ## split to test1 and test2
    names, emos  = np.array(names), np.array(emos)
    indices = np.arange(len(names))
    random.shuffle(indices)
    test1_names, test1_emos = names[:int(len(names)/2)], emos[:int(len(names)/2)]
    test2_names, test2_emos = names[int(len(names)/2):], emos[int(len(names)/2):]
    print(f'test1: {len(test1_names)}')
    print(f'test2: {len(test2_names)}')
    return test1_names, test1_emos, test2_names, test2_emos


def normalize_dataset_format(data_root, save_root):

    ## input path
    train_video, train_label = os.path.join(data_root, 'video-labeled'),   os.path.join(data_root, 'label-disdim.csv')
    semi_video,  semi_label  = os.path.join(data_root, 'video-unlabeled'), os.path.join(data_root, 'semi-label.csv')

    ## step1: 将semi分成两部分，一部分作为test1, 一部分作为test2
    train_names, train_emos = read_train_label(train_label)
    test1_names, test1_emos, test2_names, test2_emos = split_semi_label(semi_label)

    ## output path
    save_train = os.path.join(save_root, 'video-train')
    save_test1 = os.path.join(save_root, 'video-test1')
    save_test2 = os.path.join(save_root, 'video-test2')
    save_label = os.path.join(save_root, 'label-6way.npz')
    if not os.path.exists(save_root):  os.makedirs(save_root)
    if not os.path.exists(save_train): os.makedirs(save_train)
    if not os.path.exists(save_test1): os.makedirs(save_test1)
    if not os.path.exists(save_test2): os.makedirs(save_test2)

    ## generate label path
    whole_corpus = {}
    for (subset, video_root, names, labels, save_root) in [ ('train', train_video, train_names, train_emos, save_train),
                                                            ('test1', semi_video,  test1_names, test1_emos, save_test1),
                                                            ('test2', semi_video,  test2_names, test2_emos, save_test2)]:
        
        whole_corpus[subset] = {}
        print (f'{subset}: sample number: {len(names)}')
        for (name, label) in zip(names, labels):
            whole_corpus[subset][name] = {'emo': label}
            # copy video
            video_path = os.path.join(video_root, f'{name}.mp4')
            assert os.path.exists(video_path), f'video does not exist.'
            video_name = os.path.basename(video_path)
            new_path = os.path.join(save_root, video_name)
            shutil.copy(video_path, new_path)

    ## 需要对视频进行加噪处理
    np.savez_compressed(save_label,
                        train_corpus=whole_corpus['train'],
                        test1_corpus=whole_corpus['test1'],
                        test2_corpus=whole_corpus['test2'])


def generate_final_test1_test2_gt(data_root):

    for test_name in ['test1', 'test2']:
        ## read name2emo
        name2emo = {}
        label_path = os.path.join(data_root, 'label-6way.npz')
        test_labels = np.load(label_path, allow_pickle=True)[f'{test_name}_corpus'].tolist()
        for name in test_labels:
            emo = test_labels[name]['emo']
            name2emo[name] = emo
        
        ## read name_mapping
        mapping_path = os.path.join(data_root, 'old2new-release.npz')
        mapping = np.load(mapping_path, allow_pickle=True)['old2new'].tolist()

        ## store to csv
        save_path = os.path.join(data_root, f'{test_name}_name_convert_label.csv')
        whole_names = [mapping[name] for name in name2emo]
        name2key = {}
        for name in name2emo:
            name2key[mapping[name]] = name2emo[name]

        # save to csv
        keynames = ['discrete']
        func_write_key_to_csv(save_path, whole_names, name2key, keynames)


# run -d toolkit/preprocess/mer2024.py
if __name__ == '__main__':

    # 步骤1：基本数据库生成
    # data_root = '/share/home/lianzheng/chinese-mer-2023/dataset/mer2024-dataset'
    # save_root = '/share/home/lianzheng/chinese-mer-2023/dataset/mer2024-dataset-process'
    # normalize_dataset_format(data_root, save_root)

    # 步骤2：给 test2 加噪，然后再把所有数据集合并 [因为这块需要多进程操作]
    '''
    python main-check.py main_mixture_multiprocess 
            --video_root='/share/home/lianzheng/chinese-mer-2023/dataset/mer2024-dataset-process/video-test2' 
            --save_root='/share/home/lianzheng/chinese-mer-2023/dataset/mer2024-dataset-process/video-test2-noise' 
    '''

    # 步骤3：将所有数据集放在 /share/home/lianzheng/chinese-mer-2023/dataset/mer2024-dataset-process/video 

    # 步骤4：检查 [ok]
    # label_path = config.PATH_TO_LABEL['MER2024']
    # for subset in ['train_corpus', 'test1_corpus', 'test2_corpus']:
    #     train_corpus = np.load(label_path, allow_pickle=True)[subset].tolist()
    #     labels = [train_corpus[name]['emo'] for name in train_corpus]

    #     # 检测标签分布情况 [ok]
    #     func_discrte_label_distribution(labels)
    #     '''
    #     都是6类标签，并且标签种类相同
    #     train_corpus: 5030
    #     test1_corpus: 1169
    #     test2_corpus: 1170
    #     total: 7369 条数据 [ok]
    #     '''

    #     # 人工挑选一些样本可视化 [ok]
    #     # names = [name for name in train_corpus]
    #     # for (name, label) in zip(names[-10:], labels[-10:]):
    #     #     print (name, label)

    # 步骤5：生成 gt label for test1 and test2 [其中涉及到 name mapping 的过程，结果存储到 .csv 中]
    # data_root = '/share/home/lianzheng/chinese-mer-2023/dataset/mer2024-dataset-process'
    # generate_final_test1_test2_gt(data_root)
    