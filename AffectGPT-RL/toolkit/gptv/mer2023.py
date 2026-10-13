import os
import re
import glob
import tqdm
import shutil
import librosa
from toolkit.utils.read_files import *
from toolkit.utils.chatgpt import get_video_emotion_batch, get_text_emotion_batch, get_multi_emotion_batch
from toolkit.utils.functions import func_label_distribution, func_plot_confusion_matrix

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score

from toolkit.globals import config
FFMPEG_PATH = config.PATH_TO_FFMPEG_Win

emos = ['worried', 'happy', 'neutral', 'angry', 'surprised', 'sad']
emo2idx = {}
for ii, emo in enumerate(emos):
    emo2idx[emo] = ii
emo2idx['surprise'] = emo2idx['surprised'] 

# 将 video 转成 mel-spec.jpgget_multi_emotion_batch
def func_video_to_melspec(video_path, mel_path):

    sample_rate=16000
    maxlen = 10 * sample_rate
    audio_path = mel_path[:-4] + '.wav'

    cmd = f"ffmpeg -i {video_path} {audio_path}"
    os.system(cmd)

    # read audio
    waveform, sr = librosa.load(audio_path)
    waveform = librosa.resample(waveform, sr, sample_rate)
    print ('waveform shape:', waveform.shape)

    # too long then clip, maxlen=10s
    if len(waveform) > maxlen:
        waveform = waveform[:maxlen]

    # change to melspec
    mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sample_rate)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spec_db, aspect='auto', origin='lower')
    plt.axis('off') # 关闭坐标轴
    plt.savefig(mel_path, bbox_inches='tight', pad_inches=0)
    
    # remove temp file
    os.remove(audio_path)


# test1: totally 411 samples
def select_samples(data_root, save_root, modality):
    
    assert modality in ['audio', 'video', 'text']
    
    # save path
    save_csv = os.path.join(save_root, 'label.csv')
    save_video = os.path.join(save_root, modality)
    if not os.path.exists(save_video): os.makedirs(save_video)

    # gain name2tran
    name2tran = {}
    trans_path = 'E:\\Dataset\\mer2023-dataset-process\\transcription-engchi-polish.csv'
    names = func_read_key_from_csv(trans_path, 'name')
    trans = func_read_key_from_csv(trans_path, 'chinese')
    for ii in range(len(names)):
        name2tran[names[ii]] = trans[ii]
    
    # read data
    names, labels, gpt4vs = [], [], []
    video_root = os.path.join(data_root, 'test1')
    label_path = os.path.join(data_root, 'test1-label.csv')
    df = pd.read_csv(label_path)
    for _, row in df.iterrows():
        name = row['name']
        label = row['discrete']
        tran = name2tran[name]
        
        # get video_path
        video_path1 = os.path.join(video_root, name + '.avi')
        video_path2 = os.path.join(video_root, name + '.mp4')
        if os.path.exists(video_path1):
            video_path = video_path1
        else:
            video_path = video_path2
        videoname = os.path.basename(video_path)

        # get save_path
        if modality == 'video':
            save_path = os.path.join(save_video, videoname)
            shutil.copy(video_path, save_path)
        elif modality == 'audio':
            save_path = os.path.join(save_video, videoname[:-4] + '.jpg')
            func_video_to_melspec(video_path, save_path)
        elif modality == 'text':
            save_path = os.path.join(save_video, videoname[:-4] + '.npy')
            np.save(save_path, tran)

        # for label
        names.append(name)
        labels.append(label)
        gpt4vs.append('')

    name2key = {}
    for ii, name in enumerate(names):
        name2key[name] = [labels[ii], gpt4vs[ii]]
    keynames = ['gt', 'gpt4v']
    func_write_key_to_csv(save_csv, names, name2key, keynames)


def func_get_response(batch, emos, modality, sleeptime, samplenum=3):
    if modality == 'video':
        response = get_video_emotion_batch(batch, emos, sleeptime, samplenum)
    elif modality == 'text':
        response = get_text_emotion_batch(batch, emos, sleeptime)
    elif modality == 'multi':
        response = get_multi_emotion_batch(batch, emos, sleeptime)
    return response

def func_get_segment_batch(batch, savename, xishu=2):
  
    segment_num = math.ceil(len(batch)/xishu)

    store = []
    for ii in range(xishu):
        segbatch = batch[ii*segment_num:(ii+1)*segment_num]
        segsave  = savename[:-4] + f"_segment_{ii+1}.npz"
        if not isinstance(segbatch, list):
            segbatch = [segbatch]
        if len(segbatch) > 0:
            store.append((segbatch, segsave))
    return store

# # 30 images may excceed the max token number of GPT-4V -> reduce to 20
def evaluate_performance_using_gpt4v(image_root, save_root, save_order, modality, bsize=20, batch_flag='flag1', sleeptime=0, xishus=[2,2,2], samplenum=3):
    # params assert
    if len(xishus) == 1: assert batch_flag in ['flag1', 'flag2']
    if len(xishus) == 2: assert batch_flag in ['flag1', 'flag2', 'flag3']
    if len(xishus) == 3: assert batch_flag in ['flag1', 'flag2', 'flag3', 'flag4']
    multiple = 1
    for item in xishus: multiple *= item
    assert multiple == bsize
    
    # create folders
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    
    # preprocess for 'multi'
    if modality == 'multi':
        image_root = os.path.split(image_root)[0] + '/video'

    # shuffle image orders
    if not os.path.exists(save_order):
        image_paths = glob.glob(image_root + '/*')
        indices = np.arange(len(image_paths))
        random.shuffle(indices)
        image_paths = np.array(image_paths)[indices]
        np.savez_compressed(save_order, image_paths=image_paths)
    else:
        image_paths = np.load(save_order, allow_pickle=True)['image_paths'].tolist()
    print (f'process sample numbers: {len(image_paths)}') # 981

    # split int batch [20 samples per batch]
    batches = []
    splitnum = math.ceil(len(image_paths) / bsize)
    for ii in range(splitnum):
        batches.append(image_paths[ii*bsize:(ii+1)*bsize])
    print (f'process batch  number: {len(batches)}') # 50 batches
    print (f'process sample number: {sum([len(batch) for batch in batches])}')
    
    # generate predictions for each batch and store
    for ii, batch in tqdm.tqdm(enumerate(batches)):
        save_path = os.path.join(save_root, f'batch_{ii+1}.npz')
        if os.path.exists(save_path): continue
        ## batch not exists -> how to deal with these false batches
        if batch_flag == 'flag1': # process the whole batch again # 20
            response = func_get_response(batch, emos, modality, sleeptime, samplenum)
            np.savez_compressed(save_path, gpt4v=response, names=batch)
        elif batch_flag == 'flag2': # split and process # 10
            stores = func_get_segment_batch(batch, save_path, xishu=xishus[0])
            for (segbatch, segsave) in stores:
                if os.path.exists(segsave): continue
                response = func_get_response(segbatch, emos, modality, sleeptime, samplenum)
                np.savez_compressed(segsave, gpt4v=response, names=segbatch)
        elif batch_flag == 'flag3': # split and process # 5
            stores = func_get_segment_batch(batch, save_path, xishu=xishus[0])
            for (segbatch, segsave) in stores:
                if os.path.exists(segsave): continue
                newstores = func_get_segment_batch(segbatch, segsave, xishu=xishus[1])
                for (newbatch, newsave) in newstores:
                    if os.path.exists(newsave): continue
                    response = func_get_response(newbatch, emos, modality, sleeptime, samplenum)
                    np.savez_compressed(newsave, gpt4v=response, names=newbatch)
        elif batch_flag == 'flag4': # split and process # 5
            stores = func_get_segment_batch(batch, save_path, xishu=xishus[0])
            for (segbatch, segsave) in stores:
                if os.path.exists(segsave): continue
                newstores = func_get_segment_batch(segbatch, segsave, xishu=xishus[1])
                for (newbatch, newsave) in newstores:
                    if os.path.exists(newsave): continue
                    new2stores = func_get_segment_batch(newbatch, newsave, xishu=xishus[2])
                    for (new2batch, new2save) in new2stores:
                        if os.path.exists(new2save): continue
                        response = func_get_response(new2batch, emos, modality, sleeptime, samplenum)
                        np.savez_compressed(new2save, gpt4v=response, names=new2batch)
                            

def func_analyze_gpt4v_outputs(gpt_path):
    
    names = np.load(gpt_path, allow_pickle=True)['names'].tolist()

    ## analyze gpt-4v
    store_results = []
    gpt4v = np.load(gpt_path, allow_pickle=True)['gpt4v'].tolist()
    gpt4v = gpt4v.replace("name",    "==========")
    gpt4v = gpt4v.replace("result",  "==========")
    gpt4v = gpt4v.split("==========")
    for line in gpt4v:
        if line.find('[') != -1:
            res = line.split('[')[1]
            res = res.split(']')[0]
            store_results.append(res)
    
    return names, store_results
    
def check_gpt4_performance(gpt4v_root):
    whole_names, whole_gpt4vs = [], []
    for gpt_path in sorted(glob.glob(gpt4v_root + '/*')):
        names, gpt4vs = func_analyze_gpt4v_outputs(gpt_path)
        print (f'number of samples: {len(names)} number of results: {len(gpt4vs)}')
        if len(names) == len(gpt4vs): 
            names = [os.path.basename(name) for name in names]
            whole_names.extend(names)
            whole_gpt4vs.extend(gpt4vs)
        else:
            print (f'error batch: {gpt_path}. Need re-test!!')
            os.system(f'rm -rf {gpt_path}')
    return whole_names, whole_gpt4vs


def get_results_and_update_label(gpt4v_root, label_path, store_path):
    ## read label_path
    names  = func_read_key_from_csv(label_path, 'name')
    labels = func_read_key_from_csv(label_path, 'gt')
    print (f'gt sample number: {len(names)}') # 411
    name2label = {}
    for ii in range(len(names)):
        name2label[names[ii]] = labels[ii]

    ## read prediction
    whole_names, whole_gpt4vs = check_gpt4_performance(gpt4v_root)
    whole_names = [name[:-4] for name in whole_names] # remove type
    print (f'pred sample number: {len(whole_names)}') # 411

    ## gain acc
    acc = 0
    name2key = {}
    for ii, name in enumerate(whole_names):
        gt = name2label[name]
        ## process for pred
        pred = whole_gpt4vs[ii]
        pred = [item for item in re.split('[\'\"]', pred) if item.strip() not in ['', ',']]
        if len(pred) != 5:
            print (whole_gpt4vs[ii])
        top1pred = pred[0]
        if top1pred == gt:
            acc += 1
        name2key[name] = [gt, ",".join(pred)]
    print ('gtp4v accuracy: %.2f' %(acc/len(whole_names)*100))
    keynames = ['gt', 'gpt4v']
    func_write_key_to_csv(store_path, whole_names, name2key, keynames)


def further_report_results(gpt4v_csv):
    names = func_read_key_from_csv(gpt4v_csv, 'name')
    gts = func_read_key_from_csv(gpt4v_csv, 'gt')
    gpt4vs = func_read_key_from_csv(gpt4v_csv, 'gpt4v')
    print (f'sample number: {len(names)}')

    # gpt4v
    test_preds, test_labels = [], []
    for ii in range(len(names)):
        test_preds.append(emo2idx[gpt4vs[ii].split(',')[0]])
        test_labels.append(emo2idx[gts[ii]])
    fscore = f1_score(test_labels, test_preds, average='weighted')
    print ('gtp4v waf: %.2f' %(fscore*100))

    # random guess
    whole_wafs = []
    candidate_labels = emos
    for repeat in range(10): # remove randomness
        test_preds = []
        for ii in range(len(names)):
            pred = candidate_labels[random.randint(0, len(candidate_labels)-1)]
            test_preds.append(emo2idx[pred])
        test_preds = np.array(test_preds)
        fscore = f1_score(test_labels, test_preds, average='weighted')
        whole_wafs.append(fscore)
    print ('random guess waf: %.2f' %(np.mean(whole_wafs)*100))

    # frequent guess
    label2count = func_label_distribution(gts)
    labels, counts = [], []
    for label in label2count:
        labels.append(label)
        counts.append(label2count[label])
    majorlabel = emo2idx[labels[np.argmax(counts)]]
    test_preds = np.array([majorlabel] * len(names))
    fscore = f1_score(test_labels, test_preds, average='weighted')
    print ('frequent guess waf: %.2f' %(fscore*100))


def plot_confusion_matrix(gpt4v_csv, save_path, fontsize, cmap):
    names = func_read_key_from_csv(gpt4v_csv, 'name')
    gts = func_read_key_from_csv(gpt4v_csv, 'gt')
    gpt4vs = func_read_key_from_csv(gpt4v_csv, 'gpt4v')

    emos = ['worried', 'happy', 'neutral', 'angry', 'surprised', 'sad']
    labels, preds = [], []
    for ii in range(len(names)):
        gt = gts[ii]
        if gt == 'surprise': gt = 'surprised'
        pred = gpt4vs[ii].split(',')[0]
        if pred not in emos: continue
        labels.append(gt)
        preds.append(pred)
    print (f'whole sample number: {len(names)}')
    print (f'process sample number: {len(labels)}')

    emo2idx = {}
    for idx, emo in enumerate(emos): emo2idx[emo] = idx
    labels = [emo2idx[label] for label in labels]
    preds = [emo2idx[pred] for pred in preds]
    target_names = [emo.lower() for emo in emos]
    func_plot_confusion_matrix(labels, preds, target_names, save_path, fontsize, cmap)


if __name__ == '__main__':

    ## step1: pre-process for dataset [for multimodal dataset, firstly process on each modality]
    # data_root = 'E:\\Dataset\\mer2023-dataset'
    # save_root = 'E:\\Dataset\\gpt4v-evaluation\\mer2023'
    # select_samples(data_root, save_root, 'video')
    # select_samples(data_root, save_root, 'audio')
    # select_samples(data_root, save_root, 'text')

    ## step2: gain prediction results [ok Kang's machine]
    save_root = '/root/dataset/mer2023'
    # for (modality, bsize, xishus) in [('video', 8, [2, 2, 2]),
    #                                   ('text', 20, [2, 2, 5]),
    #                                   ('multi', 6, [2, 3])]:
    #     if modality in ['video', 'text']:
    #         flags = ['flag1', 'flag1', 'flag1', 'flag2', 'flag2', 'flag3', 'flag4']
    #     else:
    #         flags = ['flag1', 'flag1', 'flag1', 'flag2', 'flag2', 'flag3']

    #     # process for each modality
    #     image_root = os.path.join(save_root, modality)
    #     gpt4v_root = os.path.join(save_root, f'{modality}-gpt4v')
    #     save_order = os.path.join(save_root, f'{modality}-order.npz')
    #     for flag in flags:
    #         evaluate_performance_using_gpt4v(image_root, gpt4v_root, save_order, modality, bsize=bsize, batch_flag=flag, sleeptime=20, xishus=xishus)
    #         check_gpt4_performance(gpt4v_root)

    ## step3: 比较 gpt4v outputs 与 labels.cvs 的结果差异
    # for modality in ['video', 'text', 'multi']:
    #     label_path = os.path.join(save_root, 'label.csv')
    #     store_path = os.path.join(save_root, f'label-gpt4v-{modality}.csv')
    #     gpt4v_root = os.path.join(save_root, f'{modality}-gpt4v')
    #     get_results_and_update_label(gpt4v_root, label_path, store_path)

    # 411 => mer2023-video: 411 gtp4v performance: 45.50
    # 411 => mer2023-text:  411 gtp4v performance: 34.06
    # 411 => mer2023-multi: 411 gtp4v performance: 63.50


    ## 进一步分析结果并得到更多统计信息
    # gpt4v_path = 'E:\\Dataset\\gpt4v-evaluation\\mer2023\label-gpt4v-multi.csv'
    # further_report_results(gpt4v_path)
    '''
    sample number: 411
    gtp4v waf: 65.39
    random guess waf: 17.87
    frequent guess waf: 10.40
    '''

    # gpt4v_path = 'E:\\Dataset\\gpt4v-evaluation\\mer2023\label-gpt4v-text.csv'
    # further_report_results(gpt4v_path)
    '''
    sample number: 411
    gtp4v waf: 34.57
    '''

    # gpt4v_path = 'E:\\Dataset\\gpt4v-evaluation\\mer2023\label-gpt4v-video.csv'
    # further_report_results(gpt4v_path)
    '''
    sample number: 411
    gtp4v waf: 46.23
    '''

    ## 绘制混淆矩阵图，进一步分析GPT4V的预测结果
    # gpt4v_path = 'E:\\Dataset\\gpt4v-evaluation\\mer2023\\label-gpt4v-multi.csv'
    # save_path  = 'E:\\Dataset\\gpt4v-evaluation\\mer2023\\mer2023-multi-cm.png'
    # plot_confusion_matrix(gpt4v_path, save_path)


    #########################################################################
    ## 测试不同采样帧数的影响 => 仅测试 video [最多同时输入24张图片]
    #########################################################################
    ## gpt4v prediction
    for (modality, bsize, xishus, samplenum) in [
                                                 ('video', 8, [2, 2, 2], 2),
                                                 ('video', 8, [2, 2, 2], 3), # default
                                                 ('video', 6, [2, 3],    4),
                                                 ('video', 3, [3],       8),
                                                ]:
        print (f'uniform sampled frames number: {samplenum}')
        if len(xishus) == 1:
            flags = ['flag1', 'flag1', 'flag1', 'flag2']
        elif len(xishus) == 2:
            flags = ['flag1', 'flag1', 'flag1', 'flag2', 'flag2', 'flag3']
        elif len(xishus) == 3:
            flags = ['flag1', 'flag1', 'flag1', 'flag2', 'flag2', 'flag3', 'flag4']
        image_root = os.path.join(save_root, modality)
        gpt4v_root = os.path.join(save_root, f'{modality}-gpt4v-samplenum{samplenum}')
        save_order = os.path.join(save_root, f'{modality}-order-samplenum{samplenum}.npz')

        ## step1: process for each modality
        # for flag in flags:
        #     evaluate_performance_using_gpt4v(image_root, gpt4v_root, save_order, modality, bsize=bsize, 
        #                                      batch_flag=flag, sleeptime=20, xishus=xishus, samplenum=samplenum)
        #     check_gpt4_performance(gpt4v_root)
        
        ## step2: calculate results
        label_path = os.path.join(save_root, 'label.csv')
        store_path = os.path.join(save_root, f'label-gpt4v-{modality}-samplenum{samplenum}.csv')
        get_results_and_update_label(gpt4v_root, label_path, store_path)