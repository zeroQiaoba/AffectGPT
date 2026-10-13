import os
import re
import cv2
import glob
import tqdm
import shutil
from sklearn.metrics import f1_score

from toolkit.utils.read_files import *
from toolkit.utils.chatgpt import get_video_emotion_batch, get_text_emotion_batch, get_multi_emotion_batch
from toolkit.utils.functions import func_label_distribution, func_plot_confusion_matrix

emos = ['negative', 'weakly negative', 'neutral', 'weakly positive', 'positive']

# def func_uniform_sample_frames(video_path, save_root):

#     # read all frames
#     frames = []
#     cap = cv2.VideoCapture(video_path)
#     while True:
#         ret, frame = cap.read()
#         if ret == False: break
#         frames.append(frame)
#     cap.release()
    
#     ## gain select frames
#     indexes = [0, int(len(frames)/2), len(frames)-1]
#     select_frames = [frames[index] for index in indexes]

#     ## copy into the tgt root
#     if not os.path.exists(save_root):
#         os.makedirs(save_root)
#     for ii, frame in enumerate(select_frames):
#         save_path = os.path.join(save_root, '%06d.jpg' %(ii))
#         cv2.imwrite(save_path, frame)

def select_samples(data_root, save_root, modality):
    
    assert modality in ['video', 'text']
    
    # save path
    save_csv = os.path.join(save_root, 'label.csv')
    save_video = os.path.join(save_root, modality)
    if not os.path.exists(save_video): os.makedirs(save_video)

    # gain name2tran
    name2tran = {}
    trans_path = os.path.join(data_root, 'transcription-engchi-polish.csv')
    names = func_read_key_from_csv(trans_path, 'name')
    trans = func_read_key_from_csv(trans_path, 'chinese')
    for ii in range(len(names)):
        name2tran[names[ii]] = trans[ii]
    
    # read data
    names, labels, gpt4vs = [], [], []
    video_root = os.path.join(data_root, 'video')
    label_path = os.path.join(data_root, 'label.npz')
    corpus = np.load(label_path, allow_pickle=True)['test_corpus'].tolist()
    for name in corpus:
        tran = name2tran[name]
        label = corpus[name]['val']

        # get video_path
        video_path = os.path.join(video_root, name + '.mp4')
        videoname = os.path.basename(video_path)

        # get save_path
        if modality == 'video':
            # save_path = os.path.join(save_video, videoname[:-4])
            # func_uniform_sample_frames(video_path, save_path)
            save_path = os.path.join(save_video, videoname)
            shutil.copy(video_path, save_path)
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

# by default, we sample 3 frames for each video
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

def test_segment_batch(vlen, xishu=2):
    
    batch = [1] * vlen
    segment_num = math.ceil(len(batch)/xishu)

    store = []
    for ii in range(xishu):
        segbatch = batch[ii*segment_num:(ii+1)*segment_num]
        if not isinstance(segbatch, list):
            segbatch = [segbatch]
        if len(segbatch) > 0:
            store.append((segbatch))
    print ([len(segbatch) for segbatch in store])
    return store


# samplenum = 3: By default, we sample 3 frames for each video
# bsize = xishus 乘积，从而实现分级处理
def evaluate_performance_using_gpt4v(image_root, save_root, save_order, modality, bsize=20, batch_flag='flag1', sleeptime=0, xishus=[2,2,2], samplenum=3):
    # params assert
    if len(xishus) == 0: assert batch_flag in ['flag1']
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
    error_num = 0
    whole_names, whole_gpt4vs = [], []
    for gpt_path in sorted(glob.glob(gpt4v_root + '/*')):
        names, gpt4vs = func_analyze_gpt4v_outputs(gpt_path)
        # print (f'number of samples: {len(names)} number of results: {len(gpt4vs)}')
        if len(names) == len(gpt4vs): 
            names = [os.path.basename(name) for name in names]
            whole_names.extend(names)
            whole_gpt4vs.extend(gpt4vs)
        else:
            # print (f'error batch: {gpt_path}. Need re-test!!')
            os.system(f'rm -rf {gpt_path}')
            error_num += 1
    # print (f'error number: {error_num}')
    return whole_names, whole_gpt4vs


def get_results_and_update_label(gpt4v_root, label_path, store_path):
    ## read label_path
    names  = func_read_key_from_csv(label_path, 'name')
    labels = func_read_key_from_csv(label_path, 'gt')
    print (f'raw sample number: {len(set(names))}')
    name2label = {}
    for ii in range(len(names)):
        name2label[names[ii]] = labels[ii]
    
    ## read prediction
    whole_names, whole_gpt4vs = check_gpt4_performance(gpt4v_root)
    whole_names = [name.rsplit('.', 1)[0] for name in whole_names] # remove type
    print (f'predict sample number: {len(set(whole_names))}')
    assert len(whole_names) == len(set(whole_names))

    ## gain performance
    name2key = {}
    test_labels, test_preds = [], []
    for ii, name in enumerate(whole_names):
        gt = name2label[name]
        test_labels.append(gt)
        ## process for pred
        pred = whole_gpt4vs[ii]
        pred = [item for item in re.split('[\'\"]', pred) if item.strip() not in ['', ',']]
        if len(pred) != 5:
            print (whole_gpt4vs[ii])

        name2key[name] = [gt, ",".join(pred)]
        top1pred = pred[0]
        if top1pred.find('negative') != -1:
            test_preds.append(-1)
        elif top1pred.find('positive') != -1:
            test_preds.append(1)
        elif top1pred.find('neutral') != -1:
            test_preds.append(0)
        else:
            print ('Error: prediction contains unrecognized samples')

    # show statistics
    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)
    non_zeros = np.array([i for i, e in enumerate(test_labels) if e != 0]) # remove 0, and remove mask
    fscore = f1_score((test_labels[non_zeros] > 0), (test_preds[non_zeros] > 0), average='weighted')
    print ('gtp4v performance: %.2f' %(fscore*100))

    # save to csv
    keynames = ['gt', 'gpt4v']
    func_write_key_to_csv(store_path, whole_names, name2key, keynames)


def further_report_results(gpt4v_csv):
    names = func_read_key_from_csv(gpt4v_csv, 'name')
    gts = func_read_key_from_csv(gpt4v_csv, 'gt')
    gpt4vs = func_read_key_from_csv(gpt4v_csv, 'gpt4v')
    print (f'sample number: {len(names)}')

    # gpt4v
    test_preds = []
    for ii in range(len(names)):
        top1pred = gpt4vs[ii].split(',')[0]
        if top1pred.find('negative') != -1:
            test_preds.append(-1)
        elif top1pred.find('positive') != -1:
            test_preds.append(1)
        elif top1pred.find('neutral') != -1:
            test_preds.append(0)
        else:
            print ('Error: prediction contains unrecognized samples')
    test_preds = np.array(test_preds)
    test_labels = np.array(gts)
    non_zeros = np.array([i for i, e in enumerate(test_labels) if e != 0]) # remove 0, and remove mask
    fscore = f1_score((test_labels[non_zeros] > 0), (test_preds[non_zeros] > 0), average='weighted')
    print ('gtp4v waf: %.2f' %(fscore*100))

    # random guess
    whole_wafs = []
    candidate_labels = [-1, 1]
    for repeat in range(10): # remove randomness
        test_preds = []
        for ii in range(len(names)):
            pred = candidate_labels[random.randint(0, len(candidate_labels)-1)]
            test_preds.append(pred)
        test_preds = np.array(test_preds)
        test_labels = np.array(gts)
        non_zeros = np.array([i for i, e in enumerate(test_labels) if e != 0]) # remove 0, and remove mask
        fscore = f1_score((test_labels[non_zeros] > 0), (test_preds[non_zeros] > 0), average='weighted')
        whole_wafs.append(fscore)
    print ('random guess waf: %.2f' %(np.mean(whole_wafs)*100))

    # frequent guess
    fake_gts = []
    for gt in gts:
        if gt > 0:
            fake_gt = 1
        elif gt < 0:
            fake_gt = -1
        fake_gts.append(fake_gt)
    label2count = func_label_distribution(fake_gts)
    labels, counts = [], []
    for label in label2count:
        labels.append(label)
        counts.append(label2count[label])
    majorlabel = labels[np.argmax(counts)]
    test_preds = np.array([majorlabel] * len(names))
    test_labels = np.array(gts)
    non_zeros = np.array([i for i, e in enumerate(test_labels) if e != 0]) # remove 0, and remove mask
    fscore = f1_score((test_labels[non_zeros] > 0), (test_preds[non_zeros] > 0), average='weighted')
    print ('frequent guess waf: %.2f' %(fscore*100))


def plot_confusion_matrix(gpt4v_csv, save_path, fontsize, cmap):
    names = func_read_key_from_csv(gpt4v_csv, 'name')
    gts = func_read_key_from_csv(gpt4v_csv, 'gt')
    gpt4vs = func_read_key_from_csv(gpt4v_csv, 'gpt4v')

    test_preds = []
    for ii in range(len(names)):
        top1pred = gpt4vs[ii].split(',')[0]
        if top1pred.find('negative') != -1:
            test_preds.append(-1)
        elif top1pred.find('positive') != -1:
            test_preds.append(1)
        elif top1pred.find('neutral') != -1:
            test_preds.append(0)
        else:
            print ('Error: prediction contains unrecognized samples')
    test_preds = np.array(test_preds)
    test_labels = np.array(gts)
    assert len(test_labels) == len(test_preds)
    print (f'raw sample number: {len(test_labels)}')
    
    emos = ['negative', 'positive']
    labels, preds = [], []
    for ii in range(len(test_labels)):
        if test_labels[ii] == 0:
            continue
        if test_labels[ii] > 0:
            labels.append('positive')
        else:
            labels.append('negative')
        if test_preds[ii] > 0:
            preds.append('positive')
        else:
            preds.append('negative')
    assert len(labels) == len(preds)
    print (f'selected sample number: {len(labels)}')

    emo2idx = {}
    for idx, emo in enumerate(emos): emo2idx[emo] = idx
    labels = [emo2idx[label] for label in labels]
    preds = [emo2idx[pred] for pred in preds]
    target_names = [emo.lower() for emo in emos]
    func_plot_confusion_matrix(labels, preds, target_names, save_path, fontsize, cmap)


def func_find_correct_samples(gpt4v_csv, image_root, store_root):
    names  = func_read_key_from_csv(gpt4v_csv, 'name')
    names  = [name+'.mp4' for name in names]
    gts    = func_read_key_from_csv(gpt4v_csv, 'gt')
    gpt4vs = func_read_key_from_csv(gpt4v_csv, 'gpt4v')

    true_root = os.path.join(store_root, 'true')
    false_root = os.path.join(store_root, 'false')
    if not os.path.exists(true_root): os.makedirs(true_root)
    if not os.path.exists(false_root): os.makedirs(false_root)

    for ii in range(len(names)):
        gt = gts[ii]
        pred = gpt4vs[ii].split(',')[0]
        if pred.find('negative') != -1:
            pred = -1
        elif pred.find('positive') != -1:
            pred = 1
        elif pred.find('neutral') != -1:
            pred = 0

        input_path = f'{image_root}/{names[ii]}'
        save_name = "_".join(gpt4vs[ii].split(','))
        if (gt>0 and pred>0) or (gt<0 and pred<0):
            save_path = f'{true_root}/{names[ii][:-4]}_gt_{gt}_pred_{save_name}.mp4'
            shutil.copy(input_path, save_path)
        elif (gt>0 and pred<0) or (gt<0 and pred>0):
            save_path = f'{false_root}/{names[ii][:-4]}_gt_{gt}_pred_{save_name}.mp4'
            shutil.copy(input_path, save_path)


if __name__ == '__main__':

    ## step1: pre-process for dataset [for multimodal dataset, firstly process on each modality]
    # => video too large, then we sample frames
    # data_root = '/share/home/lianzheng/chinese-mer-2023/dataset/sims-process'
    # save_root = '/share/home/lianzheng/emotion-data/gpt4v-evaluation/sims'
    # select_samples(data_root, save_root, 'video')
    # select_samples(data_root, save_root, 'text')

    ## step2: gain prediction results [ok Kang's machine]
    save_root = '/root/dataset/sims'
    # modality, bsize, xishus = 'video', 8, [2,2,2]
    # modality, bsize, xishus = 'text', 20, [2,2,5]
    # modality, bsize, xishus = 'multi', 6, [2,3]
    # image_root = os.path.join(save_root, modality)
    # gpt4v_root = os.path.join(save_root, f'{modality}-gpt4v')
    # save_order = os.path.join(save_root, f'{modality}-order.npz')
    # for flag in ['flag1', 'flag2', 'flag3', 'flag4']:
    # for flag in ['flag3']:
        # evaluate_performance_using_gpt4v(image_root, gpt4v_root, save_order, modality, bsize=bsize, batch_flag=flag, sleeptime=0, xishus=xishus)
        # check_gpt4_performance(gpt4v_root)

    ## step4: 比较 gpt4v outputs 与 labels.cvs 的结果差异
    # for modality in ['video', 'text', 'multi']:
    #     label_path = os.path.join(save_root, 'label.csv')
    #     store_path = os.path.join(save_root, f'label-gpt4v-{modality}.csv')
    #     gpt4v_root = os.path.join(save_root, f'{modality}-gpt4v')
    #     get_results_and_update_label(gpt4v_root, label_path, store_path)

    # video: 457 -> 456 -> gpt4v performance: 76.13
    # text:  457 -> 457 -> gpt4v performance: 70.07
    # multi: 457 -> 453 -> gpt4v performance: 81.24

    ## 进一步分析结果并得到更多统计信息
    # gpt4v_path = 'E:\\Dataset\\gpt4v-evaluation\\sims\label-gpt4v-multi.csv'
    # further_report_results(gpt4v_path)
    '''
    sample number: 453
    gtp4v waf: 81.24
    random guess waf: 51.23
    frequent guess waf: 50.47
    '''

    ## 绘制混淆矩阵图，进一步分析GPT4V的预测结果
    # gpt4v_path = 'E:\\Dataset\\gpt4v-evaluation\\sims\\label-gpt4v-multi.csv'
    # save_path  = 'E:\\Dataset\\gpt4v-evaluation\\sims\\sims-multi-cm.png'
    # plot_confusion_matrix(gpt4v_path, save_path)

    ## 找到拒识样本，并分析拒识case
    # from toolkit.utils.functions import func_find_refused_samples
    # label_path = 'E:\\Dataset\\gpt4v-evaluation\\sims\\label.csv'
    # gpt4v_path = 'E:\\Dataset\\gpt4v-evaluation\\sims\\label-gpt4v-multi.csv'
    # store_root = 'E:\\Dataset\\gpt4v-evaluation\\sims\\refuse_samples'
    # image_root = 'E:\\Dataset\\gpt4v-evaluation\\sims\\video'
    # func_find_refused_samples(label_path, gpt4v_path, image_root, store_root)

    ## 找到识别正确和识别错误样本并进行结果展示
    gpt4v_path = 'E:\\Dataset\\gpt4v-evaluation\\sims\\label-gpt4v-multi.csv'
    store_root = 'E:\\Dataset\\gpt4v-evaluation\\sims\\correct_samples'
    image_root = 'E:\\Dataset\\gpt4v-evaluation\\sims\\video'
    func_find_correct_samples(gpt4v_path, image_root, store_root)

    #########################################################################
    ## 测试不同采样帧数的影响 => 仅测试 video [最多同时输入24张图片]
    #########################################################################
    ## gpt4v prediction
    # # => sims samplenum=3 时候的结果变化有点大，我建议修改采样策略重新验证
    # for (modality, bsize, xishus, samplenum) in [
    #                                             #  ('video', 8, [2, 2, 2], 2),
    #                                              ('video', 8, [2, 2, 2], 3), # default
    #                                             #  ('video', 6, [2, 3],    4),
    #                                             #  ('video', 3, [3],       8),
    #                                             #  ('video', 1, [],       16),
    #                                             #  ('video', 1, [],       24),
    #                                             ]:
    #     print (f'uniform sampled frames number: {samplenum}')
    #     if len(xishus) == 0:
    #         flags = ['flag1']
    #     elif len(xishus) == 1:
    #         flags = ['flag1', 'flag1', 'flag1', 'flag2']
    #     elif len(xishus) == 2:
    #         flags = ['flag1', 'flag1', 'flag1', 'flag2', 'flag2', 'flag3']
    #     elif len(xishus) == 3:
    #         flags = ['flag1', 'flag1', 'flag1', 'flag2', 'flag2', 'flag3', 'flag4']
    #     image_root = os.path.join(save_root, modality)
    #     gpt4v_root = os.path.join(save_root, f'{modality}-gpt4v-samplenum{samplenum}')
    #     save_order = os.path.join(save_root, f'{modality}-order-samplenum{samplenum}.npz')

    #     ## step1: process for each modality
    #     # for flag in flags:
    #     #     evaluate_performance_using_gpt4v(image_root, gpt4v_root, save_order, modality, bsize=bsize, 
    #     #                                      batch_flag=flag, sleeptime=20, xishus=xishus, samplenum=samplenum)
    #     #     check_gpt4_performance(gpt4v_root)
        
    #     ## step2: calculate results
    #     label_path = os.path.join(save_root, 'label.csv')
    #     store_path = os.path.join(save_root, f'label-gpt4v-{modality}-samplenum{samplenum}.csv')
    #     get_results_and_update_label(gpt4v_root, label_path, store_path)

    