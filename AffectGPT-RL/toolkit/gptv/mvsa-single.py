import os
import re
import glob
import tqdm
import shutil
from sklearn.metrics import f1_score

from toolkit.utils.read_files import *
from toolkit.utils.chatgpt import get_social_multi_batch, get_social_image_batch, get_social_text_batch
from toolkit.utils.functions import func_label_distribution, func_plot_confusion_matrix

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

emos = ['negative', 'neutral', 'positive']
emo2idx = {}
for idx, emo in enumerate(emos): emo2idx[emo] = idx

def func_merge_label(text_major, image_major):
    labels = list(set([text_major, image_major]))
    if len(labels) == 1:
        return labels[0]
    else: # 可能有三种组合
        if 'positive' in labels and 'negative' in labels:
            return ""
        elif 'positive' in labels:
            return 'positive'
        elif 'negative' in labels:
            return 'negative'

def func_random_select_subset(sample_num, partial=0.1):
    indices = np.arange(sample_num)
    random.shuffle(indices)
    indices = indices[:int(len(indices)*partial)]
    return indices

def select_samples(data_root, save_root, partial=None):
    
    # read (names, labels, gpt4vs)
    names, labels, gpt4vs = [], [], []
    label_path = os.path.join(data_root, 'labelResultAll.txt')
    lines = func_read_text_file(label_path)
    for line in lines[1:]:
        name, ann = line.split()
        ## 读取每个模态的 major label
        text_label, image_label = ann.split(',')
        merge_label = func_merge_label(text_label, image_label)
        if merge_label == '':
            continue
        ## 保存结果
        names.append(name)
        labels.append(merge_label)
        gpt4vs.append('')
    label2count = func_label_distribution(labels)
    print (len(labels))
    print (label2count)
    ''' check 完全一致
    4511 
    {'positive': 2683, 'negative': 1358, 'neutral': 470}
    '''

    # update (names, labels, gpt4vs) -> randomly select 10% for testing
    if partial is not None:
        indices = func_random_select_subset(len(names), partial)
        names  = np.array(names)[indices]
        labels = np.array(labels)[indices]
        gpt4vs = np.array(gpt4vs)[indices]
    label2count = func_label_distribution(labels)
    print (len(labels))
    print (label2count)
    '''
    451
    '''

    # 存储结果
    for modality in ['image', 'text']:
        # origin data path
        ori_root = os.path.join(data_root, 'data')

        # save path
        save_csv = os.path.join(save_root, 'label.csv')
        save_data = os.path.join(save_root, modality)
        if not os.path.exists(save_data): os.makedirs(save_data)

        for ii, name in enumerate(names):
            # get video_path
            image_path = os.path.join(ori_root, name + '.jpg')
            text_path  = os.path.join(ori_root, name + '.txt')
            # get save_path
            if modality == 'image':
                save_path = os.path.join(save_data, name + '.jpg')
                shutil.copy(image_path, save_path)
            elif modality == 'text':
                save_path = os.path.join(save_data, name + '.npy')
                lines = func_read_text_file(text_path) # 默认字幕文件为1行
                assert len(lines) == 1
                np.save(save_path, lines[0])

        name2key = {}
        for ii, name in enumerate(names):
            name2key[name] = [labels[ii], gpt4vs[ii]]
        keynames = ['gt', 'gpt4v']
        func_write_key_to_csv(save_csv, names, name2key, keynames)


# 分析 social media 下的情感识别结果
def func_get_response(batch, emos, modality, sleeptime):
    if modality == 'image':
        response = get_social_image_batch(batch, emos, sleeptime)
    elif modality == 'text':
        response = get_social_text_batch(batch, emos, sleeptime)
    if modality == 'multi':
        response = get_social_multi_batch(batch, emos, sleeptime)
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
def evaluate_performance_using_gpt4v(image_root, save_root, save_order, modality, bsize=20, batch_flag='flag1', sleeptime=0, xishus=[2,2,2]):
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
        image_root = os.path.split(image_root)[0] + '/image'

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
            response = func_get_response(batch, emos, modality, sleeptime)
            np.savez_compressed(save_path, gpt4v=response, names=batch)
        elif batch_flag == 'flag2': # split and process # 10
            stores = func_get_segment_batch(batch, save_path, xishu=xishus[0])
            for (segbatch, segsave) in stores:
                if os.path.exists(segsave): continue
                response = func_get_response(segbatch, emos, modality, sleeptime)
                np.savez_compressed(segsave, gpt4v=response, names=segbatch)
        elif batch_flag == 'flag3': # split and process # 5
            stores = func_get_segment_batch(batch, save_path, xishu=xishus[0])
            for (segbatch, segsave) in stores:
                if os.path.exists(segsave): continue
                newstores = func_get_segment_batch(segbatch, segsave, xishu=xishus[1])
                for (newbatch, newsave) in newstores:
                    if os.path.exists(newsave): continue
                    response = func_get_response(newbatch, emos, modality, sleeptime)
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
                        response = func_get_response(new2batch, emos, modality, sleeptime)
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
        print (f'number of samples: {len(names)} number of results: {len(gpt4vs)}')
        if len(names) == len(gpt4vs): 
            names = [os.path.basename(name) for name in names]
            whole_names.extend(names)
            whole_gpt4vs.extend(gpt4vs)
        else:
            print (f'error batch: {gpt_path}. Need re-test!!')
            os.remove(gpt_path)
            error_num += 1
    # print (f'error number: {error_num}')
    return whole_names, whole_gpt4vs


def get_results_and_update_label(gpt4v_root, label_path, store_path):
    ## read label_path
    names  = func_read_key_from_csv(label_path, 'name')
    labels = func_read_key_from_csv(label_path, 'gt')
    print (f'sample number: {len(names)}')
    name2label = {}
    for ii in range(len(names)):
        name2label[names[ii]] = labels[ii]
    
    ## read prediction
    whole_names, whole_gpt4vs = check_gpt4_performance(gpt4v_root)
    whole_names = [int(name.rsplit('.', 1)[0]) for name in whole_names]
    print (f'sample number: {len(whole_names)}')

    ## gain acc
    acc = 0
    name2key = {}
    for ii, name in enumerate(whole_names):
        gt = name2label[name]
        ## process for pred
        pred = whole_gpt4vs[ii]
        pred = [item for item in re.split('[\'\"]', pred) if item.strip() not in ['', ',']]
        if len(pred) != 3:
            print (whole_gpt4vs[ii])
        ## calculate res
        top1pred = pred[0]
        if not (top1pred in emos and gt in emos):
            print (top1pred, gt)
        if top1pred == gt:
            acc += 1
        name2key[name] = [gt, ",".join(pred)]
    print ('gtp4v performance: %.2f' %(acc/len(whole_names)*100))
    keynames = ['gt', 'gpt4v']
    func_write_key_to_csv(store_path, whole_names, name2key, keynames)


def further_report_results(gpt4v_csv):
    names = func_read_key_from_csv(gpt4v_csv, 'name')
    gts = func_read_key_from_csv(gpt4v_csv, 'gt')
    gpt4vs = func_read_key_from_csv(gpt4v_csv, 'gpt4v')

    acc = 0
    for ii in range(len(names)):
        gt = gts[ii]
        pred = gpt4vs[ii].split(',')[0]
        if pred == gt:
            acc += 1
    print (f'sample number: {len(names)}')
    print ('gpt4v acc: %.2f' %(acc/len(names)*100))
    
    # random guess
    whole_accs = []
    candidate_labels = list(set(gts))
    for repeat in range(10): # remove randomness
        acc = 0
        for ii in range(len(names)):
            pred = candidate_labels[random.randint(0, len(candidate_labels)-1)]
            if pred == gts[ii]:
                acc += 1
        whole_accs.append(acc/len(names)*100)
    print ('random guess acc: %.2f' %(np.mean(whole_accs)))

    # frequent guess
    label2count = func_label_distribution(gts)
    labels, counts = [], []
    for label in label2count:
        labels.append(label)
        counts.append(label2count[label])
    majorlabel = labels[np.argmax(counts)]
    acc = 0
    for ii in range(len(names)):
        pred = majorlabel
        if pred == gts[ii]:
            acc += 1
    print ('frequent guess acc: %.2f' %(acc/len(names)*100))


# 多个gpt-4v的samplenum结果，只统计overlap sample的wf1得分，看看能不能减小方差
def overlap_report_results(gpt4v_paths):

    # gain overlap names
    namecounts = {}
    for gpt4v_csv in gpt4v_paths:
        print (gpt4v_csv)
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
        test_preds = np.array(test_preds)
        test_labels = np.array(gts)
        non_zeros = np.array([i for i, e in enumerate(test_labels) if e != 0]) # remove 0, and remove mask
        fscore = f1_score((test_labels[non_zeros] > 0), (test_preds[non_zeros] > 0), average='weighted')
        print ('gtp4v waf: %.2f' %(fscore*100))

        # store into namecounts
        for name in names:
            if name not in namecounts: namecounts[name] = 0
            namecounts[name] += 1
    
    # find has all predicted samples
    process_names = []
    for name in namecounts:
        if namecounts[name] == len(gpt4v_paths):
            process_names.append(name)
    print (f'process names: {len(process_names)}')
    for gpt4v_csv in gpt4v_paths:
        print (gpt4v_csv)
        names = func_read_key_from_csv(gpt4v_csv, 'name')
        gts = func_read_key_from_csv(gpt4v_csv, 'gt')
        gpt4vs = func_read_key_from_csv(gpt4v_csv, 'gpt4v')

        # gpt4v
        test_labels, test_preds = [], []
        for ii in range(len(names)):
            gt = gts[ii]
            name = names[ii]
            top1pred = gpt4vs[ii].split(',')[0]
            if name not in process_names: continue
            # post-process predictions
            if top1pred.find('negative') != -1:
                test_preds.append(-1)
            elif top1pred.find('positive') != -1:
                test_preds.append(1)
            elif top1pred.find('neutral') != -1:
                test_preds.append(0)
            test_labels.append(gt)
        test_preds = np.array(test_preds)
        test_labels = np.array(test_labels)
        print (f'sample number: {len(test_preds)}')
        non_zeros = np.array([i for i, e in enumerate(test_labels) if e != 0]) # remove 0, and remove mask
        fscore = f1_score((test_labels[non_zeros] > 0), (test_preds[non_zeros] > 0), average='weighted')
        print ('gtp4v waf: %.2f' %(fscore*100))


def plot_confusion_matrix(gpt4v_csv, save_path, fontsize, cmap):
    names = func_read_key_from_csv(gpt4v_csv, 'name')
    gts = func_read_key_from_csv(gpt4v_csv, 'gt')
    gpt4vs = func_read_key_from_csv(gpt4v_csv, 'gpt4v')

    labels, preds = [], []
    for ii in range(len(names)):
        gt = gts[ii]
        pred = gpt4vs[ii].split(',')[0]
        if pred not in emos: continue
        labels.append(gt)
        preds.append(pred)
    print (f'whole sample number: {len(names)}')
    print (f'process sample number: {len(labels)}')

    labels = [emo2idx[label] for label in labels]
    preds = [emo2idx[pred] for pred in preds]
    target_names = [emo.lower() for emo in emos]
    func_plot_confusion_matrix(labels, preds, target_names, save_path, fontsize, cmap)


if __name__ == '__main__':

    ## step1: pre-process for dataset [for multimodal dataset, firstly process on each modality]
    # data_root = 'F:\\MVSA-Single'
    # save_root = 'E:\\Dataset\\gpt4v-evaluation\\mvsa-single' ## 测试全部数据
    # select_samples(data_root, save_root, partial=None)
    # save_root = 'E:\\Dataset\\gpt4v-evaluation\\mvsa-single-subset' ## 仅测试test set
    # select_samples(data_root, save_root, partial=0.1) 

    ## step2: gain prediction results
    save_root = 'E:\\Dataset\\gpt4v-evaluation\\mvsa-single-subset'
    for (modality, bsize, xishus) in [
                                    #  ('multi', 8,  [2,2,2]),
                                       ('image', 20, [2,2,5]),
                                       ('text',  20, [2,2,5]),
                                      ]:
        image_root = os.path.join(save_root, modality)
        gpt4v_root = os.path.join(save_root, f'{modality}-gpt4v')
        save_order = os.path.join(save_root, f'{modality}-order.npz')
        # for flag in ['flag1', 'flag1', 'flag1', 'flag2', 'flag2', 'flag3', 'flag4']:
        #     evaluate_performance_using_gpt4v(image_root, gpt4v_root, save_order, modality, bsize=bsize, batch_flag=flag, sleeptime=20, xishus=xishus)
        #     check_gpt4_performance(gpt4v_root)

        ## 比较 gpt4v outputs 与 labels.cvs 的结果差异
        label_path = os.path.join(save_root, 'label.csv')
        store_path = os.path.join(save_root, f'label-gpt4v-{modality}.csv')
        get_results_and_update_label(gpt4v_root, label_path, store_path)

    '''
    video: 451 -> 409 -> gpt4v performance: 58.68
    text:  451 -> 451 -> gpt4v performance: 57.65
    multi: 451 -> 449 -> gpt4v performance: 61.25
    '''
    
    ## 进一步分析结果并得到更多统计信息
    # gpt4v_path = 'E:\\Dataset\\gpt4v-evaluation\\mvsa-single-subset\\label-gpt4v-multi.csv'
    # further_report_results(gpt4v_path)
    '''
    random guess acc: 33.96
    frequent guess acc: 55.68
    '''

    ## 绘制混淆矩阵图，进一步分析GPT4V的预测结果
    # gpt4v_path = 'E:\\Dataset\\gpt4v-evaluation\\mvsa-single-subset\\label-gpt4v-multi.csv'
    # save_path  = 'E:\\Dataset\\gpt4v-evaluation\\mvsa-single-subset\\confusion-mvsa-single-multi.png'
    # plot_confusion_matrix(gpt4v_path, save_path, fontsize=16, cmap=plt.cm.YlOrBr)
    
    