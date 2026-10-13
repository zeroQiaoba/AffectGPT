import os
import re
import glob
import tqdm
import shutil
from toolkit.utils.read_files import *
from toolkit.utils.chatgpt import get_image_emotion_batch
from toolkit.utils.functions import func_label_distribution, func_plot_confusion_matrix

id2emo = {'1': 'Surprise',
          '2': 'Fear',
          '3': 'Disgust',
          '4': 'Happiness',
          '5': 'Sadness',
          '6': 'Anger',
          '7': 'Neutral'}
emos = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']

def select_samples_for_RAFDB_basic(data_root, save_root):

    # save path
    save_csv = os.path.join(save_root, 'label.csv')
    save_image = os.path.join(save_root, 'image')
    if not os.path.exists(save_image): os.makedirs(save_image)

    # read data
    names, labels, gpt4vs = [], [], []
    image_root = os.path.join(data_root, 'basic/Image/original/original')
    label_path = os.path.join(data_root, 'basic/EmoLabel/list_patition_label.txt')
    lines = func_read_text_file(label_path)
    for line in lines:
        name, label = line.split()
        label = id2emo[label]
        
        # only save test
        if name.startswith('train_'): continue

        # for image
        input_path = os.path.join(image_root, name)
        save_path  = os.path.join(save_image, name)
        shutil.copy(input_path, save_path)

        # for label
        names.append(name)
        labels.append(label)
        gpt4vs.append('')

    name2key = {}
    for ii, name in enumerate(names):
        name2key[name] = [labels[ii], gpt4vs[ii]]
    keynames = ['gt', 'gpt4v']
    func_write_key_to_csv(save_csv, names, name2key, keynames)


# # 30 images may excceed the max token number of GPT-4V -> reduce to 20
# def evaluate_performance_using_gpt4v(image_root, save_root, save_order, bsize=20, batch_flag=True):
#     if not os.path.exists(save_root):
#         os.makedirs(save_root)
    
#     # shuffle image orders
#     if not os.path.exists(save_order):
#         image_paths = glob.glob(image_root + '/*')
#         indices = np.arange(len(image_paths))
#         random.shuffle(indices)
#         image_paths = np.array(image_paths)[indices]
#         np.savez_compressed(save_order, image_paths=image_paths)
#     else:
#         image_paths = np.load(save_order, allow_pickle=True)['image_paths']
#     print (f'process sample numbers: {len(image_paths)}') # 981

#     # split int batch [20 samples per batch]
#     batches = []
#     splitnum = math.ceil(len(image_paths) / bsize)
#     for ii in range(splitnum):
#         batches.append(image_paths[ii*bsize:(ii+1)*bsize])
#     print (f'process batch  number: {len(batches)}') # 50 batches
#     print (f'process sample number: {sum([len(batch) for batch in batches])}')
    
#     # generate predictions for each batch and store
#     for ii, batch in tqdm.tqdm(enumerate(batches)):
#         save_path = os.path.join(save_root, f'batch_{ii+1}.npz')
#         if os.path.exists(save_path): continue
#         if batch_flag:
#             response = get_image_emotion_batch(batch, emos)
#             np.savez_compressed(save_path, gpt4v=response, names=batch)
#         else: # error when batch process, then process for each sample
#             for jj, item in enumerate(batch):
#                 save_path = os.path.join(save_root, f'batch_{ii+1}_sample_{jj+1}.npz')
#                 if os.path.exists(save_path): continue
#                 response = get_image_emotion_batch([item], emos)
#                 np.savez_compressed(save_path, gpt4v=response, names=[item])


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
    
## 30 images may excceed the max token number of GPT-4V -> reduce to 20
def evaluate_performance_using_gpt4v(image_root, save_root, save_order, bsize=20, batch_flag='flag1', sleeptime=0, grey_flag=False, xishus=[2,2,2]):
    # params assert
    if len(xishus) == 1: assert batch_flag in ['flag1', 'flag2']
    if len(xishus) == 2: assert batch_flag in ['flag1', 'flag2', 'flag3']
    if len(xishus) == 3: assert batch_flag in ['flag1', 'flag2', 'flag3', 'flag4']
    multiple = 1
    for item in xishus: multiple *= item
    assert multiple == bsize

    # define save folder
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    
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
            response = get_image_emotion_batch(batch, emos, sleeptime, grey_flag)
            np.savez_compressed(save_path, gpt4v=response, names=batch)
        elif batch_flag == 'flag2': # split and process # 10
            stores = func_get_segment_batch(batch, save_path, xishu=xishus[0])
            for (segbatch, segsave) in stores:
                if os.path.exists(segsave): continue
                response = get_image_emotion_batch(segbatch, emos, sleeptime, grey_flag)
                np.savez_compressed(segsave, gpt4v=response, names=segbatch)
        elif batch_flag == 'flag3': # split and process # 5
            stores = func_get_segment_batch(batch, save_path, xishu=xishus[0])
            for (segbatch, segsave) in stores:
                if os.path.exists(segsave): continue
                newstores = func_get_segment_batch(segbatch, segsave, xishu=xishus[1])
                for (newbatch, newsave) in newstores:
                    if os.path.exists(newsave): continue
                    response = get_image_emotion_batch(newbatch, emos, sleeptime, grey_flag)
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
                        response = get_image_emotion_batch(new2batch, emos, sleeptime, grey_flag)
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
    print (f'sample number: {len(names)}') # 3068
    name2label = {}
    for ii in range(len(names)):
        name2label[names[ii]] = labels[ii]
    
    ## read prediction
    whole_names, whole_gpt4vs = check_gpt4_performance(gpt4v_root)
    print (f'sample number: {len(whole_names)}') # 3067

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
        ## calculate res
        top1pred = pred[0]
        if top1pred == gt:
            acc += 1
        name2key[name] = [gt, ",".join(pred)]
    print ('gtp4v performance: %.2f' %(acc/len(whole_names)*100))
    keynames = ['gt', 'gpt4v']
    func_write_key_to_csv(store_path, whole_names, name2key, keynames)


import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
def draw_confusion_matrix(label_path, save_path):

    # read inputs
    labels = func_read_key_from_csv(label_path, 'gt')
    preds  = func_read_key_from_csv(label_path, 'gpt4v')
    preds = [pred.split(',')[0] for pred in preds]
    print ('label: ', set(labels))
    print ('pred: ',  set(preds))
    print ('sample number: ', len(preds))

    # convert into idx
    newlabels, newpreds = [], []
    emo2id = {}
    for ii, emo in enumerate(emos): emo2id[emo] = ii
    for ii in range(len(labels)):
        if preds[ii] in emo2id:
            newlabels.append(emo2id[labels[ii]])
            newpreds.append(emo2id[preds[ii]])
    print (f'polish sample number: {len(newlabels)}')
    cm = confusion_matrix(newlabels, newpreds) # gain cm

    # plot confusion matric
    fig, ax = plt.subplots()
    target_names = emos
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=20, fontsize=10)
    plt.yticks(tick_marks, target_names, fontsize=10)
    plt.tight_layout()
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]),
                 horizontalalignment="center",
                 fontsize = 10, 
                 color="white" if cm[i, j] > thresh else "black")
    plt.savefig(save_path, format='png')


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

    emo2idx = {}
    for idx, emo in enumerate(emos): emo2idx[emo] = idx
    labels = [emo2idx[label] for label in labels]
    preds = [emo2idx[pred] for pred in preds]
    target_names = [emo.lower() for emo in emos]
    func_plot_confusion_matrix(labels, preds, target_names, save_path, fontsize, cmap)


def func_find_correct_samples(gpt4v_csv, image_root, store_root):
    names  = func_read_key_from_csv(gpt4v_csv, 'name')
    gts    = func_read_key_from_csv(gpt4v_csv, 'gt')
    gpt4vs = func_read_key_from_csv(gpt4v_csv, 'gpt4v')

    true_root = os.path.join(store_root, 'true')
    false_root = os.path.join(store_root, 'false')
    if not os.path.exists(true_root): os.makedirs(true_root)
    if not os.path.exists(false_root): os.makedirs(false_root)

    for ii in range(len(names)):
        gt = gts[ii]
        pred = gpt4vs[ii].split(',')[0]
        input_path = f'{image_root}/{names[ii]}'
        save_name = "_".join(gpt4vs[ii].split(','))
        if gt == pred:
            save_path = f'{true_root}/{names[ii][:-4]}_gt_{gt}_pred_{save_name}.jpg'
        else:
            save_path = f'{false_root}/{names[ii][:-4]}_gt_{gt}_pred_{save_name}.jpg'
        shutil.copy(input_path, save_path)


def calculate_overlap_times(rgb_path, grey_path, save_path):
    # gain (name2gt, name2preds)
    name2gt = {}
    name2preds = {}
    for gpt4v_csv in [rgb_path, grey_path]:
        print (f'process on {gpt4v_csv}')
        gts    = func_read_key_from_csv(gpt4v_csv, 'gt')
        names  = func_read_key_from_csv(gpt4v_csv, 'name')
        gpt4vs = func_read_key_from_csv(gpt4v_csv, 'gpt4v')

        for ii, name in enumerate(names):
            gt = gts[ii]
            pred = gpt4vs[ii].split(',')[0]
            if name not in name2gt: name2gt[name] = []
            if name not in name2preds: name2preds[name] = []
            name2gt[name].append(gt)
            name2preds[name].append(pred)

    ## check
    satisfy_number = 0
    for name in name2gt:
        assert len(set(name2gt[name])) == 1
        if len(name2preds[name]) == 2:
            satisfy_number += 1
    print (f'whole number: {len(name2preds)}')  
    print (f'satisfy number: {satisfy_number}')

    ## calculate repeat times => 大于81%的重叠率
    same, diff = 0, 0
    for name in name2preds:
        preds = name2preds[name]
        if len(preds) == 2: # 有两个预测结果
            if len(set(preds)) == 1:
                same += 1
            elif len(set(preds)) == 2:
                diff += 1
    print (f'same number: {same}')
    print (f'diff number: {diff}')

    ## 绘制混淆矩阵图
    rgb_preds, grey_preds = [], []
    for name in name2preds:
        preds = name2preds[name]
        if len(preds) == 2: # 必须有两个预测结果
            if preds[0] not in emos or preds[1] not in emos:
                continue
            rgb_preds.append(preds[0])
            grey_preds.append(preds[1])
    print (f'filter number: {len(grey_preds)}')

    emo2idx = {}
    for idx, emo in enumerate(emos): emo2idx[emo] = idx
    rgb_preds  = [emo2idx[pred] for pred in rgb_preds]
    grey_preds = [emo2idx[pred] for pred in grey_preds]
    target_names = [emo.lower() for emo in emos]
    func_plot_confusion_matrix(rgb_preds, grey_preds, target_names, save_path, fontsize=12, cmap=plt.cm.Purples)


if __name__ == '__main__':

    ## step1: pre-process for dataset
    # rafdb_root = 'F:\\RAF-DB\\BUPT-ShanLi'
    # save_root  = 'E:\\Dataset\\gpt4v-evaluation\\rafdb-basic'
    # select_samples_for_RAFDB_basic(rafdb_root, save_root)

    ## step2: gain prediction results [ok Kang's machine]
    save_root = '/root/dataset/rafdb-basic'
    # image_root = os.path.join(save_root, 'image')
    # gpt4v_root = os.path.join(save_root, 'gpt4v')
    # save_order = os.path.join(save_root, 'image_order.npz')
    # evaluate_performance_using_gpt4v(image_root, gpt4v_root, save_order)
    # evaluate_performance_using_gpt4v(image_root, gpt4v_root, save_order, batch_flag=False) # 实在不行了，就逐个样本处理
    
    ## step3: 检查gpt4的生成结果 => 删除错误，其余重新测试 step2，[在step2/step3之间不断迭代，直到所有结果都正确为止]
    # => 只有一个样本有问题，主要原因的人物有点暴露，模型拒绝识别
    # gpt4v_root = os.path.join(save_root, 'gpt4v')
    # check_gpt4_performance(gpt4v_root)

    ## step4: 比较 gpt4v outputs 与 labels.cvs 的结果差异
    # gpt4v_root = os.path.join(save_root, 'gpt4v')
    # label_path = os.path.join(save_root, 'label.csv')
    # store_path = os.path.join(save_root, 'label-gpt4v.csv')
    # get_results_and_update_label(gpt4v_root, label_path, store_path)
    # 3068 => 3067 => gtp4v performance: 75.81
    # refuse to recognize: test_2617.jpg
    
    ## step5: draw confusion matrics
    # label_path = os.path.join(save_root, 'label-gpt4v.csv')
    # cm_path = os.path.join(save_root, 'cm.png')
    # draw_confusion_matrix(label_path, cm_path)

    ## 进一步分析结果并得到更多统计信息
    # gpt4v_path = 'E:\\Dataset\\gpt4v-evaluation\\rafdb-basic\\label-gpt4v.csv'
    # further_report_results(gpt4v_path)
    '''
    sample number: 3067
    gpt4v acc: 75.81
    random guess acc: 14.57
    frequent guess acc: 38.64
    '''

    ## 绘制混淆矩阵图，进一步分析GPT4V的预测结果
    # gpt4v_path = 'E:\\Dataset\\gpt4v-evaluation\\rafdb-basic\\label-gpt4v.csv'
    # save_path  = 'E:\\Dataset\\gpt4v-evaluation\\rafdb-basic\\rafdb-basic-cm.png'
    # plot_confusion_matrix(gpt4v_path, save_path)

    ## 找到拒识样本，并分析拒识case
    # from toolkit.utils.functions import func_find_refused_samples
    # label_path = 'E:\\Dataset\\gpt4v-evaluation\\rafdb-basic\\label.csv'
    # gpt4v_path = 'E:\\Dataset\\gpt4v-evaluation\\rafdb-basic\\label-gpt4v.csv'
    # store_root = 'E:\\Dataset\\gpt4v-evaluation\\rafdb-basic\\refuse_samples'
    # image_root = 'E:\\Dataset\\gpt4v-evaluation\\rafdb-basic\\image'
    # func_find_refused_samples(label_path, gpt4v_path, image_root, store_root)

    ## 找到识别正确和识别错误样本并进行结果展示
    # gpt4v_path = 'E:\\Dataset\\gpt4v-evaluation\\rafdb-basic\\label-gpt4v.csv'
    # store_root = 'E:\\Dataset\\gpt4v-evaluation\\rafdb-basic\\correct_samples'
    # image_root = 'E:\\Dataset\\gpt4v-evaluation\\rafdb-basic\\image'
    # func_find_correct_samples(gpt4v_path, image_root, store_root)


    ## --------------------- 测试灰度图下的结果 ----------------------------
    ## case1: testing
    # bsize, xishus = 20, [2, 2, 5]
    # save_root = '/root/dataset/rafdb-basic'
    # image_root = os.path.join(save_root, 'image')
    # gpt4v_root = os.path.join(save_root, 'gpt4v-grey')
    # save_order = os.path.join(save_root, 'order-grey.npz')
    # for flag in ['flag1', 'flag1', 'flag1', 'flag2', 'flag2', 'flag3', 'flag4']:
    #     evaluate_performance_using_gpt4v(image_root, gpt4v_root, save_order, bsize=bsize, batch_flag=flag, sleeptime=20, grey_flag=True, xishus=xishus)
    #     check_gpt4_performance(gpt4v_root)
    
    ## case2: evaluation
    # label_path = os.path.join(save_root, 'label.csv')
    # store_path = os.path.join(save_root, 'label-gpt4v-grey.csv')
    # get_results_and_update_label(gpt4v_root, label_path, store_path)
    # 3068 -> 3068 -> gpt4v performance: 74.28 [it seems grey and rgb does not has much difference]

    ## 统计 grey and rgb 预测结果的重叠率
    rgb_path  = 'E:\\Dataset\\gpt4v-evaluation\\rafdb-basic\\label-gpt4v.csv'
    grey_path = 'E:\\Dataset\\gpt4v-evaluation\\rafdb-basic\\label-gpt4v-grey.csv'
    save_path = 'E:\\Dataset\\gpt4v-evaluation\\rafdb-basic\\rafdb-rgb-grey-cm.png'
    calculate_overlap_times(rgb_path, grey_path, save_path)
