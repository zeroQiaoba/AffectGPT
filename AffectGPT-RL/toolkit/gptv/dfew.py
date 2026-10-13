import os
import re
import cv2
import glob
import tqdm
import shutil
from sklearn.metrics import confusion_matrix

from toolkit.utils.read_files import *
from toolkit.utils.chatgpt import get_video_emotion_batch
from toolkit.utils.functions import func_label_distribution, func_plot_confusion_matrix

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

emos = ['Sad', 'Neutral', 'Angry', 'Fear', 'Surprise', 'Happy', 'Disgust']
emo2idx = {}
for ii, emo in enumerate(emos):
    emo2idx[emo] = ii

def func_uniform_sample_frames(video_path, save_root):

    # read all frames
    frames = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if ret == False: break
        frames.append(frame)
    cap.release()
    
    ## gain select frames
    # indexes = [0, len(frames)-1] # sample 2 frames
    indexes = [0, int(len(frames)/2), len(frames)-1] # sample 3 frames
    select_frames = [frames[index] for index in indexes]

    ## copy into the tgt root
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    for ii, frame in enumerate(select_frames):
        save_path = os.path.join(save_root, '%06d.jpg' %(ii))
        cv2.imwrite(save_path, frame)

def select_samples(data_root, save_root):
    
   # save path
    save_csv   = os.path.join(save_root, 'label.csv')
    save_video = os.path.join(save_root, 'video')
    if not os.path.exists(save_video): os.makedirs(save_video)

    # read data
    names, labels, gpt4vs = [], [], []
    image_root = os.path.join(data_root, 'all_parts')
    label_path = os.path.join(data_root, 'label_fold1.csv')
    df = pd.read_csv(label_path)
    for _, row in df.iterrows():
        name  = str(int(row['name']))
        label = row['gt']
    
        # for video
        video_path = os.path.join(image_root, name+'.mp4')
        save_root  = os.path.join(save_video, name)
        func_uniform_sample_frames(video_path, save_root)

        # for label
        names.append(name)
        labels.append(label)
        gpt4vs.append('')

    name2key = {}
    for ii, name in enumerate(names):
        name2key[name] = [labels[ii], gpt4vs[ii]]
    keynames = ['gt', 'gpt4v']
    func_write_key_to_csv(save_csv, names, name2key, keynames)


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
def evaluate_performance_using_gpt4v(image_root, save_root, save_order, bsize=20, batch_flag='flag1', sleeptime=0, samplenum=3, xishus=[2,2,2]):
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
            response = get_video_emotion_batch(batch, emos, sleeptime, samplenum)
            np.savez_compressed(save_path, gpt4v=response, names=batch)
        elif batch_flag == 'flag2': # split and process # 10
            stores = func_get_segment_batch(batch, save_path, xishu=xishus[0])
            for (segbatch, segsave) in stores:
                if os.path.exists(segsave): continue
                response = get_video_emotion_batch(segbatch, emos, sleeptime, samplenum)
                np.savez_compressed(segsave, gpt4v=response, names=segbatch)
        elif batch_flag == 'flag3': # split and process # 5
            stores = func_get_segment_batch(batch, save_path, xishu=xishus[0])
            for (segbatch, segsave) in stores:
                if os.path.exists(segsave): continue
                newstores = func_get_segment_batch(segbatch, segsave, xishu=xishus[1])
                for (newbatch, newsave) in newstores:
                    if os.path.exists(newsave): continue
                    response = get_video_emotion_batch(newbatch, emos, sleeptime, samplenum)
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
                        response = get_video_emotion_batch(new2batch, emos, sleeptime, samplenum)
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
            os.system(f'rm -rf {gpt_path}')
            error_num += 1
    print (f'error number: {error_num}')
    return whole_names, whole_gpt4vs


# ------ Performance Evaluation ------
def func_remove_dulpicate(whole_names, whole_gpt4vs):
    updata_names, update_gpt4vs = [], []
    for ii, name in enumerate(whole_names):
        if name not in updata_names:
            updata_names.append(whole_names[ii])
            update_gpt4vs.append(whole_gpt4vs[ii])
    return updata_names, update_gpt4vs

def func_remove_falsepreds(whole_names, whole_gpt4vs):
    updata_names, update_gpt4vs = [], []
    for ii, gpt4vs in enumerate(whole_gpt4vs):
        gpt4vs = gpt4vs.strip()
        if len(gpt4vs) > 1: # remove empty ones
            updata_names.append(whole_names[ii])
            update_gpt4vs.append(whole_gpt4vs[ii])
    return updata_names, update_gpt4vs

def func_convert_to_index(test_labels, test_preds):
    # print (f'old number: {len(test_labels)}')
    new_labels, new_preds = [], []
    for ii in range(len(test_labels)):
        if test_preds[ii] in emo2idx:
            new_preds.append(emo2idx[test_preds[ii]])
            new_labels.append(emo2idx[test_labels[ii]])
    # print (f'new number: {len(new_labels)}')
    return new_labels, new_preds

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
    whole_names = [int(name) for name in whole_names]
    whole_names, whole_gpt4vs = func_remove_dulpicate(whole_names, whole_gpt4vs) # remove multiple predicted
    whole_names, whole_gpt4vs = func_remove_falsepreds(whole_names, whole_gpt4vs) # remove remove false preds
    print (f'sample number: {len(set(whole_names))}')
    assert len(whole_names) == len(set(whole_names))

    ## gain acc
    acc = 0
    name2key = {}
    test_labels, test_preds = [], []
    for ii, name in enumerate(whole_names):
        gt = name2label[name]
        ## process for pred
        pred = whole_gpt4vs[ii]
        pred = [item for item in re.split('[\'\"]', pred) if item.strip() not in ['', ',']]
        if len(pred) != 5: # only print for error cases
            print (whole_gpt4vs[ii])
        top1pred = pred[0]
        if top1pred == gt:
            acc += 1
        name2key[name] = [gt, ",".join(pred)]
        test_labels.append(gt)
        test_preds.append(top1pred)
    print ('Accuracy: %.2f' %(acc/len(whole_names)*100))
    keynames = ['gt', 'gpt4v']
    func_write_key_to_csv(store_path, whole_names, name2key, keynames)

    # using Sun's Metric
    test_labels, test_preds = func_convert_to_index(test_labels, test_preds)
    conf_mat = confusion_matrix(y_pred=test_preds, y_true=test_labels)
    print(f'Confusion Matrix:\n{conf_mat}')
    class_acc = conf_mat.diagonal() / conf_mat.sum(axis=1)
    print(f"Class Accuracies: {[f'{i:.2%}' for i in class_acc]}")
    uar = np.mean(class_acc)
    war = conf_mat.trace() / conf_mat.sum()
    print(f'UAR: {uar:.2%}, WAR: {war:.2%}')


def further_report_results(gpt4v_csv):
    names = func_read_key_from_csv(gpt4v_csv, 'name')
    gts = func_read_key_from_csv(gpt4v_csv, 'gt')
    gpt4vs = func_read_key_from_csv(gpt4v_csv, 'gpt4v')

    # gpt4v
    test_preds = []
    for ii in range(len(names)):
        pred = gpt4vs[ii].split(',')[0]
        test_preds.append(pred)
    test_labels = gts
    test_labels, test_preds = func_convert_to_index(test_labels, test_preds)
    conf_mat = confusion_matrix(y_pred=test_preds, y_true=test_labels)
    class_acc = conf_mat.diagonal() / conf_mat.sum(axis=1)
    uar = np.mean(class_acc)
    war = conf_mat.trace() / conf_mat.sum()
    print(f'gpt4v => WAR: {war:.2%} UAR: {uar:.2%}, ')

    # random guess
    whole_uars = []
    whole_wars = []
    candidate_labels = list(set(gts))
    for repeat in range(10): # remove randomness
        test_preds = []
        for ii in range(len(names)):
            pred = candidate_labels[random.randint(0, len(candidate_labels)-1)]
            test_preds.append(pred)
        test_labels = gts
        test_labels, test_preds = func_convert_to_index(test_labels, test_preds)
        conf_mat = confusion_matrix(y_pred=test_preds, y_true=test_labels)
        class_acc = conf_mat.diagonal() / conf_mat.sum(axis=1)
        uar = np.mean(class_acc)
        war = conf_mat.trace() / conf_mat.sum()
        whole_uars.append(uar)
        whole_wars.append(war)
    print(f'random guess => WAR: {np.mean(whole_wars):.2%} UAR: {np.mean(whole_uars):.2%}, ')

    # frequent guess
    label2count = func_label_distribution(gts)
    labels, counts = [], []
    for label in label2count:
        labels.append(label)
        counts.append(label2count[label])
    majorlabel = labels[np.argmax(counts)]
    test_labels = gts
    test_preds = [majorlabel] * len(names)
    test_labels, test_preds = func_convert_to_index(test_labels, test_preds)
    conf_mat = confusion_matrix(y_pred=test_preds, y_true=test_labels)
    class_acc = conf_mat.diagonal() / conf_mat.sum(axis=1)
    uar = np.mean(class_acc)
    war = conf_mat.trace() / conf_mat.sum()
    print(f'frequent guess => WAR: {war:.2%} UAR: {uar:.2%}, ')


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
            save_path = f'{true_root}/{names[ii]}_gt_{gt}_pred_{save_name}'
        else:
            save_path = f'{false_root}/{names[ii]}_gt_{gt}_pred_{save_name}'
        shutil.copytree(input_path, save_path)


if __name__ == '__main__':

    ###############################################
    ########## uniform select 2 frames ############
    ###############################################
    ## step1: pre-process for dataset [random sample 2 frames]
    # data_root = '/share/home/lianzheng/emotion-data/DFEW'
    # save_root = '/share/home/lianzheng/emotion-data/gpt4v-evaluation/dfew'
    # select_samples(data_root, save_root)

    ## step2: gain prediction results [ok Kang's machine]
    # bsize, xishus, samplenum = 12, [2, 2, 3], 2 # reduce cost
    # save_root = '/root/dataset/dfew'
    # image_root = os.path.join(save_root, 'video')
    # gpt4v_root = os.path.join(save_root, 'gpt4v')
    # save_order = os.path.join(save_root, 'order.npz')
    # for flag in ['flag1', 'flag1', 'flag1', 'flag2', 'flag2', 'flag3', 'flag4']:
    #     evaluate_performance_using_gpt4v(image_root, gpt4v_root, save_order, bsize=bsize, batch_flag=flag, sleeptime=20, samplenum=2, xishus=xishus)
    #     check_gpt4_performance(gpt4v_root)

    ## step3: 比较 gpt4v outputs 与 labels.cvs 的结果差异
    # label_path = os.path.join(save_root, 'label.csv')
    # store_path = os.path.join(save_root, 'label-gpt4v.csv')
    # get_results_and_update_label(gpt4v_root, label_path, store_path)
    # 2341 -> 2337 -> gtp4v performance: 43.69
    '''
    Accuracy: 43.69
    old number: 2337
    new number: 2331
    Confusion Matrix:
    [[194  89  18  40   7  24   5]
    [ 87 321  42  20  24  33   4]
    [ 43 149 161  27  26  25   4]
    [ 25  37  14  65  27   7   3]
    [ 28 117  23  35  69  18   4]
    [ 41 176  16  20  24 209   1]
    [  5  10   3   1   4   4   2]]
    Class Accuracies: ['51.46%', '60.45%', '37.01%', '36.52%', '23.47%', '42.92%', '6.90%']
    UAR: 36.96%, WAR: 43.80%
    '''

    ## 进一步分析结果并得到更多统计信息
    # gpt4v_path = 'E:\\Dataset\\gpt4v-evaluation\\dfew\\label-gpt4v.csv'
    # further_report_results(gpt4v_path)
    '''
    gpt4v => WAR: 43.80% UAR: 36.96%,
    random guess => WAR: 14.43% UAR: 14.49%,
    frequent guess => WAR: 22.81% UAR: 14.29%,
    '''

    ## 绘制混淆矩阵图，进一步分析GPT4V的预测结果
    # gpt4v_path = 'E:\\Dataset\\gpt4v-evaluation\\dfew\\label-gpt4v.csv'
    # save_path  = 'E:\\Dataset\\gpt4v-evaluation\\dfew\\dfew-cm.png'
    # plot_confusion_matrix(gpt4v_path, save_path)

    ## 找到拒识样本，并分析拒识case
    # from toolkit.utils.functions import func_find_refused_samples
    # label_path = 'E:\\Dataset\\gpt4v-evaluation\\dfew\\label.csv'
    # gpt4v_path = 'E:\\Dataset\\gpt4v-evaluation\\dfew\\label-gpt4v.csv'
    # store_root = 'E:\\Dataset\\gpt4v-evaluation\\dfew\\refuse_samples'
    # image_root = 'E:\\Dataset\\gpt4v-evaluation\\dfew\\video'
    # func_find_refused_samples(label_path, gpt4v_path, image_root, store_root)

    ## 找到识别正确和识别错误样本并进行结果展示
    # gpt4v_path = 'E:\\Dataset\\gpt4v-evaluation\\dfew\\label-gpt4v.csv'
    # store_root = 'E:\\Dataset\\gpt4v-evaluation\\dfew\\correct_samples'
    # image_root = 'E:\\Dataset\\gpt4v-evaluation\\dfew\\video'
    # func_find_correct_samples(gpt4v_path, image_root, store_root)


    ###############################################
    ########## uniform select 3 frames ############
    ###############################################
    ## => change to 3-frame for further test
    ## step1: pre-process for dataset [random sample 3 frames]
    # data_root = '/share/home/lianzheng/emotion-data/DFEW'
    # save_root = '/share/home/lianzheng/emotion-data/gpt4v-evaluation/dfew-threeframe'
    # select_samples(data_root, save_root)

    ## step2: gain prediction results [ok Kang's machine]
    # bsize, xishus = 8, [2, 2, 2] # reduce cost
    # save_root = '/root/dataset/dfew-threeframe'
    # image_root = os.path.join(save_root, 'video')
    # gpt4v_root = os.path.join(save_root, 'gpt4v')
    # save_order = os.path.join(save_root, 'order.npz')
    # for flag in ['flag1', 'flag1', 'flag1', 'flag2', 'flag2', 'flag3', 'flag4']:
    #     evaluate_performance_using_gpt4v(image_root, gpt4v_root, save_order, bsize=bsize, batch_flag=flag, sleeptime=20, samplenum=3, xishus=xishus)
    #     check_gpt4_performance(gpt4v_root)

    ## step3: 比较 gpt4v outputs 与 labels.cvs 的结果差异
    # label_path = os.path.join(save_root, 'label.csv')
    # store_path = os.path.join(save_root, 'label-gpt4v.csv')
    # get_results_and_update_label(gpt4v_root, label_path, store_path)

    '''
    2341 -> 2337

    Accuracy: 54.73
    Confusion Matrix:
    [[267  34  19  29  14  15   1]
    [109 300  45  21  26  22  11]
    [ 36  89 220  18  51  15   5]
    [ 19  19  16  92  29   3   2]
    [ 21 100  24  42  94   7   4]
    [ 29 107  12   9  24 303   2]
    [  6  12   2   2   2   2   3]]
    Class Accuracies: ['70.45%', '56.18%', '50.69%', '51.11%', '32.19%', '62.35%', '10.34%']
    UAR: 47.62%, WAR: 54.80%
    '''

    ## 绘制混淆矩阵图，进一步分析GPT4V的预测结果
    # gpt4v_path = 'E:\\Dataset\\gpt4v-evaluation\\dfew-threeframe\\label-gpt4v.csv'
    # save_path  = 'E:\\Dataset\\gpt4v-evaluation\\dfew-threeframe\\confusion-dfew-threeframe.png'
    # plot_confusion_matrix(gpt4v_path, save_path, fontsize=12, cmap=plt.cm.Greens)
    '''
    whole sample number: 2337
    process sample number: 2334
    '''

    ## 找到拒识样本，并分析拒识case
    from toolkit.utils.functions import func_find_refused_samples
    label_path = 'E:\\Dataset\\gpt4v-evaluation\\dfew-threeframe\\label.csv'
    gpt4v_path = 'E:\\Dataset\\gpt4v-evaluation\\dfew-threeframe\\label-gpt4v.csv'
    store_root = 'E:\\Dataset\\gpt4v-evaluation\\dfew-threeframe\\refuse_samples'
    # image_root = 'E:\\Dataset\\gpt4v-evaluation\\dfew-threeframe\\video' # actually, it shoud be this, but for convenience, we use dfew
    image_root = 'E:\\Dataset\\gpt4v-evaluation\\dfew\\video'
    func_find_refused_samples(label_path, gpt4v_path, image_root, store_root)

    ## 找到识别正确和识别错误样本并进行结果展示
    # gpt4v_path = 'E:\\Dataset\\gpt4v-evaluation\\dfew-threeframe\\label-gpt4v.csv'
    # store_root = 'E:\\Dataset\\gpt4v-evaluation\\dfew-threeframe\\correct_samples'
    # image_root = 'E:\\Dataset\\gpt4v-evaluation\\dfew-threeframe\\video'
    # func_find_correct_samples(gpt4v_path, image_root, store_root)