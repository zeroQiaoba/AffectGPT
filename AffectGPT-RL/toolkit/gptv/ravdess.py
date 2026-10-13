import os
import re
import cv2
import glob
import tqdm
from sklearn.metrics import confusion_matrix

from toolkit.utils.read_files import *
from toolkit.utils.chatgpt import get_video_emotion_batch
from toolkit.utils.functions import func_label_distribution, func_plot_confusion_matrix

emos = ['surprised', 'neutral', 'disgust', 'sad', 'happy', 'calm', 'fearful', 'angry']
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
    indexes = [0, int(len(frames)/2), len(frames)-1]
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
    image_root = os.path.join(data_root, 'image')
    label_path = os.path.join(data_root, 'label.csv')
    df = pd.read_csv(label_path)
    for _, row in df.iterrows():
        name  = row['name']
        label = row['gt']
    
        # for video
        video_path = os.path.join(image_root, name)
        save_root  = os.path.join(save_video, name[:-4])
        func_uniform_sample_frames(video_path, save_root)

        # for label
        names.append(name)
        labels.append(label)
        gpt4vs.append('')
    print ('labels: ', set(labels))

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
    names = func_read_key_from_csv(label_path, 'name')
    names = [name[:-4] for name in names]
    labels = func_read_key_from_csv(label_path, 'gt')
    print (f'sample number: {len(names)}')
    name2label = {}
    for ii in range(len(names)):
        name2label[names[ii]] = labels[ii]
    
    ## read prediction
    whole_names, whole_gpt4vs = check_gpt4_performance(gpt4v_root)
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
    print ('gtp4v performance: %.2f' %(acc/len(whole_names)*100))
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
    print (f'sample number: {len(names)}')

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
    print(f'gpt4v => WAR: {war:.2%}, UAR: {uar:.2%}')

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
    print(f'random guess => WAR: {np.mean(whole_wars):.2%}, UAR: {np.mean(whole_uars):.2%}')

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
    print(f'frequent guess => WAR: {war:.2%}, UAR: {uar:.2%}')


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


if __name__ == '__main__':

    ###############################################
    ########## uniform select 2 frames ############
    ###############################################
    ## step1: pre-process for dataset [random sample 2 frames]
    # data_root = '/share/home/lianzheng/emotion-data/RAVDESS'
    # save_root = '/share/home/lianzheng/emotion-data/gpt4v-evaluation/ravdess'
    # select_samples(data_root, save_root)

    ## step2: gain prediction results [ok Kang's machine]
    # bsize, xishus, samplenum = 12, [2, 2, 3], 2 # reduce cost
    # save_root = '/root/dataset/ravdess'
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

    # 1440 -> 1440 -> performance: 22.29
    '''
    sample number: 1440
    gtp4v performance: 22.29
    old number: 1440
    new number: 1440
    Confusion Matrix:
    [[ 28  89   6  22  31   0   1  15]
    [  6  68   1  10   9   0   0   2]
    [  2  69  38  43  19   0   4  17]
    [  5  92  12  60  14   0   1   8]
    [  8  66   2  13  99   1   0   3]
    [  8 115   4  19  38   5   1   2]
    [ 32  81   8  33  18   3   9   8]
    [  5 122  15  16  17   1   2  14]]
    Class Accuracies: ['14.58%', '70.83%', '19.79%', '31.25%', '51.56%', '2.60%', '4.69%', '7.29%']
    UAR: 25.33%, WAR: 22.29%
    '''


    ###############################################
    ########## uniform select 3 frames ############
    ###############################################
    ## step1: pre-process for dataset [random sample 3 frames]
    # data_root = '/share/home/lianzheng/emotion-data/RAVDESS'
    # save_root = '/share/home/lianzheng/emotion-data/gpt4v-evaluation/ravdess-threeframe'
    # select_samples(data_root, save_root)

    ## step2: gain prediction results [ok Kang's machine]
    bsize, xishus = 8, [2, 2, 2] # reduce cost
    save_root = '/root/dataset/ravdess-threeframe'
    image_root = os.path.join(save_root, 'video')
    gpt4v_root = os.path.join(save_root, 'gpt4v')
    save_order = os.path.join(save_root, 'order.npz')
    # for flag in ['flag1', 'flag1', 'flag1', 'flag2', 'flag2', 'flag3', 'flag4']:
    #     evaluate_performance_using_gpt4v(image_root, gpt4v_root, save_order, bsize=bsize, batch_flag=flag, sleeptime=20, samplenum=3, xishus=xishus)
    #     check_gpt4_performance(gpt4v_root)

    ## step3: 比较 gpt4v outputs 与 labels.cvs 的结果差异
    # label_path = os.path.join(save_root, 'label.csv')
    # store_path = os.path.join(save_root, 'label-gpt4v.csv')
    # get_results_and_update_label(gpt4v_root, label_path, store_path)

    '''
    1440 -> 1440 -> Acccuracy: 34.31

    old number: 1440
    new number: 1440
    Confusion Matrix:
    [[ 53  61  11  18  33   0   3  13]
    [  7  74   1   8   2   1   1   2]
    [ 10  18  76  49   6   1   0  32]
    [  6  44  14  97  15   5   2   9]
    [  4  23   4  11 149   0   0   1]
    [  5  80   5   8  82   9   0   3]
    [ 51  36  20  43  13   1   9  19]
    [ 32  63  32  22  15   1   0  27]]
    Class Accuracies: ['27.60%', '77.08%', '39.58%', '50.52%', '77.60%', '4.69%', '4.69%', '14.06%']
    UAR: 36.98%, WAR: 34.31%
    '''

    ## 进一步分析结果并得到更多统计信息
    # gpt4v_path = 'E:\\Dataset\\gpt4v-evaluation\\ravdess-threeframe\\label-gpt4v.csv'
    # further_report_results(gpt4v_path)
    '''
    sample number: 1440
    gpt4v => WAR: 34.31%, UAR: 36.98%
    random guess => WAR: 11.86%, UAR: 11.76%
    frequent guess => WAR: 13.33%, UAR: 12.50%
    '''

    ## 绘制混淆矩阵图，进一步分析GPT4V的预测结果
    gpt4v_path = 'E:\\Dataset\\gpt4v-evaluation\\ravdess-threeframe\\label-gpt4v.csv'
    save_path  = 'E:\\Dataset\\gpt4v-evaluation\\ravdess-threeframe\\ravdess-threeframe-cm.png'
    plot_confusion_matrix(gpt4v_path, save_path)