import os
import re
import glob
import tqdm
import shutil
from sklearn.metrics import confusion_matrix

from toolkit.utils.read_files import *
from toolkit.utils.chatgpt import get_micro_emotion_batch
from toolkit.utils.functions import func_label_distribution, func_plot_confusion_matrix

# emos = ['Fear', 'Surprise', 'Anger', 'Contempt', 'Disgust', 'Sadness', 'Other', 'Happiness']
emos = ['Anger', 'Contempt', 'Happiness', 'Surprise', 'Other']

emo2idx = {}
for ii, emo in enumerate(emos):
    emo2idx[emo] = ii

def func_name_to_jpgpath(name, image_root):
    start, end = name[:-4].split('_')
    image_paths = glob.glob(os.path.join(image_root, start + '*' + end + '.jpg'))
    assert len(image_paths) == 1
    return image_paths[0]

def preprocess_dataset(data_root, save_root):
    # save path
    save_csv = os.path.join(save_root, 'label.csv')
    save_image = os.path.join(save_root, 'image')
    if not os.path.exists(save_image): os.makedirs(save_image)

    # read datat
    names, labels, gpt4vs = [], [], []
    image_root = os.path.join(data_root, 'image')
    label_path = os.path.join(data_root, 'label.csv')
    df = pd.read_csv(label_path)
    for _, row in df.iterrows():
        name = row['jpg_name']
        label = row['emotion']
        if label not in emos:
            continue
        
        # get video_path
        image_path = func_name_to_jpgpath(name, image_root)
        name = os.path.basename(image_path)
        save_path = os.path.join(save_image, name)
        shutil.copy(image_path, save_path)
        
        # for label
        names.append(name)
        labels.append(label)
        gpt4vs.append('')

    print ('candidate labels: ', set(labels))
    print (func_label_distribution(labels))
    name2key = {}
    for ii, name in enumerate(names):
        name2key[name] = [labels[ii], gpt4vs[ii]]
    keynames = ['gt', 'gpt4v']
    func_write_key_to_csv(save_csv, names, name2key, keynames)
    '''
    {'Anger': 57, 'Sadness': 6, 'Surprise': 15, 'Fear': 8, 'Other': 26, 'Happiness': 26, 'Disgust': 9, 'Contempt': 11}
    {'Anger': 57, 'Surprise': 15, 'Other': 26, 'Happiness': 26, 'Contempt': 11}
    '''

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
def evaluate_performance_using_gpt4v(image_root, save_root, save_order, bsize=20, batch_flag='flag1', sleeptime=0, xishus=[2,2,2]):
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
            response = get_micro_emotion_batch(batch, emos, sleeptime)
            np.savez_compressed(save_path, gpt4v=response, names=batch)
        elif batch_flag == 'flag2': # split and process # 10
            stores = func_get_segment_batch(batch, save_path, xishu=xishus[0])
            for (segbatch, segsave) in stores:
                if os.path.exists(segsave): continue
                response = get_micro_emotion_batch(segbatch, emos, sleeptime)
                np.savez_compressed(segsave, gpt4v=response, names=segbatch)
        elif batch_flag == 'flag3': # split and process # 5
            stores = func_get_segment_batch(batch, save_path, xishu=xishus[0])
            for (segbatch, segsave) in stores:
                if os.path.exists(segsave): continue
                newstores = func_get_segment_batch(segbatch, segsave, xishu=xishus[1])
                for (newbatch, newsave) in newstores:
                    if os.path.exists(newsave): continue
                    response = get_micro_emotion_batch(newbatch, emos, sleeptime)
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
                        response = get_micro_emotion_batch(new2batch, emos, sleeptime)
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

    print (f'old number: {len(test_labels)}')
    new_labels, new_preds = [], []
    for ii in range(len(test_labels)):
        if test_preds[ii] in emo2idx:
            new_preds.append(emo2idx[test_preds[ii]])
            new_labels.append(emo2idx[test_labels[ii]])
    print (f'new number: {len(new_labels)}')
    return new_labels, new_preds

# test on three cases
def get_results_and_update_label(gpt4v_root, label_path, store_path):
    
    ## read prediction
    whole_names, whole_gpt4vs = check_gpt4_performance(gpt4v_root)
    whole_names = [name[:-4] for name in whole_names]
    whole_names, whole_gpt4vs = func_remove_dulpicate(whole_names, whole_gpt4vs) # remove multiple predicted
    whole_names, whole_gpt4vs = func_remove_falsepreds(whole_names, whole_gpt4vs) # remove remove false preds
    print (f'pred sample number: {len(set(whole_names))}') # 589
    assert len(whole_names) == len(set(whole_names))
    name2gpt4v = {}
    for ii in range(len(whole_names)):
        name2gpt4v[whole_names[ii]] = whole_gpt4vs[ii]

    ## read label_path
    names = func_read_key_from_csv(label_path, 'name')
    names = [name[:-4] for name in names]
    labels = func_read_key_from_csv(label_path, 'gt')
    print (f'gt sample number: {len(names)}') # 603
    name2gt = {}
    for ii in range(len(names)):
        name2gt[names[ii]] = labels[ii]

    ## gain acc
    acc = 0
    name2key = {}
    test_labels, test_preds = [], []
    for name in name2gpt4v:
        gt = name2gt[name]
        pred = name2gpt4v[name]

        # gain acc
        pred = [item for item in re.split('[\'\"]', pred) if item.strip() not in ['', ',']]
        if len(pred) != 5: # only print for error cases
            print (name2gpt4v[name])
        top1pred = pred[0]
        if top1pred == gt:
            acc += 1
        test_labels.append(gt)
        test_preds.append(top1pred)
        name2key[name] = [gt, ",".join(pred)]
    print ('gtp4v performance: %.2f' %(acc/len(name2key)*100))
    print (func_label_distribution(test_labels))
    keynames = ['gt', 'gpt4v']
    save_names = [name for name in name2key]
    func_write_key_to_csv(store_path, save_names, name2key, keynames)

    # using Sun's Metric
    test_labels, test_preds = func_convert_to_index(test_labels, test_preds)
    conf_mat = confusion_matrix(y_pred=test_preds, y_true=test_labels)
    print(f'Confusion Matrix:\n{conf_mat}')
    class_acc = conf_mat.diagonal() / conf_mat.sum(axis=1)
    print(f"Class Accuracies: {[f'{i:.2%}' for i in class_acc]}")


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
    for repeat in range(10): # remove randomness
        candidate_labels = list(set(gts)) # [positive, negative]
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

    emos = ['Anger', 'Contempt', 'Happiness', 'Surprise', 'Other']
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

    ## step1: rename for all samples
    # data_root = 'G:\\SAMM'
    # save_root = 'E:\\Dataset\\gpt4v-evaluation\\samm'
    # preprocess_dataset(data_root, save_root)

    ## step2: gain prediction results [ok Kang's machine]
    bsize, xishus = 20, [2, 2, 5] # reduce cost
    save_root = '/root/dataset/samm'
    image_root = os.path.join(save_root, 'image')
    gpt4v_root = os.path.join(save_root, 'gpt4v')
    save_order = os.path.join(save_root, 'order.npz')
    # for flag in ['flag1', 'flag1', 'flag1', 'flag2', 'flag2', 'flag3', 'flag4']:
    #     evaluate_performance_using_gpt4v(image_root, gpt4v_root, save_order, bsize=bsize, batch_flag=flag, sleeptime=20, xishus=xishus)
    #     check_gpt4_performance(gpt4v_root)

    ## step4: 比较 gpt4v outputs 与 labels.cvs 的结果差异
    # label_path = os.path.join(save_root, 'label.csv')
    # store_path = os.path.join(save_root, 'label-gpt4v.csv')
    # get_results_and_update_label(gpt4v_root, label_path, store_path)
    
    '''
    gtp4v performance: 17.04
    {'Other': 26, 'Happiness': 26, 'Anger': 57, 'Surprise': 15, 'Contempt': 11}
    old number: 135
    new number: 133
    Confusion Matrix:
    [[ 2  7  0  0 48]
    [ 0  0  0  0 10]
    [ 0  1  1  0 24]
    [ 1  3  0  0 11]
    [ 0  5  0  0 20]]
    Class Accuracies: ['3.51%', '0.00%', '3.85%', '0.00%', '80.00%']

    # => 基本上全部识别为 Other
    emos = ['Anger', 'Contempt', 'Happiness', 'Surprise', 'Other']
    '''

    ## 进一步分析结果并得到更多统计信息
    # gpt4v_path = 'E:\\Dataset\\gpt4v-evaluation\\samm\\label-gpt4v.csv'
    # further_report_results(gpt4v_path)
    '''
    sample number: 135
    gpt4v acc: 17.04
    random guess acc: 20.37
    frequent guess acc: 42.22
    '''

    ## 绘制混淆矩阵图，进一步分析GPT4V的预测结果
    gpt4v_path = 'E:\\Dataset\\gpt4v-evaluation\\samm\\label-gpt4v.csv'
    save_path  = 'E:\\Dataset\\gpt4v-evaluation\\samm\\samm-cm.png'
    plot_confusion_matrix(gpt4v_path, save_path)
