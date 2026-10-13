import os
import re
import glob
import tqdm
import shutil
from toolkit.utils.read_files import *
from toolkit.utils.functions import func_label_distribution, func_plot_confusion_matrix
from toolkit.utils.chatgpt import get_image_emotion_batch

emos = ['Surprise', 'Contempt', 'Happiness', 'Anger', 'Neutral', 'Sadness', 'Fear', 'Disgust']

# AffectNet: 8 class should have 4000 samples
def select_samples(data_root, save_root):
    
   # save path
    save_csv = os.path.join(save_root, 'label.csv')
    save_image = os.path.join(save_root, 'image')
    if not os.path.exists(save_image): os.makedirs(save_image)

    # read data
    names, labels, gpt4vs = [], [], []
    image_root = os.path.join(data_root, 'image')
    label_path = os.path.join(data_root, 'label_8cls.csv')
    df = pd.read_csv(label_path)
    for _, row in df.iterrows():
        name  = row['name']
        label = row['gt']
    
        # for image
        input_path = os.path.join(image_root, name)
        save_path  = os.path.join(save_image, os.path.basename(name))
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

# each time split into 2 splits
def func_get_segment_batch(batch, savename, xishu=2):
    assert len(batch) % xishu == 0
    segment_num = math.ceil(len(batch)/xishu)

    store = []
    for ii in range(xishu):
        segbatch = batch[ii*segment_num:(ii+1)*segment_num]
        segsave  = savename[:-4] + f"_segment_{ii+1}.npz"
        if not isinstance(segbatch, list):
            segbatch = [segbatch]
        store.append((segbatch, segsave))
    return store

# 30 images may excceed the max token number of GPT-4V -> reduce to 20
def evaluate_performance_using_gpt4v(image_root, save_root, save_order, bsize, batch_flag='flag1', sleeptime=0):
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
    print (f'process sample numbers: {len(image_paths)}')

    # split int batch [20 samples per batch]
    batches = []
    splitnum = math.ceil(len(image_paths) / bsize)
    for ii in range(splitnum):
        batches.append(image_paths[ii*bsize:(ii+1)*bsize])
    print (f'process batch  number: {len(batches)}')
    print (f'process sample number: {sum([len(batch) for batch in batches])}')
    
    # generate predictions for each batch and store
    for ii, batch in tqdm.tqdm(enumerate(batches)):
        save_path = os.path.join(save_root, f'batch_{ii+1}.npz')
        if os.path.exists(save_path): continue
        ## batch not exists -> how to deal with these false batches
        if batch_flag == 'flag1': # process the whole batch again # 24
            response = get_image_emotion_batch(batch, emos, sleeptime)
            np.savez_compressed(save_path, gpt4v=response, names=batch)
        elif batch_flag == 'flag2': # split and process # 12
            stores = func_get_segment_batch(batch, save_path, xishu=2)
            for (segbatch, segsave) in stores:
                if os.path.exists(segsave): continue
                response = get_image_emotion_batch(segbatch, emos, sleeptime)
                np.savez_compressed(segsave, gpt4v=response, names=segbatch)
        elif batch_flag == 'flag3': # split and process # 6
            stores = func_get_segment_batch(batch, save_path, xishu=2)
            for (segbatch, segsave) in stores:
                if os.path.exists(segsave): continue
                newstores = func_get_segment_batch(segbatch, segsave, xishu=2)
                for (newbatch, newsave) in newstores:
                    if os.path.exists(newsave): continue
                    response = get_image_emotion_batch(newbatch, emos, sleeptime)
                    np.savez_compressed(newsave, gpt4v=response, names=newbatch)
        elif batch_flag == 'flag4': # split and process # 1 per sample
            stores = func_get_segment_batch(batch, save_path, xishu=2)
            for (segbatch, segsave) in stores:
                if os.path.exists(segsave): continue
                newstores = func_get_segment_batch(segbatch, segsave, xishu=2)
                for (newbatch, newsave) in newstores:
                    if os.path.exists(newsave): continue
                    new2stores = func_get_segment_batch(newbatch, newsave, xishu=6)
                    for (new2batch, new2save) in new2stores:
                        if os.path.exists(new2save): continue
                        response = get_image_emotion_batch(new2batch, emos, sleeptime)
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
    error_number = 0
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
            error_number += 1
    print (f'error number: {error_number}')
    return whole_names, whole_gpt4vs


def get_results_and_update_label(gpt4v_root, label_path, store_path):
    ## read label_path
    names  = func_read_key_from_csv(label_path, 'name')
    labels = func_read_key_from_csv(label_path, 'gt')
    # preprocess for name
    names = [os.path.basename(name) for name in names]
    print (f'sample number: {len(names)}') # 4000
    name2label = {}
    for ii in range(len(names)):
        name2label[names[ii]] = labels[ii]
    
    ## read prediction
    whole_names, whole_gpt4vs = check_gpt4_performance(gpt4v_root)
    print (f'sample number: {len(whole_names)}') # 4000

    ## gain acc
    acc = 0
    name2key = {}
    for ii, name in enumerate(whole_names):
        gt = name2label[name]
        ## process for pred
        pred = whole_gpt4vs[ii]
        pred = [item for item in re.split('[\'\"]', pred) if item.strip() not in ['', ',']]
        assert len(pred) == 5, 'must return top5 predictions'
        top1pred = pred[0]
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

    ## step1: pre-process for dataset
    # data_root = 'E:\\Dataset\\gpt4v-evaluation\\AffectNet_for_gpt4v_evaluation'
    # save_root = 'E:\\Dataset\\gpt4v-evaluation\\affectnet'
    # select_samples(data_root, save_root)

    ## step2: gain prediction results [ok Kang's machine]
    # => sleeptime: deal with token per minute
    # => batch_flag: deal with request per day3
    # => bsize: should fixed for flag1-flag3
    save_root = '/root/dataset/affectnet'
    image_root = os.path.join(save_root, 'image')
    gpt4v_root = os.path.join(save_root, 'gpt4v')
    save_order = os.path.join(save_root, 'image_order.npz')
    # evaluate_performance_using_gpt4v(image_root, gpt4v_root, save_order, bsize=24, batch_flag='flag1', sleeptime=0)
    # evaluate_performance_using_gpt4v(image_root, gpt4v_root, save_order, bsize=24, batch_flag='flag2', sleeptime=12) # 每次接口调用间隔12s，避免1min token限制
    # evaluate_performance_using_gpt4v(image_root, gpt4v_root, save_order, bsize=24, batch_flag='flag3', sleeptime=12) # 每次接口调用间隔12s，避免1min token限制
    # evaluate_performance_using_gpt4v(image_root, gpt4v_root, save_order, bsize=24, batch_flag='flag4', sleeptime=12) # 每次接口调用间隔12s，避免1min token限制

    ## step3: 检查gpt4的生成结果 => 删除错误，其余重新测试 step2，[在step2/step3之间不断迭代，直到所有结果都正确为止]
    # check_gpt4_performance(gpt4v_root)

    ## step4: 比较 gpt4v outputs 与 labels.cvs 的结果差异
    # label_path = os.path.join(save_root, 'label.csv')
    # store_path = os.path.join(save_root, 'label-gpt4v.csv')
    # get_results_and_update_label(gpt4v_root, label_path, store_path)
    # 4000 => 4000 => gtp4v performance: 42.77

    ## 进一步分析结果并得到更多统计信息
    # gpt4v_path = 'E:\\Dataset\\gpt4v-evaluation\\affectnet\\label-gpt4v.csv'
    # further_report_results(gpt4v_path)
    '''
    sample number: 4000
    gpt4v acc: 42.77
    random guess acc: 12.73
    frequent guess acc: 12.50
    '''

    ## 绘制混淆矩阵图，进一步分析GPT4V的预测结果
    gpt4v_path = 'E:\\Dataset\\gpt4v-evaluation\\affectnet\\label-gpt4v.csv'
    save_path  = 'E:\\Dataset\\gpt4v-evaluation\\affectnet\\affectnet-cm.png'
    plot_confusion_matrix(gpt4v_path, save_path)

 