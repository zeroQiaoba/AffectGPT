import os
import re
import glob
import shutil
from toolkit.utils.read_files import *
from toolkit.utils.chatgpt import get_image_emotion_batch
from toolkit.utils.functions import func_label_distribution, func_plot_confusion_matrix

id2emo = {  '0': 'neutral',
            '1': 'anger',
            '2': 'contempt',
            '3': 'disgust',
            '4': 'fear',
            '5': 'happy',
            '6': 'sadness',
            '7': 'surprise'}

emos = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

def select_samples_for_ckplus(data_root, save_root):

    # save path
    save_csv = os.path.join(save_root, 'label.csv')
    save_video = os.path.join(save_root, 'image')
    if not os.path.exists(save_video): os.makedirs(save_video)

    # read data
    names, labels, gpt4vs = [], [], []
    video_root = os.path.join(data_root, 'extended-cohn-kanade-images/cohn-kanade-images')
    label_root = os.path.join(data_root, 'Emotion_labels/Emotion')
    for speaker in os.listdir(label_root):
        for clip in os.listdir(label_root + '/' + speaker):
            label_paths = glob.glob(os.path.join(label_root, speaker, clip, '*.txt'))
            if len(label_paths) != 1: continue
            label_path = label_paths[0]
            label = id2emo[func_read_text_file(label_path)[0][0]]
            assert label != 'neutral'

            # gain last three frames
            frame_paths = glob.glob(os.path.join(video_root, speaker, clip, '*.png'))
            for frame_path in sorted(frame_paths)[-3:]:
                framename = os.path.basename(frame_path)
                save_path = os.path.join(save_video, framename)
                shutil.copy(frame_path, save_path)
                # for label
                names.append(framename)
                labels.append(label)
                gpt4vs.append('')

    # debug
    print (f'label numbers: {len(set(labels))}')
    name2key = {}
    for ii, name in enumerate(names):
        name2key[name] = [labels[ii], gpt4vs[ii]]
    keynames = ['gt', 'gpt4v']
    func_write_key_to_csv(save_csv, names, name2key, keynames)

# 30 images may excceed the max token number of GPT-4V -> reduce to 20
def evaluate_performance_using_gpt4v_ckplus(image_root, save_root, save_order, bsize=20):
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
        image_paths = np.load(save_order, allow_pickle=True)['image_paths']
    print (f'process sample numbers: {len(image_paths)}') # 981

    # split int batch [20 samples per batch]
    batches = []
    splitnum = math.ceil(len(image_paths) / bsize)
    for ii in range(splitnum):
        batches.append(image_paths[ii*bsize:(ii+1)*bsize])
    print (f'process batch  number: {len(batches)}') # 50 batches
    print (f'process sample number: {sum([len(batch) for batch in batches])}')
    
    # generate predictions for each batch and store
    for ii, batch in enumerate(batches):
        save_path = os.path.join(save_root, f'batch_{ii+1}.npz')
        if os.path.exists(save_path): continue
        response = get_image_emotion_batch(batch, emos)
        np.savez_compressed(save_path, 
                            gpt4v=response,
                            names=batch)


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
    print (f'sample number: {len(whole_names)}')

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

    ## step1: process for ck+
    # data_root = 'F:\\CK+'
    # save_root = 'E:\\Dataset\\gpt4v-evaluation\\ckplus'
    # select_samples_for_ckplus(data_root, save_root)

    ## step2: gain prediction results [ok Kang's machine]
    save_root = '/root/dataset/ckplus'
    # image_root = os.path.join(save_root, 'image')
    # gpt4v_root = os.path.join(save_root, 'gpt4v')
    # save_order = os.path.join(save_root, 'image_order.npz')
    # evaluate_performance_using_gpt4v_ckplus(image_root, gpt4v_root, save_order)

    ## step3: 检查gpt4的生成结果 => 删除错误，其余重新测试 step2
    # gpt4v_root = os.path.join(save_root, 'gpt4v')
    # check_gpt4_performance(gpt4v_root)

    ## step4: 比较 gpt4v outputs 与 labels.cvs 的结果差异
    # gpt4v_root = os.path.join(save_root, 'gpt4v')
    # label_path = os.path.join(save_root, 'label.csv')
    # store_path = os.path.join(save_root, 'label-gpt4v.csv')
    # get_results_and_update_label(gpt4v_root, label_path, store_path)
    # 981 => 981 => gtp4v performance: 69.72

    ## 进一步分析结果并得到更多统计信息
    # gpt4v_path = 'E:\\Dataset\\gpt4v-evaluation\\ckplus\\label-gpt4v.csv'
    # further_report_results(gpt4v_path)
    '''
    sample number: 981
    gpt4v acc: 69.72
    random guess acc: 14.61
    frequent guess acc: 25.38
    '''

    ## 绘制混淆矩阵图，进一步分析GPT4V的预测结果
    # gpt4v_path = 'E:\\Dataset\\gpt4v-evaluation\\ckplus\\label-gpt4v.csv'
    # save_path  = 'E:\\Dataset\\gpt4v-evaluation\\ckplus\\ckplus-cm.png'
    # plot_confusion_matrix(gpt4v_path, save_path)
    
    ## 找到拒识样本，并分析拒识case
    from toolkit.utils.functions import func_find_refused_samples
    label_path = 'E:\\Dataset\\gpt4v-evaluation\\ckplus\\label.csv'
    gpt4v_path = 'E:\\Dataset\\gpt4v-evaluation\\ckplus\\label-gpt4v.csv'
    store_root = 'E:\\Dataset\\gpt4v-evaluation\\ckplus\\refuse_samples'
    image_root = 'E:\\Dataset\\gpt4v-evaluation\\ckplus\\image'
    func_find_refused_samples(label_path, gpt4v_path, image_root, store_root)