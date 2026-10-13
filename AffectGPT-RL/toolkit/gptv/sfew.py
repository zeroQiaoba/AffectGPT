import os
import re
import glob
import tqdm
import shutil
from toolkit.utils.read_files import *
from toolkit.utils.chatgpt import *
from toolkit.utils.functions import func_label_distribution, func_plot_confusion_matrix

emos = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

#########################
## Dataset preprocess
#########################
# should have 436 test images 
# => one overlap samples, thus 435 samples
def select_samples(data_root, save_root):
    
    # save path
    save_csv = os.path.join(save_root, 'label.csv')
    save_video = os.path.join(save_root, 'image')
    if not os.path.exists(save_video): os.makedirs(save_video)

    # read data
    names, labels, gpt4vs = [], [], []
    for emo in os.listdir(data_root):
        for frame_path in glob.glob(os.path.join(data_root, emo, '*')):
            framename = os.path.basename(frame_path)
            save_path = os.path.join(save_video, framename)
            shutil.copy(frame_path, save_path)

            # for label
            names.append(framename)
            labels.append(emo)
            gpt4vs.append('')

    name2key = {}
    for ii, name in enumerate(names):
        name2key[name] = [labels[ii], gpt4vs[ii]]
    keynames = ['gt', 'gpt4v']
    func_write_key_to_csv(save_csv, names, name2key, keynames)


def find_overlap_names(label_path):
    names  = func_read_key_from_csv(label_path, 'name')
    labels = func_read_key_from_csv(label_path, 'gt')
    name2count = {}
    for name in names:
        if name not in name2count:
            name2count[name] = 0
        else:
            print (f'overlap name: {name}')


#########################
## GPT4v predictions
#########################
def func_get_response(batch, emos, modality, sleeptime, template):
    if modality == 'video':
        response = get_video_emotion_batch(batch, emos, sleeptime)
    elif modality == 'text':
        response = get_text_emotion_batch(batch, emos, sleeptime)
    elif modality == 'multi':
        response = get_multi_emotion_batch(batch, emos, sleeptime)
    elif modality == 'image':
        response = get_image_emotion_batch(batch, emos, sleeptime, template)
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

## 30 images may excceed the max token number of GPT-4V -> reduce to 20
def evaluate_performance_using_gpt4v(image_root, save_root, save_order, modality, template='case0', bsize=20, batch_flag='flag1', sleeptime=0, xishus=[2,2,5]):
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
        if batch_flag == 'flag1': # process the whole batch again # 20
            response = func_get_response(batch, emos, modality, sleeptime, template)
            np.savez_compressed(save_path, gpt4v=response, names=batch)
        elif batch_flag == 'flag2': # split and process # 10
            stores = func_get_segment_batch(batch, save_path, xishu=xishus[0])
            for (segbatch, segsave) in stores:
                if os.path.exists(segsave): continue
                response = func_get_response(segbatch, emos, modality, sleeptime, template)
                np.savez_compressed(segsave, gpt4v=response, names=segbatch)
        elif batch_flag == 'flag3': # split and process # 5
            stores = func_get_segment_batch(batch, save_path, xishu=xishus[0])
            for (segbatch, segsave) in stores:
                if os.path.exists(segsave): continue
                newstores = func_get_segment_batch(segbatch, segsave, xishu=xishus[1])
                for (newbatch, newsave) in newstores:
                    if os.path.exists(newsave): continue
                    response = func_get_response(newbatch, emos, modality, sleeptime, template)
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
                        response = func_get_response(new2batch, emos, modality, sleeptime, template)
                        np.savez_compressed(new2save, gpt4v=response, names=new2batch)


#################################################################
## GPT4v output polish: remove false predictions and try again
#################################################################
# for top5
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

# for most likely category
def func_analyze_gpt4v_outputs_v2(gpt_path):
    
    names = np.load(gpt_path, allow_pickle=True)['names'].tolist()

    ## analyze gpt-4v
    store_results = []
    gpt4v = np.load(gpt_path, allow_pickle=True)['gpt4v'].tolist()
    gpt4v = gpt4v.replace("name",    "==========")
    gpt4v = gpt4v.replace("result",  "==========")
    gpt4v = gpt4v.split("==========")
    for item in gpt4v:
        for emo in emos:
            if emo in item:
                store_results.append(emo)
  
    return names, store_results

## 临时采用：针对 template='case2' 的情况进行分析
def check_gpt4_performance(gpt4v_root, template='case0'):
    whole_names, whole_gpt4vs = [], []
    for gpt_path in sorted(glob.glob(gpt4v_root + '/*')):
        if template in ['case0', 'case1']:
            names, gpt4vs = func_analyze_gpt4v_outputs(gpt_path)
        else:
            names, gpt4vs = func_analyze_gpt4v_outputs_v2(gpt_path)
        print (f'number of samples: {len(names)} number of results: {len(gpt4vs)}')
        if len(names) == len(gpt4vs): 
            names = [os.path.basename(name) for name in names]
            whole_names.extend(names)
            whole_gpt4vs.extend(gpt4vs)
        else:
            print (f'error batch: {gpt_path}. Need re-test!!')
            os.remove(gpt_path)
    return whole_names, whole_gpt4vs


#########################
## Calculate results
#########################
def get_results_and_update_label(gpt4v_root, label_path, store_path, template):
    print (gpt4v_root)

    ## read label_path
    names  = func_read_key_from_csv(label_path, 'name')
    labels = func_read_key_from_csv(label_path, 'gt')
    print (f'sample number: {len(names)}') # 435
    name2label = {}
    for ii in range(len(names)):
        name2label[names[ii]] = labels[ii]
    
    ## read prediction
    whole_names, whole_gpt4vs = check_gpt4_performance(gpt4v_root, template)
    print (f'sample number: {len(whole_names)}')

    ## gain acc
    acc = 0
    name2key = {}
    for ii, name in enumerate(whole_names):
        gt = name2label[name]
        ## process for pred
        pred = whole_gpt4vs[ii]
        if template in ['case0', 'case1']:
            pred = [item for item in re.split('[\'\"]', pred) if item.strip() not in ['', ',']]
            assert len(pred) == 5, 'must return top5 predictions'
            top1pred = pred[0]
        else:
            top1pred = pred
        if top1pred == gt:
            acc += 1
        name2key[name] = [gt, ",".join(pred)]
    print ('gtp4v performance: %.2f' %(acc/len(whole_names)*100))
    keynames = ['gt', 'gpt4v']
    func_write_key_to_csv(store_path, whole_names, name2key, keynames)


#########################
## Further analysis
#########################
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


# test GPT-4V's prediction stability
def analyze_prediciton_stability(random_csvs):

    # gain (name2gt, name2preds)
    name2gt = {}
    name2preds = {}
    for gpt4v_csv in random_csvs:
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
        if len(name2preds[name]) == len(random_csvs):
            satisfy_number += 1
    print (f'whole number: {len(name2preds)}')  # 435
    print (f'satisfy number: {satisfy_number}') # 434

    ## calculate repeat times
    maxcounts = []
    for name in name2preds:
        preds = name2preds[name]
        if len(preds) == len(random_csvs): # 10次运行均在这个数据集上产生了预测结果
            pred2count = func_label_distribution(preds)
            maxcount = max([pred2count[pred] for pred in pred2count])
            maxcounts.append(maxcount)
    print (f'processed number: {len(maxcounts)}')
    
    ## return frequence
    counts = func_label_distribution(maxcounts)
    for key in sorted(counts): print (key, counts[key])


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
    # data_root = 'F:\\SFEW_2.0\\Val\\Val'
    # save_root = 'E:\\Dataset\\gpt4v-evaluation\\sfew2'
    # select_samples(data_root, save_root)
    # label_path = os.path.join(save_root, 'label.csv')
    # find_overlap_names(label_path)

    ## step2: gain prediction results [ok Kang's machine]
    save_root = '/root/dataset/sfew2'
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
    # 435 => 435 => gtp4v performance: 57.24

    ## 进一步分析结果并得到更多统计信息
    # gpt4v_path = 'E:\\Dataset\\gpt4v-evaluation\\sfew2\\label-gpt4v.csv'
    # further_report_results(gpt4v_path)
    '''
    sample number: 435
    gpt4v acc: 57.24
    random guess acc: 13.98
    frequent guess acc: 19.77
    '''

    ## 绘制混淆矩阵图，进一步分析GPT4V的预测结果
    # gpt4v_path = 'E:\\Dataset\\gpt4v-evaluation\\sfew2\\label-gpt4v.csv'
    # save_path  = 'E:\\Dataset\\gpt4v-evaluation\\sfew2\\sfew2-cm.png'
    # plot_confusion_matrix(gpt4v_path, save_path)


    #################################################################
    # => 修改 evaluate_performance_using_gpt4v 为分级式预测
    #################################################################
    ## 进行随机性测试 => run 10 times and store all results
    # for time in range(10):
    #     modality, bsize, xishus = 'image', 20, [2, 2, 5]
    #     flags = ['flag1', 'flag1', 'flag1', 'flag2', 'flag2', 'flag3', 'flag4']
    #     image_root = os.path.join(save_root, modality)
    #     gpt4v_root = os.path.join(save_root, f'gpt4v-random{time+1}')
    #     save_order = os.path.join(save_root, f'order-random{time+1}.npz')
    #     print (f'process gpt4v root: {gpt4v_root}')
    #     for flag in flags:
    #         evaluate_performance_using_gpt4v(image_root, gpt4v_root, save_order, modality, bsize=bsize, batch_flag=flag, sleeptime=20, xishus=xishus)
    #         check_gpt4_performance(gpt4v_root)

    ## 计算结果
    # for time in range(10):  
    #     label_path = os.path.join(save_root, 'label.csv')
    #     store_path = os.path.join(save_root, f'label-gpt4v-random{time+1}.csv')
    #     gpt4v_root = os.path.join(save_root, f'gpt4v-random{time+1}')
    #     get_results_and_update_label(gpt4v_root, label_path, store_path)
    '''
    # 但是第一次运行的结果，和我论文中的是一模一样的，后面再运行产生了差异
    gtp4v performance: 57.24
    gtp4v performance: 55.17
    gtp4v performance: 55.86
    gtp4v performance: 55.63
    gtp4v performance: 55.86
    gtp4v performance: 54.71
    gtp4v performance: 55.07
    gtp4v performance: 52.64
    gtp4v performance: 55.40
    gtp4v performance: 56.32
    '''
    
    ## 分析10次结果中每个样本的结果重叠情况 => 只考虑所有predition都有的sample
    # random_csvs = glob.glob(save_root + '/label-gpt4v-random*.csv')
    # analyze_prediciton_stability(random_csvs)
    '''
    4: 5
    5: 21
    6: 35
    7: 38
    8: 63
    9: 49
    10: 223
    '''


    ######################################################################
    ## 测试不同 prompt template 的影响 => ChatGPT已经对于template比较robust
    ######################################################################
    # template = 'case0' # 原始的prompt
    # template = 'case1' # 删除 Please play the role of a facial expression classification expert. 分析其影响
    # template = 'case2' # 只选择最有可能的 category
    for template in ['case1', 'case2']:
        save_root = 'E:\\Dataset\\gpt4v-evaluation\\sfew2'
        modality, bsize, xishus = 'image', 20, [2, 2, 5]
        flags = ['flag1', 'flag1', 'flag1', 'flag2', 'flag2', 'flag3', 'flag4']
        image_root = os.path.join(save_root, modality)
        gpt4v_root = os.path.join(save_root, f'gpt4v-template-{template}')
        save_order = os.path.join(save_root, f'order-template-{template}.npz')
        print (f'process gpt4v root: {gpt4v_root}')
        # for flag in flags:
        #     evaluate_performance_using_gpt4v(image_root, gpt4v_root, save_order, modality, template=template, bsize=bsize, batch_flag=flag, sleeptime=20, xishus=xishus)
        #     check_gpt4_performance(gpt4v_root, template)
        
        # 分析结果 [结果没有明显差距]
        label_path = os.path.join(save_root, 'label.csv')
        store_path = os.path.join(save_root, f'label-gpt4v-template-{template}.csv')
        get_results_and_update_label(gpt4v_root, label_path, store_path, template)
        '''
        case1: 435 -> 435 -> gtp4v performance: 57.93
        case2: 435 -> 435 -> gtp4v performance: 54.25
        '''
