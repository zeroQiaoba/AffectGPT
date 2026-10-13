import os
import cv2
import glob
import tqdm
import shutil
import random
import pickle
import numpy as np
import pandas as pd

from toolkit.globals import *
from toolkit.utils.read_files import func_read_text_file


# (iter, meanloss)
def func_iter_to_meanloss(out_file):
    tgt_items = []
    lines = func_read_text_file(out_file)
    for line in lines:
        if line.find('meanloss') != -1:
            iter = int(line.split('|')[0].split()[-1])
            meanloss = float(line.split('meanloss: ')[-1])
            if iter % 2000 == 0: # 每 2000 epoch 测试一次
                tgt_items.append([iter, meanloss])
    return tgt_items

# 将 video processor 放置在 目标文件夹内
def update_processor(model_root, processor_path):
    for model_dir in glob.glob(model_root + '/*'):
        processor_name = os.path.basename(processor_path)
        tgt_path = os.path.join(model_dir, processor_name)
        shutil.copy(processor_path, tgt_path)

# 依据out文件，将备选的epoch模型保存在一个folder下，并把processor也放在指定路径下面，并把模型重命名
def selected_models_for_videomae(out_file, model_root, save_root, model_name='videomae-base'):

    # analyze (iter, meanloss)
    tgt_items = func_iter_to_meanloss(out_file)
    
    # (iter, meanloss) -> model_dir
    tgt_models = []
    for tgt_item in tgt_items:
        item, meanloss = tgt_item
        select_models = glob.glob(f'{model_root}/*{item}*meanloss:{meanloss}*')
        if len(select_models) == 1: # 如果存在重叠的情况，则不测试这个model
            tgt_models.append(select_models[0])
    
    out_name = os.path.basename(out_file)[:-4]
    save_dir = os.path.join(save_root, out_name)
    if not os.path.exists(save_root): os.makedirs(save_root)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    for model in tgt_models:
        modelname = os.path.basename(model)
        newname = modelname[modelname.find('_epoch:')+len('_'):].rsplit('_', 1)[0]
        save_path = os.path.join(save_dir, newname)
        cmd = f'mv {model} {save_path}'
        os.system(cmd)
    
    # 更新 processor
    if model_name == 'videomae-base':
        processor_path = os.path.join(config.PATH_TO_PRETRAINED_MODELS, 'transformers/videomae-base/preprocessor_config.json')
    elif model_name == 'videomae-large':
        processor_path = os.path.join(config.PATH_TO_PRETRAINED_MODELS, 'transformers/videomae-large/preprocessor_config.json')
    update_processor(save_dir, processor_path)


# 生成所有 videomae models 抽取特征的脚本
def generate_all_feature_extraction_cmds(model_name, model_root):
    cmds = []
    for ii, model_dir in enumerate(sorted(glob.glob(model_root + '/*'))):
        cmd = f"nohup python -u extract_vision_huggingface.py --dataset=MER2023 --feature_level='UTTERANCE' --model_name={model_name} --model_dir={model_dir} --gpu=0 > temp{ii}.out &"
        cmds.append(cmd)
    return cmds


# 检测特征是否都提取完整了
def check_feature_completeness():
    feature_root = config.PATH_TO_FEATURES['MER2023']
    for feature_name in os.listdir(feature_root):
        if feature_name.startswith('videomae'):
            feature_dir = os.path.join(feature_root, feature_name)
            samples = glob.glob(feature_dir + '/*')
            sample_num = len(samples)
            if sample_num != 0:
                index = random.randint(0, sample_num-1)
                feature_shape = np.load(samples[index]).shape
            else:
                feature_shape = (0, 0)
            print (f'{feature_name} => shape: {feature_shape}  number: {sample_num}')


def generate_attention_model_cmds():
    whole_cmds = []
    feature_root = config.PATH_TO_FEATURES['MER2023']
    for feature_name in os.listdir(feature_root):
        if feature_name.startswith('videomae'):
            cmd = f"python -u main-release.py --model='attention' --feat_type='utt' --dataset=MER2023 --audio_feature={feature_name} --text_feature={feature_name} --video_feature={feature_name} --save_root='./savedtemp' --gpu="
            print (cmd)
            whole_cmds.append(cmd)
    return  whole_cmds


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def performance_analysis(res_root, prefix='videomae-base', model_name='pretrain01'):

    ## 计算 baseline 结果
    baseline_accs = []
    for res_path in glob.glob(res_root + f'/test1_features:{prefix}-UTT*'):
        res_name = os.path.basename(res_path)
        acc = float(res_name.split('_f1:')[1].split('_')[0])
        baseline_accs.append(acc)
    baseline_top2acc = np.mean(np.sort(baseline_accs)[-2:])
        
    ## 计算不同epoch下的结果
    model2res = {}
    for res_path in glob.glob(res_root + f'/test1_features:{prefix}-*{model_name}*'):
        res_name = os.path.basename(res_path)
        # gain items
        model = res_name[res_name.find('epoch'):].split('_meanloss')[0]
        loss  = float(res_name.split('_meanloss:')[1].split('_')[0])
        acc   = float(res_name.split('_f1:')[1].split('_')[0])
        
        # save
        if model not in model2res: 
            model2res[model] = []
        model2res[model].append((loss, acc))
    # 找到 top 2 acc
    for model in model2res:
        reses = model2res[model]
        loss = reses[0][0]
        accs = [item[1] for item in reses]
        top2acc = np.mean(np.sort(accs)[-2:])
        model2res[model] = (loss, top2acc)
    

    ## 绘制结果曲线图 => 一个是acc 一个是loss
    losses = [model2res[model][0] for model in sorted(model2res)]
    accs   = [model2res[model][1] for model in sorted(model2res)]
    baselines = [baseline_top2acc] * len(losses)
    xx = np.arange(len(losses))

    save_path = f'{model_name}-pretrain-acc.jpg'
    fontsize = 20
    plt.figure()
    plt.plot(xx, accs, color='k')
    plt.plot(xx, baselines, color='b')
    plt.xlabel('iters', fontsize=fontsize)
    plt.ylabel('acc', fontsize=fontsize)
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, format='png')
    plt.close()

    save_path = f'{model_name}-pretrain-loss.jpg'
    plt.figure()
    plt.plot(xx, losses, color='r')
    plt.xlabel('iters', fontsize=fontsize)
    plt.ylabel('meanloss', fontsize=fontsize)
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, format='png')
    plt.close()
    
def feature_conversion_from_sunlicai(input_pkl, save_root):
    if not os.path.exists(save_root): os.makedirs(save_root)
    saved_features = pickle.load(open(input_pkl, 'rb'))
    for sample, sample_dict in tqdm.tqdm(saved_features.items()):
        sample_name = os.path.basename(sample).rsplit('.', 1)[0]
        sample_file = os.path.join(save_root, f'{sample_name}.npy')
        sample_feature = sample_dict['feature']
        np.save(sample_file, sample_feature)


def generate_csv_folder_for_sun(process_datasets, save_root):
    for dataset in process_datasets:
        save_dir = os.path.join(save_root, dataset)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        openface_root = config.PATH_TO_RAW_FACE[dataset]
        video_files = sorted(glob.glob(os.path.join(openface_root, "*/*.npy")))
        print (f'process sample number: {len(video_files)}')
        print (f'first file: {video_files[0]}')
        print (f'last file: {video_files[-1]}')

        train_label_list = []
        for video_file in video_files:
            train_label_list.append([video_file, 0])

        for csvname in ['train.csv', 'val.csv', 'test.csv']:
            train_split_file = os.path.join(save_dir, csvname)
            df = pd.DataFrame(train_label_list)
            df.to_csv(train_split_file, header=None, index=False, sep=' ')


def feature_conversion_main(save_root):
    for pkl_path in glob.glob(save_root + '/*.pkl'):
        # MER2023-videomaebase-299.pkl
        # CMUMOSI-videomae-large-K400-VoxCeleb2-9.pkl
        # SIMS-videomae-base-VoxCeleb2-99.pkl # => pretrain-videomae-base-VoxCeleb2-99
        pklname = os.path.basename(pkl_path)[:-4]
        dataset, featname = pklname.split('-', 1)
        save_root = os.path.join(config.PATH_TO_FEATURES[dataset], 'pretrain-'+featname)
        if os.path.exists(save_root):
            print ('error!!')
            continue
        feature_conversion_from_sunlicai(pkl_path, save_root)


def compress_images_to_npy(data_root, save_root):
    
    for video_root in sorted(glob.glob(data_root + '/*')):
        print (f'process on {video_root}')

        ## read (frames, framenames)
        frames = []
        for frame_name in tqdm.tqdm(sorted(os.listdir(video_root))):
            if frame_name.endswith('.jpg') or frame_name.endswith('.bmp'):
                frame_path = os.path.join(video_root, frame_name)
                frame = cv2.imread(frame_path)
                frames.append(frame)
        print (f'process frame number: {len(frames)}')

        videoname = os.path.basename(video_root)
        save_dir = os.path.join(save_root, videoname)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, videoname+'.npy')
        np.save(save_path, frames)


# run -d toolkit/utils/videomae_utils.py
if __name__ == '__main__':
    # import fire
    # fire.Fire()

    ## 将目标模型移动到指定文件夹内
    # project_root = '/share/home/lianzheng/chinese-mer-2023'
    # # out_file = os.path.join(project_root, 'videomae-base-polish-pretrain01.out')
    # out_file = os.path.join(project_root, 'videomae-base-polish-pretrain02.out')
    # model_root = os.path.join(project_root, 'saved-others/model')
    # save_root = os.path.join(project_root, 'saved-others/select-model')
    # selected_models_for_videomae(out_file, model_root, save_root, model_name='videomae-base')

    ## 生成特征提取的脚本
    # whole_cmds = []
    # model_root = os.path.join(project_root, 'saved-others/select-model/videomae-base-polish-pretrain01')
    # whole_cmds.extend(generate_all_feature_extraction_cmds('videomae-base', model_root))
    # model_root = os.path.join(project_root, 'saved-others/select-model/videomae-base-polish-pretrain02')
    # whole_cmds.extend(generate_all_feature_extraction_cmds('videomae-base', model_root))
    # for cmd in whole_cmds:
    #     print (cmd)
    
    ## 判断一下特征是否都提取完整了
    # check_feature_completeness()

    ## 生成模型训练脚本并测试特征性能
    # generate_attention_model_cmds()

    ## 性能分析
    # res_root = os.path.join(project_root, 'savedtemp-unimodal/result')
    # performance_analysis(res_root, prefix='videomae-base', model_name='pretrain01')
    # performance_analysis(res_root, prefix='videomae-base', model_name='pretrain02')
    # performance_analysis(res_root, prefix='videomae-large')

    ## 测试孙总提取的videomae-base预训练特征
    # for (input_pkl, save_name) in [('overall_feature_ckpt99.pkl',  'sunlicai-videomae-base-celebvox'),
    #                                ('overall_feature_ckpt149.pkl', 'sunlicai-videomae-base-merunlabel-50epoch'),
    #                                ('overall_feature_ckpt199.pkl', 'sunlicai-videomae-base-merunlabel-100epoch'),
    #                                ('overall_feature_ckpt299.pkl', 'sunlicai-videomae-base-merunlabel-200epoch')]:

    # for (input_pkl, save_name) in [('k400_overall_feature_ckpt800.pkl',  'sunlicai-k400-ckpt800'),
    #                                ('k400_overall_feature_ckpt1600.pkl', 'sunlicai-k400-ckpt1600'),
    #                                ('ssv2_overall_feature_ckpt800.pkl',  'sunlicai-ssv2-ckpt800'),
    #                                ('ssv2_overall_feature_ckpt2400.pkl', 'sunlicai-ssv2-ckpt2400')]:
    # for (input_pkl, save_name) in [('kinetics-400_pretrain_e1600_checkpoint_input_size224.pkl',  'sunlicai-k400-e1600-size224')]:

    # for (input_pkl, save_name) in [('overall_feature_ckpt49_initialize_from_k400_weight_base.pkl',  'sunlicai-k400-mer50'),
    #                                ('overall_feature_ckpt349.pkl',  'sunlicai-vox-mer250'),
    #                                ('overall_feature_ckpt399.pkl',  'sunlicai-vox-mer300'),
    #                                ('overall_feature_ckpt449.pkl',  'sunlicai-vox-mer350')]:

    # for (input_pkl, save_name) in [('overall_feature_ckpt99_initialize_from_k400_weight_base.pkl',   'sunlicai-k400-mer100-base'),
    #                                ('overall_feature_ckpt149_initialize_from_k400_weight_base.pkl',  'sunlicai-k400-mer150-base'),
    #                                ('overall_feature_ckpt49_initialize_from_k400_weight_large.pkl',  'sunlicai-k400-mer50-large'),
    #                                ('videomae_large_kinetics-400_pretrain_e1600_checkpoint_input_size224_after_debug.pkl',  'sunlicai-k400-large')]:

    # for (input_pkl, save_name) in [
    #                                ('overall_feature_ckpt199_initialize_from_k400_weight_large.pkl', 'sunlicai-k400-mer200-large'),
    #                                ('overall_feature_ckpt249_initialize_from_k400_weight_large.pkl', 'sunlicai-k400-mer250-large'),
    #                                ('overall_feature_ckpt299_initialize_from_k400_weight_large.pkl', 'sunlicai-k400-mer300-large'),
    #                                ('overall_feature_ckpt299_initialize_from_k400_weight_base.pkl', 'sunlicai-k400-mer300-base'),
    #                                ('overall_feature_ckpt399_initialize_from_k400_weight_base.pkl', 'sunlicai-k400-mer400-base'),
    #                                ('overall_feature_ckpt499_initialize_from_k400_weight_base.pkl', 'sunlicai-k400-mer500-base'),
    #                                ('overall_feature_ckpt599_initialize_from_k400_weight_base.pkl', 'sunlicai-k400-mer600-base'),
    #                                ]:
    #     save_root = os.path.join(config.PATH_TO_FEATURES['MER2023'], save_name)
    #     if os.path.exists(save_root):
    #         print ('error!!')
    #         continue
    #     feature_conversion_from_sunlicai(input_pkl, save_root)

    ## 用于生成folder提取sun的预训练特征 [模型选择如下]
    '''
    sunlicai-k400-mer300-base: **60.38+0.36**  overall_feature_ckpt299_initialize_from_k400_weight_base.pkl
    sunlicai-k400-mer50-large: **59.78+0.15**  overall_feature_ckpt49_initialize_from_k400_weight_large.pkl
    '''

    ## 孙总特征提取脚本
    # process_datasets = ['MER2023', 'CMUMOSI', 'CMUMOSEI', 'SIMS', 'SIMSv2', 'MELD', 'IEMOCAPSix']
    # save_root = '/share/home/lianzheng/chinese-mer-2023/feature_extraction/visual/videomae-sun'
    # ## 生成train.csv/val.csv/test.csv用于测试
    # # generate_csv_folder_for_sun(process_datasets, save_root)
    # ## 将特征进行解析并存储
    # feature_conversion_main(save_root)

    ## 提取卓凡的特征 
    # => !! 直接在输入端改成处理帧级别特征 !! 
    # => 这种存储方式，产生的文件太多太大了，还是需要修改孙总的代码
    process_datasets = ['affwild2']
    ## 将图片压缩到npy格式中
    ori_root = '/share/home/lianzheng/emotion-data/affwild2/ori_faces'
    save_root = '/share/home/lianzheng/emotion-data/affwild2/openface_face'
    compress_images_to_npy(ori_root, save_root)
    ## 生成train.csv/val.csv/test.csv用于测试
    # save_root = '/share/home/lianzheng/chinese-mer-2023/feature_extraction/visual/videomae-sun'
    # generate_csv_folder_for_sun(process_datasets, save_root)
    ## 将特征进行解析并存储
    # feature_conversion_main(save_root)
