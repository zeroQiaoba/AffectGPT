import os
import time
import glob
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

import decord
decord.bridge.set_bridge('torch')

from my_affectgpt.tasks import *
from my_affectgpt.models import *
from my_affectgpt.runners import *
from my_affectgpt.processors import *
from my_affectgpt.datasets.builders import *
from my_affectgpt.common.config import Config
from my_affectgpt.common.dist_utils import get_rank
from my_affectgpt.common.registry import registry
from my_affectgpt.conversation.conversation_video import Chat
from my_affectgpt.datasets.builders.image_text_pair_builder import * # pull in every dataset class

import config
from toolkit.utils.read_files import *


# use the root that stores the largest number of checkpoints
def search_for_ckpt_root(root_candidates):
    if len(root_candidates) == 0:
        return ''
    
    # keep the root that holds the most checkpoint files
    maxcount = 0
    targetroot = ''
    for root in root_candidates:
        count = len([path for path in os.listdir(root) if path.startswith('checkpoint_')])
        print (root, '==>', count)
        if count > maxcount:
            maxcount = count
            targetroot = root
    print ('================================================')
    print (f'Targetroot: epoch range: 0-{maxcount-1}')
    
    # report when the last checkpoint of the target root was created
    last_file = sorted(glob.glob(targetroot + '/checkpoint*'))[-1]
    file_stat = Path(last_file).stat()
    creation_time = file_stat.st_ctime
    print("Targetroot: Last ckpt creation time:", datetime.fromtimestamp(creation_time))
    print ('================================================')
    return targetroot


# case1: default => last epoch
# case2: inference_cfg.test_epoch == a; run that epoch only
# case3: inference_cfg.test_epochs == a-b; run a, a+skip_epoch, ... up to b
def get_ckpt3_candidates(ckpt3_root, inference_cfg):
    
    if inference_cfg.test_epoch != 'xxx':
        cur_epoch = inference_cfg.test_epoch
        ckpts = glob.glob("%s/*%06d*.pth" %(ckpt3_root, int(cur_epoch)))
        assert len(ckpts) == 1, 'Error: (ckpt, epoch) combination is not exists or contain multiple candidates!'
        return [ckpts[0]]
    
    elif inference_cfg.test_epochs == 'xxx-xxx':
        last_ckpt = sorted(glob.glob("%s/*.pth" %(ckpt3_root)))[-1]
        last_epoch=  int(last_ckpt.split('_')[-3])
        assert last_epoch > 10, f'Error: too less training time to conduct automatic inference!'
        return [last_ckpt]
    
    else:
        start_epoch, end_epoch = inference_cfg.test_epochs.split('-')
        skip_epoch = int(inference_cfg.skip_epoch) 
        whole_ckpts = []
        for cur_epoch in range(int(start_epoch), int(end_epoch)+1):
            if cur_epoch % skip_epoch == 0:
                ckpts = glob.glob("%s/*%06d*.pth" %(ckpt3_root, int(cur_epoch)))
                assert len(ckpts) == 1, 'Error: (ckpt, epoch) combination is not exists or contain multiple candidates!'
                whole_ckpts.append(ckpts[0])
        return whole_ckpts


# face_or_frame comes from the training config, and must agree with the video data used at training time
def get_face_or_frame(datasets_cfg, outside_face_or_frame):
    if outside_face_or_frame is not None:
        return outside_face_or_frame
    
    face_or_frame_candidates = []
    if 'mercaptionplus' in datasets_cfg:
        face_or_frame_candidates.append(datasets_cfg['mercaptionplus'].face_or_frame)
    if 'mer25ov1000' in datasets_cfg:
        face_or_frame_candidates.append(datasets_cfg['mer25ov1000'].face_or_frame)
    assert len(set(face_or_frame_candidates)) == 1, f'must has the unified face_or_frame type'
    face_or_frame = list(set(face_or_frame_candidates))[0]
    return face_or_frame


def get_name2cls(dataset):
    if dataset == 'MER2023':          return MER2023_Dataset()
    if dataset == 'MER2024':          return MER2024_Dataset()
    if dataset == 'MELD':             return MELD_Dataset()
    if dataset == 'IEMOCAPFour':      return IEMOCAPFour_Dataset()
    if dataset == 'CMUMOSI':          return CMUMOSI_Dataset()
    if dataset == 'CMUMOSEI':         return CMUMOSEI_Dataset()
    if dataset == 'SIMS':             return SIMS_Dataset()
    if dataset == 'SIMSv2':           return SIMSv2_Dataset()
    if dataset == 'OVMERDPlus':       return OVMERDPlus_Dataset()
    print ('dataset cls not provided!')
    return None


# priority: zeroshot > dataset specific
def get_user_message(dataset_cls, zeroshot, outside_user_message):
    if outside_user_message is not None:
        user_message = outside_user_message
    elif zeroshot: # predict ov labels
        user_message = dataset_cls.func_get_qa_ovlabel(sample=None, question_only=True)
    else: # dataset-specific prompt
        if dataset in ['MER2023', 'MER2024', 'MELD', 'IEMOCAPFour']:
            user_message = dataset_cls.func_get_qa_onehot_w_candidates(sample=None, question_only=True)
        elif dataset in ['CMUMOSI', 'CMUMOSEI', 'SIMS', 'SIMSv2']:
            user_message = dataset_cls.func_get_qa_valence(sample=None, question_only=True)
        elif dataset == 'OVMERDPlus':
            user_message = dataset_cls.func_get_qa_ovlabel(sample=None, question_only=True)
    return user_message


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AffectGPT Inference Process")
    parser.add_argument("--cfg-path", default='xxx', help="path to configuration file.")
    parser.add_argument("--options",  nargs="+", help="override some settings in the used config, format: --option xx=xx yy=yy zz=zz")
    parser.add_argument("--dataset", default='OVMERDPlus', help="evaluate dataset: one name in inference_datasets, comma-separated names, or 'inferenceData' for all")
    parser.add_argument('--zeroshot', action='store_true', default=False, help='whether testing on zeroshot performance?') # which prompt is used
    ## the parameters below are rarely needed by the model
    parser.add_argument('--outside_user_message',  default=None, help="we use the outside user message, rather than dataset dependent.")
    parser.add_argument('--outside_face_or_frame', default=None, help="we use the outside face_or_frame, rather than dataset dependent.")
    args = parser.parse_args()
    cfg = Config(args)
    model_cfg = cfg.model_cfg
    datasets_cfg = cfg.datasets_cfg
    inference_cfg = cfg.inference_cfg
    device = 'cuda:{}'.format(inference_cfg.gpu)
    ## the only datasets this script handles, keep in sync with get_name2cls()
    inference_datasets = ['OVMERDPlus', 'MER2023', 'MER2024', 'MELD', 'IEMOCAPFour', 'CMUMOSI', 'CMUMOSEI', 'SIMS', 'SIMSv2']


    print ('======== Step1: cfg pre-analysis ========')
    # ckpt_root and ckpt_name are both accepted here => (ckpt3_root)
    # by default it is derived from os.path.basename(args.cfg_path)
    if inference_cfg.ckpt_root not in ['', 'xxx']:
        ckpt3_root = inference_cfg.ckpt_root
    elif inference_cfg.ckpt_name not in ['', 'xxx']:
        cfg_name = os.path.basename(args.cfg_path)[:-len('.yaml')]
        ckpt3_root = os.path.join('output', cfg_name, inference_cfg.ckpt_name)
        assert inference_cfg.ckpt_name.startswith(cfg_name) # has to agree with how train.py names the directory
    else:
        print ('strat searching for suitable ckpt_root')
        cfg_name = os.path.basename(args.cfg_path)[:-len('.yaml')]
        root_candidates = glob.glob(os.path.join('output', cfg_name, cfg_name+'*'))
        ckpt3_root = search_for_ckpt_root(root_candidates)
    print ('processed ckpt3 root:')
    print (ckpt3_root)

    # (ckpt3_root) => processed epochs
    print ('processed ckpt3 epochs:')
    whole_ckpt3s = get_ckpt3_candidates(ckpt3_root, inference_cfg)
    for item in whole_ckpt3s: print (os.path.basename(item))

    # => (face_or_frame); it has to match the face_or_frame the model was trained with
    face_or_frame = get_face_or_frame(datasets_cfg, args.outside_face_or_frame)
    print (f'Read data type: {face_or_frame}')
    print ('=======================================')


    ## main process for each ckpt3 candidates
    for ii, ckpt_3 in enumerate(whole_ckpt3s):

        ##############################################################
        print (f'======== Step2: initial model; using ckpt_3: {os.path.basename(ckpt_3)} ========')
        model_cfg.ckpt_3 = ckpt_3 # ckpt_3 has the highest priority
        if ii == 0: # first-round: initialize models
            model_cls = registry.get_model_class(model_cfg.arch) # affectgpt
            model = model_cls.from_config(model_cfg)
        if ii > 0:  # second-round: overwrite the trainable params with the new ckpt_3
            ckpt = torch.load(model_cfg.ckpt_3, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt['model'], strict=False)
        model = model.to(device).eval() # !! reduce randomness during the inference

        if model_cfg.arch.startswith('affectgpt_r1'): # r1 only use policy model
            chat = Chat(model.model, model_cfg, device=device)
        else: # other case use the normal model
            chat = Chat(model, model_cfg, device=device)
        ##############################################################


        print ('======== Step3: Inferece ========')
        if args.dataset == 'inferenceData':
            process_datasets = inference_datasets
        else:
            names = args.dataset.split(',')
            for name in names: assert name in inference_datasets, f'unsupported dataset: {name}'
            process_datasets = names
        print ('process datasets: ', process_datasets)

        ## for each dataset
        for dataset in process_datasets:
            print (f'current dataset: {dataset}')
            ## the train / inference state dataset_cls keeps for itself
            dataset_cls = get_name2cls(dataset)
            dataset_cls.needed_data = dataset_cls.get_needed_data(face_or_frame)
            dataset_cls.vis_processor = BaseProcessor()
            dataset_cls.img_processor = BaseProcessor()
            vis_processor_cfg = inference_cfg.get("vis_processor") # read vis processor
            img_processor_cfg = inference_cfg.get("img_processor") # read img processor
            if vis_processor_cfg is not None:
                dataset_cls.vis_processor = registry.get_processor_class(vis_processor_cfg.train.name).from_config(vis_processor_cfg.train)
            if img_processor_cfg is not None:
                dataset_cls.img_processor = registry.get_processor_class(img_processor_cfg.train.name).from_config(img_processor_cfg.train)
            dataset_cls.n_frms = model_cfg.vis_processor.train.n_frms


            ## read the content of each dataset
            test_names = dataset_cls.read_test_names()
            name2subtitle = dataset_cls.name2subtitle

            ## where the results go; skip when they are already there
            save_root = os.path.join(inference_cfg.base_root + f'-{dataset.lower()}', # output/results-{dataset}/ckpt3_name
                                    os.path.basename(ckpt3_root)) 
            if not os.path.exists(save_root): os.makedirs(save_root)
            epoch = os.path.basename(cfg.model_cfg.ckpt_3)[:-4]
            save_path = '%s/%s.npz' %(save_root, epoch) # output/result-{dataset}/ckpt3_name/epochname
            if os.path.exists(save_path): continue

            ## main loop; this is the time-consuming part
            name2reason = {}
            for ii, name in enumerate(test_names):
                subtitle = name2subtitle[name]
                print (f'process on {ii}|{len(test_names)}: {name} | {subtitle}')

                # read the paths in the form dataset_cls expects
                sample = {'name': name}
                video_path, image_path, audio_path, face_npy = None, None, None, None
                if hasattr(dataset_cls, '_get_video_path'): video_path = dataset_cls._get_video_path(sample)
                if hasattr(dataset_cls, '_get_audio_path'): audio_path = dataset_cls._get_audio_path(sample)
                if hasattr(dataset_cls, '_get_face_path'):  face_npy   = dataset_cls._get_face_path(sample)
                if hasattr(dataset_cls, '_get_image_path'): image_path = dataset_cls._get_image_path(sample)
                sample_data = dataset_cls.read_frame_face_audio_text(video_path, face_npy, audio_path, image_path)

                # => img_list
                audio_llms, frame_llms, face_llms, image_llms, multi_llms = None, None, None, None, None
                audio_hiddens, audio_llms = chat.postprocess_audio(sample_data)  
                frame_hiddens, frame_llms = chat.postprocess_frame(sample_data)
                face_hiddens,  face_llms  = chat.postprocess_face(sample_data)
                _,             image_llms = chat.postprocess_image(sample_data)
                if face_or_frame.startswith('multiface'):
                    _, multi_llms = chat.postprocess_multi(face_hiddens, audio_hiddens)
                elif face_or_frame.startswith('multiframe'):
                    _, multi_llms = chat.postprocess_multi(frame_hiddens, audio_hiddens)

                img_list = {}
                img_list['audio'] = audio_llms
                img_list['frame'] = frame_llms
                img_list['face']  = face_llms
                img_list['image'] = image_llms
                img_list['multi'] = multi_llms

                # get prompt (if use zeroshot => ov labels; else => dataset specific question)
                user_message = get_user_message(dataset_cls, args.zeroshot, args.outside_user_message)
                prompt = dataset_cls.get_prompt_for_multimodal(face_or_frame, subtitle, user_message)
                
                # => call function
                response = chat.answer_sample(prompt=prompt, img_list=img_list,
                                            num_beams=1, temperature=1, do_sample=True, top_p=0.9, 
                                            max_new_tokens=1200, max_length=2000) # llama: max_token_num=2048
                name2reason[name] = response
                print (response)

            print ('save results')
            np.savez_compressed(save_path, name2reason=name2reason)
