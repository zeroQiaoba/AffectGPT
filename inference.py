import os
import glob
import random
import argparse
import numpy as np
import pandas as pd

import torch
import torch.backends.cudnn as cudnn

import decord
decord.bridge.set_bridge('torch')

from video_llama.tasks import *
from video_llama.models import *
from video_llama.runners import *
from video_llama.processors import *
from video_llama.datasets.builders import *
from video_llama.common.config import Config
from video_llama.common.dist_utils import get_rank
from video_llama.common.registry import registry
from video_llama.conversation.conversation_video import Chat, Conversation, default_conversation, SeparatorStyle

def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def upload_imgorvideo(video_path, chat_state, subtitle=None):
    chat_state = default_conversation.copy()
    chat_state = Conversation(
        system= "",
        roles=("Human", "Assistant"),
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.SINGLE,
        sep="###",
    )
    img_list = []
    chat.upload_video(video_path, chat_state, img_list, subtitle)
    return chat_state, img_list
 
def gradio_ask(user_message, chat_state):
    chat.ask(user_message, chat_state)
    return chat_state

def gradio_answer(chat_state, img_list, num_beams, temperature):
    output_text, _ = chat.answer(conv=chat_state,
                                 img_list=img_list,
                                 num_beams=num_beams,
                                 temperature=temperature,
                                 max_new_tokens=300, 
                                 max_length=2000) # llama: max_token_num=2048
    return output_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference Process for Multimodal Emotion Reasoning")

    # default config
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--cfg-path", default='eval_configs/video_llama_eval_withaudio_stage3.yaml', help="path to configuration file.")
    parser.add_argument("--options", nargs="+", help="override some settings in the used config, format: --option xx=xx yy=yy zz=zz")

    # input for emotion reasoning dataset
    parser.add_argument('--video_root',  type=str, default="instruction-dataset/EmoReason/video-process",  help='video root')
    parser.add_argument('--label_path',  type=str, default="instruction-dataset/EmoReason/gt-eng.csv",  help='label path')
    parser.add_argument('--no_subtitle', action='store_true', default=False, help='whether use subtitle in the inference (A+V+T)')
    parser.add_argument('--user_message',type=str, default="From what clues can we infer the person's emotional state?", help='input user message')
    parser.add_argument('--save_root',   type=str, default=None,  help='where to save output')

    # test multiple saved files (accelerate)
    parser.add_argument('--ckpt_root',    type=str, default=None,  help='test multiple files')
    parser.add_argument('--test_epochs',  type=str, default=None,  help='test which epochs')
    args = parser.parse_args()
    cfg = Config(args)
    
    ## figure out test ckpts
    whole_ckpts = []
    start_epoch, end_epoch = args.test_epochs.split('-')
    for cur_epoch in range(int(start_epoch), int(end_epoch)+1):
        ckpts = glob.glob("%s/*%06d*.pth" %(args.ckpt_root, int(cur_epoch)))
        if len(ckpts) == 0: print('Error: target ckpt is not exists!')
        if len(ckpts) >  1: print('Error: target ckpt has at least 2!')
        ckpt = ckpts[0]
        whole_ckpts.append(ckpt)
    print ('test ckpts:')
    for item in whole_ckpts: print (item)

    for ii, ckpt in enumerate(whole_ckpts):
        print (f'=================== Process on {ckpt} =================')

        print ('Step1: initial models')
        cfg.model_cfg.ckpt_3 = ckpt # ckpt_3 has the highest priority
        model_config = cfg.model_cfg
        model_config.device_8bit = args.gpu_id # for low-resource run-up
        print("Load Checkpoint: {}".format(model_config.ckpt_3))

        if ii == 0: # first-round: initialize models
            model_cls = registry.get_model_class(model_config.arch)
            model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
            
        if ii > 0:  # second-round: update models
            ckpt = torch.load(model_config.ckpt_3, map_location="cpu")
            model.load_state_dict(ckpt['model'], strict=False)
            model = model.to('cuda:{}'.format(args.gpu_id))

        model = model.eval() # !! reduce randomness during the inference
        vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))


        print ('Step2: feed-forward process')
        whole_results = {}
        df = pd.read_csv(args.label_path)
        for _, row in df.iterrows():
            name     = row['names']
            emotion  = row['emotions']
            subtitle = row['subtitles']
            reason   = row['reasons']
            print (f'process on {name}')
            video_path = os.path.join(args.video_root, f'{name}.avi')
            user_message = args.user_message
            
            # process for one file
            chat_state, img_list = [], []
            if args.no_subtitle: subtitle = None
            chat_state, img_list = upload_imgorvideo(video_path, chat_state, subtitle=subtitle)
            chat_state = gradio_ask(user_message, chat_state)
            response = gradio_answer(chat_state, img_list, num_beams=1, temperature=1)
            print (f'assistant: {user_message}')
            print (f'answer: {response}')
            whole_results[name] = {
                'emotion': emotion,
                'subtitle': subtitle,
                'reason': reason,
                'pred_reasons': response,
            }


        print ('Step3: save results for one ckpt_3')
        save_root = os.path.basename(args.ckpt_root)
        if not os.path.exists(args.save_root): os.makedirs(args.save_root)
        epoch = os.path.basename(cfg.model_cfg.ckpt_3).split('_')[1]
        save_path = f'{args.save_root}/epoch-{epoch}.npz'
        np.savez_compressed(save_path,
                            whole_results=whole_results)