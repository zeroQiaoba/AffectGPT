# AffectGPT 

We aim to train an audio-video-text aligned model  to deal with explainable multimodal emotion reasoning. Specifically, we modify the code in [Video-LLaMA](https://github.com/DAMO-NLP-SG/Video-LLaMA) and make it support for audio-video-text aligned training. Meanwhile, we unify the input for different datasets, remove some unnecessary modules, add model.eval() during inference, etc. More details can be found in our implementation. We provide the code in **./AffectGPT**. Currently, we only provide the code training with **EMER-V1 dataset**.

1. Environment Preparation

```shell
conda env create -f environment.yml
```

- If raise errors about "OSError: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory", please run "pip install -U torch torchaudio --no-cache-dir"
- If your Cuda version is low (such as 10.2), please check the install instructions for pytorch-relate packages in "https://pytorch.org/get-started/previous-versions"



2. Dataset Preparation

```shell
# instruction-dataset/LLaVA
1. llava_instruct_150k.json: https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_instruct_150k.json
2. COCO train2017: https://cocodataset.org/#download

# instruction-dataset/EmoReason => i.e., EMER-V1 dataset
1. Download MER2023 Dataset
2. gt-eng.csv and gt-chi.csv

# instruction-dataset/MiniGPT-4
https://drive.google.com/file/d/1nJXhoEcy3KTExr17I7BXqY5Y9Lx_-n-9/view

# instruction-dataset/VideoChat
videochat_instruct_11k.json: https://drive.google.com/file/d/1C-7xmf42QUEi4ApXTcxBHr5nLvTWXyUi
webvid: https://github.com/m-bain/webvid. We also modify download.py for acceleration
```



3. Model Preparation

```shell
# models/finetune_vicuna7b_audiobranch.pth
https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-Series/resolve/main/finetune-vicuna7b-v2.pth

# models/finetune_vicuna7b_videobranch.pth
https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-Series/resolve/main/finetune_vicuna7b_audiobranch.pth

# models/blip2_pretrained_flant5xxl.pth
https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth

# models/eva_vit_g.pth
https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth

# models/imagebind_huge.pth
https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth

# models/pretrained_minigpt4.pth
https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze

# models/vicuna-7b-v0
llama-7b-hf: https://huggingface.co/decapoda-research/llama-7b-hf/tree/main
vicuna-7b-delta-v0: https://huggingface.co/lmsys/vicuna-7b-delta-v0/tree/main
python3 apply_delta.py --base {path for llama-7b-hf} --delta {path for vicuna-7b-delta-v0} --target vicuna-7b-v0

# ./bert-base-uncased
https://huggingface.co/bert-base-uncased
```



4. Training and Inference

```shell
conda activate videollama

# training with one gpu
python train.py --cfg-path=./train_configs/multimodal_llama_stage3_finetune.yaml

# training with 8 gpu
torchrun --nproc_per_node=8 train.py --cfg-path=./train_configs/multimodal_llama_stage3_finetune.yaml

# inference for one epoch (one emotion reasoning performance)
python inference.py --gpu-id=0 --cfg-path="eval_configs/video_llama_eval_withaudio_stage3.yaml" \
					--ckpt_root="video_llama/output/multimodal_llama_stage3_finetune/20230722155059_100_1000_1000" \
					--test_epochs="0-0"

# inference for multiple epochs (one emotion reasoning performance)
python inference.py --gpu-id=0 --cfg-path="eval_configs/video_llama_eval_withaudio_stage3.yaml" \
					--ckpt_root="video_llama/output/multimodal_llama_stage3_finetune/20230722155059_100_1000_1000" \
					--test_epochs="0-9"
```

