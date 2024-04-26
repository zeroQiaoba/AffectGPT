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
