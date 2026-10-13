#######################################################################################################
# Stage 0: environment setup and dataset preparation
#######################################################################################################
# Download audio/, video/ and openface_face/
# [Option 1] download the full dataset from https://huggingface.co/datasets/MERChallenge/MER2025
# [Option 2] download */*_track2_train_human and */*_track2_train_mercaptionplus from https://huggingface.co/datasets/MERChallenge/MER2026

# All the subtitles go into subtitle_chieng.csv

# The label files of the three datasets also go under dataset/

#######################################################################################################
# Stage 1: cold start — train the model to produce output in <think></think><answer></answer> format
#######################################################################################################
# [1] Training
CUDA_VISIBLE_DEVICES=0 python train.py \
    --cfg-path=train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_face_r1stage1.yaml

# [2] Inference
CUDA_VISIBLE_DEVICES=0 python inference_hybird.py --zeroshot --dataset='OVMERDPlus' \
    --outside_user_message="Please recognize all possible emotional states of the character." \
    --cfg-path=train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_face_r1stage1.yaml \
    --options "inference.test_epochs=10-60" "inference.skip_epoch=5"


#####################################################################################################################
# Stage 2: reinforcement learning — align the token-level training loss with the semantic-level evaluation metric
#####################################################################################################################
# [1] Training
CUDA_VISIBLE_DEVICES=0 python train.py \
    --cfg-path=train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_face_r1stage2_mer25ov1000_continue60_g4_formatbeta00_ovlabel_numrestrict3.yaml

# [2] Inference
CUDA_VISIBLE_DEVICES=0 python inference_hybird.py --zeroshot --dataset='OVMERDPlus' \
    --outside_user_message="Please recognize all possible emotional states of the character." \
    --cfg-path=train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_face_r1stage2_mer25ov1000_continue60_g4_formatbeta00_ovlabel_numrestrict3.yaml \
    --options "inference.test_epochs=5-60" "inference.skip_epoch=5"


#####################################################################################################################
# Stage 3: score calculation
#####################################################################################################################
CUDA_VISIBLE_DEVICES=0 python statistic.py

