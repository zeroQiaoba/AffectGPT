# Explainable Multimodal Emotion Reasoning (EMER) & AffectGPT 



## Motivation

Emotions are related to multi-faceted clues, such as facial expressions, prosody, gestures (or micro-gestures), etc. It is inappropriate to identify emotions from just one aspect. **EMER provides a general format for all emotion-related tasks, aiming to integrate multiple clues and generate more comprehensive descriptions.**

<img src="image\example.png" alt="example " style="zoom:80%;" />

Details can be found in our paper: [**Explainable Multimodal Emotion Reasoning**](https://arxiv.org/pdf/2306.15401.pdf)

```tex
@article{lian2023explainable,
  title={Explainable Multimodal Emotion Reasoning},
  author={Lian, Zheng and Sun, Licai and Xu, Mingyu and Sun, Haiyang and Xu, Ke and Wen, Zhuofan and Chen, Shun and Liu, Bin and Tao, Jianhua},
  journal={arXiv preprint arXiv:2306.15401},
  year={2023}
}
```



## EMER Dataset

To construct the initial dataset, we select samples from MER 2023.

(1) Download [**Raw MER 2023 Dataset**](https://dl.acm.org/doi/abs/10.1145/3581783.3612836)

To download the dataset, please fill out an [**EULA**](https://drive.google.com/file/d/1LOW2e6ZuyUjurVF0SNPisqSh4VzEl5lN) and send it to **lianzheng2016@ia.ac.cn**.



(2) EMER-V1

Due to the high annotation cost, we only select 100 non-neutral samples to form the initial dataset. See **https://arxiv.org/abs/2306.15401v3** for more details. We provide the annotated results in **./EMER/dataset-v1**. (100 samples)



(3) EMER-V2

However, the description obtained in previous manner is short and cannot cover multi-faceted clues. Therefore, we use GPT-4V to provide initial annotations, combining with manual check and ChatGPT's reasoning capabilities. See **https://arxiv.org/abs/2306.15401** for more details. We provide the annotated results in **./EMER/dataset-v2**. (332 samples)



## AffectGPT

We aim to train an audio-video-text aligned model  to deal with explainable multimodal emotion reasoning. Specifically, we modify the code in [Video-LLaMA](https://github.com/DAMO-NLP-SG/Video-LLaMA) and make it support for audio-video-text aligned training. Meanwhile, we unify the input for different datasets, remove some unnecessary modules, add model.eval() during inference, etc. More details can be found in our implementation. We provide the code in **./AffectGPT**. Currently, we only provide the code training with **EMER-V1 dataset**.

More experimental results can be found in **https://arxiv.org/abs/2306.15401v3**.
