# Explainable Multimodal Emotion Reasoning (EMER) & AffectGPT 

To construct the initial dataset, we select samples from MER 2023.

(1) Download [**Raw MER 2023 Dataset**](https://dl.acm.org/doi/abs/10.1145/3581783.3612836)

To download the dataset, please fill out an [**EULA**](https://drive.google.com/file/d/1LOW2e6ZuyUjurVF0SNPisqSh4VzEl5lN) and send it to **lianzheng2016@ia.ac.cn**.



(2) EMER-V1

Due to the high annotation cost, we only select 100 non-neutral samples to form the initial dataset. See **https://arxiv.org/abs/2306.15401v3** for more details. We provide the annotated results in **./dataset-v1**. (100 samples)

```shell
## gt-chi.csv and gt-eng.csv provide Chinese and English version annotations.
names: video name
emotions: discrete labels provided by MER2023 
subtitles: corrected subtitles
reasons: EMER labels
```



(3) EMER-V2

However, the description obtained in previous manner is short and cannot cover multi-faceted clues. Therefore, we use GPT-4V to provide initial annotations, combining with manual check and ChatGPT's reasoning capabilities. See **https://arxiv.org/abs/2306.15401v5** for more details. We provide the annotated results in **./dataset-v2**. (332 samples)

```shell
## final-EMER-reason.csv
names: video name
chinese: Chinese version EMER annotation
english: English version EMER annotation
```



