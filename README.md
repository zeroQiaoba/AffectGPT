<h3 align="center"><a href="https://openreview.net/pdf?id=Y8lfuSoqQz" style="color:#9C276A">
OV-MER &</a><a href="https://openreview.net/pdf?id=xmbdACI0xu" style="color:#9C276A">
AffectGPT &</a><a href="https://arxiv.org/pdf/2507.04278" style="color:#9C276A">
EmoPrefer &</a><a href="https://arxiv.org/pdf/2605.06126" style="color:#9C276A">
AffectGPT-RL</a></h3> 

<h5 align="center"> If our project helps you, please give us a star ⭐ on GitHub to support us. 🙏🙏 </h2>

<h5 align="center">


[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](AffectGPT/LICENSE)
</h5>

## ✨ OV-MER

**OV-MER** transitions from *traditional MER* to a framework that enables the prediction of *any number and category of emotions*, thereby advancing emotion AI toward real-world applicability by capturing the full spectrum of human emotions.

**(a) Task Comparison**: We compare the differences among three tasks (one-hot MER, multi-label MER, and OV-MER) across three aspects (label space, label number, and annotation manner).

**(b) Label Comparison**: We provide an example to visualize the one-hot and OV labels.

<img src="OV-MER/assert/task.png" width="800" />

## 🚀 AffectGPT
We provide specifically designed framework, AffectGPT, for the OV-MER task.

<img src="AffectGPT/assert/dataset-pipeline.png" width="800" />


## ✨ EmoPrefer
We propose **EmoPrefer**, a pioneering work exploring the potential of LLMs in decoding human emotion preferences. Specifically, we construct the first emotion preference dataset, **EmoPrefer-Data**, featuring high-quality preference annotations from experts. Additionally, we introduce **EmoPrefer-Bench**, which evaluates the performance of various MLLMs and prompting techniques in preference prediction, while also revealing new strategies to enhance their performance.

<img src="EmoPrefer/assert/framework.png" width="800" />

## 🚀 AffectGPT-RL
**Open-Vocabulary Multimodal Emotion Recognition (OV-MER)** removes the restriction of a predefined label space, enabling fine-grained emotion understanding. Instead of relying on discriminative classifiers, OV-MER uses generative models to capture the full spectrum of emotions, and evaluates them with **emotion wheels (EWs)**. Existing approaches are trained with a *token-level* loss, which is **misaligned** with the metrics actually used in OV-MER: *joyful* and *awful* share high token-level similarity yet are semantically far apart. Worse, EW-based metrics are **non-differentiable** — they involve stochastic sampling, discrete word-to-word clustering, and set-cardinality operations — so they cannot be optimized by gradient backpropagation at all. **AffectGPT-RL** addresses this with reinforcement learning, which can optimize non-differentiable objectives. EW-based metrics are used as **reward functions**, so that the training objective is directly aligned with the evaluation metric. Extensive experiments reveal the role of reinforcement learning in this task.

<img src="AffectGPT-RL/image/overall.png" width="800" />

## 📑 Citation

If you find this project useful for your research and applications, please cite using this BibTeX:
```bibtex
# OV-MER task, OV-MERD dataset
@inproceedings{lian2025ov,
  title={OV-MER: Towards Open-Vocabulary Multimodal Emotion Recognition},
  author={Lian, Zheng and Sun, Haiyang and Sun, Licai and Chen, Haoyu and Chen, Lan and Gu, Hao and Wen, Zhuofan and Chen, Shun and Siyuan, Zhang and Yao, Hailiang and others},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning},
  year={2025}
}

# MER-Caption dataset, MER-Caption+ dataset, AffectGPT Framework
@inproceedings{lian2025affectgpt,
  title={AffectGPT: A New Dataset, Model, and Benchmark for Emotion Understanding with Multimodal Large Language Models},
  author={Lian, Zheng and Chen, Haoyu and Chen, Lan and Sun, Haiyang and Sun, Licai and Ren, Yong and Cheng, Zebang and Liu, Bin and Liu, Rui and Peng, Xiaojiang and others},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning},
  year={2025}
}

# EmoPrefer
@inproceedings{lian2026emoprefer,
  title={EmoPrefer: Can Large Language Models Understand Human Emotion Preferences?},
  author={Lian, Zheng and Sun, Licai and Chen, Lan and Chen, Haoyu and Cheng, Zebang and Zhang, Fan and Jia, Ziyu and Ma, Ziyang and Ma, Fei and Peng, Xiaojiang and others},
  booktitle={Proceedings of the International Conference on Learning Representations, {ICLR}},
  year={2026}
}

# AffectGPT-RL
@inproceedings{lian2026affectgptrl,
  title={AffectGPT-RL: Revealing Roles of Reinforcement Learning in Open-Vocabulary Emotion Recognition},
  author={Lian, Zheng and Zhang, Fan and Chen, Lan and Zhang, Yazhou and Liu, Rui and Wu, Jinyang and Chen, Haoyu and Li, Xiaobai and Peng, Xiaojiang and He, Bin and Tao, Jianhua},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}

# EMER task
@article{lian2023explainable,
  title={Explainable Multimodal Emotion Recognition},
  author={Lian, Zheng and Sun, Haiyang and Sun, Licai and Gu, Hao and Wen, Zhuofan and Zhang, Siyuan and Chen, Shun and Xu, Mingyu and Xu, Ke and Chen, Kang and others},
  journal={arXiv preprint arXiv:2306.15401},
  year={2023}
}

# MER2023 Dataset
@inproceedings{lian2023mer,
  title={Mer 2023: Multi-label learning, modality robustness, and semi-supervised learning},
  author={Lian, Zheng and Sun, Haiyang and Sun, Licai and Chen, Kang and Xu, Mngyu and Wang, Kexin and Xu, Ke and He, Yu and Li, Ying and Zhao, Jinming and others},
  booktitle={Proceedings of the 31st ACM international conference on multimedia},
  pages={9610--9614},
  year={2023}
}

# MER2024 Dataset
@inproceedings{lian2024mer,
  title={Mer 2024: Semi-supervised learning, noise robustness, and open-vocabulary multimodal emotion recognition},
  author={Lian, Zheng and Sun, Haiyang and Sun, Licai and Wen, Zhuofan and Zhang, Siyuan and Chen, Shun and Gu, Hao and Zhao, Jinming and Ma, Ziyang and Chen, Xie and others},
  booktitle={Proceedings of the 2nd International Workshop on Multimodal and Responsible Affective Computing},
  pages={41--48},
  year={2024}
}

# MER2025 Dataset
@inproceedings{lian2025mer,
  title={Mer 2025: When affective computing meets large language models},
  author={Lian, Zheng and Liu, Rui and Xu, Kele and Liu, Bin and Liu, Xuefei and Zhang, Yazhou and Liu, Xin and Li, Yong and Cheng, Zebang and Zuo, Haolin and others},
  booktitle={Proceedings of the 33th ACM International Conference on Multimedia},
  year={2025}
}

# MER2026 Dataset
@article{lian2026mer,
  title={MER 2026: From Discriminative Emotion Recognition to Generative Emotion Understanding},
  author={Lian, Zheng and Peng, Xiaojiang and Xu, Kele and Jia, Ziyu and Che, Xinyi and Fang, Kuofei and Cheng, Zebang and Ma, Fei and Cui, Laizhong and Zhang, Yazhou and others},
  journal={arXiv preprint arXiv:2604.19417},
  year={2026}
}
```

## 🔒 License

This project is released under the Apache 2.0 license as found in the LICENSE file. The service is a research preview intended for **non-commercial use ONLY**. Please get in touch with us if you find any potential violations.
