<!-- omit in toc -->
# EAT
[![Platform](https://img.shields.io/badge/Platform-linux-lightgrey)](https://github.com/cwx-worst-one/EAT)
[![Python](https://img.shields.io/badge/Python-3.8+-orange)](https://github.com/cwx-worst-one/EAT)
[![Pytorch](https://img.shields.io/badge/PyTorch-1.13+-brightgreen)](https://github.com/cwx-worst-one/EAT)
[![fairseq](https://img.shields.io/badge/fairseq-0.12.2-blue)](https://github.com/cwx-worst-one/EAT)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](https://github.com/cwx-worst-one/EAT)

**Guides**
- [Model Checkpoints](#model-checkpoints)
- [Feature Extraction](#feature-extraction)
- [Pre-Training](#pre-training)
- [Fine-Tuning](#fine-tuning)

<!-- omit in toc -->
## Introduction 
EAT is an audio SSL model with high effectiveness and efficiency during self-supervised pre-training. You can find details in the paper [EAT: Self-Supervised Pre-Training with Efficient Audio Transformer](https://arxiv.org/abs/2401.03497). 

## Model Checkpoints
You could download the EAT checkpoints by Google Drive. 
- AS-2M [Pre-trained](https://drive.google.com/file/d/1PFUcDbvtZfxFcyaRv3RHsjy_QhvC1QBp/view?usp=sharing)
- AS-2M Pre-trained+[Fine-tuned]() (AS-2M)
- AS-2M Pre-trained+[Fine-tuned]() (AS-20K)

## Feature Extraction
We provide the script for extracting audio features from the last layer of EAT encoder. The features are stored in `.npy` format and the sample rate of the extracted features is ~50Hz. EAT could provide frame-level features and utterance-level features (denoted by the CLS token).  
To extract latent representations from audio clips, you could use our pre-trained [checkpoint](https://drive.google.com/file/d/1PFUcDbvtZfxFcyaRv3RHsjy_QhvC1QBp/view?usp=sharing) or your owns, then please run the script `feature_extract.sh` by:
```bash
bash feature_extract.sh 
``` 

## Pre-Training 
TODO

## Fine-Tuning
TODO

<!-- omit in toc -->
## Performance
Pre-training on AS-2M, EAT gains state-of-the-art (SOTA) performance on several audio and speech classification datasets including AS-20K, AS-2M, ESC-50 and SPC-2.  
![](src/performance.png)

<!-- omit in toc -->
## Efficiency
EAT achieves a total pre-training time reduction of ~15x compared to BEATs and ~10x relative to Audio-MAE. It costs only 10 epochs during EAT's pre-training on AS-2M. 
![Alt text](src/efficiency.png)  





<!-- omit in toc -->
## TODO 
- [x] release the feature extraction codes
- [ ] release the model checkpoints for pre-training and fine-tuning
- [ ] release the main pre-trained codes
- [ ] release the fine-tuned codes
- [ ] release the inferrence codes 


<!-- omit in toc -->
## Citation
If you find our EAT code and paper useful, please cite the following paper:
```
@article{chen2024eat,
  title={EAT: Self-Supervised Pre-Training with Efficient Audio Transformer},
  author={Chen, Wenxi and Liang, Yuzhe and Ma, Ziyang and Zheng, Zhisheng and Chen, Xie},
  journal={arXiv preprint arXiv:2401.03497},
  year={2024}
}
```
