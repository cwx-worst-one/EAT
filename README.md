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
- [Data Preparation](#data-preparation)
- [Pre-Training](#pre-training)
- [Fine-Tuning](#fine-tuning)
- [Inference](#inference)

<!-- omit in toc -->
## Introduction 
EAT is an audio SSL model with high effectiveness and efficiency during self-supervised pre-training. You can find details in the paper [EAT: Self-Supervised Pre-Training with Efficient Audio Transformer](https://arxiv.org/abs/2401.03497). 

## Model Checkpoints
You could download the EAT checkpoints by Google Drive. 
- AS-2M [Pre-trained](https://drive.google.com/file/d/1PFUcDbvtZfxFcyaRv3RHsjy_QhvC1QBp/view?usp=sharing)
- AS-2M Pre-trained+[Fine-tuned](https://drive.google.com/file/d/1FNZ4LotG-VLRwrQJacsQyKQZnEah4i4w/view?usp=sharing) (AS-2M)
- AS-2M Pre-trained+[Fine-tuned](https://drive.google.com/file/d/1TyRG2xczQ6rvnkvEn0p2A-KbgSPKxcEI/view?usp=drive_link) (AS-20K)

## Feature Extraction
We provide the script for extracting audio features from the last layer of EAT encoder. The features are stored in `.npy` format and the sample rate of the extracted features is ~50Hz. EAT could provide frame-level features and utterance-level features (denoted by the CLS token).  
To extract latent representations from audio clips, you could use our pre-trained [checkpoint](https://drive.google.com/file/d/1PFUcDbvtZfxFcyaRv3RHsjy_QhvC1QBp/view?usp=sharing) or your owns, then please run the script `feature_extract.sh` by:
```bash
bash feature_extract.sh 
``` 

## Data Preparation
The main dataset in our experiment is [AudioSet](https://research.google.com/audioset/). Data manifest is available at [here](). We follow the file format in [wav2vec](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec) and [data2vec](https://github.com/facebookresearch/fairseq/tree/main/examples/data2vec), where `.tsv` format file is for index while `.lbl` and `.csv` format files are specific for classification task.  You could modify the files for your own database. 

## Pre-Training 
TODO

## Fine-Tuning
TODO

## Inference 
For inference on AudioSet audio clips with fine-tuned models, you could use our EAT checkpoints fine-tuning on [AS-2M](https://drive.google.com/file/d/1FNZ4LotG-VLRwrQJacsQyKQZnEah4i4w/view?usp=sharing) or [AS-20K](https://drive.google.com/file/d/1TyRG2xczQ6rvnkvEn0p2A-KbgSPKxcEI/view?usp=drive_link)
and run the script `inference.sh` by: 
```bash
bash inference.sh 
``` 
An example output is as follows:
```python
# top_k_prediction = 12
{'Percussion': 0.5227, 'Drum kit': 0.4365, 'Vibraphone': 0.4196, 'Drum': 0.3161, 
'Music': 0.3035, 'Snare drum': 0.2766, 'Glockenspiel': 0.2248, 'Marimba, xylophone': 0.223, 
'Cymbal': 0.213, 'Bass drum': 0.2069, 'Hi-hat': 0.1961, 'Mallet percussion': 0.1704}
```


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
- [x] release the model checkpoints for pre-training and fine-tuning
- [x] release the inferrence codes 
- [ ] release the pre-trained codes
- [ ] release the fine-tuned codes


<!-- omit in toc -->
## Citation
If you find our EAT codes and models useful, please cite the following paper:
```
@article{chen2024eat,
  title={EAT: Self-Supervised Pre-Training with Efficient Audio Transformer},
  author={Chen, Wenxi and Liang, Yuzhe and Ma, Ziyang and Zheng, Zhisheng and Chen, Xie},
  journal={arXiv preprint arXiv:2401.03497},
  year={2024}
}
```
