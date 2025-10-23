# CLOVER 
## Confusion-Driven Self-Supervised Progressively Weighted Ensemble Learning for Non-Exemplar Class Incremental Learning

| **[1 Introduction](#introduction)** 
| **[2 Requirements](#requirements)**
| **[3 Usage](#usage)**
| **[4 Citation](#citation)**
| **[5 Acknowledgments](#acknowledgments)** |

<a id="introduction"></a>
## Introduction

Official code for NeurIPS 2025 paper "[Confusion-Driven Self-Supervised Progressively Weighted Ensemble Learning for Non-Exemplar Class Incremental Learning]()".

> Non-exemplar class incremental learning (NECIL) aims to continuously assimilate new knowledge while retaining previously acquired knowledge in scenarios where prior examples are unavailable. A prevalent strategy within NECIL mitigates knowledge forgetting by freezing the feature extractor after training on the initial task. However, this freezing mechanism does not provide explicit training to differentiate between new and old classes, resulting in overlapping feature representations. To address this challenge, we propose a Confusion-driven seLf-supervised prOgressiVely weighted Ensemble leaRning (CLOVER) framework for NECIL. Firstly, we introduce a confusion-driven self-supervised learning approach that enhances representation extraction by guiding the model to distinguish between highly confusable classes, thereby reducing class representation overlap. Secondly, we develop a progressively weighted ensemble learning method that gradually adjusts weights to integrate diverse knowledge more effectively, further minimizing representation overlap. Finally, extensive experiments demonstrate that our proposed method achieves state-of-the-art results on the CIFAR100, TinyImageNet, and ImageNet-Subset NECIL benchmarks.

[//]: # (<div align=center><img src="CLOVER.png", width="90%"></div>)
![image](CLOVER.png?raw=true "inference")

<a id="requirements"></a>
## Requirements

Clone this repository:

```bash
git clone https://github.com/MLMIP/CLOVER.git
cd CLOVER/
```

Install the dependencies:

```bash
conda create -n CLOVER python=3.10
conda activate CLOVER
pip install -r requirements.txt
```

<a id="usage"></a>

## Usage

### Dataset Storage Format

Make sure your dataset is placed in the "./dataset" path and the directory structure is:

```
dataset/
├── CIFAR100/
│   └─── cifar-100-python/
│       ├── file.txt~
│       ├── meta
│       ├── test
│       └── train
├── tiny-imagenet-200/
│   ├── test/
│   ├── train/
│   ├── val/
│   ├── wnids.txt
│   └── words.txt.txt
└── seed_1993_subset_100_imagenet /
    ├── data/
    │   ├── samples
    │   ├── train
    │   └── test
    ├── ImageNet-Subset.tar
    ├── test.txt
    └── train.txt
```

### Running CLOVER

If you prepare your custom data following the above storage format, you can start training by executing the following command in the terminal.

Run CLOVER on CIFAR100 50 base classes, 5 tasks, 10 classes each:
```bash
bash script/cifar100/50+5x10.sh
```

Run CLOVER on CIFAR100 50 base classes, 10 tasks, 5 classes each:
```bash
bash script/cifar100/50+10x5.sh
```

Run CLOVER on CIFAR100 40 base classes, 20 tasks, 3 classes each:
```bash
bash script/cifar100/40+20x3.sh
```

<a id="citation"></a>

## Citation

If you find our work is useful for your research, please consider citing:

```
@inproceedings{
    anonymous2025confusiondriven,
    title={Confusion-Driven Self-Supervised Progressively Weighted Ensemble Learning for Non-Exemplar Class Incremental Learning},
    author={Anonymous},
    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
    year={2025},
    url={https://openreview.net/forum?id=yflq8Bhjrw}
}
```

<a id="acknowledgments"></a>

## Acknowledgments

Our code is inspired by [SEED](https://github.com/grypesc/SEED), [PASS](https://github.com/Impression2805/CVPR21_PASS) and [PGLS](https://github.com/MLMIP/PGLS). 
