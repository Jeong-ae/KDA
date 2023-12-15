# KDA : semi-supervised Knowledge Distillation with feature Augmentation
Implementation of "KDA : semi-supervised Knowledge Distillation with feature Augmentation"

This repository contains strong FeatMatch baseline implementation.
"[FeatMatch: Feature-Based Augmentation for Semi-Supervised Learning](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630460.pdf)[ECCV 2020]"]

## Datasets used
- Cifar10/100
- Mini-ImageNet

## Semi-Supervised Knowledge Distillation
- Response based distillation (KL Divergence loss)
- Consistency loss

## Teacher Models and Student Models
- CNN-13
- ResNet-18
- ViT_Tiny
- Wide-ResNet

## Installation
```
pip install -r requirements.txt

```

## Training
```
sh train.sh
```

### Running arguments

    -cf CONFIG: training configd on
    --ckpt CHECKPOINT: location of teacher model checkpoint
    --teacher TEACHER: activate teacher model
    --kld KL-Divergence: add kl divergence loss


## Results

Here are the quantitative results on different datasets. Numbers represent accuracy.

For CIFAR-100, mini-ImageNet, CIFAR-10, we follow the conventional evaluation method.
The model is evaluated directly on the test set, and the median of the last _K_ (_K_=10 in our case) testing accuracies is reported.

For our proposed KD and top-K setting, we observed that selecting CNN as teacher, ViT as student achieves best accuracy.

### CIFAR-100
\#label 4k | cnn13(S) | ResNet18(S) | ViT(S)
--- | --- | --- | ---
cnn13(T) 69.09 | 68.89 | 51.32 | **80.67**
ResNet(T) 51.19 | 68.75 | 51.95 | **81.62**
ViT(T) 76.03 | 51.50 | 55.18 | **78.6**

### CIFAR-10
\#label 250 | WRN(S) | ResNet18(S) | CNN13(S) | ViT(S)
--- | --- | --- | --- | ---
WRN(T) 91.72 | **92.20** | 73.08 | 84.80 | 86.00 

### CIFAR-10
\#label 4k  | WRN(S) | ResNet18(S) | CNN13(S) | ViT(S)
--- | --- | --- | --- | ---
ResNet18(T) 69.14 | 56.42 | 68.88 | 45.36 | 80.94

## Citation
    @inproceedings{kuo2020featmatch,
      title={Featmatch: Feature-based augmentation for semi-supervised learning},
      author={Kuo, Chia-Wen and Ma, Chih-Yao and Huang, Jia-Bin and Kira, Zsolt},
      booktitle={European Conference on Computer Vision},
      pages={479--495},
      year={2020},
      organization={Springer}
    }

[svhn]: http://ufldl.stanford.edu/housenumbers/
[cifar]: https://www.cs.toronto.edu/~kriz/cifar.html
[mini_imagenet]: https://github.com/twitter/meta-learning-lstm/tree/master/data/miniImagenet
[zca]: https://drive.google.com/drive/folders/14DDmdqMvBSp45ivk589-jpVq9Q4as0xA?usp=sharing

[Chia-Wen Kuo]: https://sites.google.com/view/chiawen-kuo/home
[Chih-Yao Ma]: https://chihyaoma.github.io/
[Jia-Bin Huang]: https://filebox.ece.vt.edu/~jbhuang/
[Zsolt Kira]: https://www.cc.gatech.edu/~zk15/
[arXiv]: https://arxiv.org/abs/2007.08505
[Project]: https://sites.google.com/view/chiawen-kuo/home/featmatch
