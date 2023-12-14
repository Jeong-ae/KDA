# semi-supervised Knowledge Distillation with feature Augmentation : KDA
Implementation of "semi-supervised Knowledge Distillation with feature Augmentation : KDA"

This repository contains strong FeatMatch baseline implementation.
"[FeatMatch: Feature-Based Augmentation for Semi-Supervised Learning](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630460.pdf)[ECCV 2020]"]

## Datasets used
- Cifar10/100
- Mini-ImageNet

## Supervised Knowledge Distillation
- response based distillation (KL Divergence loss)
- Consistency loss

## Teacher Models and Student Models
- CNN-13
- ResNet-18
- ViT_Tiny
- Wide-ResNet

## Installation
```
pipinstall -r requirements.txt

```

## Training
```
sh train.sh
```

### Running arguments

    -cf CONFIG: training configd on
    --teacher TEACHER: activate teacher model
    --kld KL-Divergence: add kl divergence loss


## Results

Here are the quantitative results on different datasets, with different number of labels. Numbers represent error rate in three runs (lower the better).

For CIFAR-100, mini-ImageNet, CIFAR-10, and SVHN, we follow the conventional evaluation method.
The model is evaluated directly on the test set, and the median of the last _K_ (_K_=10 in our case) testing accuracies is reported.

For our proposed DomainNet setting, we reserve 1% of validation data, which is much fewer than the 5% of labeled data.
The model is evaluated on the validation data, and the model with the best validation accuracy is selected.
Finally, we report the test accuracy of the selected model.

### CIFAR-100
\#labels | 4k | 10k
--- | --- | --- 
paper | 31.06 ± 0.41 |  26.83 ± 0.04 
repo | 30.79 ± 0.35 | 26.88 ± 0.13

### mini-ImageNet
\#labels | 4k | 10k
--- | --- | --- 
paper | 39.05 ± 0.06 | 34.79 ± 0.22 
repo | 38.94 ± 0.19 | 34.84 ± 0.19

### DomainNet
_r<sub>u<sub>_ | 0% | 25% | 50% | 75%
--- | --- | --- | --- | ---
paper | 40.66 ± 0.60 | 46.11 ± 1.15 | 54.01 ± 0.66 | 58.30 ± 0.93 
repo | 40.47 ± 0.23 | 43.40 ± 0.25 | 52.49 ± 1.06 | 56.20 ± 1.25

### SVHN
\#labels | 250 | 1k | 4k
--- | --- | --- | ---
paper | 3.34 ± 0.19 | 3.10 ± 0.06 | 2.62 ± 0.08
repo | 3.62 ± 0.12 | 3.02 ± 0.04 | 2.61 ± 0.02

### CIFAR-10
\#labels | 250 | 1k | 4k
--- | --- | --- | ---
paper | 7.50 ± 0.64 | 5.76 ± 0.07 |  4.91 ± 0.18
repo | 7.38 ± 0.94 | 6.04 ± 0.24 | 5.19 ± 0.05

## Acknowledgement
This work was funded by DARPA’s Learning with Less Labels (LwLL) program under agreement HR0011-18-S-0044 and DARPAs Lifelong Learning Machines (L2M) program under Cooperative Agreement HR0011-18-2-0019.

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
