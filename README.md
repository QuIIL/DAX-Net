# DAX-Net

## About
![image](https://hackmd.io/_uploads/Skvo63ijT.png)
> Overview of DAX-Net. DAX-Net includes a feature extraction block, a feature fusion block, and a prediction block. The feature extraction block comprises two simultaneous CNN- and Transformer-based branches, generating two heterogeneous feature representations. During training, the feature fusion block fuses the two feature representations via summation and the prediction block uses them to separately conduct categorical classification and ordinal classification. DAX-Net is optimized using two loss functions that are tailored to the two classification tasks. At inference, the two feature representations are adaptively fused to focus more on the informative features.

## Datasets
There are five datasets for two organs (colon & prostate) used in this study:

1. Colon dataset:
    - [x] [CTrain, CValid & CTest-I](https://drive.google.com/file/d/1KsLvqNdwAnw_WunVyOqi-TIF77BTsn8K/view?usp=sharing)
    - [x] [CTest-II](https://drive.google.com/file/d/1taYhjlHydhe6TMn4f5J5Lz9SJ-b0IQeS/view) (Independent test set).

2. Prostate dataset:
    - [x] [PTrain, PValid & PTest-I](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OCYCMP)
    - [x] [PTest-II](https://gleason2019.grand-challenge.org/) (Independent test set)
    - [ ] PTest-III (comming soon).

![image](https://hackmd.io/_uploads/S1JGVaoiT.png)
> Dataset stastitics.

For convinience, please prepare the downloaded datasets as below directory tree:

```
datasets
|__KBSMC_colon_tma_cancer_grading_1024 # CTrain, CValid, CTest-I
|__KBSMC_test_2 # CTest-II
|__prostate_harvard # PTrain, PValid, PTest-I
|__prostate_miccai_2019_patches_690_80_step05_test # PTest-II
|__AGGC22_patch_512_c08 # PTest-III
```

## How to run
### Requirements

Install necessary libraries in requirements.txt:

```
pip install -r requirements.txt
```

### Training
For training DAX-Net, please run the below command
```
CUDA_VISIBLE_DEVICES=0 python3.8 train_val.py \
--dataset [DATASET NAME] \
--run_info=MULTI
```

`DATASET NAME` has two options: `colon_tma` and `prostate_uhu`. Set `colon_tma` for training, validating and saving checkpoints using on CTrain and CValid datasets. And set `prostate_uhu` for PTrain, PValid.

### Reproducion

For reproduce our results, please run below command:

```
CUDA_VISIBLE_DEVICES=0 python3.8 test_only.py --dataset=[DATASET NAME] \
--checkpoint=[CHECKPOINT PATH] \
--run_info=MULTI
```

`DATASET NAME` has five options: `colon_tma_test_1`, `colon_tma_test_2`, `prostate_uhu`, `prostate_ubc` and `aggc2022`, for evaluating DAX-Net on CTest-I, CTest-II, PTest-I, PTest-II and PTest-III, respectively.

For `[CHECKPOINT PATH]`, refers to the checkpoint links for corresponding organs: [Colon]() and [Prostate]().

The results should be the same as those reported in Tables 2 and 3 in our paper.

## Authors

Doanh C. Bui, Boram Song, Kyungeun Kim and Jin Tae Kwak

## Citation

If any part of this code is used, please give appropriate citation to our paper. <br />

BibTex entry: <br />
```
@article{bui2024daxnet,
  title={DAX-Net: a dual-branch dual-task adaptive cross-weight feature fusion network for robust multi-class cancer classification in pathology images},
  author={Bui, Doanh C. and Kim, Kyungeun and Song, Boram and Kwak, Jin Tae},
  journal={},
  pages={},
  year={},
  publisher={}
}
```
