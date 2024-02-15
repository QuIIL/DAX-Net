import os
import csv
import glob
import random
from collections import Counter
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data
from torchvision import transforms
from imgaug import augmenters as iaa

####

def print_data_count(label_list):
    count = []
    for i in range(5):
        count.append(label_list.count(i))
    count.append(len(label_list))
    return count

class DatasetSerialForVisualization(data.Dataset):

    def __init__(self, pair_list, shape_augs=None, input_augs=None, has_aux=False, test_aux=False):
        self.test_aux = test_aux
        self.pair_list = pair_list
        self.shape_augs = shape_augs
        self.input_augs = input_augs

    def __getitem__(self, idx):
        pair = self.pair_list[idx]
        # print(pair)
        input_img = cv2.imread(pair[0])
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        img_label = pair[1]
        # print(input_img.shape)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0., 0., 0.],
                                 std=[1., 1., 1.])
        ])

        if not self.test_aux:

            # shape must be deterministic so it can be reused
            if self.shape_augs is not None:
                shape_augs = self.shape_augs.to_deterministic()
                input_img = shape_augs.augment_image(input_img)

            # additional augmenattion just for the input
            if self.input_augs is not None:
                input_img = self.input_augs.augment_image(input_img)

            input_img = np.array(input_img).copy()
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0., 0., 0.],
                                     std=[1., 1., 1.])
            ])

            out_img = np.array(transform(input_img)).transpose(1, 2, 0)
        else:
            out_img = []
            for idx in range(5):
                input_img_ = input_img.copy()
                if self.shape_augs is not None:
                    shape_augs = self.shape_augs.to_deterministic()
                    input_img_ = shape_augs.augment_image(input_img_)
                input_img_ = iaa.Sequential(self.input_augs[idx]).augment_image(input_img_)
                input_img_ = np.array(input_img_).copy()
                input_img_ = np.array(transform(input_img_)).transpose(1, 2, 0)
                out_img.append(input_img_)
        return np.array(out_img), img_label, pair[0]

    def __len__(self):
        return len(self.pair_list)
    
class DatasetSerial(data.Dataset):

    def __init__(self, pair_list, shape_augs=None, input_augs=None, has_aux=False, test_aux=False):
        self.test_aux = test_aux
        self.pair_list = pair_list
        self.shape_augs = shape_augs
        self.input_augs = input_augs

    def __getitem__(self, idx):
        pair = self.pair_list[idx]
        # print(pair)
        input_img = cv2.imread(pair[0])
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        img_label = pair[1]
        # print(input_img.shape)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0., 0., 0.],
                                 std=[1., 1., 1.])
        ])

        if not self.test_aux:

            # shape must be deterministic so it can be reused
            if self.shape_augs is not None:
                shape_augs = self.shape_augs.to_deterministic()
                input_img = shape_augs.augment_image(input_img)

            # additional augmenattion just for the input
            if self.input_augs is not None:
                input_img = self.input_augs.augment_image(input_img)

            input_img = np.array(input_img).copy()
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0., 0., 0.],
                                     std=[1., 1., 1.])
            ])

            out_img = np.array(transform(input_img)).transpose(1, 2, 0)
        else:
            out_img = []
            for idx in range(5):
                input_img_ = input_img.copy()
                if self.shape_augs is not None:
                    shape_augs = self.shape_augs.to_deterministic()
                    input_img_ = shape_augs.augment_image(input_img_)
                input_img_ = iaa.Sequential(self.input_augs[idx]).augment_image(input_img_)
                input_img_ = np.array(input_img_).copy()
                input_img_ = np.array(transform(input_img_)).transpose(1, 2, 0)
                out_img.append(input_img_)
        return np.array(out_img), img_label

    def __len__(self):
        return len(self.pair_list)
    
class DatasetSerialWholeCNNViT(data.Dataset):

    def __init__(self, pair_list, shape_augs=None, shape_augs1=None, input_augs=None, has_aux=False, test_aux=False):
        self.test_aux = test_aux
        self.pair_list = pair_list
        self.shape_augs = shape_augs
        self.shape_augs1 = shape_augs1
        self.input_augs = input_augs

    def __getitem__(self, idx):
        pair = self.pair_list[idx]
        # print(pair)
        input_img = cv2.imread(pair[0])
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        img_label = pair[1]
        # print(input_img.shape)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0., 0., 0.],
                                 std=[1., 1., 1.])
        ])

        if not self.test_aux:

            # shape must be deterministic so it can be reused
            if self.shape_augs is not None:
                shape_augs = self.shape_augs.to_deterministic()
                shape_augs1 = self.shape_augs1.to_deterministic()
                input_img1 = input_img.copy()
                input_img = shape_augs.augment_image(input_img)
                input_img1 = shape_augs1.augment_image(input_img1)

            # additional augmenattion just for the input
            if self.input_augs is not None:
                input_img = self.input_augs.augment_image(input_img)
                input_img1 = self.input_augs.augment_image(input_img1)

            input_img = np.array(input_img).copy()
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0., 0., 0.],
                                     std=[1., 1., 1.])
            ])

            out_img = np.array(transform(input_img)).transpose(1, 2, 0)

            input_img1 = np.array(input_img1).copy()
            out_img1 = np.array(transform(input_img1)).transpose(1, 2, 0)

        else:
            out_img = []
            for idx in range(5):
                input_img_ = input_img.copy()
                if self.shape_augs is not None:
                    shape_augs = self.shape_augs.to_deterministic()
                    input_img_ = shape_augs.augment_image(input_img_)
                input_img_ = iaa.Sequential(self.input_augs[idx]).augment_image(input_img_)
                input_img_ = np.array(input_img_).copy()
                input_img_ = np.array(transform(input_img_)).transpose(1, 2, 0)
                out_img.append(input_img_)
        
        return np.array(out_img), np.array(out_img1), img_label

    def __len__(self):
        return len(self.pair_list)


class DatasetSerialWSI(data.Dataset):
    def __init__(self, path_list):
        self.path_list = path_list

    def __getitem__(self, idx):
        input_img = cv2.imread(self.path_list[idx])
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = np.array(input_img).copy()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0., 0., 0.],
                                 std=[1., 1., 1.])
        ])
        input_img = np.array(transform(input_img)).transpose(1, 2, 0)
        location = self.path_list[idx].split('/')[-1].split('.')[0].split('_')
        return input_img, location

    def __len__(self):
        return len(self.path_list)

def prepare_colon_tma_data(
        data_root_dir=None):
    
    def load_data_info(pathname):
        file_list = glob.glob(pathname)
        label_list = [int(file_path.split('_')[-1].split('.')[0]) for file_path in file_list]
        print(Counter(label_list))
        return list(zip(file_list, label_list))

    set_tma01 = load_data_info('%s/tma_01/*.jpg' % data_root_dir)
    set_tma02 = load_data_info('%s/tma_02/*.jpg' % data_root_dir)
    set_tma03 = load_data_info('%s/tma_03/*.jpg' % data_root_dir)
    set_tma04 = load_data_info('%s/tma_04/*.jpg' % data_root_dir)
    set_tma05 = load_data_info('%s/tma_05/*.jpg' % data_root_dir)
    set_tma06 = load_data_info('%s/tma_06/*.jpg' % data_root_dir)
    set_wsi01 = load_data_info('%s/wsi_01/*.jpg' % data_root_dir)  # benign exclusively
    set_wsi02 = load_data_info('%s/wsi_02/*.jpg' % data_root_dir)  # benign exclusively
    set_wsi03 = load_data_info('%s/wsi_03/*.jpg' % data_root_dir)  # benign exclusively

    train_set = set_tma01 + set_tma02 + set_tma03 + set_tma05 + set_wsi01
    valid_set = set_tma06 + set_wsi03
    test_set = set_tma04 + set_wsi02

    # print dataset detail
    train_label = [train_set[i][1] for i in range(len(train_set))]
    val_label = [valid_set[i][1] for i in range(len(valid_set))]
    test_label = [test_set[i][1] for i in range(len(test_set))]

    print(print_data_count(train_label))
    print(print_data_count(val_label))
    print(print_data_count(test_label))
    
    return train_set, valid_set, test_set

def prepare_colon_tma_data_test_1(
        data_root_dir=None):
    
    data_root_dir = './datasets/KBSMC_colon_tma_cancer_grading_512/'

    def load_data_info(pathname):
        file_list = glob.glob(pathname)
        label_list = [int(file_path.split('_')[-1].split('.')[0]) for file_path in file_list]
        print(Counter(label_list))
        return list(zip(file_list, label_list))

    set_tma04 = load_data_info('%s/tma_04/*.jpg' % data_root_dir)
    set_wsi02 = load_data_info('%s/wsi_02/*.jpg' % data_root_dir)  # benign exclusively

    train_set = None
    valid_set = None
    test_set = set_tma04 + set_wsi02
    # print dataset detail
    test_label = [test_set[i][1] for i in range(len(test_set))]
    print(print_data_count(test_label))
    
    return train_set, valid_set, test_set

def prepare_colon_tma_data_test_2(
        data_root_dir=None):
    
    data_root_dir = './datasets/KBSMC_test_2/'

    def load_data_info(pathname):
        print(pathname)
        file_list = glob.glob(pathname)
        label_list = [int(file_path.split('_')[-1].split('.')[0]) for file_path in file_list]
        print(Counter(label_list))
        return list(zip(file_list, label_list))

    test_set_2 = []
    for folder in glob.glob('%s/*' % data_root_dir):
        set_info = load_data_info('%s/**/*.png' % folder)
        test_set_2.append(set_info)

    train_set = None
    valid_set = None
    test_set = test_set_2[0]
    for i in range(1, len(test_set_2)): test_set += test_set_2[i]
    # test_set = [i for i in test_set if os.path.basename(i[0]).split('.')[0].split('_')[-1] == '4']
    test_label = [test_set[i][1] for i in range(len(test_set))]
    print(print_data_count(test_label))
    
    return train_set, valid_set, test_set


def prepare_colon_wsi_patch(data_visual=False):
    def load_data_info_from_list(data_dir, path_list):
        file_list = []
        for WSI_name in path_list:
            pathname = glob.glob(f'{data_dir}/{WSI_name}/*/*.png')
            file_list.extend(pathname)
            label_list = [int(file_path.split('_')[-1].split('.')[0]) - 1 for file_path in file_list]
        print(Counter(label_list))
        list_out = list(zip(file_list, label_list))
        return list_out

    data_root_dir = '/media/data1/trinh/data/workspace_data/colon_wsi/patches_colon_edit_MD/colon_45WSIs_1144_08_step05_05'
    data_visual = '/media/data1/trinh/data/workspace_data/colon_wsi/patches_colon_edit_MD/colon_45WSIs_1144_01_step05_visualize/patch_512/'

    df_test = [] #Note: Will be update later

    if data_visual:
        test_set = load_data_info_from_list(data_visual, df_test)
    else:
        test_set = load_data_info_from_list(data_root_dir, df_test)
    return test_set


def prepare_prostate_uhu_data():
    def load_data_info(pathname, parse_label=True, label_value=0, cancer_test=False):
        file_list = glob.glob(pathname)

        if cancer_test:
            file_list_bn = glob.glob(pathname.replace('*.jpg', '*0.jpg'))
            file_list = [elem for elem in file_list if elem not in file_list_bn]
            label_list = [int(file_path.split('_')[-1].split('.')[0])-1 for file_path in file_list]
        else:
            if parse_label:
                label_list = [int(file_path.split('_')[-1].split('.')[0]) for file_path in file_list]
            else:
                label_list = [label_value for file_path in file_list]
        print(Counter(label_list))
        return list(zip(file_list, label_list))

    data_root_dir = './datasets/prostate_harvard/'
    data_root_dir_train = f'{data_root_dir}/patches_train_750_v0/'
    data_root_dir_validation = f'{data_root_dir}/patches_validation_750_v0/'
    data_root_dir_test = f'{data_root_dir}/patches_test_750_v0/'

    train_set_111 = load_data_info('%s/ZT111*/*.jpg' % data_root_dir_train)
    train_set_199 = load_data_info('%s/ZT199*/*.jpg' % data_root_dir_train)
    train_set_204 = load_data_info('%s/ZT204*/*.jpg' % data_root_dir_train)
    valid_set = load_data_info('%s/ZT76*/*.jpg' % data_root_dir_validation)
    test_set = load_data_info('%s/patho_1/*/*.jpg' % data_root_dir_test)

    train_set = train_set_111 + train_set_199 + train_set_204
    return train_set, valid_set, test_set


def prepare_prostate_ubc_data(fold_idx=0):
    def load_data_info(pathname, parse_label=True, label_value=0):
        file_list = glob.glob(pathname)
        cancer_test = False
        if cancer_test:
            file_list_bn = glob.glob(pathname.replace('*.jpg', '*0.jpg'))
            file_list = [elem for elem in file_list if elem not in file_list_bn]
            label_list = [int(file_path.split('_')[-1].split('.')[0]) for file_path in file_list]
            label_dict = {2: 0, 3: 1, 4: 2}
            label_list = [label_dict[k] for k in label_list]
        else:
            if parse_label:
                label_list = [int(file_path.split('_')[-1].split('.')[0]) for file_path in file_list]
            else:
                label_list = [label_value for file_path in file_list]
            label_dict = {0: 0, 2: 1, 3: 2, 4: 3}
            label_list = [label_dict[k] for k in label_list]
        print(Counter(label_list))
        return list(zip(file_list, label_list))

    assert fold_idx < 3, "Currently only support 5 fold, each fold is 1 TMA"

    data_root_dir = './datasets/'
    data_root_dir_train_ubc = f'{data_root_dir}/prostate_miccai_2019_patches_690_80_step05_test/'
    test_set_ubc = load_data_info('%s/*/*.jpg' % data_root_dir_train_ubc)
    return test_set_ubc

def prepare_aggc2022_data(fold_idx=0, nr_classes=4):
    def load_data_info(pathname, parse_label=True, label_value=0):
        file_list = glob.glob(pathname)
        file_list = [file_path for file_path in file_list if int(file_path.split('/')[-1].split('.')[0].split('_')[-1]) > 1]
        if parse_label:
            label_list = [int(file_path.split('/')[-1].split('.')[0].split('_')[-1]) - 2 for file_path in file_list if int(file_path.split('/')[-1].split('.')[0].split('_')[-1]) > 1]
        else:
            label_list = [label_value for file_path in file_list]
        print(Counter(label_list))
        return list(zip(file_list, label_list))

    # 1000 ~ 6158
    data_root_dir = './datasets/AGGC22_patch_512_c08/'
    train_set_1 = load_data_info('%s/Subset1_Train_image/**/**' % data_root_dir)
    train_set_2 = load_data_info('%s/Subset2_Train_image/**/**' % data_root_dir)
    train_set_3 = load_data_info('%s/Subset3_Train_image/**/**/**' % data_root_dir)
    return train_set_1 + train_set_2 + train_set_3

import pandas as pd
def load_a_dataset(csv_name, gt_list, data_root_dir, data_root_dir_2, nr_claases, down_sample=True):
    #csv_path = f'/data2/trinh/data/patch_data/KBSMC/gastric_cancer/gastric_wsi/{csv_name}'
    def load_data_info_from_list(path_list, gt_list, data_root_dir, nr_claases):
        file_list = []
        for tma_name in path_list:
            # print(tma_name)
            pathname = glob.glob(f'{data_root_dir}/{tma_name}/*.jpg')
            file_list.extend(pathname)
        label_list = [int(file_path.split('_')[-1].split('.')[0]) for file_path in file_list]
        label_list = [gt_list[i] for i in label_list]
        list_out = list(zip(file_list, label_list))

        list_out = [list_out[i] for i in range(len(list_out)) if list_out[i][1] < nr_claases]
        return list_out
    
    csv_path = f'../datasets/gastric/{csv_name}'
    df = pd.read_csv(csv_path).iloc[:, :3]
    train_list = list(df.query('Task == "train"')['WSI'])
    valid_list = list(df.query('Task == "val"')['WSI'])
    test_list = list(df.query('Task == "test"')['WSI'])
    train_set = load_data_info_from_list(train_list, gt_list, data_root_dir, nr_claases)

    if down_sample:
        train_normal = [train_set[i] for i in range(len(train_set)) if train_set[i][1] == 0]
        train_tumor = [train_set[i] for i in range(len(train_set)) if train_set[i][1] != 0]

        random.shuffle(train_normal)
        train_normal = train_normal[: len(train_tumor) // 3]
        train_set = train_normal + train_tumor

    valid_set = load_data_info_from_list(valid_list, gt_list, data_root_dir_2, nr_claases)
    test_set = load_data_info_from_list(test_list, gt_list, data_root_dir_2, nr_claases)
    return train_set, valid_set, test_set

def prepare_gastric_data(patch_size=1024, nr_classes=8):
    """ 8 classes in total only choose 5"""

    if nr_classes == 3:
        gt_train_local = {1: 4,  # "BN", #0
                          2: 4,  # "BN", #0
                          3: 0,  # "TW", #2
                          4: 1,  # "TM", #3
                          5: 2,  # "TP", #4
                          6: 4,  # "TLS", #1
                          7: 4,  # "papillary", #5
                          8: 4,  # "Mucinous", #6
                          9: 4,  # "signet", #7
                          10: 4,  # "poorly", #7
                          11: 4  # "LVI", #ignore
                          }
    elif nr_classes == 4:
        gt_train_local = {1: 0,  # "BN", #0
                          2: 0,  # "BN", #0
                          3: 1,  # "TW", #2
                          4: 2,  # "TM", #3
                          5: 3,  # "TP", #4
                          6: 4,  # "TLS", #1
                          7: 4,  # "papillary", #5
                          8: 4,  # "Mucinous", #6
                          9: 4,  # "signet", #7
                          10: 4,  # "poorly", #7
                          11: 4  # "LVI", #ignore
                          }
    elif nr_classes == 5:
        gt_train_local = {1: 0,  # "BN", #0
                          2: 0,  # "BN", #0
                          3: 1,  # "TW", #2
                          4: 2,  # "TM", #3
                          5: 3,  # "TP", #4
                          6: 8,  # "TLS", #1
                          7: 8,  # "papillary", #5
                          8: 8,  # "Mucinous", #6
                          9: 4,  # "signet", #7
                          10: 4,  # "poorly", #7
                          11: 8  # "LVI", #ignore
                          }
    elif nr_classes == 6:
        gt_train_local = {1: 0,  # "BN", #0
                          2: 0,  # "BN", #0
                          3: 2,  # "TW", #2
                          4: 2,  # "TM", #3
                          5: 2,  # "TP", #4
                          6: 1,  # "TLS", #1
                          7: 3,  # "papillary", #5
                          8: 4,  # "Mucinous", #6
                          9: 5,  # "signet", #7
                          10: 5,  # "poorly", #7
                          11: 6  # "LVI", #ignore
                          }
    elif nr_classes == 8:
        gt_train_local = {1: 0,  # "BN", #0
                          2: 0,  # "BN", #0
                          3: 2,  # "TW", #2
                          4: 3,  # "TM", #3
                          5: 4,  # "TP", #4
                          6: 1,  # "TLS", #1
                          7: 5,  # "papillary", #5
                          8: 6,  # "Mucinous", #6
                          9: 7,  # "signet", #7
                          10: 7,  # "poorly", #7
                          11: 8  # "LVI", #ignore
                          }
    elif nr_classes == 10:
        gt_train_local = {1: 0,  # "BN", #0
                          2: 0,  # "BN", #0
                          3: 1,  # "TW", #2
                          4: 2,  # "TM", #3
                          5: 3,  # "TP", #4
                          6: 4,  # "TLS", #1
                          7: 5,  # "papillary", #5
                          8: 6,  # "Mucinous", #6
                          9: 7,  # "signet", #7
                          10: 8,  # "poorly", #7
                          11: 9  # "LVI", #ignore
                          }
    else:
        gt_train_local = {1: 0,  # "BN", #0
                          2: 0,  # "BN", #0
                          3: 1,  # "TW", #2
                          4: 2,  # "TM", #3
                          5: 3,  # "TP", #4
                          6: 8,  # "TLS", #1
                          7: 8,  # "papillary", #5
                          8: 5,  # "Mucinous", #6
                          9: 4,  # "signet", #7
                          10: 4,  # "poorly", #7
                          11: 8  # "LVI", #ignore
                          }

    csv_her02 = 'gastric_cancer_wsi_1024_80_her01_split.csv'
    csv_addition = 'gastric_wsi_addition_PS1024_ano08_split.csv'

    data_her_root_dir = f'../datasets/gastric/gastric_wsi/gastric_cancer_wsi_1024_80_her01_step05_bright230_resize05'
    data_her_root_dir_2 = f'../datasets/gastric/gastric_wsi/gastric_cancer_wsi_1024_80_her01_step10_bright230_resize05'
    data_add_root_dir = f'../datasets/gastric/gastric_wsi_addition/gastric_wsi_addition_PS1024_ano08_step05_bright230_resize05'
    data_add_root_dir_2 = f'../datasets/gastric/gastric_wsi_addition/gastric_wsi_addition_PS1024_ano08_step10_bright230_resize05'

    # data_her_root_dir = f'/data2/lju/gastric/gastric_wsi/gastric_cancer_wsi_1024_80_her01_step10_bright230_resize05'
    # data_her_root_dir_2 = f'/data2/lju/gastric/gastric_wsi/gastric_cancer_wsi_1024_80_her01_step10_bright230_resize05'
    # data_add_root_dir = f'/data2/lju/gastric/gastric_wsi_addition/gastric_wsi_addition_PS1024_ano08_step10_bright230_resize05'
    # data_add_root_dir_2 = f'/data2/lju/gastric/gastric_wsi_addition/gastric_wsi_addition_PS1024_ano08_step10_bright230_resize05'
    train_set, valid_set, test_set = load_a_dataset(csv_her02, gt_train_local,data_her_root_dir, data_her_root_dir_2, nr_classes)
    train_set_add, valid_set_add, test_set_add = load_a_dataset(csv_addition, gt_train_local, data_add_root_dir, data_add_root_dir_2, nr_classes, down_sample=False)
    train_set += train_set_add
    valid_set += valid_set_add
    test_set += test_set_add
    #train_label, valid_label, test_label = print_number_of_sample(train_set + valid_set + test_set, valid_set, test_set)
    #return train_set, valid_set, test_set, train_label, valid_label, test_label
    return train_set, valid_set, test_set

def visualize(ds, batch_size, nr_steps=100):
    data_idx = 0
    cmap = plt.get_cmap('jet')
    for i in range(0, nr_steps):
        if data_idx >= len(ds):
            data_idx = 0
        for j in range(1, batch_size + 1):
            sample = ds[data_idx + j]
            if len(sample) == 2:
                img = sample[0]
            else:
                img = sample[0]
                # TODO: case with multiple channels
                aux = np.squeeze(sample[-1])
                aux = cmap(aux)[..., :3]  # gray to RGB heatmap
                aux = (aux * 255).astype('unint8')
                img = np.concatenate([img, aux], axis=0)
                img = cv2.resize(img, (40, 80), interpolation=cv2.INTER_CUBIC)
            plt.subplot(1, batch_size, j)
            plt.title(str(sample[1]))
            plt.imshow(img)
        plt.show()
        data_idx += batch_size