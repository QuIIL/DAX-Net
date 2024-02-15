import os
import argparse
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage
from tensorboardX import SummaryWriter
from imgaug import augmenters as iaa
from misc.train_ultils_all_iter import *
import importlib

import dataset as dataset
from config import Config
from loss.ceo_loss import count_pred
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, cohen_kappa_score

####

class Tester(Config):
    def __init__(self, _args=None):
        super(Tester, self).__init__(_args=_args)
        if _args is not None:
            self.__dict__.update(_args.__dict__)
            print(self.run_info)
    
    ####
    def infer_step(self, net, batch, device):
        net.eval()  # infer mode

        imgs, true = batch
        imgs = imgs.permute(0, 3, 1, 2)  # to NCHW

        # push data to GPUs and convert to float32
        imgs = imgs.to(device).float()
        true = true.to(device).long()  # not one-hot

        # -----------------------------------------------------------
        with torch.no_grad():  # dont compute gradient
            out_net = net(imgs, tax=True)  # a list contains all the out put of the network
            if "CLASS" in self.task_type:
                logit_class = out_net
                prob = nn.functional.softmax(logit_class, dim=-1)
                return dict(logit_c=prob.cpu().numpy(),  # from now prob of class task is called by logit_c
                            true=true.cpu().numpy())

            if "REGRESS" in self.task_type:
                if "rank_ordinal" in self.loss_type:
                    logits, probas = out_net[0], out_net[1]
                    predict_levels = probas > 0.5
                    pred = torch.sum(predict_levels, dim=1)
                    return dict(logit_r=pred.cpu().numpy(),
                                true=true.cpu().numpy())
                if "rank_dorn" in self.loss_type:
                    pred, softmax = net(imgs)
                    return dict(logit_r=pred.cpu().numpy(),
                                true=true.cpu().numpy())
                if "soft_label" in self.loss_type:
                    logit_regress = (self.nr_classes - 1) * out_net
                    return dict(logit_r=logit_regress.cpu().numpy(),
                                true=true.cpu().numpy())
                if "FocalOrdinal" in self.loss_type:
                    logit_regress = out_net
                    pred = count_pred(logit_regress)
                    return dict(logit_r=pred.cpu().numpy(),
                                true=true.cpu().numpy())
                else:
                    logit_regress = out_net
                    return dict(logit_r=logit_regress.cpu().numpy(),
                                true=true.cpu().numpy())

            if "MULTI" in self.task_type:
                logit_class, logit_regress = out_net[0], out_net[1]
                prob = nn.functional.softmax(logit_class, dim=-1)
                return dict(logit_c=prob.cpu().numpy(),
                            logit_r=logit_regress.cpu().numpy(),
                            true=true.cpu().numpy())
            

    ####
    def run_once(self, data_root_dir, dataset_=None, fold_idx=None):
        log_dir = self.log_dir
        check_manual_seed(self.seed)
        
        if dataset_ == 'colon_tma_test_1':
            self.dataset = "colon_tma"
            _, _, test_pairs = getattr(dataset, ('prepare_colon_tma_data_test_1'))()
        
        elif dataset_ == 'colon_tma_test_2':
            self.dataset = "colon_tma"
            _, _, test_pairs = getattr(dataset, ('prepare_colon_tma_data_test_2'))()
        
        elif dataset_ == 'prostate_uhu':
            self.dataset = "prostate_uhu"
            _, _, test_pairs = getattr(dataset, ('prepare_%s_data' % self.dataset))()
        
        elif dataset_ == 'prostate_ubc':
            self.dataset = "prostate_ubc"
            test_pairs = getattr(dataset, ('prepare_%s_data' % self.dataset))()

        elif dataset_ == 'aggc2022':
            self.dataset = "aggc2022"
            test_pairs = getattr(dataset, ('prepare_%s_data' % self.dataset))()

        # --------------------------- Dataloader
        infer_augmentors = self.infer_augmentors()  # HACK at has_aux
        test_dataset = dataset.DatasetSerial(test_pairs[:], has_aux=False,
                                             shape_augs=iaa.Sequential(infer_augmentors[0]))


        test_loader = data.DataLoader(test_dataset,
                                      num_workers=self.nr_procs_valid,
                                      batch_size=self.infer_batch_size,
                                      shuffle=False, drop_last=False)

        device = 'cuda'

        # Define network
        net_def = importlib.import_module('model_lib.efficientnet_pytorch.model')  # dynamic import
        net = net_def.jl_efficientnet(task_mode=self.task_type.lower(), pretrained=True)

        PATH_model = args.checkpoint
        net = torch.nn.DataParallel(net).to(device)
        checkpoint = torch.load(PATH_model)
        pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print('Num params:', pytorch_total_params)
        net.load_state_dict(checkpoint)

        net.eval()
        logits_c = []
        trues = []

        # Evaluating
        with tqdm(desc='Epoch %d - evaluation', unit='it', total=len(test_loader)) as pbar:
            for it, (images, gts) in enumerate(iter(test_loader)):
                results = self.infer_step(net, (images, gts), device)
                logits_c.append(results['logit_c'])
                trues.append(results['true'])
                pbar.update()

        logits_c = np.concatenate(logits_c, axis=0)
        trues = np.concatenate(trues)
        preds_c = np.argmax(logits_c, axis=-1)

        # For class indices 1, 2, 3, 4
        if max(trues) == 4:
            trues -= 1

        print('----------------------------- Predictions by classification head -----------------------------')
        print('Precision: ', precision_score(trues, preds_c, average='macro'))
        print('Recall: ', recall_score(trues, preds_c, average='macro'))
        print('F1: ', f1_score(trues, preds_c, average='macro'))
        print('Accuracy: ', accuracy_score(trues, preds_c))
        print('Kw:', cohen_kappa_score(trues, preds_c, weights='quadratic'))
        print('Confusion matrix: ')
        print(confusion_matrix(trues, preds_c))

        return

    ####
    def run(self, data_root_dir=None, dataset=None):
        self.run_once(data_root_dir, dataset, self.fold_idx)
        return

####
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--view', help='view dataset', action='store_true')
    parser.add_argument('--run_info', type=str, default='REGRESS_rank_dorn',
                        help='CLASS, REGRESS, MULTI + loss, '
                             'loss ex: MULTI_mtmr, REGRESS_rank_ordinal, REGRESS_rank_dorn'
                             'REGRESS_FocalOrdinalLoss, REGRESS_soft_ordinal')
    parser.add_argument('--dataset', type=str, default='colon_tma', help='colon_tma_test_1, colon_tma_test_2, prostate_uhu, prostate_ubc, panda_512')
    parser.add_argument('--data_root_dir', type=str, default='../../anhnguyen/dataset/KBSMC_512_test2/KBSMC_test_2/')
    parser.add_argument('--seed', type=int, default=5, help='number')
    parser.add_argument('--alpha', type=int, default=5, help='number')
    parser.add_argument('--checkpoint', type=str, default='/home/compu/doanhbc/JCO_Learning-pytorch/experiments_dir/log_prostate_uhu_20230327_HUBER_SEESAW_350x350/MULTI_ce_mse_cancer_Effi_seed5_BS64/_net_1550.pth')
    parser.add_argument('--log_path', type=str, default='', help='log path')
    args = parser.parse_args()

    tester = Tester(_args=args)

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    tester.run(data_root_dir=args.data_root_dir, dataset=args.dataset)