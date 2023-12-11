import numpy as np
import os
from termcolor import cprint
import math
from sklearn.cluster import KMeans
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.append('.')
from train import ssltrainer
from model import FeatMatch
from loss import common
from util import misc, metric
from util.command_interface import command_interface
from util.reporter import Reporter
from model.teacher import TeacherNetwork

class FeatMatchTrainer(ssltrainer.SSLTrainer):
    def __init__(self, args, config):
        super().__init__(args, config)
        self.fu, self.pu = [], []
        self.fp, self.yp, self.lp = None, None, None
        self.criterion = getattr(common, self.config['loss']['criterion'])
        self.kld = getattr(common, self.config['loss']['kld'])
        self.attr_objs.extend(['fu', 'pu', 'fp', 'yp', 'lp'])
        self.load(args.mode)
    
    def init_teacher(self):
        arch = self.args.ckpt.split('/')[-2]
        model = TeacherNetwork(backbone=arch,
                               num_classes=self.config['model']['classes'],
                               devices=self.args.devices,
                               num_heads=self.config['model']['num_heads'],
                               amp=self.args.amp)
        print(f'Use [{arch}] teacher model with [{misc.count_n_parameters(model):,}] parameters')
        return model
    def init_model(self):
        model = FeatMatch(backbone=self.config['model']['backbone'],
                          num_classes=self.config['model']['classes'],
                          devices=self.args.devices,
                          num_heads=self.config['model']['num_heads'],
                          amp=self.args.amp,
                          pretrain = self.args.pretrain)
        print(f'Use [{self.config["model"]["backbone"]}] student model with [{misc.count_n_parameters(model):,}] parameters')
        return model 

    def get_labeled_featrues(self):
        mode = self.model.training
        self.model.eval()
        with torch.no_grad():
            labeled_dset = self.dataloader_train.labeled_dset.dataset
            xl = torch.stack([self.Tval(xi) for xi in labeled_dset.get_x()])
            #bs = (self.config['train']['bsl'] + self.config['train']['bsu'])*self.config['transform']['data_augment']['K']
            bs = self.config['train']['bsl'] + self.config['train']['bsu']
            fl = []
            for i in range(math.ceil(len(xl) / bs)):
                xli = self.Tnorm(xl[i*bs:min((i+1)*bs, len(xl))].to(self.default_device))
                fli = self.model.extract_feature(xli)
                fl.append(fli.detach().clone().float().cpu())
            fl = torch.cat(fl)
            yl = torch.tensor(labeled_dset.y).cpu()
        self.model.train(mode)

        return fl, yl

    def get_unlabeled_features(self, thres=0.5, max_iter=5):
        if len(self.fu) == 0:
            return None, None

        fu = torch.cat(self.fu).detach().clone()
        pu = torch.cat(self.pu).detach().clone()
        prob, yu = torch.max(pu, dim=1)

        flag = False
        for _ in range(max_iter):
            idx_thres = (prob > thres)
            yu_ = yu[idx_thres]
            class_distribution = torch.stack([torch.sum(yu_ == i) for i in range(self.config['model']['classes'])])

            if not (class_distribution > self.config['model']['pk']).all():
                thres = thres / 2.
            else:
                flag = True
                break

        del self.fu[:]
        del self.pu[:]

        if flag:
            return fu[idx_thres], yu[idx_thres]
        else:
            return None, None

    def extract_fp(self):
        fl, yl = self.get_labeled_featrues() #(250,128), (250)
        fu, yu = self.get_unlabeled_features()
        pk = self.config['model']['pk'] #20, number of prototypes per epoch
        rl = self.config['model']['l_ratio'] #0.5

        all_classes = torch.unique(yl, sorted=True)
        fp, yp, lp = [], [], []
        for yi in torch.unique(yl, sorted=True):
            if fu is None:  # all prototypes extracted from labeled data
                fpi = self.extract_fp_per_class(fl[yl == yi], pk, record_mean=True)
                pkl = len(fpi)
                fp.append(fpi)
                yp.append(torch.full((pkl,), yi, device=self.default_device, dtype=torch.long))
                lp.append(torch.ones_like(yp[-1]))
            else:
                if pk == 1:
                    fpi = self.extract_fp_per_class(torch.cat([fl[yl == yi], fu[yu == yi]]), 1, record_mean=True)
                    pkl = len(fpi)
                    fp.append(fpi)
                    yp.append(torch.full((pkl,), yi, device=self.default_device, dtype=torch.long))
                    lp.append(torch.ones_like(yp[-1]))
                else:
                   # print("unlabeled in")
                    # prototypes extracted from labeled data
                    fpi = self.extract_fp_per_class(fl[yl == yi], max(1, int(round(pk*rl))), record_mean=True)
                    pkl = len(fpi)
                    fp.append(fpi)
                    yp.append(torch.full((pkl,), yi, device=self.default_device, dtype=torch.long))
                    lp.append(torch.ones_like(yp[-1]))
                    # prototypes extracted from unlabeled data
                    fpi = self.extract_fp_per_class(fu[yu == yi], max(1, pk - pkl), record_mean=True)
                    pku = len(fpi)
                    fp.append(fpi)
                    yp.append(torch.full((pku,), yi, device=self.default_device, dtype=torch.long))
                    lp.append(torch.zeros_like(yp[-1]))
        self.fp = torch.cat(fp).to(self.default_device) #(200, 128)
        self.yp = torch.cat(yp).to(self.default_device) #(200)
        self.lp = torch.cat(lp).to(self.default_device) #(200)

        self.fp, self.yp, self.lp = self.retrieve_topk()
       # print("shape", self.fp.shape)

    def retrieve_topk(self):
        labels, counts = torch.unique(self.yp, return_counts=True)
        top_k_labels = labels[counts.sort(descending=True).indices][:5]
        indices = torch.where(self.yp.unsqueeze(1) == top_k_labels)[0]
        selected_fp = self.fp[indices]
        selected_yp = self.yp[indices]
        selected_lp = self.lp[indices]
        return selected_fp, selected_yp, selected_lp

    def extract_fp_per_class(self, fx, n, record_mean=True):
        if n == 1:
            fp = torch.mean(fx, dim=0, keepdim=True)
        elif record_mean:
            n = n-1
            fm = torch.mean(fx, dim=0, keepdim=True)
            if n >= len(fx):
                fp = fx
            else:
                fp = self.kmeans(fx, n, 'cosine')
            fp = torch.cat([fm, fp], dim=0)
        else:
            if n >= len(fx):
                fp = fx
            else:
                fp = self.kmeans(fx, n, 'cosine')

        return fp #(20, 128)

    @staticmethod
    def kmeans(fx, n, metric='cosine'):
        device = fx.device

        if metric == 'cosine':
            fn = fx / torch.clamp(torch.norm(fx, dim=1, keepdim=True), min=1e-20)
        elif metric == 'euclidean':
            fn = fx
        else:
            raise KeyError
        fn = fn.detach().cpu().numpy()
        fx = fx.detach().cpu().numpy()

        labels = KMeans(n_clusters=n).fit_predict(fn)
        fp = np.stack([np.mean(fx[labels == li], axis=0) for li in np.unique(labels)])
        fp = torch.FloatTensor(fp).to(device)

        return fp

    def data_mixup(self, xl, prob_xl, xu, prob_xu, alpha=0.75):
        Nl = len(xl)

        x = torch.cat([xl, xu], dim=0)
        prob = torch.cat([prob_xl, prob_xu], dim=0).detach()

        idx = torch.randperm(x.shape[0])
        x_, prob_ = x[idx], prob[idx]

        l = np.random.beta(alpha, alpha)
        l = max(l, 1 - l)

        x = l * x + (1 - l) * x_
        prob = l * prob + (1 - l) * prob_
        prob = prob / prob.sum(dim=1, keepdim=True)

        xl, xu = x[:Nl], x[Nl:]
        probl, probu = prob[:Nl], prob[Nl:]

        return xl, probl, xu, probu

    def train1(self, xl, yl, xu):
        # Forward pass on original data
        bsl, bsu, k, c = len(xl), len(xu), xl.size(1), self.config['model']['classes'] # 64, 128, k:1, c:10
        x = torch.cat([xl, xu], dim=0).reshape(-1, *xl.shape[2:]) #(192*2, 3, 32, 32)
        logits_x = self.model(x) # (192*2, 10)
        logits_x = logits_x.reshape(bsl + bsu, k, c) #(192, 2, 10)
        logits_xl, logits_xu = logits_x[:bsl], logits_x[bsl:] #(64,2,10),(128,2,10)

        # Compute pseudo label
        prob_xu_fake = torch.softmax(logits_xu[:, 0].detach(), dim=1) # 첫번째 증강 변형의 로짓만 선택 ; (128, 10), detach : 다른 tensor 객체로 분리, 특징은 requires grad가 false로 바뀜
        prob_xu_fake = prob_xu_fake ** (1. / self.config['transform']['data_augment']['T']) # temperature
        prob_xu_fake = prob_xu_fake / prob_xu_fake.sum(dim=1, keepdim=True) # 확률값을 정규화 하여 각 샘플의 확률 합이 1이 되도록 함
        prob_xu_fake = prob_xu_fake.unsqueeze(1).repeat(1, k, 1) #(128, 2, 10)

        # Mixup perturbation
        xu = xu.reshape(-1, *xu.shape[2:])
        xl = xl.reshape(-1, *xl.shape[2:])
        prob_xl_gt = torch.zeros(len(xl), c, device=xl.device)
        prob_xl_gt.scatter_(dim=1, index=yl.unsqueeze(1).repeat(1, k).reshape(-1, 1), value=1.)
        xl_mix, probl_mix, xu_mix, probu_mix = self.data_mixup(xl, prob_xl_gt, xu, prob_xu_fake.reshape(-1, c))

        # Forward pass on mixed data
        Nl = len(xl_mix)
        x_mix = torch.cat([xl_mix, xu_mix], dim=0)
        logits_x_mix = self.model(x_mix)
        logits_xl_mix, logits_xu_mix = logits_x_mix[:Nl], logits_x_mix[Nl:]

        # CLF loss
        loss_pred = self.criterion(None, probl_mix, logits_xl_mix, None)

        # Mixup loss
        loss_con = self.criterion(None, probu_mix, logits_xu_mix, None)

        # Graph loss
        loss_graph = torch.tensor(0.0, device=self.default_device)

        # Total loss
        coeff = self.get_consistency_coeff()
        loss = loss_pred + coeff*self.config['loss']['mix']*loss_con
        # Prediction
        pred_x = torch.softmax(logits_xl[:, 0].detach(), dim=1)

        return pred_x, loss, loss_pred, loss_con, loss_graph

    def train2(self, xl, yl, xu):
        bsl, bsu, k, c = len(xl), len(xu), xl.size(1), self.config['model']['classes']
        x = torch.cat([xl, xu], dim=0).reshape(-1, *xl.shape[2:])
        logits_xg, _, fx, _, _ = self.model(x, self.fp)
        logits_xg = logits_xg.reshape(bsl + bsu, k, c)
        logits_xgl, logits_xgu = logits_xg[:bsl], logits_xg[bsl:]
        fxu = fx.reshape(bsl + bsu, k, fx.size(1))[bsl:]
        self.fu.append(fxu[:, 0].detach().clone().float().cpu())

        # Compute pseudo label
        prob_xu_fake = torch.softmax(logits_xgu[:, 0].detach(), dim=1)
        self.pu.append(prob_xu_fake.detach().clone().float().cpu())
        prob_xu_fake = prob_xu_fake ** (1. / self.config['transform']['data_augment']['T'])
        prob_xu_fake = prob_xu_fake / prob_xu_fake.sum(dim=1, keepdim=True)
        prob_xu_fake = prob_xu_fake.unsqueeze(1).repeat(1, k, 1)

        # Mixup perturbation
        xu = xu.reshape(-1, *xu.shape[2:])
        xl = xl.reshape(-1, *xl.shape[2:])
        prob_xl_gt = torch.zeros(len(xl), c, device=xl.device)
        prob_xl_gt.scatter_(dim=1, index=yl.unsqueeze(1).repeat(1, k).reshape(-1, 1), value=1.)
        xl_mix, probl_mix, xu_mix, probu_mix = self.data_mixup(xl, prob_xl_gt, xu, prob_xu_fake.reshape(-1, c))

        # Forward pass on mixed data
        Nl = len(xl_mix)
        x_mix = torch.cat([xl_mix, xu_mix], dim=0)
        logits_xg_mix, logits_xf_mix, _, _, _ = self.model(x_mix, self.fp)
        logits_xgl_mix, logits_xgu_mix = logits_xg_mix[:Nl], logits_xg_mix[Nl:]
        logits_xfl_mix, logits_xfu_mix = logits_xf_mix[:Nl], logits_xf_mix[Nl:]

        coeff = self.get_consistency_coeff()

        # KLD Loss
        logits_teacher_mix = self.teacher(x_mix, self.fp)
        logits_teacher_l_mix, logits_teacher_u_mix = logits_teacher_mix[:Nl], logits_teacher_mix[Nl:]
        loss_kld_l = self.kld(None, probl_mix, logits_teacher_l_mix, None)
        loss_kld_u = self.kld(None, probu_mix, logits_teacher_u_mix, None)
        loss_kld = loss_kld_l + coeff*(self.config['loss']['mix'] * loss_kld_u)

        # CLF loss
        loss_pred = self.criterion(None, probl_mix, logits_xgl_mix, None)

        # Mixup loss
        loss_con = self.criterion(None, probu_mix, logits_xgu_mix, None)

        # Graph loss
        loss_graph = self.criterion(None, probu_mix, logits_xfu_mix, None)

        # Total loss
        loss = loss_pred + coeff * (self.config['loss']['mix'] * loss_con + self.config['loss']['graph'] * loss_graph)
        if self.args.kld:
            loss = loss + loss_kld
      #  print(loss_kld, loss)
        # Prediction
        pred_x = torch.softmax(logits_xgl[:, 0].detach(), dim=1)

        return pred_x, loss, loss_pred, loss_con, loss_graph

    def eval1(self, x, y):
        logits_x = self.model(x)

        # Compute pseudo label
        prob_fake = torch.softmax(logits_x.detach(), dim=1)
        prob_fake = prob_fake ** (1. / self.config['transform']['data_augment']['T'])
        prob_fake = prob_fake / prob_fake.sum(dim=1, keepdim=True)

        # Mixup perturbation
        prob_gt = torch.zeros(len(y), self.config['model']['classes'], device=x.device)
        prob_gt.scatter_(dim=1, index=y.unsqueeze(1), value=1.)
        x_mix, prob_mix, _, _ = self.data_mixup(x, prob_gt, x, prob_fake)

        # Forward pass on mixed data
        logits_x_mix = self.model(x_mix)

        # CLF loss and Mixup loss
        loss_con = loss_pred = self.criterion(None, prob_mix, logits_x_mix, None)

        # Graph loss
        loss_graph = torch.tensor(0.0, device=self.default_device)

        # Total loss
        coeff = self.get_consistency_coeff()
        loss = loss_pred + coeff*self.config['loss']['mix']*loss_con

        # Prediction
        pred_x = torch.softmax(logits_x.detach(), dim=1)

        return pred_x, loss, loss_pred, loss_con, loss_graph

    def eval2(self, x, y):
        logits_xg, logits_xf, _, _, _ = self.model(x, self.fp)

        # Compute pseudo label
        prob_fake = torch.softmax(logits_xg.detach(), dim=1)
        prob_fake = prob_fake ** (1. / self.config['transform']['data_augment']['T'])
        prob_fake = prob_fake / prob_fake.sum(dim=1, keepdim=True)

        # Mixup perturbation
        prob_gt = torch.zeros(len(y), self.config['model']['classes'], device=x.device)
        prob_gt.scatter_(dim=1, index=y.unsqueeze(1), value=1.)
        x_mix, prob_mix, _, _ = self.data_mixup(x, prob_gt, x, prob_fake)

        # Forward pass on mixed data
        logits_xg_mix, logits_xf_mix, _, _, _ = self.model(x_mix, self.fp)

        # CLF loss and Mixup loss
        loss_con = loss_pred = self.criterion(None, prob_mix, logits_xg_mix, None)

        # Graph loss
        loss_graph = self.criterion(None, prob_mix, logits_xf_mix, None)

        # Total loss
        coeff = self.get_consistency_coeff()
        loss = loss_pred + coeff*(self.config['loss']['mix']*loss_con + self.config['loss']['graph']*loss_graph)

        # Prediction
        pred_x = torch.softmax(logits_xg.detach(), dim=1)

        return pred_x, loss, loss_pred, loss_con, loss_graph

    def forward_train(self, data):
        self.model.train()
        xl = data[0].reshape(-1, *data[0].shape[2:])
        xl = self.Tnorm(xl.to(self.default_device)).reshape(data[0].shape)
        yl = data[1].to(self.default_device)
        xu = data[2].reshape(-1, *data[2].shape[2:])
        xu = self.Tnorm(xu.to(self.default_device)).reshape(data[2].shape)

        if self.curr_iter < self.config['train']['pretrain_iters']:
            self.model.set_mode('pretrain')
            pred_x, loss, loss_pred, loss_con, loss_graph = self.train1(xl, yl, xu)
        elif self.curr_iter == self.config['train']['pretrain_iters']:
            self.model.set_mode('train')
            self.extract_fp()
            pred_x, loss, loss_pred, loss_con, loss_graph = self.train2(xl, yl, xu)
        else:
            self.model.set_mode('train')
            if self.curr_iter % self.config['train']['sample_interval'] == 0: # update prototype 주기
                self.extract_fp()
            pred_x, loss, loss_pred, loss_con, loss_graph = self.train2(xl, yl, xu)

        results = {
            'y_pred': torch.max(pred_x, dim=1)[1].detach().cpu().numpy(),
            'y_true': yl.cpu().numpy(),
            'loss': {
                'all': loss.detach().cpu().item(),
                'pred': loss_pred.detach().cpu().item(),
                'con': loss_con.detach().cpu().item(),
                'graph': loss_graph.detach().cpu().item()
            }
        }

        return loss, results

    def forward_eval(self, data):
        self.model.eval()
        x = self.Tnorm(data[0].to(self.default_device))
        y = data[1].to(self.default_device)

        if self.curr_iter < self.config['train']['pretrain_iters']:
            self.model.set_mode('pretrain')
            pred_x, loss, loss_pred, loss_con, loss_graph = self.eval1(x, y)
        else:
            self.model.set_mode('train')
            pred_x, loss, loss_pred, loss_con, loss_graph = self.eval2(x, y)

        results = {
            'y_pred': torch.max(pred_x, dim=1)[1].detach().cpu().numpy(),
            'y_true': y.cpu().numpy(),
            'loss': {
                'all': loss.detach().cpu().item(),
                'pred': loss_pred.detach().cpu().item(),
                'con': loss_con.detach().cpu().item(),
                'graph': loss_graph.detach().cpu().item()
            }
        }

        return results


if __name__ == '__main__':
    args, config, save_root = command_interface()

    r = args.rand_seed
    reporter = Reporter(save_root, args)

    for i in range(args.iters):
        args.rand_seed = r + i
        cprint(f'Run iteration [{i+1}/{args.iters}] with random seed [{args.rand_seed}]', attrs=['bold'])

        setattr(args, 'save_root', save_root/f'run{i}')
        if args.mode == 'resume' and not args.save_root.exists():
            args.mode = 'new'
        args.save_root.mkdir(parents=True, exist_ok=True)

        trainer = FeatMatchTrainer(args, config)
        if args.mode != 'test':
            trainer.train()

        acc_val, acc_test = trainer.test()
        acc_median = metric.median_acc(os.path.join(args.save_root, 'results.txt'))
        reporter.record(acc_val, acc_test, acc_median)
        with open(args.save_root/'final_result.txt', 'w') as file:
            file.write(f'Val acc: {acc_val*100:.2f} %')
            file.write(f'Test acc: {acc_test*100:.2f} %')
            file.write(f'Median acc: {acc_median*100:.2f} %')

    reporter.report()