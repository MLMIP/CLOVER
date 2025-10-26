import copy
import os.path
import random

import numpy as np
import torch
from torch import nn

from argparse import ArgumentParser
from itertools import compress

from matplotlib import pyplot as plt

from .mvgb import ClassMemoryDataset, ClassDirectoryDataset, MyDummyDataset
from .gmm import GaussianMixture
from .incremental_learning import Inc_Learning_Appr

torch.backends.cuda.matmul.allow_tf32 = False


def softmax_temperature(x, dim, tau=1.0):
    return torch.softmax(x / tau, dim=dim)


class Appr(Inc_Learning_Appr):
    """Class implementing the joint baseline"""

    def __init__(self, model, device, nepochs=200, ftepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5,
                 clipgrad=10000,
                 momentum=0, wd=0, ftwd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, patience=5, fix_bn=False,
                 eval_on_train=False,
                 logger=None, max_experts=999, gmms=1, alpha=1.0, tau=3.0, shared=0, use_multivariate=False,
                 use_nmc=False,
                 initialization_strategy="first", compensate_drifts=False,
                 batch_size=128, save_models=False, loop_choose=False, balance=0,
                 w0=0, w1=0,
                 ft_lr=None, ft_wd=None, dataset='cifar100', taskcla=None):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset=None)
        self.balance = balance

        self.max_experts = max_experts
        self.model.bbs = self.model.bbs[:max_experts]
        self.taskcla = taskcla

        # Param for PWP
        self.w0 = w0
        self.w1 = w1

        self.dataset = dataset

        self.first_model = None

        # 为微调阶段使用新的学习率和wd
        self.ft_lr = ft_lr if ft_lr is not None else lr
        self.ft_wd = ft_wd if ft_wd is not None else wd

        self.gmms = gmms
        self.alpha = alpha
        self.tau = tau
        self.patience = patience
        self.use_multivariate = use_multivariate
        self.use_nmc = use_nmc
        self.ftepochs = ftepochs
        self.ftwd = ftwd
        self.compensate_drifts = compensate_drifts
        self.model.to(device)
        self.experts_distributions = []

        self.shared_layers = []
        self.batch_size = batch_size
        self.save_models = save_models
        if shared > 0:
            self.shared_layers = ["conv1_starting.weight", "bn1_starting.weight", "bn1_starting.bias", "layer1"]
            if shared > 1:
                self.shared_layers.append("layer2")
                if shared > 2:
                    self.shared_layers.append("layer3")
                    if shared > 3:
                        self.shared_layers.append("layer4")

        self.initialization_strategy = initialization_strategy

        self.prototype_dict = [{} for _ in range(max_experts)]
        self.radius = []

        self.temp = 0.1
        self.train_history = {i: [] for i in range(max_experts)}
        self.train_history_task = {i: [] for i in range(max_experts)}

        self.class_weight_per_expert = None
        self.weight_per_expert = None
        self.now_task = 0

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--max-experts',
                            help='Maximum number of experts',
                            type=int,
                            default=999)
        parser.add_argument('--gmms',
                            help='Number of gaussian models in the mixture',
                            type=int,
                            default=1)
        parser.add_argument('--shared',
                            help='Number of shared blocks',
                            type=int,
                            default=0)
        parser.add_argument('--initialization-strategy',
                            help='How to initialize experts weight',
                            type=str,
                            choices=["first", "random"],
                            default="first")
        parser.add_argument('--ftepochs',
                            help='Number of epochs for finetuning an expert',
                            type=int,
                            default=100)
        parser.add_argument('--ftwd',
                            help='Weight decay for finetuning',
                            type=float,
                            default=0)
        parser.add_argument('--use-multivariate',
                            help='Use multivariate distribution',
                            action='store_true',
                            default=True)
        parser.add_argument('--use-nmc',
                            help='Use nearest mean classifier instead of bayes',
                            action='store_true',
                            default=False)
        parser.add_argument('--alpha',
                            help='relative weight of kd loss',
                            type=float,
                            default=0.99)
        parser.add_argument('--tau',
                            help='softmax temperature',
                            type=float,
                            default=3.0)
        parser.add_argument('--compensate-drifts',
                            help='Drift compensation using MLP feature adaptation',
                            action='store_true',
                            default=False)
        return parser.parse_known_args(args)

    def train(self, t, trn_loader, val_loader):
        """Main train structure"""
        return self.train_loop(t, trn_loader, val_loader)

    def train_loop(self, t, trn_loader, val_loader):
        self.now_task = t

        trained_expert = t
        class_end = sum([i for _, i in self.model.taskcla[:t + 1]])
        class_start = class_end - self.model.taskcla[t][1]

        if t < self.max_experts:
            print(f"Training backbone on task {t}:")
            if not os.path.exists(f"{self.logger.exp_path}/models/model_{t}.pth"):
                self.train_backbone(t, trn_loader, val_loader)
                if self.save_models:
                    torch.save(self.model, f"{self.logger.exp_path}/models/model_{t}.pth")
            else:
                print(f"model exists for task {t}")
                print(f"load exists model for task {t}")
                self.model = torch.load(f"{self.logger.exp_path}/models/model_{t}.pth")
                self.model.device = self.device
                self.model.taskcla = self.taskcla

            self.experts_distributions.append([])

            self.train_history_task[t].append(t)
            for item in range(class_start, class_end):
                self.train_history[t].append(item)
        else:
            trained_expert = 0

        print(f"Creating distributions for task {t}:")
        self.create_distributions(t, trn_loader, val_loader)

        print(f"Calculate_class_weight_per_expert until task {t}:")
        self.calculate_class_weight_per_expert(t)

        return trained_expert

    def train_backbone(self, t, trn_loader, val_loader):

        if self.initialization_strategy == "random" or t == 0:
            if self.dataset[0] in ['tinyImagenet200_fecam', 'imagenet_subset_fecam', 'cifar100_fecam']:
                if self.dataset[0] == 'tinyImagenet200_fecam':
                    dataset_name = 'tinyimagenet200'
                elif self.dataset[0] == 'imagenet_subset_fecam':
                    dataset_name = 'imagenet100'
                elif self.dataset[0] == 'cifar100_fecam':
                    dataset_name = 'cifar100'
                self.model.bbs.append(self.model.bb_fun(num_classes=self.model.taskcla[t][1],
                                                        num_features=self.model.num_features,
                                                        dataset=dataset_name))
            else:
                use_last_block = True
                if self.dataset[0] in ['tinyImagenet200_224']:
                    print("use_last_block=False")
                    use_last_block = False
                else:
                    print("use_last_block=True")
                self.model.bbs.append(
                    self.model.bb_fun(num_classes=self.model.taskcla[t][1], num_features=self.model.num_features,
                                      use_last_block=use_last_block))
        else:
            self.model.bbs.append(copy.deepcopy(self.model.bbs[0]))

        model = self.model.bbs[t]

        noise_std = 0.5
        print(f'noise_std: {noise_std}')

        model.fc = nn.Linear(self.model.num_features, 7 * self.model.taskcla[t][1])

        if t == 0:
            for param in model.parameters():
                param.requires_grad = True
        else:
            for name, param in model.named_parameters():
                param.requires_grad = True
                for layer_not_to_train in self.shared_layers:
                    if layer_not_to_train in name:
                        model.get_parameter(name).data = self.model.bbs[0].get_parameter(name).data
                        param.requires_grad = False

        print(f'The expert has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')
        print(f'The expert has {sum(p.numel() for p in model.parameters() if not p.requires_grad):,} shared parameters')

        model.to(self.device)
        optimizer, lr_scheduler = self._get_optimizer(t)

        train_total_losses_list, test_total_losses_list = [], []
        total_dis_loss_list = []

        for epoch in range(self.nepochs):

            train_loss, valid_loss = [], []
            train_hits, val_hits = 0, 0
            model.train()
            for images, targets in trn_loader:
                targets -= self.model.task_offset[t]

                images, targets = images.to(self.device), targets.to(self.device)

                # Generate confusion classes
                size = images.shape[1:]
                noise = torch.randn_like(images) * noise_std
                images = torch.stack([images,
                                      images + noise,
                                      torch.rot90(images, 1, (2, 3)),
                                      torch.rot90(images, 2, (2, 3)),
                                      torch.rot90(images, 3, (2, 3)),
                                      torch.stack([images[:, 1, :, :], images[:, 2, :, :], images[:, 0, :, :]], 1),
                                      torch.stack([images[:, 2, :, :], images[:, 0, :, :], images[:, 1, :, :]], 1)
                                      ],
                                     1)
                images = images.view(-1, *size)
                targets = torch.stack([targets * 7 + k for k in range(7)], 1).view(-1)

                bsz = images.shape[0]
                optimizer.zero_grad()
                out = model(images)
                loss = nn.functional.cross_entropy(out, targets, label_smoothing=0.0)
                # print(f"out.shape: {out.shape}")
                # print(f"targets.shape: {targets.shape}")
                # print(f"out: {out}")
                # print(f"targets: {targets}")
                # print(f"loss={loss}")
                # break
                optimizer.zero_grad()
                loss.backward()
                if self.dataset[0] not in ['tinyImagenet200', 'tinyImagenet200_fecam']:
                    # clipgrad=1.0;
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.clipgrad)
                optimizer.step()
                _, preds = torch.max(out, dim=1)
                train_hits += preds.eq(targets.expand_as(preds)).cpu().sum()
                train_loss.append(bsz * loss.item())

            lr_scheduler.step()

            # eval per 5 epochs
            if epoch % 5 == 0:
                model.eval()
                for images, targets in val_loader:
                    targets -= self.model.task_offset[t]

                    images, targets = images.to(self.device), targets.to(self.device)
                    out = model(images)
                    # only calculate accuracy of raw classes
                    out = out[:, ::7]

                    val_loss = nn.functional.cross_entropy(out, targets, label_smoothing=0.0)
                    val_hits += float(torch.sum((torch.argmax(out, dim=1) == targets)))
                    valid_loss.append(float(images.shape[0] * val_loss))

            train_loss = sum(train_loss) / len(trn_loader.dataset)
            valid_loss = sum(valid_loss) / len(val_loader.dataset)

            train_total_losses_list.append(train_loss)
            test_total_losses_list.append(valid_loss)

            train_acc = train_hits / len(trn_loader.dataset) / 7
            val_acc = val_hits / len(val_loader.dataset)

            print(f"Epoch: {epoch} Train loss: {train_loss:.2f} Val loss: {valid_loss:.2f} "
                  f"Train acc: {100 * train_acc:.2f} Val acc: {100 * val_acc:.2f}")

        """绘制训练/测试损失随epoch变化的折线图"""
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        ax[0, 0].plot(np.arange(len(train_total_losses_list)), train_total_losses_list, label="train loss")
        ax[0, 0].plot(np.arange(len(total_dis_loss_list)), total_dis_loss_list, label="dis loss")
        ax[0, 0].plot(np.arange(len(test_total_losses_list)), test_total_losses_list, label="test loss")
        ax[0, 0].set(xlabel='epoch',
                     ylabel='loss',
                     title='total_losses_list')
        ax[0, 0].legend()

        plt.savefig(f"{self.logger.exp_path}/figures/loss_{t}.png")

        # remove fc
        model.fc = nn.Identity()
        self.model.bbs[t] = model

    @torch.no_grad()
    def create_distributions(self, t, trn_loader, val_loader):
        """ Create distributions for task t"""
        self.model.eval()
        classes = self.model.taskcla[t][1]
        self.model.task_offset.append(self.model.task_offset[-1] + classes)

        transforms = val_loader.dataset.transform

        for bb_num in range(min(self.max_experts, t + 1)):
            model = self.model.bbs[bb_num]
            for index in range(classes):

                c = index + self.model.task_offset[t]

                train_indices = torch.tensor(trn_loader.dataset.labels) == c
                if self.dataset[0] in ['tinyImagenet200', 'tinyImagenet200_64', 'tinyImagenet200_128',
                                       'tinyImagenet200_224', 'tinyImagenet200_fecam']:
                    ds = trn_loader.dataset.images[train_indices]  # 获取类c的所有样本
                    ds = MyDummyDataset(ds, transforms)
                elif isinstance(trn_loader.dataset.images, list):
                    train_images = list(compress(trn_loader.dataset.images, train_indices))
                    ds = ClassDirectoryDataset(train_images, transforms)
                else:
                    ds = trn_loader.dataset.images[train_indices]  # 获取类c的所有样本
                    ds = ClassMemoryDataset(ds, transforms)  # 将样本构成数据集, 带transforms的
                loader = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, num_workers=trn_loader.num_workers,
                                                     shuffle=False)  # 构建dataloader
                eps = 1e-8
                from_ = 0
                class_features = torch.full((2 * len(ds), self.model.num_features), fill_value=-999999999.0,
                                            device=self.model.device)  # (class_num*2,64); *2是后续需要记录翻转后的特征
                for images in loader:
                    bsz = images.shape[0]
                    images = images.to(self.device)
                    features = model.extract_feature(images)

                    class_features[from_: from_ + bsz] = features
                    features = model.extract_feature(torch.flip(images, dims=(3,)))

                    class_features[from_ + bsz: from_ + 2 * bsz] = features
                    from_ += 2 * bsz

                # Calculate distributions
                cov_type = "full" if self.use_multivariate else "diag"  # "full", 多变量高斯分布？
                is_ok = False
                while not is_ok:
                    try:
                        # 为专家bb_num生成对应的多类高斯分布,
                        gmm = GaussianMixture(self.gmms, class_features.shape[1], covariance_type=cov_type,
                                              eps=eps).to(
                            self.device)
                        gmm.fit(class_features, delta=1e-3, n_iter=100)  # 用高斯分布拟合特征
                    except RuntimeError:
                        eps = 10 * eps
                        print(
                            f"WARNING: Covariance matrix is singular. Increasing eps to: {eps:.7f} but this may hurt results")
                    else:
                        is_ok = True

                if len(gmm.mu.data.shape) == 2:
                    gmm.mu.data = gmm.mu.data.unsqueeze(1)
                self.experts_distributions[bb_num].append(gmm)

    @torch.no_grad()
    def eval(self, t, val_loaders, all_tst_load=None):
        val_loader = val_loaders[t]

        """Contains the evaluation code"""
        total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
        self.model.eval()
        for images, targets in val_loader:
            targets = targets.to(self.device)

            features = self.model(images.to(self.device))
            # Calculate TIL and CIL accuracy
            hits_taw, hits_tag = self.calculate_metrics(features, targets, t)
            # Log
            total_loss = 0
            total_acc_taw += hits_taw.sum().item()
            total_acc_tag += hits_tag.sum().item()
            total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    @torch.no_grad()
    def calculate_metrics(self, features, targets, t):
        """Contains the main Task-Aware and Task-Agnostic metrics"""

        taw_pred, tag_pred = self.predict_class_bayes(t, features, targets)

        hits_taw = (taw_pred == targets).float()
        hits_tag = (tag_pred == targets).float()
        return hits_taw, hits_tag

    @torch.no_grad()
    def predict_class_bayes(self, t, features, targets=None):
        log_probs = torch.full((features.shape[0], len(self.experts_distributions), len(self.experts_distributions[0])),
                               fill_value=-1e8, device=features.device)
        mask = torch.full_like(log_probs, fill_value=False, dtype=torch.bool)

        for bb_num, _ in enumerate(self.experts_distributions):
            for c, class_gmm in enumerate(self.experts_distributions[bb_num]):
                c += self.model.task_offset[bb_num]
                log_probs[:, bb_num, c] = class_gmm.score_samples(features[:, bb_num])
                mask[:, bb_num, c] = True

        from_ = self.model.task_offset[t]
        to_ = self.model.task_offset[t + 1]

        # Task-Aware
        taw_log_probs = log_probs[:, :t + 1, from_:to_].clone()
        taw_log_probs = softmax_temperature(taw_log_probs, dim=2, tau=self.tau)
        confidences = torch.sum(taw_log_probs, dim=1)
        taw_class_id = torch.argmax(confidences, dim=1) + self.model.task_offset[t]

        # Task-Agnostic
        log_probs = softmax_temperature(log_probs, dim=2, tau=self.tau)
        # weight predict
        log_probs = log_probs[:] * self.class_weight_per_expert[:mask.shape[1], :mask.shape[2]].to(self.device)
        confidences = torch.sum(log_probs, dim=1)  # 求所有专家的预测概率平均值

        tag_class_id = torch.argmax(confidences, dim=1)  # 求出未知task-id时的预测结果
        return taw_class_id, tag_class_id

    def _get_optimizer(self, expert_id, milestones=[60, 120, 160]):
        """Returns the optimizer"""
        train_params = []
        shared_params = []
        for k, v in self.model.bbs[expert_id].named_parameters():
            if v.requires_grad:
                train_params.append(v)
                continue
            for layer_not_to_train in self.shared_layers:
                if layer_not_to_train in k:
                    shared_params.append(v)

        train_params = {'params': train_params, 'lr': self.lr, 'weight_decay': self.wd}
        shared_params = {'params': shared_params, 'lr': self.lr * 0.1, 'weight_decay': 2e-4}
        total_params = [train_params]
        optimizer = torch.optim.SGD(total_params, lr=self.lr, weight_decay=self.wd, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)

        return optimizer, scheduler

    @torch.no_grad()
    def calculate_class_weight_per_expert(self, t):
        class_numbers = sum([i for _, i in self.model.taskcla[:t + 1]])
        class_total = sum([i for _, i in self.model.taskcla])

        self.class_weight_per_expert = [[0 for i in range(class_total)] for _ in range(min(self.max_experts, t + 1))]

        # PWP
        w0 = self.w0
        w1 = self.w1

        if self.weight_per_expert is None:
            self.weight_per_expert = [0]

        for i in range(1, min(t, self.max_experts)):
            self.weight_per_expert[i] = min(self.weight_per_expert[i] + w1, 0.2)
        if t != 0 and t < self.max_experts:
            self.weight_per_expert.append(w0)
        self.weight_per_expert[0] = 1 - sum(self.weight_per_expert[1:])

        print(f'The weight of each expert: {self.weight_per_expert}')

        for task in range(t + 1):
            total_weight = sum(self.weight_per_expert[:min(task + 1, self.max_experts)])
            weight_per_expert = [v / total_weight for v in self.weight_per_expert[:min(task + 1, self.max_experts)]]
            for cls in range(self.model.task_offset[task], self.model.task_offset[task + 1]):
                for e in range(len(weight_per_expert)):
                    self.class_weight_per_expert[e][cls] = weight_per_expert[e]
        # The weight of each expert on each class
        self.class_weight_per_expert = torch.asarray(self.class_weight_per_expert)
