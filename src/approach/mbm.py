import copy
import os.path
import random

import numpy as np
import torch

from argparse import ArgumentParser
from itertools import compress
from torch import nn
from torch.utils.data import Dataset
from torch.distributions import MultivariateNormal

from .mvgb import ClassMemoryDataset, ClassDirectoryDataset
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
                 batch_size=128, save_models=False):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset=None)
        self.max_experts = max_experts
        self.model.bbs = self.model.bbs[:max_experts]
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
            # conv+bn+10*conv(11)
            self.shared_layers = ["conv1_starting.weight", "bn1_starting.weight", "bn1_starting.bias", "layer1"]
            if shared > 1:
                # conv+bn+10*conv+10*conv(21)
                self.shared_layers.append("layer2")
                if shared > 2:
                    # conv+bn+10*conv+10*conv+10*conv(31)
                    self.shared_layers.append("layer3")
                    if shared > 3:
                        self.shared_layers.append("layer4")
        # 延后train_backbone的训练范围；有base阶段的Setting下, 第一个任务(base阶段)是不参与backbone训练的, 而是用base模型预训练模型
        self.offset = 0

        self.initialization_strategy = initialization_strategy

        # 为每个专家都保存对应的类原型
        # 1、每个专家仅都保存自身训练过的类原型
        # 2、每个专家保存所有类的原型
        # self.prototype_dict[t][c]: 保存在专家t中的类c的原型
        self.prototype_dict = [{} for _ in range(max_experts)]
        self.radius = 0
        self.protoAug_weight = 0  # 保持与PASS、SSRE一样的权重; 使用小权重
        self.temp = 0.1
        # 记录每个专家训练过哪些类
        self.train_history = {i: [] for i in range(max_experts)}
        # 记录每个专家训练过哪些任务
        self.train_history_task = {i: [] for i in range(max_experts)}

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


    def train_loop(self, t, trn_loader, val_loader):
        train = True
        if os.path.exists(f"{self.logger.exp_path}/models/model_{t}.pth"):
            print("model exists for task {}".format(t))
            # 读取模型、原型、train_history和train_history_task
            # self.model.load_state_dict(torch.load(f"{self.logger.exp_path}/models/model_{t}.pth"))
            self.model = torch.load(f"{self.logger.exp_path}/models/model_{t}.pth")
            # self.prototype_dict = torch.load(f"{self.logger.exp_path}/models/prototype_dict_{t}.pth")
            # self.train_history = torch.load(f"{self.logger.exp_path}/models/train_history_{t}.pth")
            # self.train_history_task = torch.load(f"{self.logger.exp_path}/models/train_history_task_{t}.pth")
            train = False

        class_end = sum([i for _, i in self.model.taskcla[:t + 1]])
        class_start = class_end - self.model.taskcla[t][1]

        if t == 0 and self.model.taskcla[0][1] == self.model.taskcla[1][1]:
            if train:
                print(f"PreTraining backbone on base_task {t}:")
                self.train_base_process(trn_loader, val_loader)
                if self.save_models:
                    torch.save(self.model, f"{self.logger.exp_path}/models/model_{t}.pth")  # 直接保存对象
                    """还是需要保存模型, 以便观察数据"""
                    # torch.save(self.model.state_dict(), f"{self.logger.exp_path}/models/model_{t}.pth")
                    # torch.save(self.prototype_dict, f"{self.logger.exp_path}/models/prototype_dict_{t}.pth")  # 直接保存对象
                    # torch.save(self.train_history, f"{self.logger.exp_path}/models/train_history_{t}.pth")  # 直接保存对象
                    # torch.save(self.train_history_task, f"{self.logger.exp_path}/models/train_history_task_{t}.pth")  # 直接保存对象
            self.experts_distributions.append([])
            # 每个专家都以base模型为起点，必然都训练过base类
            for sub_experts in range(self.max_experts):
                self.train_history_task[sub_experts].append(t)
                for item in range(class_start, class_end):
                    self.train_history[sub_experts].append(item)
            self.offset = 1
            # 训练结束后，保存改该task的原型到对应的专家中
            # self.proto_save(t, trn_loader, first_task=True)

        elif t < self.max_experts + self.offset:  # 任务数未超过专家数, 则直接选择第t个专家来学习
            if train:
                print(f"Training backbone on task {t}:")
                self.train_backbone(t, trn_loader, val_loader)
                if self.save_models:
                    torch.save(self.model, f"{self.logger.exp_path}/models/model_{t}.pth")  # 直接保存对象
                    """还是需要保存模型, 以便观察数据"""
                    # torch.save(self.model.state_dict(), f"{self.logger.exp_path}/models/model_{t}.pth")
                    # torch.save(self.prototype_dict, f"{self.logger.exp_path}/models/prototype_dict_{t}.pth")  # 直接保存对象
                    # torch.save(self.train_history, f"{self.logger.exp_path}/models/train_history_{t}.pth")  # 直接保存对象
                    # torch.save(self.train_history_task, f"{self.logger.exp_path}/models/train_history_task_{t}.pth")  # 直接保存对象

            self.experts_distributions.append([])
            self.train_history_task[t].append(t)
            for item in range(class_start, class_end):
                self.train_history[t].append(item)
            # 训练结束后，保存改该task的原型到对应的专家中
            # self.proto_save(t, trn_loader, first_task=True)

        elif t >= self.max_experts + self.offset:
            # 同样的模型理论上应该会得到相同的结果
            bb_to_finetune = self._choose_backbone_to_finetune(t, trn_loader, val_loader)
            if train:
                print(f"Finetuning backbone {bb_to_finetune} on task {t}:")
                self.finetune_backbone(t, bb_to_finetune, trn_loader, val_loader)
                """还是需要保存模型, 以便观察数据"""
                if self.save_models:
                    # torch.save(self.model.state_dict(), f"{self.logger.exp_path}/model_{t}.pth")
                    torch.save(self.model, f"{self.logger.exp_path}/models/model_{t}.pth")  # 直接保存对象
                    # torch.save(self.prototype_dict, f"{self.logger.exp_path}/prototype_dict_{t}.pth")  # 直接保存对象
                    # torch.save(self.train_history, f"{self.logger.exp_path}/train_history_{t}.pth")  # 直接保存对象
                    # torch.save(self.train_history_task, f"{self.logger.exp_path}/train_history_task_{t}.pth")  # 直接保存对象

            self.train_history_task[bb_to_finetune].append(t)
            for item in range(class_start, class_end):
                self.train_history[bb_to_finetune].append(item)
            # 训练结束后，保存改该task的原型到对应的专家中
            # self.proto_save(bb_to_finetune, trn_loader)

        print(f"Creating distributions for task {t}:")
        self.create_distributions(t, trn_loader, val_loader)

    def train_base_process(self, trn_loader, val_loader):
        """训练Base阶段数据，将此模型作为增量任务中每个专家的初始化模型"""
        t = 0
        model = self.model.bb_fun(num_classes=self.model.taskcla[t][1], num_features=self.model.num_features)

        model.fc = nn.Linear(self.model.num_features, self.model.taskcla[t][1])  # 后接分类头, 使用CE损失训练模型

        for param in model.parameters():  # base阶段或第一个task训练整个模型
            param.requires_grad = True

        print(f'The expert has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')
        print(
            f'The expert has {sum(p.numel() for p in model.parameters() if not p.requires_grad):,} shared parameters\n')

        model.to(self.device)
        # 这里是将所有参数一共进行训练了
        optimizer, lr_scheduler = self._get_optimizer(t, self.wd)
        for epoch in range(self.nepochs):
            train_loss, valid_loss = [], []
            train_hits, val_hits = 0, 0
            model.train()
            for images, targets in trn_loader:
                targets -= self.model.task_offset[t]
                bsz = images.shape[0]
                images, targets = images.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                out = model(images)
                # 计算损失值
                loss = self.criterion(t, out, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clipgrad)
                optimizer.step()
                train_hits += float(torch.sum((torch.argmax(out, dim=1) == targets)))
                train_loss.append(float(bsz * loss))
            lr_scheduler.step()

            model.eval()
            with torch.no_grad():
                for images, targets in val_loader:
                    targets -= self.model.task_offset[t]
                    bsz = images.shape[0]
                    images, targets = images.to(self.device), targets.to(self.device)
                    out = model(images)
                    loss = self.criterion(t, out, targets)

                    val_hits += float(torch.sum((torch.argmax(out, dim=1) == targets)))
                    valid_loss.append(float(bsz * loss))

            train_loss = sum(train_loss) / len(trn_loader.dataset)
            valid_loss = sum(valid_loss) / len(val_loader.dataset)
            train_acc = train_hits / len(trn_loader.dataset)
            val_acc = val_hits / len(val_loader.dataset)

            print(f"Epoch: {epoch} Train loss: {train_loss:.2f} Val loss: {valid_loss:.2f} "
                  f"Train acc: {100 * train_acc:.2f} Val acc: {100 * val_acc:.2f}")
        model.fc = nn.Identity()  # 训练完后, 舍弃分类头
        self.model.preTrain_model = model  # 为何需要再次赋值？对象不是引用的吗

    def train_backbone(self, t, trn_loader, val_loader):
        if self.model.preTrain_model is not None:
            # 用Base阶段训练的模型作为专家的初始化参数
            self.model.bbs.append(copy.deepcopy(self.model.preTrain_model))  # 复制之前的旧专家作为新专家
        else:
            if self.initialization_strategy == "random" or t == 0:  # 添加首个专家, 一个resnet32的模型
                self.model.bbs.append(
                    self.model.bb_fun(num_classes=self.model.taskcla[t][1], num_features=self.model.num_features))
            else:
                self.model.bbs.append(copy.deepcopy(self.model.bbs[0]))  # 复制之前的旧专家作为新专家
        model = self.model.bbs[t]

        model.fc = nn.Linear(self.model.num_features, self.model.taskcla[t][1])  # 后接分类头, 使用CE损失训练模型
        if t == 0:
            for param in model.parameters():  # base阶段或第一个task训练整个模型
                param.requires_grad = True
        else:
            for name, param in model.named_parameters():
                param.requires_grad = True
                for layer_not_to_train in self.shared_layers:  # 非base阶段, 固定浅层, 只训练部分深层模型
                    if layer_not_to_train in name:
                        model.get_parameter(name).data = self.model.bbs[0].get_parameter(name).data
                        param.requires_grad = False

        print(f'The expert has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')
        print(
            f'The expert has {sum(p.numel() for p in model.parameters() if not p.requires_grad):,} shared parameters\n')

        model.to(self.device)
        optimizer, lr_scheduler = self._get_optimizer(t, self.wd)
        for epoch in range(self.nepochs):
            train_loss, valid_loss = [], []
            train_hits, val_hits = 0, 0
            model.train()
            for images, targets in trn_loader:
                targets -= self.model.task_offset[t]
                bsz = images.shape[0]
                images, targets = images.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                out = model(images)
                # 计算损失值
                loss = self.criterion(t, out, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clipgrad)
                optimizer.step()
                train_hits += float(torch.sum((torch.argmax(out, dim=1) == targets)))
                train_loss.append(float(bsz * loss))
            lr_scheduler.step()

            model.eval()
            with torch.no_grad():
                for images, targets in val_loader:
                    targets -= self.model.task_offset[t]
                    bsz = images.shape[0]
                    images, targets = images.to(self.device), targets.to(self.device)
                    out = model(images)
                    loss = self.criterion(t, out, targets)

                    val_hits += float(torch.sum((torch.argmax(out, dim=1) == targets)))
                    valid_loss.append(float(bsz * loss))

            train_loss = sum(train_loss) / len(trn_loader.dataset)
            valid_loss = sum(valid_loss) / len(val_loader.dataset)
            train_acc = train_hits / len(trn_loader.dataset)
            val_acc = val_hits / len(val_loader.dataset)

            print(f"Epoch: {epoch} Train loss: {train_loss:.2f} Val loss: {valid_loss:.2f} "
                  f"Train acc: {100 * train_acc:.2f} Val acc: {100 * val_acc:.2f}")
        model.fc = nn.Identity()  # 训练完后, 舍弃分类头
        self.model.bbs[t] = model  # 为何需要再次赋值？对象不是引用的吗

    @torch.no_grad()
    def _choose_backbone_to_finetune(self, t, trn_loader, val_loader):
        self.create_distributions(t, trn_loader, val_loader)  # 先在所有专家中为当前任务创建分布
        expert_overlap = torch.zeros(self.max_experts, device=self.device)  # 记录新任务分布与每个专家已有分布的相似度（KL散度）
        for bb_num in range(self.max_experts):
            classes_in_t = self.model.taskcla[t][1]  # 当前任务的类数
            new_distributions = self.experts_distributions[bb_num][-classes_in_t:]  # 新任务在当前专家的分布
            kl_matrix = torch.zeros((len(new_distributions), len(new_distributions)), device=self.device)  # (10,10),
            for o, old_gauss_ in enumerate(new_distributions):
                # 构建一个多元高斯分布, 一个特征维度就相当于一个变量元
                old_gauss = MultivariateNormal(old_gauss_.mu.data[0][0], covariance_matrix=old_gauss_.var.data[0][0])
                for n, new_gauss_ in enumerate(new_distributions):
                    new_gauss = MultivariateNormal(new_gauss_.mu.data[0][0],
                                                   covariance_matrix=new_gauss_.var.data[0][0])
                    # 计算新任务类间的KL散度
                    kl_matrix[n, o] = torch.distributions.kl_divergence(new_gauss, old_gauss)
            # 计算新任务类间KL散度之和作为选择专家的判断指标
            expert_overlap[bb_num] = torch.mean(kl_matrix)
            self.experts_distributions[bb_num] = self.experts_distributions[bb_num][:-classes_in_t]  # 清除刚刚生成的新任务分布
        print(f"Expert overlap:{expert_overlap}")
        # 选择KL散度最大的专家, 即最能将新任务类辨别出来的专家
        bb_to_finetune = torch.argmax(expert_overlap)
        self.model.task_offset = self.model.task_offset[:-1]  # 清除create_distributions()代表的变化
        return int(bb_to_finetune)  # 返回新任务类分布间KL散度和最大的专家, 即最能将新任务类辨别出来的专家

    def finetune_backbone(self, t, bb_to_finetune, trn_loader, val_loader):

        """想要在微调专家时, 额外用该专家训练过的类原型, 增强原型生成特征与当前任务的特征一同输入分类头, 以期保留旧类的决策边界"""

        # 在训练前先生成新类在专家t中的原型; 应该没必要在训练前为新类保存原型吧
        # self.proto_save(t, trn_loader)

        old_model = copy.deepcopy(self.model.bbs[bb_to_finetune])  # 获取无梯度旧模型用于蒸馏
        for name, param in old_model.named_parameters():
            param.requires_grad = False
        old_model.eval()

        model = self.model.bbs[bb_to_finetune]
        for name, param in model.named_parameters():
            param.requires_grad = True
            for layer_not_to_train in self.shared_layers:
                if layer_not_to_train in name:
                    param.requires_grad = False
        # 专门新建一个分类头求分类损失, 即CE损失
        # model.fc = nn.Linear(self.model.num_features, self.model.taskcla[t][1])

        # 创建可以识别所有以见类的分类头, 方便我们使用原型构建良好的决策边界; 这样的话就不能将target限制在0~class_num中了, 而需要使用真实标签
        model.fc = nn.Linear(self.model.num_features, sum([i for _, i in self.model.taskcla[:t + 1]]))
        model.to(self.device)
        print(f'The expert has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')
        print(
            f'The expert has {sum(p.numel() for p in model.parameters() if not p.requires_grad):,} shared parameters\n')

        # SGD&MultiStepLR(30,60,80)
        optimizer, lr_scheduler = self._get_optimizer(bb_to_finetune, wd=self.ftwd, milestones=[30, 60, 80])
        for epoch in range(self.ftepochs):
            train_loss, valid_loss = [], []
            train_hits, val_hits = 0, 0
            model.train()
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
            for images, targets in trn_loader:
                # targets -= self.model.task_offset[t]  # 将下标从0开始统计, 以方便计算CE损失; 不进行修改, 为配合原型增强使用
                bsz = images.shape[0]  # B
                images, targets = images.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                with torch.no_grad():
                    # 获取新数据在旧模型的特征, 用于蒸馏
                    old_features = old_model(images)  # resnet with fc as identity returns features by default

                # 获取经分类头得到的预测结果, 特征
                out, features = model(images, return_features=True)
                # 计算损失
                loss = self.criterion(t, out, targets, features, old_features, bb_to_finetune, model)
                # 因为分类头只预测与新类相关的出书，故
                loss.backward()
                # 梯度裁剪函数, 防止梯度爆炸; clipgrad=1参数可以理解为梯度（默认是L2 范数）范数的最大阈值
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clipgrad)
                optimizer.step()
                train_hits += float(torch.sum((torch.argmax(out, dim=1) == targets)))  # 用于计算训练准确率
                train_loss.append(float(bsz * loss))  # 记录损失值

            lr_scheduler.step()

            model.eval()
            with torch.no_grad():
                for images, targets in val_loader:
                    # targets -= self.model.task_offset[t]
                    bsz = images.shape[0]
                    images, targets = images.to(self.device), targets.to(self.device)
                    with torch.no_grad():
                        old_features = old_model(images)  # resnet with fc as identity returns features by default
                    out, features = model(images, return_features=True)
                    # 也要计算测试集的损失, 以便调整学习率等
                    loss = self.criterion(t, out, targets, features, old_features, bb_to_finetune, model)

                    # 计算测试集准确率
                    val_hits += float(torch.sum((torch.argmax(out, dim=1) == targets)))
                    valid_loss.append(float(bsz * loss))

            train_loss = sum(train_loss) / len(trn_loader.dataset)
            valid_loss = sum(valid_loss) / len(val_loader.dataset)
            train_acc = train_hits / len(trn_loader.dataset)
            val_acc = val_hits / len(val_loader.dataset)

            print(f"Epoch: {epoch} Train loss: {train_loss:.2f} Val loss: {valid_loss:.2f} "
                  f"Train acc: {100 * train_acc:.2f} Val acc: {100 * val_acc:.2f}")

        # 重置分类头
        model.fc = nn.Identity()
        self.model.bbs[bb_to_finetune] = model

        # torch.save(self.model.state_dict(), f"{self.logger.exp_path}/model_{t}.pth")
        return old_model

    @torch.no_grad()
    def create_distributions(self, t, trn_loader, val_loader):
        """ Create distributions for task t"""
        self.model.eval()
        classes = self.model.taskcla[t][1]  # 获取当前任务的类别数
        self.model.task_offset.append(self.model.task_offset[-1] + classes)  # [0,10,20,....,100]
        transforms = val_loader.dataset.transform
        for bb_num in range(min(self.max_experts, t + 1)):  # 每个专家都将生成关于新task若干类的高斯分布
            eps = 1e-8
            model = self.model.bbs[bb_num]
            for c in range(classes):
                c = c + self.model.task_offset[t]  # 获取真实下标(0,10)-->(20,30)
                train_indices = torch.tensor(trn_loader.dataset.labels) == c  # 筛选出类c的样本
                if isinstance(trn_loader.dataset.images, list):
                    train_images = list(compress(trn_loader.dataset.images, train_indices))
                    ds = ClassDirectoryDataset(train_images, transforms)
                else:
                    ds = trn_loader.dataset.images[train_indices]  # 获取类c的所有样本
                    ds = ClassMemoryDataset(ds, transforms)  # 将样本构成数据集, 带transforms的
                loader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=trn_loader.num_workers,
                                                     shuffle=False)  # 构建dataloader
                from_ = 0  # 记录数据的下标, 类似index
                class_features = torch.full((2 * len(ds), self.model.num_features), fill_value=-999999999.0,
                                            device=self.model.device)  # (class_num*2,64)
                for images in loader:
                    bsz = images.shape[0]
                    images = images.to(self.device)
                    features = model(images)
                    class_features[from_: from_ + bsz] = features
                    features = model(torch.flip(images, dims=(3,)))  # 额外计算翻转后的特征
                    class_features[from_ + bsz: from_ + 2 * bsz] = features
                    from_ += 2 * bsz

                # Calculate distributions
                cov_type = "full" if self.use_multivariate else "diag"  # "full", 多变量高斯分布？
                is_ok = False
                while not is_ok:
                    try:  # 为专家bb_num生成对应的多类高斯分布
                        gmm = GaussianMixture(self.gmms, class_features.shape[1], covariance_type=cov_type, eps=eps).to(
                            self.device)
                        gmm.fit(class_features, delta=1e-3, n_iter=100)  # 用高斯分布拟合特征
                    except RuntimeError:
                        eps = 10 * eps
                        print(
                            f"WARNING: Covariance matrix is singular. Increasing eps to: {eps:.7f} but this may hurt results")
                    else:
                        is_ok = True  # 拟合成功就可以, 失败则增加eps值

                if len(gmm.mu.data.shape) == 2:
                    gmm.mu.data = gmm.mu.data.unsqueeze(1)
                self.experts_distributions[bb_num].append(gmm)  # 载入一个类的高斯分布; mu:(1,1,64),var:(1,1,64,64)

    @torch.no_grad()
    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
        self.model.eval()
        for images, targets in val_loader:
            targets = targets.to(self.device)
            # Forward current model
            features = self.model(images.to(self.device))
            # 计算出TIL和CIL的准确率
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
        taw_pred, tag_pred = self.predict_class_bayes(t, features)  # 获取TIL和CIL的预测结果
        hits_taw = (taw_pred == targets).float()
        hits_tag = (tag_pred == targets).float()  # 计算准确率
        return hits_taw, hits_tag

    @torch.no_grad()
    def predict_class_bayes(self, t, features):
        log_probs = torch.full((features.shape[0], len(self.experts_distributions), len(self.experts_distributions[0])),
                               fill_value=-1e8, device=features.device)  # 记录关于各个类的预测概率, (B,k,num_class)
        mask = torch.full_like(log_probs, fill_value=False, dtype=torch.bool)  # (B,k,num_class)
        for bb_num, _ in enumerate(self.experts_distributions):
            for c, class_gmm in enumerate(self.experts_distributions[bb_num]):  # 遍历每个专家的每个类分布
                c += self.model.task_offset[bb_num]
                log_probs[:, bb_num, c] = class_gmm.score_samples(features[:, bb_num])  # 计算特征与每个分布的对数似然度
                mask[:, bb_num, c] = True  # 没明白这个的作用, 统计有效的专家数, 以计算所有专家的平均置信度

        from_ = self.model.task_offset[t]
        to_ = self.model.task_offset[t + 1]

        # Task-Aware
        taw_log_probs = log_probs[:, :t + 1, from_:to_].clone()  # 已知task-id, 则预测范围将限定在这个Task所属的类中
        taw_log_probs = softmax_temperature(taw_log_probs, dim=2, tau=self.tau)
        confidences = torch.sum(taw_log_probs, dim=1)  # 计算所有专家的预测概率只和
        taw_class_id = torch.argmax(confidences, dim=1) + self.model.task_offset[t]  # 在已经task-id的情况下进行预测的结果
        # Task-Agnostic
        log_probs = softmax_temperature(log_probs, dim=2, tau=self.tau)  # 未知task-id的情况下, 在所有类中进行分类
        confidences = torch.sum(log_probs, dim=1) / torch.sum(mask, dim=1)  # 求所有专家的预测概率平均值
        tag_class_id = torch.argmax(confidences, dim=1)  # 求出未知task-id时的预测结果
        return taw_class_id, tag_class_id

    def criterion(self, t, outputs, targets, features=None, old_features=None, model_id=-1, model=None):
        """Returns the loss value"""
        # LCD
        ce_loss = nn.functional.cross_entropy(outputs, targets, label_smoothing=0.0)
        # ce_loss = nn.functional.cross_entropy(outputs / self.temp, targets, label_smoothing=0.0)
        if old_features is not None:  # Knowledge distillation loss on features
            # LKD
            kd_loss = nn.functional.mse_loss(features, old_features)
            total_loss = (1 - self.alpha) * ce_loss + self.alpha * kd_loss
            if self.protoAug_weight == 0:
                return total_loss

            # LPA, 仅在微调阶段进行
            proto_aug = []
            proto_aug_label = []
            # 获取当前专家所训练过的所有类; 目前保持的都是任务id而非类id，需要调整
            index = self.train_history[model_id]
            for _ in range(self.batch_size):
                # 随机选择一个旧样本进行原型增强
                np.random.shuffle(index)
                # 使用高斯噪声随机增强一个旧类的原型
                temp = self.prototype_dict[model_id][index[0]] + np.random.normal(0, 1, 64) * self.radius
                proto_aug.append(temp)
                proto_aug_label.append(index[0])

            proto_aug = torch.from_numpy(np.float32(np.asarray(proto_aug))).float().to(self.device)
            proto_aug_label = torch.from_numpy(np.asarray(proto_aug_label)).to(self.device)
            # 得用全连接层
            soft_feat_aug = model.fc(proto_aug)
            loss_protoAug = nn.CrossEntropyLoss()(soft_feat_aug / self.temp, proto_aug_label)
            total_loss += loss_protoAug * self.protoAug_weight
            return total_loss

        return ce_loss

    def _get_optimizer(self, num, wd, milestones=[60, 120, 160]):
        """Returns the optimizer"""

        # 获取requires_grad=True的参数, 实现真正的冻结浅层
        train_params = []
        shared_params = []
        for k, v in self.model.bbs[num].named_parameters():
            if v.requires_grad:
                train_params.append(v)
                continue
            for layer_not_to_train in self.shared_layers:
                if layer_not_to_train in k:
                    shared_params.append(v)
        # lr: 0.05; weight_decay: 5e-4
        train_params = {'params': train_params, 'lr': self.lr, 'weight_decay': wd}
        # lr: 0.005; weight_decay: 2e-4
        shared_params = {'params': shared_params, 'lr': self.lr * 0.1, 'weight_decay': 2e-4}

        # total_params = [train_params]
        total_params = [train_params, shared_params]

        optimizer = torch.optim.SGD(total_params, lr=self.lr, weight_decay=wd, momentum=0.9)
        # optimizer = torch.optim.SGD(self.model.bbs[num].parameters(), lr=self.lr, weight_decay=wd, momentum=0.9)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)
        return optimizer, scheduler

    def proto_save(self, model_id, loader, first_task=False):
        """为任务t保存在专家t中的原型"""
        print("Save prototype in expert {}".format(model_id))
        features = []
        labels = []
        model = self.model.bbs[model_id]

        model.eval()
        with torch.no_grad():
            for i, (images, target) in enumerate(loader):
                images, target = images.to(self.device), target.to(self.device)
                # 将原图分别旋转0, 90, 180, 270, 得到四张不同标签的图像
                # images = torch.stack([torch.rot90(images, k, (2, 3)) for k in range(4)], 1)
                # (4*B,3,32,32)
                images = images.view(-1, 3, 32, 32)
                # 将标签下标*4，为不同旋转的原图对应的标签
                # target = torch.stack([target * 4 + k for k in range(4)], 1).view(-1)

                # (4*B,512)
                feature = model.extract_feature(images)
                # 舍弃最后无法构成一个batch的数据
                # if feature.shape[0] == self.args.batch_size * 4:

                if feature.shape[0] == self.batch_size:
                    # 记录标签和特征，保存在数组中
                    labels.append(target.cpu().numpy())
                    features.append(feature.cpu().numpy())
        # 将训练集所有图像的标签和特征合并在一起：labels(Dt*4,),feature(Dt*4,512)
        labels_set = np.unique(labels)
        labels = np.array(labels)
        labels = np.reshape(labels, labels.shape[0] * labels.shape[1])
        features = np.array(features)
        features = np.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))
        feature_dim = features.shape[1]

        radius = []

        for item in labels_set:
            # 每个旋转的类都视为一个单独的类，并保存对应的原型
            index = np.where(item == labels)[0]  # 所有item类的下标
            feature_classwise = features[index]  # 所有item类的特征: [500,512]
            self.prototype_dict[model_id][item] = np.mean(feature_classwise, axis=0)  # 计算所有item类数据的均值作为类均值:[512]

            # 记录每个专家所训练过的类
            self.train_history[model_id].append(item)

            if first_task:
                # base阶段确定原型的半径, 即确定高斯噪音的增强度
                cov = np.cov(feature_classwise.T)
                radius.append(np.trace(cov) / feature_dim)
        if first_task:
            self.radius = np.sqrt(np.mean(radius))
            print("radius: ", self.radius)

    def eval_within_expert(self, tst_loader, max_tasks, task):
        """计算每个专家关于已训练任务的准确率, 以便了解每个专家内的稳定性和可塑性情况, 并根据这一情况调整每个专家的训练策略"""
        transforms = tst_loader[0].dataset.transform
        # 记录每个专家关于每个已训练类的准确率
        expert_acc = np.zeros((self.max_experts, max_tasks))
        for e_id in range(self.max_experts):
            if task not in self.train_history_task[e_id]:
                continue
            # 先从tst_loader中筛选出专家e_id训练过的数据
            for t_id in self.train_history_task[e_id]:
                # 获取任务t_id的测试集
                sub_loader = tst_loader[t_id]

                # 获取任务t_id所包含的类
                class_end = sum([i for _, i in self.model.taskcla[:t_id + 1]])
                class_start = class_end - self.model.taskcla[t_id][1]

                total_num = 0
                total_acc_tag = 0

                # 筛选任务t_id所包含的类的数据
                for c_id in range(class_start, class_end):
                    train_indices = torch.tensor(sub_loader.dataset.labels) == c_id  # 筛选出类c的样本
                    if isinstance(sub_loader.dataset.images, list):
                        train_images = list(compress(sub_loader.dataset.images, train_indices))
                        ds = ClassDirectoryDataset(train_images, transforms)
                    else:
                        ds = sub_loader.dataset.images[train_indices]  # 获取类c的所有样本
                        ds = ClassMemoryDataset(ds, transforms)  # 将样本构成数据集, 带transforms的
                    # 获取了专家e_id所训练过的任务t_id中包含的类c_id的数据
                    loader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=sub_loader.num_workers,
                                                         shuffle=False)  # 构建dataloader
                    for images in loader:
                        bsz = images.shape[0]
                        total_num += bsz

                        features = self.model(images.to(self.device))
                        # 获取当前专家得到的特征
                        features = features[:, e_id]

                        # 记录关于各个类的预测概率, (B,k,num_class)
                        log_probs = torch.full((features.shape[0], 1, len(self.experts_distributions[0])),
                                               fill_value=-1e8, device=features.device)
                        # mask = torch.full_like(log_probs, fill_value=False, dtype=torch.bool)  # (B,k,num_class)
                        for c, class_gmm in enumerate(self.experts_distributions[e_id]):
                            c += self.model.task_offset[e_id]
                            # if c in range(class_start, class_end):
                            if c in self.train_history[e_id]:
                                # 只选择当前任务相关的类, 只计算任务t_id内每个类的预测概率
                                log_probs[:, 0, c] = class_gmm.score_samples(features)
                                # mask[:, 0, c] = True  # 没明白这个的作用

                        # 计算每个专家内部的准确率
                        log_probs = softmax_temperature(log_probs, dim=2, tau=self.tau)  # 未知task-id的情况下, 在所有类中进行分类
                        confidences = torch.sum(log_probs, dim=1)  # 求所有专家的预测概率平均值, 此处实际上只统计一个专家的预测概率
                        tag_class_id = torch.argmax(confidences, dim=1)  # 求出未知task-id时的预测结果
                        hits = (tag_class_id == c_id).float()
                        total_acc_tag += hits.sum().item()
                if total_num == 0:
                    expert_acc[e_id][t_id] = 0
                else:
                    # 记录专家e_id关于已训练过的任务t_id的准确率
                    expert_acc[e_id][t_id] = total_acc_tag / total_num
        return expert_acc
