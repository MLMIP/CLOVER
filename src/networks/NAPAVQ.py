import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np

import os
import copy
from collections import Counter


class network(nn.Module):
    def __init__(self, numclass, feature_extractor):
        super(network, self).__init__()
        # Fθ
        self.feature = feature_extractor
        # 为什么需要全连接层作为分类头？分类方式不都是基于特征表征z与cv的距离来进行分类的吗？
        self.fc = nn.Linear(512, numclass, bias=True)  # Liner(512,50*4)

    def forward(self, input):
        x = self.feature(input)
        x = self.fc(x)
        return x

    def Incremental_learning(self, numclass):
        # 增量扩展全连接层fc，为新类增加新权重
        weight = self.fc.weight.data
        bias = self.fc.bias.data
        in_feature = self.fc.in_features
        out_feature = self.fc.out_features

        self.fc = nn.Linear(in_feature, numclass, bias=True)
        self.fc.weight.data[:out_feature] = weight[:out_feature]
        self.fc.bias.data[:out_feature] = bias[:out_feature]

    def feature_extractor(self, inputs):
        return self.feature(inputs)


class NAVQ(nn.Module):

    def __init__(self, num_classes=100, feat_dim=2, device=None, with_grow=1):
        super(NAVQ, self).__init__()
        self.num_classes = num_classes  # 50, 55, ..., 100
        self.feat_dim = feat_dim  # 512
        self.device = device
        self.cvs = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))  # (num_classes,512)
        self.edges = torch.zeros([self.num_classes, self.num_classes]).to(self.device)  # (num_classes,num_classes)
        self.edges.fill_diagonal_(1)  # 主对角线元素赋值为1
        self.print_edges_counter = 0  # 记录边总数

        # {i:j,...},将cv的下标i与对应的类j构成字典
        self.cv_class = dict(zip(range(self.num_classes), [i for i in range(self.num_classes)]))
        self.class_indices = [[i] for i in range(self.num_classes)]  # 将原本的类顺序变化为[0,1,...,99]
        self.num_cvs = self.num_classes

        # CV之间的连通性，根据边的值来判断的？
        self.cv_connectedness = torch.zeros([self.num_classes, self.num_classes]).to(self.device)
        self.cv_connectedness.fill_diagonal_(1)

    def add_cvs(self, num_classes_to_add):
        # [10*4,512]
        new_cvs = torch.randn(num_classes_to_add, self.feat_dim, device=self.device)
        # [old+10*4,512]
        self.cvs = nn.Parameter(
            torch.cat((self.cvs,
                       new_cvs)).to(
                self.device))

        for i in range(num_classes_to_add):
            # 将cv的下标与类下标用字典关联
            self.cv_class.update({self.num_cvs + i: self.num_classes + i})
            self.class_indices.append([self.num_cvs + i])

        self.num_classes += num_classes_to_add
        self.num_cvs += num_classes_to_add

        # 之更新新CV（可是好像没有这个属性呀）
        self.optimizer.add_param_group({"params": new_cvs})

        # 扩展边
        edges_new = torch.zeros([self.num_cvs, self.num_cvs])
        edges_new[:self.num_cvs - num_classes_to_add, :self.num_cvs - num_classes_to_add] = self.edges
        self.edges = edges_new.to(self.device)
        # 对角线填1
        self.edges.fill_diagonal_(1)

        # 扩展表示连接关系的数组
        cv_connectedness_new = torch.zeros([self.num_cvs, self.num_cvs])
        cv_connectedness_new[:self.num_cvs - num_classes_to_add,
        :self.num_cvs - num_classes_to_add] = self.cv_connectedness
        self.cv_connectedness = cv_connectedness_new.to(self.device)
        self.cv_connectedness.fill_diagonal_(1)

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        # TODO move k,epsilon
        self.k = 15

        # The edge-strength decay constant. Set at a small value for sparse graph generation. For dense
        # graphs, use a higher value (close to 1.0 but less than 1.0).

        epsilon = 0.9
        e_min = 0.9 ** 10
        self.print_edges_counter += 1

        # 计算与各CV间的距离(B*4,old_class_num*4)
        dist_to_cvs = torch.cdist(x, self.cvs)
        # 按距离升序排序
        sorted_dists, ordered_cvs = torch.sort(dist_to_cvs, dim=1)

        # correct_cvs = torch.tensor(labels)
        correct_cvs = labels.clone().detach()

        # 选择最近的前k个cv
        kth_closest_cvs = ordered_cvs[:, :self.k]

        # # ------------------------------------------------------------------------------------------------------------

        # 距离最近的CV: (B*4,)
        closest_cvs_list = kth_closest_cvs[:, 0].tolist()
        # counting the number of times the closest cvs appear in the input, encounters is the number of time by
        # which we multiply the edge strength in the input by epsilon in the non optimised code

        visited_node_encounters = torch.zeros(self.num_cvs, device=self.device)
        # 统计计数器？可以统计出该batch所有样本的预测结果的频率
        visits_counter = Counter(closest_cvs_list)
        for i in range(self.num_cvs):
            if i in visits_counter:
                visited_node_encounters[i] = visits_counter[i]
        encounters = (torch.ones(self.num_cvs, self.num_cvs, device=self.device) * visited_node_encounters).T

        # closest cvs tensor gives the number of times each cv was considered as a closest cv of another cv.
        # this is the condition where we set the edge strength to 1 in the non-optimised code
        closest_cv_encounters = torch.zeros(self.num_cvs, self.num_cvs, device=self.device)
        for i, i_k in enumerate(kth_closest_cvs.tolist()):
            for cv in i_k:
                closest_cv_encounters[closest_cvs_list[i], cv] += 1

        epsilon = epsilon * torch.ones(self.num_cvs, self.num_cvs, device=self.device)

        # this array is used to set the start of the edge strength to either
        #   1 (if the edge is considered as a closest neighbour connection in this iteration) or
        #   current value (if the edge is not considered as closest neighbour connection in this iteration)
        #   0 (if there has been no updates on this edge so far).

        # this identify whether its an edge that should updated in this iteration
        closest_cvs_existence = torch.gt(closest_cv_encounters, 0)

        # recalculate encounters by subtracting the closest cv occurrences to identify the number by which the edge
        # strength should be multiplied by epsilon
        # here we ignore the order of the operations and assume
        # if encounters> closest_cv_encounters:
        #   multiply edge strength  by epsilon**(encounters-closest_cv_encounters) times
        # else
        #   edge strength=1
        encounters = (encounters - closest_cv_encounters) * (torch.gt((encounters - closest_cv_encounters), 0))

        # if first_time: self.edges = closest_cvs_existence since we're setting the edge strength to 1 at
        # closest_cvs_existence we dont need to update the value to 1 again when encounters<closest cv
        # encounters
        self.edges = torch.max(self.edges, closest_cvs_existence)
        self.edges = self.edges * (epsilon ** encounters)
        self.edges = self.edges * (1 - (self.edges < e_min) * 1)

        scale_factor = torch.diagonal(
            torch.pow(torch.cdist(x, torch.index_select(self.cvs, 0, correct_cvs)), 2), 0)

        x_cvs = torch.pow(dist_to_cvs, 2)

        d_pos = scale_factor

        d_neg = (x_cvs * torch.index_select(
            torch.logical_and((self.edges > 0), self.cv_connectedness < 1),
            0,
            correct_cvs))

        exp_d_neg_neighbours = torch.exp(-0.001 * d_neg) * (d_neg > 0)  # d_neg>0 to remove non neighbours
        w_d_neg = exp_d_neg_neighbours / exp_d_neg_neighbours.sum(keepdim=True, dim=1)
        w_d_neg[w_d_neg != w_d_neg] = 0  # removing nan
        w_d_neg = w_d_neg.detach()

        mu = (d_pos - (w_d_neg * d_neg).sum(dim=1))
        loss = (nn.ReLU()(mu)).sum() / x.size(0)

        self.edges = (self.edges + self.edges.T) / 2

        return loss


def _get_dist_each_class(feature, navq):
    features = feature.unsqueeze(1)
    cvs = navq.cvs.unsqueeze(0).repeat(feature.size(0), 1, 1)
    dist = torch.cdist(features, cvs).squeeze(1)

    return -dist


class NAPAVQ:
    def __init__(self, feature_extractor, taskcla, init_class_num, perTask_class_num, num_features, batch_size, device):
        # self.file_name = file_name  # cifar100_40_20*3_cifar_100-20
        # self.args = args
        self.batch_size = batch_size
        self.kd_weight = 10.0
        self.protoAug_weight = 10.0
        self.temp = 0.1
        self.epochs = 100  # 100
        self.learning_rate = 0.001  # 0.001
        self.model = network(init_class_num, feature_extractor)  # feature_extractor + classifier; *4是采用了SSL
        # self.model = network(init_class_num * 4, feature_extractor)  # feature_extractor + classifier; *4是采用了SSL
        self.radius = 0
        self.init_class_num = init_class_num
        self.num_class = init_class_num  # 50, 目前可见类数
        self.task_size = perTask_class_num  # 5, 每个任务所含类数
        self.device = device
        self.old_model = None

        # 数据集在训练时指定
        # self.data_composer = DataComposer(args.data_name, args.shuffle, args.seed)
        # self.train_dataset = self.data_composer.get_train_dataset()
        # self.test_dataset = self.data_composer.get_test_dataset()
        self.train_loader = None
        self.test_loader = None

        # 因为专家0不是按顺序训练所有数据, 需要偏置将预测结果转化为相应结果
        self.trained_task = []

        # self.old_dataset = self.data_composer.get_test_dataset()
        # self.new_dataset = self.data_composer.get_test_dataset()
        # self.old_test_loader = None
        # self.new_test_loader = None

        # CV
        self.navq = NAVQ(
            num_classes=self.num_class,
            feat_dim=num_features,
            device=device,
        )
        # 记录旧样本的特征表征原型, 用于生成式reheasal
        self.prototype_dict = {}

        self.steps = -1  # -1/0

    def extract_feature(self, images):
        self.model.eval()
        with torch.no_grad():
            features = self.model.feature_extractor(images)
        return features

    def targets_transfer(self, target, current_task, raw=False):
        """因为每个专家训练的任务不一定连续，故需要将targets转化为连续的值，或将模型的预测结果转化为原始值"""
        if raw:
            # 因为任务是一个个来的，所以只需要考虑将当期的原石标签转化为适合模型的标签
            # 计算当前任务的本专家的训练的第几个任务
            task_order = self.trained_task.index(current_task)
            # 获取真实标签和我们所需的连续标签间的偏置值
            offset = (current_task - task_order) * self.task_size
            # 转化为连续的值
            target -= torch.tensor([offset for _ in range(len(target))])
        else:

            # 记录标签在整体上的任务id和当期模型上的id
            raw_orders = []
            model_orders = []
            for i in range(target):
                if i < self.init_class_num:
                    model_orders.append(0)
                    raw_orders.append(0)
                    continue
                i -= self.init_class_num
                model_order = i // self.task_size + 1
                model_orders.append(model_order)
                raw_orders.append(self.trained_task.index(model_order))
            #
            target += torch.tensor([(raw_orders[i] - model_orders[i]) * self.task_size for i in range(len(target))])

        return target

    def _before_update(self, current_task):

        # setting the gradients of old coding vectors to be 0
        if current_task > 0:
            # 通过将旧类CV的梯度归零，实现冻结旧CV的效果
            # classes_old = range((self.num_class - self.task_size) * 4)
            classes_old = range(self.num_class - self.task_size)
            self.navq.cvs.grad[classes_old, :] *= 0

    def before_train(self, current_task, train_loader, test_loader):
        self.model.eval()
        # 获取当前task相关的数据loader
        self.trained_task.append(current_task)

        # self.train_loader, self.test_loader = self._get_train_and_test_dataloader(classes)
        # 直接在训练前指定loader
        self.train_loader, self.test_loader = train_loader, test_loader

        if current_task > 0:
            # 更新扩展全连接层
            # self.model.Incremental_learning(4 * self.num_class)
            self.model.Incremental_learning(self.num_class)

            self.model.to(self.device)
            # 为什么是在新任务训练前，用新任务的数据保存原型？
            self.proto_save(self.model, self.train_loader, current_task)
            # 新增随机初始化的CV
            # self.navq.add_cvs(self.task_size * 4)
            self.navq.add_cvs(self.task_size)

        self.model.train()
        self.model.to(self.device)

    def train(self, current_task, old_class=0, epochs=100):
        if current_task == 0:
            # base阶段采用较大的学习率(10倍于增量阶段)
            # 分别确定模型和CV的optimizer
            self.opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=2e-4)
            self.optimizer_cvs = optim.SGD(self.navq.parameters(), lr=5)
        else:
            self.opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate * 0.1, weight_decay=2e-4)
            self.optimizer_cvs = optim.SGD(self.navq.parameters(), lr=5)

        # 为何要保存一份？
        self.navq.optimizer = self.optimizer_cvs

        # 学习率变化设置: 以固定间隔成倍衰减初始学习率
        scheduler = StepLR(self.opt, step_size=45, gamma=0.1)
        scheduler_sng = StepLR(self.optimizer_cvs, step_size=20, gamma=0.1)

        for epoch in range(epochs):
            total_loss, total_pass_kd, total_dce, total_na = 0, 0, 0, 0

            for images, target in self.train_loader:
                if self.steps != -1:
                    self.steps += 1
                    if self.steps == 2:
                        self.steps = 0
                        break

                target = self.targets_transfer(target, current_task, raw=True)
                images, target = images.to(self.device), target.to(self.device)

                # self-supervised learning based label augmentation similar to PASS
                # images = torch.stack([torch.rot90(images, k, (2, 3)) for k in range(4)], 1)
                # images = images.view(-1, 3, 32, 32)
                # target = torch.stack([target * 4 + k for k in range(4)], 1).view(-1)

                self.opt.zero_grad()
                self.optimizer_cvs.zero_grad()

                # 直接一步到位计算损失值
                loss_dce, loss_pass_kd, loss_na = self._compute_loss(images, target, old_class)
                loss = loss_dce + loss_pass_kd + loss_na

                # 统计此epoch中所有数据的Ldce、Lna和Ltotal的损失
                total_dce += loss_dce.item()
                if loss_pass_kd == 0:
                    total_pass_kd += 0
                else:
                    total_pass_kd += loss_pass_kd.item()
                total_na += loss_na.item()
                total_loss += loss.item()

                # 损失值反向传播得到梯度
                loss.backward()
                # 冻结旧CV参数, 不更新旧CV
                self._before_update(current_task)
                # 更新模型和CV
                self.opt.step()
                self.optimizer_cvs.step()

            # 学习率更新
            scheduler.step()
            scheduler_sng.step()

            # 记录训练集准确率, 便于观察
            accuracy_ncm_train = self._test(self.train_loader, current_task)
            accuracy_ncm_test = self._test(self.test_loader, current_task)

            total_dce /= len(self.train_loader)
            total_pass_kd /= len(self.train_loader)
            total_na /= len(self.train_loader)
            total_loss /= len(self.train_loader)

            print(f"Epoch: {epoch} Train loss: {total_loss:.2f} Train DCELoss: {total_dce:.2f} "
                  f"Train PASS&KDloss: {total_pass_kd:.2f} Train NAloss: {total_na:.2f} "
                  f"Train acc: {accuracy_ncm_train:.2f} Val acc: {accuracy_ncm_test:.2f}")

    def _test(self, testloader, current_task):
        self.model.eval()
        total = 0.0
        correct_ncm = 0.0

        task_order = self.trained_task.index(current_task)
        # 获取真实标签和我们所需的连续标签间的偏置值
        offset = task_order * self.task_size
        total_class_num = self.init_class_num + offset

        for imgs, labels in testloader:
            if self.steps != -1:
                self.steps += 1
                if self.steps == 2:
                    self.steps = 0
                    break

            imgs, labels = imgs.to(self.device), labels.to(self.device)

            with torch.no_grad():
                features = self.model.feature(imgs)
                features_norm = (features.T / torch.norm(features.T, dim=0)).T

            total += len(labels)

            cvs_copy = self.navq.cvs.detach().clone()
            cvs_norm = (cvs_copy.T / torch.norm(cvs_copy.T, dim=0)).T.to(self.device)

            selected_class_indices = [self.navq.class_indices[i] for i in range(0, total_class_num)]
            selected_class_indices_flat = [item for sublist in selected_class_indices for item in sublist]

            filtered_cvs = torch.index_select(cvs_norm, 0,
                                              torch.tensor(selected_class_indices_flat).to(self.device))
            filtered_targets = [self.navq.cv_class[i] for i in selected_class_indices_flat]

            result = []
            for target in features_norm.cpu().numpy():
                x = target - filtered_cvs.cpu().numpy()
                x = np.linalg.norm(x, ord=2, axis=1)
                x = np.argmin(x)
                result.append(filtered_targets[x])

            predicts_ncm = torch.tensor(result)

            # 将连续的预测值转化回原始标签
            predicts_ncm = self.targets_transfer(predicts_ncm.cpu(), current_task, raw=False)
            correct_ncm += (predicts_ncm == labels.cpu()).sum()

        accuracy_ncm = correct_ncm.item() / total
        self.model.train()

        return accuracy_ncm

    def _compute_loss(self, imgs, target, old_class=0):
        # 特征表征Z
        feature = self.model.feature(imgs)
        # 计算Z与各CV间的欧氏距离
        output = _get_dist_each_class(feature, self.navq)
        output, target = output.to(self.device), target.to(self.device)

        # 计算Ldce, 用于拉近Z与对应CV的距离; self.args.temp=0.1
        loss_dce = nn.CrossEntropyLoss()(output / self.temp, target)
        # 计算Lna, 用于拉远Z与其他混淆类的CV的距离
        loss_na = self.navq(feature, target)

        if self.old_model is None:
            # base阶段的损失
            return loss_dce, 0, loss_na
        else:
            # 计算新数据在旧模型上的特征, 使用蒸馏的方式指导新模型训练
            feature_old = self.old_model.feature(imgs)
            loss_kd = torch.dist(feature, feature_old, 2)

            index = np.arange(old_class)

            # code for NA-PA, 由于旧类CV不更新，复习旧类原型实际上是让新类CV远离旧类CV，以保持旧类CV不与新类重叠
            # [4*64]; 随机在旧类中选择batch_size个旧类，可重复选择同一类多次, *4是因为SSL扩展了数据量导致的
            # random_indices = index[np.random.choice(len(index), size=self.batch_size, replace=True)] * 4
            random_indices = index[np.random.choice(len(index), size=self.batch_size, replace=True)]

            # 获取选中类对应的均值原型, [B]
            proto_list = [self.prototype_dict[i] for i in random_indices]
            proto_array = np.array(proto_list)  # [B,64]
            # 获取类对应的邻域类下标, [1, B, 55]
            proto_neighbours = self.navq.edges.cpu().numpy()[[random_indices]]
            # 为每个类随机选择邻居类中的一个, [1]?应该是[8]，一个样本对应一个才对
            picked_neighbour_indices = np.array([np.random.choice(r.nonzero()[0]) for r in proto_neighbours])
            # 获取其中一个邻居类的均值原型, [1,64]
            picked_neighbours = np.array([self.prototype_dict[i] for i in picked_neighbour_indices])
            # 邻域类在增强原型中所在比例α的分布
            gammas = np.random.uniform(0.5, 1, self.batch_size)
            # 使用全部原型进行增强的方法
            # gammas = np.random.uniform(0.5, 1, len(index))
            # 利用邻域原型增强类原型
            proto_aug = proto_array * gammas[:, None] + picked_neighbours * (1 - gammas)[:, None]

            # 载入GPU
            proto_aug = torch.tensor(proto_aug, dtype=torch.float).to(self.device)
            proto_aug_label = torch.from_numpy(random_indices).to(self.device)

            # 用增强的旧原型作为特征z计算Lna（训练新样本与旧样本的数量相同，每训练一个新任务batch，则训练同样一个batch的旧样本）
            loss_na += self.navq(proto_aug, proto_aug_label)

            # 计算Ldce
            soft_feat_aug = _get_dist_each_class(proto_aug, self.navq)
            loss_protoAug = nn.CrossEntropyLoss()(soft_feat_aug / self.temp, proto_aug_label)

            return loss_dce, self.protoAug_weight * loss_protoAug + self.kd_weight * loss_kd, loss_na

    def after_train(self, current_task):
        # 训练前后为什么都保存一次原型？
        self.proto_save(self.model, self.train_loader, current_task)

        # 基于Z与CV的距离进行分类的准确率
        accuracy_ncm = self._test(self.test_loader, current_task)
        print('Final, accuracy_ncm:%.5f' % (accuracy_ncm,))

        # if current_task > 0:
        #     old_accuracy_ncm = self._test(self.old_test_loader, current_task)
        #     new_accuracy_ncm = self._test(self.new_test_loader, current_task)
        #     print('Final, old_accuracy_ncm:%.5f' % (
        #         old_accuracy_ncm,))
        #     print('Final, new_accuracy_ncm:%.5f' % (
        #         new_accuracy_ncm,))

        # 每个任务的类范围：[num_class-task_size, num_class)
        self.num_class += self.task_size

        # 方便蒸馏使用
        self.navq.old_class_indices = copy.deepcopy(self.navq.class_indices)

        self.old_model = copy.deepcopy(self.model)
        self.old_model.to(self.device)
        self.old_model.eval()

        self.old_navq = copy.deepcopy(self.navq)
        self.old_navq.to(self.device)
        self.old_navq.eval()

    def proto_save(self, model, loader, current_task):
        features = []
        labels = []
        model.eval()
        with torch.no_grad():
            for images, target in loader:
                target = self.targets_transfer(target, current_task, raw=True)

                images, target = images.to(self.device), target.to(self.device)
                # 将原图分别旋转0, 90, 180, 270, 得到四张不同标签的图像
                # images = torch.stack([torch.rot90(images, k, (2, 3)) for k in range(4)], 1)
                # # (4*B,3,32,32)
                # images = images.view(-1, 3, 32, 32)
                # # 将标签下标*4，为不同旋转的原图对应的标签
                # target = torch.stack([target * 4 + k for k in range(4)], 1).view(-1)

                feature = model.feature(images)
                # 设计训练集最后无法构成一个batch的数据
                if feature.shape[0] == self.batch_size:
                    # 记录标签和特征，保存在数组中
                    labels.append(target.cpu().numpy())
                    features.append(feature.cpu().numpy())
        # 将训练集所有图像的标签和特征合并在一起：labels(Dt*4,),feature(Dt*4,768)
        labels_set = np.unique(labels)
        labels = np.array(labels)
        labels = np.reshape(labels, labels.shape[0] * labels.shape[1])
        # labels = np.reshape(labels, labels.shape[0])
        features = np.array(features)
        features = np.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))

        for item in labels_set:
            index = np.where(item == labels)[0]  # 所有item类的下标
            feature_classwise = features[index]  # 所有item类的特征: [500,512]
            self.prototype_dict[item] = np.mean(feature_classwise, axis=0)  # 计算所有item类数据的均值作为类均值:[512]
