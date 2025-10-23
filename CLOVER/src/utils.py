import os
import torch
import random
import numpy as np


def seed_everything(seed=0):
    """Fix all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def print_summary_weighted(acc_taw, acc_tag, forg_taw, forg_tag, task_weight=None):
    """Print summary of results"""
    if task_weight is None:
        task_weight = [1 for _ in range(acc_taw.shape[0])]

    for name, metric in zip(['TAw Acc', 'TAg Acc', 'TAw Forg', 'TAg Forg'], [acc_taw, acc_tag, forg_taw, forg_tag]):
        print('*' * 108)
        print(name)
        avgs = []
        for i in range(metric.shape[0]):
            print('\t', end='')
            for j in range(metric.shape[1]):
                print('{:5.1f}% '.format(100 * metric[i, j]), end='')

            avg = 0
            if np.trace(metric) == 0.0:
                if i > 0:
                    for k in range(i):
                        avg += 100 * metric[i, k] * task_weight[k]
                    avg /= sum(task_weight[:i])
            else:
                for k in range(i + 1):
                    avg += 100 * metric[i, k] * task_weight[k]
                avg /= sum(task_weight[:i+1])

            print('\tAvg.:{:5.1f}% \n'.format(avg), end='')
            avgs.append(avg)
        if "Acc" in name:
            print('Average incremental list:{}% \n'.format(avgs), end='')
            print('Average incremental:{:5.1f}% \n'.format(np.mean(avgs)), end='')

    print('*' * 108)


def print_summary(acc_taw, acc_tag, forg_taw, forg_tag):
    for name, metric in zip(['TAw Acc', 'TAg Acc', 'TAw Forg', 'TAg Forg'], [acc_taw, acc_tag, forg_taw, forg_tag]):
        print('*' * 108)
        print(name)
        avgs = []
        for i in range(metric.shape[0]):
            print('\t', end='')
            for j in range(metric.shape[1]):
                print('{:5.1f}% '.format(100 * metric[i, j]), end='')
            if np.trace(metric) == 0.0:
                if i > 0:
                    avg = 100 * metric[i, :i].mean()
            else:
                avg = 100 * metric[i, :i + 1].mean()
            print('\tAvg.:{:5.1f}% \n'.format(avg), end='')
            avgs.append(avg)
        if "Acc" in name:
            print('Average incremental:{:5.1f}% \n'.format(np.mean(avgs)), end='')

    print('*' * 108)
