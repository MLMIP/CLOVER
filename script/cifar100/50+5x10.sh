#!/bin/bash

#CUDA_VISIBLE_DEVICES=0 python src/main_incremental.py --approach clover --gmms 1 --max-experts 5 --use-multivariate --nepochs 200 --tau 3 --batch-size 128 --num-workers 8 --datasets cifar100_icarl_224 --num-tasks 6 --nc-first-task 50 --lr 0.1 --weight-decay 1e-4 --ft-lr 0.1 --ft-weight-decay 1e-4 --clipping 1 --alpha 0.99 --ftepochs 0 --use-test-as-val --network resnet18 --extra-aug fetril --momentum 0.9 --exp-name cifar50+5x10 --seed 0 --shared 1 --w0 0.10 --w1 0.05 --save-models

CUDA_VISIBLE_DEVICES=0 python src/main_incremental.py --approach clover --gmms 1 --max-experts 2 --use-multivariate --nepochs 2 --tau 3 --batch-size 128 --num-workers 8 --datasets cifar100_icarl_224 --num-tasks 6 --nc-first-task 50 --lr 0.1 --weight-decay 1e-4 --ft-lr 0.1 --ft-weight-decay 1e-4 --clipping 1 --alpha 0.99 --ftepochs 0 --use-test-as-val --network resnet18 --extra-aug fetril --momentum 0.9 --exp-name cifar50+5x10 --seed 0 --shared 1 --w0 0.10 --w1 0.05 --save-models
