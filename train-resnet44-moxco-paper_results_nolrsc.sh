#!/usr/bin/env bash
# Train binarized ResNets on CIFAR-10 where warmstart is over no lrsc version SGD on GPU:"1"
DEPTH=44

for i in 0 1 2 3
do
    python3 main_binary_reg_vanilla_small_44_adaMoM_hypergradient_2024_may_gpu1.py --model resnet --resume results/resnet44_MAY2024_nolrsc1  --model_config "{'depth': 44}" --save resnet44_MAY20240_moxco_nolrscwarmup$i --dataset cifar10 --batch-size 128 --gpu 1 --epochs 300 --reg_rate 1e-5  --lr 0.025  --projection_mode prox --freeze_epoch 200  --alpha 0.5 --beta 0.9 --alpha_exp_config 3 --resetting_window 0.87 --reset_alpha 0.7 --reset_beta 0.3 --temperature 0.15  --hyper_lr  0.01
done

wait
echo all processes complete



