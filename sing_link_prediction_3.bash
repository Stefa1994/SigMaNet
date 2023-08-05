#!/bin/bash

epochs=('300' '500' '1000')
datasets=(
    'bitcoin_alpha' 'bitcoin_otc' 
    'slashdot' 'epinions'
)
num_filters='64'
lr='1e-3'
task="four_class_signed_digraph" #"five_class_signed_digraph")
num_classes='4'

for epochs in "${epochs[@]}"; do
    for data in "${datasets[@]}"; do
    # SSSNET
        command=("python3 sign_link_prediction.py --dataset=$data --task=$task --num_classes=$num_classes --hidden=64 --K=1 --num_layers=2 --epochs=$epochs --dropout=0.5 --lr=$lr --method SSSNET")
        echo "${command}"
        $command
# SNEA
        command=("python3 sign_link_prediction.py --dataset=$data --task=$task --num_classes=$num_classes --num_layers=2 --epochs=$epochs --dropout=0.5 --lr=$lr --method SNEA")
        echo "${command}"
        $command
    done
done


for epochs in "${epochs[@]}"; do
    for data in "${datasets[@]}"; do
# SiGAT
        command=("python3 sign_link_prediction.py --dataset=$data --task=$task --num_classes=$num_classes--num_layers=2 --epochs=$epochs --dropout=0.5 --lr=$lr --method SiGAT")
        echo "${command}"
        $command
    done
done
