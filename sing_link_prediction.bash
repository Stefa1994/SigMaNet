#!/bin/bash

epochs=('300' '500' '1000')
datasets=(
    'bitcoin_alpha' 'bitcoin_otc' 'slashdot' 'epinions'
)
num_filters='64'
lr='1e-3'
weight_decay='5e-4'
task="four_class_signed_digraph" # "five_class_signed_digraph")
num_classes='4'

for epochs in "${epochs[@]}"; do
    for data in "${datasets[@]}"; do
    #for epochs in "${epochs[@]}"; do
# MSGNN
        q_values=(0.01 0.05 0.1 0.15 0.2 0.25)
        for q in "${q_values[@]}"; do
            command=("python3 sign_link_prediction.py --dataset=$data --num_classes=$num_classes --q=$q --hidden=$num_filters --task=$task --K=2 --num_layers=2 --epochs=$epochs --dropout=0.5 --lr=$lr --method=MSGNN")
            echo "${command}"
            $command
        done

# SigMaNet
        command=("python3 sign_link_prediction.py --dataset=$data --task=$task --num_classes=$num_classes --hidden=$num_filters --K=1 --num_layers=2 --epochs=$epochs --dropout=0.5 --lr=$lr --method=SigMaNet")
        echo "${command}"
        $command

# QuaterGCN
        command=("python3 sign_link_prediction.py --dataset=$data --task=$task  --num_classes=$num_classes --hidden=$num_filters --K=1 --num_layers=2 --epochs=$epochs --dropout=0.5 --lr=$lr")
        echo "${command}"
        $command
    done
done