#!/bin/bash

epochs=('300' '500' '1000')
epochs_1='300'
datasets=(
    'bitcoin_alpha' 
    'bitcoin_otc' 'slashdot' 'epinions'
)
lr='1e-3'
task="four_class_signed_digraph" #"five_class_signed_digraph")
num_classes='4'


for epochs in "${epochs[@]}"; do
    for data in "${datasets[@]}"; do
    #SGCN
        command=("python3 sign_link_prediction.py --dataset=$data --task=$task --num_classes=$num_classes --num_layers=2 --epochs=$epochs --dropout=0.5 --lr=$lr --method SGCN")
        echo "${command}"
        $command
    done
done

for epochs in "${epochs[@]}"; do
    for data in "${datasets[@]}"; do
    # SDGNN
        command=("python3 sign_link_prediction.py --dataset=$data --task=$task --num_classes=$num_classes --num_layers=2 --epochs=$epochs_1 --dropout=0.5 --lr=$lr --method SDGNN")
        echo "${command}"
        $command
    done
done

