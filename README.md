# SigMaNet: One Laplacian to Rule Them All
This repository is the offical PyTorch implementation of SigMaNet: both its implementation and code for running other convolutional graph networks is present.

## Enviroment Setup
The experiments were conducted under this specific environment:

1. Ubuntu 20.04.3 LTS
2. Python 3.8.10
3. CUDA 10.2
4. Torch 1.11.0 (with CUDA 10.2)

In addition, torch-scatter, torch-sparse and torch-geometric are needed to handle scattered graphs and train Graph Neural Networks (GNNs) for a wide range of applications related to structured data. For these three packages, follow the official instructions for [torch-scatter](https://github.com/rusty1s/pytorch_scatter), [torch-sparse](https://github.com/rusty1s/pytorch_sparse), and [torch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

Finally, Pytorch Geometric Signed Directed [GitHub Pages](https://github.com/SherylHYX/pytorch_geometric_signed_directed) must be installed.
Other library are listed below

## Repository structure

The repository contains 8 different files that are used to launch experiments by invoking the different models used for comparisons for divided in the different task. 
Furthermore, the repository contains two folders:
- **data** contains the syntactic graphs in *synthetic* and the WikiRfa dataset in *wikirfa*.
- **src** contains all the model implementations used for running the experiments. Futhermore, it stores two other foldes **utils** and **layer**.

## Run code

```
cd src
python3 node_SigMaNet.py --dataset dataset_nodes500_alpha0.05_beta0.2
python3 Edge_SigMaNet.py --dataset dataset_nodes500_alpha0.05_beta0.2 --task direction --noisy -N
```




## License

SigMaNet is released under the [MIT License](https://choosealicense.com/licenses/mit/)

## Acknowledgements

The template is borrowed from [MagNet](https://github.com/matthew-hirn/magnet) and Pytorch-Geometric Signed Directed. We thank the authors for the excellent repositories.


