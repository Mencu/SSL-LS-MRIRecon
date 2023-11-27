# Self-supervised Low-rank plus Sparse Network for Radial MRI Reconstruction

This is the companion repository of the NeurIPS 2023 workshop paper: [Self-supervised Low-rank plus Sparse Network for Radial MRI Reconstruction](https://neurips.cc/virtual/2023/79273)

This project was part of the interdisciplinary project at the Techincal University of Munich and has the aim of improving the acquisition speed of magnetic resonance imaging (MRI) using limited radially sampled data.

The repository contains pytorch lightning scripts to train each model described in the paper (plus another cascade of convolutions). Each technique has its own independnt folder to completely atomize and to customize each.


### Project Overview

An important aspect of this project is that all the models use **radial sampling** for the k-space data, which can help reduce moving artifacts, at the same time ensuring patient comfort and high resolution scans.


### Datasets

We have used an in-house dataset for our experiments. The dataset contains 128 samples from 16 subjects. We have generated reconstructions from 1 and form 4 patient heartbeats.

### How to Use

First, you need a conda or mamba environment that supports all the required packages. The required packages are inside `environment.yml` and an environment can be created and activated with:

```
conda create -n mri -f environment.yml
conda activate mri
```

Secondly, you need to pre-process your kspace/image data pairs and store them. Examples are provided in `ssl_mri_reconstruction/create_collapsed.py` and `ssl_mri_reconstruction/create_collapsed_gpunufft.py`, which use the non-uniform FFT for radial data.

Thirdly, head into the desired technique's folder, adapt the paths (indicated by `<your_path_here>`) to find the data and run the trainings scripts with:

```
python train.py
```

:exclamation: Important :exclamation:
Some training scripts require a working Pytorch installation of [Optox](https://github.com/VLOGroup/optox/tree/master) and [Merlin](https://github.com/midas-tum/merlin).

Feel free to change the training or dataloading files as you desire. 

If you feel that the project helped your research, don't hesitate to cite it:
```
@inproceedings{
mancu2023selfsupervised,
title={{Self-supervised Low-rank plus Sparse Network for Radial {MRI} Reconstruction}},
author={Andrei Mancu and Wenqi Huang and Gastao Lima da Cruz and Daniel Rueckert and Kerstin Hammernik},
booktitle={NeurIPS 2023 Workshop on Deep Learning and Inverse Problems},
year={2023},
url={https://openreview.net/forum?id=hUOHV4SKNw}
}
```


### Milestones

- [X] CGSense reconstructions in order to check torchkbnufft
- [X] Supervised learning approach for baseline MRI reconstruction
    - Deep Cascade of CNNs[[1]](#1) with cartesian (original)
    - [X] Deep Cascade of CNNs with radial sampling (modified)
- [X] Self-supervised physics-guided deep learning reconstruction[[2]](#2) with radial sampling
- [X] Adapt the NN model to L+S[[3]](#3)[[4]](#4)


-----

### Literature Resources

<a id="1">[1]</a> Schlemper, Jo & Caballero, Jose & Hajnal, Joseph & Price, Anthony & Rueckert, Daniel. (2017). [A Deep Cascade of Convolutional Neural Networks for Dynamic MR Image Reconstruction.](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8067520) IEEE Transactions on Medical Imaging. PP. 10.1109/TMI.2017.2760978. 

<a id="2">[2]</a> Yaman, Burhaneddin & Hosseini, Seyed Amir Hossein & Moeller, Steen & Ellermann, Jutta & Uğurbil, Kâmil & Akçakaya, Mehmet. (2020). [Self-supervised learning of physics-guided reconstruction neural networks without fully sampled reference data.](https://onlinelibrary.wiley.com/doi/epdf/10.1002/mrm.28378?saml_referrer) Magnetic resonance in medicine. 84. 10.1002/mrm.28378. 

<a id="3">[3]</a> Otazo R, Candès E, Sodickson DK. [Low-rank plus sparse matrix decomposition for accelerated dynamic MRI with separation of background and dynamic components.](https://onlinelibrary.wiley.com/doi/10.1002/mrm.25240) Magn Reson Med. 2015 Mar;73(3):1125-36. doi: 10.1002/mrm.25240. Epub 2014 Apr 23. PMID: 24760724; PMCID: PMC4207853.

<a id="4">[4]</a> Huang, Wenqi & Ziwen, Ke & Cui, Zhuo-Xu & Jing Cheng, Jing Cheng & Qiu, Zhilang & Jia, Sen & Ying, Lei & Zhu, Yanjie & Liang, Dong. (2021). [Deep Low-Rank Plus Sparse Network for Dynamic MR Imaging.](https://www.researchgate.net/publication/353440776_Deep_Low-Rank_Plus_Sparse_Network_for_Dynamic_MR_Imaging) Medical Image Analysis. 73. 102190. 10.1016/j.media.2021.102190. 

