# On Learning Sets of Symmetric Elements
A Pytorch implementation of The ICML 2020 paper "On Learning Sets of Symmetric Elements" by Haggai Maron, Or Litany, Gal Chechik, Ethan Fetaya
https://arxiv.org/pdf/2002.08599.pdf
## Abstract
Learning from unordered sets is a fundamental learning setup, recently attracting increasing attention. Research in this area has focused on the case where elements of the set are represented by feature vectors, and far less emphasis has been given to the common case where set elements themselves adhere to their own symmetries. That case is relevant to numerous applications, from deblurring image bursts to multi-view 3D shape recognition and reconstruction.
In this paper, we present a principled approach to learning sets of general symmetric elements. We first characterize the space of linear layers that are equivariant both to element reordering and to the inherent symmetries of elements, like translation in the case of images. We further show that networks that are composed of these layers, called Deep Sets for Symmetric Elements layers (DSS), are universal approximators of both invariant and equivariant functions. DSS layers are also straightforward to implement. Finally, we show that they improve over existing set-learning architectures in a series of experiments with images, graphs, and point-clouds.
## Data
A data generator for the synthetic experiment is provided. All Other datasets should be downloaded independently and handled according to the paper.
Specifically, for generating burst image deblurring data one can use the following [repository](https://github.com/FrederikWarburg/Burst-Image-Deblurring)

## Prerequisites
python 3

Pytorch 1.2

torch_geometric

## Bibtex:

```
@InProceedings{Maron_2020_ICML,
author={Maron, Haggai and Litany, Or and Chechik, Gal and Fetaya, Ethan},
title = {On Learning Sets of Symmetric Elements},
booktitle = {Proceedings of the International Conference on Machine Learning (ICML 2020)},
month = {July},
year = {2020}
}
```
