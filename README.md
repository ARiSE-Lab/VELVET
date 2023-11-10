<p align="center">
  <img src="images/columbia-logo.png" width="120" />
  <img src="images/ibm-logo.jpg" width="120" />
</p>

# VELVET: a noVel Ensemble Learning approach to automatically locate VulnErable sTatements ([SANER 2022](https://saner2022.uom.gr/restrack))

This [paper](https://arxiv.org/abs/2112.10893) presents VELVET, a novel ensemble learning approach to locate vulnerable statements. Our model combines graph-based and sequence-based neural networks to successfully capture the local and global context of a program graph and effectively understand code semantics and vulnerable patterns. This work is done by researchers from Columbia University and IBM Research.

## Updates

__Nov, 2023__: __For the convenience of PyTorch users and easier adoption or customization, and to enable VELVET's potential of integrating the latest deep-learning techniques, we are re-implementing VELVET with PyTorch, taking advantage of the latest pre-trained code LM checkpoints and GNN architectures. Please check [VELVET-PyTorch](https://github.com/Robin-Y-Ding/VELVET-PyToch).__

## Data

This paper considers two datasets as the main resources for the evaluation:
* [Juliet Test Suite for C/C++](https://samate.nist.gov/SRD/testsuite.php)
* [IBM D2A Dataset](https://developer.ibm.com/exchanges/data/all/d2a/). Our processed function-level dataset can be found [here](https://drive.google.com/drive/folders/1Q-yApGmz-HyNdrgN8jxy2ugG-cmmGu7B?usp=sharing).

## Approach
<p align="center">
  <img src="images/Workflow_ensemble.png" width="600" />
</p>

Graph-based neural networks are effective at understanding the semantic order of programs, since they directly learn control flows and data dependencies with the pre-defined edges. However, training involves a message passing algorithm where nodes only communicate with their neighbors. The ability to learn long-range dependencies is limited by the number of message passing iterations, which are typically set to a small number (e.g., less than eight) due to computational cost. Such a limitation will result in an inherently local model. In contrast, Transformer allows global, program-wise information aggregation, and without pre-defined edges, the self-attention mechanism of Transformer is expected to encode considerable code semantics – which can be complementary to those defined explicitly by the code graph. Therefore, to learn the diversity of vulnerable patterns, we separately train these two distinct models and use their predictions in an ensemble learning setting at inference time.

Our implementation for the model can be found [here](src/).

## Citation
```
@inproceedings{ding2022velvet,
author = {Y. Ding and S. Suneja and Y. Zheng and J. Laredo and A. Morari and G. Kaiser and B. Ray},
booktitle = {2022 IEEE International Conference on Software Analysis, Evolution and Reengineering (SANER)},
title = {VELVET: a noVel Ensemble Learning approach to automatically locate VulnErable sTatements},
year = {2022},
issn = {1534-5351},
pages = {959-970},
keywords = {location awareness;codes;neural networks;static analysis;software;data models;security},
doi = {10.1109/SANER53432.2022.00114},
url = {https://doi.ieeecomputersociety.org/10.1109/SANER53432.2022.00114},
publisher = {IEEE Computer Society},
address = {Los Alamitos, CA, USA},
month = {mar}
}

```
