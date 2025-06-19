# Self-Regulating Distribution Alignment

Authors: [QYw12](https://github.com/QYw12)).
<hr />

## Highlights

![main figure](model.jpg)
> **<p align="justify"> Abstract:** Although large-scale pre-trained vision-language models (VLMs) exhibit significant potential for cross-domain visual tasks, existing prompt-learning-based unsupervised domain adaptation (UDA) methods suffer from source domain overfitting and target domain performance degradation. This paper experimentally demonstrates that conventional prompt learning exhibits insufficient cross-domain generalization due to optimization heavily biased toward the source distribution. To address this challenge, we propose a Self-Regulating Distribution Alignment (SRDA) framework. Its core innovation is a dual-branch co-optimization mechanism that dynamically balances cross-domain semantic alignment with pre-trained knowledge preservation. Specifically, the self-regulating multimodal prompt branch incorporates three constraints: semantic consistency regularization, dual-domain collaborative contrastive regularization, and text semantic diversity enhancement. This design suppresses prompt overfitting to the source domain while preserving CLIP's zero-shot generalization capability. The cross-domain alignment branch introduces dynamic dual-domain feature bank and Cross-domain Collaborative Dual Attention (CCDA) module, achieving fine-grained local semantic calibration through moving average prototypes and a dual-layer attention mechanism. Extensive experiments validate SRDA's effectiveness on downstream UDA tasks.* </p>

<details>
  
<summary>Main Contributions</summary>

1)	This work experimentally reveals the inherent contradiction in existing prompt-learning based UDA methods between source domain overfitting and target domain performance degradation. we demonstrate that under frozen pre-trained parameters, the overfitting of prompt vector optimization towards the source distribution constitutes the core issue hindering cross-domain generalization.

2)	The proposed SRDA framework, which simultaneously achieves cross-domain semantic alignment and preserves VLMs' zero-shot generalization capability through a co-optimization paradigm of decoupled multimodal prompt learning and cross-domain fine-grained alignment.

3)	Extensive experiments conducted on three benchmark cross-domain datasets (Office-Home, Office-31, and VisDA-2017) validate the effectiveness of the SRDA framework, achieving classification accuracies of 87.0\%, 92.6\%, and 90.1\% respectively.
   
</details>


## Results
### SRDA in comparison with existing prompt tuning methods
Results reported below show accuracy across 3 UDA datasets with ViT-B/16 backbone. Our SRDA method adopts the paradigm of multi-modal prompt tuning.

| Method                                                    | Office-Home Acc. | Office-31 Acc. |  VisDA-2017 Acc.  | 
|-----------------------------------------------------------|:---------:|:----------:|:---------:|
| [CLIP](https://arxiv.org/abs/2103.00020)                  |   82.1   |   77.5    |   88.9   | 
| [CoOp](https://arxiv.org/abs/2109.01134)                  |   83.9   |   89.4    |   82.7   |
| [CoCoOp](https://arxiv.org/abs/2203.05557)                |   84.1   |   88.9    |   84.2   | 
| [VP](https://arxiv.org/abs/2203.17274)                    |   81.7   |   77.4    |   88.7   | 
| [VPT-deep](https://arxiv.org/abs/2203.12119)              |   83.9   |   89.4    |   86.2   | 
| [MaPLe](https://arxiv.org/abs/2210.03117)                 |   84.2   |   89.6    |   83.5   |
| [DAPL](https://arxiv.org/abs/2202.06687)                  |   84.4   |   81.2    |   89.5   |
| [PDA](https://arxiv.org/abs/2312.09553)                   |   85.7   |   91.2    |   89.7   | 
| [SRDA](Ours)                                              |   **87.0**   |   **92.6**    | **90.1** |

## Installation 
For installation and other package requirements, please follow the instructions as follows. 
This codebase is tested on Ubuntu 18.04 LTS with python 3.7. Follow the below steps to create environment and install dependencies.

* Setup conda environment.
```bash
# Create a conda environment
conda create -y -n pda python=3.7

# Activate the environment
conda activate srda

# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/get-started/previous-versions/ if your cuda version is different
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
```

* Install dassl library.
```bash
# Instructions borrowed from https://github.com/KaiyangZhou/Dassl.pytorch#installation

# Clone this repo
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch

# Install dependencies
pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
cd ..
```

* Clone SRDA code repository and install requirements.
```bash
# Clone SRDA code base
git clone https://github.com/QYw12/Self-Regulating Distribution Alignment.git
cd Self-Regulating Distribution Alignment

# Install requirements
pip install -r requirements.txt
```

## Data Preparation
Please follow the instructions to prepare all datasets.
Datasets list:
- [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?pli=1&resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw)
- [Office-31](https://faculty.cc.gatech.edu/~judy/domainadapt/#datasets_code)
- [VisDA-2017](http://ai.bu.edu/visda-2017/#download)


## Training and Evaluation
Please follow the instructions for training, evaluating, and reproducing the results.
Firstly, you need to **modify the directory of data by yourself**.
### Training 
```bash
# Example: trains on Office-Home dataset, and the source domian is art and the target domain is clipart (a-c)
bash scripts/srda/main_srda.sh officehome b16_ep30_officehome SRDA ViT-B/16 2 a-c 0
```

### Evaluation
```bash
# Example: evaluates on Office-Home dataset, and the source domian is art and the target domain is clipart (a-c)
bash scripts/srda/eval_srda.sh officehome b16_ep30_officehome SRDA ViT-B/16 2 a-c 0
```
The details are in each method folder in [scripts folder](scripts/).

## Supported Methods
Supported methods in this codespace are as follows:

| Method                    |                   Paper                        |                             Code                                     |               Script                           |
|---------------------------|:----------------------------------------------:|:--------------------------------------------------------------------:|:----------------------------------------------:|
| CoOp                      | [IJCV 2022](https://arxiv.org/abs/2109.01134)  |  [link](https://github.com/KaiyangZhou/CoOp)                         |  [link](scripts/coop)                          |
| CoCoOp                    | [CVPR 2022](https://arxiv.org/abs/2203.05557)  |  [link](https://github.com/KaiyangZhou/CoOp)                         |  [link](scripts/cocoop)                        |
| VP                        | [-](https://arxiv.org/abs/2203.17274)          |  [link](https://github.com/hjbahng/visual_prompting)                 |  -                                             |
| VPT                       | [ECCV 2022](https://arxiv.org/abs/2203.12119)  |  [link](https://github.com/KMnP/vpt)                                 |  [link](scripts/vpt)                           |
| IVLP & MaPLe              | [CVPR 2023](https://arxiv.org/abs/2210.03117)  |  [link](https://github.com/muzairkhattak/multimodal-prompt-learning) |  [link](scripts/ivlp) & [link](scripts/maple)  |
| DAPL                      | [TNNLS 2023](https://arxiv.org/abs/2202.06687) |  [link](https://github.com/LeapLabTHU/DAPrompt)                      |  [link](scripts/dapl)                          |
| PDA                      | [AAAI 2024](https://arxiv.org/abs/2312.09553v2) |  [link](https://github.com/BaiShuanghao/Prompt-based-Distribution-Alignment)                   |  [link](scripts/pda)                          |


## Citation
If our code is helpful to your research or projects, please consider citing:
```bibtex
```

## Contact
If you have any questions, please create an issue on this repository or contact at qy@stu.xjtu.edu.cn.

## Acknowledgements

Our style of reademe refers to [PDA](https://github.com/BaiShuanghao/Prompt-based-Distribution-Alignment). 
And our code is based on [CoOp and CoCoOp](https://github.com/KaiyangZhou/CoOp), [DAPL](https://github.com/LeapLabTHU/DAPrompt/tree/main) ,[MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning) and [PDA](https://github.com/BaiShuanghao/Prompt-based-Distribution-Alignment)etc. repository. We thank the authors for releasing their codes. If you use their codes, please consider citing these works as well.

