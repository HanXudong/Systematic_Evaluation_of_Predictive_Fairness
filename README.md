# Systematic_Evaluation_of_Predictive_Fairness

Source codes for AACL 2022 paper "Systematic Evaluation of Predictive Fairness"

If you use the code, please cite the following paper:

```
@inproceedings{han-etal-2022-systematic,
    title = "Systematic Evaluation of Predictive Fairness",
    author = "Han, Xudong  and
      Shen, Aili  and
      Cohn, Trevor  and
      Baldwin, Timothy  and
      Frermann, Lea",
    booktitle = "Proceedings of the 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = nov,
    year = "2022",
    address = "Online only",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.aacl-main.6",
    pages = "68--81",
    abstract = "Mitigating bias in training on biased datasets is an important open problem. Several techniques have been proposed, however the typical evaluation regime is very limited, considering very narrow data conditions. For instance, the effect of target class imbalance and stereotyping is under-studied. To address this gap, we examine the performance of various debiasing methods across multiple tasks, spanning binary classification (Twitter sentiment), multi-class classification (profession prediction), and regression (valence prediction). Through extensive experimentation, we find that data conditions have a strong influence on relative model performance, and that general conclusions cannot be drawn about method efficacy when evaluating only on standard datasets, as is current practice in fairness research.",
}

```

# Quick Links
+ [Overview](#overview)

+ [Requirements](#requirements)

+ [Data Preparation](#data-preparation)

+ [Source Code](#source-code)

+ [Experiments](#experiments)

# Overview

In this work, we first propose a framework to manipulate the dataset distributions for classification tasks, and then describe how we adopt debiasing methods to a regression setting.

# Requirements

The model is implemented using PyTorch and FairLib.

```
tqdm==4.62.3
numpy==1.22
docopt==0.6.2
pandas==1.3.4
scikit-learn==1.0
torch==1.10.0
PyYAML==6.0
seaborn==0.11.2
matplotlib==3.5.0
pickle5==0.0.12
transformers==4.11.3
sacremoses==0.0.53
```

Alternatively, you can install the fairlib directly:
```
pip install fairlib
```

# Data Preparation

```python
from fairlib import datasets

datasets.prepare_dataset("moji", "data/deepmoji")
datasets.prepare_dataset("bios", "data/bios")

```

# Source Code

## Data distribution manipulation

The function for manipulating data distributions is shown in `src/generalized_BT.py`

Please check the interactive demo for the usages of this function.
https://hanxudong.github.io/fairlib/tutorial_notebooks/tutorial_manipulate_dist.html

## Debiasing Regression Tasks

```python
    def regression_init(self):
        if not self.args.regression:
            self.regression_label = np.array(
                [0 for _ in range(len(self.protected_label))])
        else:
            # Discretize variable into equal-sized buckets
            if self.split == "train":
                bin_labels, bins = pd.qcut(
                    self.y, q=self.args.n_bins, labels=False, 
                    duplicates = "drop", retbins = True)
                self.args.regression_bins = bins
            else:
                bin_labels = pd.cut(
                    self.y, bins=self.args.regression_bins, labels=False, 
                    duplicates = "drop", include_lowest = True)
            bin_labels = np.nan_to_num(bin_labels, nan=0)
            
            # Reassign labels
            self.regression_label, self.y = np.array(self.y), bin_labels
```

# Experiments

All experiments can be reproduced by using fairlib. Taking the vanilla model over the Bios dataset as an example,
```
python fairlib --project_dir Vanilla --dataset Bios_gender --emb_size 768 --num_classes 28 --batch_size 1024 --lr 0.003 --hidden_size 300 --n_hidden 2 --dropout 0 --base_seed 9822168 --exp_id Vanilla_joint_-1.0_0_0 --epochs_since_improvement 10 --num_groups 2 --epochs 50 --results_dir /data/cephfs/punim0478/xudongh1/experimental_results/dist_joint/ --GBT --GBTObj joint --GBT_N 30000 --GBT_alpha 1.25 
```

- `--GBT` indicates that manipulation is applied to the training dataset
- `--GBTObj joint` means that **Joint Balance** is the manipulation method
- `--GBT_N 30000` sets the training dataset size to be 30k
- `--GBT_alpha 1.25` is the alpha value (x-axis) in Figure 5. 

To automatically generate scripts for experiments, please check the `src\gen_exps_dist.py`, which includes several key hyperparameters
- `DIST_TYPE`: the manipulation type, e.g., joint and g_cond_y.
- `code_dir`: dir to fairlib
- `results_dir`: dir to save experimental results
- `exps["_GBT_alpha"]`: the range of alpha values
