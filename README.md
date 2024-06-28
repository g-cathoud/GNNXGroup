# GNNXGroup

This repository contains the code for the Master thesis *'Decoding Chemical Predictions: Group Contribution Methods for XAI'*. This branch focuses on the molecule models. For the reaction models, please visit the `reaction_models` branch.

## Getting Started

To reproduce the results obtained in the study, please follow the steps below.

Please make sure you have Python, Pytorch, Pytorch geometric, and RDKit  installed. 

#### Prepare dataset
```bash
python 0_preparing_dataset.py
```

#### Prepare dataset
```bash
python 1_linear_regression.py
```

#### Prepare dataset
```bash
python 2_training.py
```

#### Prepare dataset
```bash
python 3_get_results.py
```

#### Prepare dataset
```bash
python 3_get_contribution_plots.py
```

