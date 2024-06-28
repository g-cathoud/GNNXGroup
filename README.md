# GNNXGroup

This repository contains the code for the Master thesis *'Decoding Chemical Predictions: Group Contribution Methods for XAI'*. This branch focuses on the molecule models. For the reaction models, please visit the `reaction_models` branch.

## Getting Started

To reproduce the results obtained in the study, please follow the steps below. Please make sure you have Python, Pytorch, Pytorch geometric, and RDKit installed. It might be necessary to create some folders for the code to work and store the correct files. 

#### Prepare dataset
```bash
python 0_preparing_dataset.py --dataset=[dataset]
```

#### Collect the groups' information and generate the group contributions ground truth
```bash
python 1_linear_regression.py --dataset=[dataset]
```

#### Train the models
```bash
python 2_training.py \
    --epochs=1000 \
    --model_name=[model] \
    --model_type=[model_type] \
    --target=[target] \
    --root_dir=[root_dir] 
    --dataset=[dataset] \
    --batch_size=32 \
    --weight_decay=0.9e-16
```

#### Extract the results from the trained models

Before extracting the results, please prepare a `training_data.json` file containing the information of the trained models to be processed. Example of the file is given bellow:

```json
{
    "gnn_models": [
        "qm9_schnet_original_H_2024-05-14_20-45-44",
        "qm9_schnet_group_gap_2024-05-16_06-11-15",
        "qm9_egnn_group_H_2024-05-25_17-26-24_custom_train",
        "alchemy_egnn_original_gap_2024-05-17_02-47-04",
        "alchemy_schnet_group_gap_2024-05-28_12-57-52_custom_train"
    ],
    "regression_models": [
        "ridge_qm9_benson_H_split_scaffold",
        "ridge_qm9_benson_gap_split_scaffold",
        "ridge_alchemy_benson_H_split_scaffold",
        "ridge_alchemy_benson_gap_split_scaffold"
    ],
    "contribution_models": [
        "qm9_schnet_group_H_2024-05-14_20-45-56_contributions.pkl",
        "qm9_egnn_group_gap_2024-05-17_17-05-33_contributions.pkl",
        "qm9_schnet_group_gap_2024-05-25_17-25-56_custom_train_contributions.pkl",
        "alchemy_schnet_group_H_2024-05-17_02-44-02_contributions.pkl",
        "alchemy_egnn_group_H_2024-05-17_05-23-49_contributions.pkl"
    ]
}
```

```bash
python 3_get_results.py
```

#### Generate the XAI plots
```bash
python 3_get_contribution_plots.py
```

