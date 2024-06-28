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
```bash
python 3_get_results.py
```

#### Generate the XAI plots
```bash
python 3_get_contribution_plots.py
```

