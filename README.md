# GNNXGroup

This repository contains the code for the Master thesis *'Decoding Chemical Predictions: Group Contribution Methods for XAI'*. This branch focuses on the reaction models. For the molecule models, please visit the `main` branch.

To reproduce the results obtained in the study, please follow the steps below. Please make sure you have Python, Pytorch, Pytorch geometric, and RDKit installed. It might be necessary to create some folders for the code to work and store the correct files. 

#### Prepare dataset
```bash
python 0_preparing_dataset.py --dataset=[dataset]
```

#### Train the models
```bash
python 1_training.py \
    --epochs=1500 \
    --model_name=[model] \
    --model_type=[model_type] \
    --target=[target] \
    --root_dir=[root_dir] 
    --dataset=[dataset] \
    --batch_size=32 \
    --weight_decay=0.9e-16
```

#### Extract the results from the trained models

Before extracting the results, please prepare a `training_data.json` file containing the information of the trained models to be processed. An example of the file is given below:

```json
{
    "reaction_models": [
        "gdb722ts_egnn_group_2024-06-04_21-51-17.pth", 
        "gdb722ts_egnn_atomic_2024-06-04_21-51-17.pth", 
        "gdb722ts_schnet_group_2024-06-04_21-51-17.pth",
        "gdb722ts_schnet_atomic_2024-06-04_21-51-17.pth",
    ],

    "reaction_models_no_contribution": [
        "gdb722ts_egnn_original-var2_2024-06-04_21-51-17.pth", 
        "gdb722ts_egnn_original-var1_2024-06-04_21-51-17.pth", 
        "gdb722ts_schnet_original-var2_2024-06-04_21-51-17.pth",
        "gdb722ts_schnet_original-var1_2024-06-04_21-51-17.pth",
    ]
}
```
```bash
python 2_get_results.py
```

