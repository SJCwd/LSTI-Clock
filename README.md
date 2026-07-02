# LSTI-Clock

This repository contains the reproducibility artifacts for the manuscript:

**LSTI-Clock: Long-Short Term Interest Adaptive Clock Model for Conversational Recommendation Systems**

submitted to **ACM Transactions on Recommender Systems (TORS)**.

## Overview

LSTI-Clock is a conversational recommendation model that incorporates temporal information to model users' long-term and short-term interests. The repository provides the source code, data preparation instructions, training and evaluation scripts, and hyperparameter tuning details required for reproducing the experimental results in the manuscript.

## Repository Structure

```text
LSTI-Clock/
├── Graph_generate/      # Code for graph construction and graph-related preprocessing
├── RL/                  # Reinforcement learning components
├── data/                # Dataset instructions
├── evaluate.py          # Evaluation script
├── gcn.py               # GCN module
├── graph_init.py        # Graph initialization
├── process_data.py      # Data preprocessing
├── RL_model.py          # Reinforcement learning model
├── sum_tree.py          # Sum-tree implementation for replay memory
├── utils.py             # Utility functions
├── README.md            # Main documentation
└── docs/
    └── tuning_details.md
```

## Environment

The code is implemented in Python. Please install the required dependencies before running the experiments.

If a `requirements.txt` file is provided, install dependencies using:

```bash
pip install -r requirements.txt
```

## Data Preparation

The experiments in the manuscript are conducted on the YELP and MovieLens datasets.

Please follow the instructions in:

```text
data/README.md
```

After placing the datasets under the `data/` directory, run:

```bash
python process_data.py
```

## Running the Model

After data preprocessing, the model can be trained and evaluated using the provided Python scripts.

Example command:

```bash
python evaluate.py
```

Please adjust the dataset name, paths, and hyperparameters according to the configuration used in the manuscript.

## Hyperparameter Tuning

The hyperparameter tuning details for the baselines reported in Table 2 are provided in:

```text
docs/tuning_details.md
```

## Reproducibility Notes

This repository includes the implementation and documentation required to reproduce the main experimental results reported in the manuscript. The datasets are not directly included due to file size and/or license restrictions.
