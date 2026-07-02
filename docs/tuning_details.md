# Hyperparameter Tuning Details

This document describes the hyperparameter tuning process for the baselines reported in Table 2 of the manuscript.

## Datasets

The experiments were conducted on two datasets:

- YELP
- MovieLens

For each dataset, the hyperparameters were tuned using the validation set. The best configuration was selected according to validation performance and then evaluated on the test set.

## Baselines

The following baselines were compared in Table 2:

- Abs Greedy
- Max Entropy
- CRM
- EAR
- SCPR
- UNICORN-FM
- UNICORN-TransE
- MCMIPL
- HutCRS
- T*-SCPR
- LSTI-Clock

## General Tuning Strategy

For each baseline, we followed the hyperparameter settings suggested by the original paper or official implementation when available. For parameters that were not explicitly specified, we searched commonly used values on the validation set.

The main tuned hyperparameters include:

- learning rate
- batch size
- embedding dimension
- hidden dimension
- dropout rate
- number of graph layers
- discount factor
- replay buffer size
- reward settings
- model-specific parameters

## Reinforcement Learning Settings

For reinforcement-learning-based methods, we tuned or followed the commonly used settings for:

- recommendation success reward
- recommendation failure reward
- attribute asking success reward
- attribute asking failure reward
- quit penalty
- discount factor
- replay memory size
- mini-batch size
- learning rate

For LSTI-Clock, the reward settings are:

- recommendation success reward: 1
- recommendation failure reward: -0.1
- attribute asking success reward: 0.01
- attribute asking failure reward: -0.1
- quit penalty: -0.3

The replay buffer size is 50,000, the mini-batch size is 128, the learning rate is 1e-4, and the discount factor is 0.999.

## Dataset-specific Settings

For LSTI-Clock, the number of local GAT layers was set separately for different datasets:

- YELP: 4
- MovieLens: 1

Other parameters were selected based on validation performance.

## Final Evaluation

After selecting the best hyperparameter configuration on the validation set, each method was evaluated on the test set. The final test results are reported in Table 2 of the manuscript.
