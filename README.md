# CML demo

This repository contains https://github.com/iterative/cml usage example.

## How to run locally

Execute following commands in repository root

```
pip3 install -r requirements.txt
python3 train.py \
    --plot_dataset dataset.png \     # path to save dataset plot
    --plot_decision_plane dp.png \   # path to save decision plane of model
    --save_metrics metrics.txt       # path to save model metrics
```


## How to run using actions

Just create pull request on branch master.
