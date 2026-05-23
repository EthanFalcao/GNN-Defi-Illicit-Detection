# GraphSAGE YAML Sweep

This folder supports GraphSAGE training from YAML files.

## Config files

- `configs/graphsage_base.yaml`: baseline paths + default hyperparameters.
- `configs/graphsage_sweep.yaml`: hyperparameter search space.

## Run sweep

```zsh
python3 src/models/graphsage/train_graphsage_yaml.py \
  --base-config configs/graphsage_base.yaml \
  --sweep-config configs/graphsage_sweep.yaml
```

## Outputs

- `reports/graphsage_sweep_results.json`: every trial and metrics.
- `reports/graphsage_best_config.yaml`: best trial config and metrics.

The script reports validation and test metrics including `f1` and `roc_auc`.

