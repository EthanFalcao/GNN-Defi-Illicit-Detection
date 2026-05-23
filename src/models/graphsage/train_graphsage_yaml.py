import copy
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score, roc_auc_score

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR))

from src.models.graphsage.graphsage import GraphSage, load_data, valid_indices, compute_class_weights
from src.losses.focal_loss import FocalLoss
from src.data.smote_graph import smote_augment_graph

BASE_CONFIG = ROOT_DIR / "configs" / "graphsage_base.yaml"
SWEEP_CONFIG = ROOT_DIR / "configs" / "graphsage_sweep.yaml"


def run_trial(cfg, trial_num, total):
    # set seed
    seed = 42 + trial_num
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # load data
    data, train_idx, val_idx, test_idx = load_data(
        graph_dir=ROOT_DIR / cfg["data"]["graph_dir"],
        splits_path=ROOT_DIR / cfg["data"]["splits_path"],
    )

    # use cpu for now
    device = "mps"
    
    # apply SMOTE if enabled
    if cfg.get("smote", {}).get("enabled", False):
        print("Applying SMOTE...")
        valid_train = train_idx[data.y[train_idx] >= 0]
        data, train_idx = smote_augment_graph(
            data, valid_train,
            k_neighbors=cfg["smote"].get("k_neighbors", 5),
            smote_k_neighbors=cfg["smote"].get("smote_k_neighbors", 5),
            random_state=seed,
        )

    data = data.to(device)
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)

    # build model
    model = GraphSage(
        in_channels=data.x.size(-1),
        hidden_channels=cfg["model"].get("hidden_channels", 128),
        out_channels=cfg["model"].get("out_channels", 2),
        dropout=cfg["model"].get("dropout", 0.3),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["optimizer"].get("lr", 1e-3),
        weight_decay=cfg["optimizer"].get("weight_decay", 5e-4),
    )

    labelled = valid_indices(data.y, train_idx)
    class_weights = compute_class_weights(data.y[labelled].cpu()).to(device)
    print(f"Class weights (licit, illicit): {class_weights.tolist()}")

    criterion = FocalLoss(
        weight=class_weights,
        gamma=cfg.get("loss", {}).get("gamma", 2.0),
        beta=cfg.get("loss", {}).get("beta", 0.9999),
    ).to(device)

    # training loop
    epochs = cfg["training"].get("epochs", 100)
    best_f1 = 0.0
    best_weights = None

    print(f"\n=== Trial {trial_num + 1}/{total} ===")
    for epoch in range(1, epochs + 1):
        # train
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        use_idx = valid_indices(data.y, train_idx)
        loss = criterion(logits[use_idx], data.y[use_idx])
        loss.backward()
        optimizer.step()
        train_loss = loss.item()

        # eval on val
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            use_idx = valid_indices(data.y, val_idx)

            if use_idx.numel() > 0:
                pred = logits[use_idx].argmax(dim=1)
                y_true = data.y[use_idx].cpu().numpy()
                y_pred = pred.cpu().numpy()
                val_f1 = f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)

                y_prob = torch.softmax(logits[use_idx], dim=1).cpu().numpy()
                val_roc = None
                if len(np.unique(y_true)) > 1:
                    try:
                        val_roc = roc_auc_score(y_true, y_prob[:, 1])
                    except Exception:
                        pass
            else:
                val_f1 = 0.0
                val_roc = None

        # save best model on val F1
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:03d} | loss={train_loss:.4f} | val_f1={val_f1:.4f} | val_roc={val_roc}")

    if best_weights is None:
        print("  WARNING: val_f1 never improved above 0.0; using final epoch weights.")
        best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # load best weights and eval on val + test
    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)

        # val metrics
        use_idx = valid_indices(data.y, val_idx)
        pred = logits[use_idx].argmax(dim=1)
        val_y_true = data.y[use_idx].cpu().numpy()
        val_y_pred = pred.cpu().numpy()
        val_f1_final = f1_score(val_y_true, val_y_pred, pos_label=1, average="binary", zero_division=0)
        val_y_prob = torch.softmax(logits[use_idx], dim=1).cpu().numpy()
        val_roc_final = (
            roc_auc_score(val_y_true, val_y_prob[:, 1])
            if len(np.unique(val_y_true)) > 1 else None
        )

        # test metrics
        use_idx = valid_indices(data.y, test_idx)
        pred = logits[use_idx].argmax(dim=1)
        test_y_true = data.y[use_idx].cpu().numpy()
        test_y_pred = pred.cpu().numpy()
        test_f1 = f1_score(test_y_true, test_y_pred, pos_label=1, average="binary", zero_division=0)
        test_y_prob = torch.softmax(logits[use_idx], dim=1).cpu().numpy()
        test_roc = (
            roc_auc_score(test_y_true, test_y_prob[:, 1])
            if len(np.unique(test_y_true)) > 1 else None
        )

    print(f"  => val_f1={val_f1_final:.4f}  test_f1={test_f1:.4f}")

    return {
        "trial": trial_num + 1,
        "config": {
            "model": cfg["model"],
            "optimizer": cfg["optimizer"],
            "training": cfg["training"],
            "loss": cfg.get("loss", {}),
        },
        "val":  {"f1": val_f1_final, "roc_auc": val_roc_final},
        "test": {"f1": test_f1,      "roc_auc": test_roc},
        "_weights": best_weights,
    }


if __name__ == "__main__":
    # load configs
    with open(BASE_CONFIG) as f:
        base_cfg = yaml.safe_load(f)
    with open(SWEEP_CONFIG) as f:
        sweep_cfg = yaml.safe_load(f)

    # build trial grid from sweep search_space
    search_space = sweep_cfg.get("search_space", {})
    keys   = list(search_space.keys())
    values = [v if isinstance(v, list) else [v] for v in search_space.values()]

    all_combos = list(itertools.product(*values))

    random.seed(42)
    random.shuffle(all_combos)

    max_trials = sweep_cfg.get("sweep", {}).get("max_trials")
    if max_trials:
        all_combos = all_combos[:max_trials]

    trials = []
    for combo in all_combos:
        cfg = copy.deepcopy(base_cfg)
        for key, val in zip(keys, combo):
            parts = key.split(".")
            d = cfg
            for p in parts[:-1]:
                if p not in d:
                    d[p] = {}
                d = d[p]
            d[parts[-1]] = val
        trials.append(cfg)

    print(f"Running {len(trials)} trials...")

    results = []
    for i, cfg in enumerate(trials):
        result = run_trial(cfg, i, len(trials))
        results.append(result)

    # find best trial by val F1
    best = max(results, key=lambda r: r["val"]["f1"])

    print(f"\n{'='*50}")
    print(f"Best: Trial {best['trial']}")
    print(f"  Val  F1={best['val']['f1']:.4f}  ROC-AUC={best['val']['roc_auc']:.4f}")
    print(f"  Test F1={best['test']['f1']:.4f}  ROC-AUC={best['test']['roc_auc']:.4f}")
    print(f"{'='*50}")

    # save outputs
    output_dir = ROOT_DIR / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "graphsage_sweep_results.json"
    config_file  = output_dir / "graphsage_best_config.yaml"
    weights_file = output_dir / "graphsage_best_weights.pt"

    torch.save(best.pop("_weights"), weights_file)

    # strip non-serialisable _weights from all results before dumping
    clean_results = [{k: v for k, v in r.items() if k != "_weights"} for r in results]
    with open(results_file, "w") as f:
        json.dump(
            {"num_trials": len(trials), "results": clean_results, "best_trial": best},
            f, indent=2,
        )

    with open(config_file, "w") as f:
        yaml.dump(best, f)

    print(f"\nSaved to {output_dir}/")