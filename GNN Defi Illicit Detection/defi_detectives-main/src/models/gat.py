import torch
import torch.nn as nn
from torch_geometric.nn.conv import GATConv
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split


# Referenced: https://arxiv.org/pdf/1710.10903 for inductive architecture
class GATModel(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels=2,
                 device="cuda",
                 heads_in=4,
                 heads_out=1,
                 concat=True,
                 dropout=0.6
        ):
        super(GATModel, self).__init__()
        # Referenced:https://towardsdatascience.com/graph-attention-networks-in-python-975736ac5c0c/
        # for channel count, Claude told me out_channels should be number of classes
        
        # Claude suggestion for getting more heads and hidden_channels
        # without hitting OOM error
        if concat:
            n_hidden = hidden_channels*heads_in
        else:
            n_hidden = hidden_channels

        self.conv1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads_in,
            concat=concat,
            dropout=dropout
        )

        self.elu1 = nn.ELU()
    
        self.conv2 = GATConv(
            in_channels=n_hidden,
            out_channels=hidden_channels,
            heads=heads_in,
            concat=concat,
            dropout=dropout
        )

        self.elu2 = nn.ELU()

        self.conv3 = GATConv(
            in_channels=n_hidden,
            out_channels=out_channels,
            heads=heads_out,
            concat=concat,
            dropout=dropout
        )

        self.device = device

    def forward(self, x, edge_index):
        conv1_out = self.conv1(x, edge_index)
        elu1_out = self.elu1(conv1_out)
        conv2_out = self.conv2(elu1_out, edge_index)
        elu2_out = self.elu2(conv2_out)
        out = self.conv3(elu2_out, edge_index)

        return out

def create_loader(X, y, edge_index, split=0.2):
    """Create the data loader and class weights

    Args:
        X (np.NDArray): data
        y (np.NDArray): labels
        edge_index (torch.tensor): edge indices of nodes
        split (float, optional): validation and test set split. Defaults to 0.2.

    Returns:
        NeighborLoader, NeighborLoader, NeighborLoader, torch.tensor: 
            train, validation, and test loaders, and raw class weights
    """
    # Claude suggested to mask out unknown transactions to help with loss
    known_mask = y != -1
    known_idx = np.where(known_mask)[0]
    known_labels = y[known_idx]

    train_ratio = 1 - split
    train_split = int(len(known_idx) * train_ratio)

    # Claude recommended stratifying with train_test_split so that the
    # test and val set had similar licit to illicit ratios
    train_idx, val_test_idx = train_test_split(
        known_idx,
        test_size=split,
        stratify=known_labels,
    )
    
    val_test_labels = y[val_test_idx]
    val_idx, test_idx = train_test_split(
        val_test_idx,
        test_size=0.5,
        stratify=val_test_labels,
    )

    train_mask = torch.zeros(len(y), dtype=torch.bool)
    val_mask = torch.zeros(len(y), dtype=torch.bool)
    test_mask = torch.zeros(len(y), dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    data = Data(
        x=torch.tensor(X),
        edge_index=edge_index,
        y=torch.tensor(y),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    train_loader = NeighborLoader(
        data,
        num_neighbors=[-1],
        input_nodes=data.train_mask,
        batch_size=1024,
    )

    val_loader = NeighborLoader(
        data,
        num_neighbors=[-1],
        input_nodes=data.val_mask,
        batch_size=1024
    )

    test_loader = NeighborLoader(
        data,
        num_neighbors=[-1],
        input_nodes=data.test_mask,
        batch_size=1024
    )

    train_labels = y[known_idx[:train_split]]
    _, cls_weights = np.unique(train_labels, return_counts=True)
    cls_weights = torch.from_numpy(cls_weights).float()
    return train_loader, val_loader, test_loader, cls_weights

# Derived from assignment3/assignment3_NLP_spr26/utils.py
def train_gat(model, loader, optimizer, criterion, device='cuda', epochs=10):
    model = model.to(device)

    model.train()
    optimizer.zero_grad()
    metrics = {
        'licit_f1_scores': [],
        'illicit_f1_scores': [],
        'losses': [],
        'licit_precision': [],
        'licit_recall': [],
        'illicit_precision': [],
        'illicit_recall': [],
        'roc_auc': []
    }

    for epoch in range(epochs):
        preds = []
        # probs = []
        targets = []
        total_loss = 0

        for _, data in enumerate(loader):
            data = data.to(device)
            source = data.x
            target = data.y 
            data_edge_index = data.edge_index
            train_mask = data.train_mask.to(device)

            out = model(source, data_edge_index).to(device)
            pred = out[train_mask].argmax(dim=1)

            loss = criterion(out[train_mask], target[train_mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            preds.append(pred)
            targets.append(target[train_mask])

        avg_loss = total_loss / len(loader)
        metrics['losses'].append(avg_loss)
        preds = torch.cat(preds).cpu()
        targets = torch.cat(targets).cpu()
        class_report_dict = classification_report(targets, preds,
                                                  labels=[0, 1],
                                                  target_names=['licit', 'illicit'],
                                                  output_dict=True,
                                                  zero_division=0)

        licit_f1 = class_report_dict['licit']['f1-score']
        illicit_f1 = class_report_dict['illicit']['f1-score']

        metrics['illicit_f1_scores'].append(illicit_f1)
        metrics['licit_f1_scores'].append(licit_f1)

        print(f"Epoch {epoch + 1} / {epochs}, Loss: {avg_loss:.4f}, F1 Score: {illicit_f1}")

    metrics['targets'] = targets
    metrics['preds'] = preds

    # Claude helped to debug Out Of Memory Error for cuda
    del source, target, data_edge_index, out, loss, train_mask
    torch.cuda.empty_cache()

    return model, metrics


def evaluate_gat(model, loader, criterion, device="cuda", dataset='val'):
    model.eval()
    total_loss = 0
    preds = []
    targets = []
    probs = []
    metrics = {}

    with torch.no_grad():
        for _, data in enumerate(loader):
            if dataset == 'val':
                mask = data.val_mask.to(device)
            else:
                mask = data.test_mask.to(device)

            source = data.x.to(device)
            target = data.y.to(device)
            data_edge_index = data.edge_index.to(device)

            out = model(source, data_edge_index).to(device)
            prob = torch.softmax(out[mask], dim=1)
            pred = out[mask].argmax(dim=1)

            loss = criterion(out[mask], target[mask])
            total_loss += loss.item()

            preds.append(pred)
            targets.append(target[mask])
            probs.append(prob)

        # Claude gave me the idea to use classification report to get
        # better metrics than just focal loss, specifically for known transactions
        preds = torch.cat(preds).cpu()
        targets = torch.cat(targets).cpu()
        probs = torch.cat(probs).cpu()
        class_report_dict = classification_report(targets, preds,
                                            labels=[0, 1],
                                            target_names=['licit', 'illicit'],
                                            output_dict=True,
                                            zero_division=0)
        class_report = classification_report(targets, preds,
                                            labels=[0, 1],
                                            target_names=['licit', 'illicit'],
                                            zero_division=0)
    # Claude helped to debug Out Of Memory Error for cuda
    del source, target, data_edge_index, out, loss, mask
    torch.cuda.empty_cache()

    metrics = {
        "total_loss": total_loss,
        "targets": targets,
        "probs": probs
    }

    print(f"Validation loss: {total_loss}")
    print(f"Validation F1: {class_report_dict['illicit']['f1-score']}")
    return metrics, class_report_dict, class_report