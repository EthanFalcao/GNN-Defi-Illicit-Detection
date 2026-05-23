import torch
import torch.nn as nn
from torch_geometric.nn.conv import GCNConv
from torch_geometric.data import Data
import numpy as np
from sklearn.metrics import f1_score, classification_report

# Referenced: https://arxiv.org/abs/1609.02907 (Kipf & Welling, 2017)
class GCNModel(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels=2,
                 device="cuda",
                 dropout=0.6
        ):
        super(GCNModel, self).__init__()

        # Layer 1: raw features -> hidden representation
        # GCNConv handles the normalized aggregation (D^-1/2 A D^-1/2)
        self.conv1 = GCNConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            improved=False,
            add_self_loops=True,
            normalize=True
        )

        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout(p=dropout)

        # Layer 2: hidden -> hidden (extra depth to capture 2-hop neighborhood)
        self.conv2 = GCNConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            improved=False,
            add_self_loops=True,
            normalize=True
        )

        self.elu2 = nn.ELU()
        self.dropout2 = nn.Dropout(p=dropout)

        # Layer 3: hidden -> class scores (licit vs illicit)
        # NOTE: returns raw logits — do NOT apply LogSoftmax/LogSigmoid here.
        # CrossEntropyLoss and FocalLoss both expect raw logits and apply
        # their own log-softmax internally. Applying log twice destroys gradients
        # and causes the model to predict all-licit (F1 = 0.00).
        self.conv3 = GCNConv(
            in_channels=hidden_channels,
            out_channels=out_channels,
            improved=False,
            add_self_loops=True,
            normalize=True
        )

        self.device = device

    def forward(self, x, edge_index):
        # Layer 1: aggregate neighbors, compress to hidden_channels
        conv1_out = self.conv1(x, edge_index)
        elu1_out = self.elu1(conv1_out)
        drop1_out = self.dropout1(elu1_out)

        # Layer 2: aggregate again — each node now sees 2 hops
        conv2_out = self.conv2(drop1_out, edge_index)
        elu2_out = self.elu2(conv2_out)
        drop2_out = self.dropout2(elu2_out)

        # Layer 3: raw logits — CrossEntropyLoss handles softmax internally
        out = self.conv3(drop2_out, edge_index)

        return out


def create_loader(X, y, edge_index, split=0.2):
    # Mask out unknown transactions (y == -1) — same as gat.py
    known_mask = y != -1
    known_idx = np.where(known_mask)[0]
    test_split = int(np.round(len(known_idx) * split))
    train_split = len(known_idx) - test_split

    train_mask = torch.zeros(len(y))
    test_mask = torch.zeros(len(y))
    train_mask[known_idx[:train_split]] = True
    test_mask[known_idx[train_split:]] = True

    data = Data(
        x=torch.tensor(X),
        edge_index=edge_index,
        y=torch.tensor(y),
        train_mask=train_mask.nonzero().squeeze().long(),
        test_mask=test_mask.nonzero().squeeze().long()
    )

    train_labels = y[known_idx[:train_split]]
    _, cls_weights = np.unique(train_labels, return_counts=True)
    return data, cls_weights


def train_gcn(model, data, optimizer, criterion, device='cuda', epochs=10):
    model = model.to(device)

    source = data.x.to(device)
    target = data.y.to(device)
    data_edge_index = data.edge_index.to(device)
    train_mask = data.train_mask.to(device)
    print(np.unique(data.y[data.train_mask]))

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        out = model(source, data_edge_index)
        loss = criterion(out[train_mask], target[train_mask])
        loss.backward()

        # Gradient clipping — same as GAT to stabilize training
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss = loss.item()
        print(f"Epoch {epoch + 1} / {epochs}, Loss: {total_loss:.4f}")

    del source, target, data_edge_index, out, loss, train_mask
    torch.cuda.empty_cache()

    return model


def evaluate_gcn(model, data, criterion, device="cuda"):
    model.eval()

    source = data.x.to(device)
    target = data.y.to(device)
    data_edge_index = data.edge_index.to(device)
    test_mask = data.test_mask.to(device)

    with torch.no_grad():
        out = model(source, data_edge_index)
        pred = out[test_mask].argmax(dim=1)

        loss = criterion(out[test_mask], target[test_mask])
        total_loss = loss.item()

        # Same reporting format as evaluate_gat — makes comparison easy
        class_report_dict = classification_report(
            target[test_mask].cpu(),
            pred.cpu(),
            labels=[0, 1],
            target_names=['licit', 'illicit'],
            output_dict=True,
            zero_division=0
        )
        class_report = classification_report(
            target[test_mask].cpu(),
            pred.cpu(),
            labels=[0, 1],
            target_names=['licit', 'illicit'],
            zero_division=0
        )

    del source, target, data_edge_index, out, loss, test_mask
    torch.cuda.empty_cache()

    print(f"Validation loss: {total_loss}")
    print(f"Validation F1 (illicit): {class_report_dict['illicit']['f1-score']:.4f}")
    return total_loss, class_report_dict, class_report