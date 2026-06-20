import torch.optim as optim
import argparse
import wandb
from torch_geometric.data import DataLoader

import torch.nn.functional as F

import numpy as np
import random

from torch_geometric.utils import to_dense_adj
import torch
import torch.nn as nn
from counting_datatset import GraphCountDataset, GraphCountDatasetNnodes
import time


class TransformerModel(nn.Module):
    def __init__(self, input_dim, args):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, args.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.d_model,
            nhead=args.nhead,
            dim_feedforward=args.d_model,
            dropout=0.1,
            activation="relu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=args.num_encoder_layers)
        self.output_layer = nn.Linear(args.d_model, 1)

    def forward(self, x, mask=None):

        embedded = self.embedding(x)

        embedded = embedded.permute(1, 0, 2)

        if mask is None:
            mask = torch.ones(x.size(0), x.size(1), device=x.device, dtype=torch.bool)

        src_key_padding_mask = ~mask.bool()

        transformer_out = self.transformer(embedded, src_key_padding_mask=src_key_padding_mask)

        transformer_out = transformer_out.permute(1, 0, 2)
        valid_edge_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled_output = (transformer_out * mask.unsqueeze(-1)).sum(
            dim=1) / valid_edge_counts

        return self.output_layer(pooled_output)


def get_graph_tokens(batch, max_nodes, rep_type, device):
    if rep_type == 'adj_rows':
        return adj_rows_process_graphs(batch, max_nodes)
    elif rep_type == 'edge_list':
        return edge_list_process_graphs(batch, max_nodes, device)
    elif rep_type == 'lap_full':
        return laplacian_eigen_process_graphs(batch, max_nodes)


def edge_list_process_graphs(batch, max_nodes, device):

    edge_list_encoded = []
    max_edges = 0
    for data in batch.to_data_list():
        num_nodes = data.num_nodes
        graph_edge_seq = []
        node_one_hot = F.one_hot(torch.arange(num_nodes), num_classes=max_nodes).float()

        for edge in data.edge_index.t():
            source, target = edge

            source_one_hot = node_one_hot[source].to(device)
            target_one_hot = node_one_hot[target].to(device)

            source_features = data.x[source].to(device) if data.x is not None else torch.zeros(
                (data.num_node_features,)).to(device)
            target_features = data.x[target].to(device) if data.x is not None else torch.zeros(
                (data.num_node_features,)).to(device)

            source_representation = torch.cat([source_one_hot, source_features], dim=0)
            target_representation = torch.cat([target_one_hot, target_features], dim=0)

            edge_representation = torch.cat([source_representation.to(device), target_representation.to(device)], dim=0)
            graph_edge_seq.append(edge_representation)
        max_edges = max(max_edges, len(graph_edge_seq))
        edge_list_encoded.append(torch.stack(graph_edge_seq, dim=0))

    padded_edges = []
    edge_masks = []
    for edge_seq in edge_list_encoded:
        num_edges = edge_seq.size(0)
        pad_size = max_edges - num_edges

        padded_edge_seq = F.pad(edge_seq, (0, 0, 0, pad_size))
        padded_edges.append(padded_edge_seq)

        edge_mask = torch.cat([torch.ones(num_edges), torch.zeros(pad_size)])
        edge_masks.append(edge_mask)

    edge_tensor = torch.stack(padded_edges, dim=0)
    edge_mask = torch.stack(edge_masks, dim=0)

    return edge_tensor, edge_mask


def adj_rows_process_graphs(batch, max_nodes):

    adj_feat_list = []
    masks = []

    for data in batch.to_data_list():

        adj = to_dense_adj(edge_index=data.edge_index, max_num_nodes=max_nodes).squeeze(0)
        padded_adj = torch.zeros(max_nodes, max_nodes, device=data.edge_index.device)
        num_nodes_in_graph = data.num_nodes
        padded_adj[:num_nodes_in_graph, :num_nodes_in_graph] = adj[:num_nodes_in_graph, :num_nodes_in_graph]

        if data.x.size(0) > max_nodes:
            raise ValueError("Number of nodes in graph is greater than max_nodes")
        node_features = data.x
        padded_node_features = torch.zeros(max_nodes, node_features.size(1), device=node_features.device)
        padded_node_features[:num_nodes_in_graph] = node_features

        combined_features = torch.cat([padded_adj, padded_node_features], dim=1)
        adj_feat_list.append(combined_features)

        mask = torch.zeros(max_nodes, device=data.edge_index.device, dtype=torch.bool)
        mask[:num_nodes_in_graph] = 1
        masks.append(mask)

    adj_feat = torch.stack(adj_feat_list, dim=0)
    mask = torch.stack(masks, dim=0)

    return adj_feat, mask


def laplacian_eigen_process_graphs(batch, max_nodes):
    lap_feat_list = []
    masks = []

    for data in batch.to_data_list():

        adj_full = to_dense_adj(edge_index=data.edge_index, max_num_nodes=max_nodes).squeeze(0)

        padded_adj = torch.zeros((max_nodes, max_nodes), device=data.edge_index.device)
        num_nodes_in_graph = data.num_nodes

        if num_nodes_in_graph > max_nodes:
            raise ValueError(
                f"Number of nodes in the graph ({num_nodes_in_graph}) "
                f"exceeds max_nodes ({max_nodes})."
            )

        padded_adj[:num_nodes_in_graph, :num_nodes_in_graph] = \
            adj_full[:num_nodes_in_graph, :num_nodes_in_graph]

        deg = padded_adj.sum(dim=1)
        deg_mat = torch.diag(deg)
        laplacian = deg_mat - padded_adj

        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)

        node_features = data.x
        padded_node_features = torch.zeros((max_nodes, node_features.size(1)),
                                           device=node_features.device)
        padded_node_features[:num_nodes_in_graph] = node_features

        combined_features = torch.cat([eigenvectors, padded_node_features], dim=1)

        mask = torch.zeros(max_nodes, device=data.edge_index.device, dtype=torch.bool)
        mask[:num_nodes_in_graph] = True

        lap_feat_list.append(combined_features)
        masks.append(mask)

    lap_feat = torch.stack(lap_feat_list, dim=0)
    mask = torch.stack(masks, dim=0)
    return lap_feat, mask


def train_model(model, criterion, optimizer, train_loader, max_nodes, device, rep_type, task_num):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        batch_seq_tokens, mask = get_graph_tokens(batch, max_nodes, rep_type, device=device)
        batch = batch.to(device)
        batch_seq_tokens = batch_seq_tokens.to(device)
        mask = mask.to(device)
        optimizer.zero_grad()
        outputs = model(batch_seq_tokens, mask)
        loss = criterion(outputs, batch.y[:, task_num].to(device).squeeze().unsqueeze(1).float())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate_model(model, loader, max_nodes, device, criterion, rep_type, task_num):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            batch_seq_tokens, mask = get_graph_tokens(batch, max_nodes, rep_type, device=device)
            batch_seq_tokens = batch_seq_tokens.to(device)
            mask = mask.to(device)
            outputs = model(batch_seq_tokens, mask).squeeze(1)
            preds.append(outputs.cpu())
            targets.append(batch.y[:, task_num].squeeze().float().cpu())

    crit = criterion(torch.cat(preds), torch.cat(targets).unsqueeze(1))
    return crit


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    dataset = GraphCountDatasetNnodes(root='dataset', name=f'count{args.n_nodes}', n_nodes=args.n_nodes)
    max_nodes = max([data.num_nodes for data in dataset])

    train_loader = DataLoader(dataset[dataset.train_idx], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset[dataset.val_idx], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset[dataset.test_idx], batch_size=args.batch_size, shuffle=False)

    node_feature_dim = dataset.num_features
    if args.rep_type == 'adj_rows' or args.rep_type == 'lap_full':
        input_dim = max_nodes + node_feature_dim
    else:
        input_dim = 2 * (max_nodes + node_feature_dim)

    model = TransformerModel(input_dim=input_dim, args=args)

    num_params = sum(p.numel() for p in model.parameters())
    print(f'num_params {num_params}')
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if args.use_wandb:
        exp_name = f"{args.num_encoder_layers} layers, {args.d_model} width"
        config = vars(args)
        config['device'] = device
        config['num_params'] = num_params
        wandb.init(project="", config=config, entity='',
                   settings=wandb.Settings(start_method='thread'),
                   name=exp_name)

    best_val_test_loss = 0
    best_test_test_loss = 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.9,
        patience=100,
        verbose=True,
        min_lr=1e-8,
    )
    total_training_start_time = time.time()
    epoch_times = []
    for epoch in range(args.num_epochs):
        start_time = time.time()
        train_loss = train_model(model, criterion, optimizer, train_loader, max_nodes, device, rep_type=args.rep_type,
                                 task_num=args.task_num)
        val_loss = evaluate_model(model, val_loader, max_nodes, device, criterion, rep_type=args.rep_type,
                                  task_num=args.task_num)
        test_loss = evaluate_model(model, test_loader, max_nodes, device, criterion, rep_type=args.rep_type,
                                   task_num=args.task_num)

        end_time = time.time()
        epoch_duration = end_time - start_time
        epoch_times.append(epoch_duration)

        average_epoch_time = sum(epoch_times) / len(epoch_times)

        print(
            f"Epoch {epoch + 1}/{args.num_epochs}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Validation MAE: {val_loss:.4f}, "
            f"Test MAE: {test_loss:.4f}"
            f"Epoch Time: {epoch_duration:.2f}s, "
            f"Average Epoch Time: {average_epoch_time:.2f}s"
        )
        scheduler.step(train_loss)
        if val_loss < best_val_test_loss:
            best_val_test_loss = val_loss
        if test_loss < best_test_test_loss:
            best_test_test_loss = test_loss
        if args.use_wandb:
            wandb.log({"Train Loss": train_loss, "Validation MAE": val_loss, "Test MAE": test_loss,
                       "Best Val MAE": best_val_test_loss, "Best Test MAE": best_test_test_loss, "Epoch": epoch,
                       "Epoch Time (s)": epoch_duration,
                       "Average Epoch Time (s)": average_epoch_time
                       })
    total_training_end_time = time.time()
    total_training_duration = total_training_end_time - total_training_start_time
    total_training_duration_min = total_training_duration / 60

    start_time = time.time()
    test_loss = evaluate_model(model, test_loader, max_nodes, device, criterion, rep_type=args.rep_type,
                               task_num=args.task_num)
    end_time = time.time()
    inference_duration = end_time - start_time
    inference_duration_min = inference_duration / 60
    print(f"Test MAE: {test_loss:.4f}")
    print(f"Total Inference Time (Sec): {inference_duration:.2f}s")
    print(f"Total Training Time (Sec): {total_training_duration:.2f}s")
    print(f"Total Inference Time (Min): {inference_duration_min:.2f}s")
    print(f"Total Training Time (Min): {total_training_duration_min:.2f}s")

    if args.use_wandb:
        wandb.log({"Test MAE": test_loss,
                   "Total Inference Time (Sec)": inference_duration,
                   "Total Training Time (Sec)": total_training_duration,
                   "Total Inference Time (Min)": inference_duration_min,
                   "Total Training Time (Min)": total_training_duration_min}
                  )
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer Model for cycles')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--d_model', type=int, default=128, help='Dimension of the embedding and transformer layers')
    parser.add_argument('--nhead', type=int, default=1, help='Number of attention heads')
    parser.add_argument('--num_encoder_layers', type=int, default=1, help='Number of transformer layers')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb for logging')
    parser.add_argument('--rep_type', type=str, help='Representation type (adj_rows or edge_list')
    parser.add_argument('--task_num', type=int, default=3, help='tri,tailed,star,cyc4,cus')
    parser.add_argument('--n_nodes', type=int, default=50, help='number of nodes')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    main(args)
