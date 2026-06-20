import networkx as nx
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import pickle
import random
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split


random.seed(42)
np.random.seed(42)

NUM_GRAPHS = 5000
TARGET_PER_CLASS = NUM_GRAPHS // 2
GRAPH_GENERATION_MODELS = ['gnp', 'random_geometric', 'scale_free', 'sbm']

DATASET_DIR = 'graph_dataset'
os.makedirs(DATASET_DIR, exist_ok=True)


def generate_connected_gnp(n, p):
    while True:
        G = nx.erdos_renyi_graph(n, p)
        if nx.is_connected(G):
            return G


def generate_disconnected_gnp(n, p, min_components=2):
    num_components = random.randint(min_components, min(4, n // 10))
    sizes = _split_size(n, num_components)
    G = nx.Graph()
    current_node = 0
    for size in sizes:
        if size <= 0:
            continue
        sub_p = p
        subgraph = generate_connected_gnp(size, sub_p)
        mapping = {node: node + current_node for node in subgraph.nodes()}
        subgraph = nx.relabel_nodes(subgraph, mapping)
        G = nx.compose(G, subgraph)
        current_node += size
    return G


def generate_connected_rgg(n, radius, dim=2):
    while True:
        G = nx.random_geometric_graph(n, radius, dim=dim)
        if nx.is_connected(G):
            return G


def generate_disconnected_rgg(n, radius, dim=2, min_components=2):
    num_components = random.randint(min_components, min(4, n // 10))
    sizes = _split_size(n, num_components)
    G = nx.Graph()
    current_node = 0
    for size in sizes:
        subgraph = generate_connected_rgg(size, radius, dim)
        mapping = {node: node + current_node for node in subgraph.nodes()}
        subgraph = nx.relabel_nodes(subgraph, mapping)
        G = nx.compose(G, subgraph)
        current_node += size
    return G


def generate_connected_scale_free(n, m):
    while True:
        G = nx.barabasi_albert_graph(n, m)
        if nx.is_connected(G):
            return G


def generate_disconnected_scale_free(n, m, min_components=2):
    num_components = random.randint(min_components, min(4, n // 10))
    sizes = _split_size(n, num_components)
    G = nx.Graph()
    current_node = 0
    for size in sizes:
        m_component = min(m, size - 1) if size > 1 else 1
        subgraph = generate_connected_scale_free(size, m_component)
        mapping = {node: node + current_node for node in subgraph.nodes()}
        subgraph = nx.relabel_nodes(subgraph, mapping)
        G = nx.compose(G, subgraph)
        current_node += size
    return G


def generate_connected_sbm(n, num_communities, sizes, p_intra, p_inter):
    attempts = 0
    while attempts < 10:
        G = nx.stochastic_block_model(
            sizes,
            [[p_intra if i == j else p_inter for j in range(num_communities)] for i in range(num_communities)]
        )
        if nx.is_connected(G):
            return G
        attempts += 1
    return generate_connected_gnp(n, p=0.5)


def generate_disconnected_sbm(n, num_communities, sizes, p_intra, p_inter):
    G = nx.stochastic_block_model(
        sizes,
        [[p_intra if i == j else 0.0 for j in range(num_communities)] for i in range(num_communities)]
    )
    if nx.is_connected(G):
        return generate_disconnected_sbm(n, num_communities, sizes, p_intra, p_inter)
    return G


def _split_size(n, num_parts):
    if n < num_parts:
        raise ValueError("Cannot split nodes into more parts than available nodes.")
    base_size = n // num_parts
    sizes = [base_size] * num_parts
    remainder = n % num_parts
    for i in range(remainder):
        sizes[i] += 1
    sizes = [max(size, 1) for size in sizes]
    return sizes


def generate_connected_sbm_wrapper(n):
    num_communities = random.randint(2, 4)
    sizes = _split_size(n, num_communities)
    p_intra = random.uniform(0.3, 0.6)
    p_inter = random.uniform(0.05, 0.2)
    return generate_connected_sbm(n, num_communities, sizes, p_intra, p_inter)


def generate_disconnected_sbm_wrapper(n):
    num_communities = random.randint(2, 4)
    sizes = _split_size(n, num_communities)
    p_intra = random.uniform(0.3, 0.6)
    p_inter = 0.0
    return generate_disconnected_sbm(n, num_communities, sizes, p_intra, p_inter)


def generate_graph(label, n_nodes):
    n = n_nodes
    model = random.choice(GRAPH_GENERATION_MODELS)
    if model == 'gnp':
        p = random.uniform(0.1, 0.5)
        if label == 1:
            G = generate_connected_gnp(n, p)
        else:
            G = generate_disconnected_gnp(n, p)
    elif model == 'random_geometric':
        radius = random.uniform(0.2, 0.5)
        if label == 1:
            G = generate_connected_rgg(n, radius)
        else:
            G = generate_disconnected_rgg(n, radius)
    elif model == 'scale_free':
        m = random.randint(2, 4)
        if label == 1:
            G = generate_connected_scale_free(n, m)
        else:
            G = generate_disconnected_scale_free(n, m)
    elif model == 'sbm':
        if label == 1:
            G = generate_connected_sbm_wrapper(n)
        else:
            G = generate_disconnected_sbm_wrapper(n)
    return G


def _verify_degree_distribution(graphs, labels):
    connected_degrees = [d for G, label in zip(graphs, labels) if label == 1 for d in dict(G.degree()).values()]
    disconnected_degrees = [d for G, label in zip(graphs, labels) if label == 0 for d in dict(G.degree()).values()]
    plt.figure(figsize=(10, 6))
    plt.hist(connected_degrees, bins=range(0, max(connected_degrees) + 2), alpha=0.5, label='Connected', density=True)
    plt.hist(disconnected_degrees, bins=range(0, max(disconnected_degrees) + 2), alpha=0.5, label='Disconnected',
             density=True)
    plt.title('Degree Distribution Comparison')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(os.path.join(DATASET_DIR, 'degree_distribution.png'))
    plt.close()
    print("Degree distribution plot saved.")


def _save_dataset(graphs, labels):
    dataset = [{'graph': G, 'label': label} for G, label in zip(graphs, labels)]
    with open(os.path.join(DATASET_DIR, 'graph_dataset.pkl'), 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Dataset saved with {len(dataset)} graphs.")


def nx_to_pyg_data(nx_graph, label):
    nodes = sorted(nx_graph.nodes())
    node_map = {n: i for i, n in enumerate(nodes)}
    edge_list = []
    for u, v in nx_graph.edges():
        edge_list.append([node_map[u], node_map[v]])
        edge_list.append([node_map[v], node_map[u]])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    num_nodes = len(nodes)
    x = torch.ones((num_nodes, 1), dtype=torch.float)
    y = torch.tensor([label], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)


def convert_list_to_pyg(data_list):
    return [
        nx_to_pyg_data(item["graph"], item["label"])
        for item in data_list
    ]


def create_connectivity_dataset(n_nodes):
    if os.path.exists(f"connectivity_dataset/{n_nodes}connectivity_train_data.pt"):
        train_data = torch.load(f"connectivity_dataset/{n_nodes}connectivity_train_data.pt")
        val_data = torch.load(f"connectivity_dataset/{n_nodes}connectivity_train_data.pt")
        test_data = torch.load(f"connectivity_dataset/{n_nodes}connectivity_train_data.pt")
        return train_data, val_data, test_data

    graphs = []
    labels = []

    print(f"Generating connected graphs over {n_nodes} nodes...")
    for _ in tqdm(range(TARGET_PER_CLASS)):
        G = generate_graph(label=1, n_nodes=n_nodes)
        graphs.append(G)
        labels.append(1)

    print(f"Generating disconnected graphs over {n_nodes} nodes...")
    for _ in tqdm(range(TARGET_PER_CLASS)):
        G = generate_graph(label=0, n_nodes=n_nodes)
        graphs.append(G)
        labels.append(0)

    combined = list(zip(graphs, labels))
    random.shuffle(combined)
    graphs[:], labels[:] = zip(*combined)

    _verify_degree_distribution(graphs, labels)

    _save_dataset(graphs, labels)

    print("Dataset generation complete.")

    pkl_file = "graph_dataset/graph_dataset.pkl"
    with open(pkl_file, "rb") as f:
        dataset = pickle.load(f)

    connected = [g for g in dataset if g["label"] == 1]
    not_connected = [g for g in dataset if g["label"] == 0]

    print("Number of connected graphs:", len(connected))
    print("Number of non-connected graphs:", len(not_connected))

    assert len(connected) == 2500, "Expected 2500 connected graphs, got {}".format(len(connected))
    assert len(not_connected) == 2500, "Expected 2500 non-connected graphs, got {}".format(len(not_connected))

    train_conn, temp_conn = train_test_split(connected, train_size=0.8, random_state=42)
    val_conn, test_conn = train_test_split(temp_conn, test_size=0.5, random_state=42)

    train_notconn, temp_notconn = train_test_split(not_connected, train_size=0.8, random_state=42)
    val_notconn, test_notconn = train_test_split(temp_notconn, test_size=0.5, random_state=42)

    train_list = train_conn + train_notconn
    val_list = val_conn + val_notconn
    test_list = test_conn + test_notconn

    random.shuffle(train_list)
    random.shuffle(val_list)
    random.shuffle(test_list)

    train_data = convert_list_to_pyg(train_list)
    val_data = convert_list_to_pyg(val_list)
    test_data = convert_list_to_pyg(test_list)

    torch.save(train_data, f"connectivity_dataset/{n_nodes}connectivity_train_data.pt")
    torch.save(val_data, f"connectivity_dataset/{n_nodes}connectivity_val_data.pt")
    torch.save(test_data, f"connectivity_dataset/{n_nodes}connectivity_test_data.pt")

    print("Saved train, val, and test sets to .pt files.")
    return train_data, val_data, test_data
