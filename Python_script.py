# GNN for predicting fertility intentions from egocentric social networks.
# Master's thesis, Applied Data Science, Utrecht University. Dimitrios Diamantidis.

# Builds one graph per participant, trains a GCN to predict their childwish,
# and runs it over every feature combination to see which features help.

import os
import random
import itertools

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch.utils.data import Subset

import pandas as pd
import networkx as nx

from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, KFold

from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

SEED = 12345
EPOCHS = 170
K_FOLDS = 5
HIDDEN_CHANNELS = 64
PARTICIPANTS_DIR = "participants"
DATA_CSV = "data_subset.csv"

ATTRIBUTES = [
    "age", "num_children", "sex", "child_free", "friend",
    "f_to_f", "relationship_type", "help", "non_f2f", "happiness",
]


# The survey answers come in as text, with a lot of special cases. These turn
# them into numbers the model can use.

def preprocess_age(age_str):
    if age_str == "50+":
        return 50
    if pd.isna(age_str) or age_str is None:
        return 0
    if age_str == "18-":
        return 18
    return int(age_str)


def preprocess_num_children(num_children):
    if num_children == "Expecting first child":
        return 0.5
    if pd.isna(num_children) or num_children == "I don't know":
        return 0
    if num_children == "More than 5":
        return 5
    return float(num_children)


def preprocess_sex(sex):
    return 1 if sex == "Female" else -1


def preprocess_happiness_child_a(value):
    less = "[PERSON 1 to 25]'s happiness in life became less after the birth of the child(ren)"
    more = "[PERSON 1 to 25]'s happiness in life increased after the birth of the child(ren)"
    same = "[PERSON 1 to 25]'s happiness in life remained the same after the birth of the child(ren)"
    not_born = "[PERSON 1 to 25]'s child has not been born yet"
    if value in ("I don't know", not_born):
        return 0
    if value in (less, 1, 2):
        return 1
    if value in (more, 4, 5):
        return 2
    if value in (same, 3):
        return 3
    return value


def preprocess_child_free(child_free):
    if child_free == "Prefers to remain childless":
        return 0
    if child_free == "Wishes to have children":
        return 1
    return 2


def preprocess_friend(friend):
    if pd.isna(friend) or friend is None:
        return 0
    return 1 if friend == "Yes, is a friend" else -1


def preprocess_relationship_type(relationship):
    if pd.isna(relationship) or relationship is None:
        return 0
    if isinstance(relationship, str):
        return float(relationship.split(",")[0].strip())
    return float(relationship)


def preprocess_numeric(value):
    # used for f_to_f, help and non_f2f, which are already numeric
    if pd.isna(value) or value is None:
        return 0
    return float(value)


def graph_to_data_object(graph, label, features):
    y = torch.tensor([label], dtype=torch.long)

    node_mapping = {node: i for i, node in enumerate(graph.nodes())}
    edge_list = [(node_mapping[u], node_mapping[v]) for u, v in graph.edges()]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    rows = []
    for _, node_data in graph.nodes(data=True):
        feats = []
        if "age" in features:
            feats.append(preprocess_age(node_data.get("age_a", None)))
        if "num_children" in features:
            feats.append(preprocess_num_children(node_data.get("num_child_a", 0)))
        if "sex" in features:
            feats.append(preprocess_sex(node_data.get("sex_a", None)))
        if "child_free" in features:
            feats.append(preprocess_child_free(node_data.get("childfree_a", None)))
        if "friend" in features:
            feats.append(preprocess_friend(node_data.get("friend_a", None)))
        if "happiness" in features:
            feats.append(preprocess_happiness_child_a(node_data.get("happiness_child_a")))
        if "relationship_type" in features:
            feats.append(preprocess_relationship_type(node_data.get("relation_a")))
        if "f_to_f" in features:
            feats.append(preprocess_numeric(node_data.get("f_to_f")))
        if "help" in features:
            feats.append(preprocess_numeric(node_data.get("help")))
        if "non_f2f" in features:
            feats.append(preprocess_numeric(node_data.get("non_f2f")))
        rows.append(feats)

    x = torch.tensor(rows, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, y=y)


class SocialNetworkDataset(Dataset):
    def __init__(self, graphs, combination, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.data_list = [
            graph_to_data_object(graph, label, combination)
            for graph, label in graphs.values()
        ]
        self.total_classes = self._count_classes()

    def _count_classes(self):
        labels = set()
        for data in self.data_list:
            if data.y.numel() > 1:
                labels.update(data.y.tolist())
            else:
                labels.add(data.y.item())
        return len(labels)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels=HIDDEN_CHANNELS):
        super().__init__()
        torch.manual_seed(SEED)
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        return self.lin(x)


class EarlyStopping:
    def __init__(self, patience=20, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.best_state = None
        self.early_stop = False

    def __call__(self, val_f1, model):
        if self.best_score is None or val_f1 > self.best_score + self.delta:
            self.best_score = val_f1
            # keep a copy of the best weights so we can put them back later
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for data in loader:
            out = model(data.x, data.edge_index, data.batch)
            _, pred = torch.max(out, dim=1)
            preds.extend(pred.cpu().numpy())
            labels.extend(data.y.cpu().numpy())
    # macro F1 because the classes are imbalanced
    return f1_score(labels, preds, average="macro"), confusion_matrix(labels, preds)


def oversample_indices(dataset, indices):
    counts = {}
    for i in indices:
        label = dataset[i].y.item()
        counts[label] = counts.get(label, 0) + 1
    max_count = max(counts.values())

    out = []
    for label, count in counts.items():
        label_indices = [i for i in indices if dataset[i].y.item() == label]
        factor = max_count // count
        remainder = max_count % count
        out.extend(label_indices * factor + random.sample(label_indices, remainder))
    random.shuffle(out)
    return out


def train_model(model, loader, epochs=EPOCHS):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for data in loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
    return model


def load_graphs():
    participant_ids = [
        d for d in os.listdir(PARTICIPANTS_DIR)
        if os.path.isdir(os.path.join(PARTICIPANTS_DIR, d))
    ]
    participant_data = pd.read_csv(DATA_CSV)
    graphs = {}

    for pid in participant_ids:
        alters_df = pd.read_csv(os.path.join(PARTICIPANTS_DIR, pid, "alters.csv"))
        graph = nx.read_graphml(os.path.join(PARTICIPANTS_DIR, pid, "edgelist.graphml"))
        row = participant_data[participant_data["nomem_encr"] == int(pid)]

        # add the participant as a central node connected to everyone in their network
        ego = "n25"
        graph.add_node(ego)
        defaults = {c: None for c in alters_df.columns if c not in ("names_a", "node_id")}
        graph.nodes[ego].update(defaults)

        if not row.empty:
            graph.nodes[ego]["has_child_a"] = row["has_child_num"].values[0]
            graph.nodes[ego]["num_child_a"] = row["num_children"].values[0]
            graph.nodes[ego]["sex_a"] = row["sex"].values[0]
            graph.nodes[ego]["age_a"] = row["age"].values[0]
            graph.nodes[ego]["happiness_child_a"] = row["happiness_num"].values[0]
            graph.nodes[ego]["childwish_a"] = row["childwish_num"].values[0]
            graph.nodes[ego]["childfree_a"] = row["has_children"].values[0]
            graph.nodes[ego]["help_child_a"] = row["no_help"].values[0]

        for node in list(graph.nodes()):
            if node != ego:
                graph.add_edge(ego, node)
            index = int(node[1:]) + 1
            match = alters_df[alters_df["names_a"] == index]
            if not match.empty:
                for key, value in match.to_dict("records")[0].items():
                    graph.nodes[node][key] = value

        graphs[pid] = graph

    participant_data["nomem_encr"] = participant_data["nomem_encr"].astype(str)
    childwish_mapping = {
        "Probably so": 2, "Absolutely so": 2,
        "Probably not": 1, "Absolutely not": 1,
    }
    childwish_dict = participant_data.set_index("nomem_encr")["childwish"].to_dict()

    for key in list(graphs.keys()):
        value = childwish_dict.get(str(key), None)
        label = childwish_mapping.get(value, 0)
        graphs[key] = [graphs[key], label]

    return graphs


def attribute_combinations(attributes):
    return [
        list(comb)
        for i in range(1, len(attributes) + 1)
        for comb in itertools.combinations(attributes, i)
    ]


def main():
    graphs = load_graphs()
    combinations = attribute_combinations(ATTRIBUTES)
    results = {}

    for idx, combination in enumerate(combinations, 1):
        dataset = SocialNetworkDataset(graphs, combination)
        torch.manual_seed(SEED)

        indices = list(range(len(dataset)))
        train_idxs, test_idxs = train_test_split(indices, test_size=0.2, random_state=42)
        train_dataset = Subset(dataset, train_idxs)
        test_dataset = Subset(dataset, test_idxs)

        kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset), 1):
            early_stopping = EarlyStopping(patience=20)
            # oversample the training fold only, never the validation fold
            os_idx = oversample_indices(train_dataset, train_idx)

            train_loader = DataLoader(Subset(train_dataset, os_idx), batch_size=64, shuffle=True)
            val_loader = DataLoader(Subset(train_dataset, val_idx), batch_size=64, shuffle=False)

            model = GCN(dataset.num_node_features, dataset.total_classes)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            criterion = torch.nn.CrossEntropyLoss()

            for _ in range(1, EPOCHS + 1):
                model.train()
                for data in train_loader:
                    optimizer.zero_grad()
                    out = model(data.x, data.edge_index, data.batch)
                    loss = criterion(out, data.y)
                    loss.backward()
                    optimizer.step()

                val_f1, _ = evaluate(model, val_loader)
                early_stopping(val_f1, model)
                if early_stopping.early_stop:
                    break

            early_stopping.restore(model)
            print(f"Combination {idx}, fold {fold}, validation F1: {val_f1:.3f}")

        # retrain on the full training set, then test once
        final_os_idx = oversample_indices(train_dataset, list(range(len(train_dataset))))
        final_loader = DataLoader(Subset(train_dataset, final_os_idx), batch_size=64, shuffle=True)
        final_model = train_model(GCN(dataset.num_node_features, dataset.total_classes), final_loader)

        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        test_f1, test_cm = evaluate(final_model, test_loader)
        results[idx] = (combination, test_f1)

        print(f"Test F1: {test_f1:.3f}")
        print(f"{idx}/{len(combinations)} combinations done\n")

    best_idx = max(results, key=lambda k: results[k][1])
    best_combo, best_f1 = results[best_idx]
    print(f"\nBest combination: {best_combo} with test F1 {best_f1:.3f}")


if __name__ == "__main__":
    main()
