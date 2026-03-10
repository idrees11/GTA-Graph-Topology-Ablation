import torch
import pandas as pd
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import os

from dataset import TopologicalDataset
from model import GINModel


# -------------------------------------------------
# Device
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# -------------------------------------------------
# Paths
# -------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DATA_DIR = os.path.join(REPO_ROOT, "data")
SUBMISSIONS_DIR = os.path.join(REPO_ROOT, "submissions")

os.makedirs(SUBMISSIONS_DIR, exist_ok=True)


# -------------------------------------------------
# Load data splits
# -------------------------------------------------
print("Loading CSV splits...")

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))


# -------------------------------------------------
# Dataset instances
# -------------------------------------------------
print("Loading datasets...")

train_dataset = TopologicalDataset(
    "MUTAG",
    topo_config="degree",
    mode="ideal"
)

ideal_test_dataset = TopologicalDataset(
    "MUTAG",
    topo_config="degree",
    mode="ideal"
)

perturbed_test_dataset = TopologicalDataset(
    "MUTAG",
    topo_config="degree",
    mode="perturbed"
)


# -------------------------------------------------
# Build graph lists
# -------------------------------------------------
print("Preparing training graphs...")

train_graphs = []

for _, row in train_df.iterrows():

    g = train_dataset[int(row.graph_index)]

    # assign label from CSV
    g.y = torch.tensor([int(row.label)], dtype=torch.long)

    train_graphs.append(g)


print("Preparing test graphs...")

ideal_test_graphs = [
    ideal_test_dataset[int(i)] for i in test_df["graph_index"]
]

perturbed_test_graphs = [
    perturbed_test_dataset[int(i)] for i in test_df["graph_index"]
]


# -------------------------------------------------
# DataLoaders
# -------------------------------------------------
train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)

ideal_test_loader = DataLoader(
    ideal_test_graphs,
    batch_size=32,
    shuffle=False
)

perturbed_test_loader = DataLoader(
    perturbed_test_graphs,
    batch_size=32,
    shuffle=False
)


# -------------------------------------------------
# Model
# -------------------------------------------------
print("Initializing model...")

model = GINModel(
    input_dim=train_dataset.num_features,
    output_dim=train_dataset.num_classes
)

model = model.to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)


# -------------------------------------------------
# Training
# -------------------------------------------------
print("Training model...")

for epoch in range(50):

    model.train()
    total_loss = 0

    for data in train_loader:

        data = data.to(device)

        optimizer.zero_grad()

        out = model(data)

        loss = F.nll_loss(out, data.y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")


# -------------------------------------------------
# Prediction Function
# -------------------------------------------------
def predict(model, loader):

    model.eval()

    predictions = []

    with torch.no_grad():

        for data in loader:

            data = data.to(device)

            out = model(data)

            pred = out.argmax(dim=1)

            predictions.extend(pred.cpu().tolist())

    return predictions


# -------------------------------------------------
# Generate Predictions
# -------------------------------------------------
print("Generating IDEAL predictions...")
ideal_predictions = predict(model, ideal_test_loader)

print("Generating PERTURBED predictions...")
perturbed_predictions = predict(model, perturbed_test_loader)


# -------------------------------------------------
# Save Submissions
# -------------------------------------------------
ideal_path = os.path.join(SUBMISSIONS_DIR, "ideal_submission.csv")
perturbed_path = os.path.join(SUBMISSIONS_DIR, "perturbed_submission.csv")

pd.DataFrame({
    "graph_index": test_df["graph_index"],
    "target": ideal_predictions
}).to_csv(ideal_path, index=False)

pd.DataFrame({
    "graph_index": test_df["graph_index"],
    "target": perturbed_predictions
}).to_csv(perturbed_path, index=False)


print("\nSubmissions generated successfully:")
print(f"Ideal predictions saved to: {ideal_path}")
print(f"Perturbed predictions saved to: {perturbed_path}")
