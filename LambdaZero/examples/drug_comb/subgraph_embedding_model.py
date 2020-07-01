from LambdaZero.examples.drug_comb.drug_comb_data import DrugCombDb, transform_to_bipartite_drug_protein_graph
from torch_geometric.data import DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F

data = DrugCombDb(pre_transform=transform_to_bipartite_drug_protein_graph)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = SubgraphEmbeddingModel().to(device), data.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

def get_train_test_splits(data, test_proportion):
    test_size = math.floor(len(data) * test_proportion)
    train_size = len(data) - test_size

    return random_split(data, [train_size, test_size])

train_set, test_set = get_train_test_splits(data, test_proportion=0.1)
train_loader = DataLoader(train_set, batch_size=256)

def train_epoch():
    model.train()

    all_loss = []
    for batch in train_loader:
        optimizer.zero_grad()
        preds = model(batch)

        loss = F.mse(preds, batch.y)
        loss.backwards()
        optimizer.step()

        all_loss.append(loss)

    return all_loss

best_test_perf = 0
for epoch in range(500):
    epoch_losses = train_epoch()

